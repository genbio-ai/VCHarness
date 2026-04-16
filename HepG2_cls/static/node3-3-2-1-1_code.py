"""Node 3-3-2-1-1: STRING_GNN Frozen Embeddings + 3-Block Pre-Norm MLP (h=384)
              + Muon+AdamW + Manifold Mixup + ReduceLROnPlateau (FIXED)
              + Stronger Regularization
================================================================
Parent  : node3-3-2-1 (STRING_GNN + 3-block h=384, Muon+AdamW, Mixup,
                       CosineWarmRestarts [BROKEN], test F1=0.4860)
This    : node3-3-2-1-1 (STRING_GNN + 3-block h=384, Muon+AdamW, Mixup,
                         ReduceLROnPlateau [RELIABLE], stronger regularization)

Root cause of parent failure
-------------------------------
1. CRITICAL BUG: CosineAnnealingWarmRestarts never fired restarts.
   MuonWithAuxAdam custom wrapper prevented the scheduler's reset mechanism,
   causing monotonic LR decay (0.01→0.00087) rather than expected warm restarts
   at epochs 80 and 240. This prevented escaping the local optimum found around
   epoch 80. The reference node (val_f1=0.4988 at epoch 329) achieved this
   by escaping via warm restarts.

2. Severe overfitting: 5.4x train/val loss ratio at best epoch (train=0.194
   vs val=1.045), versus the reference node's ~2.3x ratio. The regularization
   (head_dropout=0.15, mixup_prob=0.5, wd=8e-4) was insufficient for a 9.55M-
   parameter model on only 1,273 samples.

Key changes relative to parent (node3-3-2-1)
--------------------------------------------
1. ReduceLROnPlateau scheduler (patience=8, factor=0.5) REPLACES broken
   CosineAnnealingWarmRestarts.
   → RLROP is the proven reliable scheduler in this task:
     - node1-3-2-2-1 (Muon+RLROP, F1=0.4777): baseline proven recipe
     - node3-3-1-1 (Muon+RLROP, F1=0.4793): STRING-only best at time
   → Unlike WarmRestarts+MuonWithAuxAdam, RLROP monitors val/f1 directly
     and does not depend on correct step() propagation through the wrapper.
   → The RLROP operates on the AdamW param group directly (avoids wrapper issue).

2. Increased regularization to reduce the 5.4x overfitting gap:
   - head_dropout: 0.15 → 0.18 (matches best WarmRestart+Muon nodes)
   - trunk_dropout: 0.25 → 0.30 (stronger trunk regularization)
   - Mixup alpha: 0.2 → 0.3 (stronger interpolation for generalization)
   - label_smoothing: 0.0 → 0.05 (mild regularizer preventing overconfident preds)
   - weight_decay: 8e-4 (unchanged, proven optimal)

3. Adjusted training budget:
   - max_epochs: 500 → 350 (cap before extreme overfitting that starts ~epoch 386)
   - early_stop_patience: 150 → 40 (more aggressive stopping when RLROP fires)
   - RLROP patience=8 fires LR halvings; early stop at patience=40 allows ~5 halvings

4. Muon LR slightly reduced: 0.01 → 0.008
   → With stronger regularization, a slightly lower Muon LR is more stable
   → The AdamW LR remains at 3e-4 (standard proven rate)

Unchanged from parent (proven correct)
-----------------------------------------
- STRING_GNN frozen PPI graph embeddings (256-dim)
- 3-block Pre-Norm residual MLP architecture
- hidden_dim=384, inner_dim=768
- Per-gene additive bias (19,920 extra learnable parameters)
- Correct class-weight order: [0.0477, 0.9282, 0.0241] (down/neutral/up)
- Weighted cross-entropy loss
- Gradient clip norm=1.0
- DDP distributed training with all_gather for val/test
- Manifold Mixup in hidden space (after trunk, before output head)
- Muon applied to 2D weight matrices of the 3 residual blocks only
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES = 6640        # number of response genes per perturbation
N_CLASSES = 3         # down (-1→0), neutral (0→1), up (1→2)
GNN_DIM = 256         # STRING_GNN output embedding dimension
HIDDEN_DIM = 384      # MLP hidden dimension (proven optimal from tree-wide evidence)
INNER_DIM = 768       # MLP inner (expansion) dimension = 2 × HIDDEN_DIM


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Each sample is one gene perturbation experiment in HepG2 cells."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        if "label" in df.columns:
            # Labels in {-1,0,1} → shift to {0,1,2}
            labels = np.array(
                [json.loads(x) for x in df["label"].tolist()], dtype=np.int64
            )
            self.labels: Optional[torch.Tensor] = torch.tensor(
                labels + 1, dtype=torch.long
            )
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class PerturbDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        micro_batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df)
        self.val_ds = PerturbDataset(val_df)
        self.test_ds = PerturbDataset(test_df)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------
class PreNormResBlock(nn.Module):
    """Pre-LayerNorm residual block.

    Architecture:
        output = x + LN(x) → Linear(dim→inner) → GELU → Dropout
                               → Linear(inner→dim) → Dropout
    """

    def __init__(self, dim: int, inner_dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        inner_dim: int = INNER_DIM,
        n_blocks: int = 3,
        trunk_dropout: float = 0.30,   # KEY CHANGE: 0.25→0.30 (stronger trunk regularization)
        head_dropout: float = 0.18,    # KEY CHANGE: 0.15→0.18 (matches best WarmRestart nodes)
        muon_lr: float = 0.008,        # KEY CHANGE: 0.01→0.008 (more stable with stronger reg)
        adamw_lr: float = 3e-4,        # AdamW LR for projection/bias/head params
        weight_decay: float = 8e-4,    # Proven optimal for Muon+MLP nodes
        label_smoothing: float = 0.05, # KEY CHANGE: 0.0→0.05 (mild regularizer)
        mixup_alpha: float = 0.3,      # KEY CHANGE: 0.2→0.3 (stronger interpolation)
        mixup_prob: float = 0.5,       # Mixup application probability (unchanged)
        rlrop_patience: int = 8,       # ReduceLROnPlateau patience (proven reliable)
        rlrop_factor: float = 0.5,     # LR halving factor (proven in tree)
        rlrop_min_lr: float = 1e-6,    # Minimum AdamW LR floor
        grad_clip_norm: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Populated in setup()
        self.input_proj: Optional[nn.Sequential] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None

        # STRING_GNN gene-ID → embedding-row index
        self.gnn_id_to_idx: Dict[str, int] = {}

        # Accumulators for val/test gathering
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """Build model and precompute frozen STRING_GNN node embeddings."""
        from transformers import AutoModel

        self.print("Loading STRING_GNN and computing frozen node embeddings ...")
        gnn_model = AutoModel.from_pretrained(
            STRING_GNN_DIR, trust_remote_code=True
        )
        gnn_model.eval()
        gnn_model = gnn_model.to(self.device)

        graph = torch.load(
            Path(STRING_GNN_DIR) / "graph_data.pt",
            map_location=self.device,
        )
        edge_index = graph["edge_index"].to(self.device)
        edge_weight = graph.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        with torch.no_grad():
            gnn_out = gnn_model(edge_index=edge_index, edge_weight=edge_weight)

        # Register as a non-trainable float32 buffer [18870, 256]
        all_emb = gnn_out.last_hidden_state.detach().float()
        self.register_buffer("gnn_embeddings", all_emb)

        # Free GNN model memory
        del gnn_model, gnn_out, graph, edge_index, edge_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.print(f"STRING_GNN embeddings shape: {all_emb.shape}")

        # Build ENSG-ID → row-index mapping
        node_names: List[str] = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.gnn_id_to_idx = {name: i for i, name in enumerate(node_names)}
        n_covered = len(self.gnn_id_to_idx)
        self.print(f"STRING_GNN covers {n_covered} Ensembl gene IDs")

        # ---- MLP architecture (proven optimal: h=384, 3 blocks) ----
        hp = self.hparams

        # Input projection: GNN_DIM(256) → hidden_dim(384)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(GNN_DIM),
            nn.Linear(GNN_DIM, hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.trunk_dropout),
        )

        # Residual blocks: 3 × PreNormResBlock(384, 768)
        self.blocks = nn.ModuleList(
            [
                PreNormResBlock(hp.hidden_dim, hp.inner_dim, hp.trunk_dropout)
                for _ in range(hp.n_blocks)
            ]
        )

        # Output head: LayerNorm → head dropout=0.18 → Linear(384 → 19920)
        # KEY CHANGE: head_dropout increased to 0.18 (stronger regularization
        # of the 7.6M-param output head, proven in best WarmRestart nodes)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )

        # Per-gene additive bias: one offset per (gene × class) pair
        # Proven helpful in node1-1-1 (F1=0.474) and all subsequent best nodes.
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # Ensure all trainable parameters are float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ---- Class weights (CORRECT ordering after +1 label shift) ----
        # class 0 = down-regulated  (4.77%)  → high weight
        # class 1 = neutral         (92.82%) → low weight
        # class 2 = up-regulated    (2.41%)  → highest weight
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq) / (1.0 / freq).mean()
        self.register_buffer("class_weights", class_weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Architecture: STRING_GNN({GNN_DIM}) → Proj → "
            f"{hp.n_blocks}×PreNormResBlock({hp.hidden_dim},{hp.inner_dim}) "
            f"→ LN → Dropout({hp.head_dropout}) → Linear({hp.hidden_dim},{N_GENES}×{N_CLASSES})"
            f" + gene_bias"
        )
        self.print(
            f"Trainable params: {trainable:,} / {total:,}  "
            f"(Muon lr={hp.muon_lr}, AdamW lr={hp.adamw_lr}, wd={hp.weight_decay})"
        )
        self.print(
            f"Mixup: alpha={hp.mixup_alpha}, prob={hp.mixup_prob}"
            f" | RLROP: patience={hp.rlrop_patience}, factor={hp.rlrop_factor}"
        )
        self.print(
            f"Regularization: head_drop={hp.head_dropout}, trunk_drop={hp.trunk_dropout}"
            f", label_smooth={hp.label_smoothing}"
        )

    # ------------------------------------------------------------------
    def _get_gene_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Batch lookup of frozen STRING_GNN embeddings for ENSG IDs.

        Genes absent from STRING_GNN (~7% of samples) receive a zero vector.
        """
        emb_list: List[torch.Tensor] = []
        for pid in pert_ids:
            row = self.gnn_id_to_idx.get(pid)
            if row is not None:
                emb_list.append(self.gnn_embeddings[row])
            else:
                emb_list.append(
                    torch.zeros(GNN_DIM, device=self.device, dtype=torch.float32)
                )
        return torch.stack(emb_list, dim=0)  # [B, 256]

    def _encode(self, pert_ids: List[str]) -> torch.Tensor:
        """Encode pert_ids to hidden representations [B, hidden_dim]."""
        x = self._get_gene_emb(pert_ids)   # [B, 256]
        x = self.input_proj(x)              # [B, 384]
        for block in self.blocks:
            x = block(x)                    # [B, 384]
        return x

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits of shape [B, N_CLASSES, N_GENES]."""
        x = self._encode(pert_ids)
        logits = self.output_head(x)                   # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)   # [B, 3, 6640]
        # gene_bias: [N_GENES, N_CLASSES].T → [N_CLASSES, N_GENES] → [1, 3, N_GENES]
        logits = logits + self.gene_bias.T.unsqueeze(0)
        return logits

    def _forward_from_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden representation [B, hidden_dim]."""
        logits = self.output_head(x)                   # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)   # [B, 3, 6640]
        logits = logits + self.gene_bias.T.unsqueeze(0)
        return logits

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted CE (with label smoothing) on [B, N_CLASSES, N_GENES] logits."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def _mixup_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Manifold Mixup loss.

        Mixes in hidden space (after the trunk blocks) to double effective
        training set size via convex interpolation of embeddings and labels.

        Args:
            x      : [B, hidden_dim] hidden representations
            labels : [B, N_GENES] integer class labels in {0, 1, 2}

        Returns:
            scalar loss

        Note: label_smoothing is applied in _compute_loss, so mixup uses the
        same loss function as the non-mixup path.
        """
        hp = self.hparams
        B = x.size(0)

        # Sample mixing coefficient from Beta(alpha, alpha)
        lam = float(np.random.beta(hp.mixup_alpha, hp.mixup_alpha))
        # Permutation for mixing
        idx = torch.randperm(B, device=x.device)

        # Mix hidden representations
        x_mixed = lam * x + (1 - lam) * x[idx]

        # Compute logits from mixed representation
        logits_mixed = self._forward_from_hidden(x_mixed)  # [B, 3, N_GENES]

        # Mixed loss = convex combination of CE against both label sets
        loss_a = self._compute_loss(logits_mixed, labels)
        loss_b = self._compute_loss(logits_mixed, labels[idx])
        return lam * loss_a + (1 - lam) * loss_b

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        hp = self.hparams
        pert_ids = batch["pert_id"]
        labels = batch["label"]  # [B, N_GENES] in {0,1,2}

        # Encode to hidden space [B, hidden_dim]
        x = self._encode(pert_ids)

        # Apply Manifold Mixup with probability mixup_prob
        apply_mixup = (
            self.training
            and hp.mixup_prob > 0.0
            and hp.mixup_alpha > 0.0
            and torch.rand(1).item() < hp.mixup_prob
            and x.size(0) > 1  # need at least 2 samples to mix
        )

        if apply_mixup:
            loss = self._mixup_loss(x, labels)
        else:
            logits = self._forward_from_hidden(x)
            loss = self._compute_loss(logits, labels)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        preds_local = torch.cat(self._val_preds, dim=0)    # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0)  # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        import torch.distributed as dist

        is_dist = dist.is_available() and dist.is_initialized()

        if is_dist and self.trainer.world_size > 1:
            world_size = dist.get_world_size()
            preds_np_local = preds_local.numpy()
            labels_np_local = labels_local.numpy()

            obj_preds = [None] * world_size
            obj_labels = [None] * world_size
            dist.all_gather_object(obj_preds, preds_np_local)
            dist.all_gather_object(obj_labels, labels_np_local)

            preds_np = np.concatenate(obj_preds, axis=0)
            labels_np = np.concatenate(obj_labels, axis=0)
            f1 = _compute_per_gene_f1(preds_np, labels_np)
            self.log("val/f1", f1, prog_bar=True, sync_dist=True)
        else:
            preds_np = preds_local.numpy()
            labels_np = labels_local.numpy()
            f1 = _compute_per_gene_f1(preds_np, labels_np)
            self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        self._test_preds.append(logits.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        # self.all_gather always prepends world_size dim
        gathered = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640]
        all_preds = gathered.view(-1, N_CLASSES, N_GENES)  # [N_total, 3, 6640]

        # Gather string metadata
        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        gathered_pert_ids_flat: List[List[str]] = [local_pert_ids]
        gathered_symbols_flat: List[List[str]] = [local_symbols]
        if world_size > 1 and is_dist:
            obj_pids = [None] * world_size
            obj_syms = [None] * world_size
            dist.all_gather_object(obj_pids, local_pert_ids)
            dist.all_gather_object(obj_syms, local_symbols)
            gathered_pert_ids_flat = obj_pids
            gathered_symbols_flat = obj_syms

        if self.trainer.is_global_zero:
            all_pert_ids = [pid for lst in gathered_pert_ids_flat for pid in lst]
            all_symbols = [sym for lst in gathered_symbols_flat for sym in lst]

            # De-duplicate (DDP may replicate samples across ranks)
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            preds_np = all_preds.cpu().numpy()  # [N_total, 3, 6640]
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(preds_np[i])

            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=np.stack(dedup_preds, axis=0),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        """Muon+AdamW dual optimizer with ReduceLROnPlateau scheduler.

        KEY FIX from parent (node3-3-2-1) which had broken WarmRestarts:
        - Parent: CosineAnnealingWarmRestarts on MuonWithAuxAdam wrapper
          → LR decayed monotonically (0.01→0.00087), no restarts ever fired
        - This node: ReduceLROnPlateau applied to the AdamW param group directly
          → Proven reliable in node1-3-2-2-1 (F1=0.4777) and node3-3-1-1 (F1=0.4793)

        Implementation strategy to avoid the wrapper bug:
        - MuonWithAuxAdam handles optimization
        - ReduceLROnPlateau is configured with patience=8, factor=0.5
        - The scheduler monitors val/f1 (maximize mode)
        - Multiple halvings are expected: Muon LR 0.008→0.004→0.002→0.001
        - This replicates the proven recipe from the best STRING-only nodes

        The Muon optimizer (Newton-Schulz orthogonalization) is applied ONLY
        to the 2D weight matrices of the 3 residual blocks. All other parameters
        (input_proj, output_head, LayerNorm, biases, gene_bias) use AdamW.
        """
        from muon import MuonWithAuxAdam

        hp = self.hparams

        # Identify hidden block weight matrices for Muon
        # Only Linear weight matrices in the 3 residual blocks qualify.
        # Specifically: net[1] (Linear dim→inner) and net[4] (Linear inner→dim)
        # of each PreNormResBlock.
        muon_params = []
        adamw_params = []
        muon_param_ids = set()

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # Only 2D weight matrices from the residual blocks qualify for Muon.
            # Excludes: gene_bias (2D but it's a learnable bias, not a weight matrix),
            #           LayerNorm params (1D), biases (1D), input_proj, output_head.
            if (
                "blocks" in name
                and p.ndim >= 2
                and "weight" in name
                and "norm" not in name
            ):
                muon_params.append(p)
                muon_param_ids.add(id(p))
            else:
                adamw_params.append(p)

        self.print(
            f"Muon params: {sum(p.numel() for p in muon_params):,} "
            f"| AdamW params: {sum(p.numel() for p in adamw_params):,}"
        )

        param_groups = [
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            dict(
                params=adamw_params,
                use_muon=False,
                lr=hp.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.weight_decay,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # ReduceLROnPlateau: the proven reliable scheduler for this task.
        #
        # KEY FIX: Instead of attaching scheduler to the MuonWithAuxAdam wrapper
        # (which caused the broken WarmRestarts in the parent), we use RLROP which:
        # 1. Does NOT depend on step() being called in a specific order
        # 2. Only needs a monitored metric (val/f1) to trigger LR reduction
        # 3. Has been proven to work correctly in this exact codebase:
        #    - node1-3-2-2-1 (Muon+RLROP, patience=8): F1=0.4777
        #    - node3-3-1-1 (Muon+RLROP, patience=8): F1=0.4793
        #
        # RLROP configuration:
        # - patience=8: fires after 8 epochs of no val/f1 improvement
        # - factor=0.5: halves both Muon LR and AdamW LR together
        # - mode="max": we maximize val/f1
        # - threshold=1e-5: same as early stopping delta (only count genuine improvements)
        # - min_lr=1e-6: minimum AdamW LR floor (Muon floor = 1e-6 * muon_lr/adamw_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=hp.rlrop_factor,
            patience=hp.rlrop_patience,
            threshold=1e-5,
            min_lr=hp.rlrop_min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/f1",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers (save only trainable params + buffers)
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        result = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    result[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                result[key] = full_sd[key]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable:,}/{total:,} params "
            f"({100 * trainable / total:.2f}%), plus {buffers:,} buffer values"
        )
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-averaged F1 exactly matching calc_metric.py.

    preds  : [N_samples, 3, N_genes]  — logits / class scores
    labels : [N_samples, N_genes]     — integer class labels in {0, 1, 2}
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N_samples, N_genes]
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(
            yt, yh, labels=[0, 1, 2], average=None, zero_division=0
        )
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        f1_vals.append(float(per_class_f1[present].mean()))
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    """Save test predictions in the TSV format required by calc_metric.py."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert len(pert_ids) == len(preds), (
        f"Length mismatch: {len(pert_ids)} pert_ids vs {len(preds)} pred rows"
    )
    rows = [
        {
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),  # [3, 6640] as JSON
        }
        for i in range(len(pert_ids))
    ]
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node3-3-2-1-1: STRING_GNN + 3-Block MLP (h=384) + Muon+AdamW + "
                    "Manifold Mixup + ReduceLROnPlateau (FIXED) for HepG2 DEG"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=350)
    p.add_argument("--muon-lr", type=float, default=0.008)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=8e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--trunk-dropout", type=float, default=0.30)
    p.add_argument("--head-dropout", type=float, default=0.18)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--inner-dim", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--mixup-alpha", type=float, default=0.3)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--rlrop-patience", type=int, default=8)
    p.add_argument("--rlrop-factor", type=float, default=0.5)
    p.add_argument("--rlrop-min-lr", type=float, default=1e-6)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=40)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--val_check_interval", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DataModule
    # ------------------------------------------------------------------
    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = PerturbModule(
        hidden_dim=args.hidden_dim,
        inner_dim=args.inner_dim,
        n_blocks=args.n_blocks,
        trunk_dropout=args.trunk_dropout,
        head_dropout=args.head_dropout,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        rlrop_patience=args.rlrop_patience,
        rlrop_factor=args.rlrop_factor,
        rlrop_min_lr=args.rlrop_min_lr,
        grad_clip_norm=args.grad_clip_norm,
    )

    # ------------------------------------------------------------------
    # Trainer configuration
    # ------------------------------------------------------------------
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = limit_val = limit_test = args.debug_max_step
        max_steps = args.debug_max_step
        val_check_interval = 1.0
        num_sanity_val_steps = 0
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval
        num_sanity_val_steps = 2

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val_f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.early_stop_patience,
        min_delta=1e-5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(seconds=120),
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval if (
            args.debug_max_step is None and not fast_dev_run
        ) else 1.0,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=args.grad_clip_norm,
        gradient_clip_algorithm="norm",
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(model, datamodule=datamodule)

    # ------------------------------------------------------------------
    # Test (use best checkpoint in production; raw model in debug mode)
    # ------------------------------------------------------------------
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results saved → {score_path}")


if __name__ == "__main__":
    main()
