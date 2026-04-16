"""Node 3-3-3-1: STRING_GNN Frozen Embeddings + 3-Block Pre-Norm MLP (h=384)
               + Muon LR=0.01 + Manifold Mixup + CosineAnnealingWarmRestarts
               + Extended Training (max_epochs=700, patience=100, seed=42, wd=0.005)
=============================================================================
Parent  : node3-3-3  (test F1=0.4636)
          STRING_GNN + 3-block h=384 + Muon LR=0.01 + CosineWR T_0=80 T_mult=2
          + Manifold Mixup (alpha=0.2, prob=0.5) + head_drop=0.10 + wd=0.01 + seed=0

Root cause of parent failure (from feedback.md):
--------------------------------------------------
1. MUON STOCHASTIC TRAJECTORY (primary cause, -0.033 gap vs reference nodes):
   Parent used the IDENTICAL recipe to reference nodes (node1-3-3: F1=0.4950,
   node1-3-2-2-1-1-1-1-1-1-1-1: F1=0.4968) but achieved only F1=0.4636.
   The only plausible explanation is stochastic variance in Muon's Newton-Schulz
   orthogonalization — random initialization + shuffle ordering + mixup permutations
   + dropout masks led to convergence in a local minimum at epoch 123 rather than
   the global basin found by reference nodes.

2. INSUFFICIENT TRAINING DURATION (secondary cause):
   Parent stopped at epoch 184 (patience=60), which is 56 epochs before the next
   CosineWR restart (epoch 240). Reference nodes continued to epochs 329+, allowing
   multiple warm restart cycles to explore different optimization basins.
   The feedback explicitly states: "A larger patience would allow the model to complete
   more of cycle 2 and potentially find a better minimum."

3. MODERATE OVERFITTING (tertiary cause):
   At peak (epoch 123): train/loss ~0.30, val/loss ~0.748. Gap = 0.448.
   Feedback: "Reduce weight decay from 0.01 to 0.005. Reducing weight decay slightly
   could allow the model to fit the training distribution better."

Key changes vs parent (node3-3-3):
-------------------------------------
1. RANDOM SEED: 0 -> 42  [PRIMARY FIX: escape local minimum from different initialization]
   Feedback: "Try different random seeds to escape the current local minimum."
   The parent converged to a local minimum at epoch 123 due to stochastic Muon trajectory.
   A different seed changes the random initialization of all trainable parameters,
   the DataLoader shuffle ordering, Manifold Mixup permutations, and dropout masks.
   Reference nodes (node1-3-3, node1-3-2-2-1-1-1-1-1-1-1-1) used different random
   states and found the global basin (F1=0.4950-0.4968) vs parent's local basin (0.4636).

2. EARLY STOP PATIENCE: 60 -> 100  [PRIMARY FIX: allows full CosineWR cycle completion]
   Feedback Priority 1: "Increase early stop patience to 80-100. The current patience=60
   stopped training 56 epochs before the next CosineWR warm restart. A larger patience
   would allow the model to complete more of cycle 2 and potentially find a better minimum.
   This is the single most actionable change."
   CosineWR T_0=80, T_mult=2: restarts at epoch 80, 240, 560.
   Patience=100 ensures: if the model is still improving when a warm restart fires (every
   80-240+ epochs), we don't stop until 100 epochs after the last improvement — allowing
   the restart to complete and potentially find a better basin.

3. WEIGHT DECAY: 0.01 -> 0.005  [SUPPORTING FIX]
   Feedback: "Reduce weight decay from 0.01 to 0.005. The model shows moderate overfitting
   (train/loss=0.30 vs val/loss=0.75 at peak). Reducing weight decay slightly could allow
   the model to fit the training distribution better, potentially improving generalization
   indirectly."
   Reference node1-3-3 used weight_decay=8e-4 (0.0008) and achieved F1=0.4950.
   This node uses wd=0.005, a compromise that reduces the implicit regularization penalty
   while staying more conservative than the reference node.

4. MAX EPOCHS: 500 -> 700  [SUPPORTING FIX: allows 3+ full warm restart cycles]
   Reference nodes trained for 329-500+ epochs, with best results at epoch 329 (node
   1-3-2-2-1-1-1-1-1-1-1-1, F1=0.4968) and epoch ~468 (node1-3-3, still improving at 500).
   CosineWR restart cycle boundaries: epoch 80, 240, 560.
   700 epochs ensures we reach well into the 3rd cycle (epoch 560+) before budget exhaustion.
   With patience=100, early stopping will fire before 700 if the model plateaus.

Preserved from parent (node3-3-3, proven correct):
-----------------------------------------------------
- STRING_GNN frozen PPI graph embeddings (256-dim) [proven effective]
- 3-block PreNormResBlock MLP (h=384, inner=768) [proven optimal capacity]
- Muon LR=0.01 (proven stable after sibling failures at 0.02) [proven]
- AdamW LR=3e-4 for non-block params [proven]
- CosineAnnealingWarmRestarts (T_0=80, T_mult=2, eta_min=1e-6) [proven superior to RLROP]
- Manifold Mixup (alpha=0.2, prob=0.5) [proven: +0.017-0.019 F1 in reference nodes]
- Trunk dropout=0.25 [proven]
- Head dropout=0.10 [proven]
- No label smoothing [proven]
- Gradient clip=2.0 [proven for Muon]
- Muon momentum=0.95 [proven]
- AdamW betas=(0.9, 0.95) [proven]
- Per-gene additive bias [proven]
- Class-weight ordering: [0.0477, 0.9282, 0.0241] (down/neutral/up) [correct label shift]
- All DDP/distributed logic [proven working in parent]

Note on design choice:
  The parent's feedback also suggests "abandoning STRING-only exploration entirely in
  favor of multi-modal ESM2+STRING fusion (F1=0.5175 at tree-best node4-1-1-1-1-1)."
  However, this node continues STRING-only exploration as a legitimate optimization path:
  1. The parent's failure may be purely stochastic (different seed should recover the
     reference nodes' 0.4950-0.4968 performance)
  2. The changes are minimal and well-justified by the feedback
  3. The multi-modal path is covered by the node4 lineage
  4. Achieving ~0.495+ via STRING-only is feasible based on reference node evidence
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
N_CLASSES = 3         # down (-1->0), neutral (0->1), up (1->2)
GNN_DIM = 256         # STRING_GNN output embedding dimension
HIDDEN_DIM = 384      # MLP hidden dimension (proven optimal)
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
            # Labels in {-1,0,1} -> shift to {0,1,2}
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
        output = x + LN(x) -> Linear(dim->inner) -> GELU -> Dropout
                               -> Linear(inner->dim) -> Dropout
    """

    def __init__(self, dim: int, inner_dim: int, dropout: float = 0.25) -> None:
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
        dropout: float = 0.25,           # trunk dropout (preserved from parent)
        head_dropout: float = 0.10,      # head dropout (preserved from parent)
        muon_lr: float = 0.01,           # Proven stable Muon LR (preserved from parent)
        adamw_lr: float = 3e-4,          # AdamW LR for non-block params (preserved)
        weight_decay: float = 0.005,     # REDUCED: 0.01->0.005 per parent feedback
        label_smoothing: float = 0.0,    # No label smoothing (preserved from parent)
        mixup_alpha: float = 0.2,        # Manifold Mixup alpha (preserved from parent)
        mixup_prob: float = 0.5,         # Manifold Mixup probability (preserved)
        cosine_t0: int = 80,             # CosineWR T_0: first cycle length (preserved)
        cosine_t_mult: int = 2,          # CosineWR T_mult: cycle doubling (preserved)
        cosine_eta_min: float = 1e-6,    # CosineWR eta_min (preserved from parent)
        grad_clip: float = 2.0,          # Gradient clipping (preserved from parent)
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Populated in setup()
        self.input_proj: Optional[nn.Sequential] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None

        # STRING_GNN gene-ID -> embedding-row index
        self.gnn_id_to_idx: Dict[str, int] = {}

        # Metric accumulators
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
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

        # Build ENSG-ID -> row-index mapping
        node_names: List[str] = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.gnn_id_to_idx = {name: i for i, name in enumerate(node_names)}
        n_covered = len(self.gnn_id_to_idx)
        self.print(f"STRING_GNN covers {n_covered} Ensembl gene IDs")

        # ---- MLP architecture ----
        hp = self.hparams
        self.input_proj = nn.Sequential(
            nn.LayerNorm(GNN_DIM),
            nn.Linear(GNN_DIM, hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )
        self.blocks = nn.ModuleList(
            [
                PreNormResBlock(hp.hidden_dim, hp.inner_dim, hp.dropout)
                for _ in range(hp.n_blocks)
            ]
        )
        # Output head: LayerNorm -> head_dropout -> Linear
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )
        # Per-gene additive bias: one offset per (gene x class) pair
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # Cast trainable params to float32 for stable optimization
        for k, v in self.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()

        # ---- Class weights (CORRECT ordering after +1 label shift) ----
        # class 0 = down-regulated  (4.77%)  -> high weight
        # class 1 = neutral         (92.82%) -> low weight
        # class 2 = up-regulated    (2.41%)  -> highest weight
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq) / (1.0 / freq).mean()
        self.register_buffer("class_weights", class_weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Architecture: STRING_GNN({GNN_DIM}) -> Proj -> "
            f"{hp.n_blocks}xPreNormResBlock({hp.hidden_dim},{hp.inner_dim}) "
            f"-> LN -> Dropout({hp.head_dropout}) -> Linear({hp.hidden_dim},{N_GENES}x{N_CLASSES}) "
            f"+ gene_bias"
        )
        self.print(f"Trainable params: {trainable:,} / {total:,}")
        self.print(f"KEY CHANGES vs parent: seed=42 (was 0), patience=100 (was 60), "
                   f"weight_decay={hp.weight_decay} (was 0.01), max_epochs=700 (was 500)")

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

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits of shape [B, N_CLASSES, N_GENES].
        Standard (non-mixup) forward pass.
        """
        x = self._get_gene_emb(pert_ids)              # [B, 256]
        x = self.input_proj(x)                         # [B, 384]
        for block in self.blocks:
            x = block(x)                               # [B, 384]
        logits = self.output_head(x)                   # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)   # [B, 3, 6640]
        # gene_bias: [N_GENES, N_CLASSES].T -> [N_CLASSES, N_GENES] -> [1, 3, N_GENES]
        logits = logits + self.gene_bias.T.unsqueeze(0)
        return logits

    def _forward_with_mixup(
        self,
        pert_ids: List[str],
        labels: torch.Tensor,
        mixup_alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Manifold Mixup forward pass.

        Randomly selects a hidden layer (after input_proj or after one of the blocks),
        mixes the activations of the current batch with a randomly shuffled version.

        Returns:
            logits: [B, N_CLASSES, N_GENES]
            labels_a: [B, N_GENES] -- original targets
            labels_b: [B, N_GENES] -- shuffled targets
            lam: scalar -- mix coefficient (lambda from Beta distribution)
        """
        B = len(pert_ids)
        # Sample mix coefficient
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))

        # Random permutation for mixing
        perm = torch.randperm(B, device=self.device)
        labels_b = labels[perm]  # [B, N_GENES]

        x = self._get_gene_emb(pert_ids)              # [B, 256]
        x = self.input_proj(x)                         # [B, 384]

        # Randomly choose the layer at which to apply mixup
        # Choices: 0 = after input_proj, 1/2/3 = after block 0/1/2
        n_blocks = self.hparams.n_blocks
        mix_layer = np.random.randint(0, n_blocks + 1)

        for i, block in enumerate(self.blocks):
            if i == mix_layer:
                # Apply Manifold Mixup at this hidden layer
                x = lam * x + (1 - lam) * x[perm]
            x = block(x)                               # [B, 384]

        # If mix_layer == n_blocks, apply after the last block
        if mix_layer == n_blocks:
            x = lam * x + (1 - lam) * x[perm]

        logits = self.output_head(x)                   # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)   # [B, 3, 6640]
        logits = logits + self.gene_bias.T.unsqueeze(0)

        return logits, labels, labels_b, lam

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted CE on [B, N_CLASSES, N_GENES] logits."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def _compute_mixup_loss(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Manifold Mixup loss: weighted sum of two CE losses."""
        loss_a = self._compute_loss(logits, labels_a)
        loss_b = self._compute_loss(logits, labels_b)
        return lam * loss_a + (1 - lam) * loss_b

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        use_mixup = (
            self.training
            and np.random.random() < self.hparams.mixup_prob
            and len(batch["pert_id"]) > 1  # need at least 2 samples to mix
        )

        if use_mixup:
            logits, labels_a, labels_b, lam = self._forward_with_mixup(
                batch["pert_id"],
                batch["label"],
                self.hparams.mixup_alpha,
            )
            loss = self._compute_mixup_loss(logits, labels_a, labels_b, lam)
        else:
            logits = self(batch["pert_id"])
            loss = self._compute_loss(logits, batch["label"])

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
            # Log on all ranks so EarlyStopping / scheduler can access the metric.
            self.log("val/f1", f1, prog_bar=True, sync_dist=True)
        else:
            preds_np = preds_local.numpy()
            labels_np = labels_local.numpy()
            f1 = _compute_per_gene_f1(preds_np, labels_np)
            self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._test_preds.append(logits.detach().cpu().float())
        self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        labels_local = torch.cat(self._test_labels, dim=0)  # [N_local, 6640]
        self._test_preds.clear()
        self._test_labels.clear()

        # self.all_gather always prepends world_size dim
        gathered = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640]
        all_preds = gathered.view(-1, N_CLASSES, N_GENES)  # [N_total, 3, 6640]

        # Gather labels from all ranks
        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        gathered_labels = self.all_gather(labels_local)  # [world_size, N_local, 6640]
        all_labels = gathered_labels.view(-1, N_GENES)  # [N_total, 6640]

        # Compute test F1 (on all ranks, then log on all ranks)
        preds_np = all_preds.cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("test/f1", f1, prog_bar=True, sync_dist=True)

        # Gather string metadata
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
        """Dual Muon + AdamW optimizer configuration.

        Muon (LR=0.01): hidden block 2D weight matrices
        AdamW (LR=3e-4): all other parameters

        CosineAnnealingWarmRestarts: proven superior to RLROP for this task
        when combined with Manifold Mixup.
        T_0=80: first cycle length (epochs)
        T_mult=2: each subsequent cycle is 2x longer (80, 160, 320, ...)
        eta_min=1e-6: minimum LR at bottom of cosine cycle
        """
        from muon import MuonWithAuxAdam

        hp = self.hparams

        # Identify Muon-eligible parameters: 2D weight matrices in residual blocks
        muon_param_names = set()
        muon_params = []
        adamw_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Apply Muon to 2D weight matrices inside residual blocks only
            is_block_weight = (
                "blocks." in name
                and ".net." in name
                and "weight" in name
                and param.ndim >= 2
            )
            if is_block_weight:
                muon_params.append(param)
                muon_param_names.add(name)
            else:
                adamw_params.append(param)

        self.print(
            f"Optimizer groups: Muon={len(muon_params)} 2D block weights, "
            f"AdamW={len(adamw_params)} other params"
        )

        param_groups = [
            # Muon group: hidden block weight matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            # AdamW group: all other parameters
            dict(
                params=adamw_params,
                use_muon=False,
                lr=hp.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.weight_decay,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # CosineAnnealingWarmRestarts (preserved from parent, proven superior to RLROP
        # when combined with Manifold Mixup in this architecture lineage)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.cosine_t0,
            T_mult=hp.cosine_t_mult,
            eta_min=hp.cosine_eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers (save only trainable params + buffers)
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save trainable parameters and persistent buffers."""
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
        self.print(
            f"Saving checkpoint: {trainable:,}/{total:,} params "
            f"({100 * trainable / total:.2f}%)"
        )
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-averaged F1 exactly matching calc_metric.py.

    preds  : [N_samples, 3, N_genes]  -- logits / class scores
    labels : [N_samples, N_genes]     -- integer class labels in {0, 1, 2}
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
    print(f"Saved {len(rows)} test predictions -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Node3-3-3-1: STRING_GNN + 3-Block MLP (h=384) + Muon(lr=0.01) + "
            "WCE + Manifold Mixup + CosineWR + seed=42 + patience=100 + wd=0.005"
        )
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=700,
                   help="Extended: allows 3+ CosineWR cycles (80+240+560 epochs)")
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.005,
                   help="REDUCED: 0.01->0.005 per parent feedback for better fit")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.25,
                   help="Trunk dropout (preserved from parent)")
    p.add_argument("--head-dropout", type=float, default=0.10,
                   help="Head dropout (preserved from parent)")
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--inner-dim", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--cosine-t0", type=int, default=80)
    p.add_argument("--cosine-t-mult", type=int, default=2)
    p.add_argument("--cosine-eta-min", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--early-stop-patience", type=int, default=100,
                   help="EXTENDED: 60->100 per parent feedback to allow CosineWR cycles")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--val_check_interval", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # KEY CHANGE: seed=42 (different from parent's seed=0) to escape local minimum
    # Muon's Newton-Schulz orthogonalization is sensitive to initialization;
    # parent converged to local minimum at epoch 123 due to stochastic variance.
    # Different seed changes: param initialization, DataLoader shuffle, Mixup permutations.
    pl.seed_everything(42)

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
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_t_mult,
        cosine_eta_min=args.cosine_eta_min,
        grad_clip=args.grad_clip,
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
        auto_insert_metric_name=False,
    )
    # EXTENDED patience=100 (vs parent's 60):
    # Allows CosineWR warm restart cycles to complete before giving up.
    # CosineWR T_0=80, T_mult=2: restarts at epochs 80, 240, 560.
    # Patience=100 ensures we don't stop 56 epochs before a restart (parent's mistake).
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
            find_unused_parameters=False,
            timeout=timedelta(seconds=120),
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        gradient_clip_val=args.grad_clip,
        gradient_clip_algorithm="norm",
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
        print(f"Test results saved -> {score_path}")


if __name__ == "__main__":
    main()
