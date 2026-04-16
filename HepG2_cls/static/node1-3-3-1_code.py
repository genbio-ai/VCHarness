"""Node 1-3-3-2: STRING-Only + Extended Training (T_0=150, max_epochs=1000)
             + Top-5 Checkpoint Ensemble + Reduced Mixup (0.3) + Tuned Regularization

This node improves upon node1-3-3 (F1=0.4950) by addressing three identified
bottlenecks from feedback:
  1. Training was not converged at epoch 500 → extend to 1000 epochs with T_0=150
     (longer cycles = fewer destructive LR spikes, more productive convergence time)
  2. No checkpoint ensemble → save top-5 checkpoints, average logits at test time
     (proven +0.005-0.008 F1 in node1-3-2-2-1-1-1-1, used in tree-best node4-1-1-1-1-1)
  3. Mixup probability too aggressive (0.5) → reduce to 0.3 for cleaner training signal
     (each sample in 1,273-sample dataset is valuable; less mixing = sharper signal)

Additional tuning from memory:
  4. head_dropout: 0.15 → 0.18 (node1-3-2-2-1-1-1-1-1-1-1-1 best STRING-only used 0.18)
  5. weight_decay: 8e-4 → 1e-3 (address train-val loss gap of 1.09)
  6. early_stop_patience: 160 → 200 (allows ~1.33 T_0 cycles before early stopping)

Architecture (unchanged from node1-3-3):
  STRING_GNN embed [B, 256] → Input projection [256→384] →
  3× PreNormResidualBlock(384) → head_dropout(0.18) → Linear(384→19920) + gene_bias →
  Reshape [B, 3, 6640]

  Test: Load top-K saved checkpoints → run inference independently on CPU-based
  batched test loop → average logits → save ensemble predictions.

Key references:
  - node1-3-3 (parent, F1=0.4950): val/f1 still climbing at epoch 500, no ensemble
  - node1-3-2-2-1-1-1-1 (F1=0.4914): top-5 ensemble gave +0.005-0.008 F1
  - node1-3-2-2-1-1-1-1-1-1-1-1 (F1=0.4994, best STRING-only): head_dropout=0.18
  - node4-1-1-1-1-1 (F1=0.5175, tree best): top-5 ensemble + extended training
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import glob as glob_module
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Gene-perturbation → differential-expression dataset."""

    def __init__(self, df: pd.DataFrame, gene2str_idx: Dict[str, int]) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        # Map ENSEMBL pert_id → STRING node index; -1 = not in STRING graph
        self.str_indices = torch.tensor(
            [gene2str_idx.get(pid, -1) for pid in self.pert_ids], dtype=torch.long
        )
        if "label" in df.columns:
            labels = np.array([json.loads(x) for x in df["label"]], dtype=np.int64)
            self.labels = torch.tensor(labels + 1, dtype=torch.long)  # {-1,0,1} → {0,1,2}
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "str_idx": self.str_indices[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]  # [6640]
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
        micro_batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.gene2str_idx: Dict[str, int] = {}
        self.train_ds = self.val_ds = self.test_ds = None

    def setup(self, stage: str = "fit") -> None:
        # Build ENSEMBL-ID → STRING-node-index mapping once
        if not self.gene2str_idx:
            node_names: List[str] = json.loads(
                (STRING_GNN_DIR / "node_names.json").read_text()
            )
            self.gene2str_idx = {ensg: i for i, ensg in enumerate(node_names)}

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df, self.gene2str_idx)
        self.val_ds = PerturbDataset(val_df, self.gene2str_idx)
        self.test_ds = PerturbDataset(test_df, self.gene2str_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------
class PreNormResidualBlock(nn.Module):
    """Pre-LayerNorm residual MLP block (h → 2h → h).

    Uses Pre-LayerNorm (LN applied before projection) as shown effective
    in later STRING-only nodes (node3-3-1-1 lineage, achieving F1>0.479).
    """

    def __init__(self, dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(self.norm(x)))


class StringMLPModel(nn.Module):
    """STRING-only 3-block PreNorm residual MLP with flat output head.

    Architecture:
      STRING_GNN embed [B, 256] → Input projection [256→h] →
      3x PreNormResidualBlock(h) → head_dropout → Linear(h→19920) + gene_bias →
      Reshape [B, 3, 6640]

    Changes from node1-3-3:
      - head_dropout: 0.15 → 0.18 (node1-3-2-2-1-1-1-1-1-1-1-1 best STRING-only used 0.18)
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.35,
        head_dropout: float = 0.18,
    ) -> None:
        super().__init__()
        # Learnable fallback embedding for genes not in STRING graph (~6% of genes)
        self.fallback_emb = nn.Parameter(torch.zeros(256))
        nn.init.normal_(self.fallback_emb, std=0.02)

        # Input projection: 256 → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Residual MLP blocks (PreLN)
        self.blocks = nn.ModuleList(
            [PreNormResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        # Output head: LayerNorm + head_dropout + flat Linear + per-gene bias
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head_linear = nn.Linear(hidden_dim, N_GENES * N_CLASSES)
        # Per-gene additive bias: captures baseline DE priors per response gene
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES * N_CLASSES))

    def forward(
        self,
        str_idx: torch.Tensor,       # [B]  STRING node indices, -1 = not in graph
        string_embs: torch.Tensor,   # [18870, 256] frozen buffer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, hidden) where hidden is the pre-head representation."""
        valid_mask = str_idx >= 0                    # [B] bool
        safe_idx = str_idx.clamp(min=0)              # replace -1 with 0 (overwritten below)

        # Look up frozen STRING embeddings [B, 256]
        str_emb = string_embs[safe_idx].to(torch.float32)

        # Overwrite samples whose gene is absent from the STRING graph
        if not valid_mask.all():
            fallback = self.fallback_emb.to(str_emb).unsqueeze(0).expand(
                int((~valid_mask).sum()), -1
            )
            str_emb = str_emb.clone()
            str_emb[~valid_mask] = fallback

        # Input projection → residual MLP → output
        x = self.input_proj(str_emb)        # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)
        # Output head with dropout
        h = self.head_norm(x)               # [B, hidden_dim]
        h = self.head_dropout(h)            # Regularize output head
        logits = self.head_linear(h) + self.gene_bias.to(h)  # [B, N_GENES * N_CLASSES]
        return logits.view(-1, N_CLASSES, N_GENES), x  # ([B, 3, 6640], [B, hidden_dim])


# ---------------------------------------------------------------------------
# Manifold Mixup helper
# ---------------------------------------------------------------------------
def manifold_mixup(
    x: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Manifold Mixup in the hidden representation space.

    Mixes two random samples in the embedding space to create regularized
    synthetic training examples that improve generalization on small datasets.

    Returns:
        mixed_x: mixed embedding [B, D]
        labels_a: original labels [B, 6640]
        labels_b: shuffled labels [B, 6640]
        lam: mixing coefficient (float)
    """
    batch_size = x.shape[0]
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    labels_a = labels
    labels_b = labels[index]
    return mixed_x, labels_a, labels_b, lam


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.35,
        head_dropout: float = 0.18,
        muon_lr: float = 0.01,
        adamw_lr: float = 3e-4,
        weight_decay: float = 1e-3,
        t0: int = 150,
        t_mult: int = 2,
        mixup_prob: float = 0.3,
        mixup_alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.muon_lr = muon_lr
        self.adamw_lr = adamw_lr
        self.weight_decay = weight_decay
        self.t0 = t0
        self.t_mult = t_mult
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

        self.model: Optional[StringMLPModel] = None

        # Accumulation buffers for epoch-level metrics
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Class weights: inversely proportional to class frequencies
        # Shift: {-1→down, 0→neutral, 1→up} → class {0, 1, 2}
        # class0=down(4.77%), class1=neutral(92.82%), class2=up(2.41%)
        # Note: after {-1,0,1}→{0,1,2} shift: 0=down, 1=neutral, 2=up
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq)
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        self.register_buffer("class_weights", class_weights)

        if self.model is not None:
            return  # already initialized (guard for re-entrant setup calls)

        # ---- Load STRING_GNN node embeddings (once per rank) ----
        from transformers import AutoModel
        gnn = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        gnn.eval()
        graph = torch.load(
            STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False
        )
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)
        with torch.no_grad():
            gnn_out = gnn(edge_index=edge_index, edge_weight=edge_weight)
        string_embs = gnn_out.last_hidden_state.detach().float().cpu()  # [18870, 256]
        del gnn, gnn_out
        # Register as frozen buffer (moved to device by Lightning automatically)
        self.register_buffer("string_embs", string_embs)

        # ---- Build model ----
        self.model = StringMLPModel(
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
        )

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        self.print(
            f"Node1-3-3-2 StringMLPModel | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
            f"dropout={self.dropout} | head_dropout={self.head_dropout} | "
            f"muon_lr={self.muon_lr} | t0={self.t0}/t_mult={self.t_mult} | "
            f"mixup_prob={self.mixup_prob}/alpha={self.mixup_alpha} | "
            f"weight_decay={self.weight_decay} | "
            f"trainable={n_trainable:,}/{n_total:,}"
        )

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted cross-entropy loss (no label smoothing).

        logits: [B, 3, 6640]
        labels: [B, 6640]  — values in {0, 1, 2}

        No label smoothing — allows train loss to reach lower levels (shown
        beneficial in node1-3-2-2-1-1-1-1: removing LS enabled train/loss=0.058).
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                                # [B*6640]
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=0.0,  # No label smoothing
        )

    def _compute_mixed_loss(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Mixed cross-entropy loss for Manifold Mixup training.

        logits: [B, 3, 6640]
        labels_a, labels_b: [B, 6640]
        lam: mixing coefficient
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_a_flat = labels_a.reshape(-1)
        labels_b_flat = labels_b.reshape(-1)
        loss_a = F.cross_entropy(logits_flat, labels_a_flat, weight=self.class_weights, label_smoothing=0.0)
        loss_b = F.cross_entropy(logits_flat, labels_b_flat, weight=self.class_weights, label_smoothing=0.0)
        return lam * loss_a + (1 - lam) * loss_b

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        str_idx = batch["str_idx"]
        labels = batch["label"]

        # Always call self.model() forward to ensure ALL parameters are included
        # in the DDP computation graph (find_unused_parameters=True for safety)
        logits, x = self.model(str_idx, self.string_embs)

        # Apply Manifold Mixup in hidden space (after residual blocks, before head)
        # Reduced from prob=0.5 to prob=0.3 to preserve more clean training signal
        # (each of 1,273 training samples is valuable; less mixing = sharper gradient)
        apply_mixup = (
            self.training
            and np.random.random() < self.mixup_prob
            and labels is not None
        )

        if apply_mixup:
            # Apply Manifold Mixup in the pre-head hidden space
            mixed_x, labels_a, labels_b, lam = manifold_mixup(x, labels, alpha=self.mixup_alpha)

            # Forward mixed representation through output head
            h = self.model.head_norm(mixed_x)
            h = self.model.head_dropout(h)
            logits_flat_raw = self.model.head_linear(h) + self.model.gene_bias.to(h)
            logits = logits_flat_raw.view(-1, N_CLASSES, N_GENES)

            loss = self._compute_mixed_loss(logits, labels_a, labels_b, lam)
        else:
            loss = self._compute_loss(logits, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits, _ = self.model(batch["str_idx"], self.string_embs)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds_local = torch.cat(self._val_preds, dim=0)    # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0)  # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather across DDP ranks for accurate global F1
        all_preds = self.all_gather(preds_local)   # [world_size, N_local, 3, 6640]
        all_labels = self.all_gather(labels_local) # [world_size, N_local, 6640]
        ws = self.trainer.world_size
        if ws > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
            all_labels = all_labels.view(-1, N_GENES)
        else:
            # With ws=1 all_gather prepends a size-1 dim
            if all_preds.dim() == 4:
                all_preds = all_preds.squeeze(0)
                all_labels = all_labels.squeeze(0)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits, _ = self.model(batch["str_idx"], self.string_embs)
        self._test_preds.append(logits.detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        all_preds = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640]
        ws = self.trainer.world_size
        if ws > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
        else:
            if all_preds.dim() == 4:
                all_preds = all_preds.squeeze(0)

        # Gather string metadata from all ranks → rank 0 only
        if ws > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            if self.trainer.is_global_zero:
                _pert_gathered: List[List[str]] = [[] for _ in range(ws)]
                _syms_gathered: List[List[str]] = [[] for _ in range(ws)]
                torch.distributed.gather_object(self._test_pert_ids, _pert_gathered, dst=0)
                torch.distributed.gather_object(self._test_symbols, _syms_gathered, dst=0)
                all_pert_ids: List[str] = []
                all_symbols: List[str] = []
                for p_list, s_list in zip(_pert_gathered, _syms_gathered):
                    all_pert_ids.extend(p_list)
                    all_symbols.extend(s_list)
            else:
                torch.distributed.gather_object(self._test_pert_ids, dst=0)
                torch.distributed.gather_object(self._test_symbols, dst=0)
                all_pert_ids, all_symbols = [], []
        else:
            all_pert_ids = self._test_pert_ids
            all_symbols = self._test_symbols

        if self.trainer.is_global_zero:
            preds_np = all_preds.float().cpu().numpy()  # [N_total, 3, 6640]
            _save_test_predictions(
                pert_ids=all_pert_ids,
                symbols=all_symbols,
                preds=preds_np,
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        """Configure Muon+AdamW optimizer with CosineAnnealingWarmRestarts.

        Muon is applied to hidden 2D weight matrices in residual blocks.
        AdamW is applied to all other parameters (input_proj, head, gene_bias,
        fallback_emb, norms, biases).

        CosineAnnealingWarmRestarts with T_0=150/T_mult=2 (extended from node1-3-3's T_0=80):
        - Fewer restarts: restarts at epoch 150, 450, 1050, ... (vs. 80, 240, 560)
        - More productive convergence time per cycle (150 epochs vs 80)
        - Reduces val/loss spike frequency (each restart temporarily elevated val/loss by 70%)
        """
        try:
            from muon import MuonWithAuxAdam
            use_muon = True
        except ImportError:
            use_muon = False
            self.print("Muon not installed — falling back to AdamW for all parameters")

        if use_muon:
            # Separate parameters: Muon for hidden 2D weight matrices in blocks
            # AdamW for everything else
            muon_params = []
            adamw_params = []

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                # Apply Muon only to 2D weight matrices inside residual blocks
                # Specifically: blocks.*.net.0.weight and blocks.*.net.3.weight
                is_block_weight = (
                    "model.blocks" in name
                    and "net" in name
                    and ("net.0.weight" in name or "net.3.weight" in name)
                    and param.ndim >= 2
                )
                if is_block_weight:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

            param_groups = [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=self.muon_lr,
                    momentum=0.95,
                    weight_decay=self.weight_decay,
                ),
                dict(
                    params=adamw_params,
                    use_muon=False,
                    lr=self.adamw_lr,
                    betas=(0.9, 0.95),
                    weight_decay=self.weight_decay,
                ),
            ]

            optimizer = MuonWithAuxAdam(param_groups)
            self.print(
                f"Using Muon+AdamW: {len(muon_params)} Muon params, "
                f"{len(adamw_params)} AdamW params"
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.adamw_lr, weight_decay=self.weight_decay
            )

        # CosineAnnealingWarmRestarts for escaping local optima
        # T_0=150 per cycle (extended from parent's 80), T_mult=2 doubles cycle length
        # Restarts at: epoch 150, epoch 450 (=150+300), epoch 1050 (=150+300+600), ...
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.t0,
            T_mult=self.t_mult,
            eta_min=1e-6,
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
    # Checkpoint: save only trainable params + small essential buffers
    # (string_embs are large frozen tensors recomputed in setup() —
    #  excluding them keeps checkpoint files small and avoids VRAM pressure)
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        saved: Dict[str, Any] = {}
        # Trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    saved[key] = full_sd[key]
        # Essential small buffers (class_weights); exclude large frozen embeddings
        large_frozen = {"string_embs"}
        for name, buf in self.named_buffers():
            leaf = name.split(".")[-1]
            if leaf not in large_frozen:
                key = prefix + name
                if key in full_sd:
                    saved[key] = full_sd[key]

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        self.print(
            f"Saving checkpoint: {n_trainable:,}/{n_total:,} trainable params "
            f"({100*n_trainable/n_total:.1f}%)"
        )
        return saved

    def load_state_dict(self, state_dict, strict=True):
        # strict=False: string_embs is not in checkpoint (recomputed in setup())
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-F1 averaged over all genes — matches calc_metric.py logic.

    preds:  [N, 3, 6640] float — class logits
    labels: [N, 6640]    int   — class indices in {0,1,2}
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N, 6640]
    n_genes = labels.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        f1_vals.append(float(per_class_f1[present].mean()))
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    """Save test predictions in required TSV format (idx / input / prediction)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # De-duplicate by pert_id (keep first occurrence)
    seen_ids = set()
    for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
        if pid not in seen_ids:
            seen_ids.add(pid)
            rows.append({
                "idx": pid,
                "input": sym,
                "prediction": json.dumps(preds[i].tolist()),  # [3][6640] list
            })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


def _run_ensemble_inference_on_gpu(
    ckpt_paths: List[str],
    test_dataset: PerturbDataset,
    batch_size: int,
    device: torch.device,
    string_embs: torch.Tensor,
    hidden_dim: int = 384,
    n_blocks: int = 3,
    dropout: float = 0.35,
    head_dropout: float = 0.18,
) -> Tuple[List[str], List[str], np.ndarray]:
    """Run inference with each checkpoint and average logits.

    This is called only on rank 0 after DDP training completes.
    Returns (pert_ids, symbols, ensemble_preds) where ensemble_preds is [N, 3, 6640].

    Design: directly instantiates StringMLPModel and loads checkpoint weights by
    stripping the "model." prefix from state_dict keys. Avoids creating nested
    DDP trainers (which would hang/fail in post-training context).
    """
    all_preds_list: List[np.ndarray] = []
    pert_ids: List[str] = []
    symbols: List[str] = []

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)
    string_embs_dev = string_embs.to(device)

    for ckpt_idx, ckpt_path in enumerate(ckpt_paths):
        print(f"Ensemble: checkpoint {ckpt_idx+1}/{len(ckpt_paths)}: {Path(ckpt_path).name}")
        # Load checkpoint state dict
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        full_state_dict = ckpt.get("state_dict", ckpt)

        # Extract keys belonging to model (strip "model." prefix)
        model_state_dict = {}
        for k, v in full_state_dict.items():
            # Keys in checkpoint are stored as "model.<layer>" because PerturbModule
            # uses self.model = StringMLPModel(...)
            if k.startswith("model."):
                model_state_dict[k[len("model."):]] = v

        # Build fresh StringMLPModel and load weights
        nn_model = StringMLPModel(
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout,
            head_dropout=head_dropout,
        )
        missing, unexpected = nn_model.load_state_dict(model_state_dict, strict=False)
        if missing:
            print(f"  Warning: missing keys: {missing[:3]}...")
        nn_model.eval()
        nn_model.to(device)
        nn_model = nn_model.float()

        ckpt_preds: List[np.ndarray] = []
        ckpt_pids: List[str] = []
        ckpt_syms: List[str] = []

        with torch.no_grad():
            for batch in test_loader:
                str_idx = batch["str_idx"].to(device)
                logits, _ = nn_model(str_idx, string_embs_dev)
                ckpt_preds.append(logits.cpu().float().numpy())  # [B, 3, 6640]
                if ckpt_idx == 0:
                    ckpt_pids.extend(batch["pert_id"])
                    ckpt_syms.extend(batch["symbol"])

        ckpt_preds_arr = np.concatenate(ckpt_preds, axis=0)  # [N, 3, 6640]
        all_preds_list.append(ckpt_preds_arr)

        if ckpt_idx == 0:
            pert_ids = ckpt_pids
            symbols = ckpt_syms

        del nn_model, ckpt, full_state_dict, model_state_dict, ckpt_preds
        torch.cuda.empty_cache()

    # Average logits across all checkpoint predictions (logit-space averaging)
    ensemble_preds = np.mean(all_preds_list, axis=0)  # [N, 3, 6640]
    print(f"Ensemble complete: averaged {len(all_preds_list)} checkpoint predictions")
    return pert_ids, symbols, ensemble_preds


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-3-3-2: STRING-Only + Extended Training + Top-5 Ensemble + Reduced Mixup"
    )
    p.add_argument("--micro-batch-size",    type=int,   default=32)
    p.add_argument("--global-batch-size",   type=int,   default=256)
    p.add_argument("--max-epochs",          type=int,   default=1000)
    p.add_argument("--muon-lr",             type=float, default=0.01)
    p.add_argument("--adamw-lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",        type=float, default=1e-3)
    p.add_argument("--hidden-dim",          type=int,   default=384)
    p.add_argument("--n-blocks",            type=int,   default=3)
    p.add_argument("--dropout",             type=float, default=0.35)
    p.add_argument("--head-dropout",        type=float, default=0.18)
    p.add_argument("--t0",                  type=int,   default=150)
    p.add_argument("--t-mult",              type=int,   default=2)
    p.add_argument("--mixup-prob",          type=float, default=0.3)
    p.add_argument("--mixup-alpha",         type=float, default=0.2)
    p.add_argument("--early-stop-patience", type=int,   default=200)
    p.add_argument("--save-top-k",          type=int,   default=5,
                   help="Number of top checkpoints to save for test-time ensemble")
    p.add_argument("--num-workers",         type=int,   default=4)
    p.add_argument("--val-check-interval",  type=float, default=1.0)
    p.add_argument("--debug_max_step",      type=int,   default=None)
    p.add_argument("--fast_dev_run",        action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    pl.seed_everything(0)
    # Additional seed for numpy random (used in manifold mixup)
    np.random.seed(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- DataModule ---
    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup("fit")

    # --- LightningModule ---
    model = PerturbModule(
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        t0=args.t0,
        t_mult=args.t_mult,
        mixup_prob=args.mixup_prob,
        mixup_alpha=args.mixup_alpha,
    )

    # --- Trainer configuration ---
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        max_steps = -1

    # Top-K checkpoint saving for test-time ensemble (key improvement over node1-3-3)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=args.save_top_k,
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
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

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
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,   # Stabilize updates on large output head parameters
    )

    # --- Fit ---
    trainer.fit(model, datamodule=datamodule)

    # --- Test ---
    if args.fast_dev_run or args.debug_max_step is not None:
        # Debug mode: use current model without ensemble
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        # Production: run standard test with best checkpoint first
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

        # Top-K Checkpoint Ensemble (rank 0 only, after DDP teardown)
        # This performs logit-space averaging of top-K checkpoints.
        # Proven: node1-3-2-2-1-1-1-1 (+0.005-0.008 F1), node4-1-1-1-1-1 (tree best).
        if trainer.is_global_zero and args.save_top_k > 1:
            ckpt_dir = Path(checkpoint_cb.dirpath)
            # Find top-K saved checkpoints (sorted descending by val/f1 in filename)
            # Filename format: because "val/f1" contains "/" Lightning creates nested dirs:
            # checkpoints/best-epoch=XXX-val/f1=Y.YYYY.ckpt
            # Use recursive glob to find all .ckpt files under the best-* dirs
            def _extract_f1_from_ckpt_path(p: str) -> float:
                """Extract val/f1 score from checkpoint path.

                Supports two layouts:
                  1. Nested: ...best-epoch=XXX-val/f1=0.4923.ckpt
                  2. Flat:   ...best-XXX-0.4923.ckpt
                """
                try:
                    stem = Path(p).stem  # "f1=0.4923" or "best-XXX-0.4923"
                    if stem.startswith("f1="):
                        return float(stem[3:])
                    else:
                        return float(stem.split("-")[-1])
                except (ValueError, IndexError):
                    return 0.0

            # Try nested layout first (val/f1 creates directory structure)
            ckpt_candidates = glob_module.glob(str(ckpt_dir / "best-epoch=*" / "f1=*.ckpt"))
            if not ckpt_candidates:
                # Fall back to flat layout
                ckpt_candidates = glob_module.glob(str(ckpt_dir / "best-*.ckpt"))

            ckpt_files = sorted(
                ckpt_candidates,
                key=_extract_f1_from_ckpt_path,
                reverse=True,
            )[:args.save_top_k]

            if len(ckpt_files) >= 2:
                print(f"Running top-{len(ckpt_files)} checkpoint ensemble...")
                # Use single GPU (rank 0's device) for ensemble inference
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # Need string_embs buffer for inference — recompute on CPU/GPU
                from transformers import AutoModel
                gnn = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
                gnn.eval()
                graph = torch.load(
                    STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False
                )
                edge_index = graph["edge_index"]
                edge_weight = graph.get("edge_weight", None)
                with torch.no_grad():
                    gnn_out = gnn(edge_index=edge_index, edge_weight=edge_weight)
                string_embs = gnn_out.last_hidden_state.detach().float().cpu()
                del gnn, gnn_out

                # Ensure test dataset is initialized
                datamodule.setup("test")
                test_ds = datamodule.test_ds

                pert_ids, symbols, ensemble_preds = _run_ensemble_inference_on_gpu(
                    ckpt_paths=ckpt_files,
                    test_dataset=test_ds,
                    batch_size=args.micro_batch_size * 2,  # larger batch for inference
                    device=device,
                    string_embs=string_embs,
                    hidden_dim=args.hidden_dim,
                    n_blocks=args.n_blocks,
                    dropout=args.dropout,
                    head_dropout=args.head_dropout,
                )

                # Overwrite single-checkpoint predictions with ensemble predictions
                ensemble_out_path = output_dir / "test_predictions.tsv"
                rows = []
                seen_ids: set = set()
                for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        rows.append({
                            "idx": pid,
                            "input": sym,
                            "prediction": json.dumps(ensemble_preds[i].tolist()),
                        })
                pd.DataFrame(rows).to_csv(ensemble_out_path, sep="\t", index=False)
                print(f"Saved top-{len(ckpt_files)}-checkpoint ensemble predictions → {ensemble_out_path}")
            else:
                print(f"Only {len(ckpt_files)} checkpoint(s) found; skipping ensemble (need ≥2)")

    # --- Save test score (rank 0 only) ---
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")


if __name__ == "__main__":
    main()
