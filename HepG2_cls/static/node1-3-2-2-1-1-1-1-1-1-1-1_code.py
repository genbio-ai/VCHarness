r"""Node 1-3-2-2-1-1-1-1-1-1-1-1: Restore Grandparent Config + Fixed Ensemble + Extended Training

Key changes from parent (node1-3-2-2-1-1-1-1-1-1-1, test F1=0.4826):

ROOT CAUSE ANALYSIS (parent failed due to 3 compounding issues):
  1. Premature early stopping (epoch 228) — model stopped 12 epochs before 2nd warm restart
     at epoch 240, never entering cycle 3 where grandparent found best (epoch 329).
  2. Over-modified regularization — reduced weight_decay (6e-4) and increased mixup_prob (0.65)
     caused faster but shallower convergence to a suboptimal basin (val_f1=0.4859 vs
     grandparent's 0.4988). The model found a local minimum quickly and couldn't escape.
  3. Excessive ensemble (15 checkpoints in a 0.005 F1 range) — averaging nearly-identical
     models from the same basin actively degraded performance vs single best checkpoint.
     Note: Single best checkpoint logged avg_f1=0.5442 before ensemble overwrote it!

CHANGES IN THIS NODE:

  1. Restore weight_decay: 6e-4 → 8e-4 (grandparent's proven value)
     — The parent's 8e-4 was empirically better. The 25% reduction caused premature
       convergence. The grandparent used 8e-4 and achieved val_f1=0.4988 at epoch 329.
     — 8e-4 provides sufficient regularization pressure to continue exploring beyond early basins.

  2. Restore mixup_prob: 0.65 → 0.5 (grandparent's proven value)
     — 0.65 was too aggressive and introduced excessive label noise that accelerated convergence
       to a suboptimal local minimum. Beta(0.2, 0.2) is bimodal but at 65%, too many batches
       contained heavily mixed examples disrupting gradient directions.
     — 0.5 provided the correct balance in grandparent: richer augmentation without instability.

  3. Restore head_dropout: 0.20 → 0.18 (grandparent's proven value)
     — 0.20 combined with other changes pushed the model into a different, worse basin.
     — 0.18 was confirmed effective by the grandparent's performance.

  4. Increase patience: 90 → 160 (key fix for premature termination)
     — The parent early-stopped at epoch 228, 12 epochs before the 2nd warm restart at 240.
       With T_0=80, T_mult=2: cycle 3 spans epochs 240-560. Patience=160 guarantees the model
       survives through the start of cycle 3 and allows meaningful exploration of that basin.
     — Grandparent used patience=70 but ran all 350 epochs without early stopping. This suggests
       the model needs very long training to find the best basin. Patience=160 prevents premature
       termination through at least one full post-restart convergence period.

  5. Increase max_epochs: 400 → 500 (extended cycle 3 exploration)
     — Parent's cycle 3 starts at epoch 240 but early stopping prevented any exploration.
     — max_epochs=500 covers 260 epochs of cycle 3 (vs 0 in parent, 110 in grandparent's 350),
       allowing thorough exploration of the new basin established after the second warm restart.
     — The grandparent's best was at epoch 329 (89 epochs into cycle 3). With max_epochs=500,
       we get 260 epochs of cycle 3, a much deeper exploration.

  6. Reduce ensemble_top_k: 15 → 5 (smarter ensemble strategy)
     — The 15-checkpoint ensemble (val_f1 range: 0.4809-0.4859) actively hurt performance.
       Averaging checkpoints from a 0.005 range provides no diversity and dilutes the best model.
     — Top-5 from a 500-epoch run with 2 warm restarts will span checkpoints from:
       * Cycle 2 peak (epochs ~80-240): models from early high-LR exploration
       * Cycle 3 peak (epochs ~240-500): models from post-restart exploration
     — This ensures meaningful diversity across LR phases while avoiding dilution from weak models.

  7. Keep fixed ensemble regex: r"val_f1[=_]([\d.]+)\.ckpt$" (working, keep as-is)
     — The parent successfully discovered all 15 checkpoints with this fix.
     — Ensemble will now actually benefit from checkpoint diversity across warm restart cycles.

  8. Disable SWA (removed from training)
     — SWA was never activated in the parent (early stopping at epoch 228 before SWA start at 270).
     — SWA + warm restarts create conflicting dynamics: SWA expects a stable convergence phase,
       but CosineAnnealingWarmRestarts periodically resets LR and explores new basins.
     — By removing SWA, we eliminate complexity and allow warm restarts to work as designed.
     — The grandparent proved F1=0.4968 without SWA; adding SWA only complicated things.

  9. Reduce save_top_k: 15 → 5 (aligned with smaller ensemble)
     — With top-5 ensemble, we only need top-5 checkpoints saved.
     — Saves disk space and ensures only the highest quality checkpoints are kept.

What is preserved from grandparent (the proven-effective settings):
  — STRING-only architecture (ESM2 consistently hurt performance across 10+ nodes)
  — 3 PreLN residual blocks, hidden=384 (proven best)
  — head_dropout=0.18 (restored to grandparent's proven value)
  — trunk_dropout=0.28 (balanced)
  — Muon LR=0.01 (proven stable convergence)
  — AdamW LR=3e-4 (proven optimal for non-Muon params)
  — No label smoothing (0.0 — removes training loss floor)
  — Flat output head + per-gene bias (flat head proven superior)
  — CosineAnnealingWarmRestarts (T_0=80, T_mult=2, eta_min=5e-7)
  — Manifold Mixup at STRING embedding level (alpha=0.2, prob=0.5 — restored)
  — grad_clip_norm=1.0 (proven stable)
  — val_f1 underscore logging (prevents Lightning format corruption)
  — class_weight_alpha=0.35 (more aggressive minority weighting)
  — weight_decay=8e-4 (restored to grandparent's proven value)
  — Fixed ensemble regex: r"val_f1[=_]([\d.]+)\.ckpt$"

Tree context:
  grandparent (node1-3-2-2-1-1-1-1-1-1, F1=0.4968) | WarmRestarts+Mixup, ensemble bug
  parent (node1-3-2-2-1-1-1-1-1-1-1, F1=0.4826)    | SWA+15ens, reduced wd, increased mixup_prob
  This node targets F1 > 0.4968 (grandparent) via:
    - Restore proven grandparent regularization (wd=8e-4, mixup_prob=0.5, head_drop=0.18)
    - Fixed ensemble regex (inherited, proven working)
    - Extended training (500 epochs) for thorough cycle 3 exploration
    - Large patience (160 epochs) to survive through cycle 3 without early stopping
    - Smarter ensemble (top-5 with diversity across restart cycles)
    - No SWA (too much complexity without benefit given warm restart schedule)
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
import re
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
        # Map ENSEMBL pert_id → STRING-node-index; -1 = not in STRING graph
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
class ResidualBlock(nn.Module):
    """Pre-LayerNorm residual MLP block (hidden_dim → hidden_dim*2 → hidden_dim)."""

    def __init__(self, dim: int, dropout: float = 0.28) -> None:
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


class PerturbMLP(nn.Module):
    """STRING-only MLP for gene perturbation response prediction.

    Architecture (per sample):
      ① STRING_GNN embedding lookup [256-dim, frozen buffer]
         (fallback learnable 256-dim for ~6% genes not in STRING)
      ② Input projection: Linear(256→hidden_dim) + LN + GELU
      ③ n_blocks × ResidualBlock(hidden_dim, trunk_dropout=0.28)
      ④ LN(hidden_dim) → Dropout(head_dropout=0.18) → Linear(hidden_dim → 6640*3) + per-gene-bias
      ⑤ reshape → [B, 3, 6640]

    Methods for Manifold Mixup:
      - _lookup_embedding(): separate embedding lookup for Mixup
      - forward_from_emb(): forward from pre-computed embeddings
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.28,        # trunk dropout
        head_dropout: float = 0.18,   # RESTORED: grandparent's proven 0.18
    ) -> None:
        super().__init__()
        # Learnable fallback embedding for genes not in STRING graph
        self.fallback_emb = nn.Parameter(torch.zeros(256))
        nn.init.normal_(self.fallback_emb, std=0.02)

        # Input projection: 256 → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Residual MLP blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        # Flat output head + per-gene additive bias.
        # head_dropout=0.18 — RESTORED to grandparent's proven value.
        # Flat head confirmed superior over factorized heads across 4+ nodes.
        # Per-gene bias confirmed helpful in node1-1-1.
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, N_GENES * N_CLASSES),
        )
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES * N_CLASSES))

    def _lookup_embedding(self, str_idx: torch.Tensor, string_embs: torch.Tensor) -> torch.Tensor:
        """Lookup STRING embeddings with fallback for missing genes. Returns [B, 256]."""
        valid_mask = str_idx >= 0
        safe_idx = str_idx.clamp(min=0)
        emb = string_embs[safe_idx].to(self.fallback_emb)
        if not valid_mask.all():
            fallback = self.fallback_emb.unsqueeze(0).expand(int((~valid_mask).sum()), -1)
            emb = emb.clone()
            emb[~valid_mask] = fallback
        return emb

    def forward_from_emb(self, emb: torch.Tensor) -> torch.Tensor:
        """Forward from pre-computed embeddings (for Mixup at embedding level). Returns [B, 3, 6640]."""
        x = self.input_proj(emb)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x) + self.gene_bias.to(x)
        return logits.view(-1, N_CLASSES, N_GENES)

    def forward(
        self,
        str_idx: torch.Tensor,       # [B]  STRING node indices, -1 = not in graph
        string_embs: torch.Tensor,   # [18870, 256] frozen buffer
    ) -> torch.Tensor:
        emb = self._lookup_embedding(str_idx, string_embs)
        return self.forward_from_emb(emb)


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.28,            # trunk dropout
        head_dropout: float = 0.18,       # RESTORED: grandparent's proven 0.18
        lr: float = 3e-4,
        muon_lr: float = 0.01,            # Proven stable
        weight_decay: float = 8e-4,       # RESTORED: grandparent's proven 8e-4
        label_smoothing: float = 0.0,     # No label smoothing (removes training loss floor)
        class_weight_alpha: float = 0.35, # More aggressive minority class focus
        cosine_t0: int = 80,              # T_0 for CosineAnnealingWarmRestarts
        cosine_t_mult: int = 2,           # T_mult for warm restarts
        cosine_eta_min: float = 5e-7,     # Minimum LR
        max_epochs: int = 500,            # KEY CHANGE: 400 → 500 (more cycle 3 coverage)
        grad_clip_norm: float = 1.0,
        use_muon: bool = True,
        mixup_alpha: float = 0.2,         # Manifold Mixup alpha
        mixup_prob: float = 0.5,          # RESTORED: grandparent's proven 0.5
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.lr = lr
        self.muon_lr = muon_lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.class_weight_alpha = class_weight_alpha
        self.cosine_t0 = cosine_t0
        self.cosine_t_mult = cosine_t_mult
        self.cosine_eta_min = cosine_eta_min
        self.max_epochs = max_epochs
        self.grad_clip_norm = grad_clip_norm
        self.use_muon = use_muon
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob

        # Build model in __init__ so configure_optimizers sees all parameters
        # (required for DDP to properly detect all parameters before wrapping)
        self.model = PerturbMLP(
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout,
            head_dropout=head_dropout,
        )

        # Accumulation buffers for epoch-level metrics
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def _load_string_embeddings(self, skip_barrier: bool = False) -> None:
        """Load STRING_GNN embeddings and register as frozen buffer.

        This is a standalone helper that does NOT require a Trainer attachment.
        Use this in ensemble-test contexts where the module is not part of a Trainer.

        Args:
            skip_barrier: If True, skips the distributed barrier call. Required
                when called outside the DDP collective context (e.g., ensemble test
                where only rank 0 runs the code).
        """
        if hasattr(self, "string_embs") and self.string_embs is not None:
            return  # already loaded

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            from transformers import AutoModel as _AM
            _AM.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        # Only barrier in DDP training context; skip when outside the collective
        if not skip_barrier and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

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
        self.register_buffer("string_embs", string_embs)

    def setup(self, stage: str = "fit") -> None:
        # Softened class weights: w_i = (1/freq_i)^alpha
        # alpha=0.35: weights [1.02, 5.64, 9.67] → ratio class0:class2 = 9.5×
        alpha = self.class_weight_alpha
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq).pow(alpha)   # temperature-scaled
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        self.register_buffer("class_weights", class_weights)

        if hasattr(self, "string_embs") and self.string_embs is not None:
            return  # already loaded (guard for re-entrant setup calls)

        # ---- Load STRING_GNN node embeddings (rank-0 downloads first) ----
        self._load_string_embeddings()

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        if hasattr(self, "trainer") and self.trainer is not None:
            self.print(
                f"Node1-3-2-2-1-1-1-1-1-1-1-1 PerturbMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
                f"trunk_dropout={self.dropout} | head_dropout={self.head_dropout} | "
                f"class_weight_alpha={self.class_weight_alpha} | label_smoothing={self.label_smoothing} | "
                f"muon_lr={self.muon_lr} | cosine_t0={self.cosine_t0} | cosine_t_mult={self.cosine_t_mult} | "
                f"cosine_eta_min={self.cosine_eta_min} | use_muon={self.use_muon} | "
                f"max_epochs={self.max_epochs} | mixup_alpha={self.mixup_alpha} | "
                f"mixup_prob={self.mixup_prob} | weight_decay={self.weight_decay} | "
                f"trainable={n_trainable:,}/{n_total:,}"
            )

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted cross-entropy without label smoothing.

        logits: [B, 3, 6640]
        labels: [B, 6640]  — values in {0, 1, 2}
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                                # [B*6640]
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,  # 0.0: no artificial loss floor
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        str_idx = batch["str_idx"]
        labels = batch["label"]

        if (self.mixup_alpha > 0
                and self.training
                and np.random.random() < self.mixup_prob):
            B = str_idx.size(0)
            # Sample mixing coefficient, enforce λ ≥ 0.5 for stability
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            lam = max(lam, 1.0 - lam)

            # Random permutation for Mixup pair selection
            perm = torch.randperm(B, device=str_idx.device)

            # Get STRING embeddings
            emb = self.model._lookup_embedding(str_idx, self.string_embs)  # [B, 256]

            # Mixed embedding
            mixed_emb = lam * emb + (1.0 - lam) * emb[perm]

            # Forward with mixed embedding
            logits = self.model.forward_from_emb(mixed_emb)  # [B, 3, 6640]

            # Mixed CE loss
            labels_b = labels[perm]
            loss = lam * self._compute_loss(logits, labels) + (1.0 - lam) * self._compute_loss(logits, labels_b)
        else:
            logits = self.model(str_idx, self.string_embs)
            loss = self._compute_loss(logits, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["str_idx"], self.string_embs)
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
            if all_labels.dim() == 3:
                all_labels = all_labels.squeeze(0)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        f1 = _compute_per_gene_f1(preds_np, labels_np)

        # Log as "val_f1" (underscore) to fix checkpoint filename substitution.
        # Slash notation "val/f1" causes Lightning to substitute 0.0000 in filenames.
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["str_idx"], self.string_embs)
        self._test_preds.append(logits.detach().cpu())
        self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> Dict[str, float]:
        """Save test predictions and compute test F1.

        Each rank saves its own samples locally as NPZ (no all_gather needed in DDP test).
        Test F1 is averaged across ranks via all_reduce (scalar only).
        Writes test_score.txt directly on rank 0.
        """
        if not self._test_preds:
            return {}

        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        labels_local = torch.cat(self._test_labels, dim=0)  # [N_local, 6640]
        self._test_preds.clear()
        self._test_labels.clear()

        # Compute local per-gene F1
        preds_np = preds_local.float().cpu().numpy()
        labels_np = labels_local.cpu().numpy()
        local_f1 = _compute_per_gene_f1(preds_np, labels_np)

        # Average F1 across ranks via all_reduce (scalar op, no tensor gathering)
        ws = self.trainer.world_size
        if ws > 1:
            f1_tensor = torch.tensor(local_f1, dtype=torch.float32, device=self.device)
            torch.distributed.all_reduce(f1_tensor, op=torch.distributed.ReduceOp.SUM)
            avg_f1 = f1_tensor.item() / ws
        else:
            avg_f1 = local_f1

        # Capture metadata before clearing
        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        n_local = len(local_pert_ids)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        # Save per-rank NPZ (no all_gather needed)
        preds_np_out = preds_local.float().cpu().numpy()
        out_path = Path(__file__).parent / "run" / f"test_predictions_rank{self.global_rank}.npz"
        np.savez_compressed(out_path, preds=preds_np_out, pert_ids=local_pert_ids, symbols=local_symbols)
        self.print(f"[Rank {self.global_rank}] Saved {n_local} local predictions → {out_path}, local_f1={local_f1:.4f}, avg_f1={avg_f1:.4f}")

        # Write test_score.txt directly on rank 0
        if self.trainer.is_global_zero:
            score_path = Path(__file__).parent / "test_score.txt"
            score_path.write_text(json.dumps({"test_f1": avg_f1}, indent=2))
            self.print(f"Test results → {score_path}")

        self.log("test_f1", avg_f1, prog_bar=False, logger=True, sync_dist=True)
        return {"test_f1": avg_f1}

    def configure_optimizers(self):
        """Configure optimizer: Muon for hidden MLP weight matrices, AdamW for everything else.

        LR schedule: CosineAnnealingWarmRestarts (T_0=80, T_mult=2, eta_min=5e-7).
        No SWA — removed because it never activated in parent (early stopping at epoch 228)
        and conflicts with warm restarts' basin-exploration dynamics.

        Muon is applied only to 2D weight matrices in hidden layers (not input/output layers,
        not biases/norms/embeddings). Per Muon skill documentation.
        """
        if self.use_muon:
            try:
                from muon import MuonWithAuxAdam
                muon_available = True
            except ImportError:
                self.print("Warning: muon not installed, falling back to AdamW")
                muon_available = False
        else:
            muon_available = False

        if muon_available and self.use_muon:
            # Identify hidden MLP weight matrices (Linear weights in residual blocks only)
            hidden_weight_names = set()
            for name, param in self.model.named_parameters():
                if (param.ndim >= 2
                        and "blocks." in name
                        and ".weight" in name
                        and "norm" not in name):
                    hidden_weight_names.add(name)

            hidden_weights = []
            other_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                # Strip "model." prefix for model params
                model_name = name[len("model."):] if name.startswith("model.") else name
                if model_name in hidden_weight_names:
                    hidden_weights.append(param)
                else:
                    other_params.append(param)

            self.print(
                f"Muon params: {sum(p.numel() for p in hidden_weights):,} | "
                f"AdamW params: {sum(p.numel() for p in other_params):,}"
            )

            param_groups = [
                # Muon for hidden weight matrices (residual block Linear weights)
                dict(
                    params=hidden_weights,
                    use_muon=True,
                    lr=self.muon_lr,       # 0.01 (proven stable)
                    weight_decay=self.weight_decay,
                    momentum=0.95,
                ),
                # AdamW for all other params (input_proj, head, gene_bias, norms, etc.)
                dict(
                    params=other_params,
                    use_muon=False,
                    lr=self.lr,
                    betas=(0.9, 0.95),
                    weight_decay=self.weight_decay,
                ),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            # Fallback: pure AdamW for all parameters
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        # CosineAnnealingWarmRestarts: periodic restarts to escape local optima
        # T_0=80: first cycle length = 80 epochs (restart at epoch 80)
        # T_mult=2: each subsequent cycle doubles in length
        #   → cycle 2 = 160 epochs (80-240), cycle 3 = 320 epochs (240-560)
        # eta_min=5e-7: both Muon and AdamW decay to ~5e-7 minimum LR per cycle
        # Within max_epochs=500: covers all of cycle 1+2 and 260 epochs of cycle 3
        # (grandparent's best was at epoch 329, 89 epochs into cycle 3 — we get 260 epochs!)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cosine_t0,         # 80
            T_mult=self.cosine_t_mult,  # 2 (integer)
            eta_min=self.cosine_eta_min,  # 5e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                # No "monitor" key — CosineAnnealingWarmRestarts is epoch-driven
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint: save only trainable params + small essential buffers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save trainable parameters and essential buffers, excluding large frozen buffers."""
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        saved: Dict[str, Any] = {}

        # Keys to always exclude (large frozen buffers)
        exclude_keys = {"string_embs"}
        # Keys that are trainable params or essential small buffers
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        essential_buffers = {
            name for name, buf in self.named_buffers()
            if name not in exclude_keys
        }
        keep_keys = trainable_keys | essential_buffers

        for key, val in full_sd.items():
            # Strip top-level prefix for matching
            rel_key = key[len(prefix):] if key.startswith(prefix) else key
            if rel_key in keep_keys or any(k in rel_key for k in keep_keys):
                saved[key] = val

        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_saved = sum(v.numel() for v in saved.values())
        pct = f"{100*n_trainable/n_total:.1f}%" if n_total > 0 else "N/A"
        # Use plain print instead of self.print() because SWA's average_model is not
        # attached to a Trainer and self.print() checks self.trainer.is_global_zero.
        print(
            f"Saving checkpoint: {n_trainable:,}/{n_total:,} trainable params ({pct}), "
            f"{n_saved:,} total saved"
        )
        return saved

    def load_state_dict(self, state_dict, strict=True):
        # strict=False: string_embs is not in checkpoint but was populated by setup()
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
    for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
        rows.append({
            "idx": pid,
            "input": sym,
            "prediction": json.dumps(preds[i].tolist()),  # [3][6640] list
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


def _discover_local_checkpoints(
    ckpt_dir: Path, top_k: int
) -> List[Path]:
    r"""Discover the top-k best checkpoints from the local checkpoint directory.

    KEY FIX (inherited from parent): Updated regex to match Lightning's actual checkpoint format.

    Lightning's ModelCheckpoint with monitor="val_f1" always produces filenames
    like "epoch=329-val_f1=0.4988.ckpt" regardless of the filename format string
    provided (Lightning substitutes metric values using param=value style).

    Old (broken): r"-([\d.]+)\.ckpt$"
      — Did NOT match "epoch=329-val_f1=0.4988.ckpt" because this expects
        a dash immediately before the decimal number, but "val_f1=" comes first.

    New (fixed): r"val_f1[=_]([\d.]+)\.ckpt$"
      — Matches "val_f1=0.4988.ckpt" (Lightning format with equals sign)
      — Also matches "val_f1_0.4988.ckpt" (alternative format with underscore)
      — Correctly extracts 0.4988 as the F1 score for ranking

    With top_k=5 (reduced from 15), the ensemble only uses the top-5 best checkpoints.
    This ensures high-quality predictions from diverse LR phases (cycle 2 and 3 peaks)
    without diluting with weaker checkpoints from the same narrow convergence basin.
    """
    all_ckpts: List[tuple] = []  # (f1_score, path)

    if not ckpt_dir.exists():
        return []

    for entry in ckpt_dir.iterdir():
        if entry.is_file() and entry.suffix == ".ckpt":
            # FIXED regex: matches "val_f1=0.4988.ckpt" (Lightning's actual format)
            # Also handles "val_f1_0.4988.ckpt" as a fallback
            f1_match = re.search(r"val_f1[=_]([\d.]+)\.ckpt$", entry.name)
            if f1_match:
                f1 = float(f1_match.group(1))
                all_ckpts.append((f1, entry))
            else:
                # Fallback: try to match any trailing decimal before .ckpt
                # e.g., "190-0.4967.ckpt" (custom format)
                f1_match_fallback = re.search(r"-([\d.]+)\.ckpt$", entry.name)
                if f1_match_fallback:
                    f1 = float(f1_match_fallback.group(1))
                    # Only use as fallback if f1 is in plausible range
                    if 0.1 <= f1 <= 1.0:
                        all_ckpts.append((f1, entry))
        elif entry.is_dir():
            # Try subdirectory: "epoch=XXX-val_f1=0.XXXX" containing "val_f1=0.XXXX.ckpt"
            ckpt_files = list(entry.glob("val_f1=*.ckpt"))
            if not ckpt_files:
                ckpt_files = list(entry.glob("*.ckpt"))
            if not ckpt_files:
                continue
            ckpt_file = ckpt_files[0]
            # Try primary pattern
            f1_match = re.search(r"val_f1[=_]([\d.]+)\.ckpt$", ckpt_file.name)
            if f1_match:
                f1 = float(f1_match.group(1))
                all_ckpts.append((f1, ckpt_file))
            else:
                # Try parent dir name
                dir_f1_match = re.search(r"val_f1[=_]([\d.]+)$", entry.name)
                if dir_f1_match:
                    f1 = float(dir_f1_match.group(1))
                    all_ckpts.append((f1, ckpt_file))

    if not all_ckpts:
        return []

    # Sort by f1 descending (best first), take top_k
    all_ckpts.sort(key=lambda x: x[0], reverse=True)
    return [ckpt for _, ckpt in all_ckpts[:top_k]]


def _assemble_test_predictions_from_npz(out_path: Path) -> None:
    """Assemble per-rank NPZ prediction files into a single TSV.

    Each DDP rank saves its local predictions as NPZ during on_test_epoch_end.
    This function reads all rank NPZ files, deduplicates by pert_id, and
    writes the final test_predictions.tsv.
    """
    run_dir = out_path.parent

    # Collect all NPZ files
    npz_files = sorted(run_dir.glob("test_predictions_rank*.npz"))
    if not npz_files:
        print(f"[Assembly] No per-rank NPZ files found in {run_dir}")
        return

    all_pert_ids: List[str] = []
    all_symbols: List[str] = []
    all_preds: List[np.ndarray] = []

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        preds = data["preds"]  # [N_local, 3, 6640]
        pert_ids = data["pert_ids"].tolist()
        symbols = data["symbols"].tolist()
        all_preds.append(preds)
        all_pert_ids.extend(pert_ids)
        all_symbols.extend(symbols)

    # Concatenate predictions from all ranks
    combined_preds = np.concatenate(all_preds, axis=0)  # [N_total, 3, 6640]

    # De-duplicate by pert_id (keep first occurrence)
    seen: set = set()
    dedup_pert = []
    dedup_sym = []
    dedup_idx = []
    for i, pid in enumerate(all_pert_ids):
        if pid and pid not in seen:
            seen.add(pid)
            dedup_pert.append(pid)
            dedup_sym.append(all_symbols[i])
            dedup_idx.append(i)

    dedup_preds = combined_preds[dedup_idx]

    _save_test_predictions(
        pert_ids=dedup_pert,
        symbols=dedup_sym,
        preds=dedup_preds,
        out_path=out_path,
    )

    # Clean up per-rank NPZ files
    for npz_path in npz_files:
        npz_path.unlink()
    print(f"[Assembly] Cleaned up {len(npz_files)} per-rank NPZ files")


def _run_ensemble_test(
    model_kwargs: Dict[str, Any],
    ckpt_paths: List[Path],
    test_dataloader: DataLoader,
    out_path: Path,
    device: torch.device,
) -> None:
    r"""Ensemble test by averaging logits from multiple checkpoints.

    This is a within-node ensemble: all checkpoints are from the same training run.
    No cross-node artifacts are used. All predictions are self-contained.

    With warm restarts, checkpoints span multiple LR phases (cycle peaks and valleys),
    providing temporal diversity. With save_top_k=5 and max_epochs=500, the top-5
    checkpoints should span both cycle 2 (epochs 80-240) and cycle 3 (epochs 240-500),
    providing meaningful LR-phase diversity (unlike parent's 15-checkpoint ensemble
    which was confined to a single cycle 2 convergence basin).

    The fixed regex r"val_f1[=_]([\d.]+)\.ckpt$" correctly discovers these checkpoints.
    """
    all_preds_list: List[np.ndarray] = []
    pert_ids_final: Optional[List[str]] = None
    symbols_final: Optional[List[str]] = None

    print(f"\n[Ensemble] Averaging logits from {len(ckpt_paths)} checkpoints:")
    for ckpt_path in ckpt_paths:
        print(f"  Loading: {ckpt_path.name if ckpt_path.is_file() else ckpt_path}")

        # Create a fresh model instance (not wrapped in DDP)
        m = PerturbModule(**model_kwargs)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        # The checkpoint may have 'state_dict' key (from Lightning) or be raw
        sd = state.get("state_dict", state)
        m.load_state_dict(sd, strict=False)

        # Set up string embeddings (without DDP)
        # skip_barrier=True because ensemble test runs only on rank 0 (no DDP collective).
        m._load_string_embeddings(skip_barrier=True)
        m.eval()
        m = m.to(device)
        if hasattr(m, "string_embs"):
            m.string_embs = m.string_embs.to(device)

        # Run inference on the test set
        preds_list: List[torch.Tensor] = []
        pert_ids_batch: List[str] = []
        symbols_batch: List[str] = []

        with torch.no_grad():
            for batch in test_dataloader:
                str_idx = batch["str_idx"].to(device)
                logits = m.model(str_idx, m.string_embs)  # [B, 3, 6640]
                preds_list.append(logits.float().cpu())
                pert_ids_batch.extend(batch["pert_id"])
                symbols_batch.extend(batch["symbol"])

        batch_preds = torch.cat(preds_list, dim=0).numpy()  # [N, 3, 6640]
        all_preds_list.append(batch_preds)

        if pert_ids_final is None:
            pert_ids_final = pert_ids_batch
            symbols_final = symbols_batch

        # Free memory before loading next checkpoint
        del m
        torch.cuda.empty_cache()

    # Average logits across all checkpoints
    avg_preds = np.mean(all_preds_list, axis=0)  # [N, 3, 6640]

    # Deduplicate (consistent with the Lightning test pipeline)
    seen: set = set()
    dedup_pert: List[str] = []
    dedup_sym: List[str] = []
    dedup_idx: List[int] = []
    for i, pid in enumerate(pert_ids_final or []):
        if pid and pid not in seen:
            seen.add(pid)
            dedup_pert.append(pid)
            dedup_sym.append((symbols_final or [])[i])
            dedup_idx.append(i)
    avg_preds = avg_preds[dedup_idx]

    _save_test_predictions(
        pert_ids=dedup_pert,
        symbols=dedup_sym,
        preds=avg_preds,
        out_path=out_path,
    )
    print(f"[Ensemble] Saved {len(dedup_pert)} predictions using {len(ckpt_paths)}-checkpoint ensemble → {out_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-3-2-2-1-1-1-1-1-1-1-1: STRING-only + Flat Head + hidden=384 + "
                    "Muon LR=0.01 + CosineWarmRestarts + Manifold Mixup + top-5 ensemble"
    )
    p.add_argument("--micro-batch-size",      type=int,   default=32)
    p.add_argument("--global-batch-size",     type=int,   default=256)
    p.add_argument("--max-epochs",            type=int,   default=500,
                   help="KEY CHANGE: 400 → 500 (extended cycle 3 exploration)")
    p.add_argument("--lr",                    type=float, default=3e-4)
    p.add_argument("--muon-lr",               type=float, default=0.01,
                   help="Muon optimizer LR (proven stable)")
    p.add_argument("--weight-decay",          type=float, default=8e-4,
                   help="RESTORED: 6e-4 → 8e-4 (grandparent's proven value, prevents premature convergence)")
    p.add_argument("--hidden-dim",            type=int,   default=384)
    p.add_argument("--n-blocks",              type=int,   default=3)
    p.add_argument("--dropout",               type=float, default=0.28,
                   help="Trunk dropout")
    p.add_argument("--head-dropout",          type=float, default=0.18,
                   help="RESTORED: 0.20 → 0.18 (grandparent's proven value)")
    p.add_argument("--label-smoothing",       type=float, default=0.0,
                   help="No label smoothing (removes training loss floor)")
    p.add_argument("--class-weight-alpha",    type=float, default=0.35,
                   help="More aggressive minority class focus (alpha=0.35)")
    p.add_argument("--cosine-t0",             type=int,   default=80,
                   help="T_0 for CosineAnnealingWarmRestarts (first cycle length)")
    p.add_argument("--cosine-t-mult",         type=int,   default=2,
                   help="T_mult for CosineAnnealingWarmRestarts (cycle length multiplier)")
    p.add_argument("--cosine-eta-min",        type=float, default=5e-7,
                   help="CosineAnnealingWarmRestarts minimum LR per cycle")
    p.add_argument("--early-stop-patience",   type=int,   default=160,
                   help="KEY CHANGE: 90 → 160 (must survive through cycle 3 start at epoch 240)")
    p.add_argument("--grad-clip-norm",        type=float, default=1.0)
    p.add_argument("--no-muon",               action="store_true",
                   help="Disable Muon optimizer, fall back to AdamW for all params")
    p.add_argument("--ensemble-top-k",        type=int,   default=5,
                   help="KEY CHANGE: 15 → 5 checkpoints (avoids dilution from similar models)")
    p.add_argument("--mixup-alpha",           type=float, default=0.2,
                   help="Manifold Mixup Beta distribution parameter (0.0 = disabled)")
    p.add_argument("--mixup-prob",            type=float, default=0.5,
                   help="RESTORED: 0.65 → 0.5 (grandparent's proven value)")
    p.add_argument("--num-workers",           type=int,   default=4)
    p.add_argument("--val-check-interval",    type=float, default=1.0)
    p.add_argument("--debug_max_step",        type=int,   default=None)
    p.add_argument("--fast_dev_run",          action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

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
        lr=args.lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        class_weight_alpha=args.class_weight_alpha,
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_t_mult,
        cosine_eta_min=args.cosine_eta_min,
        max_epochs=args.max_epochs,
        grad_clip_norm=args.grad_clip_norm,
        use_muon=not args.no_muon,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
    )

    # --- Trainer configuration ---
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = 1.0
        max_steps = args.debug_max_step
    else:
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        max_steps = -1

    # KEY CHANGE: save_top_k=5 (from 15).
    # With warm restarts across 500 epochs, top-5 checkpoints should span:
    # - Cycle 2 peak (epochs ~80-240): high-val_f1 from first stable convergence
    # - Cycle 3 peak (epochs ~240-500): high-val_f1 from post-restart exploration
    # Top-5 ensures only truly high-quality checkpoints are averaged, avoiding dilution
    # from weaker models that hurt the parent's 15-checkpoint ensemble.
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=5,         # KEY CHANGE: 15 → 5 for quality over quantity
        save_last=True,
    )
    # KEY CHANGE: patience=160 (from 90).
    # The parent's critical failure was early stopping at epoch 228, 12 epochs before
    # the second warm restart at epoch 240. With patience=160, the model is guaranteed
    # to survive the restart at epoch 240 and explore at least some of cycle 3.
    # The grandparent used patience=70 but ran all 350 epochs without triggering —
    # this node needs patience > 160 to avoid the parent's premature termination.
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stop_patience,  # 160 (extended for warm restart cycle 3)
        min_delta=1e-5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # NOTE: SWA (StochasticWeightAveraging) is intentionally NOT included.
    # SWA was never activated in parent (stopped at epoch 228 < SWA start epoch 270).
    # Moreover, SWA + CosineAnnealingWarmRestarts creates conflicting dynamics:
    # SWA needs stable convergence, but WarmRestarts periodically resets to explore new basins.
    # The grandparent achieved F1=0.4968 without SWA — warm restarts alone are sufficient.

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(
            find_unused_parameters=True,  # fallback_emb may be unused when all str_idx >= 0
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
        gradient_clip_val=args.grad_clip_norm,
    )

    # --- Fit ---
    trainer.fit(model, datamodule=datamodule)

    # --- Test ---
    test_results = None
    if args.fast_dev_run or args.debug_max_step is not None:
        # Debug mode: standard Lightning test (all ranks, no ensemble)
        test_results = trainer.test(model, datamodule=datamodule)

        # Assemble per-rank NPZ files into final TSV on rank 0
        if trainer.is_global_zero:
            _assemble_test_predictions_from_npz(output_dir / "test_predictions.tsv")
    else:
        # Production mode:
        # Step 1: Run standard test with single best checkpoint (all ranks participate)
        # This writes the initial test_predictions.tsv via on_test_epoch_end
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

        # Assemble per-rank NPZ files into initial TSV
        if trainer.is_global_zero:
            _assemble_test_predictions_from_npz(output_dir / "test_predictions.tsv")

        # Step 2: Ensemble test on rank 0 (overwrite test_predictions.tsv with top-5 ensemble)
        # KEY FIX (inherited): regex r"val_f1[=_]([\d.]+)\.ckpt$" matches Lightning format.
        # KEY CHANGE: top_k reduced to 5 (from 15) for better ensemble quality.
        # With 500 epochs and two warm restarts, top-5 spans diverse LR phases.
        if trainer.is_global_zero and args.ensemble_top_k > 1:
            ckpt_dir = output_dir / "checkpoints"
            top_ckpts = _discover_local_checkpoints(ckpt_dir, top_k=args.ensemble_top_k)
            if len(top_ckpts) >= 2:
                print(f"\n[Ensemble] Running {len(top_ckpts)}-checkpoint ensemble test...")
                _run_ensemble_test(
                    model_kwargs=dict(
                        hidden_dim=args.hidden_dim,
                        n_blocks=args.n_blocks,
                        dropout=args.dropout,
                        head_dropout=args.head_dropout,
                        lr=args.lr,
                        muon_lr=args.muon_lr,
                        weight_decay=args.weight_decay,
                        label_smoothing=args.label_smoothing,
                        class_weight_alpha=args.class_weight_alpha,
                        cosine_t0=args.cosine_t0,
                        cosine_t_mult=args.cosine_t_mult,
                        cosine_eta_min=args.cosine_eta_min,
                        max_epochs=args.max_epochs,
                        grad_clip_norm=args.grad_clip_norm,
                        use_muon=not args.no_muon,
                        mixup_alpha=args.mixup_alpha,
                        mixup_prob=args.mixup_prob,
                    ),
                    ckpt_paths=top_ckpts,
                    test_dataloader=datamodule.test_dataloader(),
                    out_path=output_dir / "test_predictions.tsv",
                    device=torch.device("cuda", 0),
                )
            else:
                print(f"\n[Ensemble] Only {len(top_ckpts)} checkpoint(s) available; "
                      "skipping ensemble (need >= 2)")

    # --- Save test score (rank 0 only) ---
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")


if __name__ == "__main__":
    main()
