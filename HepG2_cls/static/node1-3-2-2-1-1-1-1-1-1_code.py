"""Node 1-3-2-2-1-1-1-1-1-1: CosineAnnealingWarmRestarts + Manifold Mixup + stronger regularization

Key changes from parent (node1-3-2-2-1-1-1-1-1, test F1=0.4968 — TREE BEST):

  1. Replace CosineAnnealingLR with CosineAnnealingWarmRestarts (T_0=80, T_mult=2, eta_min=5e-7)
     — Parent used a single cosine cycle (T_max=300), which can get stuck in local optima
       as the LR monotonically approaches eta_min with no restart mechanism.
     — CosineAnnealingWarmRestarts provides periodic LR boosts (restarts) that help escape
       local optima. Schedule: cycle 1 = epochs 0-80, cycle 2 = epochs 80-240, cycle 3 = 240-560
       (within max_epochs=350 budget). Each restart gives the optimizer a fresh shot at
       exploring the loss landscape from a higher LR.
     — Parent got stuck in a plateau at epochs 183-240. Restarts at epoch 80 and 240 may
       break this plateau and discover better optima.

  2. Add Manifold Mixup at STRING embedding level (mixup_alpha=0.2, mixup_prob=0.5)
     — Mixup regularization at the embedding level creates a smoother loss landscape and
       reduces overfitting by training on convex combinations of training examples.
     — Parent had a persistent overfitting gap: train/loss=0.053 vs val/loss=0.225 (gap=0.172).
     — λ ~ Beta(0.2, 0.2) with λ = max(λ, 1-λ) enforced to keep dominant sample ≥50%.
     — Mixed embedding: emb_mixed = λ*emb_a + (1-λ)*emb_b
     — Mixed loss: λ*CE(logits, labels_a) + (1-λ)*CE(logits, labels_b)
     — Applied with probability mixup_prob=0.5 to avoid over-regularizing small batches.

  3. Increase weight_decay: 5e-4 → 8e-4
     — Addresses persistent overfitting. Stronger L2 regularization to complement Mixup.

  4. Reduce class_weight_alpha: 0.4 → 0.35
     — More aggressive inverse-frequency weighting for minority classes.
     — α=0.35: weights [1.02, 5.64, 9.67] → ratio class0:class2 = 9.5× (more focus on rare classes)
     — α=0.40: weights [1.03, 5.06, 8.03] → ratio class0:class2 = 7.8×

  5. Extend patience: 50 → 70
     — Warm restarts produce temporary LR spikes followed by re-convergence.
       Patience=70 ensures we don't stop during a temporary post-restart regression.

  6. Extend max_epochs: 300 → 350
     — Covers the second full restart cycle (T_0=80, T_mult=2: cycle 2 ends at epoch 240,
       cycle 3 starts at 240). Epoch 350 gives 110 epochs of cycle 3 exploration.

  7. Increase save_top_k: 7 → 10
     — More checkpoints across diverse LR phases (post-restart peaks and valleys).
       With warm restarts, models at different cycle phases may have complementary strengths.

  8. Fix checkpoint filename: "best-{epoch:03d}-val_f1={val_f1:.4f}" → "{epoch:03d}-{val_f1:.4f}"
     — Avoids Lightning's double-prefix issue where the monitor key "val_f1" is prepended
       again in subdir mode, creating paths like "val_f1=val_f1=0.4967.ckpt".

What is preserved from parent:
  — STRING-only architecture (ESM2 consistently hurt performance)
  — 3 PreLN residual blocks, hidden=384 (proven best)
  — head_dropout=0.18 (inherited as proven overfitting reduction)
  — trunk_dropout=0.28 (slightly reduced from 0.30, balanced)
  — Muon LR=0.01 (proven stable convergence)
  — AdamW LR=3e-4 (proven optimal for non-Muon params)
  — No label smoothing (0.0 — removes training loss floor)
  — Flat output head + per-gene bias (flat head proven superior)
  — grad_clip_norm=1.0 (proven stable)
  — val_f1 underscore logging (bug fix from parent)

Tree context:
  node1-3-2-2-1-1-1-1 (grandparent)   | F1=0.4914 | RLROP + no label smooth + top-5 ensemble
  node1-3-2-2-1-1-1-1-1 (parent)      | F1=0.4968 | CosineAnnealingLR + head_drop=0.18 + α=0.4 + top-7 [TREE BEST]
  This node targets F1 > 0.4968 via warm restarts + manifold mixup + stronger regularization
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

    New methods for Manifold Mixup:
      - _lookup_embedding(): separate embedding lookup for Mixup
      - forward_from_emb(): forward from pre-computed embeddings
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.28,        # trunk dropout
        head_dropout: float = 0.18,   # head dropout
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
        # head_dropout=0.18 — mild regularization to reduce overfitting.
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
        head_dropout: float = 0.18,       # head dropout: reduces overfitting
        lr: float = 3e-4,
        muon_lr: float = 0.01,            # Proven stable (enables clean convergence)
        weight_decay: float = 8e-4,       # KEY CHANGE: 5e-4 → 8e-4 (stronger regularization)
        label_smoothing: float = 0.0,     # No label smoothing (removes training loss floor)
        class_weight_alpha: float = 0.35, # KEY CHANGE: 0.4 → 0.35 (more aggressive minority focus)
        cosine_t0: int = 80,              # KEY CHANGE: T_0 for CosineAnnealingWarmRestarts
        cosine_t_mult: int = 2,           # KEY CHANGE: T_mult for warm restarts
        cosine_eta_min: float = 5e-7,     # Minimum LR
        max_epochs: int = 350,            # KEY CHANGE: 300 → 350 (covers warm restart cycle 3)
        grad_clip_norm: float = 1.0,
        use_muon: bool = True,
        mixup_alpha: float = 0.2,         # KEY CHANGE: Manifold Mixup alpha
        mixup_prob: float = 0.5,          # KEY CHANGE: probability to apply Mixup
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
        self._test_labels: List[torch.Tensor] = []  # For computing test F1
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
        # alpha=0.35: weights [1.02, 5.64, 9.67] → ratio class0:class2 = 9.5× (more aggressive)
        # alpha=0.40: weights [1.03, 5.06, 8.03] → ratio class0:class2 = 7.8×
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
                f"Node1-3-2-2-1-1-1-1-1-1 PerturbMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
                f"trunk_dropout={self.dropout} | head_dropout={self.head_dropout} | "
                f"class_weight_alpha={self.class_weight_alpha} | label_smoothing={self.label_smoothing} | "
                f"muon_lr={self.muon_lr} | cosine_t0={self.cosine_t0} | cosine_t_mult={self.cosine_t_mult} | "
                f"cosine_eta_min={self.cosine_eta_min} | use_muon={self.use_muon} | "
                f"max_epochs={self.max_epochs} | mixup_alpha={self.mixup_alpha} | "
                f"mixup_prob={self.mixup_prob} | trainable={n_trainable:,}/{n_total:,}"
            )

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted cross-entropy without label smoothing.

        logits: [B, 3, 6640]
        labels: [B, 6640]  — values in {0, 1, 2}

        Uses temperature-scaled class weights (alpha=0.35, more aggressive minority focus)
        to balance gradient signal for rare up/down-regulated genes.
        label_smoothing=0.0 (no floor) — removes artificial training loss ceiling.
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
        Writes test_score.txt directly on rank 0 (more reliable than trainer.test() return).
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

        # Write test_score.txt directly on rank 0 (reliable, avoids trainer.test() return capture issues)
        if self.trainer.is_global_zero:
            score_path = Path(__file__).parent / "test_score.txt"
            score_path.write_text(json.dumps({"test_f1": avg_f1}, indent=2))
            self.print(f"Test results → {score_path}")

        self.log("test_f1", avg_f1, prog_bar=False, logger=True)
        return {"test_f1": avg_f1}

    def configure_optimizers(self):
        """Configure optimizer: Muon for hidden MLP weight matrices, AdamW for everything else.

        LR schedule: CosineAnnealingWarmRestarts (T_0=80, T_mult=2, eta_min=5e-7).

        KEY CHANGE: Replaces CosineAnnealingLR (single cycle) with CosineAnnealingWarmRestarts.
        Rationale:
          - Single cosine cycle from parent got stuck in plateau at epochs 183-240.
          - Warm restarts provide periodic LR boosts that help escape local optima.
          - Schedule: cycle 1=epochs 0-80, cycle 2=epochs 80-240, cycle 3=epochs 240-560.
          - Within max_epochs=350 budget, cycle 3 runs from epoch 240 to epoch 350.
          - Each restart triggers a fresh exploration from higher LR, potentially escaping
            the plateau behavior seen in the parent's training run.

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
                    lr=self.muon_lr,       # 0.01 (proven stable, unchanged from parent)
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
        # KEY CHANGE: Replaces single CosineAnnealingLR cycle with warm restarts.
        # - T_0=80: first cycle length = 80 epochs (restart at epoch 80)
        # - T_mult=2: each subsequent cycle doubles in length
        #   → cycle 2 = 160 epochs (80-240), cycle 3 = 320 epochs (240-560)
        # - eta_min=5e-7: both Muon and AdamW decay to ~5e-7 minimum LR per cycle
        # - Within max_epochs=350: covers all of cycle 1+2 and 110 epochs of cycle 3
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
        """Save trainable parameters and essential buffers, excluding large frozen buffers.

        Uses Lightning's base state_dict() which correctly handles DDP wrapping
        (adds/strips 'module.' prefix as needed). We then filter to keep only:
        - Trainable parameters (all model weights that require gradients)
        - Small persistent buffers (e.g., class_weights)
        - Excludes large frozen buffers (string_embs: [18870, 256])
        """
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
        self.print(
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
    """Discover the top-k best checkpoints from the local checkpoint directory.

    Handles multiple checkpoint filename formats:
      New format: "{epoch:03d}-{val_f1:.4f}.ckpt"  e.g. "190-0.4967.ckpt"
      Old format: "best-epoch=XXX-val_f1=0.XXXX.ckpt"
      Subdir format: "best-epoch=XXX-val_f1=" containing "val_f1=0.XXXX.ckpt"

    The combined regex for matching F1 scores at end of filename
    for both old and new formats (last numeric part before .ckpt extension).

    NOTE: With the val_f1 metric logging fix (underscore instead of slash),
    the checkpoint filenames correctly contain the actual val_f1 value,
    making this function's top-k selection genuinely effective.
    """
    all_ckpts: List[tuple] = []  # (f1_score, path)

    if not ckpt_dir.exists():
        return []

    for entry in ckpt_dir.iterdir():
        if entry.is_file() and entry.suffix == ".ckpt":
            # Combined pattern: matches both "190-0.4967.ckpt" and "best-epoch=XXX-val_f1=0.4967.ckpt"
            # Uses the last number before .ckpt as the F1 score
            f1_match = re.search(r"-([\d.]+)\.ckpt$", entry.name)
            if f1_match:
                f1 = float(f1_match.group(1))
                all_ckpts.append((f1, entry))
        elif entry.is_dir():
            # Try subdirectory: "best-epoch=XXX-val_f1=0.XXXX" containing "0.XXXX.ckpt"
            epoch_match = re.match(r"best-epoch=(\d+)-val_f1", entry.name)
            if not epoch_match:
                continue
            ckpt_files = list(entry.glob("val_f1=*.ckpt"))
            if not ckpt_files:
                # Try alternative pattern
                ckpt_files = list(entry.glob("*.ckpt"))
            if not ckpt_files:
                continue
            ckpt_file = ckpt_files[0]
            f1_match = re.match(r"val_f1=([\d.]+)\.ckpt$", ckpt_file.name)
            if f1_match:
                f1 = float(f1_match.group(1))
                all_ckpts.append((f1, ckpt_file))
            else:
                # Try to extract from parent dir name
                dir_f1_match = re.search(r"val_f1=([\d.]+)$", entry.name)
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

    Args:
        out_path: Path for the final assembled TSV file.
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
    """Ensemble test by averaging logits from multiple checkpoints.

    This is a within-node ensemble: all checkpoints are from the same training run.
    No cross-node artifacts are used. All predictions are self-contained.

    With warm restarts, checkpoints span multiple LR phases (cycle peaks and valleys),
    providing temporal diversity. With save_top_k=10, the ensemble selects the 10
    best checkpoints by val_f1 from across all restart cycles.

    Args:
        model_kwargs: Keyword arguments to instantiate PerturbModule
        ckpt_paths:   Paths to checkpoints to ensemble (top-k by val_f1)
        test_dataloader: DataLoader for the test set
        out_path:     Path to save final ensemble predictions TSV
        device:       Device for inference (typically cuda:0)
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

        # Set up string embeddings (without DDP) — use _load_string_embeddings to avoid
        # trainer dependency (setup() calls self.print which requires self.trainer).
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
        description="Node1-3-2-2-1-1-1-1-1-1: STRING-only + Flat Head + hidden=384 + "
                    "Muon LR=0.01 + CosineWarmRestarts + Manifold Mixup + top-10 ensemble"
    )
    p.add_argument("--micro-batch-size",      type=int,   default=32)
    p.add_argument("--global-batch-size",     type=int,   default=256)
    p.add_argument("--max-epochs",            type=int,   default=350,
                   help="KEY CHANGE: 300 → 350 (covers warm restart cycle 3 from epoch 240)")
    p.add_argument("--lr",                    type=float, default=3e-4)
    p.add_argument("--muon-lr",               type=float, default=0.01,
                   help="Muon optimizer LR (proven stable from node3-3-1-1 and parent)")
    p.add_argument("--weight-decay",          type=float, default=8e-4,
                   help="KEY CHANGE: 5e-4 → 8e-4 (stronger L2 regularization to reduce overfitting)")
    p.add_argument("--hidden-dim",            type=int,   default=384)
    p.add_argument("--n-blocks",              type=int,   default=3)
    p.add_argument("--dropout",               type=float, default=0.28,
                   help="Trunk dropout (0.30 → 0.28 to balance head_dropout increase)")
    p.add_argument("--head-dropout",          type=float, default=0.18,
                   help="Head dropout (proven overfitting reduction from parent)")
    p.add_argument("--label-smoothing",       type=float, default=0.0,
                   help="No label smoothing (removes training loss floor, proven in parent)")
    p.add_argument("--class-weight-alpha",    type=float, default=0.35,
                   help="KEY CHANGE: 0.4 → 0.35 (more aggressive minority class focus)")
    p.add_argument("--cosine-t0",             type=int,   default=80,
                   help="KEY CHANGE: T_0 for CosineAnnealingWarmRestarts (first cycle length)")
    p.add_argument("--cosine-t-mult",         type=int,   default=2,
                   help="KEY CHANGE: T_mult for CosineAnnealingWarmRestarts (cycle length multiplier)")
    p.add_argument("--cosine-eta-min",        type=float, default=5e-7,
                   help="CosineAnnealingWarmRestarts minimum LR per cycle")
    p.add_argument("--early-stop-patience",   type=int,   default=70,
                   help="KEY CHANGE: 50 → 70 (longer patience for warm restart post-restart regression)")
    p.add_argument("--grad-clip-norm",        type=float, default=1.0)
    p.add_argument("--no-muon",               action="store_true",
                   help="Disable Muon optimizer, fall back to AdamW for all params")
    p.add_argument("--ensemble-top-k",        type=int,   default=10,
                   help="KEY CHANGE: 7 → 10 checkpoints for richer ensemble across restart cycles")
    p.add_argument("--mixup-alpha",           type=float, default=0.2,
                   help="KEY CHANGE: Manifold Mixup Beta distribution parameter (0.0 = disabled)")
    p.add_argument("--mixup-prob",            type=float, default=0.5,
                   help="KEY CHANGE: probability to apply Mixup in each training step")
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

    # save_top_k=10: Extended from parent's 7 for richer ensemble across restart cycles.
    # With warm restarts, models at cycle peaks (just after restart) and valleys (end of cycle)
    # may have complementary strengths. Top-10 captures more diverse temporal states.
    # KEY FIX: filename="{epoch:03d}-{val_f1:.4f}" — no "best-" prefix avoids double-prefix
    # issue in Lightning's ModelCheckpoint subdir mode. Compatible with _discover_local_checkpoints
    # using r"-([\d.]+)\.ckpt$" pattern.
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="{epoch:03d}-{val_f1:.4f}",  # No val_f1= prefix (fixes double-prefix bug)
        monitor="val_f1",
        mode="max",
        save_top_k=10,        # KEY CHANGE: 7 → 10 for richer ensemble across restart cycles
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stop_patience,  # 70 (extended for warm restart post-restart regression)
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

        # Step 2: Ensemble test on rank 0 (overwrite test_predictions.tsv with top-10 ensemble)
        # This is a within-node ensemble; no cross-node artifacts are used.
        # With warm restarts, top-10 checkpoints span multiple LR phases providing diversity.
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

        # Synchronize all ranks after rank-0 ensemble
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    # --- Save test score (rank 0 only) ---
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")


if __name__ == "__main__":
    main()
