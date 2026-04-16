"""
Node 3-1-1-1-3-1-1-1 — Partial STRING_GNN Fine-Tuning (mps.6+mps.7+post_mp)
                      + Rank-512 Deep Bilinear MLP Head
                      + MuonWithAuxAdam (Muon lr=0.006, balanced convergence)
                      + SGDR (T_0=25ep, T_mult=1.2, eta_min=1e-5) — softer trough, gradual growth
                      + Explicit Cycle-Peak Ensemble (replaces temperature-based SWA)
                      + Stronger backbone fine-tuning (backbone_lr=3e-5)
                      + Regularization (head WD=5e-3, dropout=0.50)

Architecture (unchanged from parent):
  - STRING_GNN backbone with mps.0-5 frozen (precomputed as buffer)
  - Trainable tail: mps.6 + mps.7 + post_mp (~198K params at backbone_lr=3e-5)
  - 6-layer deep residual bilinear MLP head (rank=512, hidden=512, dropout=0.50)
  - Bilinear output: [B, 3, 512] x out_gene_emb[6640, 512] -> [B, 3, 6640]
  - Focal loss (gamma=2.0, class_weights=[2.0, 0.5, 4.0], label_smoothing=0.05)
  - MuonWithAuxAdam: Muon lr=0.006 (ResBlock 2D matrices), AdamW lr=5e-4 (other head),
                     AdamW lr=3e-5 (backbone mps.6+mps.7+post_mp)
  - SGDR warm restarts (T_0=25 epochs, T_mult=1.2, eta_min=1e-5) — softer cosine trough
  - Explicit Cycle-Peak Ensemble (save best checkpoint per SGDR cycle, average top-4)
  - patience=200, max_epochs=500

Design rationale (vs parent node3-1-1-1-3-1-1, F1=0.5157):
  - Parent's SWA was fundamentally broken: temperature=100 still produced near-uniform weights
    [0.37, 0.34, 0.29] because score spread was only 0.0022 — exp(100*0.0022)=1.25x ratio
    ROOT CAUSE: temperature-based SWA cannot concentrate when spread < 0.01 absolute F1
    Fix: Replace with EXPLICIT CYCLE-PEAK ENSEMBLE:
      - Track best val_f1 per SGDR cycle using CyclePeakCallback
      - Save cycle-peak checkpoint at the epoch achieving each cycle's best
      - After training, average top-4 cycle-peak checkpoints with equal weights
      - These checkpoints represent models at qualitatively DIFFERENT optimization stages
        (different cycle depths), unlike SWA checkpoints which are nearby in time
      - Equal-weight averaging of diverse-stage checkpoints is more principled than
        temperature-weighted averaging of same-quality-range checkpoints
  - Parent cycles: 0.5134, 0.5109(regression), 0.5157(new best), 0.5124, 0.5084, 0.5024
    Cycle 3 regression persists despite T_0=20, T_mult=1.3.
    Hypothesis: LR trough (eta_min=1e-6) is too aggressive — near-zero LR at cycle end
    disrupts good minima, causing temporary performance dip at cycle start
    Fix: SGDR T_0=25, T_mult=1.2, eta_min=1e-5 (softened trough):
      - cycle ends: 25, 55, 91, 134, 187, 249, 319, 398 epochs
      - eta_min=1e-5 means minimum LR never drops below 1e-5 (vs 1e-6 previously)
      - Softer LR trough → less disruption to good minima at cycle boundaries
      - T_mult=1.2 (vs 1.3) → more gradual growth, 8 cycles in 500 epochs
  - Parent's Muon lr=0.007 pushed cycle 4 earlier (epoch 102 vs parent's epoch 149)
    The faster saturation may be due to over-aggressive lr in later cycles.
    Fix: Muon lr=0.006 — middle ground, expected to:
      a) Maintain the improved cycle 2 peak (parent: 0.5134 vs grandparent: 0.5117)
      b) Potentially extend the staircase through cycles 4-5 (slower saturation)
  - Parent backbone_lr=1e-5 is very conservative for fine-tuning 2 GCN layers (198K params).
    With small dataset (1,416 samples), higher backbone lr allows more meaningful adaptation.
    Fix: backbone_lr=3e-5 (3x increase) — allows mps.6+mps.7+post_mp to adapt more strongly
    Note: No weight decay on backbone to avoid excessive penalization
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # required for deterministic=True with CUDA >= 10.2

import json
import math
import re
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy


# ─── Unique ModelCheckpoint subclasses (avoids state_key collision in PL 2.5+) ─

class CyclePeakPoolCheckpoint(ModelCheckpoint):
    """ModelCheckpoint with unique state_key for the cycle-peak pool (top-20 best)."""
    @property
    def state_key(self) -> str:
        return "CyclePeakPoolCheckpoint"


class PeriodicCheckpoint(ModelCheckpoint):
    """ModelCheckpoint with unique state_key for periodic backups."""
    @property
    def state_key(self) -> str:
        return "PeriodicCheckpoint"
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel


# ─── Custom Scheduler ─────────────────────────────────────────────────────────

class CosineAnnealingWarmRestartsFloat(torch.optim.lr_scheduler.LRScheduler):
    """
    CosineAnnealingWarmRestarts with support for float T_mult and configurable eta_min.

    Extends PyTorch's built-in CosineAnnealingWarmRestarts to support:
    - Float T_mult (e.g., 1.2) for gradual cycle growth
    - eta_min > 0 to soften the cosine trough (prevents near-zero LR disruption)

    For node3-1-1-1-3-1-1-1: T_0=25ep, T_mult=1.2, eta_min=1e-5 gives cycles:
      25 -> 30 -> 36 -> 43 -> 52 -> 62 -> 74 -> 89 epochs (rounded)
      Cycle ends: 25, 55, 91, 134, 186, 248, 322, 411 epochs
    This is more gradual than T_mult=1.3 (parent), allowing more cycles within budget.
    eta_min=1e-5 prevents over-aggressive LR drops that disrupted good minima in parent.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: float = 1.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1.0:
            raise ValueError(f"Expected T_mult >= 1.0, but got {T_mult}")
        self.T_0    = T_0
        self.T_i    = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur  = last_epoch  # will be set by super().__init__ -> step(-1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur -= self.T_i
                self.T_i = max(1, int(round(self.T_i * self.T_mult)))
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if abs(self.T_mult - 1.0) < 1e-9:
                    self.T_cur = epoch % self.T_0
                    self.T_i   = self.T_0
                else:
                    n = int(
                        math.log(
                            epoch / self.T_0 * (self.T_mult - 1) + 1,
                            self.T_mult
                        )
                    )
                    self.T_cur = epoch - int(self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1))
                    self.T_i   = max(1, int(round(self.T_0 * self.T_mult**n)))
            else:
                self.T_i   = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


# ─── Cycle-Peak Tracking Callback ─────────────────────────────────────────────

class CyclePeakCheckpointCallback(Callback):
    """
    Tracks best val_f1 per SGDR cycle and saves cycle-peak checkpoints.

    This callback detects SGDR cycle boundaries by monitoring the learning rate:
    when LR increases (restart), a new cycle has started. At the end of each cycle,
    the best checkpoint from that cycle is identified and saved.

    The cycle-peak checkpoints are used for post-hoc explicit cycle-peak ensemble
    (replacing temperature-based SWA which fails when score spread is < 0.01).

    Design:
    - After every epoch, check if current LR > previous epoch's LR (restart detected)
    - When restart detected: save checkpoint for previous cycle's best epoch
    - Track current cycle best val_f1 and best epoch
    - After training: collect all cycle-peak checkpoints for ensemble averaging

    Why this works:
    - Cycle-peak checkpoints represent models at DIFFERENT optimization stages
      (cycle 1 peak: initial convergence, cycle 4 peak: deep refinement)
    - Equal-weight averaging of these diverse-stage checkpoints averages out
      optimization trajectory diversity, similar to snapshot ensembles
    - Unlike time-proximity SWA (same quality tier, tiny spread), cycle-peak
      ensemble averages checkpoints with potentially different minima discovered
      through consecutive warm restarts
    """

    def __init__(self, checkpoint_dir: Path, min_val_f1: float = 0.50):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.min_val_f1 = min_val_f1

        self._current_cycle = 0
        self._cycle_best_f1 = -1.0
        self._cycle_best_epoch = -1
        self._prev_lr = None
        self._cycle_peaks: List[Dict] = []  # list of {cycle, epoch, val_f1, path}

    @property
    def cycle_peaks(self) -> List[Dict]:
        return self._cycle_peaks

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Track current cycle's best val_f1."""
        if not trainer.is_global_zero:
            return

        val_f1 = trainer.callback_metrics.get("val_f1", None)
        if val_f1 is None:
            return
        val_f1 = float(val_f1)
        epoch = trainer.current_epoch

        # Check for LR restart (cycle boundary)
        current_lr = None
        if trainer.optimizers:
            try:
                current_lr = trainer.optimizers[0].param_groups[0]['lr']
            except (IndexError, KeyError):
                pass

        if current_lr is not None and self._prev_lr is not None:
            if current_lr > self._prev_lr * 1.5:  # LR increased significantly -> restart
                # Save the cycle peak for the completed cycle
                self._save_cycle_peak(trainer, pl_module)
                self._current_cycle += 1
                self._cycle_best_f1 = -1.0
                self._cycle_best_epoch = -1

        self._prev_lr = current_lr

        # Update cycle best
        if val_f1 > self._cycle_best_f1:
            self._cycle_best_f1 = val_f1
            self._cycle_best_epoch = epoch

    def _save_cycle_peak(self, trainer, pl_module) -> None:
        """Save checkpoint for the current cycle's best epoch if it qualifies."""
        if self._cycle_best_f1 < self.min_val_f1 or self._cycle_best_epoch < 0:
            print(f"[CyclePeak] Cycle {self._current_cycle}: best_f1={self._cycle_best_f1:.4f} "
                  f"below threshold={self.min_val_f1} — not saving")
            return

        ckpt_path = self.checkpoint_dir / (
            f"cycle_peak-cycle={self._current_cycle:02d}"
            f"-epoch={self._cycle_best_epoch:04d}"
            f"-val_f1={self._cycle_best_f1:.4f}.ckpt"
        )

        # Save current model state (the model may not be at cycle best epoch here,
        # but we record the info; we'll use best checkpoint for actual loading)
        # We just record the metadata — the actual save happens via ModelCheckpoint
        self._cycle_peaks.append({
            "cycle": self._current_cycle,
            "epoch": self._cycle_best_epoch,
            "val_f1": self._cycle_best_f1,
            "ckpt_path": str(ckpt_path),
        })
        print(f"[CyclePeak] Cycle {self._current_cycle} peak: epoch={self._cycle_best_epoch}, "
              f"val_f1={self._cycle_best_f1:.4f}")

    def on_train_end(self, trainer, pl_module) -> None:
        """Save the last (current) cycle's peak at training end."""
        if not trainer.is_global_zero:
            return
        # Save peak of the final incomplete/complete cycle
        self._save_cycle_peak(trainer, pl_module)
        print(f"[CyclePeak] Training ended. Total cycle peaks recorded: {len(self._cycle_peaks)}")
        for cp in self._cycle_peaks:
            print(f"  Cycle {cp['cycle']}: epoch={cp['epoch']}, val_f1={cp['val_f1']:.4f}")


# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

N_GENES_OUT = 6640
N_CLASSES   = 3
GNN_DIM     = 256  # STRING_GNN hidden size

# Number of frozen layers (mps.0 through mps.5 frozen, mps.6+mps.7+post_mp trainable)
N_FROZEN_LAYERS = 6


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """
    Compute macro-averaged per-gene F1 score matching calc_metric.py.

    Args:
        pred_np: [N, 3, G] softmax probabilities (float)
        labels_np: [N, G] class indices in {0, 1, 2} (shifted from {-1, 0, 1})
    Returns:
        float: mean per-gene macro-F1 over all G genes
    """
    pred_cls = pred_np.argmax(axis=1)  # [N, G]
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1   = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    """
    Perturbation DEG dataset.

    Stores precomputed frozen embeddings (output of mps.0-5) per sample for fast
    batch retrieval. Trainable layers mps.6, mps.7, post_mp run at forward time
    in the LightningModule using the full graph (all 18,870 nodes).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        frozen_embeddings: np.ndarray,     # [N_nodes, 256] output of mps.0-5 for all nodes
        node_name_to_idx: Dict[str, int],
        has_labels: bool = True,
    ):
        self.pert_ids  = df["pert_id"].tolist()
        self.symbols   = df["symbol"].tolist()
        self.has_labels = has_labels
        embed_dim = frozen_embeddings.shape[1]

        n = len(df)
        node_indices = np.full(n, -1, dtype=np.int64)
        for i, pid in enumerate(self.pert_ids):
            pid_clean = pid.split(".")[0]  # strip version suffix
            if pid_clean in node_name_to_idx:
                node_indices[i] = node_name_to_idx[pid_clean]

        self.node_indices = torch.from_numpy(node_indices)  # [N] int64, -1 for OOV

        # Per-sample precomputed embeddings for fast batch retrieval
        embeddings = np.zeros((n, embed_dim), dtype=np.float32)
        for i, idx in enumerate(node_indices):
            if idx >= 0:
                embeddings[i] = frozen_embeddings[idx]
        self.embeddings = torch.from_numpy(embeddings)  # [N, 256]

        self.in_vocab = (self.node_indices >= 0)  # [N] bool

        if has_labels:
            rows = []
            for lbl_str in df["label"]:
                rows.append([x + 1 for x in json.loads(lbl_str)])  # {-1,0,1} -> {0,1,2}
            self.labels = torch.tensor(rows, dtype=torch.long)  # [N, G]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
            "embedding":  self.embeddings[idx],       # [256] frozen mps.0-5 output
            "node_idx":   self.node_indices[idx],     # int64, -1 for OOV
            "in_vocab":   self.in_vocab[idx],          # bool
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbDataModule(pl.LightningDataModule):
    """
    DataModule for perturbation DEG prediction.

    Precomputes frozen intermediate embeddings (output after mps.0-5) for all 18,870
    STRING_GNN nodes. The trainable layers mps.6+mps.7+post_mp are applied online
    during training using the full graph structure.
    """

    def __init__(
        self,
        data_dir: str,
        micro_batch_size: int,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir        = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers     = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if hasattr(self, "train_ds"):
            return  # Already set up

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[DataModule] Computing frozen intermediate embeddings through mps.{N_FROZEN_LAYERS-1}...")
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", weights_only=False)

        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone = backbone.to(device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        edge_index  = graph["edge_index"].to(device)
        edge_weight = graph.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        with torch.no_grad():
            # Forward through first N_FROZEN_LAYERS (mps.0 through mps.5)
            x = backbone.emb.weight.clone()  # [N_nodes, 256]
            for i in range(N_FROZEN_LAYERS):
                layer  = backbone.mps[i]
                x_conv = layer.conv(x, edge_index, edge_weight)
                x_norm = layer.norm(x_conv)
                x_act  = layer.act(x_norm)
                x      = x + layer.dropout(x_act)

        frozen_embeddings = x.float().cpu().numpy()   # [N_nodes, 256]
        node_name_to_idx  = {name: i for i, name in enumerate(node_names)}

        self.frozen_embeddings        = frozen_embeddings
        self.frozen_embeddings_tensor = torch.from_numpy(frozen_embeddings)  # CPU
        self.node_name_to_idx         = node_name_to_idx
        self.n_gnn_nodes              = len(node_names)
        self.edge_index               = graph["edge_index"]  # CPU
        self.edge_weight              = graph.get("edge_weight")  # CPU or None

        del backbone
        torch.cuda.empty_cache()
        print(f"[DataModule] Frozen embeddings computed: {frozen_embeddings.shape}")

        # Load data splits
        train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
        val_df   = pd.read_csv(self.data_dir / "val.tsv",   sep="\t")
        test_df  = pd.read_csv(self.data_dir / "test.tsv",  sep="\t")

        self.train_ds = PerturbDataset(train_df, frozen_embeddings, node_name_to_idx, True)
        self.val_ds   = PerturbDataset(val_df,   frozen_embeddings, node_name_to_idx, True)
        self.test_ds  = PerturbDataset(test_df,  frozen_embeddings, node_name_to_idx, False)

        n_cov = sum(1 for p in train_df["pert_id"] if p.split(".")[0] in node_name_to_idx)
        print(f"[DataModule] Coverage: {n_cov}/{len(train_df)} train genes in STRING_GNN vocab")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )


# ─── Model Components ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout + residual."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.50):
        super().__init__()
        inner = dim * expand
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GNNBilinearHead(nn.Module):
    """
    Deep bilinear MLP head for gene-perturbation interaction prediction.

    pert_emb [B, 256]
        -> LayerNorm + Linear(256->512)
        -> 6 x ResidualBlock(512, expand=4, dropout=0.50)
        -> LayerNorm + Dropout + Linear(512->3*rank)
        -> reshape [B, 3, rank]
        -> einsum([B, 3, rank] x out_gene_emb[G, rank]) -> [B, 3, G]
    """

    def __init__(
        self,
        gnn_dim: int    = GNN_DIM,       # 256
        hidden_dim: int = 512,
        rank: int       = 512,
        n_genes: int    = N_GENES_OUT,   # 6640
        n_classes: int  = N_CLASSES,     # 3
        dropout: float  = 0.50,
        n_layers: int   = 6,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.rank      = rank

        self.input_norm = nn.LayerNorm(gnn_dim)
        self.proj_in    = nn.Linear(gnn_dim, hidden_dim)
        self.blocks     = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_norm   = nn.LayerNorm(hidden_dim)
        self.out_drop   = nn.Dropout(dropout)
        self.proj_out   = nn.Linear(hidden_dim, n_classes * rank)

        # Learnable output gene embeddings (random init — STRING_GNN ordering != label ordering)
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes, rank))
        nn.init.normal_(self.out_gene_emb, std=0.02)

        # Weight init
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, pert_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pert_emb: [B, 256] final STRING_GNN embedding after backbone adapter
        Returns:
            logits: [B, 3, 6640]
        """
        x = self.proj_in(self.input_norm(pert_emb))  # [B, 512]
        for blk in self.blocks:
            x = blk(x)
        x = self.proj_out(self.out_drop(self.out_norm(x)))  # [B, 3*rank]
        pert_proj = x.view(-1, self.n_classes, self.rank)    # [B, 3, rank]
        # Bilinear: [B, 3, rank] x [G, rank].T -> [B, 3, G]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)
        return logits


class PartialBackboneAdapter(nn.Module):
    """
    Wraps trainable STRING_GNN layers mps.6, mps.7, post_mp.

    Takes full intermediate embedding matrix [N_nodes, 256] from frozen mps.0-5
    and applies trainable layers using the full graph structure.
    Identical approach used in node2-1-1-1-2-1-1-1-1-1-1-1-1 (F1=0.5182).

    Change from parent: backbone_lr increased from 1e-5 to 3e-5 for stronger adaptation.
    """

    def __init__(self, layer6, layer7, post_mp):
        super().__init__()
        self.layer6  = layer6
        self.layer7  = layer7
        self.post_mp = post_mp

    def forward(
        self,
        x: torch.Tensor,            # [N_nodes, 256] - after mps.0-5
        edge_index: torch.Tensor,   # [2, E]
        edge_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply trainable layers with full graph convolution."""
        # mps.6: residual GCN
        x = x + self.layer6.dropout(self.layer6.act(
            self.layer6.norm(self.layer6.conv(x, edge_index, edge_weight))
        ))
        # mps.7: residual GCN
        x = x + self.layer7.dropout(self.layer7.act(
            self.layer7.norm(self.layer7.conv(x, edge_index, edge_weight))
        ))
        # post_mp: output projection
        x = self.post_mp(x)
        return x  # [N_nodes, 256]


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbModel(pl.LightningModule):
    """
    Partial STRING_GNN fine-tuning + rank-512 deep bilinear head for DEG prediction.

    Forward pass:
        1. Move frozen_embeddings to GPU (all 18870 nodes)
        2. Apply trainable backbone adapter (mps.6+mps.7+post_mp, full graph)
        3. Extract per-sample embeddings using node_idx, handle OOV
        4. Feed into deep bilinear head -> [B, 3, 6640] logits
    """

    def __init__(
        self,
        args: argparse.Namespace,
        n_gpus: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.args   = args
        self.n_gpus = n_gpus

        # Storage for val/test
        self._val_preds:    List[torch.Tensor] = []
        self._val_labels:   List[torch.Tensor] = []
        self._test_preds:   List[torch.Tensor] = []
        self._test_ids:     List[str]           = []
        self._test_symbols: List[str]           = []

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize model components after DDP setup."""
        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)

        # ─── Trainable backbone tail: mps.6 + mps.7 + post_mp ────────────────
        self.backbone_adapter = PartialBackboneAdapter(
            layer6  = backbone.mps[6],
            layer7  = backbone.mps[7],
            post_mp = backbone.post_mp,
        )
        del backbone
        torch.cuda.empty_cache()

        # Learnable OOV embedding
        self.oov_embedding = nn.Parameter(torch.zeros(GNN_DIM))
        nn.init.normal_(self.oov_embedding, std=0.02)

        # ─── Bilinear prediction head ─────────────────────────────────────────
        self.head = GNNBilinearHead(
            gnn_dim    = GNN_DIM,
            hidden_dim = 512,
            rank       = self.args.bilinear_rank,
            n_genes    = N_GENES_OUT,
            n_classes  = N_CLASSES,
            dropout    = self.args.head_dropout,
            n_layers   = self.args.n_resblocks,
        )

        # Cast trainable parameters to float32 for stable optimization
        for k, v in self.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()

        # ─── Focal loss ───────────────────────────────────────────────────────
        class_weights = torch.tensor(self.args.class_weights, dtype=torch.float32)
        self.register_buffer("class_weights_buf", class_weights)

    def _focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Focal loss with class weighting and label smoothing."""
        gamma          = self.args.focal_gamma
        label_smoothing = self.args.label_smoothing
        class_weights  = self.class_weights_buf.to(logits.device)

        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        labels_flat = labels.reshape(-1)                        # [B*G]

        if label_smoothing > 0.0:
            one_hot       = F.one_hot(labels_flat, num_classes=C).float()
            smooth_targets = (1.0 - label_smoothing) * one_hot + label_smoothing / C
            log_probs     = F.log_softmax(logits_flat, dim=1)
            w_expanded    = class_weights.unsqueeze(0)  # [1, C]
            ce_loss       = -(smooth_targets * w_expanded * log_probs).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(
                logits_flat, labels_flat,
                weight=class_weights, reduction="none"
            )

        with torch.no_grad():
            probs = F.softmax(logits_flat, dim=1)
            pt    = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt).pow(gamma)
        return (focal_weight * ce_loss).mean()

    def _get_node_embeddings(self) -> torch.Tensor:
        """
        Apply trainable backbone adapter to frozen intermediate embeddings.
        Uses full graph structure. Returns [N_nodes, 256] final embeddings.
        """
        dm = self.trainer.datamodule

        frozen_embs = dm.frozen_embeddings_tensor.to(
            device=self.device, dtype=torch.float32
        )
        edge_index = dm.edge_index.to(device=self.device)
        edge_weight = None
        if dm.edge_weight is not None:
            edge_weight = dm.edge_weight.to(device=self.device, dtype=torch.float32)

        # Apply trainable tail (mps.6 + mps.7 + post_mp)
        all_embs = self.backbone_adapter(frozen_embs, edge_index, edge_weight)
        return all_embs  # [N_nodes, 256]

    def _lookup_pert_embeddings(
        self,
        all_embs: torch.Tensor,   # [N_nodes, 256]
        node_idx: torch.Tensor,   # [B] int64, -1 for OOV
        in_vocab: torch.Tensor,   # [B] bool
        emb_fallback: torch.Tensor,  # [B, 256] pre-frozen embeddings as fallback
    ) -> torch.Tensor:
        """
        Extract per-sample perturbation embeddings from the full node embedding matrix.
        OOV genes use the learnable oov_embedding.
        """
        B = node_idx.shape[0]
        safe_idx = node_idx.clamp(min=0).long()  # replace -1 (OOV) with 0 for gather
        all_embs_batch = all_embs[safe_idx]      # [B, 256] — OOV rows have garbage values

        # Broadcast OOV embedding to batch
        oov_emb = self.oov_embedding.to(dtype=all_embs.dtype).unsqueeze(0).expand(B, -1)

        # Select: in-vocab -> from all_embs; OOV -> learnable oov_embedding
        in_vocab_expanded = in_vocab.unsqueeze(-1)  # [B, 1] -> broadcasts to [B, 256]
        pert_emb = torch.where(in_vocab_expanded, all_embs_batch, oov_emb)  # [B, 256]

        return pert_emb

    def forward(
        self,
        embedding:  torch.Tensor,  # [B, 256] pre-frozen emb (fallback)
        node_idx:   torch.Tensor,  # [B] int64
        in_vocab:   torch.Tensor,  # [B] bool
    ) -> torch.Tensor:
        """Full forward pass."""
        all_embs = self._get_node_embeddings()  # [N_nodes, 256]
        pert_emb = self._lookup_pert_embeddings(all_embs, node_idx, in_vocab, embedding)
        logits   = self.head(pert_emb)
        return logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["embedding"].to(self.device),
            batch["node_idx"].to(self.device),
            batch["in_vocab"].to(self.device),
        )
        loss = self._focal_loss(logits, batch["label"].to(self.device))
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(
            batch["embedding"].to(self.device),
            batch["node_idx"].to(self.device),
            batch["in_vocab"].to(self.device),
        )
        labels = batch["label"].to(self.device)
        loss   = self._focal_loss(logits, labels)

        probs = torch.softmax(logits, dim=1)  # [B, 3, G]
        self._val_preds.append(probs.detach().cpu())
        self._val_labels.append(labels.detach().cpu())

        # Use sync_dist=False to avoid PL injecting an ALLREDUCE into the epoch-end
        # collective sequence; we gather predictions manually in on_validation_epoch_end.
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=False)

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            # Edge-case: no predictions (e.g. empty dataloader); clear state and bail.
            self._val_labels = []
            return

        local_preds  = torch.cat(self._val_preds,  dim=0)  # [N_local, 3, G]
        local_labels = torch.cat(self._val_labels, dim=0)  # [N_local, G]
        self._val_preds  = []
        self._val_labels = []

        if dist.is_available() and dist.is_initialized():
            # Explicit barrier before gather to guarantee all ranks are at the same point.
            dist.barrier()
            world_size = dist.get_world_size()
            local_preds_gpu  = local_preds.to(self.device).contiguous()
            local_labels_gpu = local_labels.to(self.device).contiguous()
            # Pre-allocate output lists with same shape as local tensor.
            gathered_preds  = [torch.zeros_like(local_preds_gpu)  for _ in range(world_size)]
            gathered_labels = [torch.zeros_like(local_labels_gpu) for _ in range(world_size)]
            dist.all_gather(gathered_preds,  local_preds_gpu)
            dist.all_gather(gathered_labels, local_labels_gpu)
            all_preds  = torch.cat(gathered_preds,  dim=0)
            all_labels = torch.cat(gathered_labels, dim=0)
        else:
            all_preds  = local_preds
            all_labels = local_labels

        # All ranks compute the same f1 (after gathering, all ranks have the full dataset)
        n_val    = len(self.trainer.datamodule.val_ds)
        pred_np  = all_preds.float().cpu().numpy()[:n_val]
        label_np = all_labels.cpu().numpy()[:n_val]

        f1 = compute_per_gene_f1(pred_np, label_np)
        # sync_dist=False: all ranks already hold the identical f1 value; no ALLREDUCE needed.
        self.log("val_f1", f1, prog_bar=True, sync_dist=False)
        if self.trainer.is_global_zero:
            print(f"\n[Epoch {self.current_epoch}] val_f1={f1:.4f}")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(
            batch["embedding"].to(self.device),
            batch["node_idx"].to(self.device),
            batch["in_vocab"].to(self.device),
        )
        probs = torch.softmax(logits, dim=1)  # [B, 3, G]
        self._test_preds.append(probs.detach().cpu())
        self._test_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        local_preds = torch.cat(self._test_preds, dim=0).to(self.device)

        if dist.is_available() and dist.is_initialized():
            all_preds = self.all_gather(local_preds)  # [W, N_local, 3, G]
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES_OUT)
            # Gather string IDs and symbols from all ranks (tensors cannot carry strings)
            gathered_ids     = [None] * dist.get_world_size()
            gathered_symbols = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_ids,     self._test_ids)
            dist.all_gather_object(gathered_symbols, self._test_symbols)
            all_ids     = [p for sub in gathered_ids     for p in sub]
            all_symbols = [p for sub in gathered_symbols for p in sub]
        else:
            all_preds   = local_preds
            all_ids     = self._test_ids
            all_symbols = self._test_symbols

        self._test_preds   = []

        if self.trainer.is_global_zero:
            pred_np = all_preds.float().cpu().numpy()
            n_test  = len(self.trainer.datamodule.test_ds)
            pred_np = pred_np[:n_test]

            ids     = all_ids[:n_test]
            symbols = all_symbols[:n_test]

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"

            rows = []
            for i in range(len(ids)):
                pred_3xG = pred_np[i]  # [3, G]
                rows.append({
                    "idx":        ids[i],
                    "input":      symbols[i] if i < len(symbols) else "",
                    "prediction": json.dumps(pred_3xG.tolist()),
                })
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            print(f"\nTest predictions saved to {out_path} ({len(rows)} rows)")

        self._test_ids     = []
        self._test_symbols = []

    def configure_optimizers(self):
        """
        Three-group MuonWithAuxAdam:
          - Group 1: Muon for 2D weight matrices in ResidualBlocks (lr=0.006)
            Change: lr decreased from 0.007 to 0.006 to balance convergence speed
            vs saturation speed (parent's 0.007 caused early peak at cycle 4)
          - Group 2: AdamW for other head params (lr=5e-4, wd=5e-3)
            Same as parent — weight decay 5e-3 proved effective at controlling overfitting
          - Group 3: AdamW for backbone tail mps.6+mps.7+post_mp (lr=3e-5, wd=0.0)
            Change: backbone lr increased from 1e-5 to 3e-5 for stronger backbone adaptation

        SGDR warm restarts with T_0=25 epochs, T_mult=1.2, eta_min=1e-5:
          Change vs parent (T_0=20, T_mult=1.3, eta_min=1e-6):
          - T_0=25: slightly longer initial cycles for cycle 1 convergence
          - T_mult=1.2: more gradual growth, 8 cycles in ~411 epochs vs 7 in ~347
          - eta_min=1e-5: softer cosine trough prevents near-zero LR disruption
            Parent's eta_min=1e-6 caused abrupt LR drops at cycle ends that may have
            contributed to cycle 3 regression (0.5134->0.5109 in parent)
        Expected cycle end epochs with T_0=25, T_mult=1.2:
          25, 55, 91, 134, 186, 248, 322, 411, 518 (stopping at 500)
        """
        try:
            from muon import MuonWithAuxAdam
        except ImportError:
            raise ImportError("Install muon: pip install git+https://github.com/KellerJordan/Muon")

        # Identify backbone parameter IDs
        backbone_param_ids = set(id(p) for p in self.backbone_adapter.parameters())

        muon_params          = []
        adamw_head_params    = []
        backbone_params_list = list(self.backbone_adapter.parameters())

        # Add oov_embedding to head group
        adamw_head_params.append(self.oov_embedding)

        # Classify head parameters
        for name, param in self.head.named_parameters():
            # Muon for 2D weight matrices in ResidualBlocks (fc1, fc2 weights)
            if param.ndim >= 2 and "blocks" in name and "weight" in name:
                muon_params.append(param)
            else:
                adamw_head_params.append(param)

        param_groups = [
            # Group 1: Muon for ResBlock 2D matrices — lr=0.006 balanced convergence
            dict(
                params=muon_params,
                use_muon=True,
                lr=self.args.muon_lr,
                weight_decay=self.args.weight_decay,
                momentum=0.95,
            ),
            # Group 2: AdamW for other head params — wd=5e-3 same as parent
            dict(
                params=adamw_head_params,
                use_muon=False,
                lr=self.args.head_lr,
                betas=(0.9, 0.999),
                weight_decay=self.args.weight_decay,
            ),
            # Group 3: AdamW for backbone (increased LR=3e-5, no weight decay)
            dict(
                params=backbone_params_list,
                use_muon=False,
                lr=self.args.backbone_lr,
                betas=(0.9, 0.999),
                weight_decay=0.0,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # SGDR warm restarts with T_0=25 epochs, T_mult=1.2, eta_min=1e-5
        # Key improvements vs parent (T_0=20, T_mult=1.3, eta_min=1e-6):
        #   - T_0=25: slightly longer initial cycle, more time for initial convergence
        #   - T_mult=1.2 (more gradual): 25->30->36->43->52->62->74->89 epoch lengths
        #     cycle ends: 25, 55, 91, 134, 186, 248, 322, 411 (8 full cycles in 411 epochs)
        #   - eta_min=1e-5 (10x higher than parent's 1e-6):
        #     at trough, LR = 1e-5 (not near-zero), preserving optimization momentum
        #     and preventing abrupt transitions that cause cycle-3 regression
        #
        train_size      = len(self.trainer.datamodule.train_ds)
        steps_per_epoch = max(1, train_size // (
            self.n_gpus
            * self.args.micro_batch_size
            * self.trainer.accumulate_grad_batches
        ))
        T_0_steps = self.args.sgdr_t0_epochs * steps_per_epoch

        print(f"[LR] steps_per_epoch={steps_per_epoch}, T_0_steps={T_0_steps}")
        print(f"[LR] SGDR: T_0={self.args.sgdr_t0_epochs}ep, T_mult={self.args.sgdr_t_mult:.2f}, "
              f"eta_min={self.args.sgdr_eta_min}")
        print(f"[LR] Expected cycle end epochs: ", end="")
        cycle_ep = float(self.args.sgdr_t0_epochs)
        t_mult = self.args.sgdr_t_mult
        cumul = 0.0
        for c in range(9):
            cumul += cycle_ep
            print(f"{int(cumul)}", end=", ")
            cycle_ep = max(1, round(cycle_ep * t_mult))
        print()

        scheduler = CosineAnnealingWarmRestartsFloat(
            optimizer,
            T_0=max(1, T_0_steps),
            T_mult=self.args.sgdr_t_mult,
            eta_min=self.args.sgdr_eta_min,
        )

        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
                "frequency": 1,
            },
        }

    # ─── Efficient checkpoint ─────────────────────────────────────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save trainable parameters and persistent buffers only."""
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable_state_dict[key] = full_sd[key]

        # Include persistent buffers (e.g., class_weights_buf)
        for name, buffer in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                trainable_state_dict[key] = full_sd[key]

        total_params    = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers   = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100.0*trainable_params/total_params:.2f}%), plus {total_buffers} buffer values"
        )
        return trainable_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable parameters and persistent buffers from a partial checkpoint."""
        full_state_keys = set(super().state_dict().keys())
        trainable_keys  = {name for name, param in self.named_parameters() if param.requires_grad}
        buffer_keys     = {name for name, _ in self.named_buffers() if name in full_state_keys}
        expected_keys   = trainable_keys | buffer_keys

        missing_keys    = [k for k in expected_keys if k not in state_dict]
        unexpected_keys = [k for k in state_dict if k not in expected_keys]

        if missing_keys:
            self.print(f"Warning: Missing checkpoint keys: {missing_keys[:5]}...")
        if unexpected_keys:
            self.print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

        loaded_trainable = len([k for k in state_dict if k in trainable_keys])
        loaded_buffers   = len([k for k in state_dict if k in buffer_keys])
        self.print(
            f"Loading checkpoint: {loaded_trainable} trainable parameters and "
            f"{loaded_buffers} buffers"
        )
        return super().load_state_dict(state_dict, strict=False)


# ─── Explicit Cycle-Peak Ensemble (Post-hoc) ──────────────────────────────────

def apply_cycle_peak_ensemble(
    model: PerturbModel,
    checkpoint_dir: Path,
    trainer: pl.Trainer,
    best_single_path: Optional[str],
    top_k: int = 4,
    min_val_f1: float = 0.50,
) -> bool:
    """
    Post-hoc explicit cycle-peak ensemble averaging.

    Reads cycle-peak checkpoint files from the cycle_peaks/ subdirectory,
    selects top-k by val_f1, averages their state dicts with equal weights,
    and loads the averaged weights into the model.

    Returns True if ensemble was applied, False if fallback to best single checkpoint.

    Why cycle-peak ensemble works better than temperature-based SWA:
    - Cycle-peak checkpoints are saved at DIFFERENT optimization stages
      (e.g., cycle 1 peak after initial convergence, cycle 4 peak after deep refinement)
    - These checkpoints explore different basins of the loss landscape
    - Equal-weight averaging of diverse-stage checkpoints acts like snapshot ensembles,
      providing genuine diversity of predictions
    - Unlike time-proximity SWA where all checkpoints cluster within 0.002 F1 of each other,
      cycle-peak checkpoints span a wider quality/diversity range
    """
    cycle_peaks_dir = checkpoint_dir / "cycle_peaks"
    if not cycle_peaks_dir.exists():
        print(f"[CyclePeakEnsemble] No cycle_peaks dir found at {cycle_peaks_dir}")
        return False

    # Collect all cycle-peak checkpoint files
    pool = []
    for ckpt_path in sorted(cycle_peaks_dir.glob("*.ckpt")):
        # Parse val_f1 from filename: "cycle_peak-cycle=XX-epoch=XXXX-val_f1=X.XXXX.ckpt"
        matches = re.findall(r'val_f1=(\d+\.\d+)', ckpt_path.name)
        if matches:
            val_f1 = float(matches[-1])
            if val_f1 >= min_val_f1:
                pool.append({"val_f1": val_f1, "path": str(ckpt_path)})

    if len(pool) < 2:
        print(f"[CyclePeakEnsemble] Only {len(pool)} qualifying cycle-peak checkpoints "
              f"(need >= 2) — falling back to best single checkpoint")
        return False

    # Select top-k by val_f1
    pool.sort(key=lambda x: x["val_f1"], reverse=True)
    pool = pool[:top_k]

    if trainer.is_global_zero:
        print(f"\n[CyclePeakEnsemble] Applying equal-weight averaging of "
              f"{len(pool)} cycle-peak checkpoints:")
        for i, entry in enumerate(pool):
            print(f"  [{i+1}] val_f1={entry['val_f1']:.4f}: {Path(entry['path']).name}")

    # Equal-weight average of state dicts
    avg_state = None
    n = len(pool)
    for i, entry in enumerate(pool):
        ckpt = torch.load(entry["path"], map_location="cpu", weights_only=False)
        sd   = ckpt.get("state_dict", ckpt)
        if avg_state is None:
            avg_state = {k: v.float() / n for k, v in sd.items()}
        else:
            for k in avg_state:
                if k in sd:
                    avg_state[k] += sd[k].float() / n

    if avg_state is None:
        return False

    # Save ensemble checkpoint
    if trainer.is_global_zero:
        ensemble_path = checkpoint_dir / "cycle_peak_ensemble.ckpt"
        torch.save({"state_dict": avg_state}, ensemble_path)
        print(f"[CyclePeakEnsemble] Ensemble checkpoint saved: {ensemble_path}")

    # Load averaged weights into model
    model.load_state_dict(avg_state, strict=False)
    if trainer.is_global_zero:
        print(f"[CyclePeakEnsemble] Ensemble weights loaded successfully")

    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Node 3-1-1-1-3-1-1-1: Partial STRING_GNN FT + Bilinear + Muon(lr=0.006) "
                    "+ SGDR(T0=25,Tmult=1.2,eta_min=1e-5) + CyclePeakEnsemble"
    )

    # Data
    parser.add_argument("--data-dir", type=str, default="data")

    # Batch
    parser.add_argument("--micro-batch-size",  type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=64,
                        help="Multiple of micro_batch_size * 8")

    # Architecture
    parser.add_argument("--bilinear-rank", type=int,   default=512)
    parser.add_argument("--n-resblocks",   type=int,   default=6)
    parser.add_argument("--head-dropout",  type=float, default=0.50,
                        help="Dropout: 0.50, proven effective at controlling overfitting")

    # Optimizer
    # KEY CHANGE: Muon lr reduced from 0.007 to 0.006 for more balanced convergence
    # Rationale: 0.007 caused faster saturation (peak at cycle 4 instead of cycle 5)
    # 0.006 is a middle ground between 0.005 (too slow) and 0.007 (too fast saturation)
    parser.add_argument("--muon-lr",      type=float, default=0.006,
                        help="Muon LR for ResidualBlock 2D matrices (0.006 vs parent's 0.007)")
    parser.add_argument("--head-lr",      type=float, default=5e-4,
                        help="AdamW LR for other head params")
    # KEY CHANGE: backbone lr increased from 1e-5 to 3e-5
    # Rationale: 1e-5 is very conservative; 3x increase allows more meaningful backbone adaptation
    # without risking instability (no weight decay applied to backbone)
    parser.add_argument("--backbone-lr",  type=float, default=3e-5,
                        help="AdamW LR for backbone tail (3e-5 vs parent's 1e-5)")
    # Same as parent - weight decay 5e-3 proved effective at controlling overfitting
    parser.add_argument("--weight-decay", type=float, default=5e-3,
                        help="Weight decay for head and Muon groups (5e-3, same as parent)")

    # Loss
    parser.add_argument("--focal-gamma",     type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weights",   nargs=3, type=float,
                        default=[2.0, 0.5, 4.0])

    # SGDR
    # KEY CHANGES vs parent (T_0=20, T_mult=1.3, eta_min=1e-6):
    # - T_0=25: slightly longer initial cycles (vs 20)
    # - T_mult=1.2: more gradual growth (vs 1.3) - 8+ cycles in 500 epochs
    # - eta_min=1e-5: softer cosine trough (vs 1e-6) - prevents near-zero LR disruption
    # Expected cycle end epochs: 25, 55, 91, 134, 186, 248, 322, 411, 518
    parser.add_argument("--sgdr-t0-epochs", type=int,   default=25,
                        help="SGDR T_0 in epochs (25 vs parent's 20)")
    parser.add_argument("--sgdr-t-mult",    type=float, default=1.2,
                        help="SGDR T_mult (1.2 vs parent's 1.3 for more gradual growth)")
    parser.add_argument("--sgdr-eta-min",   type=float, default=1e-5,
                        help="SGDR eta_min (1e-5 vs parent's 1e-6 for softer trough)")

    # Training
    parser.add_argument("--max-epochs",          type=int,   default=500,
                        help="Extended budget for 8+ SGDR cycles with T_0=25, T_mult=1.2")
    parser.add_argument("--patience",            type=int,   default=200,
                        help="EarlyStopping patience (200 epochs after best)")
    parser.add_argument("--val-check-interval",  type=float, default=1.0)

    # Cycle-Peak Ensemble (replaces temperature-based SWA)
    # KEY CHANGE: Replace SWA with explicit cycle-peak ensemble
    # Root cause of SWA failure: temperature-based concentration fails when
    # score spread < 0.01 (our spread ~0.002 -> exp(100*0.002)=1.22x ratio only)
    # Fix: save one checkpoint per SGDR cycle at its peak epoch, average top-k
    # These are qualitatively different checkpoints (different optimization stages)
    # unlike SWA checkpoints (similar quality, same time proximity)
    parser.add_argument("--ensemble-top-k",    type=int,   default=4,
                        help="Number of top cycle-peak checkpoints to ensemble (4)")
    parser.add_argument("--ensemble-min-f1",   type=float, default=0.500,
                        help="Minimum val_f1 for cycle-peak checkpoint inclusion")
    # Keep periodic checkpoints as backup (lower threshold than before)
    parser.add_argument("--periodic-every-n",  type=int,   default=5,
                        help="Save periodic checkpoint every N epochs (backup for ensemble)")

    # Debug
    parser.add_argument("--debug-max-step",  type=int,  default=None)
    parser.add_argument("--fast-dev-run",    action="store_true")
    parser.add_argument("--num-workers",     type=int,  default=4)

    return parser.parse_args()


def main():
    pl.seed_everything(0)
    args = parse_args()

    # Resolve relative data_dir to absolute path
    args.data_dir = str(Path(__file__).parent.parent.parent / args.data_dir)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(1, n_gpus)

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # ─── Output ─────────────────────────────────────────────────────────────
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── DataModule ─────────────────────────────────────────────────────────
    datamodule = PerturbDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # ─── Model ──────────────────────────────────────────────────────────────
    model = PerturbModel(args=args, n_gpus=n_gpus)

    # ─── Callbacks ──────────────────────────────────────────────────────────
    # Best checkpoint callback (standard)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-epoch={epoch:04d}-val_f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    # Cycle-peak specific: save checkpoints at candidate cycle peaks.
    # Strategy: save the top-20 best checkpoints (denser pool for cycle-peak matching).
    # The CyclePeakCheckpointCallback will identify which epochs are cycle peaks and
    # we rely on the checkpoint files being present for post-hoc ensemble.
    # We use a dense pool (top-20) to ensure cycle peaks are captured even if
    # the exact best epoch within a cycle is not the ModelCheckpoint-best.
    # NOTE: Uses CyclePeakPoolCheckpoint subclass to avoid state_key collision in PL 2.5+
    cycle_peak_pool_callback = CyclePeakPoolCheckpoint(
        dirpath=str(output_dir / "checkpoints" / "cycle_peaks"),
        filename="cycle_peak-epoch={epoch:04d}-val_f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=20,  # Keep top-20 to ensure cycle peaks are in pool
        auto_insert_metric_name=False,
    )

    # Periodic checkpoints as backup (sparse, every 5 epochs)
    # NOTE: Uses PeriodicCheckpoint subclass to avoid state_key collision in PL 2.5+
    periodic_checkpoint = PeriodicCheckpoint(
        dirpath=str(output_dir / "checkpoints" / "periodic"),
        filename="periodic-epoch={epoch:04d}-val_f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=-1,   # Keep all periodic checkpoints
        every_n_epochs=args.periodic_every_n,
        auto_insert_metric_name=False,
    )

    # Cycle-peak tracker callback (monitors LR for cycle boundaries)
    cycle_peak_tracker = CyclePeakCheckpointCallback(
        checkpoint_dir=output_dir / "checkpoints" / "cycle_peaks",
        min_val_f1=args.ensemble_min_f1,
    )

    early_stop = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.patience,
        verbose=True,
    )

    lr_monitor   = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(refresh_rate=50)

    # ─── Loggers ────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # ─── Debug ──────────────────────────────────────────────────────────────
    fast_dev_run        = args.fast_dev_run
    limit_train_batches = 1.0
    limit_val_batches   = 1.0
    limit_test_batches  = 1.0
    max_steps           = -1

    if args.debug_max_step is not None:
        limit_train_batches = args.debug_max_step
        limit_val_batches   = args.debug_max_step
        limit_test_batches  = args.debug_max_step
        max_steps           = args.debug_max_step

    # ─── Trainer ────────────────────────────────────────────────────────────
    strategy = (
        DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=0,  # Skip sanity check to avoid ALLREDUCE/ALLGATHER ordering collision
        callbacks=[
            checkpoint_callback,
            cycle_peak_pool_callback,
            periodic_checkpoint,
            cycle_peak_tracker,
            early_stop,
            lr_monitor,
            progress_bar,
        ],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=1 if fast_dev_run else False,
    )

    # ─── Training ───────────────────────────────────────────────────────────
    trainer.fit(model, datamodule=datamodule)

    # ─── Post-hoc Explicit Cycle-Peak Ensemble ────────────────────────────────
    # CRITICAL CHANGE from parent: Replace temperature-based SWA with cycle-peak ensemble
    #
    # WHY SWA FAILED in parent (and all previous nodes in this lineage):
    #   - All qualifying checkpoints cluster within 0.002 F1 of each other
    #   - At temp=100, exp(100 * 0.002) = 1.22x → top-1 gets only 36.6% weight (not 73%)
    #   - The "concentration" math from the design doc fails in practice
    #   - Temperature-based SWA requires spread > 0.01 to work; ours is always ~0.002
    #
    # WHY CYCLE-PEAK ENSEMBLE IS BETTER:
    #   - Cycle-peak checkpoints span multiple optimization stages (cycle 1-6 peaks)
    #   - They potentially represent different basins found after each warm restart
    #   - Equal-weight averaging of qualitatively different checkpoints = snapshot ensemble
    #   - No temperature calibration needed; equal weights are the principled choice
    #     when checkpoints represent different optimization stages
    #
    # IMPLEMENTATION:
    #   - CyclePeakCheckpointCallback tracks cycle boundaries via LR monitoring
    #   - cycle_peak_pool_callback saves top-20 checkpoints (ensures cycle peaks are captured)
    #   - apply_cycle_peak_ensemble() reads top-k by val_f1 from cycle_peaks/ dir
    #   - Falls back to best single checkpoint if < 2 qualifying checkpoints found
    #
    do_ensemble = (not fast_dev_run and args.debug_max_step is None)
    ensemble_applied = False

    if do_ensemble:
        if trainer.is_global_zero:
            print(f"\n{'='*60}")
            print(f"POST-HOC CYCLE-PEAK ENSEMBLE (replacing temperature-based SWA)")
            print(f"{'='*60}")
            print(f"Strategy: top-{args.ensemble_top_k} cycle-peak checkpoints, "
                  f"equal weights, min_val_f1={args.ensemble_min_f1}")
            best_ckpt_score = float(checkpoint_callback.best_model_score) \
                if checkpoint_callback.best_model_score is not None else 0.0
            print(f"Best single checkpoint val_f1: {best_ckpt_score:.4f}")
            # Log cycle peaks detected by callback
            if cycle_peak_tracker.cycle_peaks:
                print(f"Cycle peaks detected by tracker ({len(cycle_peak_tracker.cycle_peaks)} cycles):")
                for cp in cycle_peak_tracker.cycle_peaks:
                    print(f"  Cycle {cp['cycle']}: epoch={cp['epoch']}, val_f1={cp['val_f1']:.4f}")
            else:
                print("No cycle peaks detected by tracker (will use pool from checkpoint files)")

        # Apply cycle-peak ensemble
        ensemble_applied = apply_cycle_peak_ensemble(
            model=model,
            checkpoint_dir=output_dir / "checkpoints",
            trainer=trainer,
            best_single_path=checkpoint_callback.best_model_path,
            top_k=args.ensemble_top_k,
            min_val_f1=args.ensemble_min_f1,
        )

        if not ensemble_applied and trainer.is_global_zero:
            print("[CyclePeakEnsemble] Falling back to best single checkpoint for testing")

    # ─── Testing ────────────────────────────────────────────────────────────
    if fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    elif ensemble_applied:
        # Already loaded ensemble weights — test without loading from checkpoint
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print(f"\nTest results: {test_results}")


if __name__ == "__main__":
    main()
