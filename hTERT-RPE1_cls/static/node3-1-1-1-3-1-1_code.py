"""
Node 3-1-1-1-3-1-1 — Partial STRING_GNN Fine-Tuning (mps.6+mps.7+post_mp)
                    + Rank-512 Deep Bilinear MLP Head
                    + MuonWithAuxAdam (Muon lr=0.007)
                    + SGDR (T_0=20ep, T_mult=1.3) — longer, convergence-friendly cycles
                    + Ultra-tight Quality-Filtered SWA (top-k=3, temp=100.0, threshold=0.510)
                    + Stronger regularization (head WD=5e-3, dropout=0.50)

Architecture:
  - STRING_GNN backbone with mps.0-5 frozen (precomputed as buffer)
  - Trainable tail: mps.6 + mps.7 + post_mp (~198K params at backbone_lr=1e-5)
  - 6-layer deep residual bilinear MLP head (rank=512, hidden=512, dropout=0.50)
  - Bilinear output: [B, 3, 512] x out_gene_emb[6640, 512] → [B, 3, 6640]
  - Focal loss (gamma=2.0, class_weights=[2.0, 0.5, 4.0], label_smoothing=0.05)
  - MuonWithAuxAdam: Muon lr=0.007 (ResBlock 2D matrices), AdamW lr=5e-4 (other head),
                     AdamW lr=1e-5 (backbone mps.6+mps.7+post_mp)
  - SGDR warm restarts (T_0=20 epochs, T_mult=1.3) — longer per-cycle for deeper convergence
  - Ultra-tight SWA (top-k=3, temp=100.0, threshold=0.510, every-2-epochs)
  - patience=190, max_epochs=450

Design rationale (vs parent node3-1-1-1-3-1, F1=0.5142):
  - Parent's SWA was near-useless: top-8 averaging at temp=10.0 gave near-uniform weights
    (0.123-0.130 each) because score spread was <0.006 → SWA gave 0 measurable gain
    Fix: reduce top-k to 3, raise threshold to 0.510, temperature=100.0
    With temp=100.0 and typical spread [0.5100, 0.5110, 0.5125], weights would be
    ~95% on top-1 vs 3% and 2% — effectively selecting the single best + tiny correction
  - Parent cycles 3-4 regressed from cycle 2 (0.5117→0.5111→0.5098), recovered at cycle 5
    Hypothesis: T_mult=1.5 caused cycles 3-4 to be 33-50ep — too short for convergence
    after restarts disturb the well-tuned weights from a good local minimum
    Fix: T_0=20, T_mult=1.3 → cycles end at 20, 46, 80, 124, 181, 255, 347
    Each cycle is ~30% longer than the previous, allowing gradual deepening of convergence
  - Parent best epoch was 149 with dropout=0.50 (from parent's epoch 93 → +60ep delay)
    Hypothesis: more aggressive overfitting suppression in later cycles requires stronger WD
    Fix: head weight decay 3e-3→5e-3 to suppress the train/val ratio growth (2.56x at ep300)
  - Parent Muon lr=0.005 produced cycle 2 peak at 0.5117 (epoch 21)
    Hypothesis: higher Muon lr can push cycle 2 peak above 0.515 if cycles are also longer
    Fix: Muon lr 0.005→0.007 for stronger early-cycle gains (based on feedback Priority 4)
  - Extended training budget (max_epochs=450, patience=190):
    With T_0=20, T_mult=1.3, cycle ends: 20, 46, 80, 124, 181, 255, 347, 443
    Need enough budget to test 7-8 cycles, 190-epoch patience allows observation of multiple
    cycles after peak before giving up
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
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ─── Custom Scheduler ─────────────────────────────────────────────────────────

class CosineAnnealingWarmRestartsFloat(torch.optim.lr_scheduler.LRScheduler):
    """
    CosineAnnealingWarmRestarts with support for float T_mult.

    PyTorch's built-in CosineAnnealingWarmRestarts only supports integer T_mult.
    This implementation extends it to support float T_mult (e.g., T_mult=1.3),
    allowing gradual cycle length growth for more fine-grained SGDR scheduling.

    For node3-1-1-1-3-1-1: T_0=20ep, T_mult=1.3 gives cycles:
      20 → 26 → 33 → 43 → 56 → 72 → 94 → 122 epochs
    This is a more gradual growth than T_mult=1.5 (used in parent), allowing
    deeper convergence per cycle without excessive length growth.
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
        self.T_cur  = last_epoch  # will be set by super().__init__ → step(-1)
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
                rows.append([x + 1 for x in json.loads(lbl_str)])  # {-1,0,1} → {0,1,2}
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
    """Residual MLP block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout + residual."""

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
        → LayerNorm + Linear(256→512)
        → 6 × ResidualBlock(512, expand=4, dropout=0.50)
        → LayerNorm + Dropout + Linear(512→3×rank)
        → reshape [B, 3, rank]
        → einsum([B, 3, rank] × out_gene_emb[G, rank]) → [B, 3, G]
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

        # Learnable output gene embeddings (random init — STRING_GNN ordering ≠ label ordering)
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
        # Bilinear: [B, 3, rank] × [G, rank].T → [B, 3, G]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)
        return logits


class PartialBackboneAdapter(nn.Module):
    """
    Wraps trainable STRING_GNN layers mps.6, mps.7, post_mp.

    Takes full intermediate embedding matrix [N_nodes, 256] from frozen mps.0-5
    and applies trainable layers using the full graph structure.
    This is identical to the approach used in node2-1-1-1-2-1-1-1-1-1-1-1-1 (F1=0.5182).
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
        4. Feed into deep bilinear head → [B, 3, 6640] logits
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

        # SWA pool (populated via periodic checkpoints at test time, not training hooks)
        self._swa_pool: List[dict] = []

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

        Args:
            all_embs: [N_nodes, 256] full node embedding matrix after backbone adapter
            node_idx: [B] node indices (-1 for OOV)
            in_vocab: [B] bool mask (True = in vocab)
            emb_fallback: [B, 256] pre-frozen per-sample embeddings (fallback for OOV)
        Returns:
            pert_emb: [B, 256]
        """
        B = node_idx.shape[0]
        # Avoid boolean-indexing assignment on bf16 tensors (PyTorch CUDA bug).
        # Use torch.where which is dtype-safe and differentiable.
        safe_idx = node_idx.clamp(min=0).long()  # replace -1 (OOV) with 0 for gather
        all_embs_batch = all_embs[safe_idx]      # [B, 256] — OOV rows have garbage values

        # Broadcast OOV embedding to batch
        oov_emb = self.oov_embedding.to(dtype=all_embs.dtype).unsqueeze(0).expand(B, -1)

        # Select: in-vocab → from all_embs; OOV → learnable oov_embedding
        in_vocab_expanded = in_vocab.unsqueeze(-1)  # [B, 1] → broadcasts to [B, 256]
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

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        if dist.is_available() and dist.is_initialized():
            all_preds  = self.all_gather(torch.cat(self._val_preds,  dim=0).to(self.device))
            all_labels = self.all_gather(torch.cat(self._val_labels, dim=0).to(self.device))
            all_preds  = all_preds.view(-1, N_CLASSES, N_GENES_OUT)
            all_labels = all_labels.view(-1, N_GENES_OUT)
        else:
            all_preds  = torch.cat(self._val_preds,  dim=0)
            all_labels = torch.cat(self._val_labels, dim=0)

        self._val_preds  = []
        self._val_labels = []

        # All ranks compute the same f1 (after gathering, all ranks have the full dataset)
        n_val    = len(self.trainer.datamodule.val_ds)
        pred_np  = all_preds.float().cpu().numpy()[:n_val]
        label_np = all_labels.cpu().numpy()[:n_val]

        f1 = compute_per_gene_f1(pred_np, label_np)
        # Log on all ranks with the same value (all ranks computed from gathered tensors).
        # sync_dist=True silences Lightning's DDP logging warning and is safe here.
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
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
          - Group 1: Muon for 2D weight matrices in ResidualBlocks (lr=0.007)
            Key change: Muon lr increased from 0.005 to 0.007 for faster early-cycle gains
          - Group 2: AdamW for other head params (lr=5e-4, wd=5e-3)
            Key change: weight decay increased from 3e-3 to 5e-3 for stronger regularization
          - Group 3: AdamW for backbone tail mps.6+mps.7+post_mp (lr=1e-5)
        SGDR warm restarts with T_0=20 epochs, T_mult=1.3 (more gradual cycle growth)
        Key change: T_0=20 (vs parent's 15), T_mult=1.3 (vs parent's 1.5)
        Expected cycles: 20, 46, 80, 124, 181, 255, 347 epochs
        Longer cycles allow deeper convergence per restart, potentially fixing cycles 3-4 regression
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
            # Group 1: Muon for ResBlock 2D matrices — higher lr=0.007 for faster early gains
            dict(
                params=muon_params,
                use_muon=True,
                lr=self.args.muon_lr,
                weight_decay=self.args.weight_decay,
                momentum=0.95,
            ),
            # Group 2: AdamW for other head params — stronger wd=5e-3 for late-cycle regularization
            dict(
                params=adamw_head_params,
                use_muon=False,
                lr=self.args.head_lr,
                betas=(0.9, 0.999),
                weight_decay=self.args.weight_decay,
            ),
            # Group 3: AdamW for backbone (lower LR, no weight decay)
            dict(
                params=backbone_params_list,
                use_muon=False,
                lr=self.args.backbone_lr,
                betas=(0.9, 0.999),
                weight_decay=0.0,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # SGDR warm restarts with T_0=20 epochs, T_mult=1.3
        # Key improvement vs parent (T_0=15, T_mult=1.5):
        #   Parent's shorter cycles (15→22→33→50→75...) may have been too short for cycles 3-4
        #   causing slight regression from cycle 2's peak before recovering at cycle 5.
        #   T_mult=1.3 with T_0=20 → cycles: 20→26→34→44→57→74→96→125 epochs
        #   (Note: implementation rounds to nearest integer each cycle)
        #   Longer cycles allow the optimizer to converge more deeply within each restart,
        #   potentially producing a more consistently ascending staircase.
        #
        # steps_per_epoch = ceil(train_size / (n_gpus * micro_batch_size * accum))
        train_size      = len(self.trainer.datamodule.train_ds)
        steps_per_epoch = max(1, train_size // (
            self.n_gpus
            * self.args.micro_batch_size
            * self.trainer.accumulate_grad_batches
        ))
        T_0_steps = self.args.sgdr_t0_epochs * steps_per_epoch

        print(f"[LR] steps_per_epoch={steps_per_epoch}, T_0_steps={T_0_steps}")
        print(f"[LR] SGDR: T_0={self.args.sgdr_t0_epochs}ep, T_mult={self.args.sgdr_t_mult:.2f}")
        print(f"[LR] Expected cycle end epochs: ", end="")
        cycle_ep = float(self.args.sgdr_t0_epochs)
        t_mult = self.args.sgdr_t_mult
        cumul = 0.0
        for c in range(8):
            cumul += cycle_ep
            print(f"{int(cumul)}", end=", ")
            cycle_ep = max(1, round(cycle_ep * t_mult))
        print()

        # Use custom scheduler to support float T_mult (e.g., 1.3).
        # PyTorch's built-in CosineAnnealingWarmRestarts only accepts int T_mult.
        scheduler = CosineAnnealingWarmRestartsFloat(
            optimizer,
            T_0=max(1, T_0_steps),
            T_mult=self.args.sgdr_t_mult,
            eta_min=1e-6,
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Node 3-1-1-1-3-1-1: Partial STRING_GNN FT + Bilinear + Muon(lr=0.007) + SGDR(T0=20,Tmult=1.3) + Ultra-tight SWA"
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
                        help="Dropout: 0.50, same as parent (effective at delaying peak epoch)")

    # Optimizer
    # KEY CHANGE: Muon lr increased from 0.005 to 0.007 for stronger early-cycle gains
    parser.add_argument("--muon-lr",      type=float, default=0.007,
                        help="Muon LR for ResidualBlock 2D matrices (0.007 vs parent's 0.005)")
    parser.add_argument("--head-lr",      type=float, default=5e-4,
                        help="AdamW LR for other head params")
    parser.add_argument("--backbone-lr",  type=float, default=1e-5,
                        help="AdamW LR for backbone tail")
    # KEY CHANGE: Weight decay increased from 3e-3 to 5e-3 for stronger regularization
    parser.add_argument("--weight-decay", type=float, default=5e-3,
                        help="Weight decay (5e-3 vs parent's 3e-3 for stronger late-cycle regularization)")

    # Loss
    parser.add_argument("--focal-gamma",     type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weights",   nargs=3, type=float,
                        default=[2.0, 0.5, 4.0])

    # SGDR
    # KEY CHANGE: T_0=20ep (vs parent's 15), T_mult=1.3 (vs parent's 1.5)
    # Longer cycles allow deeper convergence per restart, potentially fixing cycles 3-4 regression
    # Expected cycle end epochs: 20, 46, 80, 124, 181, 255, 347, 443
    parser.add_argument("--sgdr-t0-epochs", type=int,   default=20,
                        help="SGDR T_0 in epochs (20 vs parent's 15 for longer initial cycles)")
    parser.add_argument("--sgdr-t-mult",    type=float, default=1.3,
                        help="SGDR T_mult (1.3 vs parent's 1.5 for more gradual cycle growth)")

    # Training
    parser.add_argument("--max-epochs",          type=int,   default=450,
                        help="Extended budget for 7-8 SGDR cycles with T_0=20, T_mult=1.3")
    parser.add_argument("--patience",            type=int,   default=190,
                        help="EarlyStopping patience for extended SGDR cycles")
    parser.add_argument("--val-check-interval",  type=float, default=1.0)

    # SWA
    # KEY CHANGE: ultra-tight SWA parameters to concentrate on only the best checkpoints
    # Parent's SWA failure: near-uniform weights (0.123-0.130) despite temp=10.0 because
    # score spread was too small (<0.006) — functionally equivalent to equal averaging of
    # 8 diverse-epoch checkpoints, which may introduce noise without quality gain.
    # Fix: threshold=0.510 (above parent's best val F1 of 0.5142 minus 0.004 margin),
    #      top-k=3 (only the very best 3 checkpoints),
    #      temperature=100.0 (with score range [0.5100, 0.5125], top-1 gets ~95%+ weight)
    parser.add_argument("--swa-threshold",   type=float, default=0.510,
                        help="Minimum val_f1 for SWA pool inclusion (0.510 vs parent's 0.505)")
    parser.add_argument("--swa-top-k",       type=int,   default=3,
                        help="Top-k checkpoints for SWA (3 vs parent's 8, focus on very best)")
    parser.add_argument("--swa-temperature", type=float, default=100.0,
                        help="SWA softmax temperature (100.0 vs parent's 10.0, concentrate on top)")
    parser.add_argument("--swa-every-n-epochs", type=int, default=2,
                        help="Save periodic checkpoint every N epochs (2, same as parent)")

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
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-epoch={epoch:04d}-val_f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    # Periodic checkpoints for SWA pool (every N epochs)
    # Keeping every-2-epochs from parent (denser sampling captures more peak-quality checkpoints)
    periodic_checkpoint = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints" / "periodic"),
        filename="periodic-epoch={epoch:04d}-val_f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=-1,   # Keep all periodic checkpoints
        every_n_epochs=args.swa_every_n_epochs,
        auto_insert_metric_name=False,
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
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
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
        num_sanity_val_steps=2,
        callbacks=[
            checkpoint_callback,
            periodic_checkpoint,
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

    # ─── Post-hoc ultra-tight quality-filtered SWA ──────────────────────────
    # KEY IMPROVEMENTS vs parent:
    #   1. threshold=0.510 (vs 0.505): higher bar, only captures checkpoints in top performance tier
    #   2. top-k=3 (vs 8): very selective, only the 3 best checkpoints
    #   3. temperature=100.0 (vs 10.0): at temp=100, if scores are [0.512, 0.511, 0.510],
    #      softmax(100 * [0.512, 0.511, 0.510]) = approx [0.731, 0.210, 0.059]
    #      → top-1 gets ~73% weight, effectively best-checkpoint selection with tiny correction
    #   4. every-2-epochs collection (same as parent): dense sampling of peak performance window
    #
    # This SWA strategy is designed to:
    # a) Only activate if we achieve checkpoints above F1=0.510 (not guaranteed)
    # b) Heavily concentrate weight on the very best checkpoint (73-95% depending on spread)
    # c) Fall back gracefully to best single checkpoint if threshold not met
    do_swa = (not fast_dev_run and args.debug_max_step is None)
    swa_applied = False
    best_single_val_f1 = None

    # Track best single-checkpoint val_f1 from ModelCheckpoint
    if do_swa and checkpoint_callback.best_model_score is not None:
        best_single_val_f1 = float(checkpoint_callback.best_model_score)
        if trainer.is_global_zero:
            print(f"\nBest single checkpoint val_f1: {best_single_val_f1:.4f}")

    if do_swa:
        # All ranks participate so no broadcast is needed (checkpoints on shared filesystem).
        # This ensures all ranks have identical SWA weights before trainer.test() is called.
        periodic_dir = output_dir / "checkpoints" / "periodic"
        pool = []

        if periodic_dir.exists():
            for ckpt_path in sorted(periodic_dir.glob("*.ckpt")):
                # Handle both "val_f1=0.4487" and "val_f1=val_f1=0.4487" (PL double-prefix)
                matches = re.findall(r'val_f1=(\d+\.\d+)', ckpt_path.name)
                if matches:
                    val_f1 = float(matches[-1])  # take last match to handle double prefix
                    if val_f1 >= args.swa_threshold:
                        pool.append({"val_f1": val_f1, "path": str(ckpt_path)})

        pool.sort(key=lambda x: x["val_f1"], reverse=True)
        pool = pool[:args.swa_top_k]

        if len(pool) >= 2:
            swa_pool_max = pool[0]["val_f1"]
            swa_pool_min = pool[-1]["val_f1"]

            if trainer.is_global_zero:
                print(f"\nSWA: {len(pool)} qualifying checkpoints (threshold={args.swa_threshold}), "
                      f"val_f1 range [{swa_pool_min:.4f}, {swa_pool_max:.4f}]")

            f1_vals = torch.tensor([e["val_f1"] for e in pool])
            weights = torch.softmax(f1_vals * args.swa_temperature, dim=0)
            if trainer.is_global_zero:
                print("SWA weights:", [f"{w:.4f}" for w in weights.tolist()])
                # Show concentration: what fraction of weight is on the top checkpoint
                print(f"SWA concentration: top-1 gets {weights[0].item()*100:.1f}% of total weight")

            avg_state = None
            for i, entry in enumerate(pool):
                ckpt = torch.load(entry["path"], map_location="cpu", weights_only=False)
                sd   = ckpt.get("state_dict", ckpt)
                w    = weights[i].item()
                if avg_state is None:
                    avg_state = {k: v.float() * w for k, v in sd.items()}
                else:
                    for k in avg_state:
                        if k in sd:
                            avg_state[k] += sd[k].float() * w

            if avg_state is not None:
                if trainer.is_global_zero:
                    swa_ckpt_path = output_dir / "checkpoints" / "swa_averaged.ckpt"
                    torch.save({"state_dict": avg_state}, swa_ckpt_path)
                    print(f"SWA checkpoint saved: {swa_ckpt_path}")

                # All ranks load SWA weights for consistent test inference across GPUs
                model.load_state_dict(avg_state, strict=False)
                swa_applied = True
                if trainer.is_global_zero:
                    print("SWA weights loaded for testing")

        else:
            if trainer.is_global_zero:
                print(f"SWA pool has {len(pool)} checkpoints (< 2) below threshold={args.swa_threshold} "
                      f"— falling back to best single checkpoint")

    # ─── Testing ────────────────────────────────────────────────────────────
    if fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    elif swa_applied:
        # Already loaded SWA weights — test without loading from checkpoint
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print(f"\nTest results: {test_results}")


if __name__ == "__main__":
    main()
