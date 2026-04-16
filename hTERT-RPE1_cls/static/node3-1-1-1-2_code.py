"""
Node 3-1-1-1-2 — Frozen STRING_GNN + Fixed Inductive Conditioning + Rank-512 Bilinear Head

Architecture:
  - Frozen STRING_GNN backbone (static embeddings, single forward pass)
  - Fixed Inductive Conditioning MLP (cond_mlp):
      * Removes the critical LayerNorm-on-cond_delta bug from sibling node3-1-1-1-1
      * Uses learnable scalar gate (initialized near-zero) to scale cond_delta
      * cond_mlp: Linear(256→512) → GELU → Dropout(0.3) → Linear(512→256, near-zero init)
      * gate: learnable scalar initialized at 0.01
      * conditioned_emb = frozen_emb + gate * cond_delta
  - Deep bilinear MLP head (6 residual layers, hidden=512, rank=512)
      * Proven configuration from node1-2-3-2 (F1=0.4996) and node2-1-3 (F1=0.5047)
  - Class-weighted focal loss with tree-best class weights [2.0, 0.5, 4.0]
      * Down-regulated: 2.0×, neutral: 0.5×, up-regulated: 4.0×
      * Proven in node2-1-3 (F1=0.5047) and node1-3-1-1-1-1 (F1=0.4976)
  - Two-group AdamW: head (lr=3e-4, wd=2e-3), cond_mlp (lr=1e-3, wd=1e-2)
  - Step-based cosine LR with correctly calculated T_max
      * T_max = steps_per_epoch_actual * 150 epochs (precisely calibrated to DDP setup)
      * 50-step linear warmup
  - EarlyStopping patience=60 (longer than sibling's 50 to exploit secondary LR phase)

Key fixes vs sibling node3-1-1-1-1:
  1. LayerNorm-on-cond_delta bug removed → replaced with learnable scalar gate
  2. T_max correctly calibrated to actual steps/epoch (not assumed)
  3. Cond_mlp dropout increased from 0.1 to 0.3 (prevents rapid memorization)
  4. Cond_mlp higher weight_decay: 1e-2 (vs 1e-3 for head)
  5. Rank=512 bilinear head (proven +0.008 F1 vs rank=256 in node1-2-3)
  6. Class weights [2.0, 0.5, 4.0] (vs aggressive [10.91, 1.0, 29.62])
  7. Head LR=3e-4 (vs 5e-4) with wd=2e-3 (vs 1e-3)

Best node precedents:
  - node2-1-3: STRING_GNN partial fine-tuning + rank=512 + class=[2.0,0.5,4.0] = F1=0.5047
  - node1-2-3-2: frozen STRING_GNN + rank=512 + class=[1.5,0.8,3.0] = F1=0.4996
  - node1-2-1-1: frozen STRING_GNN + pert_matrix conditioning + rank=256 = F1=0.4900
  - node3-1-1-1-1 (sibling): inductive cond_mlp with LN-bug = F1=0.4671 (would be higher without bug)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Required for deterministic=True with einsum on CUDA >= 10.2
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import json
import math
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

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR = "/home/Models/STRING_GNN"

N_GENES_OUT  = 6640
N_CLASSES    = 3
GNN_DIM      = 256   # STRING_GNN hidden size
HEAD_DIM     = 512   # MLP hidden dimension
BILINEAR_RANK = 512  # Bilinear interaction rank (proven best in node2-1-3 and node1-2-3-2)
N_RESIDUAL_LAYERS = 6  # Proven best architecture

# Class weights for focal loss
# Strategy from node2-1-3 (F1=0.5047): [down=2.0, neutral=0.5, up=4.0]
# This is less aggressive than sibling's [10.91, 1.0, 29.62] and more focused
# on minority up-regulated class
CLASS_WEIGHTS = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """
    Compute macro-averaged per-gene F1 score matching calc_metric.py logic.

    Args:
        pred_np: [N, 3, G] softmax probabilities (float)
        labels_np: [N, G] class indices in {0, 1, 2} (already shifted from {-1, 0, 1})
    Returns:
        float: mean per-gene macro-F1 over all G genes
    """
    pred_cls = pred_np.argmax(axis=1)  # [N, G]
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal cross-entropy loss for multi-class classification.

    Focal loss down-weights well-classified examples and focuses training on
    hard examples. Useful for the 88.9% neutral class imbalance in DEG task.

    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    Uses class weights [2.0, 0.5, 4.0] — proven in node2-1-3 (F1=0.5047).
    These are much less aggressive than [10.91, 1.0, 29.62] from the parent,
    avoiding the train-loss-collapse problem while still emphasizing minority classes.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, G] unnormalized logits (C=3 classes, G=6640 genes)
            targets: [B, G] class indices in {0, 1, 2}
        Returns:
            scalar loss
        """
        # [B, C, G] → [B*G, C]
        B, C, G = logits.shape
        logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        targets_flat = targets.reshape(-1)                      # [B*G]

        # Log-softmax probabilities for focal weight computation
        log_probs = F.log_softmax(logits_flat, dim=1)           # [B*G, C]
        probs     = torch.exp(log_probs)                        # [B*G, C]

        # Gather log-prob and prob at target class
        target_log_prob = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)  # [B*G]
        target_prob     = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)       # [B*G]

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1.0 - target_prob).pow(self.gamma)   # [B*G]

        # Per-class weight
        if self.weight is not None:
            class_w = self.weight.to(logits.device)[targets_flat]   # [B*G]
        else:
            class_w = torch.ones_like(focal_weight)

        # Label smoothing: blend target log-prob with mean log-prob
        if self.label_smoothing > 0:
            smooth_loss  = -log_probs.mean(dim=1)                    # [B*G]
            ce_loss      = -target_log_prob                          # [B*G]
            loss_per_pos = (
                (1 - self.label_smoothing) * ce_loss
                + self.label_smoothing * smooth_loss
            )
        else:
            loss_per_pos = -target_log_prob                         # [B*G]

        # Apply focal weighting and class weights
        weighted_loss = focal_weight * class_w * loss_per_pos       # [B*G]

        # Normalize by sum of weights for scale consistency
        denom = class_w.sum().clamp(min=1.0)
        return (weighted_loss.sum() / denom)


# ─── Residual MLP Block ────────────────────────────────────────────────────────

class ResidualMLPBlock(nn.Module):
    """
    Residual MLP block with LayerNorm pre-activation design.
    Pattern: LN → Linear(d→d*expand) → GELU → Dropout → Linear(d*expand→d) → residual

    This is the proven architecture from node1-2 (F1=0.4912), node1-2-3-2 (F1=0.4996),
    and node2-1-3 (F1=0.5047). Six of these blocks form the deep MLP backbone.
    """

    def __init__(self, hidden_dim: int, expand: int = 4, dropout: float = 0.2):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_dim)
        self.fc1    = nn.Linear(hidden_dim, hidden_dim * expand)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(dropout)
        self.fc2    = nn.Linear(hidden_dim * expand, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


# ─── Main Model ───────────────────────────────────────────────────────────────

class InductiveBilinearModel(nn.Module):
    """
    Frozen STRING_GNN + Fixed Inductive Conditioning + Rank-512 Bilinear Head.

    Architecture:
    1. Static STRING_GNN embeddings [N, 256] (precomputed, frozen throughout training)
       → Perturbed gene's embedding extracted by pert_id → frozen_emb [B, 256]
       → OOV handled by learnable fallback embedding

    2. Inductive Conditioning MLP (FIXED vs sibling):
       → cond_mlp: Linear(256→512) → GELU → Dropout(0.3) → Linear(512→256, near-zero init)
       → cond_delta = cond_mlp(frozen_emb)  [NO LayerNorm on cond_delta — the critical fix!]
       → conditioned_emb = frozen_emb + gate * cond_delta  [gate: learnable scalar, init=0.01]
       The scalar gate starts near-zero so training begins near the node1-2 baseline.
       The gate learns to scale the conditioning strength appropriately during training.

    3. Deep Bilinear MLP Head (rank=512, 6 residual layers):
       → input_proj: LayerNorm → Linear(256→512)
       → 6x ResidualMLPBlock(512, expand=4, dropout=0.2)
       → out_proj: Linear(512→3×512) → reshape [B, 3, 512]
       → bilinear: einsum([B, 3, 512] × out_gene_emb[6640, 512]) → [B, 3, 6640]

    Key design decisions:
    - rank=512 (vs node1-2's 256): +0.008 F1 improvement proven in node1-2-3-2 and node2-1-3
    - Near-zero scalar gate (not LayerNorm): preserves node1-2 baseline at start of training
    - cond_mlp dropout=0.3 (vs sibling's 0.1): prevents rapid memorization of training perturbations
    - Learnable out_gene_emb [6640, 512] randomly initialized (NOT from STRING_GNN positions,
      which would cause semantic misalignment as noted in node1-3 analysis)
    """

    def __init__(
        self,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        head_dim: int = HEAD_DIM,
        bilinear_rank: int = BILINEAR_RANK,
        n_residual_layers: int = N_RESIDUAL_LAYERS,
        head_dropout: float = 0.2,
        cond_dropout: float = 0.3,
    ):
        super().__init__()

        # ── OOV fallback embedding ───────────────────────────────────────────
        # Learnable 256-dim embedding for genes not in STRING_GNN vocabulary
        # (~6.4% of genes based on node1-2 coverage analysis)
        self.oov_embedding = nn.Parameter(torch.zeros(GNN_DIM, dtype=torch.float32))
        nn.init.normal_(self.oov_embedding, std=0.02)

        # ── Inductive Conditioning MLP (FIXED design) ────────────────────────
        # 2-layer MLP mapping frozen PPI embedding → conditioning delta
        # Final linear initialized near-zero (std=0.01) so training starts near node1-2 baseline
        cond_fc1 = nn.Linear(GNN_DIM, GNN_DIM * 2)
        cond_fc2 = nn.Linear(GNN_DIM * 2, GNN_DIM)
        # Near-zero init for final linear: preserves frozen embedding signal early in training
        nn.init.normal_(cond_fc2.weight, std=0.01)
        nn.init.zeros_(cond_fc2.bias)

        self.cond_mlp = nn.Sequential(
            cond_fc1,
            nn.GELU(),
            nn.Dropout(cond_dropout),  # 0.3 (vs sibling's 0.1) — prevents rapid memorization
            cond_fc2,
            # CRITICAL FIX: NO LayerNorm here!
            # Sibling applied LayerNorm(256) to cond_delta which rescaled near-zero init to unit
            # variance, immediately dominating the signal. Here we leave cond_delta unscaled.
        )

        # Learnable scalar gate: starts near-zero (0.01) to ensure gradual conditioning
        # As training progresses, gate grows to scale the delta appropriately
        # This replaces the LayerNorm-on-cond_delta approach that failed in sibling
        self.cond_gate = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

        # Input normalization: normalize the frozen embeddings to consistent scale
        # Applied to frozen_emb BEFORE the cond_mlp sees it
        self.emb_norm = nn.LayerNorm(GNN_DIM)

        # ── Deep Bilinear MLP Head ────────────────────────────────────────────
        # Architecture: LN → Linear(256→512) → 6×ResidualBlock(512,expand=4) → Linear(512→3×512)
        # → reshape [B,3,512] → einsum with out_gene_emb[6640,512] → [B,3,6640]

        self.input_proj = nn.Sequential(
            nn.LayerNorm(GNN_DIM),
            nn.Linear(GNN_DIM, head_dim),
        )

        self.residual_blocks = nn.ModuleList([
            ResidualMLPBlock(head_dim, expand=4, dropout=head_dropout)
            for _ in range(n_residual_layers)
        ])

        # Output projection: maps to [B, 3*rank] → reshape [B, 3, rank]
        self.out_proj = nn.Linear(head_dim, n_classes * bilinear_rank)

        # Learnable output gene embeddings [G, rank] — random init (NOT STRING_GNN positions)
        # as semantic misalignment occurs when STRING_GNN node order ≠ DEG label positions
        self.out_gene_emb = nn.Parameter(
            torch.randn(n_genes_out, bilinear_rank) * 0.02
        )

        self.n_classes     = n_classes
        self.bilinear_rank = bilinear_rank
        self.n_genes_out   = n_genes_out

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Node3-1-1-1-2] Trainable params: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.2f}%)")

    def forward(
        self,
        gnn_emb: torch.Tensor,   # [B, 256] float32 — precomputed STRING_GNN embeddings
        in_vocab: torch.Tensor,  # [B] bool — True if gene is in STRING_GNN vocabulary
    ) -> torch.Tensor:

        batch_size = gnn_emb.shape[0]
        device = gnn_emb.device

        # ── OOV handling ────────────────────────────────────────────────────
        # Replace OOV zero-vectors with learnable fallback embedding
        oov_fill = self.oov_embedding.unsqueeze(0).expand(batch_size, -1)  # [B, 256]
        in_v = in_vocab.to(device)
        frozen_emb = torch.where(
            in_v.unsqueeze(1).expand_as(gnn_emb),
            gnn_emb,
            oov_fill,
        )  # [B, 256]

        # ── Inductive Conditioning (FIXED) ───────────────────────────────────
        # Normalize frozen embeddings before conditioning (consistent scale)
        emb_normed = self.emb_norm(frozen_emb)  # [B, 256]

        # Compute conditioning delta via MLP
        # NO LayerNorm on cond_delta (the critical fix vs sibling!)
        cond_delta = self.cond_mlp(emb_normed)   # [B, 256]

        # Scale delta by learnable scalar gate (starts ~0.01, grows during training)
        # This ensures initial conditioning has minimal impact, growing gradually
        conditioned_emb = frozen_emb + self.cond_gate * cond_delta  # [B, 256]

        # ── Deep Bilinear MLP Head ────────────────────────────────────────────
        # Input projection: [B, 256] → [B, 512]
        h = self.input_proj(conditioned_emb)

        # 6 residual MLP blocks
        for block in self.residual_blocks:
            h = block(h)

        # Output projection: [B, 512] → [B, 3×rank]
        h = self.out_proj(h)  # [B, 3*bilinear_rank]
        h = h.view(batch_size, self.n_classes, self.bilinear_rank)  # [B, 3, rank]

        # Bilinear interaction with output gene embeddings
        # einsum([B, 3, rank] × [G, rank]) → [B, 3, G]
        logits = torch.einsum("bcr,gr->bcg", h, self.out_gene_emb)  # [B, 3, G]

        return logits


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        gnn_embs: torch.Tensor,        # [N, 256] float32
        in_vocab: torch.Tensor,        # [N] bool
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long or None
    ):
        self.pert_ids  = pert_ids
        self.symbols   = symbols
        self.gnn_embs  = gnn_embs
        self.in_vocab  = in_vocab
        self.labels    = labels

    def __len__(self): return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":  self.pert_ids[idx],
            "symbol":   self.symbols[idx],
            "gnn_emb":  self.gnn_embs[idx],
            "in_vocab": self.in_vocab[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    out = {
        "pert_id":  [b["pert_id"]  for b in batch],
        "symbol":   [b["symbol"]   for b in batch],
        "gnn_emb":  torch.stack([b["gnn_emb"]  for b in batch]),
        "in_vocab": torch.stack([b["in_vocab"] for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data", micro_batch_size=8, num_workers=2):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        # Load pre-computed GNN embedding cache (pre-computed in main() before DDP init)
        cache_path = Path(__file__).parent / "run" / "gnn_emb_cache.pt"
        gnn_cache: Dict = torch.load(cache_path, weights_only=False)

        def load_split(fname, has_lbl):
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            labels = None
            if has_lbl and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)

            # Look up precomputed static GNN embeddings from cache
            gnn_embs_list = []
            in_vocab_list = []
            for pid in df["pert_id"].tolist():
                emb, in_v = gnn_cache.get(pid, (torch.zeros(GNN_DIM), False))
                gnn_embs_list.append(emb)
                in_vocab_list.append(in_v)

            gnn_embs = torch.stack(gnn_embs_list, dim=0)   # [N, 256]
            in_vocab = torch.tensor(in_vocab_list, dtype=torch.bool)

            return PerturbDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(),
                gnn_embs, in_vocab, labels
            )

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=self.num_workers,
            pin_memory=True, drop_last=shuffle,
            persistent_workers=self.num_workers > 0
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    """Gather tensors from all DDP ranks, handling variable-size padding."""
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))
    pad    = max_sz - local_p.shape[0]
    p = local_p.to(device); l = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)
    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p); dist.all_gather(gl, l)
    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class InductiveBilinearLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_head: float          = 3e-4,    # MLP head + out_gene_emb learning rate
        lr_cond: float          = 1e-3,    # Conditioning MLP + gate learning rate
        wd_head: float          = 2e-3,    # Weight decay for head params
        wd_cond: float          = 1e-2,    # Higher weight decay for cond_mlp (prevents memorization)
        focal_gamma: float      = 2.0,
        label_smoothing: float  = 0.0,
        total_steps: int        = 6600,    # Cosine LR total steps (computed in main)
        warmup_steps: int       = 50,      # Linear warmup steps (matching node1-2)
        max_epochs: int         = 200,
        head_dropout: float     = 0.2,
        cond_dropout: float     = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []
        self._test_labels:   List[torch.Tensor] = []

    def setup(self, stage=None):
        self.model = InductiveBilinearModel(
            head_dropout=self.hparams.head_dropout,
            cond_dropout=self.hparams.cond_dropout,
        )
        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=CLASS_WEIGHTS,
            label_smoothing=self.hparams.label_smoothing,
        )

    def forward(self, gnn_emb, in_vocab):
        return self.model(gnn_emb, in_vocab)

    def _loss(self, logits, labels):
        return self.focal_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        logits = self(batch["gnn_emb"], batch["in_vocab"])
        loss = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["gnn_emb"], batch["in_vocab"])
        if "label" in batch:
            loss = self._loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        lp = torch.cat(self._val_preds,  0)
        ll = torch.cat(self._val_labels, 0)
        if self.trainer.world_size > 1:
            lp, ll = _gather_tensors(lp, ll, self.device, self.trainer.world_size)
        probs_np  = torch.softmax(lp, dim=1).numpy()
        labels_np = ll.numpy()
        f1 = compute_per_gene_f1(probs_np, labels_np)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        # Log gate value for monitoring (sync_dist=True required for epoch-level DDP logging)
        gate_val = float(self.model.cond_gate.item())
        self.log("cond_gate", gate_val, prog_bar=False, sync_dist=True)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["gnn_emb"], batch["in_vocab"])
        probs  = torch.softmax(logits, dim=1)  # [B, 3, G]
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (torch.cat(self._test_labels, 0) if self._test_labels
                        else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long))

        if self.trainer.world_size > 1:
            all_probs, all_labels = _gather_tensors(
                local_probs, dummy_labels, self.device, self.trainer.world_size
            )
            all_pert = [None] * self.trainer.world_size
            all_syms = [None] * self.trainer.world_size
            dist.all_gather_object(all_pert, self._test_pert_ids)
            dist.all_gather_object(all_syms, self._test_symbols)
            all_pert = [p for sub in all_pert for p in sub]
            all_syms = [s for sub in all_syms for s in sub]
        else:
            all_probs, all_labels = local_probs, dummy_labels
            all_pert, all_syms   = self._test_pert_ids, self._test_symbols

        if self.trainer.is_global_zero:
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id (DDP padding may create duplicates)
            seen_pids: set = set()
            dedup_perts, dedup_syms, dedup_probs_list, dedup_label_rows = [], [], [], []
            for pid, sym, prob_row, lbl_row in zip(
                all_pert, all_syms, all_probs.numpy(), all_labels.numpy()
            ):
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    dedup_perts.append(pid)
                    dedup_syms.append(sym)
                    dedup_probs_list.append(prob_row)
                    dedup_label_rows.append(lbl_row)

            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pid, sym, probs in zip(dedup_perts, dedup_syms, dedup_probs_list):
                    fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")

            self.print(f"[Node3-1-1-1-2] Saved {len(dedup_perts)} test predictions → {pred_path}")

            if all_labels.any():
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[Node3-1-1-1-2] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    # ── Optimizer: two-group AdamW + step-based cosine LR ─────────────────────

    def configure_optimizers(self):
        hp = self.hparams

        # Identify conditioning MLP parameters (higher wd to prevent memorization)
        cond_ids = set(id(p) for n, p in self.model.named_parameters()
                       if "cond_mlp" in n or "cond_gate" in n or "oov_embedding" in n)

        # Head parameters: MLP body, out_gene_emb, input_proj, etc.
        head_params = [p for n, p in self.model.named_parameters()
                       if id(p) not in cond_ids]
        cond_params = [p for n, p in self.model.named_parameters()
                       if id(p) in cond_ids]

        param_groups = [
            {"params": head_params, "lr": hp.lr_head, "weight_decay": hp.wd_head},
            {"params": cond_params, "lr": hp.lr_cond, "weight_decay": hp.wd_cond},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Step-based cosine annealing with linear warmup
        # total_steps computed in main() as steps_per_epoch_actual * expected_epochs
        # This is the CRITICAL FIX vs sibling: T_max is calibrated to actual DDP training speed
        warmup_steps = hp.warmup_steps
        total_steps  = hp.total_steps
        eta_min_ratio = 1e-6 / hp.lr_head

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step + 1) / float(warmup_steps)
            # Cosine decay from 1.0 to eta_min_ratio
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            # Clamp progress to [0, 1] to prevent implicit LR restart (sibling's bug)
            progress = min(progress, 1.0)
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_val

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable parameters + buffers ─────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items()
              if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving ckpt: {trained:,}/{total:,} params ({100*trained/total:.2f}%)"
        )
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Pre-compute STRING_GNN static embeddings ─────────────────────────────────

def build_static_gnn_embeddings(
    pert_ids: List[str],
    device: torch.device,
) -> Dict[str, Tuple[torch.Tensor, bool]]:
    """
    Build static STRING_GNN embeddings for a list of unique pert_ids.

    This is a single GNN forward pass (no cond_emb injection) to extract
    fixed static embeddings. This follows the proven approach from node1-2 (F1=0.4912),
    node1-2-3-2 (F1=0.4996), and node2-1-3 (F1=0.5047).

    Perturbation-specific adaptation is handled by the inductive cond_mlp,
    NOT by the GNN forward (unlike the parent node3-1-1-1 which used cond_emb injection).

    Returns a dict: {pert_id: (emb_256d_float32_cpu, in_vocab_bool)}
    """
    model_dir   = Path(STRING_GNN_DIR)
    node_names  = json.loads((model_dir / "node_names.json").read_text())
    name_to_idx = {n: i for i, n in enumerate(node_names)}

    gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    gnn_model = gnn_model.to(device)
    gnn_model.eval()

    graph = torch.load(model_dir / "graph_data.pt", weights_only=False)
    edge_index  = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"]
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    # Single forward pass: no cond_emb, just static PPI embeddings
    # Much faster than parent's per-sample cond_emb injection
    with torch.no_grad():
        out = gnn_model(edge_index=edge_index, edge_weight=edge_weight)
        all_embs = out.last_hidden_state.cpu().float()  # [18870, 256]

    # Clean up GNN model from GPU
    del gnn_model
    torch.cuda.empty_cache()

    # Build per-pert_id lookup dict
    unique_pids = list(set(pert_ids))
    result: Dict[str, Tuple[torch.Tensor, bool]] = {}
    for pid in unique_pids:
        pid_clean = pid.split(".")[0]  # strip version suffix
        if pid_clean in name_to_idx:
            idx = name_to_idx[pid_clean]
            result[pid] = (all_embs[idx], True)
        else:
            result[pid] = (torch.zeros(GNN_DIM), False)

    return result


def _precompute_gnn_cache(args):
    """
    Pre-compute static STRING_GNN embeddings BEFORE DDP initialization.

    Uses file-based polling so non-rank-0 processes wait without DDP barrier.
    Faster than parent's cond_emb approach (1 pass vs 1 pass/unique_pert_id).
    """
    import time as _time

    rank       = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cache_path  = Path(__file__).parent / "run" / "gnn_emb_cache.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path = cache_path.with_suffix(".pt.ready")

    if not sentinel_path.exists():
        if rank == 0:
            print("[Pre-compute] Building static STRING_GNN embeddings (rank 0, one-time)...",
                  flush=True)
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            data_dir = Path(args.data_dir)
            all_pids = list({
                pid
                for fname in ["train.tsv", "val.tsv", "test.tsv"]
                for pid in pd.read_csv(data_dir / fname, sep="\t")["pert_id"].tolist()
            })
            gnn_cache = build_static_gnn_embeddings(all_pids, device)
            tmp_path = cache_path.with_suffix(".pt.tmp")
            torch.save(gnn_cache, tmp_path)
            tmp_path.rename(cache_path)
            sentinel_path.touch()
            in_vocab_count = sum(1 for (_, iv) in gnn_cache.values() if iv)
            print(f"[Pre-compute] Cached {len(gnn_cache)} embeddings "
                  f"({in_vocab_count} in-vocab) → {cache_path}", flush=True)
        else:
            print(f"[Rank {rank}] Waiting for STRING_GNN cache from rank 0...", flush=True)
            while not sentinel_path.exists():
                _time.sleep(3)
            print(f"[Rank {rank}] STRING_GNN cache ready.", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 3-1-1-1-2 — Frozen STRING_GNN + Fixed Inductive Conditioning + Rank-512 Bilinear Head"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--lr-head",           type=float, default=3e-4)
    p.add_argument("--lr-cond",           type=float, default=1e-3)
    p.add_argument("--wd-head",           type=float, default=2e-3)
    p.add_argument("--wd-cond",           type=float, default=1e-2)
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--label-smoothing",   type=float, default=0.0)
    p.add_argument("--micro-batch-size",  type=int,   default=8)
    p.add_argument("--global-batch-size", type=int,   default=64)
    p.add_argument("--max-epochs",        type=int,   default=200)
    p.add_argument("--patience",          type=int,   default=60)
    p.add_argument("--warmup-steps",      type=int,   default=50)
    p.add_argument("--total-steps",       type=int,   default=None,
                   help="Cosine LR total steps (auto-computed if not set)")
    p.add_argument("--head-dropout",      type=float, default=0.2)
    p.add_argument("--cond-dropout",      type=float, default=0.3)
    p.add_argument("--num-workers",       type=int,   default=2)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",    type=int,   default=None)
    p.add_argument("--fast-dev-run",      action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute static STRING_GNN embeddings before DDP init
    _precompute_gnn_cache(args)

    # ── Correctly calibrate total_steps for actual DDP training ───────────────
    # CRITICAL FIX vs sibling: steps_per_epoch depends on n_gpus and global_batch_size
    # train size = 1416 samples
    # With drop_last=True: floor(1416 / micro_batch) steps per GPU
    # With DDP: effective steps_per_epoch = floor(train_size / global_batch_size) approximately
    # More precisely: steps_per_epoch ≈ ceil(train_size / (micro_batch_size * n_gpus)) / accum
    # accum = global_batch_size // (micro_batch_size * n_gpus)
    # → steps_per_epoch ≈ train_size // global_batch_size
    train_size = 1416
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    # Steps per epoch = (samples_per_gpu_per_epoch) / accum
    # samples_per_gpu = floor(train_size / n_gpus) with drop_last
    samples_per_gpu = train_size // n_gpus
    steps_per_epoch = max(1, samples_per_gpu // (args.micro_batch_size * accum))

    # Target: cover ~150 epochs with the cosine schedule for the secondary improvement phase
    # The secondary LR-decay phase (proven critical in node1-2: epochs 70-98 = +0.04 F1)
    # requires the LR to decay to low values. With patience=60, training lasts ~150-200 epochs.
    expected_epochs = 150  # Conservative: training may stop earlier but LR should reach minimum by 150 epochs
    if args.total_steps is None:
        total_steps = steps_per_epoch * expected_epochs
    else:
        total_steps = args.total_steps

    print(f"[Main] n_gpus={n_gpus}, accum={accum}, steps_per_epoch={steps_per_epoch}, "
          f"total_steps={total_steps}", flush=True)

    dm = PerturbDataModule(
        args.data_dir, args.micro_batch_size, args.num_workers
    )
    lit = InductiveBilinearLitModule(
        lr_head=args.lr_head,
        lr_cond=args.lr_cond,
        wd_head=args.wd_head,
        wd_cond=args.wd_cond,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        head_dropout=args.head_dropout,
        cond_dropout=args.cond_dropout,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.patience, min_delta=1e-5
    )
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)
    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps    = -1
    limit_train  = 1.0
    limit_val    = 1.0
    limit_test   = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps   = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2
        limit_test  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=1800))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)


if __name__ == "__main__":
    main()
