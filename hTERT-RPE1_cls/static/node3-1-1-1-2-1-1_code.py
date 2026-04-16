"""
Node 3-1-1-1-2-2 — Frozen STRING_GNN + Rank-512 Bilinear Head with Enhanced Class Weighting

Architecture:
  - Fully FROZEN STRING_GNN backbone (all 8 GCN layers + post_mp precomputed once)
    * Complete frozen forward pass: emb → mps.0-7 → post_mp → static buffer
    * Eliminates backbone fine-tuning instability observed in parent node3-1-1-1-2-1
    * Parent's backbone fine-tuning caused epoch 0 val_f1=0.272 (destabilization)
    * Frozen backbone provides clean 0.38+ starting point every run
  - Deep bilinear MLP head (6 residual layers, hidden=512, rank=512)
    * Proven configuration from node2-1-3 (F1=0.5047) and node1-2-3-2 (F1=0.4996)
  - Enhanced class-weighted focal loss with label smoothing
    * Class weights [2.5, 0.5, 5.0] — stronger minority class emphasis vs [2.0, 0.5, 4.0]
    * Label smoothing = 0.05 — improves calibration on imbalanced ternary labels
    * Based on feedback from parent node3-1-1-1-2-1 (Section 8 of DATA_ABSTRACT)
  - Single-group AdamW with calibrated cosine LR schedule
    * lr_head = 3e-4 (proven optimal in node2-1-3 and node3 lineage)
    * weight_decay = 1.5e-3 (slight increase over node2-1-3's 1e-3 for mild regularization)
    * total_steps = 2200 (targeting ~100 epoch LR minimum, longer than parent's 1600)
    * warmup_steps = 100 (matching node2-1-3)
  - EarlyStopping patience=50

Key changes vs parent node3-1-1-1-2-1 (F1=0.4952):
  1. REVERT to fully frozen STRING_GNN backbone (removes partial fine-tuning instability)
  2. Simplify to single optimizer group (head only, no backbone group)
  3. Increase class weights from [2.0, 0.5, 4.0] → [2.5, 0.5, 5.0] (more minority class emphasis)
  4. Add label smoothing = 0.05 (improves calibration, helps minority class predictions)
  5. Increase total_steps from 1600 → 2200 (LR minimum at ~100 epochs vs ~72 epochs)
  6. Increase weight_decay from 1e-3 → 1.5e-3 (mild additional regularization)
  7. Precompute uses FULL STRING_GNN forward pass (not partial frozen-prefix buffer)

Performance rationale:
  - Parent's backbone fine-tuning caused -0.0051 regression from grandparent (0.5003 → 0.4952)
  - Frozen STRING_GNN: node3-1-1-1-2=0.5003, node1-2-3-2=0.4996, node2-1-2=0.5011
  - Target: F1 ≈ 0.503-0.508 by combining frozen backbone reliability with better class weighting

Best node precedents:
  - node2-1-3: partial fine-tuning + rank=512 + class=[2.0,0.5,4.0] = F1=0.5047 (tree best)
  - node3-1-1-1-2 (grandparent): frozen + rank=512 + cond_mlp (inactive) = F1=0.5003
  - node1-2-3-2: frozen + rank=512 + class=[1.5,0.8,3.0] = F1=0.4996
  - node2-1-2: frozen + rank=256 + class=[1.5,0.8,2.5] = F1=0.5011
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Required for deterministic=True with einsum on CUDA >= 10.2
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import json
import math
import argparse
import time as _time
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
# Strategy: [2.5, 0.5, 5.0] — stronger than parent's [2.0, 0.5, 4.0]
# Down-regulated: 2.5x, neutral: 0.5x, up-regulated: 5.0x
# Rationale: up-regulated class (3% of labels) needs more emphasis than [2.0,0.5,4.0] provides
# Based on feedback from parent node3-1-1-1-2-1 suggesting [2.5,0.5,5.0] or [3.0,0.5,6.0]
CLASS_WEIGHTS = torch.tensor([2.5, 0.5, 5.0], dtype=torch.float32)

# Label smoothing: 0.05 — light smoothing to improve calibration on minority classes
# Helps avoid overconfident predictions on the heavily imbalanced 88.9% neutral class
LABEL_SMOOTHING = 0.05


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
    Focal cross-entropy loss for multi-class classification with label smoothing.

    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    Uses class weights [2.5, 0.5, 5.0] — enhanced minority class emphasis over
    the tree best node2-1-3's [2.0, 0.5, 4.0], targeting better recall on
    up-regulated genes (3% of labels) which contribute significantly to per-gene macro-F1.

    Label smoothing = 0.05 improves calibration on the heavily imbalanced ternary labels.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
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
        B, C, G = logits.shape
        logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        targets_flat = targets.reshape(-1)                      # [B*G]

        log_probs = F.log_softmax(logits_flat, dim=1)           # [B*G, C]
        probs     = torch.exp(log_probs)                        # [B*G, C]

        target_log_prob = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)  # [B*G]
        target_prob     = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)       # [B*G]

        focal_weight = (1.0 - target_prob).pow(self.gamma)   # [B*G]

        if self.weight is not None:
            class_w = self.weight.to(logits.device)[targets_flat]   # [B*G]
        else:
            class_w = torch.ones_like(focal_weight)

        if self.label_smoothing > 0:
            smooth_loss  = -log_probs.mean(dim=1)
            ce_loss      = -target_log_prob
            loss_per_pos = (
                (1 - self.label_smoothing) * ce_loss
                + self.label_smoothing * smooth_loss
            )
        else:
            loss_per_pos = -target_log_prob

        weighted_loss = focal_weight * class_w * loss_per_pos       # [B*G]

        denom = class_w.sum().clamp(min=1.0)
        return (weighted_loss.sum() / denom)


# ─── Residual MLP Block ────────────────────────────────────────────────────────

class ResidualMLPBlock(nn.Module):
    """
    Residual MLP block with LayerNorm pre-activation design.
    Pattern: LN → Linear(d→d*expand) → GELU → Dropout → Linear(d*expand→d) → residual

    Proven architecture from node1-2 (F1=0.4912), node1-2-3-2 (F1=0.4996),
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

class FrozenGNNBilinearModel(nn.Module):
    """
    Fully Frozen STRING_GNN + Rank-512 Deep Bilinear MLP Head.

    KEY DIFFERENCE from parent node3-1-1-1-2-1:
    - STRING_GNN is FULLY FROZEN (all 8 layers + post_mp precomputed once)
    - No backbone fine-tuning (eliminates early-epoch instability)
    - Simpler, more reliable architecture that consistently achieves F1 ≈ 0.50-0.503

    Architecture:
    1. STRING_GNN (fully frozen, precomputed):
       - Full forward pass: emb → mps.0-7 → post_mp → node_emb [18870, 256]
       - Stored as a non-parameter buffer (computed once before training)
       - Per-sample embedding lookup: node_emb[node_idx] → [B, 256]

    2. Deep Bilinear MLP Head (rank=512, 6 residual layers):
       → input_proj: LayerNorm → Linear(256→512)
       → 6x ResidualMLPBlock(512, expand=4, dropout=0.2)
       → out_proj: Linear(512→3×512) → reshape [B, 3, 512]
       → bilinear: einsum([B, 3, 512] × out_gene_emb[6640, 512]) → [B, 3, 6640]
    """

    def __init__(
        self,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        head_dim: int = HEAD_DIM,
        bilinear_rank: int = BILINEAR_RANK,
        n_residual_layers: int = N_RESIDUAL_LAYERS,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        # ── Static frozen embeddings from full STRING_GNN forward pass ────────
        # Precomputed once via build_gnn_full_embeddings() before training.
        # Shape: [N_nodes, 256] (18870 nodes)
        # This is the FULL STRING_GNN output (emb + mps.0-7 + post_mp).
        self.register_buffer("gnn_embs", None)  # [N_nodes, 256] or None

        # ── OOV fallback embedding ───────────────────────────────────────────
        self.oov_embedding = nn.Parameter(torch.zeros(GNN_DIM, dtype=torch.float32))
        nn.init.normal_(self.oov_embedding, std=0.02)

        # ── Deep Bilinear MLP Head ────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.LayerNorm(GNN_DIM),
            nn.Linear(GNN_DIM, head_dim),
        )

        self.residual_blocks = nn.ModuleList([
            ResidualMLPBlock(head_dim, expand=4, dropout=head_dropout)
            for _ in range(n_residual_layers)
        ])

        self.out_proj = nn.Linear(head_dim, n_classes * bilinear_rank)

        # Learnable output gene embeddings [G, rank] — random init
        self.out_gene_emb = nn.Parameter(
            torch.randn(n_genes_out, bilinear_rank) * 0.02
        )

        self.n_classes     = n_classes
        self.bilinear_rank = bilinear_rank
        self.n_genes_out   = n_genes_out

    def forward(
        self,
        node_idx: torch.Tensor,   # [B] long — GNN node indices for each sample
        in_vocab: torch.Tensor,   # [B] bool — True if gene is in STRING_GNN vocabulary
    ) -> torch.Tensor:

        batch_size = node_idx.shape[0]
        device = node_idx.device
        in_v = in_vocab.to(device)

        # ── Frozen embedding lookup ──────────────────────────────────────────
        # gnn_embs is the precomputed full STRING_GNN output [N_nodes, 256]
        safe_idx = node_idx.clamp(0, self.gnn_embs.shape[0] - 1)
        emb_from_gnn = self.gnn_embs[safe_idx]  # [B, 256]

        # Replace OOV positions with learnable fallback
        oov_fill = self.oov_embedding.unsqueeze(0).expand(batch_size, -1)  # [B, 256]
        sample_emb = torch.where(
            in_v.unsqueeze(1).expand_as(emb_from_gnn),
            emb_from_gnn,
            oov_fill,
        )  # [B, 256]

        # ── Deep Bilinear MLP Head ────────────────────────────────────────────
        h = self.input_proj(sample_emb)   # [B, 512]

        for block in self.residual_blocks:
            h = block(h)

        h = self.out_proj(h)  # [B, 3*bilinear_rank]
        h = h.view(batch_size, self.n_classes, self.bilinear_rank)  # [B, 3, rank]

        logits = torch.einsum("bcr,gr->bcg", h, self.out_gene_emb)  # [B, 3, G]

        return logits


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        node_indices: torch.Tensor,    # [N] long — GNN node indices (-1 if OOV)
        in_vocab: torch.Tensor,        # [N] bool
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long or None
    ):
        self.pert_ids     = pert_ids
        self.symbols      = symbols
        self.node_indices = node_indices
        self.in_vocab     = in_vocab
        self.labels       = labels

    def __len__(self): return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
            "node_idx":   self.node_indices[idx],
            "in_vocab":   self.in_vocab[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    out = {
        "pert_id":  [b["pert_id"]  for b in batch],
        "symbol":   [b["symbol"]   for b in batch],
        "node_idx": torch.stack([b["node_idx"]  for b in batch]),
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
        # node_name_to_idx: built from node_names.json, set externally before setup
        self.node_name_to_idx: Optional[Dict[str, int]] = None

    def setup(self, stage=None):
        assert self.node_name_to_idx is not None, \
            "node_name_to_idx must be set before DataModule.setup()"

        def load_split(fname, has_lbl):
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            labels = None
            if has_lbl and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)

            node_indices_list = []
            in_vocab_list     = []
            for pid in df["pert_id"].tolist():
                pid_clean = pid.split(".")[0]
                if pid_clean in self.node_name_to_idx:
                    node_indices_list.append(self.node_name_to_idx[pid_clean])
                    in_vocab_list.append(True)
                else:
                    node_indices_list.append(0)  # placeholder, replaced by OOV
                    in_vocab_list.append(False)

            node_indices = torch.tensor(node_indices_list, dtype=torch.long)
            in_vocab     = torch.tensor(in_vocab_list, dtype=torch.bool)

            return PerturbDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(),
                node_indices, in_vocab, labels
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

class FrozenGNNLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_head: float          = 3e-4,    # MLP head + out_gene_emb learning rate
        wd_head: float          = 1.5e-3,  # Weight decay (slightly increased for regularization)
        focal_gamma: float      = 2.0,
        label_smoothing: float  = 0.05,    # Light smoothing for minority class calibration
        total_steps: int        = 2200,    # Cosine LR total steps (targeting ~100 epoch window)
        warmup_steps: int       = 100,     # Linear warmup steps (matching node2-1-3)
        max_epochs: int         = 200,
        head_dropout: float     = 0.2,
        # Reference to precomputed GNN embeddings path — set externally before training
        gnn_embs_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["gnn_embs_path"])
        # Store path for lazy loading in setup
        self._gnn_embs_path = gnn_embs_path
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []
        self._test_labels:   List[torch.Tensor] = []

    def setup(self, stage=None):
        # Load precomputed frozen STRING_GNN embeddings
        assert self._gnn_embs_path, "gnn_embs_path must be set"

        gnn_embs = torch.load(
            self._gnn_embs_path, weights_only=False
        ).float()  # [N_nodes, 256]

        # Build the main model
        self.model = FrozenGNNBilinearModel(
            head_dropout=self.hparams.head_dropout,
        )
        # Register the frozen GNN embeddings as a proper buffer
        # (register_buffer handles DDP broadcasting and device movement correctly)
        self.model.register_buffer("gnn_embs", gnn_embs)

        # Cast trainable params to float32 for stable optimization
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=CLASS_WEIGHTS,
            label_smoothing=self.hparams.label_smoothing,
        )

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.model.parameters())
        self.print(f"[Node3-1-1-1-2-2] Trainable params: {n_trainable:,} / {n_total:,} "
                   f"({100*n_trainable/n_total:.2f}%)")

    def forward(self, node_idx, in_vocab):
        return self.model(node_idx, in_vocab)

    def _loss(self, logits, labels):
        return self.focal_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_idx"], batch["in_vocab"])
        loss = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_idx"], batch["in_vocab"])
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
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_idx"], batch["in_vocab"])
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

            # Deduplicate by pert_id
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

            self.print(f"[Node3-1-1-1-2-2] Saved {len(dedup_perts)} test predictions → {pred_path}")

            if all_labels.any():
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[Node3-1-1-1-2-2] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    # ── Optimizer: single-group AdamW + step-based cosine LR ─────────────────

    def configure_optimizers(self):
        hp = self.hparams

        # Single optimizer group — all trainable head parameters
        # Frozen GNN backbone: no optimizer group needed
        all_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            all_params,
            lr=hp.lr_head,
            weight_decay=hp.wd_head,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Calibrated cosine annealing with linear warmup
        # total_steps=2200 targets ~100-epoch window with 22 steps/epoch
        # LR reaches minimum by ~100 epochs, well within the max_epochs=200 window
        # This is longer than parent's 1600 (72 epochs) to allow more secondary improvement
        # Reference: parent achieved best at epoch 26 out of ~72 epoch window
        # With 2200 steps, LR is still at ~72% of peak at epoch 26, enabling late-phase refinement
        warmup_steps  = hp.warmup_steps
        total_steps   = hp.total_steps
        eta_min_ratio = 1e-6 / hp.lr_head

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            # Clamp to [0, 1] to prevent implicit LR restart after T_max
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


# ─── Pre-compute full STRING_GNN embeddings ───────────────────────────────────

def build_gnn_full_embeddings(
    device: torch.device,
    out_dir: Path,
) -> None:
    """
    Pre-compute the full STRING_GNN forward pass output:
    emb → mps.0-7 → post_mp → gnn_embs [N_nodes, 256]

    This is the FULL frozen output (all 8 GCN layers + output projection),
    unlike the partial frozen prefix used in the parent node3-1-1-1-2-1.

    The full precompute is used because:
    1. Frozen backbone avoids early-epoch instability from backbone fine-tuning
    2. Full pass captures richer PPI topology than frozen prefix alone
    3. Static computation can be done once, saving ~5.43M params of GPU memory
    """
    model_dir = Path(STRING_GNN_DIR)

    embs_path      = out_dir / "gnn_full_embs.pt"
    sentinel_path  = out_dir / "gnn_full_precompute.ready"

    if not sentinel_path.exists():
        print("[Pre-compute] Building full STRING_GNN embeddings...", flush=True)

        gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
        gnn_model = gnn_model.to(device)
        gnn_model.eval()

        graph = torch.load(model_dir / "graph_data.pt", weights_only=False)
        edge_index  = graph["edge_index"].to(device)
        edge_weight = graph["edge_weight"]
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        # Run FULL forward pass: emb → mps.0-7 → post_mp
        with torch.no_grad():
            outputs = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
                output_hidden_states=False,
            )
            gnn_embs = outputs.last_hidden_state.cpu().float()  # [N_nodes, 256]

        # Save full frozen embeddings
        torch.save(gnn_embs, embs_path)
        print(f"[Pre-compute] Saved full GNN embeddings: {gnn_embs.shape}", flush=True)

        # Clean up GPU memory
        del gnn_model
        torch.cuda.empty_cache()

        sentinel_path.touch()
        print(f"[Pre-compute] Complete. Saved to {out_dir}", flush=True)
    else:
        print(f"[Pre-compute] Found existing full GNN embeddings cache at {out_dir}", flush=True)


def _precompute_with_rank_guard(out_dir: Path):
    """
    Pre-compute full STRING_GNN embeddings before DDP initialization.
    Uses file-based polling so non-rank-0 processes wait.
    """
    rank       = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    sentinel_path = out_dir / "gnn_full_precompute.ready"

    if not sentinel_path.exists():
        if rank == 0:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            out_dir.mkdir(parents=True, exist_ok=True)
            build_gnn_full_embeddings(device, out_dir)
        else:
            print(f"[Rank {rank}] Waiting for full STRING_GNN precompute from rank 0...", flush=True)
            while not sentinel_path.exists():
                _time.sleep(3)
            print(f"[Rank {rank}] Full STRING_GNN precompute ready.", flush=True)
    else:
        print(f"[Rank {rank}] Found existing full STRING_GNN precompute cache.", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 3-1-1-1-2-2 — Frozen STRING_GNN + Rank-512 Bilinear Head with Enhanced Class Weighting"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--lr-head",           type=float, default=3e-4)
    p.add_argument("--wd-head",           type=float, default=1.5e-3)
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--label-smoothing",   type=float, default=0.05)
    p.add_argument("--micro-batch-size",  type=int,   default=8)
    p.add_argument("--global-batch-size", type=int,   default=64)
    p.add_argument("--max-epochs",        type=int,   default=200)
    p.add_argument("--patience",          type=int,   default=50)
    p.add_argument("--warmup-steps",      type=int,   default=100)
    p.add_argument("--total-steps",       type=int,   default=None,
                   help="Cosine LR total steps (auto-computed if not set)")
    p.add_argument("--head-dropout",      type=float, default=0.2)
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

    # Pre-compute full STRING_GNN embeddings before DDP init
    _precompute_with_rank_guard(out_dir)

    # Load precomputed embeddings path and node mapping
    model_dir     = Path(STRING_GNN_DIR)
    node_names    = json.loads((model_dir / "node_names.json").read_text())
    node_name_to_idx = {n: i for i, n in enumerate(node_names)}

    gnn_embs_path = str(out_dir / "gnn_full_embs.pt")

    # ── Calibrate total_steps for actual DDP training ─────────────────────────
    # Targeting ~100 epoch window: LR reaches minimum by epoch 100
    # With ~22 steps/epoch, total_steps = 22 × 100 = 2200
    # This is longer than parent's 1600 (72 epochs), allowing secondary improvement
    # Cosine schedule design: still at ~72% of peak LR at epoch 26 (parent's best epoch)
    train_size = 1416
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    samples_per_gpu = train_size // n_gpus
    steps_per_epoch = max(1, samples_per_gpu // (args.micro_batch_size * accum))

    # Targeting ~100 epoch window
    expected_epochs = 100
    if args.total_steps is None:
        total_steps = steps_per_epoch * expected_epochs
        # Ensure at least 1000 steps for meaningful cosine decay
        total_steps = max(1000, total_steps)
    else:
        total_steps = args.total_steps

    print(f"[Main] n_gpus={n_gpus}, accum={accum}, steps_per_epoch={steps_per_epoch}, "
          f"total_steps={total_steps}", flush=True)

    dm = PerturbDataModule(
        args.data_dir, args.micro_batch_size, args.num_workers
    )
    dm.node_name_to_idx = node_name_to_idx

    lit = FrozenGNNLitModule(
        lr_head=args.lr_head,
        wd_head=args.wd_head,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        head_dropout=args.head_dropout,
        gnn_embs_path=gnn_embs_path,
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
