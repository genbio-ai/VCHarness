"""
Node 3-1-1-1-2-1 — Partial STRING_GNN Fine-tuning + Rank-512 Bilinear Head (No Conditioning)

Architecture:
  - Partial STRING_GNN backbone fine-tuning (mps.6 + mps.7 + post_mp layers, ~198K params)
    * Frozen early layers (mps.0-5 + emb): precomputed as a static buffer
    * Trainable final 2 GCN layers + output projection: task-adaptive PPI embedding update
    * Following the node2-1-3 blueprint (tree best, F1=0.5047)
  - NO inductive conditioning MLP: confirmed zero contribution in parent node3-1-1-1-2
    * The cond_gate stayed near zero (range -0.007 to +0.012) throughout all 84 epochs
    * Removing it reduces parameters and overfitting pressure
  - Deep bilinear MLP head (6 residual layers, hidden=512, rank=512)
    * Proven configuration from node1-2-3-2 (F1=0.4996) and node2-1-3 (F1=0.5047)
  - Class-weighted focal loss with tree-best class weights [2.0, 0.5, 4.0]
    * Down-regulated: 2.0x, neutral: 0.5x, up-regulated: 4.0x
    * Proven in node2-1-3 (F1=0.5047)
  - Three-group AdamW: backbone (lr=5e-5, wd=1e-3), head (lr=3e-4, wd=1e-3)
  - Calibrated cosine LR schedule: total_steps=1600 (targeting ~72 epoch window)
    * Aligns LR minimum with realistic training duration before early stopping
    * 100-step linear warmup (matching node2-1-3)
  - EarlyStopping patience=50

Key changes vs parent node3-1-1-1-2:
  1. Remove cond_mlp entirely (confirmed zero contribution in parent, gate stayed ~0)
  2. Add partial STRING_GNN backbone fine-tuning (mps.6+mps.7+post_mp, backbone lr=5e-5)
  3. Reduce total_steps from 3300 to 1600 (align LR schedule with actual training window)
  4. Reduce patience from 60 to 50 (tree best node2-1-3 used patience=50)
  5. Use backbone_lr=5e-5 (matching tree best node2-1-3: mps.6+mps.7+post_mp trainable)
  6. Three-group AdamW instead of two-group

Best node precedents:
  - node2-1-3: partial STRING_GNN backbone + rank=512 + class=[2.0,0.5,4.0] = F1=0.5047 (tree best)
  - node2-1-1-2: partial STRING_GNN backbone + rank=512 = F1=0.5000
  - node1-2-3-2: frozen STRING_GNN + rank=512 + class=[1.5,0.8,3.0] = F1=0.4996
  - node3-1-1-1-2 (parent): frozen STRING_GNN + cond_mlp (inactive) = F1=0.5003
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
# Strategy from node2-1-3 (F1=0.5047): [down=2.0, neutral=0.5, up=4.0]
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

    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    Uses class weights [2.0, 0.5, 4.0] — proven in node2-1-3 (F1=0.5047).
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

class PartialFinetuneModel(nn.Module):
    """
    Partial STRING_GNN Backbone Fine-tuning + Rank-512 Bilinear Head.
    NO conditioning MLP (confirmed zero contribution in parent node3-1-1-1-2).

    Architecture:
    1. STRING_GNN with partial fine-tuning:
       - Frozen: emb + mps.0-5 (precomputed as static buffer)
       - Trainable: mps.6, mps.7, post_mp (~198K backbone params)
       - This is the node2-1-3 blueprint (tree best, F1=0.5047)
       Per-sample forward through the fine-tunable portion only.

    2. No conditioning (removed — gate stayed near-zero in parent):
       - Parent's cond_mlp contributed exactly zero signal throughout 84 epochs
       - Removing it reduces parameters and overfitting

    3. Deep Bilinear MLP Head (rank=512, 6 residual layers):
       → input_proj: LayerNorm → Linear(256→512)
       → 6x ResidualMLPBlock(512, expand=4, dropout=0.2)
       → out_proj: Linear(512→3×512) → reshape [B, 3, 512]
       → bilinear: einsum([B, 3, 512] × out_gene_emb[6640, 512]) → [B, 3, 6640]

    Key design decisions:
    - Partial backbone fine-tuning: proven +0.005 F1 in node2-1-3 over frozen STRING_GNN
    - rank=512: +0.008 F1 vs rank=256 (proven in node1-2-3-2 and node2-1-3)
    - No conditioning: zero contribution confirmed in parent, simplifies optimization
    - Learnable out_gene_emb [6640, 512]: randomly initialized (not STRING_GNN positions)
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

        # ── Static pre-computed embeddings from frozen GNN portion ───────────
        # These are precomputed embeddings from mps.0-5 (frozen layers).
        # We store them as a non-parameter buffer and run only the trainable
        # layers (mps.6, mps.7, post_mp) during forward pass.
        # This is initialized via setup() in the LightningModule.
        self.register_buffer("frozen_embs", None)  # [N_nodes, 256] or None

        # ── OOV fallback embedding ───────────────────────────────────────────
        self.oov_embedding = nn.Parameter(torch.zeros(GNN_DIM, dtype=torch.float32))
        nn.init.normal_(self.oov_embedding, std=0.02)

        # ── Backbone trainable layers (mps.6 + mps.7 + post_mp) ─────────────
        # These are loaded from STRING_GNN but remain trainable during training
        # Initialized in setup() after loading the full model.
        # We keep them as a sub-module for optimizer grouping.
        self.backbone_trainable = None  # initialized in setup

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

        # ── Retrieve partially-updated embeddings from trainable backbone ────
        # backbone_trainable produces [N_nodes, 256] embeddings
        if self.backbone_trainable is not None:
            all_embs = self.backbone_trainable()  # [N_nodes, 256]
        else:
            # Fallback to frozen pre-computed (shouldn't happen after setup)
            all_embs = self.frozen_embs

        # ── Per-sample embedding lookup ──────────────────────────────────────
        # OOV genes (not in STRING_GNN vocab) get learnable fallback
        in_v = in_vocab.to(device)

        # Gather embeddings for in-vocab samples
        safe_idx = node_idx.clamp(0, all_embs.shape[0] - 1)
        emb_from_gnn = all_embs[safe_idx]  # [B, 256]

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


class TrainableBackbone(nn.Module):
    """
    Wrapper for the trainable portion of STRING_GNN (mps.6 + mps.7 + post_mp).

    During forward, it takes the pre-computed frozen embeddings (output of mps.0-5)
    and runs the final trainable GCN layers + output projection to produce the
    updated node embeddings.

    This is the partial fine-tuning approach from node2-1-3 (F1=0.5047, tree best).
    """

    def __init__(
        self,
        gnn_model: nn.Module,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        frozen_mid_embs: torch.Tensor,  # [N_nodes, 256] - output of mps.0-5
    ):
        super().__init__()
        # Trainable GNN layers
        self.mps_6 = gnn_model.mps[6]
        self.mps_7 = gnn_model.mps[7]
        self.post_mp = gnn_model.post_mp

        # Frozen graph structure
        self.register_buffer("edge_index", edge_index)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        # Frozen intermediate embeddings (output of frozen mps.0-5)
        self.register_buffer("frozen_mid_embs", frozen_mid_embs)

    def forward(self) -> torch.Tensor:
        """Run trainable GCN layers on frozen intermediate embeddings."""
        h = self.frozen_mid_embs  # [N_nodes, 256]
        h = self.mps_6(h, self.edge_index, self.edge_weight)
        h = self.mps_7(h, self.edge_index, self.edge_weight)
        h = self.post_mp(h)
        return h  # [N_nodes, 256]


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
                    node_indices_list.append(0)  # placeholder, will be replaced by OOV
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

class PartialFinetuneLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_head: float          = 3e-4,    # MLP head + out_gene_emb learning rate
        lr_backbone: float      = 5e-5,    # Trainable STRING_GNN backbone layers LR
        wd_head: float          = 1e-3,    # Weight decay for head params (node2-1-3 proven)
        wd_backbone: float      = 1e-3,    # Weight decay for backbone trainable params
        focal_gamma: float      = 2.0,
        label_smoothing: float  = 0.0,
        total_steps: int        = 1600,    # Cosine LR total steps (calibrated to ~72 epochs)
        warmup_steps: int       = 100,     # Linear warmup steps (matching node2-1-3)
        max_epochs: int         = 200,
        head_dropout: float     = 0.2,
        # References to shared GNN data — set externally before training
        node_name_to_idx: Optional[Dict[str, int]] = None,
        frozen_mid_embs_path: Optional[str] = None,  # path to precomputed frozen embeddings
        gnn_edge_index_path: Optional[str] = None,
        gnn_edge_weight_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "node_name_to_idx", "frozen_mid_embs_path",
            "gnn_edge_index_path", "gnn_edge_weight_path"
        ])
        # Store paths for lazy loading in setup
        self._frozen_mid_embs_path = frozen_mid_embs_path
        self._gnn_edge_index_path = gnn_edge_index_path
        self._gnn_edge_weight_path = gnn_edge_weight_path
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []
        self._test_labels:   List[torch.Tensor] = []

    def setup(self, stage=None):
        # Load precomputed data
        assert self._frozen_mid_embs_path, "frozen_mid_embs_path must be set"
        assert self._gnn_edge_index_path, "gnn_edge_index_path must be set"

        frozen_mid_embs = torch.load(
            self._frozen_mid_embs_path, weights_only=False
        ).float()  # [N_nodes, 256]

        edge_index = torch.load(
            self._gnn_edge_index_path, weights_only=False
        )
        edge_weight = None
        if self._gnn_edge_weight_path and Path(self._gnn_edge_weight_path).exists():
            edge_weight = torch.load(
                self._gnn_edge_weight_path, weights_only=False
            )

        # Reconstruct trainable backbone using a fresh GNN model instance
        # The pretrained weights initialize the trainable layers (mps.6, mps.7, post_mp)
        gnn_model = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        trainable_backbone = TrainableBackbone(
            gnn_model=gnn_model,
            edge_index=edge_index,
            edge_weight=edge_weight,
            frozen_mid_embs=frozen_mid_embs,
        )
        del gnn_model

        # Build the main model
        self.model = PartialFinetuneModel(
            head_dropout=self.hparams.head_dropout,
        )
        self.model.backbone_trainable = trainable_backbone

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
        self.print(f"[Node3-1-1-1-2-1] Trainable params: {n_trainable:,} / {n_total:,} "
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

            self.print(f"[Node3-1-1-1-2-1] Saved {len(dedup_perts)} test predictions → {pred_path}")

            if all_labels.any():
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[Node3-1-1-1-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    # ── Optimizer: three-group AdamW + step-based cosine LR ──────────────────

    def configure_optimizers(self):
        hp = self.hparams

        # Identify backbone trainable parameters
        backbone_ids = set(
            id(p) for n, p in self.model.backbone_trainable.named_parameters()
            if p.requires_grad
        )
        # All other params go to head group
        head_params     = [p for p in self.model.parameters()
                           if p.requires_grad and id(p) not in backbone_ids]
        backbone_params = [p for p in self.model.backbone_trainable.parameters()
                           if p.requires_grad]

        param_groups = [
            {"params": head_params,     "lr": hp.lr_head,     "weight_decay": hp.wd_head},
            {"params": backbone_params, "lr": hp.lr_backbone, "weight_decay": hp.wd_backbone},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Calibrated cosine annealing with linear warmup
        # total_steps=1600 targets ~72-epoch window with 22 steps/epoch
        # This allows LR to reach its minimum well before early stopping (~72 epochs)
        # contrasting with parent's total_steps=3300 (never completed before early stop at epoch 83)
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


# ─── Pre-compute STRING_GNN intermediate embeddings ───────────────────────────

def build_gnn_precompute_data(
    device: torch.device,
    out_dir: Path,
) -> None:
    """
    Pre-compute:
    1. frozen_mid_embs: output of frozen layers mps.0-5, shape [N_nodes, 256]
    2. edge_index and edge_weight for trainable backbone forward pass

    These are saved to out_dir for use during training.
    """
    model_dir = Path(STRING_GNN_DIR)

    frozen_mid_path  = out_dir / "gnn_frozen_mid_embs.pt"
    edge_index_path  = out_dir / "gnn_edge_index.pt"
    edge_weight_path = out_dir / "gnn_edge_weight.pt"
    sentinel_path    = out_dir / "gnn_precompute.ready"

    if not sentinel_path.exists():
        print("[Pre-compute] Building STRING_GNN intermediate embeddings...", flush=True)

        gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
        gnn_model = gnn_model.to(device)
        gnn_model.eval()

        graph = torch.load(model_dir / "graph_data.pt", weights_only=False)
        edge_index  = graph["edge_index"].to(device)
        edge_weight = graph["edge_weight"]
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        # Run frozen prefix: emb + mps.0-5
        with torch.no_grad():
            # Start from embedding table
            h = gnn_model.emb.weight  # [N_nodes, 256]

            # Run mps.0 through mps.5 (frozen layers)
            for layer_idx in range(6):
                h = gnn_model.mps[layer_idx](h, edge_index, edge_weight)

            frozen_mid_embs = h.cpu().float()  # [N_nodes, 256]

        # Save frozen intermediate embeddings
        torch.save(frozen_mid_embs, frozen_mid_path)
        print(f"[Pre-compute] Saved frozen mid embeddings: {frozen_mid_embs.shape}", flush=True)

        # Save graph topology for trainable forward pass
        torch.save(edge_index.cpu(), edge_index_path)
        if edge_weight is not None:
            torch.save(edge_weight.cpu(), edge_weight_path)

        # Clean up GPU memory
        del gnn_model
        torch.cuda.empty_cache()

        sentinel_path.touch()
        print(f"[Pre-compute] Complete. Saved to {out_dir}", flush=True)
    else:
        print(f"[Pre-compute] Found existing cache at {out_dir}", flush=True)


def _precompute_with_rank_guard(args, out_dir: Path):
    """
    Pre-compute STRING_GNN data before DDP initialization.
    Uses file-based polling so non-rank-0 processes wait.
    """
    rank       = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    sentinel_path = out_dir / "gnn_precompute.ready"

    if not sentinel_path.exists():
        if rank == 0:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            out_dir.mkdir(parents=True, exist_ok=True)
            build_gnn_precompute_data(device, out_dir)
        else:
            print(f"[Rank {rank}] Waiting for STRING_GNN precompute from rank 0...", flush=True)
            while not sentinel_path.exists():
                _time.sleep(3)
            print(f"[Rank {rank}] STRING_GNN precompute ready.", flush=True)
    else:
        print(f"[Rank {rank}] Found existing STRING_GNN precompute cache.", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 3-1-1-1-2-1 — Partial STRING_GNN Fine-tuning + Rank-512 Bilinear Head"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--lr-head",           type=float, default=3e-4)
    p.add_argument("--lr-backbone",       type=float, default=5e-5)
    p.add_argument("--wd-head",           type=float, default=1e-3)
    p.add_argument("--wd-backbone",       type=float, default=1e-3)
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--label-smoothing",   type=float, default=0.0)
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

    # Pre-compute STRING_GNN data before DDP init
    _precompute_with_rank_guard(args, out_dir)

    # Load precomputed cache paths and node mapping
    model_dir     = Path(STRING_GNN_DIR)
    node_names    = json.loads((model_dir / "node_names.json").read_text())
    node_name_to_idx = {n: i for i, n in enumerate(node_names)}

    frozen_mid_path = str(out_dir / "gnn_frozen_mid_embs.pt")
    edge_index_path = str(out_dir / "gnn_edge_index.pt")
    ew_path         = str(out_dir / "gnn_edge_weight.pt")
    ew_path_str     = ew_path if Path(ew_path).exists() else None

    # ── Calibrate total_steps for actual DDP training ─────────────────────────
    # node2-1-3 best epoch was ~32 epochs; node2-1-1-2 best was ~64 epochs
    # With total_steps=1600 and ~22 steps/epoch → LR minimum at ~72 epochs
    # This ensures the model can exploit the secondary improvement phase
    # and LR reaches its minimum before early stopping (patience=50)
    train_size = 1416
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    samples_per_gpu = train_size // n_gpus
    steps_per_epoch = max(1, samples_per_gpu // (args.micro_batch_size * accum))

    # Targeting ~72 epoch window: LR reaches minimum by epoch 72
    expected_epochs = 72
    if args.total_steps is None:
        total_steps = steps_per_epoch * expected_epochs
        # Ensure at least 800 steps for meaningful cosine decay
        total_steps = max(800, total_steps)
    else:
        total_steps = args.total_steps

    print(f"[Main] n_gpus={n_gpus}, accum={accum}, steps_per_epoch={steps_per_epoch}, "
          f"total_steps={total_steps}", flush=True)

    dm = PerturbDataModule(
        args.data_dir, args.micro_batch_size, args.num_workers
    )
    dm.node_name_to_idx = node_name_to_idx

    lit = PartialFinetuneLitModule(
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        wd_head=args.wd_head,
        wd_backbone=args.wd_backbone,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        head_dropout=args.head_dropout,
        frozen_mid_embs_path=frozen_mid_path,
        gnn_edge_index_path=edge_index_path,
        gnn_edge_weight_path=ew_path_str,
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
