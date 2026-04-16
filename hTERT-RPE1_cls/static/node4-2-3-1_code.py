"""
Node 4-2-3-1: STRING_GNN Partial FT + Intermediate Capacity Head + Cosine Warm Restarts
             + Sample Activity Weighting + Dedicated out_gene_emb LR/WD

Architecture Strategy:
  - STRING_GNN backbone with partial fine-tuning:
      * Frozen layers (mps.0-6) + embedding: pre-computed as a buffer
      * Trainable layers: mps.7 + post_mp (~67K params, backbone_lr=1e-5)
  - Intermediate-capacity 5-layer Residual Bilinear MLP head (rank=320, hidden=320, dropout=0.4):
      ResidualBlock × 5 → bilinear output (rank-320 decomposition)
      ~7.3M total trainable parameters
      - Balance between parent's 4.2M (too low) and grandparent's 17M (too high)
      - Target 5,200:1 parameter-to-sample ratio (vs 2,966:1 too low, 12,000:1 too high)
  - Loss: Focal cross-entropy (gamma=2.0) with class weights [2.0, 0.5, 4.0]
      + per-sample loss weighting by non-zero label count (activity weighting)
  - Optimizer: MuonWithAuxAdam
      - Muon (lr=0.005) for ResBlock 2D weight matrices
      - AdamW (backbone_lr=1e-5) for backbone
      - AdamW (gene_emb_lr=1e-4, gene_emb_wd=1e-2) for out_gene_emb — dedicated low LR
      - AdamW (head_lr=5e-4) for other head params (norms, biases, input_proj)
  - Scheduler: Cosine warm restarts (T_0=1200 steps) — restored from single cosine
  - Gradient clipping: 1.0

Key Improvements Over Parent (node4-2-3, F1=0.4953):
  1. INTERMEDIATE HEAD CAPACITY (primary change):
     - 5 ResBlocks instead of 4 (hidden=320 vs 256, rank=320 vs 256)
     - ~7.3M trainable params vs ~4.2M
     - Addresses the underfitting bottleneck confirmed in node4-2-3 feedback
     - Evidence: 4.2M params produced F1=0.4953 (underfitting); 17M → 0.5069 (overfitting)
     - 7.3M is the untested mid-point, expected to balance expressiveness and generalization
  2. WARM RESTART SCHEDULE (key schedule change):
     - Restores cosine warm restarts (T_0=1200 steps) instead of single cosine decay
     - Evidence: node4-2-1-2 (4×256/rank=256 + T_0=1200) achieved F1=0.5075 while
       node4-2-3 (identical capacity + single cosine) achieved only 0.4953 (−0.0122!)
     - This strongly suggests the warm restart staircase mechanism is critical for performance
     - T_0=1200 matches the proven setting from node4-2-1 (0.5076) and node4-2-1-2 (0.5075)
  3. RETAIN ALL PROVEN INNOVATIONS:
     - Dedicated out_gene_emb lr=1e-4/wd=1e-2 (proved sound in parent; keep)
     - Per-sample activity weighting (log(1+n_nonzero)/mean; keep)
     - Gradient clipping=1.0 (keep)
     - Dropout=0.4 (keep)
  4. PATIENCE=80 and max_epochs=350 to allow 5+ full warm restart cycles

Differentiation from parent and lineage context:
  - node4-2-3: 4×256/rank=256 + single cosine → F1=0.4953 (underfitting + no staircase)
  - node4-2-1: 6×512/rank=512 + T_0=1200 → F1=0.5076 (overfitting but staircase works)
  - node4-2-1-2: 4×256/rank=256 + T_0=1200 → F1=0.5075 (staircase rescues capacity)
  - THIS NODE: 5×320/rank=320 + T_0=1200 → targeting F1≥0.508 (intermediate capacity + staircase)

Memory Sources:
  - node4-2-3/memory/feedback.md: "intermediate capacity (~7-10M params: e.g., 5× ResBlocks
    hidden=320-384, rank=320-384)"; "retain activity weighting, gradient clipping"
  - node4-2-1-2/memory: "4×256 + T_0=1200 → 0.5075, staircase rescues performance"
  - node4-2-1/memory/feedback.md: "T_0=1200 enables deeper Cycle 2 convergence"
  - collected_memory: class weights [2.0, 0.5, 4.0], focal gamma=2.0, muon_lr=0.005 proven optimal
  - collected_memory: dedicated out_gene_emb low-LR group (lr=1e-4, wd=1e-2) is sound
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
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
from muon import MuonWithAuxAdam
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_DIM        = 256
N_NODES        = 18870

# Class weights: down(-1)=2.0, neutral(0)=0.5, up(+1)=4.0
# Proven effective across all high-performing tree nodes
CLASS_WEIGHTS = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)

# Focal loss gamma: focuses training on hard examples
FOCAL_GAMMA = 2.0


# ─── Focal Loss with Sample Activity Weighting ────────────────────────────────

def focal_cross_entropy_weighted(
    logits: torch.Tensor,          # [B, C, L]
    labels: torch.Tensor,          # [B, L] long
    class_weights: torch.Tensor,   # [C]
    gamma: float = 2.0,
    sample_weights: Optional[torch.Tensor] = None,  # [B] float, or None
) -> torch.Tensor:
    """
    Focal cross-entropy loss for multi-output 3-class classification.
    Logits: [B, 3, L], labels: [B, L], class_weights: [3]
    sample_weights: [B] per-sample weight (optional)

    Per-sample weighting is applied before averaging.
    """
    B, C, L = logits.shape
    # Reshape to [B*L, C] for cross_entropy
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)   # [B*L, C]
    labels_flat = labels.reshape(-1)                         # [B*L]

    # Standard cross-entropy per element (no reduction)
    ce_loss = F.cross_entropy(
        logits_flat, labels_flat,
        weight=class_weights.to(logits_flat.device),
        reduction="none",
    )  # [B*L]

    # Focal weight: (1 - p_t)^gamma
    with torch.no_grad():
        probs = F.softmax(logits_flat.float(), dim=-1)        # [B*L, C]
        pt    = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [B*L]
        focal_weight = (1.0 - pt).pow(gamma)

    per_element_loss = focal_weight * ce_loss  # [B*L]

    if sample_weights is not None:
        # Reshape to [B, L] for per-sample weighting
        per_element_loss = per_element_loss.view(B, L)
        # sample_weights: [B] → [B, 1]
        sw = sample_weights.to(per_element_loss.device).float().unsqueeze(1)
        per_element_loss = per_element_loss * sw
        loss = per_element_loss.mean()
    else:
        loss = per_element_loss.mean()

    return loss


# ─── Per-Gene F1 ──────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Mirrors calc_metric.py: per-gene macro-F1 over present classes."""
    pred_cls = pred_np.argmax(axis=1)   # [N, L]
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]; yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class StringGNNDataset(Dataset):
    def __init__(
        self,
        pert_ids:    List[str],
        symbols:     List[str],
        node_indices: torch.Tensor,       # [N] long, -1 for unknown
        labels:      Optional[torch.Tensor] = None,  # [N, 6640] long
        sample_weights: Optional[torch.Tensor] = None,  # [N] float
    ):
        self.pert_ids     = pert_ids
        self.symbols      = symbols
        self.node_indices = node_indices
        self.labels       = labels
        self.sample_weights = sample_weights

    def __len__(self): return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
            "node_index": self.node_indices[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        if self.sample_weights is not None:
            item["sample_weight"] = self.sample_weights[idx]
        return item


def collate_fn(batch):
    out = {
        "pert_id":    [b["pert_id"]    for b in batch],
        "symbol":     [b["symbol"]     for b in batch],
        "node_index": torch.stack([b["node_index"] for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    if "sample_weight" in batch[0]:
        out["sample_weight"] = torch.stack([b["sample_weight"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class StringGNNDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data", micro_batch_size=8, num_workers=4):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        gnn_dir    = Path(STRING_GNN_DIR)
        node_names = json.loads((gnn_dir / "node_names.json").read_text())
        node_name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(node_names)}

        def compute_sample_weights(labels_tensor: torch.Tensor) -> torch.Tensor:
            """
            Compute per-sample activity weights based on non-zero label count.
            Weight = log(1 + n_nonzero) / mean_log_activity
            Upweights samples with more non-zero labels (more informative perturbations).
            Labels are already shifted: 0=down, 1=neutral, 2=up, so non-neutral = label != 1.
            """
            n_nonzero = (labels_tensor != 1).sum(dim=1).float()  # [N]
            log_activity = torch.log1p(n_nonzero)
            mean_activity = log_activity.mean().clamp(min=1.0)
            weights = log_activity / mean_activity
            weights = weights.clamp(min=0.2, max=3.0)
            return weights

        def load_split(fname: str, has_lbl: bool, compute_weights: bool = False):
            df  = pd.read_csv(self.data_dir / fname, sep="\t")
            idxs = torch.tensor(
                [node_name_to_idx.get(pid, -1) for pid in df["pert_id"].tolist()],
                dtype=torch.long,
            )
            labels = None
            sample_weights = None
            if has_lbl and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
                if compute_weights:
                    sample_weights = compute_sample_weights(labels)
            return StringGNNDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(), idxs, labels,
                sample_weights=sample_weights,
            )

        # Only compute sample weights for training split
        self.train_ds = load_split("train.tsv", True, compute_weights=True)
        self.val_ds   = load_split("val.tsv",   True, compute_weights=False)
        self.test_ds  = load_split("test.tsv",  True, compute_weights=False)

        # Store graph data for the LightningModule
        graph = torch.load(gnn_dir / "graph_data.pt", weights_only=False)
        self.edge_index  = graph["edge_index"]
        self.edge_weight = graph.get("edge_weight", None)

        n_unknown = sum(
            1 for ds in (self.train_ds, self.val_ds, self.test_ds)
            for ni in ds.node_indices.tolist() if ni == -1
        )
        total = len(self.train_ds) + len(self.val_ds) + len(self.test_ds)
        print(f"[Node4-2-3-1] {n_unknown}/{total} samples not in STRING_GNN "
              f"→ learned fallback embedding.")

        if self.train_ds.sample_weights is not None:
            sw = self.train_ds.sample_weights
            print(f"[Node4-2-3-1] Activity weights: min={sw.min():.3f}, "
                  f"mean={sw.mean():.3f}, max={sw.max():.3f}")

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=self.num_workers,
            pin_memory=True, drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Residual Block ───────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """MLP residual block used in the bilinear head."""
    def __init__(self, hidden_dim: int, expand: int = 4, dropout: float = 0.4):
        super().__init__()
        mid = hidden_dim * expand
        self.fc1 = nn.Linear(hidden_dim, mid, bias=False)
        self.fc2 = nn.Linear(mid, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x + residual


# ─── Model ────────────────────────────────────────────────────────────────────

class StringGNNIntermediateBilinearModel(nn.Module):
    """
    STRING_GNN (partial fine-tuning: mps.7 + post_mp) +
    Intermediate-capacity 5-layer Residual MLP (hidden=320) + Rank-320 Bilinear Head.

    Architecture:
      1. Pre-computed frozen backbone (mps.0-6 + emb) stored as buffer
      2. Trainable tail: mps.7 + post_mp
      3. Fallback embedding for unknown genes
      4. Linear projection: gnn_dim(256) → hidden_dim(320)
      5. 5× ResidualBlock(hidden=320, expand=4, dropout=0.4)
      6. Bilinear head: fc_bilinear produces [B, 3*rank] decomposed into
         [B, 3, rank] × [n_genes_out, rank]^T → [B, 3, n_genes_out]

    Total trainable parameters: ~7.3M
    - Intermediate between parent's 4.2M (underfitting) and grandparent's 17M (overfitting)
    - Parameter-to-sample ratio: ~7.3M / 1,416 ≈ 5,200:1
    """

    def __init__(
        self,
        edge_index:  torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        gnn_dim:     int = GNN_DIM,
        n_genes_out: int = N_GENES_OUT,
        n_classes:   int = N_CLASSES,
        hidden_dim:  int = 320,    # Intermediate: between 256 (parent) and 512 (grandparent)
        n_layers:    int = 5,      # Intermediate: between 4 (parent) and 6 (grandparent)
        expand:      int = 4,
        bilinear_rank: int = 320,  # Intermediate: between 256 (parent) and 512 (grandparent)
        dropout:     float = 0.4,  # Proven effective
    ):
        super().__init__()
        self.gnn_dim      = gnn_dim
        self.n_classes    = n_classes
        self.n_genes_out  = n_genes_out
        self.bilinear_rank = bilinear_rank

        # ── Backbone ──────────────────────────────────────────────────────────
        # Load full STRING_GNN
        full_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        # Register frozen graph tensors
        self.register_buffer("edge_index",  edge_index)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        # Store the backbone modules
        self.emb     = full_gnn.emb         # Embedding(18870, 256) - frozen
        self.mps_0_6 = nn.ModuleList([full_gnn.mps[i] for i in range(7)])  # frozen
        self.mps_7   = full_gnn.mps[7]      # trainable
        self.post_mp = full_gnn.post_mp     # trainable

        # Freeze emb + mps.0-6
        for p in self.emb.parameters():
            p.requires_grad_(False)
        for layer in self.mps_0_6:
            for p in layer.parameters():
                p.requires_grad_(False)

        # Fallback embedding for genes absent from STRING_GNN
        self.fallback_emb = nn.Parameter(torch.randn(gnn_dim) * 0.02)

        # ── Head ──────────────────────────────────────────────────────────────
        # Input projection: gnn_dim(256) → hidden_dim(320)
        # Now gnn_dim != hidden_dim, so this is a true projection
        self.input_proj = nn.Sequential(
            nn.LayerNorm(gnn_dim),
            nn.Linear(gnn_dim, hidden_dim, bias=False),
        )

        # Residual blocks (intermediate capacity)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Bilinear decomposition: hidden_dim → n_classes × bilinear_rank
        self.fc_bilinear = nn.Linear(hidden_dim, n_classes * bilinear_rank, bias=False)

        # Output gene embedding matrix: [n_genes_out, bilinear_rank]
        # Intermediate rank=320 (between 256 and 512)
        self.out_gene_emb = nn.Embedding(n_genes_out, bilinear_rank)
        nn.init.xavier_uniform_(self.out_gene_emb.weight)

        # Cache for precomputed frozen intermediate embeddings
        self._frozen_emb_cache: Optional[torch.Tensor] = None

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Node4-2-3-1] Trainable: {n_trainable:,} / {n_total:,} params "
              f"({100*n_trainable/n_total:.2f}%)")

    def _compute_backbone_embs(self) -> torch.Tensor:
        """
        Run partial forward:
          emb.weight → mps.0-6 (frozen, cached) → mps.7 (trainable) → post_mp (trainable)
        Returns node_emb [N_nodes, 256].
        """
        ei = self.edge_index
        ew = self.edge_weight

        # Compute/use cached frozen intermediate (output after mps.6)
        if self._frozen_emb_cache is None:
            x = self.emb.weight  # [N, 256]
            for layer in self.mps_0_6:
                x = layer(x, ei, ew)
            self._frozen_emb_cache = x.detach()

        x = self._frozen_emb_cache  # [N, 256] — no grad

        # Trainable tail
        x = self.mps_7(x, ei, ew)  # [N, 256]
        x = self.post_mp(x)         # [N, 256]
        return x

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_indices: [B] long — STRING_GNN node indices (-1 = unknown)
        Returns:
            logits: [B, 3, 6640]
        """
        # Full partial-backbone pass
        node_emb = self._compute_backbone_embs()  # [N_nodes, 256]

        # Extract perturbed gene embeddings; handle unknowns
        known_mask = (node_indices >= 0)
        safe_idx   = node_indices.clamp(min=0)
        pert_emb   = node_emb[safe_idx, :]        # [B, 256]
        if not known_mask.all():
            fallback = self.fallback_emb.unsqueeze(0).expand_as(pert_emb)
            pert_emb = torch.where(
                known_mask.unsqueeze(-1).expand_as(pert_emb),
                pert_emb, fallback,
            )

        # Ensure float32 for head
        pert_emb = pert_emb.float()

        # Input projection: gnn_dim(256) → hidden_dim(320)
        h = self.input_proj(pert_emb)   # [B, hidden_dim]

        # Residual blocks
        for block in self.res_blocks:
            h = block(h)                # [B, hidden_dim]

        # Bilinear decomposition
        # [B, hidden_dim] → [B, n_classes*rank] → [B, n_classes, rank]
        blin = self.fc_bilinear(h).view(-1, self.n_classes, self.bilinear_rank)  # [B, 3, rank]

        # Output gene embeddings: [n_genes_out, rank]
        out_embs = self.out_gene_emb.weight   # [6640, rank]

        # [B, 3, rank] @ [rank, 6640] → [B, 3, 6640]
        logits = torch.matmul(blin, out_embs.T)
        return logits


# ─── Helpers for DDP gathering ────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
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

class StringGNNIntermediateLitModule(pl.LightningModule):

    def __init__(
        self,
        backbone_lr:   float = 1e-5,
        head_lr:       float = 5e-4,
        muon_lr:       float = 0.005,
        weight_decay:  float = 1e-3,
        gene_emb_lr:   float = 1e-4,   # Dedicated reduced LR for out_gene_emb
        gene_emb_wd:   float = 1e-2,   # Strong WD for out_gene_emb
        t0_steps:      int   = 1200,   # Cosine warm restart cycle length (proven setting)
        warmup_steps:  int   = 100,
        eta_min_frac:  float = 1e-6,   # LR lower bound fraction (eta_min = eta_min_frac * peak_lr)
        focal_gamma:   float = FOCAL_GAMMA,
        use_activity_weighting: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str]          = []
        self._test_symbols:  List[str]          = []
        self._test_labels:   List[torch.Tensor] = []

    def setup(self, stage=None):
        dm = self.trainer.datamodule if self.trainer is not None else None
        if dm is None:
            raise RuntimeError("DataModule must be attached to the trainer.")

        self.model = StringGNNIntermediateBilinearModel(
            edge_index  = dm.edge_index,
            edge_weight = dm.edge_weight,
        )
        # Cast all trainable parameters to float32 for stable optimization
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(self, node_indices):
        return self.model(node_indices)

    def _loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return focal_cross_entropy_weighted(
            logits, labels,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
            sample_weights=sample_weights if self.hparams.use_activity_weighting else None,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        sw     = batch.get("sample_weight", None)
        loss   = self._loss(logits, batch["label"], sample_weights=sw)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        if "label" in batch:
            # Val loss without sample weighting for a fair comparison
            loss = self._loss(logits, batch["label"], sample_weights=None)
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
        f1 = compute_per_gene_f1(lp.numpy(), ll.numpy())
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        probs  = torch.softmax(logits.float(), dim=1)
        self._test_preds.append(probs.detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (
            torch.cat(self._test_labels, 0) if self._test_labels
            else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        )
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

        # Deduplicate (DDP DistributedSampler pads the dataset)
        seen: set = set()
        keep: List[int] = []
        for i, pid in enumerate(all_pert):
            if pid not in seen:
                seen.add(pid)
                keep.append(i)
        if len(keep) < len(all_pert):
            self.print(f"[Node4-2-3-1] Deduplicating: {len(all_pert)} → {len(keep)}")
            all_probs  = all_probs[keep]
            all_labels = all_labels[keep]
            all_pert   = [all_pert[i] for i in keep]
            all_syms   = [all_syms[i]  for i in keep]

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pid, sym, probs in zip(all_pert, all_syms, all_probs.numpy()):
                    fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")
            self.print(f"[Node4-2-3-1] Saved predictions → {pred_path}")
            if all_labels.any():
                f1 = compute_per_gene_f1(all_probs.numpy(), all_labels.numpy())
                self.print(f"[Node4-2-3-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear(); self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Separate parameters into groups:
        # 1. Backbone trainable (mps.7 + post_mp) — AdamW at backbone_lr
        # 2. Head 2D weight matrices (ResBlocks, bilinear) — Muon
        # 3. out_gene_emb — dedicated AdamW with LOW LR (1e-4) and HIGH WD (1e-2)
        # 4. Other head params (norms, biases, input_proj, fallback_emb) — AdamW at head_lr

        backbone_params = (
            list(self.model.mps_7.parameters()) +
            list(self.model.post_mp.parameters())
        )
        backbone_param_ids = {id(p) for p in backbone_params}

        # out_gene_emb — dedicated low-LR group
        gene_emb_params = list(self.model.out_gene_emb.parameters())
        gene_emb_param_ids = {id(p) for p in gene_emb_params}

        # Head 2D matrices for Muon: fc1.weight, fc2.weight in ResBlocks, fc_bilinear.weight
        # Note: input_proj.1 (Linear 256→320) is a boundary projection — excluded from Muon
        head_2d_matrices = []
        head_other       = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in backbone_param_ids:
                continue  # handled separately
            if id(param) in gene_emb_param_ids:
                continue  # handled separately
            # Muon: 2D weight matrices in hidden layers (not embeddings, not boundary projections)
            if (param.ndim >= 2
                    and "input_proj.1" not in name  # input projection (boundary)
                    and "fallback_emb" not in name):
                head_2d_matrices.append(param)
            else:
                head_other.append(param)

        param_groups = [
            # Group 0: Backbone (mps.7 + post_mp) — AdamW
            {
                "params":       backbone_params,
                "use_muon":     False,
                "lr":           hp.backbone_lr,
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.weight_decay,
            },
            # Group 1: Head 2D matrices — Muon
            {
                "params":       head_2d_matrices,
                "use_muon":     True,
                "lr":           hp.muon_lr,
                "momentum":     0.95,
                "weight_decay": hp.weight_decay,
            },
            # Group 2: out_gene_emb — dedicated low-LR, high-WD AdamW
            # 6640×320=2.1M params; needs strong regularization despite reduced size
            {
                "params":       gene_emb_params,
                "use_muon":     False,
                "lr":           hp.gene_emb_lr,   # 1e-4 (5x lower than standard)
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.gene_emb_wd,   # 1e-2 (strong L2 constraint)
            },
            # Group 3: Other head params (norms, biases, input_proj, fallback_emb) — AdamW
            {
                "params":       head_other,
                "use_muon":     False,
                "lr":           hp.head_lr,
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.weight_decay * 0.1,  # lighter on non-matrix params
            },
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # Cosine warm restarts schedule (T_0=1200 steps)
        # This is the key change from parent's single cosine:
        # - node4-2-1-2 (identical 4×256/rank=256 capacity) achieved 0.5075 with T_0=1200
        # - node4-2-3 (same capacity) achieved only 0.4953 with single cosine
        # - The staircase mechanism (each restart allows a new performance jump) is critical
        #
        # T_0=1200 steps at 2 GPUs, micro_batch=8, global_batch=32, accum=2:
        #   steps_per_epoch ≈ ceil(1416/8) / 2 accum ≈ 45 optimizer steps/epoch per GPU
        #   cycle_length_epochs = 1200 / 45 ≈ 26-27 epochs per cycle
        # With patience=80, we allow ~3 epochs post-best → fits ~4-5 full cycles
        warmup = hp.warmup_steps
        T_0    = hp.t0_steps
        eta_min_frac = hp.eta_min_frac

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / float(max(1, warmup))
            step_after = step - warmup
            # Cosine warm restarts: find current cycle position
            cycle_step = step_after % T_0
            progress   = float(cycle_step) / float(T_0)
            return eta_min_frac + 0.5 * (1.0 - eta_min_frac) * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
                "frequency": 1,
            },
        }

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
        self.print(f"Saving ckpt: {trained}/{total} params ({100*trained/total:.2f}%)")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 4-2-3-1 — STRING_GNN Partial FT + Intermediate Capacity Head + "
                    "Cosine Warm Restarts + Activity Weighting"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--backbone-lr",       type=float, default=1e-5)
    p.add_argument("--head-lr",           type=float, default=5e-4)
    p.add_argument("--muon-lr",           type=float, default=0.005)
    p.add_argument("--weight-decay",      type=float, default=1e-3)
    p.add_argument("--gene-emb-lr",       type=float, default=1e-4)
    p.add_argument("--gene-emb-wd",       type=float, default=1e-2)
    p.add_argument("--micro-batch-size",  type=int,   default=8)
    p.add_argument("--global-batch-size", type=int,   default=32)
    p.add_argument("--max-epochs",        type=int,   default=350)
    p.add_argument("--patience",          type=int,   default=80)
    p.add_argument("--t0-steps",          type=int,   default=1200,
                   help="Cosine warm restart cycle length in optimizer steps")
    p.add_argument("--warmup-steps",      type=int,   default=100)
    p.add_argument("--eta-min-frac",      type=float, default=1e-6,
                   help="Fraction of peak LR as eta_min for cosine restarts")
    p.add_argument("--no-activity-weighting", action="store_true", default=False)
    p.add_argument("--num-workers",       type=int,   default=4)
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

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm  = StringGNNDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = StringGNNIntermediateLitModule(
        backbone_lr   = args.backbone_lr,
        head_lr       = args.head_lr,
        muon_lr       = args.muon_lr,
        weight_decay  = args.weight_decay,
        gene_emb_lr   = args.gene_emb_lr,
        gene_emb_wd   = args.gene_emb_wd,
        t0_steps      = args.t0_steps,
        warmup_steps  = args.warmup_steps,
        eta_min_frac  = args.eta_min_frac,
        use_activity_weighting = not args.no_activity_weighting,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath   = str(out_dir / "checkpoints"),
        filename  = "best-{epoch:04d}-{val_f1:.4f}",
        monitor   = "val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max",
                            patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)
    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps    = -1
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False
    if args.debug_max_step is not None:
        max_steps = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2
        limit_test  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator   = "gpu",
        devices       = n_gpus,
        num_nodes     = 1,
        strategy      = strategy,
        precision     = "bf16-mixed",
        max_epochs    = args.max_epochs,
        max_steps     = max_steps,
        accumulate_grad_batches = accum,
        gradient_clip_val = 1.0,   # Proven effective (stabilizes Muon LR updates)
        limit_train_batches  = limit_train,
        limit_val_batches    = limit_val,
        limit_test_batches   = limit_test,
        val_check_interval   = (
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps = 2,
        callbacks     = [ckpt_cb, es_cb, lr_cb, pb_cb],
        logger        = [csv_logger, tb_logger],
        log_every_n_steps = 10,
        deterministic = False,   # nll_loss2d has no deterministic CUDA impl
        default_root_dir = str(out_dir),
        fast_dev_run  = fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 4-2-3-1 — STRING_GNN Partial FT + Intermediate Capacity Head + "
            "Cosine Warm Restarts + Activity Weighting\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
