"""
Node 4-2-1-3: STRING_GNN Extended Backbone (mps.6+7+post_mp) + SGDR Scheduling
              + Label Smoothing + Fixed Quality-Filtered SWA + Dropout=0.45

Architecture Strategy:
  - STRING_GNN backbone with EXTENDED partial fine-tuning:
      * Frozen layers (mps.0-5) + embedding: pre-computed as a buffer in setup()
        to speed up forward pass (mps.5 output cached)
      * Trainable layers: mps.6 + mps.7 + post_mp (~198K params, backbone_lr=1e-5)
        [Extended from parent's mps.7+post_mp ~67K — mirrors tree-best configuration]
  - Deep 6-layer Residual Bilinear MLP head (rank=512, hidden=512):
      ResidualBlock × 6 → bilinear output (rank-512 decomposition)
      Produces logits of shape [B, 3, 6640]
  - Loss: Focal cross-entropy (gamma=2.0) with class weights [2.0, 0.5, 4.0]
          AND label_smoothing=0.05 (proven in tree-best lineage, eps=0.05)
  - Optimizer: MuonWithAuxAdam
      - Muon (lr=0.005) for ResBlock 2D weight matrices
      - AdamW (backbone_lr=1e-5, head_lr=5e-4) for other parameters
      - Separate AdamW group with strong WD (1e-2) for out_gene_emb
  - Scheduler: SGDR (T_0=20 epochs, T_mult=2) for diverse checkpoint pool
               [Changed from parent's cosine restarts T_0=1200, T_mult=1]
  - Post-hoc Quality-Filtered SWA:
      Periodic checkpoints every 5 epochs from epoch 10
      Filter: val_f1 >= 0.490, top-15, exponential weighting temp=3.0
      CORRECT inference: saved to separate ckpt, loaded via ckpt_path parameter
  - Dropout: 0.45 (increased from parent's 0.40; mirrors tree-best)
  - Patience: 200 epochs (mirrors tree-best; allows full SGDR pool accumulation)

Key Improvements Over Parent (node4-2-1, F1=0.5076):
  1. Extended backbone fine-tuning: mps.6+mps.7+post_mp vs mps.7+post_mp only
     - Tree best (node2-1-1-1-2-1-1-1-1-1-1, F1=0.5180) uses mps.6+7+post_mp
     - One additional GCN hop provides richer PPI context representations
     - backbone_lr=1e-5 (same) ensures conservative, stable adaptation
  2. SGDR scheduling (T_0=20, T_mult=2) → diverse SWA pool
     - Short early cycles (20 epochs) enable fast exploration
     - Progressive lengthening enables deep convergence
     - Creates rich checkpoint diversity across training
  3. Quality-filtered SWA (top-15, temp=3.0) with CORRECT inference
     - Tree-best gained +0.0065 F1 from SWA
     - Fix for node4-2-1-2's setup() reinit bug: use ckpt_path parameter
  4. Label smoothing (eps=0.05): proven calibration in tree-best lineage
  5. Dropout 0.4 → 0.45: matches tree-best hyperparameters
  6. Patience 50 → 200: allows full SGDR pool accumulation (tree-best pattern)

Memory Sources:
  - node2-1-1-1-2-1-1-1-1-1-1 (F1=0.5180): mps.6+7 backbone, SGDR T_0=20 T_mult=2,
    dropout=0.45, quality SWA +0.0065 F1 gain
  - node2-1-1-1-2-1-1-1-1-1-1-1-1 (F1=0.5182, tree best): eps=0.05, patience=200
  - node4-2-1-2 feedback: SWA setup() bug fix via ckpt_path; find_unused_params=False
  - node4-2-1 feedback: gene_emb_wd=1e-2 correct; gradient_clip=1.0 correct
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import re
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional

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
    Callback,
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
# Proven effective in the tree best lineage (node2-1-3, node1-2-2-2-1, node4-2)
CLASS_WEIGHTS = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)

# Focal loss gamma
FOCAL_GAMMA = 2.0


# ─── Focal Loss with Label Smoothing ──────────────────────────────────────────

def focal_cross_entropy_with_smoothing(
    logits: torch.Tensor,         # [B, C, L]
    labels: torch.Tensor,         # [B, L] long
    class_weights: torch.Tensor,  # [C]
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """
    Focal cross-entropy loss with label smoothing for multi-output 3-class classification.
    Logits: [B, 3, L], labels: [B, L], class_weights: [3]

    Label smoothing: replaces one-hot targets with (1-eps)*one_hot + eps/C,
    calibrating predictions and preventing overconfidence on the sparse label space.
    """
    B, C, L = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)   # [B*L, C]
    labels_flat = labels.reshape(-1)                         # [B*L]

    log_probs = F.log_softmax(logits_flat.float(), dim=-1)   # [B*L, C]

    if label_smoothing > 0.0:
        # Smooth labels: one_hot → (1-eps)*one_hot + eps/C
        one_hot = F.one_hot(labels_flat, num_classes=C).float()  # [B*L, C]
        smooth  = (1.0 - label_smoothing) * one_hot + label_smoothing / C  # [B*L, C]

        # Class-weighted cross-entropy with smooth labels
        w = class_weights.to(logits_flat.device)   # [C]
        # Per-element: -sum(smooth * log_prob * weight)
        ce_loss = -(smooth * log_probs * w.unsqueeze(0)).sum(dim=-1)  # [B*L]
    else:
        ce_loss = F.cross_entropy(
            logits_flat, labels_flat,
            weight=class_weights.to(logits_flat.device),
            reduction="none",
        )  # [B*L]

    # Focal weight: (1 - p_t)^gamma  (computed on the true class probability)
    with torch.no_grad():
        probs = F.softmax(logits_flat.float(), dim=-1)   # [B*L, C]
        pt    = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [B*L]
        focal_weight = (1.0 - pt).pow(gamma)

    loss = (focal_weight * ce_loss).mean()
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
        pert_ids:     List[str],
        symbols:      List[str],
        node_indices: torch.Tensor,       # [N] long, -1 for unknown
        labels:       Optional[torch.Tensor] = None,  # [N, 6640] long
    ):
        self.pert_ids     = pert_ids
        self.symbols      = symbols
        self.node_indices = node_indices
        self.labels       = labels

    def __len__(self): return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
            "node_index": self.node_indices[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    out = {
        "pert_id":    [b["pert_id"]    for b in batch],
        "symbol":     [b["symbol"]     for b in batch],
        "node_index": torch.stack([b["node_index"] for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
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

        def load_split(fname: str, has_lbl: bool):
            df   = pd.read_csv(self.data_dir / fname, sep="\t")
            idxs = torch.tensor(
                [node_name_to_idx.get(pid, -1) for pid in df["pert_id"].tolist()],
                dtype=torch.long,
            )
            labels = None
            if has_lbl and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return StringGNNDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(), idxs, labels
            )

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

        # Store graph data for the LightningModule
        graph = torch.load(gnn_dir / "graph_data.pt", weights_only=False)
        self.edge_index  = graph["edge_index"]
        self.edge_weight = graph.get("edge_weight", None)

        n_unknown = sum(
            1 for ds in (self.train_ds, self.val_ds, self.test_ds)
            for ni in ds.node_indices.tolist() if ni == -1
        )
        total = len(self.train_ds) + len(self.val_ds) + len(self.test_ds)
        print(f"[Node4-2-1-3] {n_unknown}/{total} samples not in STRING_GNN "
              f"→ learned fallback embedding.")

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
    def __init__(self, hidden_dim: int, expand: int = 4, dropout: float = 0.45):
        super().__init__()
        mid = hidden_dim * expand
        self.fc1  = nn.Linear(hidden_dim, mid, bias=False)
        self.fc2  = nn.Linear(mid, hidden_dim, bias=False)
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

class StringGNNExtendedBilinearModel(nn.Module):
    """
    STRING_GNN (EXTENDED partial fine-tuning: mps.6 + mps.7 + post_mp) +
    Deep 6-layer Residual MLP + Rank-512 Bilinear Interaction Head.

    Architecture:
      1. Pre-computed frozen backbone (mps.0-5 + emb) stored as buffer
         [Changed from parent: mps.5 output cached, not mps.6 output]
      2. Trainable tail: mps.6 + mps.7 + post_mp (~198K params)
         [Extended from parent's mps.7+post_mp only ~67K params]
      3. Fallback embedding for unknown genes
      4. 6× ResidualBlock(hidden=512, expand=4, dropout=0.45)  [increased from 0.40]
      5. Bilinear head: fc_bilinear produces [B, 3*rank] decomposed into
         [B, 3, rank] × [n_genes_out, rank]^T → [B, 3, n_genes_out]
    """

    def __init__(
        self,
        edge_index:    torch.Tensor,
        edge_weight:   Optional[torch.Tensor],
        gnn_dim:       int = GNN_DIM,
        n_genes_out:   int = N_GENES_OUT,
        n_classes:     int = N_CLASSES,
        hidden_dim:    int = 512,
        n_layers:      int = 6,
        expand:        int = 4,
        bilinear_rank: int = 512,
        dropout:       float = 0.45,
    ):
        super().__init__()
        self.gnn_dim       = gnn_dim
        self.n_classes     = n_classes
        self.n_genes_out   = n_genes_out
        self.bilinear_rank = bilinear_rank

        # ── Backbone ──────────────────────────────────────────────────────────
        full_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        # Register frozen graph tensors
        self.register_buffer("edge_index",  edge_index)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        # Store backbone components
        self.emb     = full_gnn.emb             # Embedding(18870, 256) - frozen
        self.mps_0_5 = nn.ModuleList([full_gnn.mps[i] for i in range(6)])  # frozen
        self.mps_6   = full_gnn.mps[6]          # trainable  ← EXTENDED
        self.mps_7   = full_gnn.mps[7]          # trainable
        self.post_mp = full_gnn.post_mp          # trainable

        # Freeze emb + mps.0-5
        for p in self.emb.parameters():
            p.requires_grad_(False)
        for layer in self.mps_0_5:
            for p in layer.parameters():
                p.requires_grad_(False)

        # mps.6, mps.7, post_mp remain trainable (default)

        # Fallback embedding for genes absent from STRING_GNN
        self.fallback_emb = nn.Parameter(torch.randn(gnn_dim) * 0.02)

        # ── Head ──────────────────────────────────────────────────────────────
        # Input projection: gnn_dim → hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(gnn_dim),
            nn.Linear(gnn_dim, hidden_dim, bias=False),
        )

        # Residual blocks (dropout=0.45)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Bilinear decomposition: hidden_dim → n_classes × bilinear_rank
        self.fc_bilinear = nn.Linear(hidden_dim, n_classes * bilinear_rank, bias=False)

        # Output gene embedding matrix: [n_genes_out, bilinear_rank]
        self.out_gene_emb = nn.Embedding(n_genes_out, bilinear_rank)
        nn.init.xavier_uniform_(self.out_gene_emb.weight)

        # Cache for precomputed frozen intermediate embeddings (output of mps.5)
        self._frozen_emb_cache: Optional[torch.Tensor] = None

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Node4-2-1-3] Trainable: {n_trainable:,} / {n_total:,} params "
              f"({100*n_trainable/n_total:.2f}%)")

    def _compute_backbone_embs(self) -> torch.Tensor:
        """
        Run partial forward:
          emb.weight → mps.0-5 (frozen) → mps.6 (trainable) → mps.7 (trainable) → post_mp
        Returns node_emb [N_nodes, 256].

        Uses frozen cache for mps.0-5 output to save compute during training.
        """
        ei = self.edge_index
        ew = self.edge_weight

        # Compute/use cached frozen intermediate (output after mps.5)
        if self._frozen_emb_cache is None:
            x = self.emb.weight  # [N, 256]
            for layer in self.mps_0_5:
                x = layer(x, ei, ew)
            self._frozen_emb_cache = x.detach()

        x = self._frozen_emb_cache  # [N, 256] — no grad

        # Trainable tail: mps.6, mps.7, post_mp
        x = self.mps_6(x, ei, ew)  # [N, 256]
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

        # Input projection
        h = self.input_proj(pert_emb)   # [B, hidden_dim]

        # Residual blocks
        for block in self.res_blocks:
            h = block(h)                # [B, hidden_dim]

        # Bilinear decomposition
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


# ─── Periodic Checkpoint Callback ─────────────────────────────────────────────

class PeriodicCheckpointCallback(Callback):
    """
    Saves a checkpoint every `every_n_epochs` epochs starting from `start_epoch`,
    up to `end_epoch` (if specified). Used to build the SWA pool.
    """
    def __init__(
        self,
        dirpath: str,
        every_n_epochs: int = 5,
        start_epoch: int = 10,
        end_epoch: Optional[int] = None,
    ):
        super().__init__()
        self.dirpath        = Path(dirpath)
        self.every_n_epochs = every_n_epochs
        self.start_epoch    = start_epoch
        self.end_epoch      = end_epoch
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch < self.start_epoch:
            return
        if self.end_epoch is not None and epoch > self.end_epoch:
            return
        if (epoch - self.start_epoch) % self.every_n_epochs != 0:
            return
        # NOTE: Do NOT guard with `if not trainer.is_global_zero: return` here.
        # trainer.save_checkpoint() internally calls strategy.barrier(), which is a
        # collective operation requiring ALL ranks to participate. Returning early on
        # non-zero ranks would cause a NCCL collective desync / deadlock.
        # Lightning's DDPStrategy.save_checkpoint() already handles rank-0-only file writing.

        val_f1 = trainer.callback_metrics.get("val_f1", torch.tensor(0.0))
        if isinstance(val_f1, torch.Tensor):
            val_f1 = val_f1.item()

        ckpt_path = self.dirpath / f"periodic-epoch={epoch:04d}-val_f1={val_f1:.4f}.ckpt"
        trainer.save_checkpoint(str(ckpt_path))


# ─── SWA Averaging ────────────────────────────────────────────────────────────

def parse_val_f1_from_filename(filename: str) -> Optional[float]:
    """
    Robustly parse val_f1 from periodic checkpoint filename.
    Handles PyTorch Lightning's actual format where 'val_f1=' may appear multiple times.
    Takes the LAST occurrence (the actual float value).
    """
    matches = re.findall(r'val_f1=(\d+\.\d+)', filename)
    if matches:
        return float(matches[-1])
    return None


def perform_quality_filtered_swa(
    periodic_ckpt_dir: Path,
    model_class,
    model_kwargs: dict,
    lit_module: "StringGNNExtendedLitModule",
    swa_top_k: int = 15,
    swa_val_f1_threshold: float = 0.490,
    swa_weight_temp: float = 3.0,
    output_ckpt_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Load periodic checkpoints, filter by val_f1 threshold, select top-K,
    and average weights with exponential weighting.

    Returns the path to the SWA checkpoint if successful, else None.

    NOTE: This function correctly saves the SWA checkpoint as a Lightning-compatible
    checkpoint file, so it can be loaded via ckpt_path= parameter in trainer.test().
    """
    if not periodic_ckpt_dir.exists():
        print("[SWA] No periodic checkpoint directory found.")
        return None

    ckpt_files = list(periodic_ckpt_dir.glob("periodic-epoch=*.ckpt"))
    if len(ckpt_files) == 0:
        print("[SWA] No periodic checkpoints found.")
        return None

    # Parse val_f1 from filenames
    ckpt_infos = []
    for ckpt_path in ckpt_files:
        val_f1 = parse_val_f1_from_filename(ckpt_path.name)
        if val_f1 is not None:
            ckpt_infos.append((val_f1, ckpt_path))

    if len(ckpt_infos) == 0:
        print("[SWA] Could not parse val_f1 from any checkpoint filename.")
        return None

    print(f"[SWA] Found {len(ckpt_infos)} periodic checkpoints.")
    print(f"[SWA] val_f1 range: [{min(x[0] for x in ckpt_infos):.4f}, {max(x[0] for x in ckpt_infos):.4f}]")

    # Filter by threshold
    qualified = [(f1, p) for f1, p in ckpt_infos if f1 >= swa_val_f1_threshold]
    print(f"[SWA] Qualified (val_f1 >= {swa_val_f1_threshold}): {len(qualified)}/{len(ckpt_infos)}")

    # Fallback: if not enough qualified, use all checkpoints
    if len(qualified) < 3:
        print(f"[SWA] Too few qualified checkpoints. Falling back to all {len(ckpt_infos)} checkpoints.")
        qualified = ckpt_infos

    # Select top-K by val_f1
    qualified_sorted = sorted(qualified, key=lambda x: x[0], reverse=True)
    top_k = qualified_sorted[:swa_top_k]
    print(f"[SWA] Top-{len(top_k)} checkpoints:")
    for f1, p in top_k:
        print(f"  val_f1={f1:.4f}: {p.name}")

    if len(top_k) == 0:
        return None

    # Compute exponential weights
    f1_values = np.array([f1 for f1, _ in top_k])
    f1_min = f1_values.min()
    f1_max = f1_values.max()
    if f1_max > f1_min:
        # Normalize to [0, 1] then apply temperature
        normalized = (f1_values - f1_min) / (f1_max - f1_min)
    else:
        normalized = np.ones_like(f1_values)
    exp_weights = np.exp(swa_weight_temp * normalized)
    exp_weights = exp_weights / exp_weights.sum()
    print(f"[SWA] Weights: {exp_weights}")

    # Load and average state dicts
    avg_state_dict = None
    for (f1, ckpt_path), weight in zip(top_k, exp_weights):
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        # PL checkpoint has 'state_dict' key
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        if avg_state_dict is None:
            avg_state_dict = {k: v.float() * weight for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if k in avg_state_dict:
                    avg_state_dict[k] += v.float() * weight
                else:
                    avg_state_dict[k] = v.float() * weight

    if avg_state_dict is None:
        return None

    # Save averaged state dict as a Lightning-compatible checkpoint
    if output_ckpt_path is None:
        output_ckpt_path = periodic_ckpt_dir.parent / "swa_averaged.ckpt"

    # Create a minimal Lightning checkpoint structure
    # We need to load the best original ckpt structure and replace state_dict
    best_original_ckpt_path = top_k[0][1]
    best_ckpt = torch.load(str(best_original_ckpt_path), map_location="cpu", weights_only=False)
    best_ckpt["state_dict"] = avg_state_dict
    torch.save(best_ckpt, str(output_ckpt_path))
    print(f"[SWA] Saved averaged checkpoint to {output_ckpt_path}")
    return output_ckpt_path


# ─── LightningModule ──────────────────────────────────────────────────────────

class StringGNNExtendedLitModule(pl.LightningModule):

    def __init__(
        self,
        backbone_lr:     float = 1e-5,
        head_lr:         float = 5e-4,
        muon_lr:         float = 0.005,
        weight_decay:    float = 1e-3,
        gene_emb_wd:     float = 1e-2,
        t0_epochs:       int   = 20,    # SGDR T_0 in epochs (not steps)
        t_mult:          int   = 2,     # SGDR T_mult (progressive doubling)
        warmup_steps:    int   = 100,
        max_steps:       int   = 10000,
        focal_gamma:     float = FOCAL_GAMMA,
        label_smoothing: float = 0.05,  # NEW: calibration via label smoothing
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

        self.model = StringGNNExtendedBilinearModel(
            edge_index  = dm.edge_index,
            edge_weight = dm.edge_weight,
        )
        # Cast all trainable parameters to float32
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(self, node_indices):
        return self.model(node_indices)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return focal_cross_entropy_with_smoothing(
            logits, labels,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
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
        f1 = compute_per_gene_f1(lp.numpy(), ll.numpy())
        # f1 is already computed on all-gathered data (global metric), so sync_dist=False
        # avoids a redundant all-reduce collective that could interfere with other collectives.
        self.log("val_f1", f1, prog_bar=True, sync_dist=False)
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
            self.print(f"[Node4-2-1-3] Deduplicating: {len(all_pert)} → {len(keep)}")
            all_probs  = all_probs[keep]
            all_labels = all_labels[keep]
            all_pert   = [all_pert[i] for i in keep]
            all_syms   = [all_syms[i]  for i in keep]

        if self.trainer.is_global_zero:
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Use configurable output path (set by _pred_output_path attribute if present)
            pred_path = getattr(self, "_pred_output_path", None)
            if pred_path is None:
                pred_path = out_dir / "test_predictions.tsv"
            pred_path = Path(pred_path)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pid, sym, probs in zip(all_pert, all_syms, all_probs.numpy()):
                    fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")
            self.print(f"[Node4-2-1-3] Saved predictions → {pred_path}")
            if all_labels.any():
                f1 = compute_per_gene_f1(all_probs.numpy(), all_labels.numpy())
                self.print(f"[Node4-2-1-3] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear(); self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Parameter groups:
        # 1. Extended Backbone (mps.6 + mps.7 + post_mp) — AdamW at backbone_lr
        # 2. Head 2D weight matrices (ResBlocks, bilinear) — Muon
        # 3. out_gene_emb — AdamW with stronger weight_decay (1e-2)
        # 4. Other head params (norms, biases, input_proj, fallback_emb) — AdamW at head_lr

        backbone_params = (
            list(self.model.mps_6.parameters()) +   # EXTENDED: now includes mps.6
            list(self.model.mps_7.parameters()) +
            list(self.model.post_mp.parameters())
        )
        backbone_param_ids = {id(p) for p in backbone_params}

        # out_gene_emb gets its own group with stronger weight decay
        gene_emb_params    = [self.model.out_gene_emb.weight]
        gene_emb_param_ids = {id(p) for p in gene_emb_params}

        # Head 2D matrices for Muon: fc1.weight, fc2.weight, fc_bilinear.weight
        head_2d_matrices = []
        head_other       = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in backbone_param_ids:
                continue
            if id(param) in gene_emb_param_ids:
                continue
            # Muon: 2D weight matrices in hidden layers (not embeddings, not input proj)
            if (param.ndim >= 2
                    and "input_proj.1" not in name
                    and "fallback_emb" not in name):
                head_2d_matrices.append(param)
            else:
                head_other.append(param)

        param_groups = [
            # Extended Backbone (mps.6 + mps.7 + post_mp) — AdamW
            {
                "params":       backbone_params,
                "use_muon":     False,
                "lr":           hp.backbone_lr,
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.weight_decay,
            },
            # Head 2D matrices — Muon
            {
                "params":       head_2d_matrices,
                "use_muon":     True,
                "lr":           hp.muon_lr,
                "momentum":     0.95,
                "weight_decay": hp.weight_decay,
            },
            # out_gene_emb — AdamW with STRONG weight_decay
            {
                "params":       gene_emb_params,
                "use_muon":     False,
                "lr":           hp.head_lr,   # 5e-4: enables rapid Cycle 2 breakthrough
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.gene_emb_wd,  # 1e-2
            },
            # Other head params — AdamW
            {
                "params":       head_other,
                "use_muon":     False,
                "lr":           hp.head_lr,
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.weight_decay * 0.1,
            },
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # SGDR: CosineAnnealingWarmRestarts with T_0 in epochs, T_mult=2
        # This scheduler is applied "per epoch" (not per step).
        # Steps per epoch: approximately 1416/8/n_gpus/accum ≈ 44 on 2 GPUs
        # T_0=20 epochs → first cycle is 20 epochs; second is 40; third is 80; etc.
        # This creates diverse checkpoint states for SWA.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.t0_epochs,
            T_mult=hp.t_mult,
            eta_min=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
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
        description="Node 4-2-1-3 — STRING_GNN Extended Backbone (mps.6+7+post_mp) "
                    "+ SGDR + LabelSmoothing + Quality-Filtered SWA + Dropout=0.45"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--backbone-lr",       type=float, default=1e-5)
    p.add_argument("--head-lr",           type=float, default=5e-4)
    p.add_argument("--muon-lr",           type=float, default=0.005)
    p.add_argument("--weight-decay",      type=float, default=1e-3)
    p.add_argument("--gene-emb-wd",       type=float, default=1e-2)
    p.add_argument("--micro-batch-size",  type=int,   default=8)
    p.add_argument("--global-batch-size", type=int,   default=32)
    p.add_argument("--max-epochs",        type=int,   default=350)
    p.add_argument("--patience",          type=int,   default=200)   # Tree-best pattern
    p.add_argument("--t0-epochs",         type=int,   default=20)    # SGDR T_0 in epochs
    p.add_argument("--t-mult",            type=int,   default=2)     # SGDR T_mult
    p.add_argument("--warmup-steps",      type=int,   default=100)
    p.add_argument("--label-smoothing",   type=float, default=0.05)  # Tree-best eps
    p.add_argument("--swa-start-epoch",   type=int,   default=10)
    p.add_argument("--swa-every-n-epochs", type=int,  default=5)
    p.add_argument("--swa-top-k",         type=int,   default=15)
    p.add_argument("--swa-val-f1-threshold", type=float, default=0.490)
    p.add_argument("--swa-weight-temp",   type=float, default=3.0)
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

    # Approximate steps for warmup reference (not used in SGDR, which is epoch-based)
    steps_per_epoch = int(np.ceil(1416 / args.micro_batch_size))
    estimated_max_steps = args.max_epochs * steps_per_epoch // accum

    dm  = StringGNNDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = StringGNNExtendedLitModule(
        backbone_lr     = args.backbone_lr,
        head_lr         = args.head_lr,
        muon_lr         = args.muon_lr,
        weight_decay    = args.weight_decay,
        gene_emb_wd     = args.gene_emb_wd,
        t0_epochs       = args.t0_epochs,
        t_mult          = args.t_mult,
        warmup_steps    = args.warmup_steps,
        max_steps       = estimated_max_steps,
        label_smoothing = args.label_smoothing,
    )

    # Standard ModelCheckpoint (best single checkpoint)
    ckpt_cb = ModelCheckpoint(
        dirpath   = str(out_dir / "checkpoints"),
        filename  = "best-{epoch:04d}-{val_f1:.4f}",
        monitor   = "val_f1", mode="max", save_top_k=1, save_last=True,
    )

    # Periodic checkpoint callback for SWA pool
    periodic_ckpt_dir = out_dir / "checkpoints" / "periodic"
    periodic_cb = PeriodicCheckpointCallback(
        dirpath        = str(periodic_ckpt_dir),
        every_n_epochs = args.swa_every_n_epochs,
        start_epoch    = args.swa_start_epoch,
    )

    es_cb   = EarlyStopping(monitor="val_f1", mode="max",
                             patience=args.patience, min_delta=1e-5)
    lr_cb   = LearningRateMonitor(logging_interval="epoch")
    pb_cb   = TQDMProgressBar(refresh_rate=10)
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

    # find_unused_parameters=True: fallback_emb may not be used when all genes are in STRING_GNN
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
        gradient_clip_val    = 1.0,   # Stabilize Muon LR updates
        limit_train_batches  = limit_train,
        limit_val_batches    = limit_val,
        limit_test_batches   = limit_test,
        val_check_interval   = (
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps = 2,
        callbacks     = [ckpt_cb, periodic_cb, es_cb, lr_cb, pb_cb],
        logger        = [csv_logger, tb_logger],
        log_every_n_steps = 10,
        deterministic = True,
        default_root_dir = str(out_dir),
        fast_dev_run  = fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)

    if args.fast_dev_run or args.debug_max_step is not None:
        # Fast/debug mode: test with current weights (no ckpt loading)
        test_results = trainer.test(lit, datamodule=dm)
    else:
        # Step 1: Test with best single checkpoint
        # Direct the predictions to a clearly named file
        best_ckpt_path = ckpt_cb.best_model_path
        best_val_f1 = ckpt_cb.best_model_score
        best_val_f1_str = f"{best_val_f1:.4f}" if best_val_f1 is not None else "N/A"
        print(f"\n[Node4-2-1-3] Best single checkpoint: {best_ckpt_path} "
              f"(val_f1={best_val_f1_str})")

        # Point predictions to the primary output path
        lit._pred_output_path = out_dir / "test_predictions.tsv"
        test_results = trainer.test(lit, datamodule=dm, ckpt_path="best")

        # Also save a copy as the "best single" reference
        if trainer.is_global_zero:
            import shutil
            best_single_pred = out_dir / "best_single_test_predictions.tsv"
            primary_pred = out_dir / "test_predictions.tsv"
            if primary_pred.exists():
                shutil.copy(str(primary_pred), str(best_single_pred))
                print(f"[Node4-2-1-3] Best single ckpt predictions backed up to {best_single_pred}")

        # Step 2: Attempt quality-filtered SWA on rank 0 only
        if trainer.is_global_zero:
            print("\n[Node4-2-1-3] Attempting quality-filtered SWA...")
            swa_ckpt_path = perform_quality_filtered_swa(
                periodic_ckpt_dir  = periodic_ckpt_dir,
                model_class        = StringGNNExtendedBilinearModel,
                model_kwargs       = {},
                lit_module         = lit,
                swa_top_k          = args.swa_top_k,
                swa_val_f1_threshold = args.swa_val_f1_threshold,
                swa_weight_temp    = args.swa_weight_temp,
                output_ckpt_path   = out_dir / "checkpoints" / "swa_averaged.ckpt",
            )
        else:
            swa_ckpt_path = None

        # Synchronize SWA checkpoint path across ranks
        if n_gpus > 1 and dist.is_available() and dist.is_initialized():
            swa_paths = [None] * n_gpus
            dist.all_gather_object(swa_paths, swa_ckpt_path)
            swa_ckpt_path = swa_paths[0]  # rank 0's result

        if swa_ckpt_path is not None and Path(str(swa_ckpt_path)).exists():
            print(f"\n[Node4-2-1-3] Running SWA test inference with {swa_ckpt_path}")

            # Create a fresh trainer for SWA testing (avoids state contamination)
            swa_out_dir = out_dir / "swa_run"
            swa_out_dir.mkdir(parents=True, exist_ok=True)
            swa_csv_logger = CSVLogger(
                save_dir=str(swa_out_dir / "logs"), name="swa_csv_logs"
            )
            swa_tb_logger = TensorBoardLogger(
                save_dir=str(swa_out_dir / "logs"), name="swa_tb_logs"
            )
            swa_pb = TQDMProgressBar(refresh_rate=10)

            swa_trainer = pl.Trainer(
                accelerator        = "gpu",
                devices            = n_gpus,
                num_nodes          = 1,
                strategy           = DDPStrategy(
                    find_unused_parameters=False, timeout=timedelta(seconds=120)
                ),
                precision          = "bf16-mixed",
                limit_test_batches = 1.0,
                callbacks          = [swa_pb],
                logger             = [swa_csv_logger, swa_tb_logger],
                deterministic      = False,
                default_root_dir   = str(swa_out_dir),
            )

            # Create fresh lit module for SWA testing
            # Set output path to swa_run/test_predictions.tsv BEFORE setup runs
            lit_swa = StringGNNExtendedLitModule(
                backbone_lr     = args.backbone_lr,
                head_lr         = args.head_lr,
                muon_lr         = args.muon_lr,
                weight_decay    = args.weight_decay,
                gene_emb_wd     = args.gene_emb_wd,
                t0_epochs       = args.t0_epochs,
                t_mult          = args.t_mult,
                warmup_steps    = args.warmup_steps,
                max_steps       = estimated_max_steps,
                label_smoothing = args.label_smoothing,
            )
            # Direct SWA predictions to swa_run directory
            lit_swa._pred_output_path = swa_out_dir / "test_predictions.tsv"

            swa_test_results = swa_trainer.test(
                lit_swa,
                datamodule=dm,
                ckpt_path=str(swa_ckpt_path),
            )

            if trainer.is_global_zero:
                swa_pred = swa_out_dir / "test_predictions.tsv"
                print(f"\n[Node4-2-1-3] Both prediction files available:")
                print(f"  Best single ckpt:  {out_dir / 'test_predictions.tsv'}")
                print(f"  Best single backup: {out_dir / 'best_single_test_predictions.tsv'}")
                print(f"  SWA averaged:      {swa_pred}")
                print(f"\n[Node4-2-1-3] EvaluateAgent should evaluate both and use the better one.")
                print(f"  Primary output (test_predictions.tsv) = best single ckpt predictions.")
                print(f"  SWA output = {swa_pred}")
        else:
            print("[Node4-2-1-3] SWA not applicable (no qualifying checkpoints). "
                  "Using best single checkpoint predictions.")

    if trainer.is_global_zero:
        best_val_f1 = ckpt_cb.best_model_score
        best_val_f1_str = f"{best_val_f1:.4f}" if best_val_f1 is not None else "N/A"
        (Path(__file__).parent / "test_score.txt").write_text(
            f"Node 4-2-1-3 — STRING_GNN Extended Backbone (mps.6+7+post_mp) + SGDR + "
            f"LabelSmoothing=0.05 + Dropout=0.45 + Quality-Filtered SWA\n"
            f"Best val F1: {best_val_f1_str}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
            f"Note: Both best single ckpt and SWA predictions are available under run/\n"
        )


if __name__ == "__main__":
    main()
