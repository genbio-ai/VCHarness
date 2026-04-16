"""
Node 4-2-1-2-1: STRING_GNN Expanded Backbone (mps.6+7+post_mp) + Fixed SWA Golden Zone

Architecture Strategy (differentiating from parent node4-2-1-2):
  Parent (node4-2-1-2): mps.7+post_mp trainable (~67K params), SWA failed due to:
    1. Wrong window (epochs 45-125 — mixed good Cycle3 and bad Cycle4 checkpoints)
    2. Critical SWA test inference bug (trainer.test() without ckpt_path resets model)
    3. Bottlenecked at 0.5075 — plateaued across node4 lineage despite head capacity changes

  This node tries: Expand trainable backbone to mps.6+7+post_mp (~198K params) + fix SWA

Key reasoning:
  The node2 lineage (F1=0.5180) uses mps.6+7+post_mp — the same backbone scope.
  The node4 lineage (0.5075) uses only mps.7+post_mp.
  This extra trainable GCN layer (mps.6) is the primary architectural differentiator
  between the two lineages, and is the most actionable path to breaking the 0.508 ceiling.

Summary of changes from parent (node4-2-1-2):
  1. Expanded backbone: mps.6 added to trainable → mps.6+7+post_mp (~198K vs ~67K)
  2. Fixed SWA golden zone: start=40, end=90 (captures cycles 2-3 peaks only)
  3. Fixed SWA test inference bug: use ckpt_path= in trainer.test()
  4. Stronger regularization: dropout=0.45 (from 0.4), gene_emb_wd=5e-2 (from 1e-2)
  5. Fixed DDP overhead: find_unused_parameters=False
  6. Increased patience=80 (expanded backbone needs more time to converge)
  7. SWA threshold=0.498 (from 0.490) — focused quality filtering

Memory Sources:
  - node2-1-1-1-2-1-1-1-1-1-1 (tree best F1=0.5180): Uses mps.6+7+post_mp → 0.5180
  - node4-2-1-2/memory/feedback.md: SWA golden zone (40-85), threshold=0.498
  - node4-2-1-2/memory/feedback.md: dropout=0.45, gene_emb_wd=5e-2 recommendations
  - node4-2-1-2/memory/feedback.md: find_unused_parameters=False fix
  - collected_memory: class_weights=[2.0,0.5,4.0], focal_gamma=2.0, no label smoothing
  - node1-2-2-3 / node4-2-1: T_0=1200, Muon lr=0.005 proven optimal
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
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
# Proven effective across the tree (node2-1-3, node1-2-2-2-1, node4-2, node2-1-1-1-2-1-1-1-1-1-1)
CLASS_WEIGHTS = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)

# Focal loss gamma: focuses training on hard examples
FOCAL_GAMMA = 2.0


# ─── Focal Loss ───────────────────────────────────────────────────────────────

def focal_cross_entropy(
    logits: torch.Tensor,      # [B, C, L]
    labels: torch.Tensor,      # [B, L] long
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal cross-entropy loss for multi-output 3-class classification.
    Logits: [B, 3, L], labels: [B, L], class_weights: [3]
    """
    B, C, L = logits.shape
    logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)   # [B*L, C]
    labels_flat  = labels.reshape(-1)                         # [B*L]

    ce_loss = F.cross_entropy(
        logits_flat, labels_flat,
        weight=class_weights.to(logits_flat.device),
        reduction="none",
    )  # [B*L]

    with torch.no_grad():
        probs = F.softmax(logits_flat.float(), dim=-1)        # [B*L, C]
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
        pert_ids:    List[str],
        symbols:     List[str],
        node_indices: torch.Tensor,       # [N] long, -1 for unknown
        labels:      Optional[torch.Tensor] = None,  # [N, 6640] long
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
            df  = pd.read_csv(self.data_dir / fname, sep="\t")
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

        graph = torch.load(gnn_dir / "graph_data.pt", weights_only=False)
        self.edge_index  = graph["edge_index"]
        self.edge_weight = graph.get("edge_weight", None)

        n_unknown = sum(
            1 for ds in (self.train_ds, self.val_ds, self.test_ds)
            for ni in ds.node_indices.tolist() if ni == -1
        )
        total = len(self.train_ds) + len(self.val_ds) + len(self.test_ds)
        print(f"[Node4-2-1-2-1] {n_unknown}/{total} samples not in STRING_GNN "
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

class StringGNNExpandedBilinearModel(nn.Module):
    """
    STRING_GNN (expanded partial fine-tuning: mps.6+7 + post_mp) +
    4-layer Reduced Residual MLP + Rank-256 Bilinear Interaction Head.

    Key Change from Parent (node4-2-1-2):
      - EXPANDED backbone: mps.6+7+post_mp trainable (~198K params)
        vs parent's mps.7+post_mp (~67K params)
      - Frozen cache now stores output after mps.5 (was after mps.6)
      - Stronger regularization: dropout=0.45, gene_emb_wd=5e-2
      - This aligns with the tree best lineage (node2-1-1-1-2-1-1-1-1-1-1, F1=0.5180)
        which uses the same mps.6+7+post_mp configuration

    Architecture:
      1. Pre-computed frozen backbone (mps.0-5 + emb) stored as buffer
      2. Trainable tail: mps.6 + mps.7 + post_mp (~198K params)
      3. Fallback embedding for unknown genes
      4. 4× ResidualBlock(hidden=256, expand=4, dropout=0.45)
      5. Bilinear head: rank-256 decomposition → [B, 3, 6640]
    """

    def __init__(
        self,
        edge_index:  torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        gnn_dim:     int = GNN_DIM,
        n_genes_out: int = N_GENES_OUT,
        n_classes:   int = N_CLASSES,
        hidden_dim:  int = 256,       # Unchanged from parent (proven capacity)
        n_layers:    int = 4,         # Unchanged from parent
        expand:      int = 4,
        bilinear_rank: int = 256,     # Unchanged from parent
        dropout:     float = 0.45,    # INCREASED from 0.40 (stronger regularization)
    ):
        super().__init__()
        self.gnn_dim      = gnn_dim
        self.n_classes    = n_classes
        self.n_genes_out  = n_genes_out
        self.bilinear_rank = bilinear_rank

        # ── Backbone ──────────────────────────────────────────────────────────
        full_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        # Register frozen graph tensors
        self.register_buffer("edge_index",  edge_index)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        # Split backbone: freeze mps.0-5, train mps.6+7+post_mp
        # KEY CHANGE: mps.6 is now trainable (was frozen in parent node4-2-1-2)
        self.emb     = full_gnn.emb                                     # Embedding — frozen
        self.mps_0_5 = nn.ModuleList([full_gnn.mps[i] for i in range(6)])  # frozen (mps.0-5)
        self.mps_6   = full_gnn.mps[6]    # trainable (NEW: added vs parent)
        self.mps_7   = full_gnn.mps[7]    # trainable
        self.post_mp = full_gnn.post_mp   # trainable

        # Freeze emb + mps.0-5
        for p in self.emb.parameters():
            p.requires_grad_(False)
        for layer in self.mps_0_5:
            for p in layer.parameters():
                p.requires_grad_(False)
        # mps.6, mps.7, post_mp are trainable (requires_grad=True by default)

        # Fallback embedding for genes absent from STRING_GNN
        self.fallback_emb = nn.Parameter(torch.randn(gnn_dim) * 0.02)

        # ── Head ──────────────────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.LayerNorm(gnn_dim),
            nn.Linear(gnn_dim, hidden_dim, bias=False),
        )

        # 4× ResidualBlock (hidden=256, dropout=0.45 — slightly higher regularization)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Bilinear decomposition: 256 → 3×256 = 768
        self.fc_bilinear = nn.Linear(hidden_dim, n_classes * bilinear_rank, bias=False)

        # Output gene embedding: 6640×256 = 1.7M params
        # Separate optimizer group with strong weight decay (5e-2, increased from 1e-2)
        self.out_gene_emb = nn.Embedding(n_genes_out, bilinear_rank)
        nn.init.xavier_uniform_(self.out_gene_emb.weight)

        # Cache for precomputed frozen intermediate embeddings (output after mps.5)
        self._frozen_emb_cache: Optional[torch.Tensor] = None

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Node4-2-1-2-1] Trainable: {n_trainable:,} / {n_total:,} params "
              f"({100*n_trainable/n_total:.2f}%) — EXPANDED backbone (mps.6+7+post_mp)")
        print(f"[Node4-2-1-2-1] Head config: 4×ResBlock(hidden=256, expand=4, dropout=0.45) + rank-256 bilinear")
        print(f"[Node4-2-1-2-1] Backbone: mps.0-5 frozen, mps.6+7+post_mp trainable (~198K params)")

    def _compute_backbone_embs(self) -> torch.Tensor:
        """
        Run partial forward:
          emb.weight → mps.0-5 (frozen, cached) → mps.6 (trainable) → mps.7 (trainable)
          → post_mp (trainable)
        Returns node_emb [N_nodes, 256].

        Uses frozen cache for mps.0-5 output to save compute during training.
        The cache stores output after mps.5 (frozen), so gradients flow from mps.6 onward.
        """
        ei = self.edge_index
        ew = self.edge_weight

        # Compute/use cached frozen intermediate (output after mps.5)
        if self._frozen_emb_cache is None:
            x = self.emb.weight  # [N, 256]
            for layer in self.mps_0_5:
                x = layer(x, ei, ew)
            self._frozen_emb_cache = x.detach()

        x = self._frozen_emb_cache  # [N, 256] — no grad, frozen at mps.5 output

        # Trainable tail: mps.6 → mps.7 → post_mp
        x = self.mps_6(x, ei, ew)   # trainable (NEW)
        x = self.mps_7(x, ei, ew)   # trainable
        x = self.post_mp(x)          # trainable
        return x

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_indices: [B] long — STRING_GNN node indices (-1 = unknown)
        Returns:
            logits: [B, 3, 6640]
        """
        node_emb = self._compute_backbone_embs()  # [N_nodes, 256]

        # Extract perturbed gene embeddings; handle unknowns with learned fallback
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

        # Input projection (LayerNorm + Linear: 256→256)
        h = self.input_proj(pert_emb)   # [B, hidden_dim=256]

        # Residual blocks (4 layers, hidden=256, dropout=0.45)
        for block in self.res_blocks:
            h = block(h)                # [B, hidden_dim=256]

        # Bilinear decomposition: [B, 256] → [B, 768] → [B, 3, 256]
        blin = self.fc_bilinear(h).view(-1, self.n_classes, self.bilinear_rank)

        # Output gene embeddings: [6640, 256]
        out_embs = self.out_gene_emb.weight

        # [B, 3, 256] @ [256, 6640] → [B, 3, 6640]
        logits = torch.matmul(blin, out_embs.T)
        return logits


# ─── SWA Utilities ────────────────────────────────────────────────────────────

def parse_val_f1_from_checkpoint_path(ckpt_path: Path) -> float:
    """
    Robustly parse val_f1 from PL checkpoint filenames using regex.
    Handles double-prefixed format: 'periodic-epoch=0025-val_f1=val_f1=0.4923.ckpt'
    """
    filename = ckpt_path.name
    matches = re.findall(r'val_f1=(\d+\.\d+)', filename)
    if matches:
        return float(matches[-1])  # Last match = actual float value
    return 0.0


def perform_quality_filtered_swa(
    ckpt_dir: Path,
    swa_start_epoch: int = 40,
    swa_end_epoch:   int = 90,
    swa_top_k: int = 10,
    swa_val_f1_threshold: float = 0.498,
    swa_weight_temperature: float = 3.0,
) -> Optional[Dict]:
    """
    Post-hoc quality-filtered exponentially-weighted SWA.

    KEY FIX from parent node4-2-1-2:
    - Now respects swa_end_epoch to focus on the 'golden zone' (cycles 2-3 peaks)
    - Higher quality threshold (0.498 vs 0.490)
    - Avoids mixing high-quality early checkpoints with weaker late-cycle checkpoints

    Collects periodic checkpoints in the golden zone [swa_start_epoch, swa_end_epoch],
    filters by val_f1 >= threshold, selects top-K, averages exponentially.

    Returns averaged state dict, or None if insufficient checkpoints found.
    """
    # Collect all periodic checkpoints within the golden zone
    all_periodic = []
    for ckpt_p in ckpt_dir.glob("periodic-*.ckpt"):
        epoch_match = re.search(r'epoch=(\d+)', ckpt_p.name)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            val_f1 = parse_val_f1_from_checkpoint_path(ckpt_p)
            all_periodic.append((epoch_num, val_f1, ckpt_p))

    if not all_periodic:
        print("[SWA] No periodic checkpoints found — skipping SWA")
        return None

    # Filter by golden zone epoch range
    in_zone = [(ep, f1, p) for ep, f1, p in all_periodic
               if swa_start_epoch <= ep <= swa_end_epoch]

    print(f"[SWA] Periodic checkpoints total: {len(all_periodic)}, "
          f"in golden zone [ep={swa_start_epoch}..{swa_end_epoch}]: {len(in_zone)}")

    # Filter by quality threshold
    qualified = [(ep, f1, p) for ep, f1, p in in_zone if f1 >= swa_val_f1_threshold]

    print(f"[SWA] After quality filter (f1>={swa_val_f1_threshold}): {len(qualified)}")

    if len(qualified) == 0:
        # Fallback 1: relax threshold, keep golden zone
        qualified = in_zone
        print(f"[SWA] Fallback 1: using all {len(qualified)} golden-zone checkpoints "
              f"(relaxed threshold)")

    if len(qualified) == 0:
        # Fallback 2: use all periodic checkpoints regardless of epoch
        qualified = all_periodic
        print(f"[SWA] Fallback 2: using all {len(qualified)} periodic checkpoints")

    if len(qualified) == 0:
        print("[SWA] No qualifying checkpoints — skipping SWA")
        return None

    # Sort by val_f1 descending, take top-K
    qualified.sort(key=lambda x: x[1], reverse=True)
    selected = qualified[:swa_top_k]

    print(f"[SWA] Selecting top-{len(selected)} checkpoints:")
    for rank_i, (ep, f1, p) in enumerate(selected):
        print(f"  rank {rank_i+1}: epoch={ep}, val_f1={f1:.4f}, path={p.name}")

    # Exponential weights: w_i = exp(-temp * rank_i / (N-1))
    N = len(selected)
    if N == 1:
        weights = [1.0]
    else:
        weights = [np.exp(-swa_weight_temperature * i / (N - 1)) for i in range(N)]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    print(f"[SWA] Weight distribution (sum=1.0):")
    for rank_i, ((ep, f1, p), w) in enumerate(zip(selected, weights)):
        print(f"  rank {rank_i+1}: weight={w:.4f}, epoch={ep}, val_f1={f1:.4f}")

    # Load checkpoints and compute weighted average
    avg_state_dict = None
    for (ep, f1, p), w in zip(selected, weights):
        try:
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
            if avg_state_dict is None:
                avg_state_dict = {k: v.float() * w for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    if k in avg_state_dict:
                        avg_state_dict[k] = avg_state_dict[k] + v.float() * w
                    else:
                        avg_state_dict[k] = v.float() * w
        except Exception as e:
            print(f"[SWA] Warning: Failed to load {p.name}: {e}")

    if avg_state_dict is None:
        print("[SWA] All checkpoints failed to load — skipping SWA")
        return None

    print(f"[SWA] Successfully created SWA state dict from {N} checkpoints")
    return avg_state_dict


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

class StringGNNExpandedLitModule(pl.LightningModule):

    def __init__(
        self,
        backbone_lr:   float = 1e-5,
        head_lr:       float = 5e-4,
        muon_lr:       float = 0.005,
        weight_decay:  float = 1e-3,
        gene_emb_wd:   float = 5e-2,    # INCREASED from 1e-2 (stronger L2 on primary overfitting source)
        gene_emb_lr:   float = 5e-4,    # Same as head_lr (proven in parent, enables Cycle 2 spike)
        t0_steps:      int   = 1200,    # Extended warm restart (proven optimal in tree)
        warmup_steps:  int   = 100,
        max_steps:     int   = 10000,
        focal_gamma:   float = FOCAL_GAMMA,
        # SWA parameters (golden zone focused)
        swa_start_epoch:      int   = 40,   # INCREASED from 20 (skip Cycle 1 warm-up phase)
        swa_end_epoch:        int   = 90,   # NEW: stop at cycle 3 peak (~epoch 80-90)
        swa_every_n_epochs:   int   = 5,
        swa_top_k:            int   = 10,   # Slightly reduced from 12 (tighter pool)
        swa_val_f1_threshold: float = 0.498, # INCREASED from 0.490 (stricter quality)
        swa_weight_temp:      float = 3.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str]          = []
        self._test_symbols:  List[str]          = []
        self._test_labels:   List[torch.Tensor] = []
        self._best_val_f1: float = 0.0

    def setup(self, stage=None):
        dm = self.trainer.datamodule if self.trainer is not None else None
        if dm is None:
            raise RuntimeError("DataModule must be attached to the trainer.")

        self.model = StringGNNExpandedBilinearModel(
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

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return focal_cross_entropy(
            logits, labels,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
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
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        if f1 > self._best_val_f1:
            self._best_val_f1 = f1
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
            self.print(f"[Node4-2-1-2-1] Deduplicating: {len(all_pert)} → {len(keep)}")
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
            self.print(f"[Node4-2-1-2-1] Saved predictions → {pred_path}")
            if all_labels.any():
                f1 = compute_per_gene_f1(all_probs.numpy(), all_labels.numpy())
                self.print(f"[Node4-2-1-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear(); self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Parameter groups:
        # 1. Backbone trainable (mps.6+7 + post_mp) — AdamW at backbone_lr
        # 2. Head 2D weight matrices (ResBlocks, bilinear) — Muon
        # 3. out_gene_emb — AdamW with gene_emb_lr=5e-4 and gene_emb_wd=5e-2 (stronger)
        # 4. Other head params (norms, biases, input_proj, fallback_emb) — AdamW at head_lr

        # KEY CHANGE: Include mps.6 in backbone_params
        backbone_params = (
            list(self.model.mps_6.parameters()) +    # NEW: added mps.6
            list(self.model.mps_7.parameters()) +
            list(self.model.post_mp.parameters())
        )
        backbone_param_ids = {id(p) for p in backbone_params}

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
            # Muon: 2D weight matrices in hidden layers
            if (param.ndim >= 2
                    and "input_proj.1" not in name
                    and "fallback_emb" not in name):
                head_2d_matrices.append(param)
            else:
                head_other.append(param)

        param_groups = [
            # Backbone (mps.6+7 + post_mp) — AdamW at backbone_lr=1e-5
            {
                "params":       backbone_params,
                "use_muon":     False,
                "lr":           hp.backbone_lr,
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.weight_decay,
            },
            # Head 2D matrices — Muon (excellent for deep MLP optimization)
            {
                "params":       head_2d_matrices,
                "use_muon":     True,
                "lr":           hp.muon_lr,
                "momentum":     0.95,
                "weight_decay": hp.weight_decay,
            },
            # out_gene_emb — AdamW with STRONG weight_decay (5e-2) and LR=5e-4
            # KEY CHANGE: gene_emb_wd increased 5× (1e-2 → 5e-2)
            # Stronger L2 on the primary overfitting source (1.7M param matrix)
            {
                "params":       gene_emb_params,
                "use_muon":     False,
                "lr":           hp.gene_emb_lr,
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.gene_emb_wd,
            },
            # Other head params — AdamW at head_lr
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

        # Cosine annealing with warm restarts (T_0=1200 steps, proven optimal)
        warmup = hp.warmup_steps
        T_0    = hp.t0_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / float(max(1, warmup))
            step_after = step - warmup
            cycle_len  = T_0
            cycle_pos  = step_after % cycle_len
            progress   = float(cycle_pos) / float(cycle_len)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

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


# ─── Periodic Checkpoint Callback ─────────────────────────────────────────────

class PeriodicCheckpointCallback(pl.Callback):
    """
    Save a checkpoint every N epochs in the golden zone [start_epoch, end_epoch].

    KEY CHANGE from parent: Added end_epoch to stop saving after the golden zone.
    This prevents contaminating the SWA pool with post-peak Cycle 4 checkpoints.
    """

    def __init__(
        self,
        every_n_epochs: int,
        dirpath: str,
        start_epoch: int = 40,
        end_epoch: int = 90,    # NEW: golden zone ceiling
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.dirpath        = Path(dirpath)
        self.start_epoch    = start_epoch
        self.end_epoch      = end_epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # Only save within the golden zone
        if epoch < self.start_epoch or epoch > self.end_epoch:
            return
        if (epoch - self.start_epoch) % self.every_n_epochs != 0:
            return

        val_f1 = trainer.callback_metrics.get("val_f1", 0.0)
        if hasattr(val_f1, "item"):
            val_f1 = val_f1.item()

        self.dirpath.mkdir(parents=True, exist_ok=True)
        fname = f"periodic-epoch={epoch:04d}-val_f1={val_f1:.4f}.ckpt"
        ckpt_path = self.dirpath / fname
        trainer.save_checkpoint(str(ckpt_path))
        if trainer.is_global_zero:
            print(f"[PeriodicCkpt] Saved epoch={epoch}, val_f1={val_f1:.4f} → {fname}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 4-2-1-2-1 — STRING_GNN Expanded Backbone (mps.6+7+post_mp) + Fixed SWA Golden Zone"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--backbone-lr",       type=float, default=1e-5)
    p.add_argument("--head-lr",           type=float, default=5e-4)
    p.add_argument("--muon-lr",           type=float, default=0.005)
    p.add_argument("--weight-decay",      type=float, default=1e-3)
    p.add_argument("--gene-emb-wd",       type=float, default=5e-2,
                   help="Weight decay for out_gene_emb — INCREASED to 5e-2 (from 1e-2)")
    p.add_argument("--gene-emb-lr",       type=float, default=5e-4,
                   help="LR for out_gene_emb — same as head_lr (enables Cycle 2 spike)")
    p.add_argument("--micro-batch-size",  type=int,   default=8)
    p.add_argument("--global-batch-size", type=int,   default=32)
    p.add_argument("--max-epochs",        type=int,   default=250)
    p.add_argument("--patience",          type=int,   default=80,
                   help="INCREASED from 60 — expanded backbone needs more convergence time")
    p.add_argument("--t0-steps",          type=int,   default=1200,
                   help="Warm restart T_0 (proven at tree best node1-2-2-3, node4-2-1)")
    p.add_argument("--warmup-steps",      type=int,   default=100)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    # SWA golden zone parameters
    p.add_argument("--swa-start-epoch",      type=int,   default=40,
                   help="INCREASED from 20 — skip Cycle 1 warm-up, focus on Cycle 2+ peaks")
    p.add_argument("--swa-end-epoch",        type=int,   default=90,
                   help="NEW: stop saving periodic checkpoints after golden zone")
    p.add_argument("--swa-every-n-epochs",   type=int,   default=5)
    p.add_argument("--swa-top-k",            type=int,   default=10,
                   help="Top checkpoints to average (reduced to 10, tighter golden pool)")
    p.add_argument("--swa-val-f1-threshold", type=float, default=0.498,
                   help="INCREASED from 0.490 — strict quality filter for golden zone")
    p.add_argument("--swa-weight-temp",      type=float, default=3.0)
    p.add_argument("--skip-swa",             action="store_true", default=False)
    # Debug
    p.add_argument("--debug-max-step",    type=int,   default=None)
    p.add_argument("--fast-dev-run",      action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    steps_per_epoch = int(np.ceil(1416 / args.micro_batch_size))
    estimated_max_steps = args.max_epochs * steps_per_epoch // accum

    dm  = StringGNNDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = StringGNNExpandedLitModule(
        backbone_lr  = args.backbone_lr,
        head_lr      = args.head_lr,
        muon_lr      = args.muon_lr,
        weight_decay = args.weight_decay,
        gene_emb_wd  = args.gene_emb_wd,
        gene_emb_lr  = args.gene_emb_lr,
        t0_steps     = args.t0_steps,
        warmup_steps = args.warmup_steps,
        max_steps    = estimated_max_steps,
        swa_start_epoch      = args.swa_start_epoch,
        swa_end_epoch        = args.swa_end_epoch,
        swa_every_n_epochs   = args.swa_every_n_epochs,
        swa_top_k            = args.swa_top_k,
        swa_val_f1_threshold = args.swa_val_f1_threshold,
        swa_weight_temp      = args.swa_weight_temp,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath   = str(ckpt_dir),
        filename  = "best-{epoch:04d}-{val_f1:.4f}",
        monitor   = "val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max",
                            patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    # Periodic checkpoint callback for SWA — golden zone only
    periodic_cb = PeriodicCheckpointCallback(
        every_n_epochs = args.swa_every_n_epochs,
        dirpath        = str(ckpt_dir),
        start_epoch    = args.swa_start_epoch,
        end_epoch      = args.swa_end_epoch,
    )

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

    # Use find_unused_parameters=True: fallback_emb is only used for batches with unknown
    # node IDs (~6.3% of data), so some batches don't use it. With find_unused=False,
    # DDP deadlocks waiting for a gradient hook that never fires for that parameter.
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
        gradient_clip_val    = 1.0,
        limit_train_batches  = limit_train,
        limit_val_batches    = limit_val,
        limit_test_batches   = limit_test,
        val_check_interval   = (
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps = 2,
        callbacks     = [ckpt_cb, es_cb, lr_cb, pb_cb, periodic_cb],
        logger        = [csv_logger, tb_logger],
        log_every_n_steps = 10,
        deterministic = False,   # nll_loss2d has no deterministic CUDA impl
        default_root_dir = str(out_dir),
        fast_dev_run  = fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)

    if args.fast_dev_run or args.debug_max_step is not None:
        trainer.test(lit, datamodule=dm)
    else:
        # Production: attempt SWA on golden zone, fall back to best single checkpoint
        swa_ckpt_path = ckpt_dir / "swa_averaged.ckpt"
        swa_applied   = False

        if not args.skip_swa and trainer.is_global_zero:
            print("\n[Node4-2-1-2-1] Attempting golden-zone quality-filtered SWA...")
            swa_sd = perform_quality_filtered_swa(
                ckpt_dir              = ckpt_dir,
                swa_start_epoch       = args.swa_start_epoch,
                swa_end_epoch         = args.swa_end_epoch,
                swa_top_k             = args.swa_top_k,
                swa_val_f1_threshold  = args.swa_val_f1_threshold,
                swa_weight_temperature = args.swa_weight_temp,
            )
            if swa_sd is not None:
                torch.save({"state_dict": swa_sd}, str(swa_ckpt_path))
                print(f"[Node4-2-1-2-1] SWA checkpoint saved → {swa_ckpt_path}")
                swa_applied = True

        # Synchronize SWA decision across all ranks to avoid deadlock
        if n_gpus > 1 and dist.is_available() and dist.is_initialized():
            swa_flag = [swa_applied]
            dist.broadcast_object_list(swa_flag, src=0)
            swa_applied = bool(swa_flag[0])

        if swa_applied and swa_ckpt_path.exists():
            # KEY FIX: Use ckpt_path= parameter so Lightning loads AFTER setup()
            # Parent bug: manually called lit.load_state_dict() BEFORE trainer.test(),
            # but trainer.test() calls setup() which re-initializes self.model,
            # overwriting the SWA weights → self-computed F1=0.2190 (random model)
            # Fix: pass ckpt_path= to trainer.test() so Lightning loads AFTER setup()
            if trainer.is_global_zero:
                print("[Node4-2-1-2-1] Running test inference with SWA model (ckpt_path=)...")
            try:
                trainer.test(lit, datamodule=dm, ckpt_path=str(swa_ckpt_path))
                if trainer.is_global_zero:
                    print("[Node4-2-1-2-1] SWA test inference completed successfully.")
            except Exception as e:
                if trainer.is_global_zero:
                    print(f"[Node4-2-1-2-1] SWA test failed ({e}), falling back to best checkpoint...")
                trainer.test(lit, datamodule=dm, ckpt_path="best")
        else:
            if trainer.is_global_zero:
                print("[Node4-2-1-2-1] No SWA applied, using best single checkpoint for test...")
            trainer.test(lit, datamodule=dm, ckpt_path="best")

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 4-2-1-2-1 — STRING_GNN Expanded Backbone (mps.6+7+post_mp) + Fixed SWA Golden Zone\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
