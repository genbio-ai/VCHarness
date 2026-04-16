"""
Node 3-1-1-3-1-1-1 – STRING_GNN FROZEN Backbone + Rank-512 Bilinear Head + Muon + Fixed SWA

Key changes over parent node3-1-1-3-1-1 (F1=0.5036):
  1. [CRITICAL] Freeze backbone completely: eliminates the SWA catastrophic failure
     (F1=0.2453 in parent due to averaging checkpoints with diverged backbone weights
     across SGDR cycles). With frozen backbone, all checkpoints have identical backbone
     weights, making SWA coherent. This aligns with the best MCTS nodes (F1=0.5180-0.5182)
     which all use frozen backbone.
  2. [CRITICAL] Fix SWA off-by-one: Store val_f1 in LightningModule.on_validation_epoch_end
     as self._last_val_f1, then read pl_module._last_val_f1 in the callback instead of
     trainer.callback_metrics["val_f1"] which still has the previous epoch's value.
  3. [HIGH] Change SGDR to epoch-based T_0=20 epochs (like best node2 family): best nodes
     use T_0=20 epochs directly with CosineAnnealingWarmRestarts scheduler in epoch mode.
     This allows the model to train for more cycles before overfitting kicks in.
  4. [HIGH] Increase patience to 200 (from 120): aligns with best node2 family, allowing
     more SGDR cycles and a richer SWA pool.
  5. [MEDIUM] SWA params: threshold=0.490 (conservative, matching successful nodes), top_k=15,
     temperature=3.0 (proven configuration from node2-1-1-1-2-1-1-1-1-1-1, F1=0.5180).
  6. [MEDIUM] Reduce weight decay to 1.5e-3 (from 3e-3): best frozen backbone nodes use 1.5e-3.
  7. [MEDIUM] Increase dropout to 0.50 (from 0.45): slightly stronger regularization for
     head-only training on 1,416 samples.

Architecture:
  - STRING_GNN backbone: FULLY FROZEN (all layers precomputed as static buffer)
    → no trainable backbone parameters, identical backbone across all checkpoints
  - Rank-512 deep residual bilinear MLP head (6 ResBlocks, hidden=512, dropout=0.50)
    Bilinear interaction: [B, 512] pert_emb @ [6640, 512] out_gene_emb -> [B, 6640] logits
    Output logit shape: [B, 3, 6640]
  - Muon optimizer (lr=0.005) for 2D head matrices (ResBlock weights)
    AdamW (lr=5e-4) for 1D head params (biases, LayerNorms, embeddings)
  - Focal loss (gamma=2.0) with class weights [2.0, 0.5, 4.0] + label smoothing 0.05
  - SGDR: CosineAnnealingWarmRestarts (T_0=20 epochs, T_mult=2, epoch-based)
  - SWA: quality-filtered top-15 exponentially-weighted checkpoint averaging
    (threshold=0.490, temp=3.0) — coherent because backbone is frozen
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
import heapq
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
from muon import MuonWithAuxAdam

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_DIM        = 256      # STRING_GNN hidden size
BILINEAR_RANK  = 512      # Bilinear interaction rank

# Class weights: [down(-1), neutral(0), up(+1)] after +1 shift → {0,1,2}
# Empirically proven in node1-2-2-3 family (F1=0.5101) and best nodes
CLASS_WEIGHTS = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)


# ─── Metric (matches calc_metric.py exactly) ──────────────────────────────────

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
    """Focal cross-entropy loss for multi-class classification."""

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
            logits:  [B, C, G] unnormalized logits (C=3 classes, G=6640 genes)
            targets: [B, G] class indices in {0, 1, 2}
        Returns:
            scalar loss
        """
        B, C, G = logits.shape
        logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        targets_flat = targets.reshape(-1)                      # [B*G]

        log_probs = F.log_softmax(logits_flat, dim=1)           # [B*G, C]
        probs     = torch.exp(log_probs)                        # [B*G, C]

        target_log_prob = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        target_prob     = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - target_prob).pow(self.gamma)

        if self.weight is not None:
            class_w = self.weight.to(logits.device)[targets_flat]
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

        weighted_loss = focal_weight * class_w * loss_per_pos
        denom = class_w.sum().clamp(min=1.0)
        return weighted_loss.sum() / denom


# ─── Bilinear MLP Head ────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual block for deep MLP head."""

    def __init__(self, hidden_dim: int, expand: int = 4, dropout: float = 0.50):
        super().__init__()
        inner = hidden_dim * expand
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.net(self.norm(x)))


class GNNBilinearHead(nn.Module):
    """
    Deep residual MLP + bilinear interaction head for DEG prediction.

    Architecture:
      pert_emb [B, 256] → project_in → [B, hidden_dim]
      → 6 x ResidualBlock(hidden_dim, expand=4, dropout)
      → project_to_rank [B, rank] (bilinear rank)
      → bilinear: [B, rank] × out_gene_emb[6640, rank]^T → [B, 6640] per-class logits
      Combined: [B, 3] class_proj × [6640, rank] → reshaped → [B, 3, 6640]

    The bilinear interaction captures gene-gene interaction patterns by projecting
    the perturbation embedding to the same space as learnable output gene embeddings.
    """

    def __init__(
        self,
        input_dim:    int = GNN_DIM,
        hidden_dim:   int = 512,
        n_layers:     int = 6,
        bilinear_rank: int = BILINEAR_RANK,
        n_genes_out:  int = N_GENES_OUT,
        n_classes:    int = N_CLASSES,
        dropout:      float = 0.50,
    ):
        super().__init__()

        # Project from 256-dim GNN embedding to hidden_dim
        self.project_in = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )

        # Deep residual MLP
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Project to bilinear rank (one per class)
        self.to_rank = nn.Linear(hidden_dim, bilinear_rank * n_classes)

        # Learnable output gene embeddings: [n_genes_out, bilinear_rank]
        self.out_gene_emb = nn.Embedding(n_genes_out, bilinear_rank)
        nn.init.normal_(self.out_gene_emb.weight, std=0.02)

        self.n_classes    = n_classes
        self.bilinear_rank = bilinear_rank
        self.n_genes_out  = n_genes_out

    def forward(self, pert_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pert_emb: [B, 256] perturbation embedding from STRING_GNN
        Returns:
            logits: [B, 3, 6640]
        """
        x = self.project_in(pert_emb)                            # [B, hidden_dim]
        for block in self.res_blocks:
            x = block(x)
        x = self.final_norm(x)                                   # [B, hidden_dim]

        # Project to [B, n_classes * bilinear_rank]
        x = self.to_rank(x)                                      # [B, n_classes * rank]
        x = x.view(-1, self.n_classes, self.bilinear_rank)       # [B, n_classes, rank]

        # Bilinear interaction: [B, n_classes, rank] @ [rank, n_genes_out] → [B, n_classes, n_genes_out]
        gene_emb = self.out_gene_emb.weight                      # [n_genes_out, rank]
        logits = torch.matmul(x, gene_emb.t())                   # [B, n_classes, n_genes_out]

        return logits  # [B, 3, 6640]


# ─── STRING_GNN Backbone Module (FULLY FROZEN) ────────────────────────────────

class StringGNNBackbone(nn.Module):
    """
    STRING_GNN backbone — fully frozen (all parameters frozen, full graph precomputed).

    KEY CHANGE from parent (node3-1-1-3-1-1):
    - Parent: partial fine-tuning of mps.6, mps.7, post_mp (~198K trainable params)
    - This node: FULLY FROZEN — all 8 GCN layers + embedding table + post_mp are frozen
      and precomputed once as a static buffer [18870, 256]

    Why frozen is critical:
    - SWA averaging of checkpoints from different SGDR cycles was catastrophic (F1=0.2453)
      because backbone+head weights co-evolved independently across cycles
    - With frozen backbone, all checkpoints have IDENTICAL backbone weights
    - SWA averages only the head weights, which are trained in the same representational space
    - Best MCTS nodes (F1=0.5180-0.5182) all use frozen backbone — this alignment is key

    The forward pass runs the full 8-layer GNN once during setup and caches the result.
    """

    def __init__(self):
        super().__init__()
        # Load STRING_GNN model
        self.model = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        graph = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", weights_only=False)
        self.register_buffer("edge_index",  graph["edge_index"])
        ew = graph.get("edge_weight", None)
        if ew is not None:
            self.register_buffer("edge_weight", ew)
        else:
            self.edge_weight = None

        # FREEZE ALL parameters (fully frozen backbone)
        for p in self.model.parameters():
            p.requires_grad = False

        # full_embs_buffer will be registered via register_buffer() during setup()
        # shape once registered: [18870, 256]

    def precompute_full_embs(self, device: torch.device):
        """
        Precompute the FULL GNN forward pass (all 8 layers) and cache as a static buffer.
        This is called once during setup. The full embedding is reused every batch.

        The official STRING_GNN forward uses residual connections:
            x = mp(x, edge_index, edge_weight) + x  (for each layer)
            x = post_mp(x)
        We apply this exactly to get consistent outputs.
        """
        self.model.eval()
        edge_index  = self.edge_index.to(device)
        edge_weight = self.edge_weight.to(device) if self.edge_weight is not None else None

        with torch.no_grad():
            # Get initial node embeddings from embedding table
            x = self.model.emb.weight.to(device)  # [18870, 256]

            # Run ALL 8 GCN layers WITH residual connections (matching official forward)
            for i in range(8):
                x = self.model.mps[i](x, edge_index, edge_weight) + x

            # Apply final projection
            x = self.model.post_mp(x)  # [18870, 256]

        # Store as buffer (idempotent: if already registered, just update the data)
        full_embs = x.detach().cpu()
        if "full_embs_buffer" not in self._buffers:
            self.register_buffer("full_embs_buffer", full_embs)
        else:
            self.full_embs_buffer.copy_(full_embs)

    def get_embeddings(self) -> torch.Tensor:
        """
        Return the precomputed full node embeddings (static, no gradient).

        Returns:
            node_embs: [18870, 256] fully computed node embeddings
        """
        return self.full_embs_buffer  # [18870, 256] — already on correct device


# ─── Full Model ───────────────────────────────────────────────────────────────

class StringGNNBilinearModel(nn.Module):
    """
    STRING_GNN FULLY FROZEN backbone + Rank-512 deep bilinear MLP head.

    KEY DIFFERENCE from parent:
    - Parent had ~198K trainable backbone params (mps.6/7/post_mp)
    - This node has 0 trainable backbone params (fully frozen)
    - All trainable params (~16.9M) are in the head

    Total trainable parameters:
      - Backbone: 0 (all frozen)
      - Head (project_in + 6 ResBlocks + to_rank + out_gene_emb): ~16.9M
      - OOV embedding: 256
      - Total: ~16.9M trainable parameters
    """

    def __init__(
        self,
        n_genes_out:   int   = N_GENES_OUT,
        n_classes:     int   = N_CLASSES,
        hidden_dim:    int   = 512,
        bilinear_rank: int   = BILINEAR_RANK,
        n_layers:      int   = 6,
        dropout:       float = 0.50,
    ):
        super().__init__()
        self.backbone = StringGNNBackbone()
        self.head     = GNNBilinearHead(
            input_dim=GNN_DIM,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            bilinear_rank=bilinear_rank,
            n_genes_out=n_genes_out,
            n_classes=n_classes,
            dropout=dropout,
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Node3-1-1-3-1-1-1] Trainable: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.2f}%)")

    def get_node_embs(self) -> torch.Tensor:
        """
        Get the precomputed static node embeddings from the frozen backbone.
        No gradient flows through this — it's a buffer lookup.

        Returns:
            node_embs: [18870, 256] — fully precomputed node embeddings
        """
        return self.backbone.get_embeddings()  # [18870, 256]


# ─── SWA: Quality-Filtered Exponentially-Weighted Checkpoint Averaging ─────────

class SWAManager:
    """
    Quality-filtered, exponentially-weighted checkpoint averaging.

    Maintains a max-heap of top-k checkpoints by val_f1. During inference,
    loads and averages checkpoint weights using exponential weights based on rank,
    controlled by temperature parameter (higher temp = sharper weight concentration
    toward the best checkpoint).

    This implements the strategy proven in:
    - node2-1-1-1-2-1-1-1-1-1-1 (F1=0.5180): top-15, temp=3.0, threshold=0.497 → +0.0065 SWA gain
    - node2-1-1-1-2-1-1-1-1-1-1-1-1 (F1=0.5182): top-25, temp=2.5 → +0.0034 SWA gain

    With a FROZEN backbone (this node), SWA is coherent because backbone weights
    are identical across all checkpoints. Only head weights vary, and they are trained
    in the same representational space throughout all SGDR cycles.
    """

    def __init__(
        self,
        top_k:         int   = 15,
        threshold:     float = 0.490,
        temperature:   float = 3.0,
        ckpt_dir:      str   = "run/checkpoints",
    ):
        self.top_k       = top_k
        self.threshold   = threshold
        self.temperature = temperature
        self.ckpt_dir    = Path(ckpt_dir)
        # List of (val_f1, ckpt_path) for qualifying checkpoints
        self._pool: List[Tuple[float, str]] = []

    def add_checkpoint(self, val_f1: float, ckpt_path: str):
        """Add a checkpoint to the pool if it meets the quality threshold."""
        if val_f1 >= self.threshold:
            self._pool.append((val_f1, ckpt_path))
            # Keep sorted by val_f1 descending
            self._pool.sort(key=lambda x: x[0], reverse=True)

    def get_top_k(self) -> List[Tuple[float, str]]:
        """Return the top-k checkpoints by val_f1."""
        return self._pool[:self.top_k]

    def compute_weights(self, scores: List[float]) -> np.ndarray:
        """
        Compute exponential weights for top-k checkpoints.
        weights[i] ∝ exp(temperature * score[i])
        This concentrates weight on better checkpoints when temp is high.
        """
        scores_arr = np.array(scores, dtype=np.float64)
        # Normalize scores to [0,1] range before exponentiation for stability
        if scores_arr.max() > scores_arr.min():
            scores_norm = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min())
        else:
            scores_norm = np.ones_like(scores_arr)
        log_weights = self.temperature * scores_norm
        log_weights -= log_weights.max()  # numerical stability
        weights = np.exp(log_weights)
        return weights / weights.sum()

    def apply_swa(self, model_state: dict, ckpt_paths_scores: List[Tuple[float, str]], device: str = "cpu") -> dict:
        """
        Perform SWA: load checkpoints and compute weighted average of their state dicts.

        Args:
            model_state: the current model state dict (keys to average)
            ckpt_paths_scores: list of (val_f1, ckpt_path) sorted by score descending
            device: device to load tensors on
        Returns:
            averaged state dict
        """
        if not ckpt_paths_scores:
            return model_state

        scores = [s for s, _ in ckpt_paths_scores]
        paths  = [p for _, p in ckpt_paths_scores]
        weights = self.compute_weights(scores)

        print(f"[SWA] Averaging {len(paths)} checkpoints "
              f"(val_f1 range: {min(scores):.4f}–{max(scores):.4f})")
        for i, (s, p, w) in enumerate(zip(scores, paths, weights)):
            print(f"  [{i+1}] val_f1={s:.4f}  weight={w:.4f}  {Path(p).name}")

        # Load the first checkpoint as base
        avg_sd = {}
        first_sd = torch.load(paths[0], map_location=device, weights_only=False)
        if "state_dict" in first_sd:
            first_sd = first_sd["state_dict"]

        # Initialize with zeros for keys that we'll average
        avg_sd = {k: torch.zeros_like(v.float()) for k, v in first_sd.items()}

        # Weighted sum
        for path, weight in zip(paths, weights):
            sd = torch.load(path, map_location=device, weights_only=False)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            for k in avg_sd:
                if k in sd:
                    avg_sd[k] += weight * sd[k].float()

        return avg_sd

    @property
    def pool_size(self) -> int:
        return len(self._pool)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class StringGNNPerturbDataset(Dataset):
    def __init__(self, pert_ids, symbols, node_indices, labels=None):
        self.pert_ids     = pert_ids
        self.symbols      = symbols
        self.node_indices = node_indices  # [N] long, -1 for OOV
        self.labels       = labels        # [N, 6640] long or None

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

    def __init__(self, data_dir="data", micro_batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        # Load STRING_GNN node name → index mapping
        node_names_list = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.node_name_to_idx = {name: i for i, name in enumerate(node_names_list)}

        def load_split(fname, has_lbl):
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            # Map pert_id → STRING_GNN node index (-1 for OOV)
            node_indices = torch.tensor(
                [self.node_name_to_idx.get(pid, -1) for pid in df["pert_id"]],
                dtype=torch.long,
            )
            labels = None
            if has_lbl and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return StringGNNPerturbDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(),
                node_indices, labels
            )

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

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


# ─── Gather helper ────────────────────────────────────────────────────────────

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


# ─── SWA Checkpoint Callback (FULLY FIXED) ────────────────────────────────────

class SWACheckpointCallback(pl.Callback):
    """
    Callback that collects periodic checkpoints for post-hoc SWA.
    Saves checkpoints every `save_every_n_epochs` epochs from `start_epoch`.

    FIXED vs parent (node3-1-1-3-1-1):
    - Parent read trainer.callback_metrics["val_f1"] which still had the PREVIOUS
      epoch's val_f1 (Lightning logs metrics before updating callback_metrics in some
      versions), causing off-by-one.
    - This node reads pl_module._last_val_f1 which is set by the LightningModule's
      own on_validation_epoch_end BEFORE the callback fires.
    - This guarantees the CURRENT epoch's correct val_f1 is used for quality gating.
    """

    def __init__(
        self,
        swa_manager:        "SWAManager",
        ckpt_dir:           str   = "run/checkpoints",
        save_every_n_epochs: int  = 3,
        start_epoch:        int   = 10,
    ):
        super().__init__()
        self.swa_manager         = swa_manager
        self.ckpt_dir            = Path(ckpt_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.start_epoch         = start_epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Save periodic checkpoints using the CURRENT epoch's val_f1 from pl_module._last_val_f1.

        The LightningModule's on_validation_epoch_end fires BEFORE this callback's
        on_validation_epoch_end in Lightning's execution order. So by the time this
        callback runs, pl_module._last_val_f1 already holds the current epoch's val_f1.

        This is the correct fix for the off-by-one issue: reading from the LightningModule
        attribute set in on_validation_epoch_end, rather than from trainer.callback_metrics
        which may have stale values.
        """
        epoch = trainer.current_epoch

        # Skip if before start epoch or not on the save interval
        if epoch < self.start_epoch:
            return
        if (epoch - self.start_epoch) % self.save_every_n_epochs != 0:
            return

        # Get CURRENT epoch's val_f1 from LightningModule attribute (correct fix)
        val_f1 = getattr(pl_module, "_last_val_f1", None)
        if val_f1 is None:
            return
        val_f1 = float(val_f1)

        # Quality gate: skip if below threshold
        if val_f1 < self.swa_manager.threshold:
            return

        # Save a checkpoint for this epoch with the correct val_f1
        ckpt_path = self.ckpt_dir / f"swa-epoch={epoch:04d}-val_f1={val_f1:.4f}.ckpt"
        if trainer.is_global_zero:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            # Use torch.save directly instead of trainer.save_checkpoint to avoid
            # triggering distributed collective operations (barrier/broadcast) inside
            # trainer.save_checkpoint in DDP mode.
            torch.save({"state_dict": pl_module.state_dict()}, str(ckpt_path))
            self.swa_manager.add_checkpoint(val_f1, str(ckpt_path))
            pl_module.print(
                f"[SWA] Saved checkpoint epoch={epoch}, val_f1={val_f1:.4f}, "
                f"pool_size={self.swa_manager.pool_size}"
            )


# ─── LightningModule ──────────────────────────────────────────────────────────

class StringGNNBilinearLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_head:         float = 5e-4,   # AdamW LR for head 1D params
        muon_lr:         float = 0.005,  # Muon LR for head 2D matrices
        weight_decay:    float = 1.5e-3, # L2 regularization (reverted from parent's 3e-3)
        focal_gamma:     float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs:      int   = 400,
        t0_epochs:       int   = 20,     # SGDR T_0 in EPOCHS (changed from step-based)
        t_mult:          int   = 2,      # SGDR T_mult for cycle lengthening
        patience:        int   = 200,    # INCREASED from 120 to match best nodes
        hidden_dim:      int   = 512,
        bilinear_rank:   int   = BILINEAR_RANK,
        n_layers:        int   = 6,
        dropout:         float = 0.50,   # INCREASED from 0.45
        # SWA hyperparameters
        swa_top_k:       int   = 15,     # Top-k checkpoints for SWA pool
        swa_threshold:   float = 0.490,  # Threshold for qualifying checkpoints
        swa_temperature: float = 3.0,    # Exponential weight temperature (proven best)
        swa_start_epoch: int   = 10,     # Start collecting SWA checkpoints
        swa_every_n:     int   = 3,      # Save checkpoint every N epochs for SWA
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []
        self._test_labels:   List[torch.Tensor] = []

        # val_f1 attribute: set in on_validation_epoch_end, read by SWACheckpointCallback
        # This is the correct fix for the off-by-one issue
        self._last_val_f1: Optional[float] = None

        # OOV embedding: learnable fallback for genes not in STRING_GNN vocabulary
        self.oov_embedding = nn.Parameter(torch.zeros(GNN_DIM))
        nn.init.normal_(self.oov_embedding, std=0.02)

        # SWA manager (initialized here for accessibility)
        self._swa_manager: Optional[SWAManager] = None

    def set_swa_manager(self, swa_manager: SWAManager):
        """Set the SWA manager from the outside (called after callback creation)."""
        self._swa_manager = swa_manager

    def setup(self, stage=None):
        self.model = StringGNNBilinearModel(
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            hidden_dim=self.hparams.hidden_dim,
            bilinear_rank=self.hparams.bilinear_rank,
            n_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
        )
        self.register_buffer("class_weights", CLASS_WEIGHTS)
        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=CLASS_WEIGHTS,
            label_smoothing=self.hparams.label_smoothing,
        )

        # Cast trainable parameters to float32 for stable optimization
        # Only head parameters are trainable (backbone is fully frozen)
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Precompute FULL frozen backbone embeddings on CPU.
        # The buffer will be automatically moved to GPU by Lightning.
        # All 8 GCN layers + post_mp are applied here, no gradient needed.
        self.model.backbone.precompute_full_embs(torch.device("cpu"))

    def _get_pert_embs(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Get perturbation embeddings from the frozen STRING_GNN for a batch.
        Handles OOV genes (-1) by substituting the learnable oov_embedding.

        Args:
            node_indices: [B] long, -1 for OOV genes
        Returns:
            pert_embs: [B, 256]
        """
        # Get precomputed static node embeddings (no gradient, buffer lookup)
        node_embs = self.model.get_node_embs()  # [18870, 256]

        # For each sample: look up its node embedding, use OOV embedding if index == -1
        batch_size = node_indices.shape[0]
        pert_embs  = torch.zeros(batch_size, GNN_DIM, dtype=node_embs.dtype,
                                 device=node_indices.device)

        valid_mask = node_indices >= 0
        oov_mask   = ~valid_mask

        if valid_mask.any():
            valid_idx = node_indices[valid_mask]
            pert_embs[valid_mask] = node_embs[valid_idx]

        if oov_mask.any():
            # Cast oov_embedding to match node_embs dtype
            pert_embs[oov_mask] = self.oov_embedding.to(node_embs.dtype).unsqueeze(0).expand(
                oov_mask.sum(), -1
            )

        return pert_embs  # [B, 256]

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """Full forward: static backbone lookup → pert_embs → bilinear head → logits [B, 3, 6640]."""
        pert_embs = self._get_pert_embs(node_indices)         # [B, 256]
        logits    = self.model.head(pert_embs)                 # [B, 3, 6640]
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        loss   = self.focal_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        if "label" in batch:
            loss = self.focal_loss(logits, batch["label"])
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
        # Store in instance attribute for SWACheckpointCallback to read
        # This is the correct fix for the off-by-one issue:
        # - LightningModule.on_validation_epoch_end fires BEFORE Callback.on_validation_epoch_end
        # - So when the callback reads pl_module._last_val_f1, it gets the CURRENT epoch's value
        self._last_val_f1 = float(f1)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
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

            # Deduplicate by pert_id (DDP sampler may pad)
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

            self.print(f"[Node3-1-1-3-1-1-1] Saved {len(dedup_perts)} test predictions → {pred_path}")

            if all_labels.any():
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[Node3-1-1-3-1-1-1] Self-computed test F1 (best ckpt) = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    # ── Optimizer: MuonWithAuxAdam (Muon for 2D head matrices, AdamW for rest) ──

    def configure_optimizers(self):
        hp = self.hparams

        # No backbone parameters are trainable (fully frozen backbone)
        # All trainable params are in the head:
        # 1. Head 2D matrices (ResBlock weights, project_in weight) → Muon
        # 2. Head 1D params (biases, LN, embeddings) + OOV embedding → AdamW

        # Head 2D matrices for Muon (hidden weight matrices in ResBlocks)
        head_2d_params = [
            p for name, p in self.model.head.named_parameters()
            if p.requires_grad and p.ndim >= 2
            # Exclude out_gene_emb.weight (embedding table, not a typical hidden matrix)
            # Include ResBlock linear weights and project_in linear weight
            and 'out_gene_emb' not in name
        ]

        # Head 1D params + embeddings → AdamW
        head_2d_ids = {id(p) for p in head_2d_params}
        head_other_params = [
            p for p in self.model.head.parameters()
            if p.requires_grad and id(p) not in head_2d_ids
        ]
        # Include OOV embedding in AdamW group
        head_other_params.append(self.oov_embedding)

        param_groups = [
            # Muon group: 2D head weight matrices
            {
                "params":       head_2d_params,
                "use_muon":     True,
                "lr":           hp.muon_lr,
                "weight_decay": hp.weight_decay,
                "momentum":     0.95,
            },
            # AdamW group: head 1D params + embeddings + OOV
            {
                "params":       head_other_params,
                "use_muon":     False,
                "lr":           hp.lr_head,
                "betas":        (0.9, 0.95),
                "weight_decay": 0.0,  # No weight decay for biases and embeddings
            },
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # EPOCH-based SGDR: CosineAnnealingWarmRestarts with T_0 in epochs
        # This is a KEY CHANGE from the parent (which used step-based T_0=200 steps ~9 epochs)
        # Best node2 family (F1=0.5180-0.5182) uses T_0=20 epochs directly in epoch mode
        # Epoch-based SGDR:
        #   - T_0=20 epochs → first restart at epoch 20 (vs ~9 epochs with step-based T_0=200)
        #   - Longer Cycle 1 means the model can train longer before the first reset
        #   - Cycle 2 (epochs 20-60), Cycle 3 (epochs 60-140), etc. — more diverse checkpoints
        # NOTE: CosineAnnealingWarmRestarts with T_0 in epochs requires epoch-level interval
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.t0_epochs,
            T_mult=max(1, int(hp.t_mult)),
            eta_min=1e-7,
        )

        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",   # EPOCH-based SGDR (key change from step-based)
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable parameters + buffers ──────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        # Include oov_embedding explicitly
        trainable_keys.add(prefix + "oov_embedding")
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        # Exclude the large full_embs_buffer from checkpoints (can be recomputed)
        buffer_keys.discard(prefix + "model.backbone.full_embs_buffer")
        # Also exclude edge_index and edge_weight buffers (they are from the pretrained model)
        buffer_keys.discard(prefix + "model.backbone.edge_index")
        buffer_keys.discard(prefix + "model.backbone.edge_weight")

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


# ─── SWA Application (post-training) ──────────────────────────────────────────

def apply_swa_and_predict(
    lit_module:  StringGNNBilinearLitModule,
    dm:          StringGNNDataModule,
    swa_manager: SWAManager,
    n_gpus:      int,
    out_dir:     Path,
    args,
):
    """
    Load SWA-averaged weights, run test inference, and save predictions.
    Called after training is complete.

    With FROZEN backbone, SWA is coherent:
    - All checkpoints have identical backbone weights (the precomputed buffer)
    - Only head weights vary across checkpoints
    - Averaging head weights from different SGDR cycles is valid because they
      all operate on the same frozen backbone representations

    The SWA device bug fix from parent is also retained:
    - No explicit SingleDeviceStrategy() — Lightning auto-handles device placement
    """
    top_k_pool = swa_manager.get_top_k()
    if len(top_k_pool) < 2:
        print(f"[SWA] Pool has {len(top_k_pool)} checkpoints, skipping SWA (need ≥2)")
        return

    print(f"\n[SWA] Starting quality-filtered SWA over {len(top_k_pool)} checkpoints...")

    # Compute SWA-averaged state dict
    current_sd = {k: v for k, v in lit_module.state_dict().items()}
    swa_sd = swa_manager.apply_swa(current_sd, top_k_pool, device="cpu")

    if not swa_sd:
        print("[SWA] SWA averaging failed, no predictions saved.")
        return

    # Load the SWA weights into a fresh model instance
    swa_lit = StringGNNBilinearLitModule(
        lr_head=args.lr_head,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        t0_epochs=args.t0_epochs,
        t_mult=args.t_mult,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        bilinear_rank=args.bilinear_rank,
        n_layers=args.n_layers,
        dropout=args.dropout,
        swa_top_k=args.swa_top_k,
        swa_threshold=args.swa_threshold,
        swa_temperature=args.swa_temperature,
        swa_start_epoch=args.swa_start_epoch,
        swa_every_n=args.swa_every_n,
    )

    # Setup model (this initializes model, buffers, etc.)
    dm.setup("test")
    swa_lit.setup("test")

    # Load the averaged state dict
    swa_lit.load_state_dict(swa_sd, strict=False)
    print("[SWA] Successfully loaded averaged weights.")

    # Save SWA averaged model checkpoint
    swa_ckpt_path = out_dir / "checkpoints" / "swa_averaged_model.ckpt"
    torch.save({"state_dict": swa_sd}, str(swa_ckpt_path))
    print(f"[SWA] Saved SWA model to {swa_ckpt_path}")

    # Redirect test predictions to swa_test_predictions.tsv temporarily
    orig_test_epoch_end = swa_lit.on_test_epoch_end

    def patched_test_epoch_end():
        """Same as original but writes to swa_test_predictions.tsv."""
        local_probs  = torch.cat(swa_lit._test_preds, 0)
        dummy_labels = (torch.cat(swa_lit._test_labels, 0) if swa_lit._test_labels
                        else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long))

        all_probs, all_labels = local_probs, dummy_labels
        all_pert, all_syms   = swa_lit._test_pert_ids, swa_lit._test_symbols

        # Write SWA predictions
        out_dir.mkdir(parents=True, exist_ok=True)
        swa_pred_path = out_dir / "swa_test_predictions.tsv"

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

        with open(swa_pred_path, "w") as fh:
            fh.write("idx\tinput\tprediction\n")
            for pid, sym, probs in zip(dedup_perts, dedup_syms, dedup_probs_list):
                fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")

        print(f"[SWA] Saved {len(dedup_perts)} SWA test predictions → {swa_pred_path}")

        if all_labels.any():
            dedup_probs_np  = np.array(dedup_probs_list)
            dedup_labels_np = np.array(dedup_label_rows)
            swa_f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
            print(f"[SWA] Self-computed SWA test F1 = {swa_f1:.4f}")

        # Replace main test_predictions.tsv with SWA predictions
        main_pred_path = out_dir / "test_predictions.tsv"
        if swa_pred_path.exists():
            import shutil
            shutil.copy(str(swa_pred_path), str(main_pred_path))
            print(f"[SWA] Replaced test_predictions.tsv with SWA predictions.")

        swa_lit._test_preds.clear(); swa_lit._test_pert_ids.clear()
        swa_lit._test_symbols.clear(); swa_lit._test_labels.clear()

    swa_lit.on_test_epoch_end = patched_test_epoch_end

    # Create a minimal trainer for inference only.
    # No explicit strategy= — let Lightning auto-select and handle device placement correctly.
    swa_trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,  # Single device for SWA inference
        num_nodes=1,
        precision="bf16-mixed",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    swa_trainer.test(swa_lit, datamodule=dm)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 3-1-1-3-1-1-1 – STRING_GNN FROZEN + Rank-512 Bilinear Head + Muon + Fixed SWA"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--lr-head",           type=float, default=5e-4)
    p.add_argument("--muon-lr",           type=float, default=0.005)
    p.add_argument("--weight-decay",      type=float, default=1.5e-3)  # Reverted to best config
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--label-smoothing",   type=float, default=0.05)
    p.add_argument("--micro-batch-size",  type=int,   default=32)
    p.add_argument("--global-batch-size", type=int,   default=64)
    p.add_argument("--max-epochs",        type=int,   default=400)
    p.add_argument("--t0-epochs",         type=int,   default=20)    # EPOCH-based (changed from step-based)
    p.add_argument("--t-mult",            type=int,   default=2)
    p.add_argument("--patience",          type=int,   default=200)   # INCREASED from 120
    p.add_argument("--hidden-dim",        type=int,   default=512)
    p.add_argument("--bilinear-rank",     type=int,   default=512)
    p.add_argument("--n-layers",          type=int,   default=6)
    p.add_argument("--dropout",           type=float, default=0.50)  # INCREASED from 0.45
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--val-check-interval",type=float, default=1.0)
    p.add_argument("--debug-max-step",    type=int,   default=None)
    p.add_argument("--fast-dev-run",      action="store_true", default=False)
    # SWA arguments
    p.add_argument("--swa-top-k",         type=int,   default=15)
    p.add_argument("--swa-threshold",     type=float, default=0.490)  # Matching proven config
    p.add_argument("--swa-temperature",   type=float, default=3.0)    # Proven best
    p.add_argument("--swa-start-epoch",   type=int,   default=10)
    p.add_argument("--swa-every-n",       type=int,   default=3)
    p.add_argument("--skip-swa",          action="store_true", default=False,
                   help="Skip SWA post-processing (for debugging)")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm  = StringGNNDataModule(
        args.data_dir, args.micro_batch_size, args.num_workers
    )

    # Create SWA manager
    swa_manager = SWAManager(
        top_k=args.swa_top_k,
        threshold=args.swa_threshold,
        temperature=args.swa_temperature,
        ckpt_dir=str(out_dir / "checkpoints"),
    )

    lit = StringGNNBilinearLitModule(
        lr_head=args.lr_head,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        t0_epochs=args.t0_epochs,
        t_mult=args.t_mult,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        bilinear_rank=args.bilinear_rank,
        n_layers=args.n_layers,
        dropout=args.dropout,
        swa_top_k=args.swa_top_k,
        swa_threshold=args.swa_threshold,
        swa_temperature=args.swa_temperature,
        swa_start_epoch=args.swa_start_epoch,
        swa_every_n=args.swa_every_n,
    )
    lit.set_swa_manager(swa_manager)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=5, save_last=True,
    )
    es_cb  = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.patience, min_delta=1e-5,
    )
    lr_cb  = LearningRateMonitor(logging_interval="epoch")  # Epoch-based logging
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    # SWA checkpoint callback — reads pl_module._last_val_f1 (correct fix for off-by-one)
    swa_ckpt_cb = SWACheckpointCallback(
        swa_manager=swa_manager,
        ckpt_dir=str(out_dir / "checkpoints"),
        save_every_n_epochs=args.swa_every_n,
        start_epoch=args.swa_start_epoch,
    )

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps    = -1
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps   = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2
        limit_test  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    accum    = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

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
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb, swa_ckpt_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,   # Gradient clipping for stable Muon training
    )

    trainer.fit(lit, datamodule=dm)

    # Test with best single checkpoint
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    # Apply SWA post-processing (on global rank 0 only, after DDP training)
    if (trainer.is_global_zero
            and not args.fast_dev_run
            and args.debug_max_step is None
            and not args.skip_swa
            and swa_manager.pool_size >= 2):
        try:
            apply_swa_and_predict(lit, dm, swa_manager, n_gpus, out_dir, args)
        except Exception as e:
            print(f"[SWA] Warning: SWA post-processing failed with error: {e}")
            print("[SWA] Keeping best single-checkpoint predictions.")

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 3-1-1-3-1-1-1 – STRING_GNN FROZEN + Rank-512 Bilinear Head + Muon + Fixed SWA\n"
            f"Key changes: FULLY FROZEN backbone (no backbone fine-tuning), off-by-one fix\n"
            f"(pl_module._last_val_f1), epoch-based SGDR T_0={args.t0_epochs},\n"
            f"patience={args.patience}, weight_decay={args.weight_decay}, dropout={args.dropout}\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
