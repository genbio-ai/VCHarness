"""
Node 3-1-1-2-1: Complete Strategic Pivot to STRING_GNN
  Partial STRING_GNN Fine-tuning (mps.6+mps.7+post_mp)
  + rank-256 Deep Residual Bilinear MLP Head
  + Class Weights [2.0, 0.5, 4.0]
  + Slow Cosine LR (total_steps=6600) for Secondary Improvement Phase

=== Motivation ===
Parent node3-1-1-2 (AIDO.Cell-100M, test F1=0.4074) represents the lowest
performance in its lineage. The feedback explicitly recommends abandoning the
AIDO.Cell approach and pivoting to STRING_GNN-based architectures, which
dominate the MCTS tree (node2-1-3: 0.5047, node1-2: 0.4912).

=== Key Design Decisions ===
1. STRING_GNN partial fine-tuning (mps.6+mps.7+post_mp, ~198K trainable backbone):
   - Provides task-specific PPI topology adaptation (proven benefit in node2-1-3)
   - Frozen prefix (mps.0-5) pre-computed once as buffer (efficiency)
   - backbone_lr=5e-5, backbone_wd=0 (conservative, no weight decay for backbone)

2. rank=256 deep bilinear head (6 ResidualBlocks, hidden=512, expand=4, dropout=0.20):
   - rank=256 vs node2-1-3's rank=512 → lower capacity, better generalization
   - Allows secondary improvement phase (similar to node2-1-2: 0.5011 at epoch 51)
   - Total head params ~14.8M; total trainable ~15M (~10,600 params/sample)

3. Class weights [2.0, 0.5, 4.0] for [down, neutral, up]:
   - Proven optimal across the MCTS tree (node2-1-3, node2-2-1-1, node1-1-2-1-1)
   - Addresses severe class imbalance (88.9% neutral, 8.1% down, 3.0% up)

4. Slow cosine LR (total_steps=6600, warmup=50 steps):
   - Mimics node2-1-3's accidentally slow LR that proved beneficial
   - At best epoch (~32-50): LR still at ~90-95% of peak
   - Enables continued learning at high LR without collapsing prematurely
   - Allows secondary improvement phase (analogous to node2-1-2's epoch 51 gain)

5. patience=80 on val_f1 (long enough to capture secondary improvement):
   - node2-1-2 (frozen+rank=256+mild_weights): secondary improvement at epoch 51
   - With partial fine-tuning+stronger weights, secondary phase may arrive earlier

=== Distinct from All Existing Nodes ===
- node2-1-3 (0.5047): rank=512 → our: rank=256 (different capacity regime)
- node2-1-2 (0.5011): FROZEN backbone, mild weights [1.5,0.8,2.5] → our: partial fine-tuning+strong weights
- node2-1-2-1 (0.5016): partial + rank=512 → our: rank=256
- node1-2-3-2 (0.4996): frozen+rank=512+total_steps=6600+lr=3e-4 → our: partial+rank=256+lr=5e-4
- node1-1-2-2 (0.4904): partial+rank=512+total_steps=1200 → our: rank=256+total_steps=6600
- Parent (0.4074): AIDO.Cell → our: STRING_GNN (complete pivot)

This specific combination (partial fine-tuning + rank=256 + class weights [2,0.5,4] +
total_steps=6600) has NOT been tried in the MCTS tree.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import math
import argparse
from pathlib import Path
from datetime import timedelta
from typing import List, Optional

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

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
N_GNN_NODES    = 18870   # STRING_GNN graph nodes
GNN_DIM        = 256     # STRING_GNN hidden dimension
N_GENES_OUT    = 6640    # Label output positions
N_CLASSES      = 3       # down(-1→0), neutral(0→1), up(+1→2)
HIDDEN_DIM     = 512     # MLP hidden dimension
RANK           = 256     # Bilinear interaction rank
N_RESBLOCKS    = 6       # Residual blocks in head MLP
EXPAND         = 4       # Expand factor for inner dim of ResidualBlock
DROPOUT        = 0.20    # Head dropout rate

# Class weights: inverse-frequency weighted for [down, neutral, up]
# Down: 8.14% → weight 2.0; Neutral: 88.86% → weight 0.5; Up: 3.0% → weight 4.0
# Proven optimal combination from node2-1-3, node2-2-1-1, node1-1-2-1-1
CLASS_WEIGHTS = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """
    Compute macro-averaged per-gene F1 score matching calc_metric.py logic.

    Args:
        pred_np:   [N, 3, G] softmax probabilities (float)
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
    Standard focal cross-entropy loss with per-class weighting.

    Unlike the parent node's neutral-balanced variant (which proved unhelpful),
    this uses the full label set without subsampling — simpler and more stable.
    Class weights [2.0, 0.5, 4.0] handle the class imbalance at the weighting level.
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
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, G] unnormalized logits
            targets: [B, G] class indices in {0, 1, 2}
        Returns:
            scalar loss
        """
        B, C, G = logits.shape
        logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        targets_flat = targets.reshape(-1)                      # [B*G]

        log_probs = F.log_softmax(logits_flat.float(), dim=1)   # [B*G, C]
        probs     = torch.exp(log_probs)

        # Gather log-prob and prob at target class
        target_log_prob = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        target_prob     = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1.0 - target_prob).pow(self.gamma)

        # Per-class weight
        if self.weight is not None:
            class_w = self.weight.to(logits.device)[targets_flat]
        else:
            class_w = torch.ones_like(focal_weight)

        # Label smoothing
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
        return weighted_loss.sum() / class_w.sum().clamp(min=1.0)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    """
    Dataset mapping perturbation gene identifiers to STRING_GNN node indices.
    Each sample: (pert_id, symbol, node_idx, [optional] label)
    """

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        node_indices: torch.Tensor,   # [N] long, -1 for OOV genes
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long or None
    ):
        self.pert_ids     = pert_ids
        self.symbols      = symbols
        self.node_indices = node_indices
        self.labels       = labels

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":  self.pert_ids[idx],
            "symbol":   self.symbols[idx],
            "node_idx": self.node_indices[idx],   # scalar tensor (0-dim long)
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    out = {
        "pert_id":  [b["pert_id"]  for b in batch],
        "symbol":   [b["symbol"]   for b in batch],
        "node_idx": torch.stack([b["node_idx"] for b in batch]),  # [B]
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])   # [B, 6640]
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbDataModule(pl.LightningDataModule):
    """
    DataModule: loads train/val/test TSV files and maps pert_ids to
    STRING_GNN node indices using node_names.json.
    """

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir        = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers     = num_workers

    def setup(self, stage=None):
        # Build Ensembl ID → STRING_GNN node index mapping
        node_names_raw = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        # node_names_raw: list of strings (Ensembl IDs), index=STRING_GNN node idx
        self.node2idx = {name: idx for idx, name in enumerate(node_names_raw)}

        self.train_ds = self._load_split("train.tsv", has_label=True)
        self.val_ds   = self._load_split("val.tsv",   has_label=True)
        self.test_ds  = self._load_split("test.tsv",  has_label=False)

        n_train = len(self.train_ds)
        n_oov   = int((self.train_ds.node_indices < 0).sum().item())
        print(f"[Node3-1-1-2-1] Train: {n_train} samples, OOV: {n_oov}")

    def _load_split(self, fname: str, has_label: bool) -> PerturbDataset:
        df = pd.read_csv(self.data_dir / fname, sep="\t")
        pert_ids = df["pert_id"].tolist()
        symbols  = df["symbol"].tolist()

        # Map pert_ids (Ensembl IDs) to STRING_GNN node indices (-1 = OOV)
        node_indices = torch.tensor(
            [self.node2idx.get(pid, -1) for pid in pert_ids],
            dtype=torch.long,
        )

        labels = None
        if has_label and "label" in df.columns:
            # Parse JSON label vectors, shift {-1,0,1} → {0,1,2}
            rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
            labels = torch.tensor(rows, dtype=torch.long)

        return PerturbDataset(pert_ids, symbols, node_indices, labels)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=(self.num_workers > 0),
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Model Architecture ───────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Pre-norm residual block: LN → Linear(d, d*expand) → GELU → Dropout → Linear(d*expand, d).
    """

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = dim * expand
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ─── DDP gather helper ────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    """Gather tensors from all DDP ranks with variable-size padding."""
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))
    pad    = max_sz - local_p.shape[0]
    p = local_p.to(device)
    l = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)
    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p)
    dist.all_gather(gl, l)
    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class StringGNNBilinearModule(pl.LightningModule):
    """
    Partial STRING_GNN fine-tuning + deep bilinear MLP head.

    Architecture:
      frozen prefix: [18870, 256] (pre-computed mps.0-5 output, no gradient)
      trainable backbone: mps.6 + mps.7 + post_mp (~198K params, backbone_lr=5e-5)
      OOV embedding: learnable 256-dim vector for genes not in STRING_GNN graph
      proj_in: Linear(256→512) + LayerNorm
      residual MLP: 6x ResidualBlock(512, expand=4, dropout=0.20)
      bilinear_proj: Linear(512 → 3*256, bias=False)
      out_gene_emb: nn.Parameter [6640, 256] (random init, fully learnable)
      logits: einsum('bcr,gr->bcg') → [B, 3, 6640]
    """

    def __init__(
        self,
        lr_backbone: float  = 5e-5,
        lr_head: float      = 5e-4,
        weight_decay: float = 1e-3,
        focal_gamma: float  = 2.0,
        total_steps: int    = 6600,
        warmup_steps: int   = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Validation / test accumulation lists
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str]          = []
        self._test_symbols:  List[str]          = []
        self._test_labels:   List[torch.Tensor] = []

    def setup(self, stage=None):
        """
        Initialize model components:
          1. Load STRING_GNN and compute frozen prefix (mps.0-5 output)
          2. Keep trainable layers: mps.6, mps.7, post_mp
          3. Build prediction head
        """
        # ── Load STRING_GNN ───────────────────────────────────────────────────
        gnn = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        gnn.eval()

        # ── Load graph data ───────────────────────────────────────────────────
        graph      = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index = graph["edge_index"]                               # [2, E] long
        edge_weight = graph.get("edge_weight", None)
        if edge_weight is None:
            # Fallback: uniform weights if not available
            edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
        else:
            edge_weight = edge_weight.float()

        # ── Compute frozen prefix (mps.0 through mps.5) ──────────────────────
        # We pre-compute the deterministic output of the first 6 GNN layers once.
        # At each training step, only mps.6+mps.7+post_mp are run (with gradient).
        # This avoids running the full 8-layer GNN at every step.
        with torch.no_grad():
            x = gnn.emb.weight.cpu().clone().float()
            for i in range(6):  # mps.0 to mps.5 inclusive (frozen)
                layer = gnn.mps[i]
                # GNNLayer forward: norm → conv → act → dropout → residual
                h = layer.norm(x)                              # LayerNorm (mode=node)
                h = layer.conv(h, edge_index, edge_weight)     # GCNConv
                h = layer.act(h)                               # GELU
                # dropout p=0.0 for STRING_GNN, but call for correctness in eval mode
                h = layer.dropout(h)
                x = x + h                                      # residual
            frozen_prefix = x.detach()

        # Register buffers (auto-moved to correct device by Lightning)
        self.register_buffer("frozen_prefix", frozen_prefix)
        self.register_buffer("edge_index",    edge_index)
        self.register_buffer("edge_weight",   edge_weight)

        # ── Trainable GNN suffix (partial fine-tuning) ───────────────────────
        # Keep mps.6, mps.7, post_mp with gradients; discard other layers.
        self.gnn_mps6    = gnn.mps[6]
        self.gnn_mps7    = gnn.mps[7]
        self.gnn_post_mp = gnn.post_mp

        # ── OOV embedding for genes not in STRING_GNN graph ──────────────────
        self.oov_emb = nn.Parameter(torch.randn(GNN_DIM) * 0.01)

        # ── Prediction head ───────────────────────────────────────────────────
        # proj_in: map 256 (GNN dim) → 512 (MLP hidden dim)
        self.proj_in = nn.Sequential(
            nn.Linear(GNN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
        )

        # 6-layer residual MLP (hidden=512, expand=4, dropout=0.20)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(HIDDEN_DIM, expand=EXPAND, dropout=DROPOUT)
            for _ in range(N_RESBLOCKS)
        ])

        # Bilinear projection: [B, 512] → [B, 3*256]
        # Class-specific interaction: for class c, logit[b,c,g] = bilinear_proj(x)[b,c,:] · out_gene_emb[g,:]
        self.bilinear_proj = nn.Linear(HIDDEN_DIM, N_CLASSES * RANK, bias=False)

        # Output gene embeddings [6640, 256]: fully learnable (random init)
        # These are initialized randomly — no pre-training from STRING_GNN or AIDO.Cell
        self.out_gene_emb = nn.Parameter(
            torch.randn(N_GENES_OUT, RANK) * (1.0 / math.sqrt(RANK))
        )

        # ── Loss function ─────────────────────────────────────────────────────
        self.criterion = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=CLASS_WEIGHTS.clone(),
            label_smoothing=0.0,
        )

        # ── Cast all trainable parameters to float32 ─────────────────────────
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Log trainable parameter count
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        n_buffers   = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"[Node3-1-1-2-1] Trainable: {n_trainable:,} / {n_total:,} "
            f"({100 * n_trainable / n_total:.1f}%), buffers: {n_buffers:,}"
        )

    def _gnn_layer_forward(
        self,
        layer,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Manually execute a single GNNLayer (norm → conv → act → dropout → residual).
        This is equivalent to layer(x, edge_index, edge_weight) but more explicit.
        """
        h = layer.norm(x)                             # LayerNorm
        h = layer.conv(h, edge_index, edge_weight)    # GCNConv
        h = layer.act(h)                              # GELU
        h = layer.dropout(h)                          # Dropout (p=0 for STRING_GNN)
        return x + h                                  # residual

    def _get_gnn_embeddings(self) -> torch.Tensor:
        """
        Run trainable GNN layers (mps.6 + mps.7 + post_mp) on the frozen prefix
        to produce the final node embeddings [18870, 256] with gradients.
        """
        x = self.frozen_prefix   # [18870, 256], no grad (buffer)

        # Run mps.6 with gradient
        x = self._gnn_layer_forward(
            self.gnn_mps6, x, self.edge_index, self.edge_weight
        )

        # Run mps.7 with gradient
        x = self._gnn_layer_forward(
            self.gnn_mps7, x, self.edge_index, self.edge_weight
        )

        # Final projection
        x = self.gnn_post_mp(x)  # [18870, 256]

        return x

    def _extract_pert_embs(
        self,
        gnn_embs: torch.Tensor,
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract perturbation gene embeddings from GNN output.
        Genes not in STRING_GNN (node_idx=-1) use the learnable OOV embedding.

        Args:
            gnn_embs:    [18870, 256] full GNN embedding matrix
            node_indices: [B] long, -1 for OOV genes
        Returns:
            [B, 256] perturbation embeddings
        """
        is_oov    = (node_indices < 0)
        safe_idx  = node_indices.clone()
        safe_idx[is_oov] = 0  # safe dummy index for gathering

        embs = gnn_embs[safe_idx]  # [B, 256]

        # Replace OOV positions with learnable OOV embedding.
        # Use torch.where to avoid in-place indexed assignment on a gradient-tracked tensor.
        if is_oov.any():
            oov_mask = is_oov.unsqueeze(1).expand_as(embs)          # [B, 256] bool
            oov_fill = self.oov_emb.float().unsqueeze(0).expand_as(embs)  # [B, 256]
            embs = torch.where(oov_mask, oov_fill, embs)

        return embs

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: STRING_GNN → MLP → bilinear → logits.

        Args:
            node_indices: [B] long, STRING_GNN node indices (-1 for OOV)
        Returns:
            logits: [B, 3, 6640]
        """
        # 1. Get GNN embeddings (partial fine-tuning of mps.6+mps.7+post_mp)
        gnn_embs = self._get_gnn_embeddings()           # [18870, 256]

        # 2. Extract perturbation gene embeddings
        pert_embs = self._extract_pert_embs(gnn_embs, node_indices)  # [B, 256]

        # 3. Project to MLP hidden dim
        x = self.proj_in(pert_embs.float())             # [B, 512]

        # 4. Residual MLP blocks
        for block in self.residual_blocks:
            x = block(x)                                # [B, 512]

        # 5. Class-specific bilinear projection
        per_class = self.bilinear_proj(x)               # [B, 3*rank]
        per_class = per_class.view(-1, N_CLASSES, RANK) # [B, 3, 256]

        # 6. Bilinear interaction with output gene embeddings
        # logit[b,c,g] = per_class[b,c,:] · out_gene_emb[g,:]
        logits = torch.einsum(
            'bcr,gr->bcg', per_class, self.out_gene_emb.float()
        )  # [B, 3, 6640]

        return logits

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_idx"])
        loss   = self.criterion(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    # ── Validation ────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_idx"])
        if "label" in batch:
            loss = self.criterion(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        lp = torch.cat(self._val_preds,  0)   # [N_local, 3, 6640]
        ll = torch.cat(self._val_labels, 0)   # [N_local, 6640]

        # Gather across DDP ranks so all ranks compute the same global metric
        if self.trainer.world_size > 1:
            lp, ll = _gather_tensors(lp, ll, self.device, self.trainer.world_size)

        probs_np  = torch.softmax(lp, dim=1).numpy()
        labels_np = ll.numpy()
        f1 = compute_per_gene_f1(probs_np, labels_np)

        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear()
        self._val_labels.clear()

    # ── Test ──────────────────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_idx"])
        probs  = torch.softmax(logits.float(), dim=1)  # [B, 3, 6640]
        self._test_preds.append(probs.detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs = torch.cat(self._test_preds, 0)  # [N_local, 3, 6640]

        if self._test_labels:
            dummy_labels = torch.cat(self._test_labels, 0)
        else:
            dummy_labels = torch.zeros(
                local_probs.shape[0], N_GENES_OUT, dtype=torch.long
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
            all_probs  = local_probs
            all_labels = dummy_labels
            all_pert   = self._test_pert_ids
            all_syms   = self._test_symbols

        if self.trainer.is_global_zero:
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate (DDP DistributedSampler may pad the dataset)
            seen_pids: set = set()
            dedup_perts, dedup_syms, dedup_probs_list, dedup_label_rows = [], [], [], []
            for pid, sym, prob_row, lbl_row in zip(
                all_pert, all_syms,
                all_probs.numpy(), all_labels.numpy(),
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

            self.print(
                f"[Node3-1-1-2-1] Saved {len(dedup_perts)} test predictions → {pred_path}"
            )

            # Self-evaluate if labels are available
            if self._test_labels:
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[Node3-1-1-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    # ── Optimizer + LR Schedule ───────────────────────────────────────────────

    def configure_optimizers(self):
        hp = self.hparams

        # Two-group AdamW:
        #   Backbone (mps.6+mps.7+post_mp): low LR (5e-5), no weight decay
        #   Head (MLP + bilinear + OOV emb): higher LR (5e-4), weight_decay=1e-3
        backbone_params = (
            list(self.gnn_mps6.parameters())
            + list(self.gnn_mps7.parameters())
            + list(self.gnn_post_mp.parameters())
        )
        backbone_ids = {id(p) for p in backbone_params}

        head_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in backbone_ids
        ]

        param_groups = [
            {
                "params":       backbone_params,
                "lr":           hp.lr_backbone,
                "weight_decay": 0.0,  # No weight decay for pretrained backbone
                "name":         "backbone",
            },
            {
                "params":       head_params,
                "lr":           hp.lr_head,
                "weight_decay": hp.weight_decay,
                "name":         "head",
            },
        ]

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

        # Step-based cosine LR with linear warmup:
        #   0 → warmup_steps: linear ramp 0 → 1
        #   warmup_steps → total_steps: cosine decay 1 → eta_min
        #   Beyond total_steps: maintain eta_min (clamp progress at 1.0)
        # With total_steps=6600 and early stopping at ~epoch 30-80:
        #   LR remains at 90-97% of peak throughout training
        #   → "essentially constant" regime proven beneficial in node2-1-3
        warmup_steps = hp.warmup_steps
        total_steps  = hp.total_steps
        min_lr_ratio = 1e-6 / max(hp.lr_head, 1e-10)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = min(1.0, float(step - warmup_steps) /
                           max(1, total_steps - warmup_steps))
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
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
        bufs    = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving ckpt: {trained:,}/{total:,} params ({100*trained/total:.1f}%), "
            f"+ {bufs:,} buffer elements"
        )
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 3-1-1-2-1: Partial STRING_GNN + rank-256 bilinear MLP"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--lr-backbone",        type=float, default=5e-5)
    p.add_argument("--lr-head",            type=float, default=5e-4)
    p.add_argument("--weight-decay",       type=float, default=1e-3)
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--total-steps",        type=int,   default=6600)
    p.add_argument("--warmup-steps",       type=int,   default=50)
    p.add_argument("--micro-batch-size",   type=int,   default=8)
    p.add_argument("--global-batch-size",  type=int,   default=128)
    p.add_argument("--max-epochs",         type=int,   default=500)
    p.add_argument("--patience",           type=int,   default=80)
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── DataModule ────────────────────────────────────────────────────────────
    dm = PerturbDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # ── LightningModule ───────────────────────────────────────────────────────
    lit = StringGNNBilinearModule(
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=1, save_last=True,
    )
    es_cb   = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.patience, min_delta=1e-5,
    )
    lr_cb   = LearningRateMonitor(logging_interval="step")
    pb_cb   = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    # ── Debug / fast-dev-run handling ─────────────────────────────────────────
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

    # ── Gradient accumulation ─────────────────────────────────────────────────
    # global_batch_size = micro_batch_size * n_gpus * accumulate_grad_batches
    # global-batch-size must be multiple of micro_batch_size * 8 (for ≤8 GPUs)
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    strategy = DDPStrategy(
        find_unused_parameters=True,  # needed since frozen_prefix has no gradient
        timeout=timedelta(seconds=120),
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
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
        gradient_clip_val=1.0,  # Prevent bf16 numerical instability
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(lit, datamodule=dm)

    # ── Test ──────────────────────────────────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        trainer.test(lit, datamodule=dm)
    else:
        trainer.test(lit, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
