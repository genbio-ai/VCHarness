"""
Node 4-1-1-1-2-1 – Intermediate Capacity Head (rank=384, 5-layer) + Stronger Regularization
                    + Gradient Clipping + Label Position Dropout

Architecture:
  - STRING_GNN completely frozen backbone (ALL 8 GCN layers + post_mp):
      * All layers frozen; embeddings precomputed once and cached
      * No gradient flows through backbone at all
      * Provides stable embeddings across warm restart cycles
  - 5-layer deep residual bilinear MLP head (rank=384, hidden=384):
      * Input: 256-dim PPI embedding from fully frozen STRING_GNN
      * ResidualBlock: [LN → Linear(dim→dim*expand) → GELU → Dropout → Linear(dim*expand→dim) + skip]
      * Bilinear output: [B, 3, rank=384] @ [rank=384, 6640] → [B, 3, 6640]
      * Trainable parameters: ~9M (reduced from 17.1M in parent)
  - Focal cross-entropy loss: gamma=2.0, class weights [down=2.0, neutral=0.5, up=4.0]
  - Label position dropout: 10% of gene positions masked to neutral class during training
  - Gradient clipping: val=1.0 (prevents gradient explosion with deep head)
  - Muon optimizer (lr=0.005) for 2D head weight matrices
  - AdamW (lr=5e-4, wd=5e-3) for remaining head parameters
  - Cosine warm restarts: T_0=600 steps (~27 epochs), T_mult=1 (same-length cycles)
  - max_epochs=300, patience=50

Key improvements vs Node 4-1-1-1-2 (parent, F1=0.4824):
  1. [PRIMARY] Reduce head capacity: rank=384, n_layers=5, hidden_dim=384 (~9M vs ~17.1M)
     Parent's 17.1M-parameter head severely overfit 1,416 training samples (val/train=5.34x at
     best epoch, 7.67x at final). Reducing to ~9M cuts params by 47%, targeting the sweet
     spot between sibling's underfitting (~4.5M, F1=0.4752) and parent's overfitting.
  2. [PRIMARY] Increase dropout: 0.30 → 0.40
     Further regularization of the head. node4-3 (F1=0.5036) used dropout=0.40 effectively.
     Tree-best node (F1=0.5182) uses dropout=0.45. 0.40 balances capacity and regularization.
  3. [PRIMARY] Stronger L2 regularization: head_wd 1e-3 → 5e-3
     Parent's val/train loss ratio reached 5.34x at best epoch. Stronger weight decay
     specifically targets the bilinear_proj (589K params) and ResBlock weights.
  4. [SECONDARY] Add gradient clipping (gradient_clip_val=1.0)
     node4-3 (F1=0.5036) used this for stability. Prevents gradient explosion in deeper
     head layers and improves training stability across warm restart LR spikes.
  5. [SECONDARY] Label position dropout (p=0.10 during training)
     Novel regularization: randomly mask 10% of gene positions to neutral class during
     training. Forces the head to learn robust gene-position-agnostic representations
     rather than memorizing specific position patterns. Reduces effective supervision
     signal from 6,640 to ~5,976 positions per forward pass.
  6. [SECONDARY] Reduce patience: 80 → 50
     Parent's 80-epoch patience tolerated prolonged plateau/decline phases (cycles 3-12).
     With 5-cycle patience (T_0≈14 epochs), 50 epochs allows 3-4 cycles of non-improvement
     before stopping, balancing exploration with compute efficiency.

Differentiation from parent node4-1-1-1-2:
  - Parent: rank=512, 6 layers, hidden=512 (~17.1M) + dropout=0.30 + head_wd=1e-3 + patience=80
  - This node: rank=384, 5 layers, hidden=384 (~9M) + dropout=0.40 + head_wd=5e-3 +
               gradient_clip=1.0 + label_pos_dropout=0.10 + patience=50

Memory sources:
  - node4-1-1-1-2/memory/feedback.md: PRIMARY recommendation = intermediate capacity head
    (rank=384, 5 layers, ~9-10M) + stronger regularization (head_wd=5e-3, dropout=0.40)
  - node4-3 (F1=0.5036): gradient clipping val=1.0 + dropout=0.40 used successfully
  - tree-best node2-1-1-1-2-1-1-1-1-1-1-1-1 (F1=0.5182): dropout=0.45 used
  - node4-1-1-1-2/memory/feedback.md: label position dropout suggestion
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
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy, SingleDeviceStrategy
from muon import MuonWithAuxAdam
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_DIM        = 256

# Focal loss class weights: emphasize minority classes (down=-1, up=+1)
# Train distribution: 8.14% down, 88.86% neutral, 3.00% up
# Weights [down=2.0, neutral=0.5, up=4.0] proven effective in tree-best nodes
FOCAL_CLASS_WEIGHTS = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)


# ─── Focal Loss ───────────────────────────────────────────────────────────────

def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    class_weights: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Focal cross-entropy loss: down-weights easy examples.
    logits:  [N, C]
    targets: [N]  (class indices 0,1,2)
    Returns: scalar mean loss
    """
    if label_smoothing > 0.0:
        ce = F.cross_entropy(
            logits, targets, weight=class_weights,
            label_smoothing=label_smoothing, reduction="none"
        )
    else:
        ce = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    pt   = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Mirrors calc_metric.py: per-gene macro F1 averaged over all genes."""
    pred_cls = pred_np.argmax(axis=1)  # [B, G]
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1   = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Stores pert_id → STRING_GNN node index mapping and labels."""

    def __init__(
        self,
        pert_ids:     List[str],
        symbols:      List[str],
        node_indices: torch.Tensor,              # [N] long, -1 for unknown
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

class PerturbationDataModule(pl.LightningDataModule):

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
            return PerturbationDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(), idxs, labels
            )

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

        graph = torch.load(gnn_dir / "graph_data.pt", weights_only=False)
        self.edge_index  = graph["edge_index"]
        self.edge_weight = graph.get("edge_weight", None)
        self.n_nodes     = len(node_names)  # 18870

        # Coverage check
        n_unknown = sum(
            1 for ds in (self.train_ds, self.val_ds, self.test_ds)
            for ni in ds.node_indices.tolist() if ni == -1
        )
        total = len(self.train_ds) + len(self.val_ds) + len(self.test_ds)
        print(f"[Node4-1-1-1-2-1] {n_unknown}/{total} samples not found in STRING_GNN graph "
              f"-> will use learned fallback embedding.")

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


# ─── Residual MLP Block ───────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Residual MLP block: LN → Linear(in→expand) → GELU → Dropout → Linear(expand→in) → Dropout + skip.
    Pre-norm design (Layer Norm before linear) for better gradient flow in deeper networks.
    """
    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


# ─── Bilinear Head ────────────────────────────────────────────────────────────

class GNNBilinearHead(nn.Module):
    """
    Intermediate-capacity deep residual MLP + bilinear interaction head.
    Input: [B, gnn_dim=256] PPI embedding of perturbed gene
    Output: [B, n_classes=3, n_genes_out=6640] logits

    Architecture:
      1. Input projection: LayerNorm(gnn_dim) → Linear(gnn_dim → hidden_dim) → GELU
      2. N_layers residual MLP blocks (hidden_dim=384, expand=4, dropout=0.40)
      3. Output norm: LayerNorm(hidden_dim)
      4. Bilinear projection: [B, hidden_dim] → [B, n_classes, rank]
      5. Output gene embeddings: [n_genes_out, rank]
      6. Logits: [B, n_classes, rank] @ [rank, n_genes_out] → [B, n_classes, n_genes_out]

    Total trainable params: ~9M (reduced from 17.1M in parent)
    Breakdown:
      - input_proj: LN(256) + Linear(256→384) + GELU = ~99K
      - 5 ResBlocks(384, expand=4): 5 × 2 × (384×1536) = ~5.9M
      - output_norm: LN(384) = 768
      - bilinear_proj: Linear(384, 3×384, bias=False) = 442,368
      - out_gene_emb: Embedding(6640, 384) = 2,549,760
      - fallback_emb: 256 (in parent model)
    """

    def __init__(
        self,
        gnn_dim:     int   = GNN_DIM,
        hidden_dim:  int   = 384,
        n_layers:    int   = 5,
        expand:      int   = 4,
        rank:        int   = 384,
        dropout:     float = 0.4,
        n_classes:   int   = N_CLASSES,
        n_genes_out: int   = N_GENES_OUT,
    ):
        super().__init__()
        self.rank     = rank
        self.n_classes = n_classes

        # Input projection (pre-norm design)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(gnn_dim),
            nn.Linear(gnn_dim, hidden_dim),
            nn.GELU(),
        )

        # Intermediate-capacity residual MLP stack (5 layers, hidden=384)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim → n_classes * rank
        self.bilinear_proj = nn.Linear(hidden_dim, n_classes * rank, bias=False)

        # Output gene embeddings [n_genes_out, rank]
        self.out_gene_emb = nn.Embedding(n_genes_out, rank)
        nn.init.xavier_uniform_(self.out_gene_emb.weight)

    def forward(self, pert_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pert_emb: [B, gnn_dim] perturbed gene PPI embedding
        Returns:
            logits: [B, n_classes, n_genes_out]
        """
        # Input projection: [B, gnn_dim] → [B, hidden_dim]
        h = self.input_proj(pert_emb)

        # Intermediate-capacity residual MLP: [B, hidden_dim]
        for block in self.residual_blocks:
            h = block(h)
        h = self.output_norm(h)

        # Bilinear: [B, hidden_dim] → [B, n_classes, rank]
        proj = self.bilinear_proj(h).view(-1, self.n_classes, self.rank)

        # [B, n_classes, rank] @ [rank, n_genes_out] → [B, n_classes, n_genes_out]
        out_embs = self.out_gene_emb.weight  # [n_genes_out, rank]
        logits = torch.matmul(proj, out_embs.T)
        return logits


# ─── Main Model ───────────────────────────────────────────────────────────────

class FrozenStringGNNModel(nn.Module):
    """
    Fully frozen STRING_GNN model:
      - ALL GCN layers (mps.0-7) + embedding table + post_mp: fully frozen
      - Embeddings precomputed once and cached (no gradient through backbone)
      - 5-layer intermediate-capacity bilinear MLP head (~9M trainable params)
      - Fallback learned embedding for genes not in STRING_GNN

    Key change: Reduced head capacity (rank=384, 5 layers, hidden=384 → ~9M) vs
    parent (rank=512, 6 layers, hidden=512 → ~17.1M) to address overfitting.
    """

    def __init__(
        self,
        edge_index:  torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        n_nodes:     int,
        head_hidden: int   = 384,
        n_layers:    int   = 5,
        expand:      int   = 4,
        rank:        int   = 384,
        dropout:     float = 0.4,
    ):
        super().__init__()
        self.n_nodes = n_nodes

        # Load STRING_GNN backbone
        self.gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        # Register graph tensors as buffers
        self.register_buffer("edge_index", edge_index)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        # Freeze ALL backbone parameters (embedding table + all 8 GCN layers + post_mp)
        self._freeze_all_backbone()

        # Precomputed buffer for frozen backbone embeddings (computed once, cached forever)
        self._frozen_embed_cache: Optional[torch.Tensor] = None

        # Fallback embedding for genes not in STRING_GNN
        self.fallback_emb = nn.Parameter(torch.randn(GNN_DIM) * 0.02)

        # Intermediate-capacity bilinear prediction head (~9M trainable params)
        self.head = GNNBilinearHead(
            gnn_dim=GNN_DIM,
            hidden_dim=head_hidden,
            n_layers=n_layers,
            expand=expand,
            rank=rank,
            dropout=dropout,
        )

        n_total   = sum(p.numel() for p in self.parameters())
        n_train   = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Node4-1-1-1-2-1] Total params: {n_total:,} | Trainable: {n_train:,}")

    def _freeze_all_backbone(self):
        """Freeze ALL STRING_GNN backbone parameters."""
        # Freeze embedding table
        for param in self.gnn.emb.parameters():
            param.requires_grad = False
        # Freeze ALL 8 GCN layers
        for i in range(8):
            for param in self.gnn.mps[i].parameters():
                param.requires_grad = False
        # Freeze post_mp
        for param in self.gnn.post_mp.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _get_frozen_embeddings(self) -> torch.Tensor:
        """
        Get frozen STRING_GNN embeddings.
        With fully frozen backbone, embeddings are constant throughout training.
        Cache them on first call for efficiency.
        """
        device = self.edge_index.device

        # Check if cache is valid and on the correct device
        if self._frozen_embed_cache is not None:
            if self._frozen_embed_cache.device == device:
                return self._frozen_embed_cache

        # Compute embeddings with no gradient
        ew = self.edge_weight
        out = self.gnn(
            edge_index=self.edge_index,
            edge_weight=ew,
        )
        embs = out.last_hidden_state  # [n_nodes, 256]

        # Cache the computed embeddings (permanent for fully frozen backbone)
        self._frozen_embed_cache = embs.detach()
        return self._frozen_embed_cache

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_indices: [B] long - STRING_GNN node indices (-1 for unknown)
        Returns:
            logits: [B, 3, 6640]
        """
        # Get frozen embeddings (no gradient, cached after first call)
        all_embs = self._get_frozen_embeddings()  # [n_nodes, 256]

        # Extract perturbation gene embeddings
        safe_idxs = node_indices.clamp(min=0)  # replace -1 with 0 temporarily
        pert_emb  = all_embs[safe_idxs]         # [B, 256]

        # Replace unknown genes with fallback embedding
        known_mask = (node_indices >= 0)
        if not known_mask.all():
            fallback = self.fallback_emb.float().unsqueeze(0).expand_as(pert_emb)
            pert_emb = torch.where(
                known_mask.unsqueeze(-1).expand_as(pert_emb),
                pert_emb, fallback,
            )

        pert_emb = pert_emb.float()  # ensure float32

        # Run bilinear head: [B, 256] → [B, 3, 6640]
        return self.head(pert_emb)

    def head_muon_params(self) -> List[torch.Tensor]:
        """2D weight matrices in ResidualBlocks and bilinear_proj for Muon optimizer."""
        params = []
        for block in self.head.residual_blocks:
            # fc1.weight and fc2.weight are 2D matrices in the block
            for name, module in block.net.named_modules():
                if isinstance(module, nn.Linear):
                    params.append(module.weight)
        # bilinear_proj.weight is also a 2D matrix
        params.append(self.head.bilinear_proj.weight)
        return params

    def head_adamw_params(self) -> List[torch.Tensor]:
        """Non-matrix head parameters + fallback embedding for AdamW optimizer."""
        muon_param_ids = {id(p) for p in self.head_muon_params()}
        params = []
        # Head parameters not in muon group
        for p in self.head.parameters():
            if id(p) not in muon_param_ids:
                params.append(p)
        # Fallback embedding
        params.append(self.fallback_emb)
        return params


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))
    pad    = max_sz - local_p.shape[0]
    p = local_p.to(device);  l = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)
    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p);  dist.all_gather(gl, l)
    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class FrozenStringGNNLitModule(pl.LightningModule):

    def __init__(
        self,
        muon_lr:           float = 5e-3,
        head_adamw_lr:     float = 5e-4,
        head_adamw_wd:     float = 5e-3,
        focal_gamma:       float = 2.0,
        label_smoothing:   float = 0.05,
        dropout:           float = 0.4,
        head_hidden:       int   = 384,
        rank:              int   = 384,
        n_layers:          int   = 5,
        warmup_steps:      int   = 500,
        T_0:               int   = 600,
        T_mult:            int   = 1,
        eta_min_factor:    float = 0.05,
        max_epochs:        int   = 300,
        label_pos_dropout: float = 0.10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str]  = []
        self._test_symbols:  List[str]  = []
        self._test_labels:   List[torch.Tensor] = []

        self._edge_index:  Optional[torch.Tensor] = None
        self._edge_weight: Optional[torch.Tensor] = None
        self._n_nodes: int = 18870

    def setup(self, stage=None):
        dm = self.trainer.datamodule if self.trainer is not None else None
        if dm is not None and hasattr(dm, "edge_index"):
            self._edge_index  = dm.edge_index
            self._edge_weight = dm.edge_weight
            self._n_nodes     = dm.n_nodes

        self.model = FrozenStringGNNModel(
            edge_index  = self._edge_index,
            edge_weight = self._edge_weight,
            n_nodes     = self._n_nodes,
            head_hidden = self.hparams.head_hidden,
            n_layers    = self.hparams.n_layers,
            rank        = self.hparams.rank,
            dropout     = self.hparams.dropout,
        )
        # Ensure trainable parameters are float32
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.register_buffer("focal_class_weights", FOCAL_CLASS_WEIGHTS)

    def forward(self, node_indices):
        return self.model(node_indices)

    def _loss(self, logits, labels):
        # Reshape for loss computation
        logits_2d = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)  # [B*G, 3]
        labels_1d = labels.reshape(-1)                               # [B*G]
        return focal_cross_entropy(
            logits_2d, labels_1d,
            gamma=self.hparams.focal_gamma,
            class_weights=self.focal_class_weights.to(logits.device),
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        labels = batch["label"]

        # Label position dropout: randomly mask gene positions to neutral class during training
        # This regularizes against memorizing specific gene position patterns.
        # Applied only during training, not validation/test.
        if self.hparams.label_pos_dropout > 0.0 and self.training:
            mask = torch.rand(labels.shape, device=labels.device) < self.hparams.label_pos_dropout
            labels = labels.clone()
            labels[mask] = 1  # class index 1 = neutral (unchanged)

        loss = self._loss(logits, labels)
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
        self._val_preds.clear();  self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        probs  = torch.softmax(logits, dim=1)
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

        # Deduplicate by pert_id (DDP DistributedSampler may pad)
        seen_pids: set = set()
        keep_indices: List[int] = []
        for i, pid in enumerate(all_pert):
            if pid not in seen_pids:
                seen_pids.add(pid)
                keep_indices.append(i)
        if len(keep_indices) < len(all_pert):
            self.print(f"[Node4-1-1-1-2-1] Deduplicating: {len(all_pert)} -> {len(keep_indices)}")
            all_probs  = all_probs[keep_indices]
            all_labels = all_labels[keep_indices]
            all_pert   = [all_pert[i] for i in keep_indices]
            all_syms   = [all_syms[i]  for i in keep_indices]

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pid, sym, probs in zip(all_pert, all_syms, all_probs.numpy()):
                    fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")
            self.print(f"[Node4-1-1-1-2-1] Saved test predictions -> {pred_path}")
            if all_labels.any():
                f1 = compute_per_gene_f1(all_probs.numpy(), all_labels.numpy())
                self.print(f"[Node4-1-1-1-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Parameter groups:
        # 1. Muon group: 2D weight matrices in ResidualBlocks + bilinear_proj
        #    (fc1.weight, fc2.weight, bilinear_proj.weight) — intermediate capacity
        # 2. AdamW group: all other head params + fallback_emb
        #    (norms, biases, out_gene_emb, input_proj, etc.)
        muon_params  = self.model.head_muon_params()
        adamw_params = self.model.head_adamw_params()

        # Verify no overlap between groups
        muon_ids  = {id(p) for p in muon_params}
        adamw_ids = {id(p) for p in adamw_params}
        assert not muon_ids.intersection(adamw_ids), "Overlap between Muon and AdamW groups!"

        # MuonWithAuxAdam: single optimizer handling both Muon and AdamW parameter groups
        optimizer = MuonWithAuxAdam([
            # Muon group for 2D hidden weight matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                momentum=0.95,
                weight_decay=hp.head_adamw_wd,  # Apply weight decay to Muon params too
            ),
            # AdamW group for non-matrix params (norms, biases, embeddings)
            dict(
                params=adamw_params,
                use_muon=False,
                lr=hp.head_adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.head_adamw_wd,
            ),
        ])

        # Cosine warm restarts with linear warmup
        # T_0=600 steps (~27 epochs at 44 steps/epoch for 1416 samples / 8 batch / 4 accum on 2 GPUs)
        # T_mult=1: same-length cycles (proven in node1-2-2-2-1 with 6 cycles giving staircase)
        # warmup_steps=500: stabilizes initial training (~11 epochs before full LR)
        def lr_lambda(step: int) -> float:
            if step < hp.warmup_steps:
                # Linear warmup
                return float(step) / max(1, hp.warmup_steps)
            # Cosine warm restarts
            step_after_warmup = step - hp.warmup_steps
            T_0 = hp.T_0
            T_mult = hp.T_mult
            if T_mult == 1:
                # Simple case: same cycle length
                t_in_cycle = step_after_warmup % T_0
                T_cur = T_0
            else:
                # Growing cycles: T_0, T_0*T_mult, T_0*T_mult^2, ...
                T_cur = T_0
                t_in_cycle = step_after_warmup
                while t_in_cycle >= T_cur:
                    t_in_cycle -= T_cur
                    T_cur = int(T_cur * T_mult)
            progress = float(t_in_cycle) / float(T_cur)
            # Cosine from 1.0 to eta_min_factor
            cos_val = 0.5 * (1.0 + np.cos(np.pi * progress))
            return hp.eta_min_factor + (1.0 - hp.eta_min_factor) * cos_val

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[lr_lambda, lr_lambda],  # Same schedule for both param groups
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
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
        description="Node 4-1-1-1-2-1 – Intermediate Capacity Head + Stronger Regularization + Gradient Clipping + Label Position Dropout"
    )
    p.add_argument("--data-dir",            type=str,   default="data")
    p.add_argument("--muon-lr",             type=float, default=5e-3)
    p.add_argument("--head-adamw-lr",       type=float, default=5e-4)
    p.add_argument("--head-adamw-wd",       type=float, default=5e-3)
    p.add_argument("--focal-gamma",         type=float, default=2.0)
    p.add_argument("--label-smoothing",     type=float, default=0.05)
    p.add_argument("--dropout",             type=float, default=0.4)
    p.add_argument("--head-hidden",         type=int,   default=384)
    p.add_argument("--rank",                type=int,   default=384)
    p.add_argument("--n-layers",            type=int,   default=5)
    p.add_argument("--warmup-steps",        type=int,   default=500)
    p.add_argument("--T-0",                 type=int,   default=600)
    p.add_argument("--T-mult",              type=int,   default=1)
    p.add_argument("--eta-min-factor",      type=float, default=0.05)
    p.add_argument("--label-pos-dropout",   type=float, default=0.10)
    p.add_argument("--gradient-clip-val",   type=float, default=1.0)
    p.add_argument("--micro-batch-size",    type=int,   default=8)
    p.add_argument("--global-batch-size",   type=int,   default=64)
    p.add_argument("--max-epochs",          type=int,   default=300)
    p.add_argument("--patience",            type=int,   default=50)
    p.add_argument("--num-workers",         type=int,   default=4)
    p.add_argument("--val-check-interval",  type=float, default=1.0)
    p.add_argument("--debug-max-step",      type=int,   default=None)
    p.add_argument("--fast-dev-run",        action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm  = PerturbationDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = FrozenStringGNNLitModule(
        muon_lr           = args.muon_lr,
        head_adamw_lr     = args.head_adamw_lr,
        head_adamw_wd     = args.head_adamw_wd,
        focal_gamma       = args.focal_gamma,
        label_smoothing   = args.label_smoothing,
        dropout           = args.dropout,
        head_hidden       = args.head_hidden,
        rank              = args.rank,
        n_layers          = args.n_layers,
        warmup_steps      = args.warmup_steps,
        T_0               = args.T_0,
        T_mult            = args.T_mult,
        eta_min_factor    = args.eta_min_factor,
        max_epochs        = args.max_epochs,
        label_pos_dropout = args.label_pos_dropout,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
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
        max_steps   = args.debug_max_step;  limit_train = args.debug_max_step
        limit_val   = 2;  limit_test = 2
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
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=args.gradient_clip_val,  # Prevents gradient explosion
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 4-1-1-1-2-1 – Intermediate Capacity Head (rank=384, 5-layer) + "
            "Stronger Regularization + Gradient Clipping + Label Position Dropout\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
