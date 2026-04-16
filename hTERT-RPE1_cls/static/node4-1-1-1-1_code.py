"""
Node 4-1-1-1-1 – Reduced-Capacity Head + Muon Optimizer + Warm Restarts + Label Smoothing

Architecture:
  - STRING_GNN partial backbone fine-tuning (same as parent):
      * Layers mps.0–5: frozen (precomputed once)
      * Layers mps.6, mps.7 + post_mp (~198K params): trainable at backbone_lr=5e-5 (AdamW)
  - REDUCED 4-layer deep residual bilinear MLP head:
      * rank=256 (was 512), n_layers=4 (was 6): ~4.5M params (was 16.9M)
      * ResidualBlock: [LN → Linear(dim→dim*expand) → GELU → Dropout → Linear(dim*expand→dim) + skip]
      * Bilinear output: [B, 3, rank=256] @ [rank=256, 6640] → [B, 3, 6640]
  - Muon optimizer for head 2D matrix weights (proven effective in tree-best nodes)
  - AdamW for backbone + head embeddings/norms/biases
  - Focal cross-entropy loss: gamma=2.0, class weights [down=2.0, neutral=0.5, up=4.0]
  - Label smoothing: 0.05 (suppresses overconfident predictions)
  - Cosine annealing with warm restarts (T_0=1200 steps, T_mult=2: growing cycles)
  - Extended warmup: 500 steps (~11 epochs) to address early training instability
  - dropout=0.35 in residual blocks (slightly stronger than parent's 0.3)

Key improvements vs Node 4-1-1-1 (parent, test F1=0.4770):
  1. [PRIMARY] Reduce head capacity: rank 512→256, n_layers 6→4
     Parent's 16.9M head vs 1,416 samples (12,000:1 ratio) causes 15.51x val/train loss gap.
     Reduced 4.5M head brings ratio to ~3,200:1 which is more manageable.
     Directly addresses the feedback's #1 highest-priority recommendation.
  2. [PRIMARY] Muon optimizer for 2D head weight matrices
     Tree-best node1-2-2-2-1 (F1=0.5099) and node1-2-2-2 (F1=0.5060) use Muon optimizer.
     Muon achieves faster convergence via orthogonalized momentum updates.
     Applied to all Linear weight matrices in head ResidualBlocks + bilinear_proj.
     AdamW handles backbone, embeddings, norms, biases.
  3. [SECONDARY] Cosine warm restarts (T_0=1200 steps, T_mult=2)
     node1-2-2-2-1 showed "staircase improvement" across 6 cycles with warm restarts.
     T_mult=2 ensures growing cycle lengths for deeper convergence per cycle.
     This is fundamentally different from parent's single-cycle cosine.
  4. [SECONDARY] Extended warmup: 200→500 steps
     Parent had 41 val_f1 regressions > 0.005 in epochs 0-40.
     500-step warmup ≈ 11 epochs provides stable ramp before cosine decay begins.
  5. [SECONDARY] Label smoothing: 0.0→0.05
     Reduces overconfident predictions (parent val/train ratio: 15.51x).
     Consistent with feedback recommendation for dropout=0.4-0.5 node variant.
  6. Dropout 0.3→0.35 in ResidualBlocks
     Slightly stronger regularization for the already-reduced capacity head.

Inspired by:
  - node4-1-1-1/memory/feedback.md: reduce rank→256, n_layers→4, head_wd→3e-3-5e-3, label_smoothing=0.05
  - node1-2-2-2-1 (tree best F1=0.5099): Muon + warm restarts + partial backbone FT
  - node1-2-2-2 (F1=0.5060): Muon optimizer for 2D head matrices
  - node2-1-3 (F1=0.5047): partial STRING_GNN (mps.6,7+post_mp) + focal loss + class weights [2,0.5,4]
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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# Muon optimizer (tree-best node1-2-2-2-1 uses Muon for hidden matrix params)
from muon import MuonWithAuxAdam

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_DIM        = 256

# Focal loss class weights: proven effective in tree-best nodes
# Train distribution: 8.14% down, 88.86% neutral, 3.00% up
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
        print(f"[Node4-1-1-1-1] {n_unknown}/{total} samples not found in STRING_GNN graph "
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
    The 2D weight matrices (Linear layers) are optimized by Muon; norms/biases by AdamW.
    """
    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.35):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim * expand)
        self.act  = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2  = nn.Linear(dim * expand, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + h


# ─── Bilinear Head ────────────────────────────────────────────────────────────

class GNNBilinearHead(nn.Module):
    """
    REDUCED-CAPACITY deep residual MLP + bilinear interaction head.
    rank=256 (was 512), n_layers=4 (was 6): ~4.5M params vs parent's 16.9M.
    This addresses the 12,000:1 parameter:sample overfitting root cause.

    Input: [B, gnn_dim=256] PPI embedding of perturbed gene
    Output: [B, n_classes=3, n_genes_out=6640] logits

    Architecture:
      1. Input projection: LayerNorm(gnn_dim) → Linear(gnn_dim → hidden_dim) → GELU
      2. N_layers=4 residual MLP blocks (hidden_dim=512, expand=4, dropout=0.35)
      3. Output norm: LayerNorm(hidden_dim)
      4. Bilinear projection: [B, hidden_dim] → [B, n_classes, rank=256]
      5. Output gene embeddings: [n_genes_out, rank=256]
      6. Logits: [B, n_classes, rank=256] @ [rank=256, n_genes_out] → [B, n_classes, n_genes_out]

    Muon targets: fc1.weight, fc2.weight in each ResidualBlock, and bilinear_proj.weight
    AdamW targets: norms, biases, input_proj, out_gene_emb
    """

    def __init__(
        self,
        gnn_dim:     int   = GNN_DIM,
        hidden_dim:  int   = 512,
        n_layers:    int   = 4,     # reduced from 6
        expand:      int   = 4,
        rank:        int   = 256,   # reduced from 512
        dropout:     float = 0.35,
        n_classes:   int   = N_CLASSES,
        n_genes_out: int   = N_GENES_OUT,
    ):
        super().__init__()
        self.rank      = rank
        self.n_classes = n_classes

        # Input projection (pre-norm design)
        # Note: input_proj Linear uses AdamW (it's the first projection layer)
        self.input_norm = nn.LayerNorm(gnn_dim)
        self.input_proj = nn.Linear(gnn_dim, hidden_dim)
        self.input_act  = nn.GELU()

        # REDUCED: 4 residual MLP blocks (was 6)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim → n_classes * rank
        # bilinear_proj.weight is a 2D matrix → Muon target
        self.bilinear_proj = nn.Linear(hidden_dim, n_classes * rank, bias=False)

        # Output gene embeddings [n_genes_out, rank] → AdamW (embedding-like)
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
        h = self.input_norm(pert_emb)
        h = self.input_proj(h)
        h = self.input_act(h)

        # Deep residual MLP: [B, hidden_dim]
        for block in self.residual_blocks:
            h = block(h)
        h = self.output_norm(h)

        # Bilinear: [B, hidden_dim] → [B, n_classes, rank]
        proj = self.bilinear_proj(h).view(-1, self.n_classes, self.rank)

        # [B, n_classes, rank] @ [rank, n_genes_out] → [B, n_classes, n_genes_out]
        out_embs = self.out_gene_emb.weight  # [n_genes_out, rank]
        logits = torch.matmul(proj, out_embs.T)
        return logits

    def muon_matrix_params(self):
        """Return 2D weight matrices suitable for Muon optimizer."""
        params = []
        for block in self.residual_blocks:
            params.append(block.fc1.weight)
            params.append(block.fc2.weight)
        params.append(self.bilinear_proj.weight)
        return params

    def adamw_params(self):
        """Return all other parameters (norms, biases, embeddings, input proj) for AdamW."""
        muon_set = set(id(p) for p in self.muon_matrix_params())
        return [p for p in self.parameters() if id(p) not in muon_set]


# ─── Main Model ───────────────────────────────────────────────────────────────

class PartialStringGNNModel(nn.Module):
    """
    Partial STRING_GNN fine-tuning model (same as parent node):
      - First 6 GCN layers (mps.0-5): frozen
      - Last 2 GCN layers (mps.6, mps.7) + post_mp: trainable at low lr (AdamW)
      - REDUCED 4-layer rank=256 residual bilinear MLP head: Muon for 2D matrices
      - Fallback learned embedding for genes not in STRING_GNN (AdamW)
    """

    def __init__(
        self,
        edge_index:  torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        n_nodes:     int,
        head_hidden: int   = 512,
        n_layers:    int   = 4,    # reduced from 6
        expand:      int   = 4,
        rank:        int   = 256,  # reduced from 512
        dropout:     float = 0.35,
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

        # Freeze layers mps.0-5 and embedding table
        self._freeze_first_layers()

        # Fallback embedding for genes not in STRING_GNN (AdamW target)
        self.fallback_emb = nn.Parameter(torch.randn(GNN_DIM) * 0.02)

        # REDUCED bilinear prediction head
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
        n_muon    = sum(p.numel() for p in self.head.muon_matrix_params())
        print(f"[Node4-1-1-1-1] Total params: {n_total:,} | Trainable: {n_train:,} | Muon: {n_muon:,}")

    def _freeze_first_layers(self):
        """Freeze mps.0-5 and embedding table. Keep mps.6, mps.7, post_mp trainable."""
        for param in self.gnn.emb.parameters():
            param.requires_grad = False
        for i in range(6):
            for param in self.gnn.mps[i].parameters():
                param.requires_grad = False

    def _compute_embeddings(self) -> torch.Tensor:
        """
        Run full STRING_GNN forward pass. Trainable layers (mps.6, mps.7, post_mp)
        produce gradients; frozen layers do not.
        Returns: [n_nodes, 256] node embeddings
        """
        out = self.gnn(
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
        )
        return out.last_hidden_state  # [n_nodes, 256]

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_indices: [B] long - STRING_GNN node indices (-1 for unknown)
        Returns:
            logits: [B, 3, 6640]
        """
        # Compute all node embeddings (with gradients for trainable backbone layers)
        all_embs = self._compute_embeddings()  # [n_nodes, 256]

        # Extract perturbation gene embeddings
        safe_idxs = node_indices.clamp(min=0)
        pert_emb  = all_embs[safe_idxs]  # [B, 256]

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

    def backbone_parameters(self):
        """Parameters of trainable backbone layers (mps.6, mps.7, post_mp) for AdamW."""
        params = []
        for i in [6, 7]:
            params.extend(self.gnn.mps[i].parameters())
        params.extend(self.gnn.post_mp.parameters())
        return params

    def head_muon_params(self):
        """2D weight matrices in head for Muon optimizer."""
        return self.head.muon_matrix_params()

    def head_adamw_params(self):
        """Head parameters that should use AdamW (embeddings, norms, biases, input proj)."""
        return self.head.adamw_params() + [self.fallback_emb]


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

class PartialStringGNNLitModule(pl.LightningModule):

    def __init__(
        self,
        backbone_lr:     float = 5e-5,
        backbone_wd:     float = 1e-4,
        muon_lr:         float = 0.005,  # Muon LR for 2D head matrices
        muon_wd:         float = 0.0,    # Muon weight decay (orthogonalization handles regularization)
        head_adamw_lr:   float = 5e-4,   # AdamW LR for head non-matrix params
        head_adamw_wd:   float = 3e-3,   # Stronger WD for head (3e-3 per feedback)
        focal_gamma:     float = 2.0,
        label_smoothing: float = 0.05,   # suppress overconfident predictions
        dropout:         float = 0.35,
        head_hidden:     int   = 512,
        rank:            int   = 256,    # reduced from 512
        n_layers:        int   = 4,      # reduced from 6
        warmup_steps:    int   = 500,    # extended from 200 to 500
        # Cosine warm restart parameters
        t0_steps:        int   = 1200,   # T_0: steps per first cycle (~27 epochs at 44 steps/epoch)
        t_mult:          int   = 2,      # T_mult: growing cycle lengths
        eta_min_factor:  float = 0.05,   # minimum LR fraction
        max_epochs:      int   = 300,    # extended to allow multiple restart cycles
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

        self.model = PartialStringGNNModel(
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
            self.print(f"[Node4-1-1-1-1] Deduplicating: {len(all_pert)} -> {len(keep_indices)}")
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
            self.print(f"[Node4-1-1-1-1] Saved test predictions -> {pred_path}")
            if all_labels.any():
                f1 = compute_per_gene_f1(all_probs.numpy(), all_labels.numpy())
                self.print(f"[Node4-1-1-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        backbone_params  = self.model.backbone_parameters()
        head_muon_params = self.model.head_muon_params()
        head_adamw_params = self.model.head_adamw_params()

        # MuonWithAuxAdam: handles both Muon (for 2D matrices) and AdamW (for everything else)
        # This is the same approach as tree-best node1-2-2-2-1 (F1=0.5099) and node1-2-2-2 (F1=0.5060)
        param_groups = [
            # Group 0: Muon for 2D head weight matrices (fc1.weight, fc2.weight, bilinear_proj.weight)
            dict(
                params=head_muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.muon_wd,
                momentum=0.95,
            ),
            # Group 1: AdamW for head non-matrix parameters (norms, biases, embeddings, input_proj)
            dict(
                params=head_adamw_params,
                use_muon=False,
                lr=hp.head_adamw_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=hp.head_adamw_wd,
            ),
            # Group 2: AdamW for trainable backbone (mps.6, mps.7, post_mp)
            dict(
                params=backbone_params,
                use_muon=False,
                lr=hp.backbone_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=hp.backbone_wd,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # Cosine warm restarts with growing cycle lengths (T_mult=2)
        # Strategy from node1-2-2-2-1 (tree best F1=0.5099):
        # "warm restarts provided genuine staircase-like improvement across 6 cycles"
        # "T_mult=2 for geometrically growing cycles" (recommended next step in node1-2-2-2-1 feedback)
        #
        # Linear warmup phase: first warmup_steps steps ramp from 0 to peak LR
        # After warmup: CosineAnnealingWarmRestarts handles the cycling
        # T_0=1200 steps ≈ 27 epochs; T_mult=2: cycles of 27, 54, 108, 216 epochs
        # This gives deep convergence within each progressively longer cycle.

        def make_lr_lambda(base_lr):
            def lr_lambda(step):
                if step < hp.warmup_steps:
                    # Linear warmup
                    return float(step) / max(1, hp.warmup_steps)
                # After warmup: cosine warm restarts
                t = step - hp.warmup_steps
                T_cur, T_0 = t, hp.t0_steps
                T_mult = hp.t_mult

                # Find which cycle we're in and progress within that cycle
                if T_mult == 1:
                    cycle = t // T_0
                    T_cur_in_cycle = t % T_0
                    T_i = T_0
                else:
                    # Geometric series: T_i = T_0 * T_mult^i
                    # Find cumulative sum at cycle boundaries
                    cumulative = 0
                    T_i = T_0
                    cycle = 0
                    while cumulative + T_i <= t:
                        cumulative += T_i
                        T_i = int(T_i * T_mult)
                        cycle += 1
                    T_cur_in_cycle = t - cumulative

                # Cosine within cycle: from 1.0 to eta_min_factor
                progress = float(T_cur_in_cycle) / max(1, T_i)
                progress = min(progress, 1.0)
                cos_val = 0.5 * (1.0 + np.cos(np.pi * progress))
                return hp.eta_min_factor + (1.0 - hp.eta_min_factor) * cos_val
            return lr_lambda

        # Each param group gets its own lambda (scales its base lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                make_lr_lambda(hp.muon_lr),
                make_lr_lambda(hp.head_adamw_lr),
                make_lr_lambda(hp.backbone_lr),
            ],
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
        description="Node 4-1-1-1-1 – Reduced Head + Muon Optimizer + Warm Restarts + Label Smoothing"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--backbone-lr",       type=float, default=5e-5)
    p.add_argument("--backbone-wd",       type=float, default=1e-4)
    p.add_argument("--muon-lr",           type=float, default=0.005)
    p.add_argument("--muon-wd",           type=float, default=0.0)
    p.add_argument("--head-adamw-lr",     type=float, default=5e-4)
    p.add_argument("--head-adamw-wd",     type=float, default=3e-3)
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--label-smoothing",   type=float, default=0.05)
    p.add_argument("--dropout",           type=float, default=0.35)
    p.add_argument("--head-hidden",       type=int,   default=512)
    p.add_argument("--rank",              type=int,   default=256)
    p.add_argument("--n-layers",          type=int,   default=4)
    p.add_argument("--micro-batch-size",  type=int,   default=8)
    p.add_argument("--global-batch-size", type=int,   default=64)
    p.add_argument("--max-epochs",        type=int,   default=300)
    p.add_argument("--patience",          type=int,   default=80)
    p.add_argument("--warmup-steps",      type=int,   default=500)
    p.add_argument("--t0-steps",          type=int,   default=1200)
    p.add_argument("--t-mult",            type=int,   default=2)
    p.add_argument("--eta-min-factor",    type=float, default=0.05)
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

    dm  = PerturbationDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = PartialStringGNNLitModule(
        backbone_lr     = args.backbone_lr,
        backbone_wd     = args.backbone_wd,
        muon_lr         = args.muon_lr,
        muon_wd         = args.muon_wd,
        head_adamw_lr   = args.head_adamw_lr,
        head_adamw_wd   = args.head_adamw_wd,
        focal_gamma     = args.focal_gamma,
        label_smoothing = args.label_smoothing,
        dropout         = args.dropout,
        head_hidden     = args.head_hidden,
        rank            = args.rank,
        n_layers        = args.n_layers,
        warmup_steps    = args.warmup_steps,
        t0_steps        = args.t0_steps,
        t_mult          = args.t_mult,
        eta_min_factor  = args.eta_min_factor,
        max_epochs      = args.max_epochs,
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
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 4-1-1-1-1 – Reduced Head (rank=256, n_layers=4) + Muon + Warm Restarts + Label Smoothing\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
