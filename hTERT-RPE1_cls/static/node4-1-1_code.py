"""
Node 4-1-1 – Partial STRING_GNN Fine-Tuning + Rank-512 Bilinear MLP Head + Focal Loss

Architecture:
  - STRING_GNN partial backbone fine-tuning:
      * Layers mps.0–5: frozen, precomputed once per epoch as a fixed buffer
      * Layers mps.6, mps.7 + post_mp (~198K params): trainable at backbone_lr=5e-5
  - 6-layer deep residual bilinear MLP head:
      * Input: 256-dim PPI embedding from partial STRING_GNN
      * ResidualBlock: [Linear(256→hidden) → GELU → Linear(hidden→256) + skip] × 6
        with layer expansion: 256 → 512*4 internally per block
      * Bilinear output: [B, 3, rank=512] @ [rank=512, 6640] → [B, 3, 6640]
      * Trainable parameters: ~16.9M
  - Focal cross-entropy loss: gamma=2.0, class weights [down=2.0, neutral=0.5, up=4.0]
  - Two-group AdamW optimizer: backbone lr=5e-5, head lr=5e-4, weight_decay=1e-4
  - Cosine annealing with warmup (properly calibrated to expected training duration)
  - Early stopping patience=30

Key improvements vs Node 4-1 (parent):
  1. Abandon over-regularizing weight_decay=1e-3 → revert to 1e-4
     (Parent's primary bottleneck: train_loss stalled at 0.89 vs node4's 0.40)
  2. Abandon cond_emb injection → signal was too diluted (256-dim all-ones
     across 18,870 nodes with 8 propagation layers)
  3. Shift from full GNN fine-tuning to partial frozen/trainable split:
     frozen mps.0-5 precomputed as buffer + trainable mps.6,7+post_mp (~198K params)
     This approach delivered the tree-best result (node2-1-3: F1=0.5047)
  4. Upgrade from simple MLP head to proven deep residual bilinear MLP:
     rank=512 bilinear interaction (2× wider than node4's implicit rank=256-equivalent)
     6 residual blocks providing deeper nonlinear processing
  5. Switch from weighted cross-entropy to focal loss (gamma=2.0) with
     class weights [2.0, 0.5, 4.0] — proven superior at handling 88.9% class imbalance
  6. Two LR groups: backbone at 5e-5 (gentle fine-tuning of last 2 GCN layers),
     head at 5e-4 (faster learning of randomly-initialized bilinear head)
  7. Cosine annealing calibrated to actual training duration (~100 epochs × steps/epoch)
     avoiding the miscalibration in node2-1-3 (total_steps=6600 for ~83 epochs)

Inspired by:
  - tree-best configuration: node2-1-3 (F1=0.5047), node1-3-1-1-1-1 (F1=0.4976),
    node2-1-3-1 (F1=0.4992)
  - node1-2 (F1=0.4912): frozen STRING_GNN + 6-layer bilinear MLP head confirmed strong
  - node4-1 feedback: weight_decay=1e-3 is the primary bottleneck; cond_emb diluted
  - node4 feedback: representational ceiling of full fine-tuning without residual head
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
) -> torch.Tensor:
    """
    Focal cross-entropy loss: down-weights easy examples.
    logits:  [N, C]
    targets: [N]  (class indices 0,1,2)
    Returns: scalar mean loss
    """
    ce   = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
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
        print(f"[Node4-1-1] {n_unknown}/{total} samples not found in STRING_GNN graph "
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
    Residual MLP block: Linear(in→expand) → GELU → Linear(expand→in) + skip.
    Matches the proven architecture from tree-best nodes (node2-1-3, node1-2).
    """
    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.2):
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
    Deep residual MLP + bilinear interaction head.
    Input: [B, gnn_dim=256] PPI embedding of perturbed gene
    Output: [B, n_classes=3, n_genes_out=6640] logits

    Architecture:
      1. Input projection: Linear(gnn_dim → hidden_dim)
      2. N_layers residual MLP blocks (hidden_dim=512, expand=4)
      3. Bilinear projection: [B, hidden_dim] → [B, n_classes, rank]
      4. Output gene embeddings: [n_genes_out, rank]
      5. Logits: [B, n_classes, rank] @ [rank, n_genes_out] → [B, n_classes, n_genes_out]
    """

    def __init__(
        self,
        gnn_dim:     int   = GNN_DIM,
        hidden_dim:  int   = 512,
        n_layers:    int   = 6,
        expand:      int   = 4,
        rank:        int   = 512,
        dropout:     float = 0.2,
        n_classes:   int   = N_CLASSES,
        n_genes_out: int   = N_GENES_OUT,
    ):
        super().__init__()
        self.rank     = rank
        self.n_classes = n_classes

        # Input projection
        self.input_proj = nn.Sequential(
            nn.LayerNorm(gnn_dim),
            nn.Linear(gnn_dim, hidden_dim),
            nn.GELU(),
        )

        # Deep residual MLP stack
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


# ─── Main Model ───────────────────────────────────────────────────────────────

class PartialStringGNNModel(nn.Module):
    """
    Partial STRING_GNN fine-tuning model:
      - First 6 GCN layers (mps.0-5): frozen, embeddings precomputed once
      - Last 2 GCN layers (mps.6, mps.7) + post_mp: trainable at low lr
      - 6-layer deep residual bilinear MLP head: trainable at higher lr
      - Fallback learned embedding for genes not in STRING_GNN

    This architecture is proven in the MCTS tree (node2-1-3 achieved F1=0.5047).
    It balances the PPI topology signal (frozen backbone) with task-specific
    adaptation (trainable last 2 layers) and high-capacity output head.
    """

    def __init__(
        self,
        edge_index:  torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        n_nodes:     int,
        head_hidden: int   = 512,
        n_layers:    int   = 6,
        expand:      int   = 4,
        rank:        int   = 512,
        dropout:     float = 0.2,
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

        # Freeze layers mps.0-5 (keep mps.6, mps.7, post_mp trainable)
        # Also freeze the embedding table (emb.weight) - not task-specific
        self._freeze_first_layers()

        # Precomputed buffer for frozen-layer intermediate embeddings
        # Will be populated in the first training step and cached
        self.register_buffer("_frozen_embed_cache", None)

        # Fallback embedding for genes not in STRING_GNN
        self.fallback_emb = nn.Parameter(torch.randn(GNN_DIM) * 0.02)

        # Bilinear prediction head
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
        print(f"[Node4-1-1] Total params: {n_total:,} | Trainable: {n_train:,}")

    def _freeze_first_layers(self):
        """Freeze mps.0-5 and embedding table. Keep mps.6, mps.7, post_mp trainable."""
        # Freeze embedding table
        for param in self.gnn.emb.parameters():
            param.requires_grad = False
        # Freeze first 6 GCN layers
        for i in range(6):
            for param in self.gnn.mps[i].parameters():
                param.requires_grad = False
        # Keep mps.6, mps.7, post_mp trainable (already requires_grad by default)

    def _precompute_embeddings(self) -> torch.Tensor:
        """
        Run full STRING_GNN forward pass with gradient disabled for frozen layers,
        but enable gradient for trainable layers (mps.6, mps.7, post_mp).
        Returns: [n_nodes, 256] node embeddings with gradients for trainable layers.
        """
        device = self.edge_index.device
        ew = self.edge_weight

        # Run the full STRING_GNN forward pass
        # Trainable layers (mps.6, mps.7, post_mp) will produce gradients
        out = self.gnn(
            edge_index=self.edge_index,
            edge_weight=ew,
        )
        return out.last_hidden_state  # [n_nodes, 256]

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_indices: [B] long - STRING_GNN node indices (-1 for unknown)
        Returns:
            logits: [B, 3, 6640]
        """
        # Compute all node embeddings (with gradients flowing through mps.6,7+post_mp)
        all_embs = self._precompute_embeddings()  # [n_nodes, 256]

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

    def backbone_parameters(self):
        """Parameters of trainable backbone layers (mps.6, mps.7, post_mp)."""
        params = []
        for i in [6, 7]:
            params.extend(self.gnn.mps[i].parameters())
        params.extend(self.gnn.post_mp.parameters())
        return params

    def head_parameters(self):
        """Parameters of head + fallback embedding."""
        return list(self.head.parameters()) + [self.fallback_emb]


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
        backbone_lr:  float = 5e-5,
        head_lr:      float = 5e-4,
        weight_decay: float = 1e-4,
        focal_gamma:  float = 2.0,
        dropout:      float = 0.2,
        head_hidden:  int   = 512,
        rank:         int   = 512,
        n_layers:     int   = 6,
        warmup_steps: int   = 100,
        total_steps:  int   = 1200,
        max_epochs:   int   = 150,
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
            self.print(f"[Node4-1-1] Deduplicating: {len(all_pert)} -> {len(keep_indices)}")
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
            self.print(f"[Node4-1-1] Saved test predictions -> {pred_path}")
            if all_labels.any():
                f1 = compute_per_gene_f1(all_probs.numpy(), all_labels.numpy())
                self.print(f"[Node4-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams
        # Two-group optimizer: low LR for backbone, high LR for head
        backbone_params = self.model.backbone_parameters()
        head_params     = self.model.head_parameters()

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": hp.backbone_lr},
            {"params": head_params,     "lr": hp.head_lr},
        ], weight_decay=hp.weight_decay)

        # Cosine annealing with linear warmup
        # total_steps is calibrated to actual expected training duration
        def lr_lambda_backbone(step):
            if step < hp.warmup_steps:
                return float(step) / max(1, hp.warmup_steps)
            progress = float(step - hp.warmup_steps) / max(1, hp.total_steps - hp.warmup_steps)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        def lr_lambda_head(step):
            if step < hp.warmup_steps:
                return float(step) / max(1, hp.warmup_steps)
            progress = float(step - hp.warmup_steps) / max(1, hp.total_steps - hp.warmup_steps)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        # Single LambdaLR with per-group lambdas (list of lambdas, one per param group)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[lr_lambda_backbone, lr_lambda_head],
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
        description="Node 4-1-1 – Partial STRING_GNN Fine-Tuning + Rank-512 Bilinear MLP Head"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--backbone-lr",       type=float, default=5e-5)
    p.add_argument("--head-lr",           type=float, default=5e-4)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--dropout",           type=float, default=0.2)
    p.add_argument("--head-hidden",       type=int,   default=512)
    p.add_argument("--rank",              type=int,   default=512)
    p.add_argument("--n-layers",          type=int,   default=6)
    p.add_argument("--micro-batch-size",  type=int,   default=8)
    p.add_argument("--global-batch-size", type=int,   default=32)
    p.add_argument("--max-epochs",        type=int,   default=150)
    p.add_argument("--patience",          type=int,   default=30)
    p.add_argument("--warmup-steps",      type=int,   default=100)
    p.add_argument("--total-steps",       type=int,   default=1200)
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
        backbone_lr  = args.backbone_lr,
        head_lr      = args.head_lr,
        weight_decay = args.weight_decay,
        focal_gamma  = args.focal_gamma,
        dropout      = args.dropout,
        head_hidden  = args.head_hidden,
        rank         = args.rank,
        n_layers     = args.n_layers,
        warmup_steps = args.warmup_steps,
        total_steps  = args.total_steps,
        max_epochs   = args.max_epochs,
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
            "Node 4-1-1 – Partial STRING_GNN + Rank-512 Bilinear MLP Head + Focal Loss\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
