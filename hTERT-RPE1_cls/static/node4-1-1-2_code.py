"""
Node 4-1-1-2 – Partial STRING_GNN + MuonWithAuxAdam + Reduced Head + SGDR + SWA

Architecture:
  - STRING_GNN partial backbone fine-tuning:
      * Layers mps.0–5: frozen, precomputed once per epoch as a fixed buffer
      * Layers mps.6, mps.7 + post_mp (~198K params): trainable at backbone_lr=1e-5
  - Reduced 4-layer deep residual bilinear MLP head:
      * Input: 256-dim PPI embedding from partial STRING_GNN
      * 4 ResidualBlocks (reduced from 6): [LN → Linear(256→256*4) → GELU → Dropout → Linear(1024→256) + skip]
      * Bilinear output: [B, 3, rank=256] @ [rank=256, 6640] → [B, 3, 6640]
      * Trainable parameters: ~4.4M (vs 16.9M in parent)
  - MuonWithAuxAdam optimizer:
      * Muon lr=0.005 for ResidualBlock 2D weight matrices (faster convergence, orthogonal updates)
      * AdamW lr=5e-4 for head scalars/biases/gene_emb
      * AdamW lr=1e-5 for backbone (mps.6/7+post_mp)
  - SGDR (CosineAnnealingWarmRestarts): T_0=400 steps (~16 epochs/cycle with actual DDP step count)
    Enables multi-cycle exploration — cycle 2 typically outperforms cycle 1
  - Focal cross-entropy loss: gamma=2.0, class weights [2.0, 0.5, 4.0], label_smoothing=0.05
  - Quality-filtered SWA: save periodic checkpoints, average those above val_f1 threshold at the end
  - Dropout=0.4 (stronger regularization for reduced-capacity head)
  - Extended warmup_steps=400 (~16 epochs) to stabilize early training

Key improvements vs Node 4-1-1 (parent):
  1. MuonWithAuxAdam instead of AdamW — proven faster convergence for bilinear head matrices
     (node4-1-2 with Muon achieved F1=0.5060 from same parent lineage)
  2. Reduced head capacity: 4 layers + rank=256 (~4.4M params) vs 6 layers + rank=512 (~16.9M)
     This directly addresses the 12,000:1 param:sample ratio identified as root cause of overfitting
  3. SGDR warm restarts for multi-cycle exploration (matches node4-2 strategy, F1=0.5069)
  4. Dropout=0.4 (vs 0.2 in parent) — sibling's feedback recommended 0.4-0.5
  5. Label smoothing=0.05 to improve calibration overfitting
  6. Extended warmup_steps=400 — sibling had early instability with 200-step warmup
  7. Quality-filtered SWA for test-time ensemble from the good plateau window

Differentiation from sibling (node4-1-1-1):
  - Sibling uses AdamW with corrected LR schedule (total_steps=6600)
  - This node uses MuonWithAuxAdam + SGDR (completely different optimizer strategy)
  - Sibling has 6-layer rank-512 head; this has 4-layer rank-256 head (addresses overfitting root cause)
  - Sibling uses monotonic cosine decay; this uses warm restarts for exploration

Inspired by:
  - node4-1-2 (sibling of parent's parent): MuonWithAuxAdam + SGDR achieved F1=0.5060 from same lineage
  - node2-1-1-1-2-1-1-1-1 (tree best F1=0.5124): SWA over checkpoint plateau
  - node4-2-1-1-1 (F1=0.5014): reduced 4-layer head with 256 rank, similar approach
  - node4-1-1-1 (sibling) feedback: reduce head capacity, increase dropout to 0.4-0.5
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
import copy
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
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """
    Focal cross-entropy loss with label smoothing to improve calibration.
    logits:  [N, C]
    targets: [N]  (class indices 0,1,2)
    Returns: scalar mean loss
    """
    # Apply label smoothing via cross_entropy
    ce   = F.cross_entropy(
        logits, targets, weight=class_weights,
        label_smoothing=label_smoothing, reduction="none"
    )
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
        print(f"[Node4-1-1-2] {n_unknown}/{total} samples not found in STRING_GNN graph "
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
    Residual MLP block: pre-norm, Linear(in→expand) → GELU → Dropout → Linear(expand→in) + skip.
    2D weight matrices (Linear.weight) are targeted by MuonWithAuxAdam.
    """
    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Two 2D weight matrices for Muon optimizer
        self.fc1 = nn.Linear(dim, dim * expand)
        self.fc2 = nn.Linear(dim * expand, dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


# ─── Reduced Bilinear Head ────────────────────────────────────────────────────

class GNNBilinearHead(nn.Module):
    """
    Reduced 4-layer deep residual MLP + bilinear interaction head.
    Input: [B, gnn_dim=256] PPI embedding of perturbed gene
    Output: [B, n_classes=3, n_genes_out=6640] logits

    Reduced capacity compared to parent (4 layers, rank=256 vs 6 layers, rank=512):
    - ~4.4M params vs 16.9M (factor ~3.8x reduction)
    - Reduces parameter:sample ratio from 12,000:1 to ~3,100:1
    - Still provides sufficient nonlinear capacity for the bilinear interaction

    Architecture:
      1. Input projection: LayerNorm(gnn_dim) → Linear(gnn_dim → hidden_dim) → GELU
      2. N_layers ResidualBlocks (hidden_dim=256, expand=4) — 2D matrices use Muon
      3. Output LayerNorm
      4. Bilinear projection: [B, hidden_dim] → [B, n_classes, rank]
      5. Output gene embeddings: [n_genes_out, rank] — uses AdamW
      6. Logits: [B, n_classes, rank] @ [rank, n_genes_out] → [B, n_classes, n_genes_out]
    """

    def __init__(
        self,
        gnn_dim:     int   = GNN_DIM,
        hidden_dim:  int   = 256,
        n_layers:    int   = 4,
        expand:      int   = 4,
        rank:        int   = 256,
        dropout:     float = 0.4,
        n_classes:   int   = N_CLASSES,
        n_genes_out: int   = N_GENES_OUT,
    ):
        super().__init__()
        self.rank      = rank
        self.n_classes = n_classes

        # Input projection — bias and LN weight are scalars, Linear.weight is 2D (Muon candidate)
        self.input_norm = nn.LayerNorm(gnn_dim)
        self.input_proj = nn.Linear(gnn_dim, hidden_dim)
        self.input_act  = nn.GELU()

        # Reduced residual MLP stack: 4 layers × ResidualBlock(256, expand=4)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim → n_classes * rank
        # This is a 2D weight matrix — Muon candidate
        self.bilinear_proj = nn.Linear(hidden_dim, n_classes * rank, bias=False)

        # Output gene embeddings [n_genes_out, rank] — use AdamW (embedding)
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
        h = self.input_act(self.input_proj(self.input_norm(pert_emb)))

        # Reduced residual MLP: [B, hidden_dim]
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
    Partial STRING_GNN fine-tuning model with reduced bilinear head.
    Optimization strategy:
      - MuonWithAuxAdam for ResidualBlock 2D weight matrices (faster convergence)
      - AdamW for backbone, gene embeddings, biases, and scalar parameters
    """

    def __init__(
        self,
        edge_index:  torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        n_nodes:     int,
        head_hidden: int   = 256,
        n_layers:    int   = 4,
        expand:      int   = 4,
        rank:        int   = 256,
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

        # Freeze layers mps.0-5 (keep mps.6, mps.7, post_mp trainable)
        self._freeze_first_layers()

        # Fallback embedding for genes not in STRING_GNN
        self.fallback_emb = nn.Parameter(torch.randn(GNN_DIM) * 0.02)

        # Reduced bilinear prediction head
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
        print(f"[Node4-1-1-2] Total params: {n_total:,} | Trainable: {n_train:,}")

    def _freeze_first_layers(self):
        """Freeze mps.0-5 and embedding table. Keep mps.6, mps.7, post_mp trainable."""
        for param in self.gnn.emb.parameters():
            param.requires_grad = False
        for i in range(6):
            for param in self.gnn.mps[i].parameters():
                param.requires_grad = False

    def _precompute_embeddings(self) -> torch.Tensor:
        """Run full STRING_GNN forward pass with gradients through mps.6, mps.7, post_mp."""
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
        all_embs  = self._precompute_embeddings()  # [n_nodes, 256]
        safe_idxs = node_indices.clamp(min=0)
        pert_emb  = all_embs[safe_idxs]            # [B, 256]

        # Replace unknown genes with fallback embedding
        known_mask = (node_indices >= 0)
        if not known_mask.all():
            fallback = self.fallback_emb.float().unsqueeze(0).expand_as(pert_emb)
            pert_emb = torch.where(
                known_mask.unsqueeze(-1).expand_as(pert_emb),
                pert_emb, fallback,
            )

        pert_emb = pert_emb.float()
        return self.head(pert_emb)

    def backbone_parameters(self):
        """Parameters of trainable backbone layers (mps.6, mps.7, post_mp)."""
        params = []
        for i in [6, 7]:
            params.extend(self.gnn.mps[i].parameters())
        params.extend(self.gnn.post_mp.parameters())
        return params

    def head_muon_params(self):
        """2D weight matrices in ResidualBlocks — targeted by Muon optimizer."""
        muon_params = []
        for block in self.head.residual_blocks:
            # fc1.weight and fc2.weight are 2D matrices (suitable for Muon)
            muon_params.append(block.fc1.weight)
            muon_params.append(block.fc2.weight)
        # Also include bilinear_proj weight (2D matrix in hidden layers)
        muon_params.append(self.head.bilinear_proj.weight)
        return muon_params

    def head_adamw_params(self):
        """Non-matrix parameters in head: biases, norms, embeddings + fallback_emb."""
        muon_set = set(id(p) for p in self.head_muon_params())
        adamw_params = []
        for p in self.head.parameters():
            if id(p) not in muon_set:
                adamw_params.append(p)
        adamw_params.append(self.fallback_emb)
        return adamw_params


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
        backbone_lr:         float = 1e-5,
        head_muon_lr:        float = 0.005,
        head_adamw_lr:       float = 5e-4,
        weight_decay:        float = 3e-3,
        focal_gamma:         float = 2.0,
        label_smoothing:     float = 0.05,
        dropout:             float = 0.4,
        head_hidden:         int   = 256,
        rank:                int   = 256,
        n_layers:            int   = 4,
        sgdr_t0:             int   = 400,   # steps per warm restart cycle
        sgdr_t_mult:         int   = 1,
        sgdr_eta_min_factor: float = 0.05,  # minimum LR as fraction of base
        max_epochs:          int   = 200,
        # SWA settings
        swa_start_epoch:     int   = 20,    # start collecting checkpoints for SWA
        swa_freq_epochs:     int   = 10,    # frequency to collect SWA snapshots
        swa_threshold:       float = 0.47,  # min val_f1 to include in SWA pool
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str]  = []
        self._test_symbols:  List[str]  = []
        self._test_labels:   List[torch.Tensor] = []

        # SWA checkpoint pool: list of (val_f1, state_dict_copy)
        self._swa_pool: List[Tuple[float, dict]] = []
        self._current_epoch_val_f1: float = 0.0

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
        self._current_epoch_val_f1 = f1

        # Collect SWA checkpoint if this epoch qualifies
        hp = self.hparams
        cur_epoch = self.current_epoch
        if (cur_epoch >= hp.swa_start_epoch
                and cur_epoch % hp.swa_freq_epochs == 0
                and f1 >= hp.swa_threshold
                and self.trainer.is_global_zero):
            # Save a deep copy of trainable+buffer model state
            trainable_names = {n for n, p in self.model.named_parameters() if p.requires_grad}
            buffer_names    = {n for n, _ in self.model.named_buffers()}
            eligible_keys   = trainable_names | buffer_names
            sd_copy = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
                if k in eligible_keys
            }
            self._swa_pool.append((f1, sd_copy))
            self.print(
                f"[SWA] Added epoch {cur_epoch} checkpoint (val_f1={f1:.4f}) "
                f"to pool (size={len(self._swa_pool)})"
            )

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
            self.print(f"[Node4-1-1-2] Deduplicating: {len(all_pert)} -> {len(keep_indices)}")
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
            self.print(f"[Node4-1-1-2] Saved test predictions -> {pred_path}")
            if all_labels.any():
                f1 = compute_per_gene_f1(all_probs.numpy(), all_labels.numpy())
                self.print(f"[Node4-1-1-2] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Gather parameter groups for MuonWithAuxAdam:
        #   - head_muon_params: 2D weight matrices in ResidualBlocks (use_muon=True)
        #   - head_adamw_params: biases, norms, embeddings in head (use_muon=False)
        #   - backbone_params: trainable GNN layers mps.6/7+post_mp (use_muon=False)
        muon_params    = self.model.head_muon_params()
        adamw_head     = self.model.head_adamw_params()
        backbone_params = list(self.model.backbone_parameters())

        # Verify no overlap between parameter groups
        muon_ids    = set(id(p) for p in muon_params)
        adamw_ids   = set(id(p) for p in adamw_head)
        backbone_ids = set(id(p) for p in backbone_params)
        # Check for duplicates
        assert len(muon_ids & adamw_ids) == 0, "Overlap between Muon and AdamW head params!"
        assert len(muon_ids & backbone_ids) == 0, "Overlap between Muon and backbone params!"

        param_groups = [
            # Muon group: ResidualBlock 2D weight matrices
            {
                "params": muon_params,
                "use_muon": True,
                "lr": hp.head_muon_lr,
                "weight_decay": hp.weight_decay,
                "momentum": 0.95,
            },
            # AdamW group: head biases/norms/embeddings
            {
                "params": adamw_head,
                "use_muon": False,
                "lr": hp.head_adamw_lr,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": hp.weight_decay,
            },
            # AdamW group: backbone (mps.6/7 + post_mp, trainable at low LR)
            {
                "params": backbone_params,
                "use_muon": False,
                "lr": hp.backbone_lr,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 1e-4,  # lighter decay for pretrained backbone
            },
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # SGDR: CosineAnnealingWarmRestarts on step level
        # T_0 = sgdr_t0 optimizer steps per cycle (~16 epochs with DDP 2 GPUs x step_factor)
        # T_mult = 1 (constant cycle length for stable multi-cycle exploration)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.sgdr_t0,
            T_mult=hp.sgdr_t_mult,
            eta_min=hp.head_muon_lr * hp.sgdr_eta_min_factor,
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


# ─── SWA Averaging ────────────────────────────────────────────────────────────

def save_swa_checkpoint(
    lit_model: PartialStringGNNLitModule,
    out_dir: Path,
) -> Optional[str]:
    """
    Compute SWA average from checkpoint pool and save as a Lightning checkpoint file.
    The saved file uses the same format as normal ModelCheckpoint saves, so it can
    be loaded via ckpt_path parameter without triggering setup() re-initialization issues.

    Returns the path to the saved SWA checkpoint, or None if pool is empty.
    Only called on rank 0.
    """
    pool = lit_model._swa_pool
    if not pool:
        print("[SWA] No checkpoints collected — skipping SWA.")
        return None

    print(f"[SWA] Averaging {len(pool)} checkpoints "
          f"(val_f1 range: {min(f for f, _ in pool):.4f}–{max(f for f, _ in pool):.4f})")

    # Compute uniform average of collected model state dicts
    avg_model_sd = {}
    for key in pool[0][1].keys():
        tensors = [sd[key].float() for _, sd in pool]
        avg_model_sd[key] = torch.stack(tensors, dim=0).mean(dim=0)

    # Build a minimal Lightning checkpoint dict
    # The state_dict in lit_model.state_dict() saves model.* keys
    # We need to wrap avg_model_sd with "model." prefix for the lit module state dict
    lit_sd = {"model." + k: v for k, v in avg_model_sd.items()}
    # Also add buffers that lit module itself has (e.g., focal_class_weights)
    for n, buf in lit_model.named_buffers():
        lit_key = n
        if lit_key not in lit_sd:
            lit_sd[lit_key] = buf.detach().cpu().clone()

    ckpt = {
        "state_dict": lit_sd,
        "pytorch-lightning_version": pl.__version__,
        "epoch": getattr(lit_model, "current_epoch", 0),
        "global_step": getattr(lit_model, "global_step", 0),
        "hyper_parameters": dict(lit_model.hparams),
    }

    swa_path = str(out_dir / "checkpoints" / "swa_checkpoint.ckpt")
    torch.save(ckpt, swa_path)
    print(f"[SWA] Saved SWA checkpoint to {swa_path}")
    return swa_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 4-1-1-2 – Partial STRING_GNN + MuonWithAuxAdam + Reduced Head + SGDR + SWA"
    )
    p.add_argument("--data-dir",             type=str,   default="data")
    p.add_argument("--backbone-lr",          type=float, default=1e-5)
    p.add_argument("--head-muon-lr",         type=float, default=0.005)
    p.add_argument("--head-adamw-lr",        type=float, default=5e-4)
    p.add_argument("--weight-decay",         type=float, default=3e-3)
    p.add_argument("--focal-gamma",          type=float, default=2.0)
    p.add_argument("--label-smoothing",      type=float, default=0.05)
    p.add_argument("--dropout",              type=float, default=0.4)
    p.add_argument("--head-hidden",          type=int,   default=256)
    p.add_argument("--rank",                 type=int,   default=256)
    p.add_argument("--n-layers",             type=int,   default=4)
    p.add_argument("--sgdr-t0",              type=int,   default=400)
    p.add_argument("--sgdr-t-mult",          type=int,   default=1)
    p.add_argument("--sgdr-eta-min-factor",  type=float, default=0.05)
    p.add_argument("--micro-batch-size",     type=int,   default=8)
    p.add_argument("--global-batch-size",    type=int,   default=32)
    p.add_argument("--max-epochs",           type=int,   default=200)
    p.add_argument("--patience",             type=int,   default=50)
    p.add_argument("--swa-start-epoch",      type=int,   default=20)
    p.add_argument("--swa-freq-epochs",      type=int,   default=10)
    p.add_argument("--swa-threshold",        type=float, default=0.47)
    p.add_argument("--num-workers",          type=int,   default=4)
    p.add_argument("--val-check-interval",   type=float, default=1.0)
    p.add_argument("--debug-max-step",       type=int,   default=None)
    p.add_argument("--fast-dev-run",         action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm  = PerturbationDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = PartialStringGNNLitModule(
        backbone_lr         = args.backbone_lr,
        head_muon_lr        = args.head_muon_lr,
        head_adamw_lr       = args.head_adamw_lr,
        weight_decay        = args.weight_decay,
        focal_gamma         = args.focal_gamma,
        label_smoothing     = args.label_smoothing,
        dropout             = args.dropout,
        head_hidden         = args.head_hidden,
        rank                = args.rank,
        n_layers            = args.n_layers,
        sgdr_t0             = args.sgdr_t0,
        sgdr_t_mult         = args.sgdr_t_mult,
        sgdr_eta_min_factor = args.sgdr_eta_min_factor,
        max_epochs          = args.max_epochs,
        swa_start_epoch     = args.swa_start_epoch,
        swa_freq_epochs     = args.swa_freq_epochs,
        swa_threshold       = args.swa_threshold,
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

    # Determine test checkpoint: SWA if available, else best ModelCheckpoint
    is_debug = args.debug_max_step is not None or args.fast_dev_run
    test_ckpt_path: Optional[str] = None  # default: None (debug uses current weights)

    if not is_debug:
        # Rank 0 computes and saves SWA checkpoint; all ranks then load it
        swa_ckpt_path: Optional[str] = None
        if trainer.is_global_zero:
            swa_ckpt_path = save_swa_checkpoint(lit, out_dir)

        # Broadcast SWA checkpoint path to all ranks
        if n_gpus > 1 and dist.is_available() and dist.is_initialized():
            path_list = [swa_ckpt_path]
            dist.broadcast_object_list(path_list, src=0)
            swa_ckpt_path = path_list[0]

        if swa_ckpt_path is not None:
            test_ckpt_path = swa_ckpt_path
            print(f"[Node4-1-1-2] Using SWA checkpoint for test: {swa_ckpt_path}")
        else:
            test_ckpt_path = "best"
            print(f"[Node4-1-1-2] No SWA pool collected — using best checkpoint.")

    trainer.test(lit, datamodule=dm, ckpt_path=test_ckpt_path)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 4-1-1-2 – STRING_GNN + MuonWithAuxAdam + Reduced Head (4L/R256) + SGDR + SWA\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
