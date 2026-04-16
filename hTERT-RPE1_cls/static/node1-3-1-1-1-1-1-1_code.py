"""
Node 1-3-1-1-1-1-1-1 (child of node1-3-1-1-1-1-1, F1=0.4976):
  STRING_GNN Partial Fine-tuning + Rank-512 Bilinear Head
  + Muon Optimizer + Epoch-based SGDR (T_0=20, T_mult=2) + Label Smoothing
  + Quality-filtered SWA (top-25, temp=3.0, threshold=0.490)

Architecture (same as parent):
  - STRING_GNN: PARTIAL fine-tuning of last 2 GCN layers (mps.6, mps.7) + post_mp
    (~198K backbone trainable params) - frozen layers mps.0-5 precomputed as buffer
  - Rank=512 bilinear MLP head (6 ResidualBlocks, hidden_dim=512, expand=4)
  - Class-weighted focal loss with label smoothing (gamma=2.0, weights=[2.0, 0.5, 4.0])
  - Three LR groups: backbone AdamW lr=1e-5, ResBlock Muon lr=0.005, head AdamW lr=5e-4

Key changes from parent (node1-3-1-1-1-1-1, F1=0.4976):
  1. REDUCE backbone_lr from 5e-5 to 1e-5:
     - The best nodes in the MCTS tree (F1=0.5180+) use backbone_lr=1e-5
     - Slower backbone updates prevent rapid overfitting to the 1,416-sample dataset
     - Allows later SGDR cycles to continue improving rather than degrading
     - Parent's 5e-5 caused peak at epoch 21 with descending staircase thereafter
  2. EPOCH-BASED SGDR (T_0=20 epochs, T_mult=2) instead of step-based (T_0=600 steps):
     - node2-1-1-1-2-1-1-1-1-1-1-1-1 (F1=0.5182) used T_0=20, T_mult=2
     - Step-based T_0=600 (~27 epochs) was too slow to restart before overfitting
     - Epoch-based T_0=20 creates restarts at epochs 20, 60, 140, 300...
     - The ascending staircase requires short cycles relative to convergence speed
  3. INCREASE DROPOUT from 0.2 to 0.45:
     - All top-performing nodes (F1>=0.51) use dropout=0.45-0.50
     - Parent's dropout=0.2 was insufficient to prevent the 2.51x train/val loss ratio
     - Higher dropout forces the model to generalize better across cycles
  4. IMPLEMENT QUALITY-FILTERED SWA:
     - Post-hoc SWA over top-25 checkpoints by val_f1 (threshold=0.490, temp=3.0)
     - Every 3 epochs from epoch 15 (dense checkpointing in warm-up phase)
     - Proven +0.003-0.007 F1 gain in nodes achieving 0.510-0.518
     - Parent had NO SWA; adding it is the highest-impact single change
  5. INCREASE PATIENCE to 200 (from 80):
     - SGDR cycles: 20, 60, 140, 300 epochs - need patience=200 for 3rd cycle
     - node2-1-1-1-2* (0.5182) used patience=200, allowing full staircase development
  6. RETAIN label_smoothing=0.05 and Muon optimizer:
     - Label smoothing reduced overconfidence (2.51x vs parent's 3.8x)
     - Muon optimizer accelerated convergence to F1=0.4976 at epoch 21
     - Both are beneficial and will be kept

Architecture (unchanged from parent):
  - Frozen mps.0-5 precomputed as embedding buffer
  - Live GNN forward for mps.6, mps.7, post_mp using precomputed intermediate activations
  - MLP head: 6 ResidualBlocks(hidden=512, expand=4) + bilinear rank=512
  - Output: [B, 3, 6640] logits

Expected performance: F1 = 0.510 – 0.520
  - Floor: parent node1-3-1-1-1-1-1 got 0.4976 (confirmed with same arch)
  - backbone_lr=1e-5: allows later staircase improvement
  - SGDR T_0=20 (epoch): ascending staircase like node2 lineage (0.5180+)
  - SWA gain: +0.003-0.007 based on best nodes
  - Combined: targeting 0.512-0.518
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import json
import math
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
    Callback,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
from muon import MuonWithAuxAdam

# ─── Constants ────────────────────────────────────────────────────────────────

N_GENES_OUT = 6640
N_CLASSES = 3
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Exact per-gene macro F1 matching calc_metric.py logic.

    Args:
        pred_logits_np: [N, 3, G] float (logits or probabilities)
        labels_np:      [N, G]    int   (class indices 0/1/2)

    Returns:
        Mean per-gene F1 score (float).
    """
    pred_classes = pred_logits_np.argmax(axis=1)  # [N, G]
    n_genes = labels_np.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = labels_np[:, g]
        yh = pred_classes[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── STRING_GNN Partial Fine-tuning ──────────────────────────────────────────

def load_string_gnn_partial(
    model_dir: Path,
    device: torch.device,
) -> Tuple[np.ndarray, List[str], Dict[str, int], object, dict]:
    """Load STRING_GNN model:
    - Run frozen forward through mps.0-5 layers to get intermediate embeddings
    - Return those intermediate embeddings (buffer) + model with mps.6, mps.7, post_mp active

    Returns:
        frozen_mps05_emb: [N_nodes, 256] numpy array (output after mps.5)
        node_names: list of node name strings
        node_name_to_idx: {name -> idx} mapping
        partial_gnn: the model with mps.0-5 frozen, mps.6+mps.7+post_mp trainable
        graph: graph data dict
    """
    node_names = json.loads((model_dir / "node_names.json").read_text())
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}
    graph = torch.load(model_dir / "graph_data.pt", weights_only=False)

    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    model = model.to(device).eval()

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Then unfreeze: mps.6, mps.7, post_mp
    for name in ["mps.6", "mps.7", "post_mp"]:
        for n, p in model.named_parameters():
            if n.startswith(name):
                p.requires_grad = True

    edge_index = graph["edge_index"].to(device)
    edge_weight = graph.get("edge_weight")
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    # Precompute intermediate activations through mps.0-5
    with torch.no_grad():
        # Manual forward through layers 0-5
        x = model.emb.weight  # [N, 256]

        for i in range(6):  # mps.0 through mps.5
            layer = model.mps[i]
            # GNNLayer forward: conv -> norm -> act -> dropout -> residual
            h = layer.conv(x, edge_index, edge_weight)
            h = layer.norm(h)
            h = layer.act(h)
            h = layer.dropout(h)
            x = x + h  # residual

        frozen_mps05_emb = x.float().cpu().numpy()  # [N, 256]

    # Clean up
    del edge_index, edge_weight
    torch.cuda.empty_cache()

    # Cast trainable parameters to float32
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    return frozen_mps05_emb, node_names, node_name_to_idx, model, graph


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        node_name_to_idx: Dict[str, int],
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        # STRING_GNN node indices
        self.node_indices = torch.tensor(
            [node_name_to_idx.get(pid, -1) for pid in self.pert_ids],
            dtype=torch.long,
        )

        self.has_labels = has_labels
        if has_labels and "label" in df.columns:
            rows = []
            for lbl_str in df["label"]:
                rows.append([x + 1 for x in json.loads(lbl_str)])
            self.labels = torch.tensor(rows, dtype=torch.long)
        else:
            self.has_labels = False

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int):
        item = {
            "pert_id":  self.pert_ids[idx],
            "symbol":   self.symbols[idx],
            "node_idx": self.node_indices[idx],
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule using partial STRING_GNN fine-tuning."""

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if hasattr(self, "train_ds"):
            return

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Load all splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        # Load STRING_GNN node mapping
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.node_name_to_idx = {name: i for i, name in enumerate(node_names)}
        self.n_nodes = len(node_names)

        # Coverage report
        train_gnn = sum(p in self.node_name_to_idx for p in dfs["train"]["pert_id"])
        if local_rank == 0:
            print(
                f"[DataModule rank {local_rank}] Coverage — "
                f"STRING_GNN: {train_gnn}/{len(dfs['train'])} train genes"
            )

        # Datasets
        self.train_ds = PerturbationDataset(dfs["train"], self.node_name_to_idx, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   self.node_name_to_idx, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  self.node_name_to_idx, True)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )


# ─── Model Components ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout + skip."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.45):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PartialGNNBilinearHead(nn.Module):
    """Prediction head combining partial STRING_GNN fine-tuning with bilinear interaction.

    Key design:
    - Frozen intermediate activations after mps.5 stored as a buffer [N_nodes, 256]
    - Trainable partial GNN: mps.6, mps.7, post_mp on top of the frozen buffer
    - MLP head (hidden=512, 6 ResBlocks) with rank=512 bilinear interaction
    - OOV fallback: learned embedding for genes not in STRING graph
    """

    def __init__(
        self,
        frozen_mps05_emb: torch.Tensor,   # [N_nodes, 256] after mps.5
        partial_gnn_model,                  # STRING_GNN model with mps.6+mps.7+post_mp trainable
        graph_data: dict,                   # graph_data.pt dict
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        bilinear_rank: int = 512,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.45,
        n_residual_layers: int = 6,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.bilinear_rank = bilinear_rank
        self.gnn_dim = gnn_dim

        # Frozen intermediate activations buffer (not trainable)
        self.register_buffer("frozen_mps05_emb", frozen_mps05_emb.float())

        # Store edge_index and edge_weight as non-trainable buffers
        edge_index = graph_data["edge_index"]  # [2, E]
        self.register_buffer("edge_index", edge_index.long())
        edge_weight = graph_data.get("edge_weight")
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight.float())
        else:
            self.edge_weight = None

        # Trainable partial GNN layers (mps.6, mps.7, post_mp)
        self.partial_gnn = partial_gnn_model

        # OOV fallback embedding for genes not in STRING graph
        self.oov_gnn_emb = nn.Embedding(1, gnn_dim)

        # Input normalization
        self.input_norm = nn.LayerNorm(gnn_dim)

        # MLP head
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim -> n_classes * bilinear_rank
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * bilinear_rank)
        self.head_dropout = nn.Dropout(dropout)

        # Output gene embeddings [n_genes_out, bilinear_rank] — random init
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, bilinear_rank))
        nn.init.normal_(self.out_gene_emb, std=0.02)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.oov_gnn_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)

    def get_node_embeddings(self) -> torch.Tensor:
        """Run trainable mps.6, mps.7, post_mp on top of frozen mps.0-5 output.

        Returns:
            node_emb: [N_nodes, 256] float32
        """
        edge_weight = getattr(self, 'edge_weight', None)

        # x = output after mps.5 (frozen buffer)
        x = self.frozen_mps05_emb  # [N, 256]

        # Run mps.6 (trainable)
        layer6 = self.partial_gnn.mps[6]
        h = layer6.conv(x, self.edge_index, edge_weight)
        h = layer6.norm(h)
        h = layer6.act(h)
        h = layer6.dropout(h)
        x = x + h  # residual

        # Run mps.7 (trainable)
        layer7 = self.partial_gnn.mps[7]
        h = layer7.conv(x, self.edge_index, edge_weight)
        h = layer7.norm(h)
        h = layer7.act(h)
        h = layer7.dropout(h)
        x = x + h  # residual

        # Run post_mp (trainable)
        x = self.partial_gnn.post_mp(x)  # [N, 256]

        return x.float()

    def forward(self, node_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_idx: [B] long tensor, -1 for OOV genes

        Returns:
            logits: [B, 3, 6640]
        """
        B = node_idx.shape[0]
        device = node_idx.device
        zeros = torch.zeros(B, dtype=torch.long, device=device)

        # Get current node embeddings via partial GNN forward
        all_node_emb = self.get_node_embeddings()  # [N_nodes, 256]

        # Look up GNN embeddings with OOV handling
        in_gnn_mask = (node_idx >= 0)
        gnn_emb  = all_node_emb[node_idx.clamp(min=0)]           # [B, 256]
        oov_emb  = self.oov_gnn_emb(zeros)                        # [B, 256]
        in_f     = in_gnn_mask.float().unsqueeze(1)               # [B, 1]
        x        = gnn_emb * in_f + oov_emb * (1.0 - in_f)       # [B, 256]

        # MLP head
        x = self.input_norm(x)
        x = self.proj_in(x)                    # [B, hidden_dim]
        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)                   # [B, hidden_dim]

        # Bilinear interaction
        x = self.head_dropout(x)
        pert_proj = self.proj_bilinear(x)                                       # [B, n_classes * rank]
        pert_proj = pert_proj.view(B, self.n_classes, self.bilinear_rank)       # [B, 3, rank]

        # logits: [B, 3, rank] x [n_genes_out, rank]^T -> [B, 3, n_genes_out]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)      # [B, 3, 6640]
        return logits


# ─── Focal Loss with Class Weights and Label Smoothing ─────────────────────────

def focal_loss_weighted_smooth(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Class-weighted focal loss with label smoothing.

    Args:
        logits: [B, C, G] raw logits
        labels: [B, G] integer class labels in {0, 1, 2}
        gamma: focal loss exponent
        class_weights: [C] per-class weight tensor
        label_smoothing: label smoothing epsilon (0.0 = no smoothing)

    Returns:
        Scalar loss value.
    """
    B, C, G = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)   # [B*G, 3]
    labels_flat = labels.reshape(-1)                         # [B*G]

    if label_smoothing > 0.0:
        # Smooth cross-entropy: -sum_c q_c * log(p_c)
        # where q_c = (1 - eps) * one_hot(c) + eps/C
        # This is equivalent to: (1-eps)*CE + eps*(-mean log p)
        # Use torch's built-in cross_entropy with label_smoothing
        ce_loss = F.cross_entropy(
            logits_flat, labels_flat,
            reduction="none",
            label_smoothing=label_smoothing,
        )
    else:
        ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction="none")

    with torch.no_grad():
        probs = F.softmax(logits_flat, dim=1)
        pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt).pow(gamma)                                # [B*G]

        if class_weights is not None:
            cw = class_weights[labels_flat]                                 # [B*G]
            focal_weight = focal_weight * cw

    return (focal_weight * ce_loss).mean()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(
    local_preds: torch.Tensor,
    local_labels: torch.Tensor,
    device: torch.device,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather variable-length tensors from all DDP ranks with padding."""
    local_size = torch.tensor([local_preds.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = int(max(s.item() for s in all_sizes))

    pad = max_size - local_preds.shape[0]
    p = local_preds.to(device)
    l = local_labels.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], dim=0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], dim=0)

    g_preds  = [torch.zeros_like(p) for _ in range(world_size)]
    g_labels = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(g_preds,  p)
    dist.all_gather(g_labels, l)

    real_preds  = torch.cat([g_preds[i][:all_sizes[i].item()].cpu()  for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][:all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


# ─── Periodic Checkpoint Callback for SWA ─────────────────────────────────────

class PeriodicCheckpointCallback(Callback):
    """Save checkpoints every `every_n_epochs` from `start_epoch` for SWA pool."""

    def __init__(
        self,
        dirpath: str,
        every_n_epochs: int = 3,
        start_epoch: int = 15,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.every_n_epochs = every_n_epochs
        self.start_epoch = start_epoch
        self._last_val_f1: float = 0.0

    def on_validation_epoch_end(self, trainer, pl_module):
        """Store the val_f1 from this epoch for correct epoch labeling."""
        val_f1 = trainer.callback_metrics.get("val_f1", None)
        if val_f1 is not None:
            self._last_val_f1 = float(val_f1)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch < self.start_epoch:
            return
        if (epoch - self.start_epoch) % self.every_n_epochs != 0:
            return

        # Use val_f1 from callback_metrics (logged during validation_epoch_end)
        val_f1 = trainer.callback_metrics.get("val_f1", None)
        val_f1_val = float(val_f1) if val_f1 is not None else 0.0

        # Call trainer.save_checkpoint on ALL ranks — it internally calls strategy.barrier(),
        # which is a collective operation requiring all-rank participation.
        # DDPStrategy.save_checkpoint() handles rank-0-only file writing internally.
        self.dirpath.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.dirpath / f"periodic-epoch={epoch:04d}-val_f1={val_f1_val:.4f}.ckpt"
        trainer.save_checkpoint(str(ckpt_path))


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """LightningModule for gene-perturbation DEG prediction.

    Key design:
    - Partial STRING_GNN fine-tuning (mps.6+mps.7+post_mp) + rank-512 bilinear head
    - 3-group optimizer: backbone AdamW (1e-5), Muon for MLP 2D matrices (0.005),
      AdamW for other head params (5e-4)
    - Epoch-based cosine warm restarts (SGDR T_0=20 epochs, T_mult=2)
    - Label smoothing (epsilon=0.05)
    - Post-hoc quality-filtered SWA (top-25, temp=3.0, threshold=0.490)
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        bilinear_rank: int = 512,
        n_residual_layers: int = 6,
        dropout: float = 0.45,
        backbone_lr: float = 1e-5,
        head_lr: float = 5e-4,
        muon_lr: float = 0.005,
        weight_decay: float = 1e-3,
        focal_gamma: float = 2.0,
        focal_class_weights: Tuple[float, float, float] = (2.0, 0.5, 4.0),
        label_smoothing: float = 0.05,
        warmup_epochs: int = 5,
        sgdr_t0_epochs: int = 20,     # T_0 in epochs (epoch-based SGDR)
        sgdr_t_mult: int = 2,         # Cycle length multiplier
        sgdr_eta_min_ratio: float = 1e-3,  # Min LR as fraction of peak
        n_nodes: int = 18870,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_labels:   List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []

    def setup(self, stage: Optional[str] = None):
        if hasattr(self, "model"):
            return

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = self.device if self.device.type != "meta" else torch.device("cpu")

        if local_rank == 0:
            print(f"[Node1-3-1-1-1-1-1-1] Loading STRING_GNN with partial fine-tuning on device={device}")

        # Load STRING_GNN with partial fine-tuning setup
        frozen_mps05_np, _, _, partial_gnn, graph_data = load_string_gnn_partial(
            STRING_GNN_DIR, device
        )
        frozen_mps05_tensor = torch.from_numpy(frozen_mps05_np).float()

        if local_rank == 0:
            trainable_bb = sum(p.numel() for p in partial_gnn.parameters() if p.requires_grad)
            total_bb = sum(p.numel() for p in partial_gnn.parameters())
            print(
                f"[Node1-3-1-1-1-1-1-1] STRING_GNN backbone: {trainable_bb}/{total_bb} params trainable "
                f"(mps.6+mps.7+post_mp)"
            )

        # Build model
        self.model = PartialGNNBilinearHead(
            frozen_mps05_emb=frozen_mps05_tensor,
            partial_gnn_model=partial_gnn,
            graph_data=graph_data,
            gnn_dim=self.hparams.gnn_dim,
            hidden_dim=self.hparams.hidden_dim,
            bilinear_rank=self.hparams.bilinear_rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=self.hparams.dropout,
            n_residual_layers=self.hparams.n_residual_layers,
        )

        # Register class weights buffer for focal loss
        cw = torch.tensor(
            list(self.hparams.focal_class_weights), dtype=torch.float32
        )
        self.register_buffer("class_weights", cw)

        # Cast all trainable parameters to float32
        for _, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        if local_rank == 0:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in self.model.parameters())
            print(f"[Node1-3-1-1-1-1-1-1] Model: {trainable:,}/{total:,} trainable params")

    def forward(self, node_idx: torch.Tensor) -> torch.Tensor:
        return self.model(node_idx)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return focal_loss_weighted_smooth(
            logits, labels,
            gamma=self.hparams.focal_gamma,
            class_weights=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_idx"])
        if "label" in batch:
            loss = self._compute_loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())
        return logits

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        local_p = torch.cat(self._val_preds,  dim=0)
        local_l = torch.cat(self._val_labels, dim=0)

        if self.trainer.world_size > 1:
            all_p, all_l = _gather_tensors(local_p, local_l, self.device, self.trainer.world_size)
        else:
            all_p, all_l = local_p, local_l

        f1 = compute_per_gene_f1(all_p.numpy(), all_l.numpy())
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_idx"])
        probs = torch.softmax(logits, dim=1)  # [B, 3, 6640]
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs = torch.cat(self._test_preds, dim=0)
        dummy_labels = torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        if self._test_labels:
            dummy_labels = torch.cat(self._test_labels, dim=0)

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
            seen_ids: set = set()
            dedup_probs: list = []
            dedup_labels: list = []
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for i, (pert_id, symbol, probs) in enumerate(
                    zip(all_pert, all_syms, all_probs.numpy())
                ):
                    if pert_id not in seen_ids:
                        seen_ids.add(pert_id)
                        fh.write(f"{pert_id}\t{symbol}\t{json.dumps(probs.tolist())}\n")
                        dedup_probs.append(probs)
                        dedup_labels.append(all_labels[i].numpy())
            self.print(
                f"[Node1-3-1-1-1-1-1-1] Saved test predictions -> {pred_path} ({len(seen_ids)} unique samples)"
            )

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-3-1-1-1-1-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # ── 3-group optimizer: backbone AdamW + Muon (MLP 2D) + AdamW (other head) ──
        #
        # Group 1: Backbone (partial GNN: mps.6, mps.7, post_mp) - AdamW with low LR
        backbone_params = [
            p for p in self.model.partial_gnn.parameters() if p.requires_grad
        ]
        backbone_param_ids = set(id(p) for p in backbone_params)

        # Group 2: Muon for ResBlock 2D weight matrices (proven to improve performance)
        # ResBlock matrices are named: res_blocks.*.net.1.weight, res_blocks.*.net.4.weight
        # Also includes proj_in.weight, proj_bilinear.weight
        # Muon is NOT applied to: embeddings, 1D params (biases, LayerNorm), out_gene_emb
        muon_params = []
        adamw_head_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in backbone_param_ids:
                continue  # Handled by backbone group
            # Muon: 2D weight matrices in hidden layers (NOT embeddings, NOT output emb)
            # Suitable: proj_in.weight, res_blocks.*.net.1.weight, res_blocks.*.net.4.weight
            # proj_bilinear.weight (maps hidden -> 3*rank), so it's an output-ish projection
            # According to Muon guidelines: NOT for output layers/embeddings
            is_2d_matrix = (p.ndim >= 2)
            is_embedding = ("emb" in name or "oov" in name)
            is_output = ("out_gene_emb" in name or "proj_bilinear" in name)
            is_norm = ("norm" in name)
            if is_2d_matrix and not is_embedding and not is_output and not is_norm:
                muon_params.append(p)
            else:
                adamw_head_params.append(p)

        param_groups = [
            # Group 1: Backbone (AdamW, low LR — reduced from 5e-5 to 1e-5)
            dict(
                params=backbone_params,
                use_muon=False,
                lr=hp.backbone_lr,
                weight_decay=hp.weight_decay,
                betas=(0.9, 0.95),
            ),
            # Group 2: ResBlock 2D matrices (Muon, higher LR)
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            # Group 3: Other head params (AdamW: 1D params, embeddings, output emb)
            dict(
                params=adamw_head_params,
                use_muon=False,
                lr=hp.head_lr,
                weight_decay=hp.weight_decay,
                betas=(0.9, 0.95),
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # ── Epoch-based Cosine Warm Restarts (SGDR) ──────────────────────────
        # T_0=20 epochs, T_mult=2: cycles at epoch 20, 60 (20+40), 140 (60+80), ...
        # This is the proven configuration from node2-1-1-1-2* (F1=0.5180+)
        # epoch-based schedule avoids step-count dependency on batch size / GPU count
        #
        # Implementation: use LambdaLR with epoch-based cycle calculation
        # Scheduler interval: "epoch" (not "step")

        warmup = hp.warmup_epochs
        T_0 = hp.sgdr_t0_epochs
        T_mult = hp.sgdr_t_mult
        eta_min_ratio = hp.sgdr_eta_min_ratio

        def lr_lambda_epoch_sgdr(current_epoch: int) -> float:
            """Combined warmup + epoch-based SGDR schedule."""
            if current_epoch < warmup:
                # Linear warmup
                return float(current_epoch + 1) / max(1, warmup)

            # SGDR (cosine warm restarts) after warmup
            epoch = current_epoch - warmup

            # Compute which cycle we're in
            if T_mult == 1:
                # Simple case: all cycles have same length T_0
                cycle = epoch // T_0
                step_in_cycle = epoch % T_0
                cycle_len = T_0
            else:
                # Geometric case: cycle i has length T_0 * T_mult^i
                t_cur = epoch
                t_i = T_0
                cycle = 0
                while t_cur >= t_i:
                    t_cur -= t_i
                    t_i = int(t_i * T_mult)
                    cycle += 1
                step_in_cycle = t_cur
                cycle_len = t_i

            # Cosine decay within cycle
            progress = step_in_cycle / max(1, cycle_len)
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(eta_min_ratio, cosine_val)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_epoch_sgdr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Epoch-based, not step-based
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable params ──────────────────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_sd = {}
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        for k, v in full_sd.items():
            if k in trainable_keys or k in buffer_keys:
                trainable_sd[k] = v
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Quality-Filtered SWA ─────────────────────────────────────────────────────

def apply_quality_filtered_swa(
    lit_module: PerturbationLitModule,
    periodic_ckpt_dir: Path,
    datamodule: "PerturbationDataModule",
    out_dir: Path,
    top_k: int = 25,
    temperature: float = 3.0,
    swa_threshold: float = 0.490,
    micro_batch_size: int = 16,
    num_workers: int = 4,
) -> Optional[float]:
    """Post-hoc quality-filtered exponentially-weighted SWA.

    Strategy:
    1. Parse all periodic checkpoints and their val_f1 scores
    2. Filter checkpoints >= swa_threshold
    3. Select top_k by val_f1 (to avoid including too-early poor checkpoints)
    4. Compute softmax-weighted average: weight_i = softmax(val_f1_i * temperature)
    5. Run test inference with averaged weights
    6. Save SWA test predictions

    Returns:
        SWA test F1 (if labels available), else None
    """
    import glob

    ckpt_files = list(periodic_ckpt_dir.glob("periodic-epoch=*.ckpt"))
    if not ckpt_files:
        print("[SWA] No periodic checkpoints found. Skipping SWA.")
        return None

    # Parse checkpoint filenames to extract val_f1 scores
    # PL-generated name format: periodic-epoch=0015-val_f1=0.5012.ckpt
    pattern = re.compile(r"periodic-epoch=(\d+)-val_f1=([\d.]+)\.ckpt")
    parsed = []
    for f in ckpt_files:
        m = pattern.search(f.name)
        if m:
            epoch = int(m.group(1))
            val_f1 = float(m.group(2))
            parsed.append((val_f1, epoch, f))
        else:
            # Try alternative parsing
            m2 = re.search(r"epoch=(\d+).*val_f1=([\d.]+)", f.name)
            if m2:
                epoch = int(m2.group(1))
                val_f1 = float(m2.group(2))
                parsed.append((val_f1, epoch, f))
            else:
                print(f"[SWA] Could not parse: {f.name}")

    if not parsed:
        print("[SWA] No parseable checkpoints found. Skipping SWA.")
        return None

    # Filter by threshold
    filtered = [(f1, ep, f) for f1, ep, f in parsed if f1 >= swa_threshold]
    print(f"[SWA] Total periodic checkpoints: {len(parsed)}, above threshold {swa_threshold}: {len(filtered)}")

    if not filtered:
        print(f"[SWA] No checkpoints above threshold {swa_threshold}. Skipping SWA.")
        return None

    # Select top_k by val_f1
    filtered.sort(key=lambda x: x[0], reverse=True)
    selected = filtered[:top_k]
    print(f"[SWA] Selected top-{len(selected)} checkpoints:")
    for f1, ep, f in selected:
        print(f"  epoch={ep:4d}  val_f1={f1:.4f}  {f.name}")

    # Compute softmax weights
    f1_vals = np.array([f1 for f1, _, _ in selected])
    logits_w = f1_vals * temperature
    logits_w = logits_w - logits_w.max()  # numerical stability
    weights = np.exp(logits_w)
    weights = weights / weights.sum()
    print(f"[SWA] Weights: min={weights.min():.4f}, max={weights.max():.4f}, sum={weights.sum():.4f}")
    print(f"[SWA] Best checkpoint weight: {weights[0]:.4f}, worst: {weights[-1]:.4f}")

    # Load and average weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start with first checkpoint's state dict
    _, _, first_ckpt = selected[0]
    sd_avg = {}
    first_sd = torch.load(str(first_ckpt), map_location=device, weights_only=False)
    if "state_dict" in first_sd:
        first_sd = first_sd["state_dict"]

    for k, v in first_sd.items():
        if v.dtype.is_floating_point:
            sd_avg[k] = v.float() * weights[0]
        else:
            sd_avg[k] = v.clone()

    # Add remaining checkpoints
    for w, (_, _, ckpt_f) in zip(weights[1:], selected[1:]):
        sd = torch.load(str(ckpt_f), map_location=device, weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        for k, v in sd.items():
            if k in sd_avg and v.dtype.is_floating_point:
                sd_avg[k] = sd_avg[k] + v.float() * w

    print("[SWA] Averaged state dict computed.")

    # Load averaged weights into module
    lit_module.load_state_dict(sd_avg)
    lit_module = lit_module.to(device)
    lit_module.eval()

    # Create a minimal trainer for test inference (single GPU)
    from lightning.pytorch.strategies import SingleDeviceStrategy

    swa_out_dir = out_dir / "swa"
    swa_out_dir.mkdir(parents=True, exist_ok=True)
    swa_csv = CSVLogger(save_dir=str(swa_out_dir / "logs"), name="swa_csv")

    swa_trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=SingleDeviceStrategy(device=device),
        precision="bf16-mixed",
        logger=[swa_csv],
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    # Make datamodule use the same settings
    datamodule.micro_batch_size = micro_batch_size
    datamodule.num_workers = num_workers
    if not hasattr(datamodule, "test_ds"):
        datamodule.setup("test")

    # Run test with SWA model
    swa_trainer.test(lit_module, datamodule=datamodule)

    # Check if SWA predictions were saved; rename them to SWA file
    pred_path = out_dir / "test_predictions.tsv"
    if pred_path.exists():
        swa_pred_path = out_dir / "test_predictions_swa.tsv"
        import shutil
        shutil.copy(str(pred_path), str(swa_pred_path))
        print(f"[SWA] SWA test predictions saved to {swa_pred_path}")

    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 1-3-1-1-1-1-1-1 — Partial STRING_GNN + Muon + Epoch SGDR + Label Smoothing + SWA"
    )
    p.add_argument("--data-dir",            type=str,   default="data")
    p.add_argument("--gnn-dim",             type=int,   default=256)
    p.add_argument("--hidden-dim",          type=int,   default=512)
    p.add_argument("--bilinear-rank",       type=int,   default=512)
    p.add_argument("--n-residual-layers",   type=int,   default=6)
    p.add_argument("--dropout",             type=float, default=0.45)
    p.add_argument("--backbone-lr",         type=float, default=1e-5)
    p.add_argument("--head-lr",             type=float, default=5e-4)
    p.add_argument("--muon-lr",             type=float, default=0.005)
    p.add_argument("--weight-decay",        type=float, default=1e-3)
    p.add_argument("--focal-gamma",         type=float, default=2.0)
    p.add_argument("--focal-class-weights", type=float, nargs=3,
                   default=[2.0, 0.5, 4.0],
                   help="Class weights for focal loss: [down, neutral, up]")
    p.add_argument("--label-smoothing",     type=float, default=0.05)
    p.add_argument("--warmup-epochs",       type=int,   default=5,
                   help="Warmup epochs before first SGDR cycle")
    p.add_argument("--sgdr-t0-epochs",      type=int,   default=20,
                   help="SGDR T_0 in epochs. T_mult=2 gives cycles: 20,40,80,160 epochs")
    p.add_argument("--sgdr-t-mult",         type=int,   default=2,
                   help="SGDR cycle length multiplier (T_mult=2 => doubling cycle lengths)")
    p.add_argument("--sgdr-eta-min-ratio",  type=float, default=1e-3,
                   help="Min LR as fraction of peak LR")
    p.add_argument("--micro-batch-size",    type=int,   default=16)
    p.add_argument("--global-batch-size",   type=int,   default=64)
    p.add_argument("--max-epochs",          type=int,   default=400)
    p.add_argument("--patience",            type=int,   default=200)
    p.add_argument("--num-workers",         type=int,   default=4)
    p.add_argument("--val-check-interval",  type=float, default=1.0)
    # SWA arguments
    p.add_argument("--swa-top-k",           type=int,   default=25,
                   help="Number of top checkpoints to use in SWA")
    p.add_argument("--swa-temperature",     type=float, default=3.0,
                   help="Softmax temperature for checkpoint weighting in SWA")
    p.add_argument("--swa-threshold",       type=float, default=0.490,
                   help="Minimum val_f1 threshold for SWA checkpoint inclusion")
    p.add_argument("--swa-every-n-epochs",  type=int,   default=3,
                   help="Frequency of periodic checkpointing for SWA pool")
    p.add_argument("--swa-start-epoch",     type=int,   default=15,
                   help="Epoch to start periodic checkpointing for SWA pool")
    p.add_argument("--no-swa",              action="store_true", default=False,
                   help="Disable post-hoc SWA (use best single checkpoint only)")
    p.add_argument("--debug-max-step",      type=int,   default=None)
    p.add_argument("--fast-dev-run",        action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataModule setup
    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    # LightningModule
    lit = PerturbationLitModule(
        gnn_dim=args.gnn_dim,
        hidden_dim=args.hidden_dim,
        bilinear_rank=args.bilinear_rank,
        n_residual_layers=args.n_residual_layers,
        dropout=args.dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        focal_class_weights=tuple(args.focal_class_weights),
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        sgdr_t0_epochs=args.sgdr_t0_epochs,
        sgdr_t_mult=args.sgdr_t_mult,
        sgdr_eta_min_ratio=args.sgdr_eta_min_ratio,
        n_nodes=dm.n_nodes,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=1, save_last=False,  # save_last=True triggers broadcast_object_list bug in PyTorch 2.7.0a0
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="epoch")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    # Periodic checkpoint callback for SWA pool
    periodic_ckpt_dir = out_dir / "periodic_checkpoints"
    periodic_ckpt_cb = PeriodicCheckpointCallback(
        dirpath=str(periodic_ckpt_dir),
        every_n_epochs=args.swa_every_n_epochs,
        start_epoch=args.swa_start_epoch,
    )

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    # Debug settings
    max_steps:           int         = -1
    limit_train_batches: float | int = 1.0
    limit_val_batches:   float | int = 1.0
    limit_test_batches:  float | int = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps           = args.debug_max_step
        limit_train_batches = args.debug_max_step
        limit_val_batches   = 2
        limit_test_batches  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=args.global_batch_size // (args.micro_batch_size * n_gpus),
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb, periodic_ckpt_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    # Train
    trainer.fit(lit, datamodule=dm)

    # Test with best single checkpoint
    if args.fast_dev_run or args.debug_max_step is not None:
        trainer.test(lit, datamodule=dm)
    else:
        trainer.test(lit, datamodule=dm, ckpt_path="best")

    # Post-hoc quality-filtered SWA (only in full training mode)
    if (
        not args.fast_dev_run
        and args.debug_max_step is None
        and not args.no_swa
        and trainer.is_global_zero
    ):
        print("\n[SWA] Starting quality-filtered SWA post-processing...")
        try:
            apply_quality_filtered_swa(
                lit_module=lit,
                periodic_ckpt_dir=periodic_ckpt_dir,
                datamodule=dm,
                out_dir=out_dir,
                top_k=args.swa_top_k,
                temperature=args.swa_temperature,
                swa_threshold=args.swa_threshold,
                micro_batch_size=args.micro_batch_size,
                num_workers=args.num_workers,
            )
            # SWA saved as test_predictions_swa.tsv; rename to use for evaluation
            swa_pred = out_dir / "test_predictions_swa.tsv"
            if swa_pred.exists():
                import shutil
                # Keep original best-checkpoint predictions as backup
                best_pred = out_dir / "test_predictions_best_single.tsv"
                if (out_dir / "test_predictions.tsv").exists():
                    shutil.copy(str(out_dir / "test_predictions.tsv"), str(best_pred))
                # Replace main prediction file with SWA
                shutil.copy(str(swa_pred), str(out_dir / "test_predictions.tsv"))
                print(f"[SWA] Replaced test_predictions.tsv with SWA predictions")
        except Exception as e:
            print(f"[SWA] SWA failed: {e}. Using best single checkpoint predictions.")


if __name__ == "__main__":
    main()
