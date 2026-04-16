"""
Node 1-1-2-2-1 – Extended Partial STRING_GNN Fine-Tuning (4 Layers) + Higher Backbone LR
                  + Calibrated Extended Schedule + Gene-Weighted Loss

Architecture:
  - STRING_GNN backbone (18,870 nodes, 256-dim PPI topology embeddings)
    * Frozen: emb, mps.0-3 (early GCN layers precomputed as buffer for efficiency)
    * Trainable: mps.4, mps.5, mps.6, mps.7, post_mp (~366K params at backbone 2-tier LR)
      - mps.4, mps.5: lower backbone LR=2e-5 (deeper layers, more conservative)
      - mps.6, mps.7, post_mp: higher backbone LR=1e-4 (later layers, more aggressive)
  - Deep 6-layer residual bilinear MLP head (hidden_dim=512, expand=4, rank=512, dropout=0.20)
  - Bilinear interaction: pert_emb [B, 512] x out_gene_emb [6640, 512] -> logits [B, 3, 6640]
  - Class-weighted focal cross-entropy (gamma=2.0, weights=[2.0, 0.5, 4.0] for [down,neutral,up])
  - Gene-activity importance weighting: up-weights positions where non-zero labels are frequent
  - Three-group AdamW (backbone_deep lr=2e-5, backbone_late lr=1e-4, head lr=5e-4, wd=1e-3)
  - Cosine annealing LR with 100-step warmup, calibrated total_steps~1650 (150 epochs)

Key changes over Parent (Node 1-1-2-2, F1=0.4904):
  1. Extend partial fine-tuning from 2 layers (mps.6+7+post_mp) to 4 layers (mps.4+5+6+7+post_mp)
     - Parent bottleneck: frozen early layers (mps.0-5) represent 75% of backbone, limiting task adaptation
     - More backbone plasticity without catastrophic forgetting (mps.0-3 remain frozen)
  2. Increase backbone LR: mps.6+7+post_mp from 5e-5 -> 1e-4; mps.4+5 at 2e-5 (2-tier LR)
     - Parent bottleneck: 5e-5 backbone LR decays to ~2.5e-5 at epoch 70 (best epoch),
       providing negligible gradient signal for remaining 50 epochs
  3. Extend LR schedule to 150 epochs (total_steps~1650 vs parent's 1200)
     - Parent's schedule decays to near-zero at ~epoch 100, leaving 50 epochs with no learning
  4. Reduce head dropout from 0.25 to 0.20
     - Parent's early plateau (epoch 40-70) suggests insufficient gradient signal; slightly
       reduced regularization allows more productive late-phase learning
  5. Increase patience to 60 epochs (from 50)
     - With extended LR schedule and more backbone plasticity, secondary improvement phase
       should appear later in training

Key insights from collected_memory:
  - node2-1-3 (tree best F1=0.5047): partial STRING_GNN FT + rank=512 + class weights [2.0,0.5,4.0]
  - node1-1-2-2 (parent F1=0.4904): backbone LR=5e-5 insufficient; plateau at epoch 40-70
  - Parent feedback: "increase backbone_lr to 7e-5 or 1e-4" and "extend to mps.4+5+6+7+post_mp"
  - node1-3 (F1=0.4120): FULL fine-tuning catastrophic; this node keeps mps.0-3 frozen
  - node1-1-2-1-1-1-1 (F1=0.5035): partial backbone FT + rank=512 + Muon confirmed 0.50+ achievable
  - n_resblocks=6 optimal (8 hurts F1=0.4705 per node2-1-1-1-1-1)
  - dropout=0.20 better than 0.25 for secondary improvement phase (node2-1-3 parent used 0.20)
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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ─── Constants ────────────────────────────────────────────────────────────────

N_GENES_OUT = 6640
N_CLASSES = 3

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
STRING_GNN_DIM = 256   # STRING_GNN hidden dimension


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


# ─── Loss ─────────────────────────────────────────────────────────────────────

def focal_cross_entropy_weighted(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
    gene_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Class-weighted focal cross-entropy loss with optional per-gene importance weighting.

    Args:
        logits:       [B, C, G] float32 - per-class logits
        targets:      [B, G]    long    - class indices 0..C-1
        gamma:        focusing parameter (0 = standard CE)
        class_weights:[C] float32 - per-class weights (optional)
        gene_weights: [G] float32 - per-gene importance weights (optional)

    Returns:
        Scalar loss.
    """
    B, C, G = logits.shape
    # Reshape for cross_entropy: [B*G, C]
    logits_2d = logits.permute(0, 2, 1).reshape(B * G, C).contiguous()  # [B*G, C]
    targets_1d = targets.reshape(B * G)                                   # [B*G]

    # Standard CE (reduction='none') with optional class weights
    ce = F.cross_entropy(
        logits_2d,
        targets_1d,
        weight=class_weights,
        reduction="none",
    )  # [B*G]

    # Focal modulation: p_t from unweighted logits
    with torch.no_grad():
        log_probs = F.log_softmax(logits_2d, dim=1)                        # [B*G, C]
        probs = log_probs.exp()                                             # [B*G, C]
        pt = probs.gather(1, targets_1d.unsqueeze(1)).squeeze(1)           # [B*G]

    focal = (1.0 - pt) ** gamma * ce   # [B*G]

    # Apply per-gene importance weights if provided
    if gene_weights is not None:
        # gene_weights: [G] -> tile to [B, G] -> reshape to [B*G]
        w = gene_weights.unsqueeze(0).expand(B, -1).reshape(B * G)   # [B*G]
        focal = focal * w

    return focal.mean()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset. Labels are optionally present."""

    def __init__(
        self,
        df: pd.DataFrame,
        pert_id_to_gnn_idx: Dict[str, int],
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        # Map pert_id (Ensembl gene ID) to STRING_GNN node index; -1 for unknown
        self.gnn_indices: List[int] = [
            pert_id_to_gnn_idx.get(pid, -1) for pid in self.pert_ids
        ]
        self.has_labels = has_labels
        if has_labels and "label" in df.columns:
            rows = []
            for lbl_str in df["label"]:
                rows.append([x + 1 for x in json.loads(lbl_str)])
            self.labels = torch.tensor(rows, dtype=torch.long)  # [N, G]
        else:
            self.has_labels = False

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int):
        item = {
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "gnn_idx": self.gnn_indices[idx],
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch: List[dict]) -> dict:
    """Simple collate: stack gnn_idx, labels; keep lists for strings."""
    result = {
        "gnn_idx": torch.tensor([item["gnn_idx"] for item in batch], dtype=torch.long),
        "pert_id": [item["pert_id"] for item in batch],
        "symbol": [item["symbol"] for item in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([item["label"] for item in batch], dim=0)
    return result


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule for perturbation DEG prediction with STRING_GNN."""

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
        self.pert_id_to_gnn_idx: Dict[str, int] = {}

    def setup(self, stage: Optional[str] = None):
        # Load STRING_GNN node names to build pert_id -> node index mapping
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        # node_names[i] is the Ensembl gene ID (pert_id) for STRING_GNN node i
        self.pert_id_to_gnn_idx = {name: i for i, name in enumerate(node_names)}

        # Load splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        self.train_ds = PerturbationDataset(dfs["train"], self.pert_id_to_gnn_idx, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   self.pert_id_to_gnn_idx, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  self.pert_id_to_gnn_idx, True)

        # Log OOV statistics
        oov_train = sum(1 for idx in self.train_ds.gnn_indices if idx == -1)
        oov_val   = sum(1 for idx in self.val_ds.gnn_indices   if idx == -1)
        print(f"[DataModule] OOV genes - train: {oov_train}/{len(self.train_ds)}, "
              f"val: {oov_val}/{len(self.val_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
        )


# ─── Model Components ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block with pre-norm: LayerNorm -> Linear -> GELU -> Dropout -> Linear + skip."""

    def __init__(self, hidden_dim: int, expand: int = 4, dropout: float = 0.20):
        super().__init__()
        inner = hidden_dim * expand
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class PerturbationHead(nn.Module):
    """
    Deep residual bilinear MLP head for perturbation DEG prediction.

    Architecture:
        gnn_emb [B, 256] -> proj_in [B, 512] -> 6xResidualBlock -> [B, 512]
        -> proj_bilinear [B, 3 * rank=512] -> reshape [B, 3, 512]
        x out_gene_emb [6640, 512] -> logits [B, 3, 6640]
    """

    def __init__(
        self,
        gnn_dim: int = STRING_GNN_DIM,
        hidden_dim: int = 512,
        n_resblocks: int = 6,
        expand: int = 4,
        dropout: float = 0.20,
        rank: int = 512,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()

        # Input projection
        self.proj_in = nn.Sequential(
            nn.LayerNorm(gnn_dim),
            nn.Linear(gnn_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Deep residual MLP
        self.resblocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
              for _ in range(n_resblocks)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear interaction head (rank=512)
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank, bias=True)
        self.out_gene_emb = nn.Embedding(n_genes_out, rank)
        nn.init.normal_(self.out_gene_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)

        self.n_classes = n_classes
        self.rank = rank

    def forward(self, gnn_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gnn_emb: [B, 256] STRING_GNN node embeddings

        Returns:
            logits: [B, 3, 6640]
        """
        B = gnn_emb.shape[0]

        # Project and process through residual MLP
        h = self.proj_in(gnn_emb)               # [B, hidden_dim]
        h = self.resblocks(h)                    # [B, hidden_dim]
        h = self.norm_out(h)                     # [B, hidden_dim]

        # Bilinear interaction: [B, hidden_dim] -> [B, 3, rank] x [6640, rank].T
        proj = self.proj_bilinear(h)             # [B, 3 * rank]
        proj = proj.view(B, self.n_classes, self.rank)  # [B, 3, rank]
        out_emb = self.out_gene_emb.weight       # [6640, rank]
        logits = torch.einsum("bcr,gr->bcg", proj, out_emb)  # [B, 3, 6640]
        return logits


# ─── Trainable GNN Backbone Wrapper ───────────────────────────────────────────

class TrainableGNNBackbone(nn.Module):
    """Wrapper for the trainable STRING_GNN layers.

    This node extends partial fine-tuning from 2 layers (mps.6+7+post_mp in parent)
    to 4 layers (mps.4+5+6+7+post_mp).

    Accepts pre-computed frozen early-layer output [18870, 256] from mps.0-3,
    and runs mps.4, mps.5, mps.6, mps.7, post_mp to produce final embeddings.

    Two-tier backbone LR:
      - mps.4, mps.5: backbone_lr_deep (lower, e.g. 2e-5)
      - mps.6, mps.7, post_mp: backbone_lr_late (higher, e.g. 1e-4)
    """

    def __init__(self, gnn_layer_4, gnn_layer_5, gnn_layer_6, gnn_layer_7, post_mp):
        super().__init__()
        self.layer_4 = gnn_layer_4
        self.layer_5 = gnn_layer_5
        self.layer_6 = gnn_layer_6
        self.layer_7 = gnn_layer_7
        self.post_mp = post_mp

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [18870, 256] pre-computed frozen early layer output (from mps.0..3)
            edge_index: [2, E] graph edge indices
            edge_weight: [E] optional edge weights

        Returns:
            emb: [18870, 256] final node embeddings
        """
        # Layer 4 (deeper, lower LR)
        x_normed = self.layer_4.norm(x)
        x_conv = self.layer_4.conv(x_normed, edge_index, edge_weight)
        x_conv = self.layer_4.act(x_conv)
        x_conv = self.layer_4.dropout(x_conv)
        x = x + x_conv

        # Layer 5 (deeper, lower LR)
        x_normed = self.layer_5.norm(x)
        x_conv = self.layer_5.conv(x_normed, edge_index, edge_weight)
        x_conv = self.layer_5.act(x_conv)
        x_conv = self.layer_5.dropout(x_conv)
        x = x + x_conv

        # Layer 6 (later, higher LR)
        x_normed = self.layer_6.norm(x)
        x_conv = self.layer_6.conv(x_normed, edge_index, edge_weight)
        x_conv = self.layer_6.act(x_conv)
        x_conv = self.layer_6.dropout(x_conv)
        x = x + x_conv

        # Layer 7 (later, higher LR)
        x_normed = self.layer_7.norm(x)
        x_conv = self.layer_7.conv(x_normed, edge_index, edge_weight)
        x_conv = self.layer_7.act(x_conv)
        x_conv = self.layer_7.dropout(x_conv)
        x = x + x_conv

        # Output projection (later, higher LR)
        x = self.post_mp(x)
        return x


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
    lbl = local_labels.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], dim=0)
        lbl = torch.cat([lbl, lbl.new_zeros(pad, *lbl.shape[1:])], dim=0)

    g_preds  = [torch.zeros_like(p) for _ in range(world_size)]
    g_labels = [torch.zeros_like(lbl) for _ in range(world_size)]
    dist.all_gather(g_preds, p)
    dist.all_gather(g_labels, lbl)

    real_preds  = torch.cat([g_preds[i][:all_sizes[i].item()].cpu()  for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][:all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


def compute_gene_activity_weights(
    data_dir: Path,
    n_genes: int = N_GENES_OUT,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Compute per-gene activity weights from training data.

    Genes that are non-zero in more training samples are considered more informative
    and are given higher importance weights. This provides a soft signal about which
    gene positions carry more task-relevant information.

    The weight is: w_g = (1 - smoothing) * normalized_activity + smoothing
    where normalized_activity = nonzero_frac_g / max(nonzero_frac)

    This ensures the minimum weight is `smoothing` (0.1) for inactive genes, and
    the maximum weight is 1.0 for the most active genes.

    Args:
        data_dir: Path to data directory containing train.tsv
        n_genes: Number of output gene positions
        smoothing: Minimum weight for least-active genes

    Returns:
        gene_weights: [n_genes] float32 tensor, values in [smoothing, 1.0]
    """
    train_df = pd.read_csv(data_dir / "train.tsv", sep="\t")

    # Count non-zero positions per gene across all training samples
    nonzero_counts = np.zeros(n_genes, dtype=np.float32)
    for lbl_str in train_df["label"]:
        labels = np.array(json.loads(lbl_str), dtype=np.int8)
        nonzero_counts += (labels != 0).astype(np.float32)

    # Normalize to [0, 1] by the max activity
    max_activity = nonzero_counts.max()
    if max_activity > 0:
        normalized = nonzero_counts / max_activity
    else:
        normalized = np.ones(n_genes, dtype=np.float32)

    # Apply smoothing: weights in [smoothing, 1.0]
    weights = (1.0 - smoothing) * normalized + smoothing

    # Normalize weights to mean=1.0 so total loss scale is preserved
    weights = weights / weights.mean()

    print(f"[GeneWeights] Activity weight range: [{weights.min():.3f}, {weights.max():.3f}], "
          f"mean={weights.mean():.3f}")

    return torch.tensor(weights, dtype=torch.float32)


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """
    LightningModule for gene-perturbation DEG prediction.
    Node 1-1-2-2-1: Extended Partial STRING_GNN Fine-Tuning (4 Layers) + Higher Backbone LR
                    + Calibrated Extended Schedule + Gene-Activity-Weighted Loss
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        n_resblocks: int = 6,
        expand: int = 4,
        dropout: float = 0.20,
        rank: int = 512,
        lr: float = 5e-4,
        backbone_lr_late: float = 1e-4,   # mps.6, mps.7, post_mp (later layers)
        backbone_lr_deep: float = 2e-5,   # mps.4, mps.5 (deeper layers, lower LR)
        weight_decay: float = 1e-3,
        warmup_steps: int = 100,
        focal_gamma: float = 2.0,
        class_weight_down: float = 2.0,
        class_weight_neutral: float = 0.5,
        class_weight_up: float = 4.0,
        max_steps_total: int = 1650,   # calibrated to ~150 epochs
        grad_clip_norm: float = 1.0,
        use_gene_weights: bool = True,
        gene_weight_smoothing: float = 0.1,
        data_dir: str = "data",
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams

        # ── Load STRING_GNN model ──────────────────────────────────────────
        gnn_model = AutoModel.from_pretrained(
            str(STRING_GNN_DIR), trust_remote_code=True
        )
        gnn_model.eval()

        # Load graph data
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
        edge_index = graph["edge_index"]           # [2, E]
        edge_weight = graph.get("edge_weight", None)  # [E] or None

        # Register edge_index as a long buffer
        self.register_buffer("_edge_index_buf", edge_index)

        # Register edge_weight (handle None gracefully)
        self._has_edge_weight = edge_weight is not None
        if self._has_edge_weight:
            self.register_buffer("_edge_weight_buf", edge_weight.float())

        # ── Precompute frozen early layer outputs (mps.0..3 + emb) ────────
        # Extended partial fine-tuning: freeze mps.0-3, fine-tune mps.4-7+post_mp
        # Previously parent froze mps.0-5 (6 layers), this node freezes only mps.0-3 (4 layers)
        # This provides ~2× more backbone adaptable capacity while keeping early layers frozen.
        with torch.no_grad():
            # Start from the embedding table
            x = gnn_model.emb.weight.clone().float()  # [18870, 256]

            # Run message passing layers 0-3 (frozen, precomputed)
            ew = edge_weight.float() if edge_weight is not None else None
            for i in range(4):
                layer = gnn_model.mps[i]
                x_normed = layer.norm(x)
                x_conv = layer.conv(x_normed, edge_index, ew)
                x_conv = layer.act(x_conv)
                x_conv = layer.dropout(x_conv)
                x = x + x_conv

        # Store frozen early-layer output as a buffer [18870, 256]
        self.register_buffer("frozen_early_emb", x.detach())
        print(f"[Model] Precomputed frozen early layers (mps.0-3) -> buffer shape: {x.shape}")

        # OOV fallback: mean of frozen early embeddings (will be updated through trainable layers)
        self._oov_idx = gnn_model.emb.num_embeddings  # sentinel value beyond valid range

        # ── Build trainable GNN backbone (mps.4, mps.5, mps.6, mps.7, post_mp) ──
        # Two-tier LR structure: mps.4+5 get backbone_lr_deep, mps.6+7+post_mp get backbone_lr_late
        self.gnn_backbone = TrainableGNNBackbone(
            gnn_layer_4=gnn_model.mps[4],
            gnn_layer_5=gnn_model.mps[5],
            gnn_layer_6=gnn_model.mps[6],
            gnn_layer_7=gnn_model.mps[7],
            post_mp=gnn_model.post_mp,
        )

        # ── Build prediction head ──────────────────────────────────────────
        self.head = PerturbationHead(
            gnn_dim=STRING_GNN_DIM,
            hidden_dim=hp.hidden_dim,
            n_resblocks=hp.n_resblocks,
            expand=hp.expand,
            dropout=hp.dropout,
            rank=hp.rank,
        )

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Register class weights as a buffer
        cw = torch.tensor(
            [hp.class_weight_down, hp.class_weight_neutral, hp.class_weight_up],
            dtype=torch.float32
        )
        self.register_buffer("class_weights_buf", cw)

        # ── Compute and register per-gene activity weights ────────────────
        if hp.use_gene_weights:
            gw = compute_gene_activity_weights(
                Path(hp.data_dir),
                n_genes=N_GENES_OUT,
                smoothing=hp.gene_weight_smoothing,
            )
            self.register_buffer("gene_weights_buf", gw)
            self._use_gene_weights = True
        else:
            self._use_gene_weights = False

        # Print parameter summary
        # Identify deep vs late backbone params for LR assignment
        deep_params = list(self.gnn_backbone.layer_4.parameters()) + \
                      list(self.gnn_backbone.layer_5.parameters())
        late_params = list(self.gnn_backbone.layer_6.parameters()) + \
                      list(self.gnn_backbone.layer_7.parameters()) + \
                      list(self.gnn_backbone.post_mp.parameters())
        head_params = list(self.head.parameters())

        n_deep = sum(p.numel() for p in deep_params)
        n_late = sum(p.numel() for p in late_params)
        n_head = sum(p.numel() for p in head_params)
        print(f"[Model] Trainable backbone deep params (mps.4+5, lr={hp.backbone_lr_deep}): {n_deep:,}")
        print(f"[Model] Trainable backbone late params (mps.6+7+post_mp, lr={hp.backbone_lr_late}): {n_late:,}")
        print(f"[Model] Head params (lr={hp.lr}): {n_head:,}")
        print(f"[Model] Total trainable: {n_deep + n_late + n_head:,}")

    def _run_trainable_gnn(self) -> torch.Tensor:
        """Run trainable GNN layers (mps.4-7, post_mp) on frozen early embeddings.

        Returns:
            all_embs: [18870, 256] - final node embeddings for all STRING_GNN nodes
        """
        x = self.frozen_early_emb.float()   # [18870, 256] on device
        edge_index = self._edge_index_buf   # [2, E] on device
        edge_weight = self._edge_weight_buf if self._has_edge_weight else None

        return self.gnn_backbone(x, edge_index, edge_weight)   # [18870, 256]

    def _get_per_sample_emb(
        self,
        gnn_idx: torch.Tensor,
        all_embs: torch.Tensor,
    ) -> torch.Tensor:
        """Look up per-sample embeddings with OOV fallback.

        Args:
            gnn_idx:  [B] long tensor of STRING_GNN node indices; -1 for OOV
            all_embs: [18870, 256] full node embedding matrix

        Returns:
            emb: [B, 256] float32 tensor
        """
        valid_mask = gnn_idx >= 0   # [B]
        safe_idx = gnn_idx.clone()
        safe_idx[~valid_mask] = 0   # clamp invalid to 0 temporarily

        emb = all_embs[safe_idx]    # [B, 256]

        # Replace OOV entries with mean of all embeddings as fallback
        if (~valid_mask).any():
            oov_emb = all_embs.mean(dim=0, keepdim=True)  # [1, 256]
            emb = emb.clone()
            n_oov = (~valid_mask).sum()
            emb[~valid_mask] = oov_emb.expand(n_oov, -1).to(emb.dtype)

        return emb.float()

    def forward(self, gnn_idx: torch.Tensor) -> torch.Tensor:
        # Run trainable GNN backbone once to get all node embeddings
        all_embs = self._run_trainable_gnn()   # [18870, 256]

        # Look up per-sample embeddings
        emb = self._get_per_sample_emb(gnn_idx, all_embs)   # [B, 256]

        # Run prediction head
        return self.head(emb)   # [B, 3, 6640]

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return focal_cross_entropy_weighted(
            logits,
            labels,
            gamma=self.hparams.focal_gamma,
            class_weights=self.class_weights_buf,
            gene_weights=self.gene_weights_buf if self._use_gene_weights else None,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["gnn_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["gnn_idx"])
        if "label" in batch:
            loss = self._compute_loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())
        return logits

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        local_p = torch.cat(self._val_preds, dim=0)
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
        logits = self(batch["gnn_idx"])
        probs = torch.softmax(logits, dim=1)   # [B, 3, 6640]
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
            self._test_labels.clear()

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
                f"[Node1-1-2-2-1] Saved test predictions -> {pred_path} "
                f"({len(seen_ids)} unique samples)"
            )

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-1-2-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Three parameter groups:
        #   1. Backbone deep params (mps.4+5) at backbone_lr_deep (lower LR for deeper layers)
        #   2. Backbone late params (mps.6+7+post_mp) at backbone_lr_late (higher LR for later layers)
        #   3. Head params at head LR
        backbone_deep_params = list(self.gnn_backbone.layer_4.parameters()) + \
                               list(self.gnn_backbone.layer_5.parameters())
        backbone_late_params = list(self.gnn_backbone.layer_6.parameters()) + \
                               list(self.gnn_backbone.layer_7.parameters()) + \
                               list(self.gnn_backbone.post_mp.parameters())
        head_params = list(self.head.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_deep_params, "lr": hp.backbone_lr_deep,  "weight_decay": hp.weight_decay},
                {"params": backbone_late_params, "lr": hp.backbone_lr_late,  "weight_decay": hp.weight_decay},
                {"params": head_params,          "lr": hp.lr,                "weight_decay": hp.weight_decay},
            ],
        )

        def lr_lambda(current_step: int):
            if current_step < hp.warmup_steps:
                return float(current_step) / max(1, hp.warmup_steps)
            progress = float(current_step - hp.warmup_steps) / max(
                1, hp.max_steps_total - hp.warmup_steps
            )
            # Clamp progress to [0, 1] to prevent unintended cosine restart beyond schedule
            progress = min(progress, 1.0)
            # Return relative scale (will be multiplied by group's base LR)
            return max(1e-6, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable params ──────────────────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        trainable_sd = {
            k: v for k, v in full_sd.items()
            if k in trainable_keys or k in buffer_keys
        }
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 1-1-2-2-1 – Extended Partial STRING_GNN Fine-Tuning (4 Layers) + Higher Backbone LR"
    )
    p.add_argument("--data-dir",              type=str,   default="data")
    p.add_argument("--hidden-dim",            type=int,   default=512)
    p.add_argument("--n-resblocks",           type=int,   default=6)
    p.add_argument("--expand",                type=int,   default=4)
    p.add_argument("--dropout",               type=float, default=0.20)
    p.add_argument("--rank",                  type=int,   default=512)
    p.add_argument("--lr",                    type=float, default=5e-4)
    p.add_argument("--backbone-lr-late",      type=float, default=1e-4)
    p.add_argument("--backbone-lr-deep",      type=float, default=2e-5)
    p.add_argument("--weight-decay",          type=float, default=1e-3)
    p.add_argument("--warmup-steps",          type=int,   default=100)
    p.add_argument("--focal-gamma",           type=float, default=2.0)
    p.add_argument("--class-weight-down",     type=float, default=2.0)
    p.add_argument("--class-weight-neutral",  type=float, default=0.5)
    p.add_argument("--class-weight-up",       type=float, default=4.0)
    p.add_argument("--grad-clip-norm",        type=float, default=1.0)
    p.add_argument("--micro-batch-size",      type=int,   default=16)
    p.add_argument("--global-batch-size",     type=int,   default=128)
    p.add_argument("--max-epochs",            type=int,   default=200)
    p.add_argument("--patience",              type=int,   default=60)
    p.add_argument("--num-workers",           type=int,   default=4)
    p.add_argument("--val-check-interval",    type=float, default=1.0)
    p.add_argument("--use-gene-weights",      action="store_true", default=True)
    p.add_argument("--no-gene-weights",       dest="use_gene_weights", action="store_false")
    p.add_argument("--gene-weight-smoothing", type=float, default=0.1)
    p.add_argument("--debug-max-step",        type=int,   default=None)
    p.add_argument("--fast-dev-run",          action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataModule
    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # Compute total training steps for LR scheduler calibration
    _train_df_size = pd.read_csv(
        Path(args.data_dir) / "train.tsv", sep="\t", usecols=["pert_id"]
    ).shape[0]
    steps_per_epoch = _train_df_size // (args.micro_batch_size * n_gpus)
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)

    # Calibrate LR schedule to 150-epoch actual convergence window
    # Extended from parent's 100-epoch calibration to allow productive LR decay
    # throughout the full expected training window
    calibration_epochs = min(args.max_epochs, 150)
    max_steps_total = max(effective_steps_per_epoch * calibration_epochs, 1)

    print(f"[Main] n_gpus={n_gpus}, accum={accum}, "
          f"effective_steps_per_epoch={effective_steps_per_epoch}")
    print(f"[Main] max_steps_total={max_steps_total} (calibrated to {calibration_epochs} epochs)")

    lit = PerturbationLitModule(
        hidden_dim=args.hidden_dim,
        n_resblocks=args.n_resblocks,
        expand=args.expand,
        dropout=args.dropout,
        rank=args.rank,
        lr=args.lr,
        backbone_lr_late=args.backbone_lr_late,
        backbone_lr_deep=args.backbone_lr_deep,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        focal_gamma=args.focal_gamma,
        class_weight_down=args.class_weight_down,
        class_weight_neutral=args.class_weight_neutral,
        class_weight_up=args.class_weight_up,
        max_steps_total=max_steps_total,
        grad_clip_norm=args.grad_clip_norm,
        use_gene_weights=args.use_gene_weights,
        gene_weight_smoothing=args.gene_weight_smoothing,
        data_dir=args.data_dir,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-4)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    # Debug / fast-dev-run settings
    max_steps: int           = -1
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
        accumulate_grad_batches=accum,
        gradient_clip_val=args.grad_clip_norm,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=args.val_check_interval if (
            args.debug_max_step is None and not args.fast_dev_run
        ) else 1.0,
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
    test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"Node 1-1-2-2-1 - Extended Partial STRING_GNN Fine-Tuning (4 Layers) + Higher Backbone LR\n"
            f"Test results from trainer: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
