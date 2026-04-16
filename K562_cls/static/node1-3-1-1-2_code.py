"""Node 1-3-1-1-2: Lineage Escape — STRING-only + K=16 NeighborhoodAttention +
Two-Stage GenePriorBias + Gene-Frequency-Weighted Loss + Tight SWA.

Strategy: Abandon the failed node1-3-1-1 scFoundation-fusion lineage (confirmed
dead end at F1~0.44 across 3+ nodes) and adopt the proven STRING-only recipe from
the node1-1-1-1-2-1 lineage (F1=0.4913), with targeted improvements drawn from
node4-2-1-2-1 (F1=0.4893) and node4-2-1-1-1 (F1=0.4868):

1. STRING-only backbone: frozen STRING_GNN + K=16 PPI neighborhood attention (K=16,
   attn_dim=64) — proven foundation from node1-1-1-1-1 (+0.010) and node1-1-1-4-1-1
2. Two-stage GenePriorBias: scale=0.5 init, 50-epoch gradient warmup — avoids the
   catastrophic disruption seen in single-stage bias nodes; contributed +0.017 F1 in
   node1-1-1-1-2-1 (proven most effective in the STRING-only lineage)
3. Gene-frequency-weighted loss: per-gene loss scaling by inverse DEG frequency
   (diversity_factor=4.0) — proven in node4-2-1-2-1 (+0.010 F1) and node4-2-3-1
4. Tight SWA: Stochastic Weight Averaging starting at epoch 170, every 5 epochs,
   ~8–10 checkpoints — node4-2-1-1-1 proved that 8 tight SWA checkpoints add +0.003
   F1; node1-3-1-2 proved SWA effective in STRING-only context (+0.0148 F1)
5. Hyperparameters from node1-1-1-1-2-1 (F1=0.4913): lr=3e-4, wd=4e-2, dropout=0.40,
   patience=25, T_max=150 (not 200 — node1-1-1-3-2 confirmed T_max=150 beats T_max=200
   by +0.0019 for this architecture)

This node is a clean escape from the contaminated scFoundation-fusion lineage,
reverting to the STRING-only architecture that consistently achieves F1~0.49+ in
the clean node1 and node4 lineages.

Memory connections:
- node1-1-1-1-2-1 (F1=0.4913): base recipe — K=16 NbAttn + GenePriorBias 50-ep warmup
- node4-2-1-2-1 (F1=0.4893): gene-frequency-weighted loss factor=4.0 works well
- node4-2-1-1-1 (F1=0.4868): tight SWA (8 checkpoints) adds +0.003 F1
- node1-3-1-2 (F1=0.4837): STRING-only + GenePriorBias + gene-div + SWA in node1 lineage
- node1-3-1-1/sibling (both F1=0.4433): scFoundation fusion lineage is dead end
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3

# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"
LABEL_GENES_TXT = DATA_ROOT / "label_genes.txt"

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights; neutral class stays approx 1."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  - softmax probabilities
        targets: [N, G]    long   - class labels in {0,1,2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)  # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)        # [N, G]
        is_pred = (y_hat == c)          # [N, G]
        present = is_true.any(dim=0)    # [G]

        tp = (is_pred & is_true).float().sum(0)
        fp = (is_pred & ~is_true).float().sum(0)
        fn = (~is_pred & is_true).float().sum(0)

        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(
            prec + rec > 0,
            2 * prec * rec / (prec + rec + 1e-8),
            torch.zeros_like(prec),
        )
        f1_per_gene += f1_c * present.float()
        n_present   += present.float()

    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Pre-computation utilities
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_string_gnn_embeddings() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load STRING_GNN and compute all node embeddings. Returns (emb[N,256], pert_id→idx)."""
    model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
    model.eval()
    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
    node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())

    edge_index = graph["edge_index"]
    ew = graph.get("edge_weight", None)

    outputs = model(edge_index=edge_index, edge_weight=ew)
    emb = outputs.last_hidden_state.float().cpu()  # [18870, 256]

    pert_to_idx = {name: i for i, name in enumerate(node_names)}
    del model
    return emb, pert_to_idx


def compute_gene_diversity_weights(diversity_factor: float = 4.0) -> torch.Tensor:
    """Compute per-gene diversity weights based on DEG frequency.

    Genes that are rarely perturbed (few DEGs across training set) get higher weight
    to encourage the model to attend to all genes equally. This is the gene-frequency-
    weighted loss strategy proven in node4-2-1-2-1 (+0.010 F1) and node4-2-3-1.

    Args:
        diversity_factor: scaling factor for weight range (larger = wider range)
    Returns:
        [N_GENES] float tensor of per-gene loss weights (mean = 1.0)
    """
    train_df = pd.read_csv(TRAIN_TSV, sep="\t")
    # Count how often each gene is differentially expressed (non-zero label)
    gene_deg_count = np.zeros(N_GENES, dtype=np.float32)
    for label_str in train_df["label"].tolist():
        labels = np.array(json.loads(label_str))
        gene_deg_count += (labels != 0).astype(np.float32)

    # Genes rarely perturbed → high weight; always perturbed → low weight
    # Use inverse frequency with smoothing to avoid extreme values
    n_samples = len(train_df)
    deg_freq = gene_deg_count / (n_samples + 1e-6)  # fraction of samples with DEG

    # Inverse frequency weighting: weight = 1 / (freq + epsilon)
    # Normalize to mean = 1.0
    epsilon = 0.1  # smoothing to prevent extreme weights
    raw_weight = 1.0 / (deg_freq + epsilon)
    # Scale by diversity_factor to amplify rare-gene signal
    normalized_weight = raw_weight / raw_weight.mean()
    # Clip to [1/diversity_factor, diversity_factor] to prevent extreme values
    clipped_weight = np.clip(
        normalized_weight * diversity_factor,
        1.0 / diversity_factor,
        diversity_factor
    )
    # Final normalize to mean 1.0
    final_weight = clipped_weight / clipped_weight.mean()
    return torch.tensor(final_weight, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Neighborhood Attention Module (K=16, proven in node1-1-1-1-1, F1=0.4846)
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Aggregate top-K PPI neighbors with learned attention weights.

    For each perturbed gene p, computes:
        attn = softmax(W_q(h_p) · W_k(h_n) / sqrt(d) for each neighbor n)
        context = sum(attn_n * W_v(h_n)) for n in topK_neighbors(p)
        output = LayerNorm(h_p + center_gate * context)

    This module proved decisive in node1-1-1-1-1 (+0.010 F1 over no-attention parent)
    and consistently matches/outperforms simple mean-pooling in the STRING lineage.

    K=16 was confirmed as optimal in node1-1-1-4-1-1 (F1=0.4913) over K=24 and K=32.
    attn_dim=64 matches the proven configuration from node1-1-1-4-1-1.
    """

    def __init__(
        self,
        dim: int = 256,
        attn_dim: int = 64,
        k: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.attn_dim = attn_dim
        self.scale = attn_dim ** -0.5

        self.q_proj = nn.Linear(dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.center_gate = nn.Linear(dim, 1)
        self.out_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_emb: torch.Tensor,        # [B, D] — the perturbed gene embeddings
        all_emb: torch.Tensor,          # [N, D] — all STRING node embeddings
        neighbor_indices: torch.Tensor, # [B, K] — top-K neighbor indices (long)
        neighbor_weights: torch.Tensor, # [B, K] — STRING edge confidence weights (float)
    ) -> torch.Tensor:
        """Return enriched embedding [B, D]."""
        B, D = query_emb.shape
        K = neighbor_indices.shape[1]

        # Gather neighbor embeddings: [B, K, D]
        neighbor_emb = all_emb[neighbor_indices.reshape(-1)].reshape(B, K, D)

        # Attention scores: [B, K]
        q = self.q_proj(query_emb).unsqueeze(1)            # [B, 1, attn_dim]
        k = self.k_proj(neighbor_emb)                       # [B, K, attn_dim]
        scores = (q * k).sum(-1) * self.scale              # [B, K]

        # Weight by STRING edge confidence (normalized over neighbors)
        conf_w = neighbor_weights / (neighbor_weights.sum(-1, keepdim=True) + 1e-6)
        scores = scores + torch.log(conf_w + 1e-6)        # bias by edge confidence

        attn = torch.softmax(scores, dim=-1)               # [B, K]
        attn = self.dropout(attn)

        # Aggregate values: [B, D]
        v = self.v_proj(neighbor_emb)                      # [B, K, D]
        context = (attn.unsqueeze(-1) * v).sum(1)          # [B, D]

        # Center-context gate: learn how much to weight neighborhood vs center
        gate = torch.sigmoid(self.center_gate(query_emb))  # [B, 1]
        return self.out_norm(query_emb + gate * context)


# ---------------------------------------------------------------------------
# GenePriorBias Module (Two-stage: frozen for warmup, then learnable)
# ---------------------------------------------------------------------------
class TwoStageGenePriorBias(nn.Module):
    """Gene-specific class bias tensor [3, 6640] with two-stage training.

    Stage 1 (epochs 0..warmup-1): bias is initialized at scale * log(CLASS_FREQ)
        and gradients are zeroed — prevents the catastrophic disruption observed when
        bias activation spikes loss. The bias is registered as a persistent buffer
        (not a Parameter) during this stage to prevent optimizer state allocation.

    Stage 2 (epochs warmup..): bias becomes learnable via a Parameter, allowing
        per-gene per-class calibration of the model's output distribution.

    The two-stage approach successfully provided +0.017 F1 in node1-1-1-1-2-1 (F1=0.4913)
    and is adopted as the default bias initialization strategy.

    bias_scale=0.5 initialization reduces the magnitude of the prior anchoring,
    allowing the model to more quickly escape the neutral-heavy initialization
    while still providing convergence guidance.
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        n_genes: int = N_GENES,
        warmup_epochs: int = 50,
        bias_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # Initialize from log class frequencies scaled by bias_scale
        log_freq = torch.tensor(
            [float(np.log(f + 1e-9)) for f in CLASS_FREQ], dtype=torch.float32
        )
        init_bias = (log_freq * bias_scale).unsqueeze(1).expand(-1, n_genes).clone()  # [3, G]

        # Register as persistent buffer initially (not updated by optimizer)
        self.register_buffer("bias_buffer", init_bias.clone())
        # Parameter that becomes active in stage 2
        self.bias = nn.Parameter(init_bias.clone(), requires_grad=False)
        # Persistent buffer tracking whether the bias was ever activated; saved in
        # checkpoint so that after ckpt_path='best' reload the forward pass correctly
        # uses the learned bias even though current_epoch resets to 0.
        self.register_buffer("_activated", torch.tensor(0, dtype=torch.long))

    def set_epoch(self, epoch: int) -> None:
        """Call at start of each epoch to control bias trainability."""
        self.current_epoch = epoch
        if epoch >= self.warmup_epochs:
            # Activate learnable bias
            if not self.bias.requires_grad:
                # Copy current buffer into parameter (in case buffer was updated)
                self.bias.data.copy_(self.bias_buffer)
                self.bias.requires_grad_(True)
            # Mark as activated so checkpoint-restored inference uses the learned bias
            self._activated.fill_(1)
        else:
            self.bias.requires_grad_(False)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G] → [B, 3, G] with gene-specific class priors added.

        Uses `_activated` buffer (persisted in checkpoint) to correctly select
        the learned bias after ckpt_path='best' reloads the model, even though
        `current_epoch` resets to 0 after re-instantiation.
        """
        if self.current_epoch < self.warmup_epochs and self._activated.item() == 0:
            # Stage 1: frozen buffer bias (no gradient computation)
            return logits + self.bias_buffer.unsqueeze(0).detach()
        else:
            # Stage 2 (or checkpoint-restored): use learnable/learned parameter bias
            return logits + self.bias.unsqueeze(0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    """K562 DEG prediction dataset."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        self.sample_indices = list(range(len(df)))
        has_label = "label" in df.columns and df["label"].notna().all()
        if has_label:
            self.labels = [
                torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
                for row in df["label"].tolist()
            ]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx": idx,
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]   # [G] in {0,1,2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx": torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":    [b["pert_id"] for b in batch],
        "symbol":     [b["symbol"]  for b in batch],
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df)
        self.val_ds   = DEGDataset(val_df)
        self.test_ds  = DEGDataset(test_df)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Lightning Model
# ---------------------------------------------------------------------------
class StringGNNNeighborBiasDEGModel(pl.LightningModule):
    """Frozen STRING_GNN + K=16 Neighborhood Attention + TwoStageGenePriorBias + Gene-Freq-Weighted Loss.

    Architecture:
        pert_id
          |
          STRING_GNN (FROZEN, pre-computed buffer [18870, 256])
          |   Direct lookup: pert_id → node_idx → node_embeddings[idx] → [B, 256]
          |
          NeighborhoodAttentionAggregator (K=16, attn_dim=64)
          |   Aggregates top-16 PPI neighbors weighted by edge confidence
          |   Center-context gate: enriched embedding [B, 256]
          |
          2-layer MLP head with bilinear gene output
          |   LayerNorm → Linear(256→256) → GELU → Dropout → [B, 256]
          |   bilinear: gene_class_emb [3, 6640, 256] → logits [B, 3, 6640]
          |
          TwoStageGenePriorBias [3, 6640]
          |   Stage 1 (epochs 0-49): frozen buffer bias (no gradient)
          |   Stage 2 (epoch 50+): learnable parameter
          |
          Output: [B, 3, 6640] logits → softmax probabilities

    Loss: weighted CE + label smoothing + per-gene diversity weighting
    """

    def __init__(
        self,
        attn_k: int = 16,
        attn_dim: int = 64,
        bilinear_dim: int = 256,
        head_dropout: float = 0.40,
        lr: float = 3e-4,
        weight_decay: float = 4e-2,
        warmup_epochs: int = 10,
        t_max: int = 150,
        label_smoothing: float = 0.05,
        bias_warmup_epochs: int = 50,
        bias_scale: float = 0.5,
        diversity_factor: float = 4.0,
        min_lr_ratio: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        # Guard: skip re-initialization if already set up (e.g., test stage after fit).
        # All parameters and buffers are restored by Lightning's checkpoint loading.
        if getattr(self, "_setup_complete", False):
            return
        self._setup_complete = True

        hp = self.hparams

        # ---- Load STRING_GNN embeddings and graph topology ----
        self.print("Precomputing STRING_GNN embeddings (frozen)...")
        string_emb, pert_to_gnn_idx = precompute_string_gnn_embeddings()
        # [18870, 256] float32 — persistent buffer (saved in checkpoint)
        self.register_buffer("node_embeddings", string_emb)  # [18870, 256]

        # ---- Build pert_id → STRING_GNN index mapping ----
        # We need all pert_ids across splits for consistent indexing
        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        all_pert_ids = (
            train_df["pert_id"].tolist() +
            val_df["pert_id"].tolist() +
            test_df["pert_id"].tolist()
        )
        unique_sorted = sorted(set(all_pert_ids))
        self.pert_to_pos = {pid: i for i, pid in enumerate(unique_sorted)}
        M = len(unique_sorted)

        gnn_idx_tensor = torch.tensor(
            [pert_to_gnn_idx.get(pid, -1) for pid in unique_sorted], dtype=torch.long
        )
        # [M] — persistent so checkpoint saves the mapping
        self.register_buffer("pert_gnn_idx", gnn_idx_tensor)

        # ---- Build K-nearest-neighbor graph from STRING PPI (using cosine sim) ----
        # For each unique pert_id, pre-compute K=16 nearest STRING PPI neighbors
        # and their edge weights based on the STRING graph edge_weight.
        # We use STRING edge confidence as weights, or cosine sim if no direct edge.
        self.print(f"Building K={hp.attn_k} neighborhood index for {M} pert_ids...")
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
        edge_index = graph["edge_index"]    # [2, E]
        edge_weight = graph.get("edge_weight", None)  # [E] or None

        # Build adjacency: for each valid pert_id in STRING, find top-K neighbors
        # by edge weight; for missing pert_ids, use random valid nodes as fallback.
        valid_mask = gnn_idx_tensor >= 0  # [M]
        valid_gnn_indices = gnn_idx_tensor[valid_mask]  # subset in STRING

        # Build sparse adjacency dict: node_idx → {neighbor_idx: weight}
        adj: Dict[int, Dict[int, float]] = {}
        E = edge_index.shape[1]
        for e_idx in range(E):
            src = int(edge_index[0, e_idx])
            dst = int(edge_index[1, e_idx])
            w = float(edge_weight[e_idx]) if edge_weight is not None else 1.0
            if src not in adj:
                adj[src] = {}
            if dst not in adj[src]:
                adj[src][dst] = 0.0
            adj[src][dst] = max(adj[src][dst], w)

        K = hp.attn_k
        # [M, K] neighbor indices and [M, K] neighbor weights
        neighbor_idx_list = []
        neighbor_w_list = []

        # Fallback: if no neighbors found, use all-zeros (index 0 with weight 1.0)
        for pos_i in range(M):
            gnn_idx = int(gnn_idx_tensor[pos_i])
            if gnn_idx < 0:
                # Not in STRING: use zeros as fallback (repeated index 0)
                neighbor_idx_list.append([0] * K)
                neighbor_w_list.append([1e-6] * K)
                continue

            nbrs = adj.get(gnn_idx, {})
            if len(nbrs) == 0:
                # No neighbors in adjacency: use self as fallback
                neighbor_idx_list.append([gnn_idx] * K)
                neighbor_w_list.append([1.0] * K)
                continue

            # Sort by weight descending, take top-K
            sorted_nbrs = sorted(nbrs.items(), key=lambda x: -x[1])
            top_k = sorted_nbrs[:K]
            # Pad to K if fewer than K neighbors exist
            while len(top_k) < K:
                top_k.append(top_k[-1])  # repeat last neighbor

            idx_row = [n[0] for n in top_k]
            w_row   = [n[1] for n in top_k]
            neighbor_idx_list.append(idx_row)
            neighbor_w_list.append(w_row)

        nb_idx = torch.tensor(neighbor_idx_list, dtype=torch.long)    # [M, K]
        nb_w   = torch.tensor(neighbor_w_list, dtype=torch.float32)   # [M, K]
        # Persistent buffers for checkpoint saving
        self.register_buffer("nb_idx", nb_idx)
        self.register_buffer("nb_w", nb_w)
        self.print(f"Neighborhood buffer built: nb_idx={nb_idx.shape}, nb_w={nb_w.shape}")

        # ---- Fallback embedding for pert_ids not in STRING ----
        self.fallback_emb = nn.Parameter(torch.zeros(1, 256))

        # ---- Neighborhood Attention Module ----
        self.nb_attn = NeighborhoodAttentionAggregator(
            dim=256,
            attn_dim=hp.attn_dim,
            k=hp.attn_k,
            dropout=0.0,  # no dropout in attention itself, only in head
        )

        # ---- Bilinear MLP head ----
        # Proven design from node1-1-1-1-1 and node1-1-1-4-1-1
        self.head_norm = nn.LayerNorm(256)
        self.head_linear = nn.Linear(256, hp.bilinear_dim)
        self.head_act = nn.GELU()
        self.head_dropout = nn.Dropout(hp.head_dropout)
        # Bilinear gene-class embedding [3, 6640, bilinear_dim]
        # This design directly maps backbone embeddings to gene-specific class logits
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.01
        )

        # ---- TwoStageGenePriorBias ----
        self.gene_prior_bias = TwoStageGenePriorBias(
            n_classes=N_CLASSES,
            n_genes=N_GENES,
            warmup_epochs=hp.bias_warmup_epochs,
            bias_scale=hp.bias_scale,
        )

        # ---- Class weights for CE loss ----
        self.register_buffer("class_weights", get_class_weights())

        # ---- Gene diversity weights for per-gene loss scaling ----
        self.print(f"Computing gene diversity weights (diversity_factor={hp.diversity_factor})...")
        gene_div_w = compute_gene_diversity_weights(diversity_factor=hp.diversity_factor)
        self.register_buffer("gene_div_weights", gene_div_w)  # [N_GENES]

        # Cast all trainable parameters to float32
        for name, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor] = []
        self._val_tgts:  List[torch.Tensor] = []
        self._val_idx:   List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    # ---- Embedding extractor ----
    def _get_gnn_emb(self, pert_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get STRING_GNN embeddings and neighborhood tensors for a batch.

        Returns:
            center_emb: [B, 256] — the perturbed gene embeddings
            nb_idx: [B, K] — top-K neighbor indices into node_embeddings
            nb_w: [B, K] — neighbor edge weights
        """
        pos = torch.tensor(
            [self.pert_to_pos[pid] for pid in pert_ids], dtype=torch.long, device=self.device
        )
        gnn_node_idx = self.pert_gnn_idx[pos]     # [B] — indices into STRING GNN nodes
        valid = gnn_node_idx >= 0
        safe_idx = gnn_node_idx.clamp(min=0)
        emb = self.node_embeddings[safe_idx]       # [B, 256]
        fallback = self.fallback_emb.expand(emb.shape[0], -1).to(emb.dtype)
        emb = torch.where(valid.unsqueeze(-1), emb, fallback)

        # Get neighborhood indices and weights
        nb_indices = self.nb_idx[pos]              # [B, K]
        nb_weights = self.nb_w[pos]                # [B, K]

        return emb.float(), nb_indices, nb_weights.float()

    # ---- Forward ----
    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        center_emb, nb_indices, nb_weights = self._get_gnn_emb(pert_ids)

        # Enrich with neighborhood attention
        enriched = self.nb_attn(
            query_emb=center_emb,
            all_emb=self.node_embeddings,
            neighbor_indices=nb_indices,
            neighbor_weights=nb_weights,
        )  # [B, 256]

        # 2-layer head with bilinear output
        h = self.head_dropout(self.head_act(self.head_linear(self.head_norm(enriched))))  # [B, bilinear_dim]

        # Bilinear interaction: h [B, D] x gene_class_emb [3, G, D] → logits [B, 3, G]
        # Efficient einsum: b,cgd -> bcg
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)  # [B, 3, G]

        # Add gene-specific class priors (two-stage bias)
        logits = self.gene_prior_bias(logits)

        return logits

    # ---- Loss with gene-frequency diversity weighting ----
    def _loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        use_diversity: bool = True,
    ) -> torch.Tensor:
        """Weighted cross-entropy with optional per-gene diversity weighting.

        Args:
            logits:  [B, 3, G]
            targets: [B, G] in {0,1,2}
            use_diversity: if True, scale loss per gene by diversity weights
        """
        B, C, G = logits.shape
        # Standard weighted CE: [B*G] per-sample loss
        ce_loss = F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),   # [B*G, 3]
            targets.reshape(-1),                        # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
            reduction="none",
        )  # [B*G]

        if use_diversity:
            # Apply per-gene diversity weights: gene axis = every B-th element in [B*G]
            # Reshape to [B, G], multiply by gene_div_weights [G], then mean
            ce_per_gene = ce_loss.reshape(B, G)           # [B, G]
            weighted    = ce_per_gene * self.gene_div_weights.unsqueeze(0)  # [B, G]
            return weighted.mean()
        else:
            return ce_loss.mean()

    # ---- Steps ----
    def on_train_epoch_start(self) -> None:
        """Update bias epoch counter at start of each training epoch."""
        self.gene_prior_bias.set_epoch(self.current_epoch)
        if self.current_epoch == self.hparams.bias_warmup_epochs:
            self.print(f"[Epoch {self.current_epoch}] GenePriorBias activated — bias now learnable")

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        pert_ids = batch["pert_id"]
        labels   = batch["labels"]  # [B, G]
        logits   = self(pert_ids)
        loss     = self._loss(logits, labels, use_diversity=True)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"], use_diversity=False)
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits, dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        # Also update bias epoch for sanity val (stage=fit, epoch=0)
        self.gene_prior_bias.set_epoch(self.current_epoch)

        local_preds = torch.cat(self._val_preds, dim=0)     # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  dim=0)     # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   dim=0)     # [N_local]
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)    # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)     # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        order  = torch.argsort(idx_flat)
        s_idx  = idx_flat[order]
        s_pred = preds_flat[order]
        s_tgt  = tgts_flat[order]
        mask   = torch.cat([torch.ones(1, dtype=torch.bool, device=s_idx.device),
                            s_idx[1:] != s_idx[:-1]])
        preds_dedup = s_pred[mask]
        tgts_dedup  = s_tgt[mask]

        f1 = compute_per_gene_f1(preds_dedup, tgts_dedup)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"], use_diversity=False)
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)    # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)    # [N_local]
        all_preds   = self.all_gather(local_preds)           # [W, N_local, 3, G]
        all_idx     = self.all_gather(local_idx)             # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            order  = torch.argsort(idx_flat)
            s_idx  = idx_flat[order]
            s_pred = preds_flat[order]
            mask   = torch.cat([torch.ones(1, dtype=torch.bool, device=s_idx.device),
                                s_idx[1:] != s_idx[:-1]])
            preds_dedup = s_pred[mask]       # [N_test, 3, G]
            unique_sid  = s_idx[mask].tolist()

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                           for i in range(len(test_df))}

            # IMPORTANT: use `enumerate` to index into preds_dedup (deduplicated array).
            # Using a position from s_idx (full sorted array) would give wrong results
            # because preds_dedup is indexed by dedup rank, not by position in s_idx.
            rows = []
            for i, sid in enumerate(unique_sid):
                pid, sym = idx_to_meta[int(sid)]
                pred_list = preds_dedup[i].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"[Node1-3-1-1-2] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ---- Checkpoint helpers ----
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full:
                    trainable[key] = full[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full:
                trainable[key] = full[key]
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        bufs  = sum(b.numel() for _, b in self.named_buffers())
        pct   = (100 * train / total) if total > 0 else 0.0
        # Use try/except because this method may be called by SWA on a model copy
        # that is not attached to a Trainer (self.print() requires Trainer attachment)
        try:
            self.print(
                f"Saving checkpoint: {train}/{total} params "
                f"({pct:.2f}%), plus {bufs} buffer values"
            )
        except RuntimeError:
            # Not attached to a Trainer (e.g., SWA average model copy)
            rank = int(os.environ.get("RANK", "0"))
            if rank == 0:
                print(
                    f"Saving checkpoint: {train}/{total} params "
                    f"({pct:.2f}%), plus {bufs} buffer values"
                )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        """Load partial checkpoint (trainable params + buffers only)."""
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer ----
    def configure_optimizers(self):
        hp = self.hparams

        # Single LR for all trainable parameters (proven in STRING-only lineage)
        all_trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(all_trainable, lr=hp.lr, weight_decay=hp.weight_decay)

        # LambdaLR: linear warmup (from 10% to 100%) + cosine decay with min_lr_ratio floor
        def lr_lambda(epoch: int) -> float:
            if epoch < hp.warmup_epochs:
                return 0.1 + 0.9 * (epoch / max(1, hp.warmup_epochs))
            cos_epoch = epoch - hp.warmup_epochs
            cos_progress = min(float(cos_epoch), float(hp.t_max)) / float(hp.t_max)
            cos_val = 0.5 * (1.0 + np.cos(np.pi * cos_progress))
            return hp.min_lr_ratio + (1.0 - hp.min_lr_ratio) * cos_val

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-3-1-1-2: STRING-only + K=16 NbAttn + TwoStageGenePriorBias + Gene-Freq-Weighted Loss + SWA"
    )
    parser.add_argument("--micro-batch-size",  type=int,   default=32)
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument("--max-epochs",        type=int,   default=220)
    parser.add_argument("--lr",                type=float, default=3e-4,
                        help="Single LR for all trainable params")
    parser.add_argument("--weight-decay",      type=float, default=4e-2)
    parser.add_argument("--attn-k",            type=int,   default=16,
                        help="K for neighborhood attention (proven optimal)")
    parser.add_argument("--attn-dim",          type=int,   default=64,
                        help="Attention projection dimension (64 proven in node1-1-1-4-1-1)")
    parser.add_argument("--bilinear-dim",      type=int,   default=256,
                        help="Bilinear output head dimension")
    parser.add_argument("--head-dropout",      type=float, default=0.40)
    parser.add_argument("--warmup-epochs",     type=int,   default=10)
    parser.add_argument("--t-max",             type=int,   default=150,
                        help="Cosine decay T_max — 150 proven optimal over 200 in node1-1-1-3-2")
    parser.add_argument("--min-lr-ratio",      type=float, default=0.05)
    parser.add_argument("--label-smoothing",   type=float, default=0.05)
    parser.add_argument("--bias-warmup-epochs", type=int,  default=50,
                        help="Epochs before GenePriorBias gradients are activated")
    parser.add_argument("--bias-scale",        type=float, default=0.5,
                        help="Scale factor for GenePriorBias initialization")
    parser.add_argument("--diversity-factor",  type=float, default=4.0,
                        help="Per-gene diversity loss weight range; proven in node4-2-1-2-1")
    parser.add_argument("--swa-start-epoch",   type=int,   default=170,
                        help="SWA starts after this epoch; tight window = 8–10 checkpoints")
    parser.add_argument("--swa-lr",            type=float, default=1e-5,
                        help="SWA learning rate (low to preserve convergence region)")
    parser.add_argument("--patience",          type=int,   default=25)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--debug-max-step",    type=int,   default=None, dest="debug_max_step")
    parser.add_argument("--fast-dev-run",      action="store_true",     dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        lim_test  = 1.0
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        lim_test  = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule + Model (Lightning calls dm.setup() internally; no manual call needed)
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)

    model = StringGNNNeighborBiasDEGModel(
        attn_k           = args.attn_k,
        attn_dim         = args.attn_dim,
        bilinear_dim     = args.bilinear_dim,
        head_dropout     = args.head_dropout,
        lr               = args.lr,
        weight_decay     = args.weight_decay,
        warmup_epochs    = args.warmup_epochs,
        t_max            = args.t_max,
        label_smoothing  = args.label_smoothing,
        bias_warmup_epochs = args.bias_warmup_epochs,
        bias_scale       = args.bias_scale,
        diversity_factor = args.diversity_factor,
        min_lr_ratio     = args.min_lr_ratio,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=3,        # Save top-3 for potential ensemble
        save_last=False,
    )
    early_stop_callback = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.patience,
        min_delta=1e-4,
        verbose=True,
    )
    lr_monitor   = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=10)

    # Tight SWA: start at swa_start_epoch, will collect ~8–10 checkpoints
    # Only apply if not in debug mode
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor, progress_bar]
    if not fast_dev_run and args.debug_max_step is None:
        swa_callback = StochasticWeightAveraging(
            swa_lrs=args.swa_lr,
            swa_epoch_start=args.swa_start_epoch,
            annealing_epochs=10,
            annealing_strategy="cos",
        )
        callbacks.append(swa_callback)

    csv_logger         = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tensorboard_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        limit_train_batches=lim_train,
        limit_val_batches=lim_val,
        limit_test_batches=lim_test,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=callbacks,
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=dm)
    else:
        test_results = trainer.test(model, datamodule=dm, ckpt_path="best")

    # Save test results
    if test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        with open(score_path, "w") as f:
            f.write(str(test_results))
        print(f"[Node1-3-1-1-2] Test results: {test_results}")


if __name__ == "__main__":
    main()
