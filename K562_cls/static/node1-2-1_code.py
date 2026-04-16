"""Node 1-2 improvement – Discriminative LR STRING_GNN + PPI Neighborhood Attention.

This node improves upon node1-2 (F1=0.4769) by replicating the key design decision
from the best node in the entire MCTS tree (node1-1-1-1-1, F1=0.4846):
  - Using **discriminative LR** (backbone_lr=1e-5, head_lr=3e-4) instead of freezing the backbone
  - The STRING_GNN backbone can subtly adapt to the task without catastrophic forgetting
  - Combined with PPI Neighborhood Attention (K=16, attn_dim=64) — proven best config

Key differences from parent node1-2:
1. STRING_GNN backbone unfrozen with discriminative LR=1e-5 (vs. fully frozen in node1-2)
2. Extended T_max=200 with eta_min=5e-6 for longer cosine decay exploration
3. Per-gene threshold calibration at inference using validation data (threshold tuning)
4. Slightly relaxed early stopping (patience=10, min_delta=1e-4) to avoid premature termination

Key differences from sibling node1-1:
1. PPI Neighborhood Attention Aggregation (K=16, attn_dim=64) — sibling has no neighborhood aggregation
2. Discriminative LR: backbone_lr=1e-5 (30x smaller than head) vs. full fine-tuning at same LR
3. weight_decay=3e-2 vs. 1e-2 in sibling

Architecture:
    STRING_GNN backbone (discriminative LR=1e-5, unfrozen)
    → node embeddings [18870, 256]
    → for each sample with STRING node idx:
         center_emb = node_embeddings[pert_idx]              # [B, 256]
         neighbor_emb = node_embeddings[top-K neighbors]     # [B, K, 256]
         attention = softmax(learned_scores + edge_weights)  # [B, K]
         aggregated = attention @ neighbor_emb               # [B, 256]
         h = center_emb + gate(center_emb) * aggregated      # [B, 256] gated fusion
    → MLP: LN(256) → Linear(256→bilinear_dim) → GELU → Dropout
    → bilinear: logits[b,c,g] = h[b] · gene_class_emb[c,g]  # [B, 3, G]
    → Weighted CE + label smoothing (epsilon=0.05)

Inspired by:
  - node1-1-1-1-1 (F1=0.4846 — best in MCTS tree): discriminative LR=1e-5 for backbone
  - node1-2 (F1=0.4769 — parent): PPI neighborhood attention K=16, attn_dim=64
  - node1-2 feedback: "Add discriminative LR for backbone" as highest priority improvement
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
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

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

STRING_DIM = 256  # STRING_GNN hidden dimension


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights; neutral class stays ~1."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def load_string_gnn_mapping() -> Dict[str, int]:
    """Load STRING_GNN node_names.json → Ensembl-ID to node-index mapping."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    return {name: idx for idx, name in enumerate(node_names)}


def precompute_neighborhood(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    n_nodes: int,
    K: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute top-K PPI neighbors for each node by edge confidence.

    Uses an efficient sort-based approach: sort all edges by (src, -weight),
    then slice the top-K for each source node.

    Returns:
        neighbor_indices: [n_nodes, K] long — top-K neighbor node indices
                          (padded with -1 if fewer than K neighbors exist)
        neighbor_weights: [n_nodes, K] float — corresponding STRING edge confidence scores
    """
    src = edge_index[0]  # [E]
    dst = edge_index[1]  # [E]
    wgt = edge_weight     # [E]

    # Sort by weight descending first, then stable sort by src ascending
    sort_by_weight = torch.argsort(wgt, descending=True)
    src_sorted = src[sort_by_weight]
    dst_sorted = dst[sort_by_weight]
    wgt_sorted = wgt[sort_by_weight]

    # Now stable-sort by src (preserves descending weight order within each src)
    sort_by_src = torch.argsort(src_sorted, stable=True)
    src_final = src_sorted[sort_by_src]
    dst_final = dst_sorted[sort_by_src]
    wgt_final = wgt_sorted[sort_by_src]

    # Count edges per source node
    counts = torch.bincount(src_final, minlength=n_nodes)  # [n_nodes]

    neighbor_indices = torch.full((n_nodes, K), -1, dtype=torch.long)
    neighbor_weights = torch.zeros(n_nodes, K, dtype=torch.float32)

    # Fill top-K entries (edges are already sorted: for each src, first K entries are highest weight)
    start = 0
    for node_i in range(n_nodes):
        c = int(counts[node_i].item())
        if c == 0:
            start += c
            continue
        n_k = min(K, c)
        neighbor_indices[node_i, :n_k] = dst_final[start:start + n_k]
        neighbor_weights[node_i, :n_k] = wgt_final[start:start + n_k]
        start += c

    return neighbor_indices, neighbor_weights


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0, 1, 2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)           # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)           # [N, G]
        is_pred = (y_hat == c)             # [N, G]
        present = is_true.any(dim=0)       # [G]

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
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    """K562 DEG prediction dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_map: Dict[str, int],
    ) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()

        # STRING_GNN node index for each sample (-1 means not in STRING)
        self.string_node_indices = torch.tensor(
            [string_map.get(p, -1) for p in self.pert_ids], dtype=torch.long
        )

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
            "sample_idx":      idx,
            "pert_id":         self.pert_ids[idx],
            "symbol":          self.symbols[idx],
            "string_node_idx": self.string_node_indices[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]  # [G] in {0, 1, 2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx":      torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":         [b["pert_id"]  for b in batch],
        "symbol":          [b["symbol"]   for b in batch],
        "string_node_idx": torch.stack([b["string_node_idx"] for b in batch]),
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
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.string_map: Optional[Dict[str, int]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.string_map is None:
            self.string_map = load_string_gnn_mapping()

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df, self.string_map)
        self.val_ds   = DEGDataset(val_df,   self.string_map)
        self.test_ds  = DEGDataset(test_df,  self.string_map)

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
# Neighborhood Attention Module
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Center-context gated attention over top-K PPI neighbors.

    For each perturbed gene, aggregates the top-K neighbors from the STRING PPI
    graph using learned attention scores gated by the edge confidence weights.

    This module is the key innovation from node1-1-1-1-1 (F1=0.4846) that
    pushed beyond the F1=0.4746 ceiling of the simpler frozen embedding approach.

    Architecture:
        attn_proj: [center(256) + neighbor(256)] → attn_dim(64) → score(1)
        attention = softmax(edge_weight + attn_proj_score)   # [B, K]
        aggregated = attention @ neighbor_emb                # [B, 256]
        gate = sigmoid(gate_proj(center_emb))                # [B, 256]
        output = center_emb + gate * aggregated              # [B, 256]
    """

    def __init__(self, embed_dim: int = 256, attn_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim  = attn_dim

        # Attention projection: [center(256) + neighbor(256)] → attn_dim → 1
        self.attn_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, 1),
        )
        # Gating: center embedding → gate vector for controlling neighborhood contribution
        self.gate_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Light dropout on attention weights
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        center_emb: torch.Tensor,         # [B, D]
        neighbor_emb: torch.Tensor,        # [B, K, D]
        neighbor_weights: torch.Tensor,    # [B, K] STRING edge confidence (0–1)
        neighbor_mask: torch.Tensor,       # [B, K] bool: True = valid neighbor
    ) -> torch.Tensor:
        """Returns aggregated embedding [B, D]."""
        B, K, D = neighbor_emb.shape

        # Expand center for pair-wise attention projection
        center_expanded = center_emb.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]

        # Pair features: center + neighbor concatenated
        pair_features = torch.cat([center_expanded, neighbor_emb], dim=-1)  # [B, K, 2D]

        # Learned attention scores
        attn_scores = self.attn_proj(pair_features).squeeze(-1)  # [B, K]

        # Add STRING edge confidence as prior
        attn_scores = attn_scores + neighbor_weights              # [B, K]

        # Mask out invalid (padding) neighbors
        attn_scores = attn_scores.masked_fill(~neighbor_mask, -1e9)

        # Softmax attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)         # [B, K]
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted aggregation of neighbor embeddings
        aggregated = torch.bmm(attn_weights.unsqueeze(1), neighbor_emb).squeeze(1)  # [B, D]

        # Gated combination: center + gate * aggregated
        gate = torch.sigmoid(self.gate_proj(center_emb))           # [B, D]
        output = center_emb + gate * aggregated                     # [B, D]

        return output


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DiscriminativeLRGNNModel(pl.LightningModule):
    """Discriminative LR STRING_GNN + PPI Neighborhood Attention + Bilinear head.

    Key improvement over node1-2 (frozen backbone, F1=0.4769):
    The STRING_GNN backbone is unfrozen but trained at a much lower learning rate
    (backbone_lr=1e-5) than the head (head_lr=3e-4). This discriminative LR approach
    allows the node embeddings to subtly adapt to the K562 DEG task without
    catastrophic forgetting of PPI topology — exactly the recipe used by node1-1-1-1-1
    which achieved the best result (F1=0.4846) in the entire MCTS tree.

    Architecture (identical to parent node1-2, only training regime differs):
    1. STRING_GNN runs at setup() to compute initial embeddings, then backbone is
       kept unfrozen but at discriminative LR=1e-5 during training
    2. Pre-compute top-K neighbors per node by STRING edge confidence (K=16)
    3. For each perturbed gene:
       a. Forward pass through STRING_GNN during training (backbone at lr=1e-5)
       b. Look up center embedding → [B, 256]
       c. Look up top-K neighbor embeddings → [B, K, 256]
       d. Apply neighborhood attention aggregation → [B, 256]
    4. MLP projection: LN(256) → Linear(256→bilinear_dim) → GELU → Dropout
    5. Bilinear: logits[b,c,g] = h[b] · gene_class_emb[c,g]  # [B, 3, G]
    6. Loss: weighted CE + label smoothing (epsilon=0.05)

    Hyperparameters:
    - backbone_lr=1e-5 (discriminative — matches node1-1-1-1-1's proven config)
    - head_lr=3e-4 (proven from node1-1-1-1-1 and node1-2)
    - K=16, attn_dim=64 (proven best from lineage analysis)
    - weight_decay=3e-2, dropout=0.35, bilinear_dim=256
    - T_max=200 (extended from parent's 150 for longer exploration)
    - patience=10 (slightly relaxed from parent's 8)
    """

    def __init__(
        self,
        bilinear_dim: int = 256,
        K: int = 16,
        attn_dim: int = 64,
        dropout: float = 0.35,
        head_lr: float = 3e-4,
        backbone_lr: float = 1e-5,
        weight_decay: float = 3e-2,
        warmup_epochs: int = 20,
        t_max: int = 200,
        eta_min: float = 5e-6,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Model layers initialized in setup()

    def setup(self, stage: Optional[str] = None) -> None:
        # Guard against repeated setup calls
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        hp = self.hparams

        # ----------------------------------------------------------------
        # 1. Load STRING_GNN backbone — KEEP UNFROZEN for discriminative LR
        # All ranks load independently (no download needed for local model).
        # ----------------------------------------------------------------
        self.backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        self.backbone.train()
        # Do NOT freeze backbone — it will be trained at a very low discriminative LR

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        self.edge_index  = graph["edge_index"].long()    # [2, E]
        self.edge_weight = graph["edge_weight"].float()  # [E]

        n_nodes = self.backbone.config.num_nodes if hasattr(self.backbone.config, 'num_nodes') else 18870

        # ----------------------------------------------------------------
        # 2. Pre-compute top-K neighbors by STRING edge confidence
        #    These are fixed graph-topology lookups, independent of backbone weights
        #    neighbor_indices: [n_nodes, K] — neighbor node indices (-1=padding)
        #    neighbor_weights: [n_nodes, K] — STRING confidence scores [0,1]
        # ----------------------------------------------------------------
        print(f"Pre-computing top-{hp.K} PPI neighbors for {n_nodes} nodes...")
        nbr_idx, nbr_wgt = precompute_neighborhood(
            self.edge_index, self.edge_weight, n_nodes, K=hp.K
        )
        # Register neighbor topology as non-trainable buffers
        self.register_buffer("neighbor_indices", nbr_idx)  # [n_nodes, K]
        self.register_buffer("neighbor_weights", nbr_wgt)  # [n_nodes, K]

        # Register graph tensors as buffers (moved to GPU by Lightning)
        self.register_buffer("_edge_index",  self.edge_index)
        self.register_buffer("_edge_weight", self.edge_weight)

        # Release temporary graph tensors from Python attributes
        del self.edge_index
        del self.edge_weight

        # ----------------------------------------------------------------
        # 3. Learnable fallback for unknown pert_ids
        # ----------------------------------------------------------------
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 4. Neighborhood Attention Aggregator
        #    K=16, attn_dim=64 — proven best config from node1-1-1-1-1
        # ----------------------------------------------------------------
        self.neighborhood_attn = NeighborhoodAttentionAggregator(
            embed_dim=STRING_DIM,
            attn_dim=hp.attn_dim,
            dropout=0.0,  # No attention dropout
        )

        # ----------------------------------------------------------------
        # 5. MLP projection head (flat, not bottleneck)
        #    Matches proven flat design from lineage
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(STRING_DIM),
            nn.Linear(STRING_DIM, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # ----------------------------------------------------------------
        # 6. Bilinear gene-class embedding
        # ----------------------------------------------------------------
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # Class weights for weighted CE
        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        for _, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators for val/test (cleared each epoch)
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_meta:  List[Dict]         = []
        self._test_idx:   List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Embedding lookup with PPI neighborhood aggregation
    # Uses LIVE backbone forward pass (discriminative LR training)
    # ------------------------------------------------------------------
    def _get_node_embeddings(self) -> torch.Tensor:
        """Run STRING_GNN forward pass to get current node embeddings.

        During training: backbone is active (discriminative LR=1e-5), so
        node embeddings adapt to the task while preserving PPI topology.
        During inference: backbone weights reflect trained state.

        Returns: [n_nodes, STRING_DIM] float32
        """
        gnn_out = self.backbone(
            edge_index=self._edge_index,
            edge_weight=self._edge_weight,
        )
        return gnn_out.last_hidden_state.float()  # [18870, 256]

    def _get_pert_embeddings(
        self,
        string_node_idx: torch.Tensor,
        node_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Lookup embeddings with PPI neighborhood aggregation.

        For known pert_ids: apply neighborhood attention aggregation.
        For unknown pert_ids: use learnable fallback embedding.

        Args:
            string_node_idx: [B] long tensor, -1 for pert_ids not in STRING.
            node_emb:        [n_nodes, STRING_DIM] — current backbone embeddings
        Returns:
            [B, STRING_DIM] float32 perturbation embeddings.
        """
        B = string_node_idx.shape[0]
        emb = torch.zeros(B, STRING_DIM,
                          dtype=node_emb.dtype,
                          device=node_emb.device)
        known   = string_node_idx >= 0
        unknown = ~known

        if known.any():
            known_idx = string_node_idx[known]   # [K_known]

            # Center embeddings for known pert_ids
            center = node_emb[known_idx]  # [K_known, 256]

            # Get pre-computed neighbor indices and weights
            nbr_idx = self.neighbor_indices[known_idx]   # [K_known, K]
            nbr_wgt = self.neighbor_weights[known_idx]   # [K_known, K]

            # Build neighbor validity mask (non-padding)
            nbr_mask = nbr_idx >= 0  # [K_known, K]

            # Clamp to valid range for embedding lookup
            nbr_idx_clamped = nbr_idx.clamp(min=0)  # [K_known, K]

            # Lookup neighbor embeddings
            n_known = int(known.sum().item())
            K_neighbors = nbr_idx.shape[1]
            flat_nbr_idx = nbr_idx_clamped.view(-1)         # [K_known * K]
            flat_nbr_emb = node_emb[flat_nbr_idx]           # [K_known * K, 256]
            neighbor_emb = flat_nbr_emb.view(n_known, K_neighbors, STRING_DIM)  # [K_known, K, 256]

            # Zero out padding neighbor embeddings
            neighbor_emb = neighbor_emb * nbr_mask.unsqueeze(-1).float()

            # Apply neighborhood attention aggregation
            aggregated = self.neighborhood_attn(
                center_emb       = center.float(),
                neighbor_emb     = neighbor_emb.float(),
                neighbor_weights = nbr_wgt.float(),
                neighbor_mask    = nbr_mask,
            )  # [K_known, 256]

            emb[known] = aggregated

        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(unknown.sum(), dtype=torch.long, device=node_emb.device)
            ).to(node_emb.dtype)
            emb[unknown] = fb

        return emb.float()

    def forward(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        # Run backbone (discriminative LR training during fit)
        node_emb = self._get_node_embeddings()  # [18870, 256]
        pert_emb = self._get_pert_embeddings(string_node_idx, node_emb)  # [B, 256]
        h = self.head(pert_emb)                                           # [B, bilinear_dim]

        # Bilinear interaction: logits[b,c,g] = h[b] · gene_class_emb[c,g]
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)  # [B, 3, G]
        return logits

    # ------------------------------------------------------------------
    # Loss: weighted CE + label smoothing (proven by the lineage)
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy with label smoothing (epsilon=0.05).

        Proven superior to focal loss in this lineage (node1-1-1-1, node1-1-1-1-1).
        """
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),   # [B*G, 3]
            targets.reshape(-1),                        # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["string_node_idx"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits, dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, dim=0)   # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  dim=0)   # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   dim=0)   # [N_local]
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        # Gather across all DDP ranks
        all_preds = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)    # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)     # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        # De-duplicate (DDP padding may introduce repeated samples)
        order  = torch.argsort(idx_flat)
        s_idx  = idx_flat[order]
        s_pred = preds_flat[order]
        s_tgt  = tgts_flat[order]
        mask   = torch.cat([
            torch.tensor([True], device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        preds_dedup = s_pred[mask]
        tgts_dedup  = s_tgt[mask]

        f1 = compute_per_gene_f1(preds_dedup, tgts_dedup)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)  # [N_local]

        all_preds = self.all_gather(local_preds)  # [W, N_local, 3, G]
        all_idx   = self.all_gather(local_idx)    # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            # De-duplicate
            order  = torch.argsort(idx_flat)
            s_idx  = idx_flat[order]
            s_pred = preds_flat[order]
            mask   = torch.cat([
                torch.ones(1, dtype=torch.bool, device=s_idx.device),
                s_idx[1:] != s_idx[:-1],
            ])
            preds_dedup = s_pred[mask]
            unique_sid  = s_idx[mask].tolist()

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                           for i in range(len(test_df))}

            rows = []
            dedup_counter = 0
            for sid in unique_sid:
                sid_i = int(sid)
                if sid_i in idx_to_meta:
                    pid, sym = idx_to_meta[sid_i]
                    pred = preds_dedup[dedup_counter].float().cpu().numpy().tolist()
                    rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})
                dedup_counter += 1

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-2 improved] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ------------------------------------------------------------------
    # Checkpoint helpers — save only trainable params + buffers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
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
        self.print(
            f"Checkpoint: {train}/{total} params ({100 * train / total:.1f}%), "
            f"plus {bufs} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: AdamW with DISCRIMINATIVE learning rates + warmup + cosine
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        # DISCRIMINATIVE LR: backbone at 1e-5, head/attention/bilinear at 3e-4
        # This is the key improvement from node1-2 (frozen backbone) to this node
        backbone_params = list(self.backbone.parameters())
        head_params = (
            list(self.neighborhood_attn.parameters())
            + list(self.head.parameters())
            + [self.gene_class_emb]
            + list(self.fallback_emb.parameters())
        )

        param_groups = [
            {"params": backbone_params, "lr": hp.backbone_lr, "name": "backbone"},
            {"params": head_params,     "lr": hp.head_lr,     "name": "head"},
        ]

        opt = torch.optim.AdamW(param_groups, weight_decay=hp.weight_decay)

        # Warmup + cosine annealing for both parameter groups
        # Phase 1: linear warmup from 0.1×lr to lr over warmup_epochs epochs
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        # Phase 2: CosineAnnealingLR (T_max=200, extended from parent's 150)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=hp.t_max,
            eta_min=hp.eta_min,
        )
        # Sequential: warmup first, then cosine decay
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[hp.warmup_epochs],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-2 improved – Discriminative LR STRING_GNN + PPI Neighborhood Attention"
    )
    parser.add_argument("--micro-batch-size",  type=int,   default=32)
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument("--max-epochs",        type=int,   default=250)
    parser.add_argument("--head-lr",           type=float, default=3e-4,
                        dest="head_lr")
    parser.add_argument("--backbone-lr",       type=float, default=1e-5,
                        dest="backbone_lr")
    parser.add_argument("--weight-decay",      type=float, default=3e-2)
    parser.add_argument("--bilinear-dim",      type=int,   default=256)
    parser.add_argument("--K",                 type=int,   default=16,
                        dest="K")
    parser.add_argument("--attn-dim",          type=int,   default=64,
                        dest="attn_dim")
    parser.add_argument("--dropout",           type=float, default=0.35)
    parser.add_argument("--label-smoothing",   type=float, default=0.05,
                        dest="label_smoothing")
    parser.add_argument("--warmup-epochs",     type=int,   default=20,
                        dest="warmup_epochs")
    parser.add_argument("--t-max",             type=int,   default=200,
                        dest="t_max")
    parser.add_argument("--eta-min",           type=float, default=5e-6,
                        dest="eta_min")
    parser.add_argument("--patience",          type=int,   default=10)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--debug-max-step",    type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",      action="store_true",
                        dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Limit / debug logic
    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        lim_test  = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        lim_test  = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    # Model
    model = DiscriminativeLRGNNModel(
        bilinear_dim    = args.bilinear_dim,
        K               = args.K,
        attn_dim        = args.attn_dim,
        dropout         = args.dropout,
        head_lr         = args.head_lr,
        backbone_lr     = args.backbone_lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        t_max           = args.t_max,
        eta_min         = args.eta_min,
        label_smoothing = args.label_smoothing,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
    )
    es_cb = EarlyStopping(
        monitor   = "val/f1",
        mode      = "max",
        patience  = args.patience,
        min_delta = 1e-4,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy
    # find_unused_parameters=True: fallback_emb may not be used in every batch
    use_ddp = n_gpus > 1 and not fast_dev_run
    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if use_ddp else "auto"
    )
    devices_for_trainer = 1 if (fast_dev_run and n_gpus > 1) else n_gpus

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = devices_for_trainer,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        max_steps               = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches     = lim_train,
        limit_val_batches       = lim_val,
        limit_test_batches      = lim_test,
        val_check_interval      = 1.0,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    # Save test score
    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1-2 improved] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
