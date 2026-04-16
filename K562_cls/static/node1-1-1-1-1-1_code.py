"""Node 1-1-1-1-1-1 – Frozen STRING_GNN + Expanded Neighborhood Aggregation (K=32, reduced attn_dim=32).

Improves on parent node1-1-1-1-1 (test F1=0.4846, best in MCTS tree) by:

1. Reduce attn_dim from 64 → 32 (PRIMARY regularization change):
   - Halves the attention projection parameters (~164K → ~82K extra params)
   - Parent feedback explicitly recommended this to close the train-val loss gap (~0.205)
   - Reduces overfitting while retaining neighborhood context benefit

2. Extend T_max from 100 → 150 (PRIMARY training schedule change):
   - Parent peaked at epoch 70 out of T_max=100; LR was decaying through the peak
   - Extending T_max gives the model more room to explore around the cosine LR peak
   - Lower eta_min (1e-7) ensures a proper long tail for fine-grained optimization

3. Extend max_epochs from 150 → 220 to accommodate the longer T_max schedule
   - Extended schedule requires more epochs to fully converge
   - patience kept at 5 to prevent over-training past the peak

4. Increase weight_decay 3e-2 → 4e-2:
   - Parent feedback: "3e-2 was insufficient to fully offset the new parameters"
   - Stronger L2 regularization to better control the combined neighborhood+head parameters

5. Expand K from 16 → 32 neighbors:
   - Captures more distant PPI regulatory relationships
   - Parent feedback mentioned K=32 as a promising direction to explore
   - With reduced attn_dim=32, the compute/memory cost increase is modest

6. RETAIN all proven components from parent:
   - Frozen STRING_GNN backbone + pre-computed embeddings
   - NeighborhoodAttention architecture (center-as-query, PPI confidence prior, gating)
   - 2-layer MLP head (bilinear_dim=256)
   - Weighted cross-entropy + label smoothing (ε=0.05)
   - lr=3e-4, 20-epoch linear warmup → CosineAnnealingLR
   - Bilinear gene-class embedding [3, 6640, 256]

Architecture:
    Pre-computed node_embeddings [18870, 256] (frozen STRING_GNN, computed once)
    Pre-computed topk_neighbors [18870, K=32] and topk_weights [18870, K=32]
    → index center by pert_id's STRING node index → center_emb [B, 256]
    → index top-K=32 neighbors → neigh_embs [B, 32, 256]
    → attention(center, neighbors, attn_dim=32) + PPI confidence weighting → context [B, 256]
    → gate(center, context) → pert_emb [B, 256]
    (fallback: learnable embedding for ~6.3% unknown pert_ids)
    → MLP head: LN(256) → Linear(256→256) → GELU → Dropout(0.4)
                → LN(256) → Linear(256→256) → GELU → Dropout(0.4)
    → [B, 256]
    → bilinear: logits[b,c,g] = h[b] · gene_class_emb[c,g]  # [B, 3, G]
    → weighted CE + label smoothing (ε=0.05)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from torch.utils.data import DataLoader, Dataset, SequentialSampler
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

STRING_DIM = 256   # STRING_GNN hidden dimension
ATTN_DIM   = 32    # Attention projection dimension (reduced from 64 to limit params/overfitting)
TOPK       = 32    # Number of top PPI neighbors to aggregate (increased from 16)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights for weighted cross-entropy."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def load_string_gnn_mapping() -> Dict[str, int]:
    """Load STRING_GNN node_names.json → Ensembl-ID to node-index mapping."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    return {name: idx for idx, name in enumerate(node_names)}


def compute_topk_neighbors(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    k: int,
) -> tuple:
    """Precompute top-K neighbors per node by edge weight.

    Args:
        edge_index: [2, E] long — source and destination indices
        edge_weight: [E] float — STRING combined_score weights
        num_nodes: total number of nodes in the graph
        k: number of top neighbors to keep per node

    Returns:
        topk_neighbors: [num_nodes, k] long — top-K neighbor indices
        topk_weights:   [num_nodes, k] float — corresponding softmaxed edge weights
                        (for nodes with fewer than K neighbors, padded with self-loops)
    """
    src = edge_index[0]
    dst = edge_index[1]
    weights = edge_weight

    # Initialize: default to self-loops (neighbor = self, weight = 0)
    topk_neighbors_np = torch.zeros(num_nodes, k, dtype=torch.long)
    topk_weights_raw  = torch.zeros(num_nodes, k, dtype=torch.float)

    # For self-loop initialization, set neighbor = node index itself
    for i in range(num_nodes):
        topk_neighbors_np[i] = i  # default: self-loop

    # Group edges by source node
    sort_idx = torch.argsort(src)
    src_sorted = src[sort_idx]
    dst_sorted = dst[sort_idx]
    wt_sorted  = weights[sort_idx]

    # Process each unique source node
    unique_srcs, counts = torch.unique_consecutive(src_sorted, return_counts=True)
    offset = 0
    for i, (node_id, cnt) in enumerate(zip(unique_srcs.tolist(), counts.tolist())):
        nb_dst = dst_sorted[offset:offset + cnt]    # [cnt] neighbor indices
        nb_wt  = wt_sorted[offset:offset + cnt]     # [cnt] weights

        # Take top-K by weight (or all if fewer than K)
        actual_k = min(k, cnt)
        topk_vals, topk_idx = torch.topk(nb_wt, actual_k)

        top_dst = nb_dst[topk_idx]
        top_wt  = topk_vals

        topk_neighbors_np[node_id, :actual_k] = top_dst
        topk_weights_raw[node_id, :actual_k]   = top_wt
        # Remaining slots stay as self-loops with weight 0

        offset += cnt

    # Softmax-normalize the weights per node (only over valid neighbors)
    # Nodes with fewer than K real neighbors have self-loops with weight=0
    # softmax gives positive attention to self-loops (preventing zeros)
    # This is intentional: fallback to self when few neighbors
    topk_weights_soft = torch.softmax(topk_weights_raw, dim=1)

    return topk_neighbors_np, topk_weights_soft


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0, 1, 2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)            # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)            # [N, G]
        is_pred = (y_hat == c)              # [N, G]
        present = is_true.any(dim=0)        # [G]

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
        sampler = SequentialSampler(self.test_ds)
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
            sampler=sampler,
        )


# ---------------------------------------------------------------------------
# Neighborhood Attention Module
# ---------------------------------------------------------------------------
class NeighborhoodAttention(nn.Module):
    """Lightweight PPI neighborhood attention with reduced attn_dim=32.

    Given a center embedding and K neighbor embeddings (with PPI confidence priors),
    computes an attention-weighted neighborhood context and gates it with the center.

    The attn_dim is reduced from 64 (parent) to 32 to cut attention parameters by half,
    reducing overfitting while retaining the neighborhood aggregation benefit.

    Architecture:
        q = W_q(center)            [B, attn_dim]
        k = W_k(neigh_embs)        [B, K, attn_dim]
        attn = softmax(q @ k^T / sqrt(d)) * ppi_weights   [B, 1, K]
        attn = attn / sum(attn)    (renormalize after PPI weighting)
        context = attn @ neigh_embs  [B, 1, 256] → squeeze → [B, 256]
        gate = sigmoid(W_gate([center, context]))   [B, 256]
        output = gate * center + (1 - gate) * context    [B, 256]
    """

    def __init__(self, emb_dim: int = 256, attn_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn_dim = attn_dim
        self.scale    = attn_dim ** -0.5

        # Reduced projection dims: 256→32 instead of 256→64
        self.W_q    = nn.Linear(emb_dim, attn_dim, bias=False)
        self.W_k    = nn.Linear(emb_dim, attn_dim, bias=False)
        # Gate projects concatenated center+context (512) to emb_dim (256)
        self.W_gate = nn.Linear(emb_dim * 2, emb_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize gate bias toward identity (center preferred initially)
        nn.init.zeros_(self.W_gate.bias)
        nn.init.xavier_uniform_(self.W_gate.weight)

    def forward(
        self,
        center: torch.Tensor,          # [B, D]
        neigh_embs: torch.Tensor,      # [B, K, D]
        ppi_weights: torch.Tensor,     # [B, K]  — softmaxed STRING confidence
    ) -> torch.Tensor:
        """Returns [B, D] context-aware perturbation embedding."""
        B, K, D = neigh_embs.shape

        # Attention scores: center queries over neighbor keys
        q = self.W_q(center)           # [B, attn_dim]
        k = self.W_k(neigh_embs.reshape(B * K, D)).reshape(B, K, self.attn_dim)  # [B, K, attn_dim]

        # Scaled dot-product attention
        attn = (q.unsqueeze(1) @ k.transpose(1, 2)) * self.scale  # [B, 1, K]
        attn = attn.squeeze(1)                                       # [B, K]

        # Incorporate PPI edge confidence as prior (multiplicative)
        attn = F.softmax(attn, dim=-1)      # [B, K]
        attn = attn * ppi_weights           # [B, K] — weight by STRING confidence
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)  # renormalize

        # Dropout on attention weights for regularization
        attn = self.dropout(attn)           # [B, K]

        # Context vector: attention-weighted sum of neighbor embeddings
        context = (attn.unsqueeze(1) @ neigh_embs).squeeze(1)  # [B, D]

        # Learnable gating: blend center and neighborhood context
        gate = torch.sigmoid(self.W_gate(
            torch.cat([center, context], dim=-1)  # [B, 2D]
        ))  # [B, D]
        output = gate * center + (1.0 - gate) * context  # [B, D]

        return output


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class NeighborhoodContextModelV2(pl.LightningModule):
    """Frozen STRING_GNN with expanded PPI neighborhood context (K=32, attn_dim=32).

    Improvements over parent node1-1-1-1-1 (F1=0.4846):
    1. Reduced attn_dim (64→32): halves attention projection parameters to close train-val gap
    2. Extended T_max (100→150): longer cosine LR decay allows continued improvement past epoch 70
    3. Increased weight_decay (3e-2→4e-2): stronger L2 regularization for neighborhood+head params
    4. Expanded K (16→32): more PPI neighbors capture broader regulatory context

    Architecture:
        1. STRING_GNN run ONCE at setup() to pre-compute embeddings [18870, 256]
           → stored as a fixed buffer; no GNN gradients during training.
        2. Pre-compute top-K=32 adjacency from graph_data.pt → topk_neighbors/weights buffers
        3. For each batch, look up center + K=32 neighbors → attention-weighted context
           (NeighborhoodAttention module with reduced attn_dim=32)
        4. Feed context-aware embedding through 2-layer MLP head (same as parent):
             LN(256) → Linear(256→256) → GELU → Dropout(0.4)
             → LN(256) → Linear(256→256) → GELU → Dropout(0.4)
        5. Bilinear output: logits[b,c,g] = h[b] · gene_class_emb[c,g]
        6. Weighted cross-entropy + label smoothing ε=0.05 (same as parent)
    """

    def __init__(
        self,
        bilinear_dim:    int   = 256,
        dropout:         float = 0.40,
        attn_dim:        int   = 32,    # REDUCED from parent's 64 to limit params
        topk:            int   = 32,    # EXPANDED from parent's 16 for richer context
        lr:              float = 3e-4,
        weight_decay:    float = 4e-2,  # INCREASED from parent's 3e-2
        warmup_epochs:   int   = 20,
        T_max:           int   = 150,   # EXTENDED from parent's 100
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
        # 1. Pre-compute STRING_GNN node embeddings (backbone stays frozen)
        # Model is at a local path — all ranks load independently (no download).
        # ----------------------------------------------------------------
        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph["edge_index"].long()
        edge_weight = graph["edge_weight"].float()

        # One-time forward pass on CPU — produces the fixed lookup table
        with torch.no_grad():
            gnn_out  = backbone(edge_index=edge_index, edge_weight=edge_weight)
            node_emb = gnn_out.last_hidden_state.float().detach()  # [18870, 256]

        # Register as buffer → Lightning moves it to GPU automatically
        self.register_buffer("node_embeddings", node_emb)

        # ----------------------------------------------------------------
        # 2. Pre-compute top-K PPI neighbors per node (K=32)
        # This is computed from the same graph_data.pt edge structure.
        # EXPANDED from K=16 (parent) to K=32 for broader regulatory context.
        # ----------------------------------------------------------------
        num_nodes = node_emb.shape[0]
        print(f"[Node1-1-1-1-1-1] Pre-computing top-{hp.topk} PPI neighbors for {num_nodes} nodes...")
        topk_nb, topk_wt = compute_topk_neighbors(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            k=hp.topk,
        )

        # Register as non-trainable buffers → auto-moved to GPU
        self.register_buffer("topk_neighbors", topk_nb)  # [N, K] long
        self.register_buffer("topk_weights",   topk_wt)  # [N, K] float

        # Release backbone memory (not needed again)
        del backbone, graph, edge_index, edge_weight, gnn_out
        print("[Node1-1-1-1-1-1] PPI neighbor buffers ready.")

        # ----------------------------------------------------------------
        # 3. Learnable fallback for unknown pert_ids (~6.4% of training data)
        # ----------------------------------------------------------------
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 4. PPI Neighborhood Attention Module (reduced attn_dim=32)
        # Reduced: W_q [256→32], W_k [256→32], W_gate [512→256]
        # Total: 32×256 + 32×256 + 256×512 + 256 = ~148K params (vs ~164K in parent)
        # ----------------------------------------------------------------
        self.neighborhood_attn = NeighborhoodAttention(
            emb_dim  = STRING_DIM,
            attn_dim = hp.attn_dim,   # 32 (reduced from 64)
            dropout  = 0.1,           # gentle dropout on attention weights
        )

        # ----------------------------------------------------------------
        # 5. Simple 2-layer MLP head (same as parent node1-1-1-1-1)
        #    dropout=0.40 retained from parent
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(STRING_DIM),
            nn.Linear(STRING_DIM, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
            nn.LayerNorm(hp.bilinear_dim),
            nn.Linear(hp.bilinear_dim, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # ----------------------------------------------------------------
        # 6. Bilinear gene-class embedding (same as parent)
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
        self._test_idx:   List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []

    # ------------------------------------------------------------------
    # Forward: PPI Neighborhood Context Aggregation (K=32)
    # ------------------------------------------------------------------
    def _get_pert_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Get context-aware perturbation embeddings with PPI neighborhood aggregation.

        For known pert_ids (in STRING graph):
          1. Look up center embedding from frozen buffer
          2. Look up top-K=32 neighbor embeddings using pre-computed adjacency
          3. Apply NeighborhoodAttention (attn_dim=32) to produce context-aware embedding
        For unknown pert_ids (string_node_idx == -1):
          - Use learnable fallback embedding (no neighborhood)

        Args:
            string_node_idx: [B] long tensor, -1 for pert_ids not in STRING.
        Returns:
            [B, STRING_DIM] float32 context-aware perturbation embeddings.
        """
        B = string_node_idx.shape[0]
        K = self.hparams.topk
        device = self.node_embeddings.device

        known   = string_node_idx >= 0
        unknown = ~known

        # Prepare output buffer
        out_emb = torch.zeros(B, STRING_DIM, dtype=torch.float32, device=device)

        # ---- Known pert_ids: neighborhood aggregation ----
        if known.any():
            known_idx  = string_node_idx[known]                         # [B_k]
            B_k        = known_idx.shape[0]

            # Center embeddings
            center_emb = self.node_embeddings[known_idx].float()        # [B_k, 256]

            # Neighbor indices and weights from pre-computed buffers
            nb_idx = self.topk_neighbors[known_idx]                     # [B_k, K=32]
            nb_wt  = self.topk_weights[known_idx].float()               # [B_k, K=32]

            # Neighbor embeddings: gather from frozen buffer
            # Flatten for efficient indexing, then reshape
            nb_idx_flat = nb_idx.reshape(-1)                             # [B_k * K]
            nb_emb_flat = self.node_embeddings[nb_idx_flat].float()      # [B_k * K, 256]
            neigh_embs  = nb_emb_flat.reshape(B_k, K, STRING_DIM)       # [B_k, K, 256]

            # Apply neighborhood attention (attn_dim=32): returns context-aware embedding
            context_emb = self.neighborhood_attn(
                center    = center_emb,   # [B_k, 256]
                neigh_embs = neigh_embs,  # [B_k, K=32, 256]
                ppi_weights = nb_wt,      # [B_k, K=32]
            )  # [B_k, 256]

            out_emb[known] = context_emb

        # ---- Unknown pert_ids: use learnable fallback ----
        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(unknown.sum(), dtype=torch.long, device=device)
            ).float()
            out_emb[unknown] = fb

        return out_emb  # [B, STRING_DIM]

    def forward(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        pert_emb = self._get_pert_embeddings(string_node_idx)  # [B, STRING_DIM]
        h = self.head(pert_emb)                                 # [B, bilinear_dim]

        # Bilinear interaction (same as parent)
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)  # [B, 3, G]
        return logits

    # ------------------------------------------------------------------
    # Loss: weighted CE + label smoothing (same as parent)
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy + mild label smoothing.

        Identical to parent node1-1-1-1-1's loss: sqrt-inverse-frequency class
        weights + label_smoothing=0.05. This proved optimal in the parent.
        """
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, 3]
            targets.reshape(-1),                       # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["string_node_idx"])
        loss   = self._loss(logits, batch["labels"])
        # NOTE: sync_dist=True intentionally omitted in training step
        # to avoid potential DDP collective deadlocks with AMP loss scaler.
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            # sync_dist=True here is safe — no AMP backward is pending.
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
        # Use sample_idx for DDP-safe deduplication
        self._test_idx.append(batch["sample_idx"].detach())
        self._test_pert_ids.extend(batch["pert_id"])
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

            pred_map: Dict[int, torch.Tensor] = {}
            for i in range(len(idx_flat)):
                gid = int(idx_flat[i].item())
                if gid not in pred_map:
                    pred_map[gid] = preds_flat[i]

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            rows = []
            for i in range(len(test_df)):
                if i not in pred_map:
                    continue
                pid = test_df.iloc[i]["pert_id"]
                sym = test_df.iloc[i]["symbol"]
                # pred_map[i] is [3, G]; .tolist() gives nested list [3, 6640]
                pred = pred_map[i].float().cpu().numpy().tolist()  # [3, 6640]
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-1-1-1-1-1] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()
        self._test_pert_ids.clear()

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
    # Optimizer: AdamW + linear warmup + CosineAnnealingLR (T_max=150)
    # Extended T_max and lower eta_min for longer LR exploration
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        # All trainable parameters: neighborhood_attn, fallback_emb, head, gene_class_emb
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=hp.lr, weight_decay=hp.weight_decay)

        # Phase 1: linear warmup from 0.1×lr to lr over warmup_epochs
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        # Phase 2: CosineAnnealingLR with extended T_max=150 and lower eta_min=1e-7
        # Extended to allow continued improvement past the parent's epoch 70 peak
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=hp.T_max,       # 150 (extended from parent's 100)
            eta_min=1e-7,         # lower floor (parent used 1e-6)
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
        description="Node1-1-1-1-1-1 – Frozen STRING_GNN + Expanded PPI Neighborhood (K=32, attn_dim=32)"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=32)
    parser.add_argument("--global-batch-size",  type=int,   default=256)
    parser.add_argument("--max-epochs",         type=int,   default=220)
    parser.add_argument("--lr",                 type=float, default=3e-4)
    parser.add_argument("--weight-decay",       type=float, default=4e-2)
    parser.add_argument("--bilinear-dim",       type=int,   default=256)
    parser.add_argument("--dropout",            type=float, default=0.40)
    parser.add_argument("--attn-dim",           type=int,   default=32,
                        dest="attn_dim")
    parser.add_argument("--topk",               type=int,   default=32)
    parser.add_argument("--warmup-epochs",      type=int,   default=20)
    parser.add_argument("--t-max",              type=int,   default=150,
                        dest="t_max")
    parser.add_argument("--label-smoothing",    type=float, default=0.05,
                        dest="label_smoothing")
    parser.add_argument("--patience",           type=int,   default=5)
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--debug-max-step",     type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",       action="store_true",
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

    # val_check_interval: run validation once per (mini-)epoch.
    val_check_interval = int(lim_train) if isinstance(lim_train, int) else 1.0

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    # Model
    model = NeighborhoodContextModelV2(
        bilinear_dim    = args.bilinear_dim,
        dropout         = args.dropout,
        attn_dim        = args.attn_dim,
        topk            = args.topk,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        T_max           = args.t_max,
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
        patience  = args.patience,   # 5 (same as parent)
        min_delta = 1e-4,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy: use DDP for multi-GPU; avoid DDP with fast_dev_run (can deadlock with AMP)
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
        val_check_interval      = val_check_interval,
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
    print(f"[Node1-1-1-1-1-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
