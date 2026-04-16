"""Node 1-1-1-1-2-1 – Frozen STRING_GNN + K=16 Neighborhood Attention + GenePriorBias.

Key improvements over node1-1-1-1-2 (parent, test F1=0.4814):
1. ADD K=16 PPI Neighborhood Attention: aggregates top-16 PPI neighbor embeddings
   using center-as-query scaled dot-product attention + learnable gating mechanism.
   Proven in node1-1-1-1-1 (F1=0.4846, +0.010 over shared parent node1-1-1-1).
   Addresses: static single-node embedding bottleneck by injecting PPI context.
   Architecture: W_q: Linear(256,64), W_k: Linear(256,64), W_gate: Linear(512,256)
   Gate: pert_emb = gate * center + (1-gate) * neighbor_context

2. RETAIN GenePriorBias: per-gene per-class learnable bias [3, 6640] with warmup=40 epochs
   (reduced from parent's 50 to allow 10 more epochs of bias learning within same budget).
   Uses persistent register_buffer('bias_active') — avoids inference-time bug (node1-3-2).
   Proven: +0.0068 F1 in parent (node1-1-1-1-2), +0.006 in independent node1-2-2-1.

3. INCREASE weight_decay 3e-2 → 4e-2 (extra regularization for 164K attn params).
4. EXTEND patience 15 → 25 (parent recommendation: peaked at epoch 63, need more convergence time).
5. REDUCE bias_warmup_epochs 50 → 40 (parent: pre-bias phase fully productive, allow more post-bias).
6. EXTEND max_epochs 250 → 300 (accommodate combined architecture convergence).

These two innovations are orthogonal:
- Neighborhood attention enriches the INPUT representation (PPI neighbor context)
- GenePriorBias calibrates the OUTPUT (per-gene DEG tendency correction)
Combined, they address different failure modes and should both contribute.
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

STRING_DIM = 256  # STRING_GNN hidden dimension


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
            "sample_idx":        idx,
            "pert_id":           self.pert_ids[idx],
            "symbol":            self.symbols[idx],
            "string_node_idx":   self.string_node_indices[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]  # [G] in {0, 1, 2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx": torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":         [b["pert_id"]  for b in batch],
        "symbol":          [b["symbol"]   for b in batch],
        "string_node_idx": torch.stack([b["string_node_idx"]   for b in batch]),
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
# GenePriorBias Module
# ---------------------------------------------------------------------------
class GenePriorBias(nn.Module):
    """Learnable per-gene-per-class bias with warmup guard.

    Shape: [N_CLASSES, N_GENES] = [3, 6640]
    Learns a static offset for each gene's class logits — effectively encoding
    "gene g is typically up/down/neutral across many perturbations".

    The `bias_active` flag is a PERSISTENT register_buffer, so it is saved in
    checkpoints and correctly loaded at test time. This avoids the inference
    bug confirmed in node1-3-2 where current_epoch defaulted to 0 on load.

    Key improvements over parent's implementation: warmup reduced 50→40 epochs,
    allowing 10 extra epochs of bias learning within the same training budget.
    """

    def __init__(self, n_classes: int = N_CLASSES, n_genes: int = N_GENES) -> None:
        super().__init__()
        # Learnable bias, initialized to zeros (neutral: no initial effect)
        self.bias = nn.Parameter(torch.zeros(n_classes, n_genes))
        # Persistent buffer: saved in checkpoint → loaded correctly at test time
        self.register_buffer("bias_active", torch.tensor(False))

    def activate(self) -> None:
        """Enable bias contribution. Called from LightningModule.on_train_epoch_start."""
        self.bias_active.fill_(True)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Add per-gene-per-class bias to logits when active.

        Args:
            logits: [B, 3, G] logit tensor
        Returns:
            [B, 3, G] logits (potentially with bias added)
        """
        if self.bias_active.item():
            # [3, G] broadcasts to [B, 3, G] automatically
            return logits + self.bias
        return logits


# ---------------------------------------------------------------------------
# Neighborhood Attention Aggregator Module
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """PPI neighborhood context aggregation via attention.

    For each perturbed gene, aggregates top-K PPI neighbor embeddings using
    scaled dot-product attention (center embedding as query) and learnable gating.

    Architecture (per batch):
        1. q = W_q(center_emb)                             [B, attn_dim]
        2. k = W_k(neigh_embs.reshape(-1, D)).reshape(B, K, attn_dim)
        3. attn = softmax(q @ k.T / sqrt(attn_dim))        [B, K]
        4. Incorporate PPI confidence: attn *= neigh_weights → re-normalize
        5. context = attn @ neigh_embs                     [B, D]
        6. gate = sigmoid(W_gate([center, context]))       [B, D]
        7. pert_emb = gate * center + (1-gate) * context  [B, D]

    For unknown pert_ids (not in STRING), returns center_emb unchanged.

    Parameters: ~164K (W_q: 16,448 + W_k: 16,448 + W_gate: 131,072 + biases)
    Proven: K=16, attn_dim=64 in node1-1-1-1-1 → test F1=0.4846 (+0.010 vs parent).
    """

    def __init__(
        self,
        dim: int = 256,
        attn_dim: int = 64,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn_dim = attn_dim
        self.scale = attn_dim ** -0.5
        self.W_q = nn.Linear(dim, attn_dim)
        self.W_k = nn.Linear(dim, attn_dim)
        self.W_gate = nn.Linear(dim * 2, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        center_emb: torch.Tensor,     # [B, D]
        neigh_embs: torch.Tensor,      # [B, K, D]
        neigh_weights: torch.Tensor,   # [B, K]  raw edge weights (zero for padding)
        valid_mask: torch.Tensor,      # [B] True if this sample has STRING neighbors
    ) -> torch.Tensor:
        """Return context-enriched embeddings [B, D].

        For samples without STRING neighbors (valid_mask=False), returns center_emb.
        """
        B, D = center_emb.shape
        K = neigh_embs.shape[1]

        # Compute attention scores: center queries neighbors
        q = self.W_q(center_emb)                               # [B, attn_dim]
        k_flat = self.W_k(neigh_embs.reshape(-1, D))            # [B*K, attn_dim]
        k = k_flat.reshape(B, K, self.attn_dim)                 # [B, K, attn_dim]

        # Scaled dot-product attention
        attn = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1) * self.scale  # [B, K]
        attn = F.softmax(attn, dim=-1)                          # [B, K]

        # Modulate by STRING PPI confidence weights (zero for padded entries)
        # This down-weights padded/low-confidence neighbors
        attn = attn * neigh_weights                              # [B, K]
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)  # re-normalize
        attn = self.attn_dropout(attn)                          # [B, K]

        # Weighted neighbor context
        context = torch.bmm(attn.unsqueeze(1), neigh_embs).squeeze(1)  # [B, D]

        # Learnable gating: how much neighborhood context to incorporate
        gate = torch.sigmoid(self.W_gate(torch.cat([center_emb, context], dim=-1)))  # [B, D]
        fused = gate * center_emb + (1.0 - gate) * context      # [B, D]

        # For unknown pert_ids (no STRING neighbors), preserve center embedding
        # valid_mask: [B] bool, True for known STRING genes
        fused = torch.where(valid_mask.unsqueeze(-1), fused, center_emb)

        return fused


# ---------------------------------------------------------------------------
# Model: Frozen STRING_GNN + Neighborhood Attention + GenePriorBias
# ---------------------------------------------------------------------------
class FrozenStringGNNNeighborPriorModel(pl.LightningModule):
    """Combined: Frozen STRING_GNN + K=16 PPI Neighborhood Attention + GenePriorBias.

    Architecture:
        1. STRING_GNN run ONCE at setup() → pre-computed node embeddings [18870, 256]
           stored as a fixed buffer; no GNN gradients during training.
        2. Pre-compute top-K=16 adjacency lists from graph_data.pt edge weights.
        3. Lookup pre-computed embedding by pert_id string_node_idx → [B, 256]
           + aggregate K=16 PPI neighbors via attention + gating → [B, 256]
           (learnable fallback for unknown pert_ids, no neighborhood)
        4. Simple 2-layer MLP projection (same as parent):
             LayerNorm(256) → Linear(256→bilinear_dim) → GELU → Dropout
             → LayerNorm(bilinear_dim) → Linear(bilinear_dim→bilinear_dim) → GELU → Dropout
        5. Bilinear output:
             logits[b,c,g] = h[b] · gene_class_emb[c,g]              → [B, 3, G]
        6. GenePriorBias: learnable [3, G] bias added to logits after warmup
             After warmup: final_logits = logits + gene_prior_bias.bias
        7. Weighted cross-entropy + mild label smoothing (eps=0.05)

    Combining both orthogonal innovations:
        - Neighborhood attention: enriches INPUT representation with PPI context
        - GenePriorBias: calibrates OUTPUT with per-gene DEG tendency
    """

    def __init__(
        self,
        bilinear_dim: int = 256,
        dropout: float = 0.40,
        lr: float = 3e-4,
        weight_decay: float = 4e-2,
        warmup_epochs: int = 20,
        T_max: int = 150,
        label_smoothing: float = 0.05,
        bias_warmup_epochs: int = 40,
        k_neighbors: int = 16,
        attn_dim: int = 64,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def _precompute_topk_neighbors(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        n_nodes: int,
        k: int,
    ):
        """Pre-compute top-K neighbors per node sorted by edge weight.

        Returns:
            topk_idx: [n_nodes, K] long — neighbor node indices (self-loop pad for missing)
            topk_wts: [n_nodes, K] float — raw edge weights (0.0 for padding)
        """
        edge_src = edge_index[0].numpy()
        edge_dst = edge_index[1].numpy()
        edge_wt  = edge_weight.numpy()

        # Sort edges by source node for efficient per-node extraction
        order = np.argsort(edge_src, kind="stable")
        edge_src_s = edge_src[order]
        edge_dst_s = edge_dst[order]
        edge_wt_s  = edge_wt[order]

        # Counts and offsets per source node
        counts  = np.bincount(edge_src_s, minlength=n_nodes)
        offsets = np.concatenate([[0], np.cumsum(counts)])

        topk_idx_np = np.zeros((n_nodes, k), dtype=np.int64)
        topk_wts_np = np.zeros((n_nodes, k), dtype=np.float32)
        # Default: self-loop for nodes with no neighbors (prevents invalid index)
        for i in range(n_nodes):
            topk_idx_np[i] = i

        for i in range(n_nodes):
            start, end = int(offsets[i]), int(offsets[i + 1])
            if start == end:
                # No neighbors → self-loop with zero weight (fallback, no context)
                continue
            nbr_dst = edge_dst_s[start:end]
            nbr_wt  = edge_wt_s[start:end]
            n_nbr   = len(nbr_dst)
            k_actual = min(k, n_nbr)
            if n_nbr <= k:
                idx = np.argsort(-nbr_wt)[:k_actual]
            else:
                # argpartition for efficiency, then sort within top-K
                part_idx = np.argpartition(-nbr_wt, k_actual)[:k_actual]
                idx = part_idx[np.argsort(-nbr_wt[part_idx])]
            topk_idx_np[i, :k_actual] = nbr_dst[idx]
            topk_wts_np[i, :k_actual] = nbr_wt[idx]
            # Remaining (k - k_actual) entries keep default self-loop / zero weight

        return (
            torch.from_numpy(topk_idx_np).long(),
            torch.from_numpy(topk_wts_np).float(),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        # Guard against repeated setup calls
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        hp = self.hparams

        # ----------------------------------------------------------------
        # 1. Pre-compute STRING_GNN node embeddings (backbone stays frozen)
        # ----------------------------------------------------------------
        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph["edge_index"].long()
        edge_weight = graph["edge_weight"].float()

        with torch.no_grad():
            gnn_out  = backbone(edge_index=edge_index, edge_weight=edge_weight)
            node_emb = gnn_out.last_hidden_state.float().detach()  # [18870, 256]

        self.register_buffer("node_embeddings", node_emb)  # [18870, 256], non-trainable

        # ----------------------------------------------------------------
        # 2. Pre-compute top-K neighbor indices and weights
        # ----------------------------------------------------------------
        n_nodes = node_emb.shape[0]
        topk_idx, topk_wts = self._precompute_topk_neighbors(
            edge_index, edge_weight, n_nodes, hp.k_neighbors
        )
        # Register as buffers → move to GPU automatically with Lightning
        self.register_buffer("topk_idx", topk_idx)    # [18870, K], non-trainable
        self.register_buffer("topk_wts", topk_wts)    # [18870, K], non-trainable

        del backbone, graph, edge_index, edge_weight, gnn_out

        # ----------------------------------------------------------------
        # 3. Learnable fallback for unknown pert_ids (~6.4% of training data)
        # ----------------------------------------------------------------
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 4. Neighborhood attention aggregator (NEW: from node1-1-1-1-1)
        # ----------------------------------------------------------------
        self.neighborhood_attn = NeighborhoodAttentionAggregator(
            dim=STRING_DIM,
            attn_dim=hp.attn_dim,
            attn_dropout=hp.attn_dropout,
        )

        # ----------------------------------------------------------------
        # 5. Simple 2-layer MLP head (same as parent)
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

        # ----------------------------------------------------------------
        # 7. GenePriorBias module (RETAINED from parent, warmup reduced 50→40)
        # ----------------------------------------------------------------
        self.gene_prior = GenePriorBias(n_classes=N_CLASSES, n_genes=N_GENES)

        # Class weights for weighted CE
        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        for _, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators for val/test
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # GenePriorBias activation hook
    # ------------------------------------------------------------------
    def on_train_epoch_start(self) -> None:
        """Activate GenePriorBias once warmup phase is complete (reduced 50→40)."""
        if (self.current_epoch >= self.hparams.bias_warmup_epochs
                and not self.gene_prior.bias_active.item()):
            self.gene_prior.activate()
            self.print(
                f"[Epoch {self.current_epoch}] GenePriorBias activated — "
                f"per-gene calibration bias contributing to logits."
            )

    # ------------------------------------------------------------------
    # Perturbation embedding with neighborhood attention
    # ------------------------------------------------------------------
    def _get_pert_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Lookup pre-computed embeddings + apply neighborhood attention.

        For known STRING nodes: apply K=16 attention-weighted neighbor aggregation.
        For unknown pert_ids: use learnable fallback (no neighborhood context).

        Args:
            string_node_idx: [B] long tensor, -1 for pert_ids not in STRING.
        Returns:
            [B, STRING_DIM] float32 perturbation embeddings.
        """
        B = string_node_idx.shape[0]
        device = self.node_embeddings.device
        dtype  = self.node_embeddings.dtype

        known   = string_node_idx >= 0   # [B] bool
        unknown = ~known                  # [B] bool

        # Initialize output buffer
        emb = torch.zeros(B, STRING_DIM, dtype=torch.float32, device=device)

        # ---- Known nodes: lookup + neighborhood attention ----
        if known.any():
            known_idx = string_node_idx[known]  # [Bk]

            # Center embeddings
            center_emb = self.node_embeddings[known_idx].float()  # [Bk, 256]

            # Neighbor embeddings: [Bk, K, 256]
            neigh_idx = self.topk_idx[known_idx]   # [Bk, K] long
            # Clamp to valid range as safety guard
            neigh_idx = neigh_idx.clamp(0, self.node_embeddings.shape[0] - 1)
            neigh_embs = self.node_embeddings[neigh_idx.reshape(-1)].float()
            neigh_embs = neigh_embs.reshape(known_idx.shape[0], self.hparams.k_neighbors, STRING_DIM)

            # Neighbor edge weights: [Bk, K]
            neigh_wts = self.topk_wts[known_idx].float()  # [Bk, K]

            # valid_mask: all known nodes have STRING neighbors
            valid_mask = torch.ones(known_idx.shape[0], dtype=torch.bool, device=device)

            # Apply neighborhood attention
            enriched = self.neighborhood_attn(
                center_emb=center_emb,
                neigh_embs=neigh_embs,
                neigh_weights=neigh_wts,
                valid_mask=valid_mask,
            )  # [Bk, 256]

            emb[known] = enriched

        # ---- Unknown nodes: learnable fallback ----
        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(unknown.sum(), dtype=torch.long, device=device)
            ).to(torch.float32)
            emb[unknown] = fb

        return emb  # [B, 256]

    def forward(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        pert_emb = self._get_pert_embeddings(string_node_idx)  # [B, 256]
        h = self.head(pert_emb)                                  # [B, bilinear_dim]

        # Bilinear interaction: h · gene_class_emb → [B, 3, G]
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)

        # GenePriorBias: add per-gene-per-class calibration (active after warmup)
        logits = self.gene_prior(logits)

        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy + mild label smoothing."""
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

            pred_map: Dict[int, torch.Tensor] = {}
            for i in range(len(idx_flat)):
                gid = int(idx_flat[i].item())
                if gid not in pred_map:
                    pred_map[gid] = preds_flat[i]  # [3, G]

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            rows = []
            for i in range(len(test_df)):
                if i not in pred_map:
                    continue
                pid = test_df.iloc[i]["pert_id"]
                sym = test_df.iloc[i]["symbol"]
                # Output format: [3, 6640] as required by calc_metric.py
                # pred_map[i] has shape [3, G] → tolist() → [[c0_g0,...], [c1_g0,...], [c2_g0,...]]
                pred = pred_map[i].float().cpu().numpy().tolist()  # [3, 6640]
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-1-1-1-2-1] Saved {len(rows)} test predictions.")

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
        self.print(f"Checkpoint: {train}/{total} params ({100 * train / total:.1f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: AdamW + linear warmup + CosineAnnealingLR
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=hp.lr, weight_decay=hp.weight_decay)

        # Phase 1: linear warmup from 0.1×lr to lr over warmup_epochs
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=hp.warmup_epochs,
        )
        # Phase 2: CosineAnnealingLR (T_max=150, extended from parent's 150)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=hp.T_max, eta_min=1e-6,
        )
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
        description="Node1-1-1-1-2-1 – Frozen STRING_GNN + Neighborhood Attention + GenePriorBias"
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=32)
    parser.add_argument("--global-batch-size",   type=int,   default=256)
    parser.add_argument("--max-epochs",          type=int,   default=300)
    parser.add_argument("--lr",                  type=float, default=3e-4)
    parser.add_argument("--weight-decay",        type=float, default=4e-2)
    parser.add_argument("--bilinear-dim",        type=int,   default=256)
    parser.add_argument("--dropout",             type=float, default=0.40)
    parser.add_argument("--warmup-epochs",       type=int,   default=20)
    parser.add_argument("--t-max",               type=int,   default=150,
                        dest="t_max")
    parser.add_argument("--label-smoothing",     type=float, default=0.05,
                        dest="label_smoothing")
    parser.add_argument("--bias-warmup-epochs",  type=int,   default=40,
                        dest="bias_warmup_epochs")
    parser.add_argument("--k-neighbors",         type=int,   default=16,
                        dest="k_neighbors")
    parser.add_argument("--attn-dim",            type=int,   default=64,
                        dest="attn_dim")
    parser.add_argument("--attn-dropout",        type=float, default=0.1,
                        dest="attn_dropout")
    parser.add_argument("--patience",            type=int,   default=25)
    parser.add_argument("--num-workers",         type=int,   default=4)
    parser.add_argument("--debug-max-step",      type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",        action="store_true",
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
    model = FrozenStringGNNNeighborPriorModel(
        bilinear_dim       = args.bilinear_dim,
        dropout            = args.dropout,
        lr                 = args.lr,
        weight_decay       = args.weight_decay,
        warmup_epochs      = args.warmup_epochs,
        T_max              = args.t_max,
        label_smoothing    = args.label_smoothing,
        bias_warmup_epochs = args.bias_warmup_epochs,
        k_neighbors        = args.k_neighbors,
        attn_dim           = args.attn_dim,
        attn_dropout       = args.attn_dropout,
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
        patience  = args.patience,   # 25 (extended for combined architecture convergence)
        min_delta = 1e-4,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy: DDP for multi-GPU, auto for single-GPU / fast_dev_run
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
    print(f"[Node1-1-1-1-2-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
