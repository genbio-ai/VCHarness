"""Node 1-1-3-2 – Frozen STRING_GNN + Neighborhood Attention + Fixed GenePriorBias + EMA

Strategy: Synthesizes the proven best techniques from the MCTS tree:

1. **Frozen STRING_GNN + Neighborhood Attention (K=16, attn_dim=64)**
   - Proven best in node1-1-1-1-1 (F1=0.4846)
   - Pre-computed embedding buffer (no GNN forward per step)
   - Lightweight attention over top-16 PPI neighbors

2. **Fixed GenePriorBias**
   - bias_init_scale=0.3 (vs 1.0 in parent → 2.5× neutral preference instead of 25×)
   - bias_warmup_epochs=10 (vs 50 in parent → bias adapts during 90%+ of training)
   - Fixes the catastrophic cold-start failure of parent (node1-1-3, F1=0.4093)
   - Validated by sibling node1-1-3-1 which escaped cold-start with same fix

3. **Proven optimal hyperparameters (matching node1-1-1-1-1)**
   - T_max=100 (vs 180 in parent → faster LR decay = stronger regularization)
   - dropout=0.40 (vs 0.35 in parent → matches best node's configuration)
   - weight_decay=0.03 (vs 0.02 in parent → matches best node's configuration)
   - warmup_epochs=5 (vs 20 in parent → faster cold-start escape)

4. **EMA (Exponential Moving Average) for test evaluation** [NOVEL]
   - Epoch-level EMA with decay=0.95
   - Stored as model buffers → saved in checkpoints
   - Test uses EMA weights for stabilized predictions
   - Recommended by sibling node1-1-3-1's feedback

Key differences from parent (node1-1-3, failed with F1=0.4093):
  - warmup_epochs: 20 → 5 (escapes cold-start 4× faster)
  - bias_init_scale: 1.0 → 0.3 (reduces neutral preference from 25× to 2.5×)
  - bias_warmup_epochs: 50 → 10 (bias adapts after 10 epochs, not 50)
  - t_max: 180 → 100 (faster LR decay = better regularization)
  - dropout: 0.35 → 0.40 (matches proven best configuration)
  - weight_decay: 0.02 → 0.03 (matches proven best configuration)
  - min_epochs: 55 → 15 (only need to pass 10-epoch bias warmup)
  - EMA added for test stability (novel)

Key differences from sibling (node1-1-3-1, F1=0.4364):
  - backbone: FROZEN (not trainable) — frozen is proven superior for this small dataset
  - T_max: 100 (not 220) — correct cosine decay schedule from node1-1-1-1-1
  - dropout: 0.40 (not 0.35) — stronger regularization
  - weight_decay: 0.03 (not 0.01) — stronger regularization
  - EMA: yes (not in sibling) — novel improvement

Expected target: F1 > 0.490 (node1-1-1-1-1's 0.4846 + GenePriorBias +0.003-0.005 + EMA +0.001-0.004)
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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights; neutral class stays ≈ 1."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def load_string_gnn_mapping() -> Dict[str, int]:
    """Load STRING_GNN node_names.json → Ensembl-ID → node-index mapping."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    return {name: idx for idx, name in enumerate(node_names)}


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0,1,2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)    # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0)

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


def build_neighbor_table(
    edge_index:  torch.Tensor,
    edge_weight: torch.Tensor,
    n_nodes:     int,
    K:           int = 16,
) -> tuple:
    """Build top-K neighbor lookup table for every STRING node.

    For each source node, selects the K outgoing neighbors with the highest
    edge confidence score (STRING combined_score, normalized to [0, 1]).

    Uses a double stable sort:
      1. Sort edges by weight descending (so that within each source group,
         neighbors appear in confidence order).
      2. Stable sort by source ascending (to group edges per source node).
    This O(E log E) approach avoids Python loops over the 786K edges.

    Returns:
        neighbor_idx : [n_nodes, K] long tensor  (-1 = no neighbor at that slot)
        neighbor_wt  : [n_nodes, K] float tensor (edge confidence weights)
    """
    src = edge_index[0].cpu()
    dst = edge_index[1].cpu()
    wt  = edge_weight.cpu().float()

    # Step 1: sort by weight descending (secondary key)
    order1 = torch.argsort(-wt, stable=True)
    src1 = src[order1]
    dst1 = dst[order1]
    wt1  = wt[order1]

    # Step 2: stable sort by source ascending (primary key)
    order2 = torch.argsort(src1, stable=True)
    src_sorted = src1[order2]
    dst_sorted = dst1[order2]
    wt_sorted  = wt1[order2]

    neighbor_idx = torch.full((n_nodes, K), -1,  dtype=torch.long)
    neighbor_wt  = torch.zeros(n_nodes, K,       dtype=torch.float32)

    if len(src_sorted) == 0:
        return neighbor_idx, neighbor_wt

    # Find boundaries between consecutive source nodes
    change_mask = torch.cat([
        torch.tensor([True]),
        src_sorted[1:] != src_sorted[:-1],
    ])
    start_positions = change_mask.nonzero(as_tuple=True)[0]  # [n_src_with_edges]
    src_at_starts   = src_sorted[start_positions]
    num_groups      = len(start_positions)

    for i in range(num_groups):
        node  = int(src_at_starts[i].item())
        start = int(start_positions[i].item())
        end   = int(start_positions[i + 1].item()) if i + 1 < num_groups else len(src_sorted)
        cnt   = min(end - start, K)
        if cnt > 0:
            neighbor_idx[node, :cnt] = dst_sorted[start:start + cnt]
            neighbor_wt[node,  :cnt] = wt_sorted[start:start + cnt]

    return neighbor_idx, neighbor_wt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    """K562 DEG prediction dataset."""

    def __init__(self, df: pd.DataFrame, string_map: Dict[str, int]) -> None:
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
            item["labels"] = self.labels[idx]  # [G] in {0,1,2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx":      torch.tensor([b["sample_idx"]     for b in batch], dtype=torch.long),
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
        self.batch_size  = batch_size
        self.num_workers = num_workers
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
        # Sequential sampler for test to avoid replicated samples in DDP
        sampler = SequentialSampler(self.test_ds)
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
            sampler=sampler,
        )


# ---------------------------------------------------------------------------
# NeighborhoodAttention Module
# ---------------------------------------------------------------------------
class NeighborhoodAttention(nn.Module):
    """Lightweight attention over top-K PPI neighbors for perturbation embedding enrichment.

    For each perturbed gene, aggregates its top-K STRING PPI neighbors using
    learned attention with edge-confidence weighting, then gates the result
    with the center gene embedding via a sigmoid gate.

    Architecture mirrors node1-1-1-1-1 (proven +0.010 F1):
        - K=16 neighbors (configurable)
        - attn_dim=64 (compact attention projection)
        - Edge confidence incorporated as log(w) additive bias to attention scores
        - Center-context gating: enriched = center + sigmoid(W_gate(center||agg)) * agg
        - Pre-norm LayerNorm on output for stable gradients
    """

    def __init__(self, emb_dim: int = 256, attn_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.attn_dim = attn_dim

        # Query, Key, Value projections (no bias for cleaner dot-product attention)
        self.W_q = nn.Linear(emb_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(emb_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(emb_dim, emb_dim,  bias=False)

        # Gating: combine center embedding with attended context
        self.W_gate = nn.Linear(emb_dim * 2, emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(emb_dim)

        # Conservative initialization: start with small gate weights
        # This ensures gate ≈ 0 initially → enriched ≈ center (stable start)
        nn.init.normal_(self.W_gate.weight, std=0.01)
        nn.init.zeros_(self.W_gate.bias)

    def forward(
        self,
        query_emb:    torch.Tensor,  # [B, D]
        neighbor_emb: torch.Tensor,  # [B, K, D]
        neighbor_wt:  torch.Tensor,  # [B, K] edge confidence weights in [0, 1]
        valid_mask:   torch.Tensor,  # [B, K] bool: True = valid neighbor slot
    ) -> torch.Tensor:
        """Returns enriched embedding [B, D]."""
        # Project to attention space
        q = self.W_q(query_emb).unsqueeze(1)   # [B, 1, attn_dim]
        k = self.W_k(neighbor_emb)              # [B, K, attn_dim]
        v = self.W_v(neighbor_emb)              # [B, K, D]

        # Scaled dot-product attention scores
        scores = (q * k).sum(-1) / (self.attn_dim ** 0.5)          # [B, K]

        # Incorporate edge confidence: high-confidence edges get higher attention
        # log(w) is negative (since w ∈ [0,1]) — edges with w→1 get score boost of ~0
        # while low-confidence edges (w→0) get penalized with large negative values
        scores = scores + neighbor_wt.clamp(min=1e-6).log()        # [B, K]

        # Mask out invalid neighbor slots (no neighbor at that position)
        scores = scores.masked_fill(~valid_mask, float('-inf'))     # [B, K]
        attn_w = torch.softmax(scores, dim=-1)                      # [B, K]

        # Safe NaN handling: if ALL neighbors are invalid, softmax yields NaN
        attn_w = torch.nan_to_num(attn_w, nan=0.0)
        attn_w = self.dropout(attn_w)                               # [B, K]

        # Weighted aggregation of value embeddings
        aggregated = (attn_w.unsqueeze(-1) * v).sum(1)             # [B, D]

        # Gated combination: center + gate * attended_context
        gate     = torch.sigmoid(
            self.W_gate(torch.cat([query_emb, aggregated], dim=-1))
        )                                                            # [B, D]
        enriched = query_emb + gate * aggregated                    # [B, D]

        return self.norm(enriched)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class StringGNNNeighborBiasEMAModel(pl.LightningModule):
    """Frozen STRING_GNN + Neighborhood Attention + Fixed GenePriorBias + EMA.

    Architecture:
        1. Pre-computed frozen STRING_GNN embeddings [18870, 256] — loaded once at setup
        2. Neighborhood attention (K=16, attn_dim=64) → enriched [B, 256]
           Aggregates top-16 PPI neighbors per perturbed gene
        3. Simple 2-layer MLP: LN → Linear(256→256) → GELU → Dropout → LN → Linear(256→256) → GELU → Dropout
        4. Scaled bilinear output: logits[b,c,g] = h[b] · gene_class_emb[c,g] / sqrt(bilinear_dim)
                                    + gene_prior_bias[c,g]

    Key fixes vs parent (node1-1-3):
        - bias_init_scale=0.3 (not 1.0): reduces neutral preference from 25× to 2.5×
        - bias_warmup_epochs=10 (not 50): bias adapts after 10 epochs, not 50
        - warmup_epochs=5 (not 20): model escapes cold-start 4× faster
        - T_max=100 (not 180): faster LR decay = stronger regularization
        - dropout=0.40 (not 0.35): stronger regularization
        - weight_decay=0.03 (not 0.02): stronger regularization

    Novel addition vs all prior nodes:
        - EMA (epoch-level, decay=0.95) stored as model buffers for test stability
    """

    def __init__(
        self,
        hidden_dim:         int   = 256,
        bilinear_dim:       int   = 256,
        dropout:            float = 0.40,
        lr:                 float = 3e-4,
        weight_decay:       float = 0.03,
        label_smoothing:    float = 0.05,
        warmup_epochs:      int   = 5,
        t_max:              int   = 100,
        bias_warmup_epochs: int   = 10,
        bias_init_scale:    float = 0.3,
        K:                  int   = 16,
        attn_dim:           int   = 64,
        ema_decay:          float = 0.95,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # All sub-modules initialized in setup()
        self._ema_param_map: Dict[str, str] = {}   # param_name → ema_buffer_name
        self._using_ema = False                     # True during test
        self._test_backup: Dict[str, torch.Tensor] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ── Load STRING_GNN: rank 0 first, then all ranks ───────────────────
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone.eval()

        # ── Load graph data (CPU only — not stored as buffer) ───────────────
        graph_data  = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph_data["edge_index"].long()
        edge_weight = graph_data["edge_weight"].float()

        # ── Pre-compute frozen STRING_GNN embeddings ─────────────────────────
        # One forward pass on CPU; result stored as non-trainable buffer.
        # Memory: [18870, 256] × 4 bytes ≈ 19 MB per GPU — negligible on H100.
        with torch.no_grad():
            gnn_out  = backbone(edge_index=edge_index, edge_weight=edge_weight)
            node_emb = gnn_out.last_hidden_state.float().cpu()  # [18870, 256]

        self.register_buffer("node_embeddings", node_emb)   # [18870, 256]

        # ── Build top-K neighbor lookup table ────────────────────────────────
        # Memory: [18870, 16] × (8+4) bytes ≈ 3.6 MB per GPU.
        neighbor_idx, neighbor_wt = build_neighbor_table(
            edge_index, edge_weight, n_nodes=node_emb.shape[0], K=hp.K
        )
        self.register_buffer("neighbor_idx", neighbor_idx)  # [18870, K]
        self.register_buffer("neighbor_wt",  neighbor_wt)   # [18870, K]

        # Backbone object no longer needed — GC will reclaim it
        del backbone, graph_data, edge_index, edge_weight, gnn_out

        # ── Fallback embedding for pert_ids not in STRING (~6.4%) ────────────
        self.fallback_emb = nn.Embedding(1, 256)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ── Neighborhood attention (K=16, attn_dim=64) ──────────────────────
        # Exact configuration proven in node1-1-1-1-1 (F1=0.4846)
        self.nbr_attn = NeighborhoodAttention(
            emb_dim=256, attn_dim=hp.attn_dim, dropout=0.1
        )

        # ── 2-layer MLP projection: 256 → hidden_dim → bilinear_dim ─────────
        # Simple architecture (no ResBlocks) proven best in node1-1-1-1 lineage.
        # dropout=0.40 matches node1-1-1-1-1's proven best configuration.
        STRING_DIM = 256
        self.mlp = nn.Sequential(
            nn.LayerNorm(STRING_DIM),
            nn.Linear(STRING_DIM,   hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
            nn.LayerNorm(hp.hidden_dim),
            nn.Linear(hp.hidden_dim, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # ── Bilinear gene-class embedding matrix [C, G, bilinear_dim] ────────
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # ── GenePriorBias [C, G] ──────────────────────────────────────────────
        # Learnable per-gene-class bias capturing gene-specific DEG propensities.
        # KEY FIX: Initialize at 0.3× log_freq (not 1.0× like parent).
        # 0.3× gives ~2.5× neutral preference (vs 25× with 1.0×), avoiding cold-start.
        # Gradients zeroed only for first bias_warmup_epochs=10 (not 50) epochs.
        log_freq  = torch.tensor(CLASS_FREQ).log()   # [-3.15, -0.08, -3.44]
        init_bias = (hp.bias_init_scale * log_freq).unsqueeze(1).expand(N_CLASSES, N_GENES).clone()
        self.gene_prior_bias = nn.Parameter(init_bias)  # [3, G]

        self.register_buffer("class_weights", get_class_weights())

        # ── Cast all trainable params to float32 for stable optimization ─────
        for _, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── EMA buffers: one shadow buffer per trainable parameter ───────────
        # Stored as model buffers → automatically saved in checkpoints.
        # Updated once per training epoch with decay=ema_decay.
        # Used for test evaluation (swapped in on_test_epoch_start).
        # EMA at 'best' checkpoint reflects averaged weights up to that epoch.
        self._ema_param_map = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                # Replace '.' with '__' for valid buffer names
                buf_name = 'ema__' + name.replace('.', '__')
                self.register_buffer(buf_name, param.data.float().clone())
                self._ema_param_map[name] = buf_name

        # Accumulators for val/test
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []
        self._test_pert_ids: List[List[str]] = []   # pert_id strings for DDP-safe deduplication

    # ── Forward helpers ───────────────────────────────────────────────────────

    def _get_pert_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Get neighborhood-attended perturbation embeddings for a batch.

        For known pert_ids (in STRING):
            1. Look up frozen pre-computed embedding from buffer [B, 256]
            2. Retrieve top-K neighbor embeddings from buffer [B, K, 256]
            3. Apply NeighborhoodAttention → enriched [B, 256]
        For unknown pert_ids (string_node_idx == -1):
            Skip attention; use the base fallback embedding directly.

        Args:
            string_node_idx: [B] long tensor, -1 for unknowns
        Returns:
            [B, 256] float perturbation embeddings
        """
        B      = string_node_idx.shape[0]
        K      = int(self.hparams.K)
        device = self.node_embeddings.device

        known_mask   = string_node_idx >= 0   # [B]
        unknown_mask = ~known_mask

        # ── Base embeddings [B, 256] ──────────────────────────────────────────
        base_emb = torch.zeros(B, 256, dtype=torch.float32, device=device)

        if known_mask.any():
            known_idx           = string_node_idx[known_mask]
            base_emb[known_mask] = self.node_embeddings[known_idx]

        if unknown_mask.any():
            n_unk = int(unknown_mask.sum().item())
            base_emb[unknown_mask] = self.fallback_emb(
                torch.zeros(n_unk, dtype=torch.long, device=device)
            ).float()

        # ── Neighbor embeddings [B, K, 256] ──────────────────────────────────
        nbr_idx = torch.full((B, K), -1, dtype=torch.long,    device=device)
        nbr_wt  = torch.zeros(B, K,      dtype=torch.float32, device=device)

        if known_mask.any():
            known_node_idx      = string_node_idx[known_mask]
            nbr_idx[known_mask] = self.neighbor_idx[known_node_idx]
            nbr_wt[known_mask]  = self.neighbor_wt[known_node_idx]

        # Valid mask: neighbor slot has a real neighbor (index ≥ 0)
        valid_mask   = nbr_idx >= 0                        # [B, K]
        safe_nbr_idx = nbr_idx.clamp(min=0)               # safe for indexing
        nbr_emb      = self.node_embeddings[safe_nbr_idx]  # [B, K, 256]

        # ── Neighborhood attention for all samples ────────────────────────────
        # Attention is cheap (~230K params) even for unknown pert_ids, and
        # importantly it PRESERVES the gradient graph to fallback_emb. Without
        # this (i.e. skipping attention for unknown), DDP detects fallback_emb
        # as unused and raises an error when find_unused_parameters=False.
        enriched = self.nbr_attn(
            query_emb    = base_emb,
            neighbor_emb = nbr_emb,
            neighbor_wt  = nbr_wt,
            valid_mask   = valid_mask,
        )   # [B, 256] — gradient graph to fallback_emb is preserved

        return enriched.float()

    def forward(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        pert_emb = self._get_pert_embeddings(string_node_idx)       # [B, 256]
        h        = self.mlp(pert_emb)                                # [B, bilinear_dim]

        # Scaled bilinear interaction (sqrt scaling prevents logit magnitude blow-up)
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)  # [B, 3, G]
        logits = logits / (self.hparams.bilinear_dim ** 0.5)

        # Add per-gene-class prior bias (learnable gene-specific DEG tendencies)
        logits = logits + self.gene_prior_bias.unsqueeze(0)            # [B, 3, G]

        return logits

    # ── GenePriorBias gradient warmup ─────────────────────────────────────────

    def on_after_backward(self) -> None:
        """Zero GenePriorBias gradient during the warmup phase.

        For the first bias_warmup_epochs epochs (default: 10), the gene_prior_bias
        tensor is used in the forward pass (providing useful class-frequency prior signal)
        but its gradient is zeroed before the optimizer step, preventing early
        corrupting updates while the bilinear head is still converging.

        KEY FIX: bias_warmup_epochs=10 (not 50 like parent) means the bias adapts
        after 10 epochs, allowing the model to learn gene-specific DEG tendencies
        during 90%+ of training (vs only 9% in the parent).
        """
        if self.current_epoch < self.hparams.bias_warmup_epochs:
            if self.gene_prior_bias.grad is not None:
                self.gene_prior_bias.grad.zero_()

    # ── EMA update ────────────────────────────────────────────────────────────

    def on_train_epoch_end(self) -> None:
        """Update EMA weights at end of each training epoch.

        EMA is updated epoch-by-epoch with decay=ema_decay (default: 0.95).
        With decay=0.95, the effective averaging window is ~1/(1-0.95) = 20 epochs,
        capturing the stable converging trajectory without being dominated by
        early poor-quality weights.

        EMA buffers are part of the model's state → automatically saved in checkpoints.
        When 'best' checkpoint is loaded for test, EMA at that epoch is available.
        """
        if not self._ema_param_map:
            return
        d = float(self.hparams.ema_decay)
        for name, param in self.named_parameters():
            if param.requires_grad and name in self._ema_param_map:
                buf = getattr(self, self._ema_param_map[name])
                buf.data.mul_(d).add_(param.data.float(), alpha=1.0 - d)

    # ── EMA swap for test evaluation ─────────────────────────────────────────

    def on_test_epoch_start(self) -> None:
        """Swap model weights to EMA weights before test evaluation.

        The 'best' checkpoint (loaded by ckpt_path='best') contains EMA buffers
        at the best validation epoch. Swapping to EMA weights provides stabilized
        predictions that average over the model's trajectory, reducing the impact
        of late-training noise/overfitting.

        Original weights are backed up and restored after test.
        """
        if not self._ema_param_map:
            return
        self._test_backup = {}
        for name, param in self.named_parameters():
            if param.requires_grad and name in self._ema_param_map:
                self._test_backup[name] = param.data.clone()
                ema_val = getattr(self, self._ema_param_map[name])
                param.data.copy_(ema_val.to(param.device))
        self._using_ema = True

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),   # [B*G, 3]
            targets.reshape(-1),                        # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ── Lightning steps ───────────────────────────────────────────────────────

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["string_node_idx"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"])
        if "labels" in batch:
            loss  = self._loss(logits, batch["labels"])
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
        self._val_preds.clear()
        self._val_tgts.clear()
        self._val_idx.clear()

        # Gather across all DDP ranks
        all_preds = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)    # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)     # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        # De-duplicate (DDP padding may replicate samples at boundaries)
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
        # Collect pert_id strings for DDP-safe deduplication
        self._test_pert_ids.append(batch["pert_id"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = torch.cat(self._test_preds, dim=0)
        local_idx   = torch.cat(self._test_idx,   dim=0)
        all_preds   = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_idx     = self.all_gather(local_idx)     # [W, N_local]

        # Gather pert_id strings across all ranks via torch.distributed.all_gather_object
        # (SequentialSampler in DDP means all ranks process same samples → dedup by pert_id)
        all_pert_ids: List[str] = []
        local_pert_ids_flat: List[str] = []
        for batch_pert_ids in self._test_pert_ids:
            local_pert_ids_flat.extend(batch_pert_ids)
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered: List[List[str]] = [None] * world_size
            torch.distributed.all_gather_object(gathered, local_pert_ids_flat)
            for rank_list in gathered:
                all_pert_ids.extend(rank_list)
        else:
            all_pert_ids = local_pert_ids_flat

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            # Deduplicate using global pert_id strings (not local sample_idx).
            # In DDP with SequentialSampler, all ranks process the same samples,
            # so local sample_idx values (0,1,2...) collide across ranks.
            # Using pert_id strings ensures each unique sample is kept once.
            assert len(all_pert_ids) == preds_flat.shape[0], \
                f"pert_id count {len(all_pert_ids)} != preds count {preds_flat.shape[0]}"

            seen: set = set()
            keep_mask: List[bool] = []
            for pid in all_pert_ids:
                if pid not in seen:
                    seen.add(pid)
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)
            keep_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=preds_flat.device)
            preds_dedup = preds_flat[keep_tensor]      # [N_test, 3, G]
            unique_pids = [p for p, k in zip(all_pert_ids, keep_mask) if k]

            # Load test metadata for symbol lookup
            test_df = pd.read_csv(TEST_TSV, sep="\t")
            pid_to_sym = dict(zip(test_df["pert_id"].tolist(), test_df["symbol"].tolist()))

            rows = []
            for i, pid in enumerate(unique_pids):
                sym = pid_to_sym.get(pid, "")
                pred_list = preds_dedup[i].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            ema_label = "(EMA weights)" if self._using_ema else "(regular weights)"
            print(f"[Node1-1-3-2] Saved {len(rows)} test predictions {ema_label}.")

        self._test_preds.clear()
        self._test_idx.clear()
        self._test_pert_ids.clear()

        # Restore original weights after test (clean up EMA swap)
        if self._using_ema and self._test_backup:
            for name, param in self.named_parameters():
                if name in self._test_backup:
                    param.data.copy_(self._test_backup[name])
            self._test_backup.clear()
            self._using_ema = False

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save trainable parameters + persistent buffers (includes EMA buffers).

        EMA buffers are included as regular buffers, so they're saved in checkpoints.
        When 'best' checkpoint is loaded at test time, EMA weights at that epoch
        are available for test evaluation.
        """
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

    # ── Optimizer & scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        hp = self.hparams

        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )

        # Linear warmup for warmup_epochs epochs: LR goes from 0.1×lr → lr
        # KEY FIX: warmup_epochs=5 (not 20) → model reaches full LR by epoch 5,
        # escaping cold-start 4× faster than parent
        warmup_sch = LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=hp.warmup_epochs
        )
        # Cosine annealing: T_max=100 (not 180) → LR decays faster
        # At epoch 50: LR ≈ 3e-4 × (1 + cos(π × 50/100))/2 = 1.5e-4 (50% decayed)
        # This matches node1-1-1-1-1's proven configuration (F1=0.4846)
        cosine_sch = CosineAnnealingLR(opt, T_max=hp.t_max, eta_min=1e-6)

        # Sequential: warmup first, then cosine
        sch = SequentialLR(
            opt,
            schedulers=[warmup_sch, cosine_sch],
            milestones=[hp.warmup_epochs],
        )
        return {
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-1-3-2 – Frozen STRING_GNN + Neighborhood Attention + Fixed GenePriorBias + EMA"
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=32)
    parser.add_argument("--global-batch-size",   type=int,   default=256)
    parser.add_argument("--max-epochs",          type=int,   default=300)
    parser.add_argument("--lr",                  type=float, default=3e-4)
    parser.add_argument("--weight-decay",        type=float, default=0.03)
    parser.add_argument("--hidden-dim",          type=int,   default=256)
    parser.add_argument("--bilinear-dim",        type=int,   default=256)
    parser.add_argument("--dropout",             type=float, default=0.40)
    parser.add_argument("--label-smoothing",     type=float, default=0.05)
    parser.add_argument("--warmup-epochs",       type=int,   default=5)
    parser.add_argument("--t-max",               type=int,   default=100)
    parser.add_argument("--bias-warmup-epochs",  type=int,   default=10,
                        dest="bias_warmup_epochs")
    parser.add_argument("--bias-init-scale",     type=float, default=0.3,
                        dest="bias_init_scale")
    parser.add_argument("--K",                   type=int,   default=16)
    parser.add_argument("--attn-dim",            type=int,   default=64)
    parser.add_argument("--ema-decay",           type=float, default=0.95,
                        dest="ema_decay")
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

    # Debug / fast-dev-run limits
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
    model = StringGNNNeighborBiasEMAModel(
        hidden_dim         = args.hidden_dim,
        bilinear_dim       = args.bilinear_dim,
        dropout            = args.dropout,
        lr                 = args.lr,
        weight_decay       = args.weight_decay,
        label_smoothing    = args.label_smoothing,
        warmup_epochs      = args.warmup_epochs,
        t_max              = args.t_max,
        bias_warmup_epochs = args.bias_warmup_epochs,
        bias_init_scale    = args.bias_init_scale,
        K                  = args.K,
        attn_dim           = args.attn_dim,
        ema_decay          = args.ema_decay,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-val_f1={val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
        auto_insert_metric_name = False,
    )
    # patience=20 with min_epochs=15 ensures training:
    #   1. Passes bias_warmup_epochs=10 (bias adapts before early stopping can fire)
    #   2. Gives neighborhood attention enough epochs to learn PPI patterns
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=20, min_delta=1e-4)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # DDP strategy for multi-GPU
    # find_unused_parameters=True: gene_prior_bias gradients are zeroed during bias warmup
    # via on_after_backward, which DDP may detect as "unused". Also EMA shadow buffers
    # shadow trainable params. Safe to keep True — negligible overhead on H100.
    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = n_gpus,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        min_epochs              = 15,      # Ensure training runs past bias warmup (10 epochs)
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

    # Save test score summary
    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1-1-3-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
