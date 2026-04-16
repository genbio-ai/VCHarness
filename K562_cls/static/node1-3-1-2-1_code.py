"""Node 1-3-1-2-1: STRING_GNN-only + GenePriorBias + Gene-Diversity-Weighted Loss + Fixed SWA + Discriminative LR.

Strategy: The parent node (node1-3-1-2, F1=0.4837) contained a critical SWA timing bug —
EarlyStopping (patience=35, best epoch=147) terminated training at epoch ~183, well before
the intended swa_start_epoch=230. SWA never activated; the 0.4837 result was achieved via
best checkpoint only, not SWA. This node fixes SWA timing and adds discriminative LR:

1. Fix SWA activation: swa_start_epoch=150 (just after expected peak at ~147),
   swa_interval=10 (every 10 epochs), patience=100 (allows ~10 SWA checkpoints near peak).
   This directly captures node4-2-1-1-1's +0.003 F1 gain from late-epoch SWA.
2. Discriminative LR: gene_class_emb + gene_prior_bias at lr=2e-4 (large embedding matrix,
   benefits from slightly slower learning), neighborhood_attn + head_proj + fallback_emb at
   lr=3e-4. node1-1-1-1-1 (F1=0.4846) used discriminative LR successfully in STRING-only lineage.
3. Slightly higher gene_diversity_factor=5.0 (from 4.0): stronger emphasis on rarely-DEG genes.
4. Extended schedule: t_max=350, max_epochs=400, patience=100 to support longer training.
5. All other proven components retained: STRING_GNN frozen, K=16 NeighborhoodAttention,
   two-stage GenePriorBias (bias_warmup=50, scale=0.5), label_smoothing=0.05, weight_decay=3e-2,
   dropout=0.35, min_lr_ratio=0.05.

Memory connections:
- node1-3-1-2 (F1=0.4837, parent): SWA never activated — root cause of underperformance
- node4-2-1-1-1 (F1=0.4868): SWA from epoch 220 with 8 checkpoints added +0.0032 F1
- node4-2-1-1-2-1 (F1=0.4852): SWA with 32 checkpoints (too many) degraded; 8 optimal
- node4-2-1-2 (F1=0.4893): best in tree — two-stage bias + gene-diversity loss + SWA zero gap
- node1-1-1-1-1 (F1=0.4846): discriminative LR proven beneficial in STRING-only lineage
- node1-2-2-1 (F1=0.4829): STRING-only + GenePriorBias ceiling without gene-diversity loss
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

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights; neutral class stays approximately 1."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  -- softmax probabilities
        targets: [N, G]    long   -- class labels in {0,1,2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)          # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)          # [N, G]
        is_pred = (y_hat == c)            # [N, G]
        present = is_true.any(dim=0)      # [G]

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


def compute_gene_diversity_weights(
    train_tsv: Path,
    n_genes: int = N_GENES,
    diversity_factor: float = 5.0,
) -> torch.Tensor:
    """Compute per-gene loss weights based on inverse DEG frequency.

    Genes that are rarely differentially expressed receive higher loss weight
    to push the model to attend to rare DEG signals, combating neutral-class dominance.

    Returns:
        weights [n_genes] float tensor, mean=1.0
    """
    df = pd.read_csv(train_tsv, sep="\t")
    deg_counts = np.zeros(n_genes, dtype=np.float32)
    total = len(df)

    for row in df["label"].tolist():
        labels = json.loads(row)
        for g, lbl in enumerate(labels):
            if lbl != 0:   # non-neutral = DEG
                deg_counts[g] += 1

    # Inverse log-frequency weighting
    log_count = np.log1p(deg_counts)
    max_log = log_count.max()
    if max_log > 0:
        # Weight = 1 + diversity_factor * (1 - normalized_log_count)
        # Rarely-DEG genes → weight close to 1 + diversity_factor
        # Frequently-DEG genes → weight close to 1.0
        weights = 1.0 + diversity_factor * (1.0 - log_count / max_log)
    else:
        weights = np.ones(n_genes, dtype=np.float32)

    # Normalize to mean=1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Pre-computation utilities
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_string_gnn_embeddings() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load STRING_GNN and compute all node embeddings. Returns (emb[N,256], pert_id→idx)."""
    import json as _json

    model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
    model.eval()
    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
    node_names = _json.loads((STRING_GNN_DIR / "node_names.json").read_text())

    edge_index = graph["edge_index"]
    ew = graph.get("edge_weight", None)

    outputs = model(edge_index=edge_index, edge_weight=ew)
    emb = outputs.last_hidden_state.float().cpu()   # [18870, 256]

    pert_to_idx = {name: i for i, name in enumerate(node_names)}
    del model
    return emb, pert_to_idx


@torch.no_grad()
def precompute_neighborhood(
    emb: torch.Tensor,
    K: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute top-K neighbor indices and normalized edge weights.

    Returns:
        neighbor_indices [N, K] long -- STRING_GNN node indices of top-K neighbors
        neighbor_weights [N, K] float -- normalized STRING confidence weights
    """
    N = emb.shape[0]
    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
    edge_index = graph["edge_index"]   # [2, E]
    ew = graph.get("edge_weight", None)

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    weights = ew.tolist() if ew is not None else [1.0] * len(src)

    adj: Dict[int, List[Tuple[int, float]]] = {}
    for s, d, w in zip(src, dst, weights):
        adj.setdefault(s, []).append((d, w))

    neighbor_indices = torch.full((N, K), -1, dtype=torch.long)
    neighbor_weights = torch.zeros(N, K, dtype=torch.float32)

    for node in range(N):
        nbrs = adj.get(node, [])
        if not nbrs:
            continue
        nbrs_sorted = sorted(nbrs, key=lambda x: -x[1])[:K]
        for j, (nb_idx, nb_w) in enumerate(nbrs_sorted):
            neighbor_indices[node, j] = nb_idx
            neighbor_weights[node, j] = nb_w

    # Normalize weights per node (softmax over valid neighbors)
    mask = neighbor_indices >= 0    # [N, K]
    raw = neighbor_weights.clone()
    raw[~mask] = -1e9
    norm_w = torch.softmax(raw, dim=-1)   # [N, K]
    norm_w[~mask] = 0.0

    return neighbor_indices, norm_w


# ---------------------------------------------------------------------------
# Neighborhood Attention Aggregator (proven design from node1-2 / node1-1-1-1-1)
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Aggregate top-K PPI neighbors for a center gene using learned attention."""

    def __init__(self, emb_dim: int = 256, attn_dim: int = 64) -> None:
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False),
        )
        self.gate_proj = nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        center_emb: torch.Tensor,       # [B, D]
        neighbor_emb: torch.Tensor,     # [B, K, D]
        neighbor_weights: torch.Tensor, # [B, K]  pre-normalized edge weights
        valid_mask: torch.Tensor,       # [B, K]  bool, True = valid neighbor
    ) -> torch.Tensor:
        """Returns aggregated representation [B, D]."""
        B, K, D = neighbor_emb.shape
        center_exp = center_emb.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]
        pair = torch.cat([center_exp, neighbor_emb], dim=-1)     # [B, K, 2D]
        attn_scores = self.attn_proj(pair).squeeze(-1)           # [B, K]

        # Combine learned scores with STRING confidence as prior
        attn_scores = attn_scores + neighbor_weights

        # Mask invalid neighbors
        attn_scores = attn_scores.masked_fill(~valid_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)        # [B, K]
        attn_weights = attn_weights * valid_mask.float()         # zero-out invalid

        # Weighted aggregation
        aggregated = (attn_weights.unsqueeze(-1) * neighbor_emb).sum(dim=1)  # [B, D]

        # Gated residual: center + gate * aggregated
        gate = torch.sigmoid(self.gate_proj(center_emb))         # [B, D]
        return center_emb + gate * aggregated                     # [B, D]


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
        "pert_id":    [b["pert_id"]  for b in batch],
        "symbol":     [b["symbol"]   for b in batch],
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
class StringGNNDEGModel(pl.LightningModule):
    """Frozen STRING_GNN + Neighborhood Attention + two-stage GenePriorBias + bilinear head.

    Key improvements over parent node1-3-1-2 (F1=0.4837):
    1. Fixed SWA timing: swa_start_epoch=150 (parent had 230, never activated due to
       EarlyStopping at epoch 183). SWA starts just after expected peak (~epoch 147).
    2. SWA interval=10 epochs: every 10 epochs → ~9-10 checkpoints in patience window.
       (proven: 8 checkpoints = optimal; 32 = degraded — node4-2-1-1-1/node4-2-1-1-2-1)
    3. Discriminative LR: gene_class_emb + gene_prior_bias at lr_emb (2e-4, slower for
       large 5.1M-parameter matrix), neighborhood_attn + head_proj + fallback_emb at lr (3e-4).
       Proven beneficial in node1-1-1-1-1 (F1=0.4846) for STRING-only architecture.
    4. gene_diversity_factor=5.0 (up from 4.0): stronger emphasis on rare-DEG genes.
    5. patience=100: ensures training reaches SWA window before EarlyStopping triggers.
    6. All other proven components unchanged: frozen STRING_GNN, K=16, bias_warmup=50,
       label_smoothing=0.05, weight_decay=3e-2, dropout=0.35, min_lr_ratio=0.05.
    """

    def __init__(
        self,
        bilinear_dim: int = 256,
        attn_dim: int = 64,
        K: int = 16,
        dropout: float = 0.35,
        lr: float = 3e-4,
        lr_emb: float = 2e-4,
        weight_decay: float = 3e-2,
        warmup_epochs: int = 20,
        t_max: int = 350,
        min_lr_ratio: float = 0.05,
        label_smoothing: float = 0.05,
        bias_warmup_epochs: int = 50,
        bias_scale_init: float = 0.5,
        gene_diversity_factor: float = 5.0,
        swa_start_epoch: int = 150,
        swa_interval: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # SWA state (stored in CPU memory; keys = parameter names)
        self._swa_params: Optional[Dict[str, torch.Tensor]] = None
        self._swa_n_averaged: int = 0
        self._swa_applied: bool = False

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Collect all pert_ids across splits ----
        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        all_pert_ids = (
            train_df["pert_id"].tolist() +
            val_df["pert_id"].tolist() +
            test_df["pert_id"].tolist()
        )

        # ---- STRING_GNN: precompute embeddings + neighborhood ----
        self.print("Precomputing STRING_GNN embeddings...")
        string_emb, pert_to_gnn_idx = precompute_string_gnn_embeddings()

        self.register_buffer("node_embeddings", string_emb)    # [18870, 256]

        # Build pert_id -> STRING_GNN node index mapping
        unique_sorted = sorted(set(all_pert_ids))
        self.pert_to_pos = {pid: i for i, pid in enumerate(unique_sorted)}

        gnn_idx_tensor = torch.tensor(
            [pert_to_gnn_idx.get(pid, -1) for pid in unique_sorted], dtype=torch.long
        )
        self.register_buffer("pert_gnn_idx", gnn_idx_tensor)   # [M]

        self.print("Precomputing PPI neighborhood tables (K={})...".format(hp.K))
        nb_indices, nb_weights = precompute_neighborhood(string_emb, K=hp.K)
        self.register_buffer("neighbor_indices", nb_indices)   # [18870, K]
        self.register_buffer("neighbor_weights", nb_weights)   # [18870, K]

        # Fallback embedding for pert_ids not in STRING
        self.fallback_emb = nn.Parameter(torch.zeros(1, 256))

        # ---- Trainable modules (Group 1: neighborhood + head) ----
        self.neighborhood_attn = NeighborhoodAttentionAggregator(
            emb_dim=256, attn_dim=hp.attn_dim
        )

        # Projection from STRING_GNN 256-dim to bilinear_dim
        self.head_proj = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # ---- Trainable modules (Group 2: gene embedding + bias) ----
        # Bilinear gene-class embedding: logits[b,c,g] = h[b] · gene_class_emb[c,g]
        # This large 5.1M param matrix benefits from slightly lower LR for stable convergence
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # GenePriorBias: per-gene per-class additive bias [3, 6640]
        # Initialized from log-class-frequencies at reduced scale (0.5x) to avoid disrupting
        # the early backbone training phase. Zero-meaned per gene for stable initialization.
        init_bias = torch.zeros(N_CLASSES, N_GENES, dtype=torch.float32)
        for c, freq in enumerate(CLASS_FREQ):
            init_bias[c, :] = float(np.log(freq + 1e-9))
        init_bias = init_bias - init_bias.mean(dim=0, keepdim=True)
        self.gene_prior_bias = nn.Parameter(init_bias * hp.bias_scale_init)

        # Register bias warmup state as a non-persistent buffer (not saved in checkpoint)
        # CRITICAL: persistent=False ensures it's re-computed in setup(), not loaded from ckpt
        self.register_buffer("bias_active", torch.zeros(1, dtype=torch.bool), persistent=False)

        # Class weights for weighted CE loss
        self.register_buffer("class_weights", get_class_weights())

        # Gene diversity weights [N_GENES] for per-gene loss scaling
        self.print("Computing gene diversity weights (factor={})...".format(hp.gene_diversity_factor))
        gene_div_weights = compute_gene_diversity_weights(
            TRAIN_TSV, N_GENES, hp.gene_diversity_factor
        )
        self.register_buffer("gene_div_weights", gene_div_weights)   # [N_GENES], mean=1.0

        # Cast trainable parameters to float32 for stable optimization
        for k, v in self.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()

        # Register gradient hook for GenePriorBias warmup phase.
        # During warmup (bias_active=False), zero out bias gradients so the backbone
        # and head learn freely without bias interference.
        # The hook is a closure capturing `self` so it can read bias_active.
        def _bias_grad_hook(grad: torch.Tensor) -> torch.Tensor:
            if not self.bias_active.item():
                return torch.zeros_like(grad)
            return grad

        self._bias_grad_hook_handle = self.gene_prior_bias.register_hook(_bias_grad_hook)

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_tgts: List[torch.Tensor]  = []
        self._test_idx:  List[torch.Tensor]  = []

    def _get_neighborhood_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Return [B, 256] neighborhood-attention-aggregated embeddings."""
        pos = torch.tensor(
            [self.pert_to_pos[pid] for pid in pert_ids], dtype=torch.long, device=self.device
        )
        gnn_node_idx = self.pert_gnn_idx[pos]   # [B]
        valid_center = gnn_node_idx >= 0
        safe_center_idx = gnn_node_idx.clamp(min=0)
        center_emb_raw = self.node_embeddings[safe_center_idx]  # [B, 256]

        fallback = self.fallback_emb.expand(center_emb_raw.shape[0], -1).to(center_emb_raw.dtype)
        center_emb = torch.where(valid_center.unsqueeze(-1), center_emb_raw, fallback).float()

        K = self.hparams.K
        nb_idx = self.neighbor_indices[safe_center_idx]   # [B, K]
        nb_wts = self.neighbor_weights[safe_center_idx]   # [B, K]
        valid_mask = nb_idx >= 0                           # [B, K] bool

        safe_nb_idx = nb_idx.clamp(min=0)
        nb_emb = self.node_embeddings[safe_nb_idx].float()  # [B, K, 256]
        nb_emb = nb_emb * valid_mask.unsqueeze(-1).float()

        aggregated = self.neighborhood_attn(
            center_emb, nb_emb, nb_wts, valid_mask
        )   # [B, 256]
        return aggregated

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        gnn_emb = self._get_neighborhood_emb(pert_ids)     # [B, 256]
        h = self.head_proj(gnn_emb)                         # [B, bilinear_dim]
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)   # [B, 3, G]

        # GenePriorBias is always added to logits; gradient is controlled via hook
        # This ensures the SWA state for bias is also included in final inference
        logits = logits + self.gene_prior_bias.unsqueeze(0)   # broadcast over B
        return logits

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Gene-diversity-weighted cross entropy loss with class weights + label smoothing."""
        B, C, G = logits.shape

        # Per-sample-per-gene CE loss (no reduction)
        ce_per_sample_gene = F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),   # [B*G, 3]
            targets.reshape(-1),                        # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
            reduction="none",
        ).reshape(B, G)  # [B, G]

        # Apply per-gene diversity weights: rare DEG genes get higher weight
        weighted = ce_per_sample_gene * self.gene_div_weights.unsqueeze(0)  # [B, G]
        return weighted.mean()

    def on_train_epoch_start(self) -> None:
        """Handle GenePriorBias warmup and SWA weight accumulation."""
        current_epoch = self.current_epoch

        # GenePriorBias warmup: activate after bias_warmup_epochs
        if current_epoch >= self.hparams.bias_warmup_epochs and not self.bias_active.item():
            self.bias_active.fill_(True)
            self.print(f"[Epoch {current_epoch}] GenePriorBias activated.")

        # SWA: accumulate model weights on ALL ranks every swa_interval epochs after
        # swa_start_epoch. The interval is measured from swa_start_epoch to ensure the
        # first checkpoint is captured exactly at swa_start_epoch.
        swa_offset = current_epoch - self.hparams.swa_start_epoch
        if swa_offset >= 0 and swa_offset % self.hparams.swa_interval == 0:
            self._swa_accumulate()

    def _swa_accumulate(self) -> None:
        """Accumulate running average of trainable parameters for SWA (all ranks)."""
        with torch.no_grad():
            if self._swa_params is None:
                # First SWA checkpoint
                self._swa_params = {
                    name: param.data.detach().float().cpu().clone()
                    for name, param in self.named_parameters()
                    if param.requires_grad
                }
                self._swa_n_averaged = 1
                self.print(
                    f"[SWA] Initialized SWA at epoch {self.current_epoch}. "
                    f"Tracking {len(self._swa_params)} param tensors."
                )
            else:
                # Running average update: E[n+1] = n/(n+1) * E[n] + 1/(n+1) * current
                n = self._swa_n_averaged
                for name, param in self.named_parameters():
                    if param.requires_grad and name in self._swa_params:
                        self._swa_params[name].mul_(n / (n + 1)).add_(
                            param.data.detach().float().cpu() / (n + 1)
                        )
                self._swa_n_averaged += 1
                if self.trainer.is_global_zero:
                    self.print(
                        f"[SWA] Updated SWA at epoch {self.current_epoch}. "
                        f"n_averaged={self._swa_n_averaged}"
                    )

    def apply_swa_weights(self) -> None:
        """Load SWA averaged weights into the model for test inference.

        This is called after trainer.fit() completes, before trainer.test().
        The SWA weights replace the current model parameters for final inference.
        """
        if self._swa_params is None or self._swa_n_averaged == 0:
            self.print("[SWA] No SWA state available. Using best checkpoint weights.")
            return

        self.print(
            f"[SWA] Applying SWA weights ({self._swa_n_averaged} checkpoints) "
            f"for test inference."
        )
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad and name in self._swa_params:
                    swa_weight = self._swa_params[name].to(
                        device=param.device, dtype=param.dtype
                    )
                    param.data.copy_(swa_weight)

        self._swa_applied = True
        # Ensure bias is active after SWA (warmup must have passed)
        self.bias_active.fill_(True)

    # ---- Steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["pert_id"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["pert_id"])
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
        local_preds = torch.cat(self._val_preds, dim=0)    # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  dim=0)    # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   dim=0)    # [N_local]
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
            self._test_tgts.append(batch["labels"].detach())
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)    # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)    # [N_local]
        # All ranks participate in all_gather (collective operation)
        all_preds   = self.all_gather(local_preds)           # [W, N_local, 3, G]
        all_idx     = self.all_gather(local_idx)             # [W, N_local]

        # Gather targets on all ranks too (collective)
        if self._test_tgts:
            local_tgts = torch.cat(self._test_tgts, dim=0)  # [N_local, G]
            all_tgts   = self.all_gather(local_tgts)          # [W, N_local, G]
        else:
            all_tgts = None

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            order = torch.argsort(idx_flat)
            s_idx = idx_flat[order]
            s_pred = preds_flat[order]
            mask = torch.cat([torch.ones(1, dtype=torch.bool, device=s_idx.device),
                              s_idx[1:] != s_idx[:-1]])
            preds_dedup = s_pred[mask]     # [N_test, 3, G]
            unique_sid  = s_idx[mask].tolist()

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                           for i in range(len(test_df))}

            rows = []
            for sid in unique_sid:
                pid, sym = idx_to_meta[int(sid)]
                dedup_pos = (s_idx == sid).nonzero(as_tuple=True)[0][0].item()
                pred_list = preds_dedup[dedup_pos].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-3-1-2-1] Saved {len(rows)} test predictions.")

            # Compute test F1 if targets are available
            if all_tgts is not None:
                tgts_flat  = all_tgts.view(-1, N_GENES)
                s_tgt = tgts_flat[order]
                tgts_dedup = s_tgt[mask]
                test_f1 = compute_per_gene_f1(preds_dedup, tgts_dedup)
                self.log("test/f1", test_f1, prog_bar=True, sync_dist=False)

        self._test_preds.clear()
        self._test_tgts.clear()
        self._test_idx.clear()

    # ---- Checkpoint helpers ----
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
            f"Checkpoint: {train}/{total} params ({100*train/total:.2f}%), "
            f"plus {bufs} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer with Discriminative LR ----
    def configure_optimizers(self):
        hp = self.hparams

        # Discriminative LR: two parameter groups
        # Group 1 (neighborhood + head + fallback): higher LR=3e-4
        #   - Smaller modules (~166K params): benefit from full LR for fast adaptation
        # Group 2 (gene_class_emb + gene_prior_bias): lower LR=lr_emb (2e-4)
        #   - Large embedding matrix (5.1M params): benefits from conservative updates
        #     for stable convergence; proven in node1-1-1-1-1 STRING-only discriminative LR
        group1_params = []
        group2_params = []
        group2_names = {"gene_class_emb", "gene_prior_bias"}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Match by parameter name (not module name)
            base_name = name.split(".")[-1] if "." in name else name
            param_name = name  # full name
            if any(g2 in param_name for g2 in group2_names):
                group2_params.append(param)
            else:
                group1_params.append(param)

        opt = torch.optim.AdamW(
            [
                {"params": group1_params, "lr": hp.lr,     "name": "attn_head"},
                {"params": group2_params, "lr": hp.lr_emb, "name": "gene_emb"},
            ],
            weight_decay=hp.weight_decay,
        )

        # WarmupCosine schedule (applied uniformly to both groups via multiplier)
        def lr_lambda(epoch: int) -> float:
            if epoch < hp.warmup_epochs:
                return (epoch + 1) / max(hp.warmup_epochs, 1)
            else:
                progress = (epoch - hp.warmup_epochs) / max(hp.t_max, 1)
                progress = min(progress, 1.0)
                cosine_val = 0.5 * (1.0 + np.cos(np.pi * progress))
                return hp.min_lr_ratio + (1.0 - hp.min_lr_ratio) * cosine_val

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
        description="Node1-3-1-2-1: STRING_GNN + NeighborhoodAttention + GenePriorBias + "
                    "GeneDiversityLoss + Fixed SWA (start=150, interval=10) + Discriminative LR"
    )
    parser.add_argument("--micro_batch_size",       type=int,   default=32)
    parser.add_argument("--global_batch_size",      type=int,   default=256)
    parser.add_argument("--max_epochs",             type=int,   default=400)
    parser.add_argument("--lr",                    type=float, default=3e-4,
                        help="LR for neighborhood attention + head projection modules")
    parser.add_argument("--lr_emb",                type=float, default=2e-4,
                        help="LR for gene_class_emb + gene_prior_bias (large embedding matrices)")
    parser.add_argument("--weight_decay",           type=float, default=3e-2)
    parser.add_argument("--bilinear_dim",           type=int,   default=256)
    parser.add_argument("--attn_dim",               type=int,   default=64)
    parser.add_argument("--k_neighbors",            type=int,   default=16)
    parser.add_argument("--dropout",               type=float, default=0.35)
    parser.add_argument("--warmup_epochs",          type=int,   default=20)
    parser.add_argument("--t_max",                  type=int,   default=350,
                        help="Cosine decay duration after warmup; extended to 350 for longer training")
    parser.add_argument("--min_lr_ratio",           type=float, default=0.05,
                        help="Cosine LR floor as fraction of peak LR (proven in node4-2-1-1-1)")
    parser.add_argument("--label_smoothing",         type=float, default=0.05)
    parser.add_argument("--bias_warmup_epochs",     type=int,   default=50,
                        help="Epochs before GenePriorBias gradients are unblocked")
    parser.add_argument("--bias_scale_init",        type=float, default=0.5,
                        help="Initial scale for GenePriorBias tensor (0.5 = half log-prior scale)")
    parser.add_argument("--gene_diversity_factor",   type=float, default=5.0,
                        help="Diversity factor for per-gene loss weighting (increased from 4.0)")
    parser.add_argument("--swa_start_epoch",        type=int,   default=150,
                        help="Epoch to start SWA accumulation (just after expected peak ~147)")
    parser.add_argument("--swa_interval",           type=int,   default=10,
                        help="Epochs between SWA checkpoints (every 10 epochs → ~9-10 checkpoints)")
    parser.add_argument("--patience",               type=int,   default=100,
                        help="EarlyStopping patience; 100 to ensure SWA activates before stopping")
    parser.add_argument("--val_check_interval",     type=float, default=1.0)
    parser.add_argument("--num_workers",            type=int,   default=4)
    parser.add_argument("--debug_max_step",         type=int,   default=None)
    parser.add_argument("--fast_dev_run",           action="store_true")
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

    # DataModule + Model
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    model = StringGNNDEGModel(
        bilinear_dim          = args.bilinear_dim,
        attn_dim              = args.attn_dim,
        K                     = args.k_neighbors,
        dropout               = args.dropout,
        lr                    = args.lr,
        lr_emb                = args.lr_emb,
        weight_decay          = args.weight_decay,
        warmup_epochs         = args.warmup_epochs,
        t_max                 = args.t_max,
        min_lr_ratio          = args.min_lr_ratio,
        label_smoothing       = args.label_smoothing,
        bias_warmup_epochs    = args.bias_warmup_epochs,
        bias_scale_init       = args.bias_scale_init,
        gene_diversity_factor = args.gene_diversity_factor,
        swa_start_epoch       = args.swa_start_epoch,
        swa_interval          = args.swa_interval,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
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

    csv_logger         = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tensorboard_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        limit_train_batches=lim_train,
        limit_val_batches=lim_val,
        limit_test_batches=lim_test,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, progress_bar],
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)

    # After training, decide test strategy:
    # - If SWA weights are available (>= 3 checkpoints), apply them and test.
    #   With swa_start_epoch=150 and patience=100, SWA will reliably activate.
    # - Otherwise, test with best checkpoint (standard behavior).
    if args.fast_dev_run or args.debug_max_step is not None:
        # Debug: test with current model (no ckpt_path)
        test_results = trainer.test(model, datamodule=dm)
    else:
        swa_available = (model._swa_params is not None and model._swa_n_averaged >= 3)
        if swa_available:
            # Apply SWA weights before testing
            model.apply_swa_weights()
            print(
                f"[Node1-3-1-2-1] Testing with SWA weights "
                f"({model._swa_n_averaged} checkpoints averaged)."
            )
            test_results = trainer.test(model, datamodule=dm)
        else:
            # SWA never activated (e.g., early stopping before swa_start_epoch)
            # Fall back to best checkpoint
            print(
                f"[Node1-3-1-2-1] SWA not available "
                f"(n_averaged={model._swa_n_averaged}). Using best checkpoint."
            )
            test_results = trainer.test(model, datamodule=dm, ckpt_path="best")

    # Save test score
    if test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        with open(score_path, "w") as f:
            f.write(str(test_results))
        print(f"[Node1-3-1-2-1] Test results: {test_results}")


if __name__ == "__main__":
    main()
