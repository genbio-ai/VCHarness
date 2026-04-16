"""Node 1-2: Frozen STRING_GNN (Neighborhood Attention K=16) + GenePriorBias — Bug-fixed version.

Critical fix over parent node1-3-2 (test F1=0.3821, val F1=0.4584):

1. **GenePriorBias test-inference bug fixed**: The parent node stored `current_epoch` as
   a plain Python int (not a buffer/parameter), so when the checkpoint was loaded for test
   inference `GenePriorBias.current_epoch` was reset to 0, disabling the calibration bias
   during test. Fix: register `bias_active` as a persistent buffer (bool) that is saved in
   the state_dict; once the warmup epoch is passed during training, `bias_active=True` is
   saved and restored at test time, ensuring bias is always applied post-warmup.

2. **Single LR instead of 3.3× discriminative LR**: The parent's 3.3× ratio
   (attn_lr=5e-4 vs head_lr=1.5e-4) was identified as causing convergence to inferior
   local minima (val F1=0.4584 < STRING-only baseline 0.4769 even on validation).
   Reverting to single lr=3e-4 matches node1-2-2-1's proven recipe (F1=0.4829).

3. **More conservative early stopping**: min_delta=5e-4 (vs parent's 2e-4) prevents
   the 59-epoch overfitting window seen in parent. Patience=10 with min_delta=5e-4
   is appropriate given the ~0.003/epoch improvement rate.

4. **Adjusted schedule**: T_max=100 with warmup=20 (total 120 epochs budget), matches the
   empirical evidence that STRING-only nodes peak at epoch 70-84. max_epochs=180 provides
   safety margin if GenePriorBias warmup=30 delays peak.

5. **GenePriorBias warmup=30**: Compromise between parent's aggressive warmup=20
   (too fast, caused instability) and node1-2-2-1's conservative warmup=50 (which worked
   but wasted 50 epochs). Warmup=30 provides stable pre-bias representation while leaving
   70+ epochs of active bias learning.

Architecture unchanged from parent:
    pert_id → STRING_GNN (FROZEN) → NeighborhoodAttention (K=16) → [B,256]
    → LayerNorm + Dropout → Bilinear(h[B,256] · gene_class_emb[3,6640,256]) → logits[B,3,6640]
    → GenePriorBias (active after warmup) → softmax probabilities

Memory connections:
- node1-3-2 (parent, F1=0.3821): Primary motivation — fix the critical test-inference bug.
  The val F1=0.4584 with bias active confirms the architecture can learn, but convergence
  to suboptimal minimum (vs STRING-only baseline 0.4769) suggests discriminative LR was
  counterproductive.
- node1-2-2-1 (F1=0.4829): Proven single-lr=3e-4 + GenePriorBias warmup=50 + STRING-only
  neighborhood attention achieved +0.006 over STRING-only baseline. Our improved version
  with fixed test-inference bug + warmup=30 should match or exceed this.
- node1-1-1-1-1 (F1=0.4846): Best STRING-only, confirms the ceiling of this architecture.
  Our target is to match or slightly exceed through better calibration.
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
    """Sqrt-inverse-frequency weights; neutral class stays close to 1."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0,1,2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)  # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)      # [N, G]
        is_pred = (y_hat == c)        # [N, G]
        present = is_true.any(dim=0)  # [G]

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
    import json as _json
    from transformers import AutoModel as _AM

    model = _AM.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
    model.eval()
    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
    node_names = _json.loads((STRING_GNN_DIR / "node_names.json").read_text())

    edge_index = graph["edge_index"]
    ew = graph.get("edge_weight", None)

    outputs = model(edge_index=edge_index, edge_weight=ew)
    emb = outputs.last_hidden_state.float().cpu()  # [18870, 256]

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
        neighbor_indices [N, K] long — STRING_GNN node indices of top-K neighbors (-1=padding)
        neighbor_weights [N, K] float — normalized STRING confidence weights
    """
    N = emb.shape[0]
    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
    edge_index = graph["edge_index"]  # [2, E]
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
    mask = neighbor_indices >= 0  # [N, K]
    raw = neighbor_weights.clone()
    raw[~mask] = -1e9
    norm_w = torch.softmax(raw, dim=-1)  # [N, K]
    norm_w[~mask] = 0.0

    return neighbor_indices, norm_w


# ---------------------------------------------------------------------------
# Neighborhood Attention Aggregator (proven design from node1-2)
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Aggregate top-K PPI neighbors for a center gene using learned attention."""

    def __init__(self, emb_dim: int = 256, attn_dim: int = 64) -> None:
        super().__init__()
        # Attention score: concat(center, neighbor) → scalar score
        self.attn_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False),
        )
        # Gate: how much neighbor context to add to center
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
        gate = torch.sigmoid(self.gate_proj(center_emb))  # [B, D]
        return center_emb + gate * aggregated              # [B, D]


# ---------------------------------------------------------------------------
# GenePriorBias — BUG-FIXED VERSION
# Key fix: `bias_active` is a persistent buffer that is saved in the state_dict.
# When checkpoint is loaded for test, bias_active=True is restored → bias applied.
# This eliminates the val/test gap caused by current_epoch=0 after checkpoint load.
# ---------------------------------------------------------------------------
class GenePriorBias(nn.Module):
    """Per-gene per-class additive bias initialized from class log-frequencies.

    BUG FIX over parent node1-3-2:
    The parent stored warmup gating state as `self.current_epoch` (plain Python int),
    which was reset to 0 on checkpoint load → bias disabled at test time → F1 collapse.

    This version uses `self.bias_active` as a `register_buffer` (persistent=True),
    ensuring it is saved and restored with the state_dict. Once warmup_epochs is
    passed during training, `bias_active` is set to True permanently. At test time
    (checkpoint loaded), bias_active=True is recovered from the state_dict.

    Proven impact:
    - node1-2-2-1: STRING-only + GenePriorBias → F1=0.4829 (+0.006 over 0.4769)
    - node4-2-1-1: scFoundation fusion + GenePriorBias → F1=0.4836 (+0.0035)
    """

    def __init__(self, n_genes: int, n_classes: int = 3, warmup_epochs: int = 30) -> None:
        super().__init__()
        self.warmup_epochs = warmup_epochs

        # Initialize at 0.3× log-frequency scale: modest prior, activated gradually.
        log_freqs = torch.tensor(
            [np.log(0.0429), np.log(0.9251), np.log(0.0320)], dtype=torch.float32
        )
        bias_init = log_freqs.unsqueeze(0).expand(n_genes, -1) * 0.3  # [G, 3]
        self.bias = nn.Parameter(bias_init.clone())  # [G, 3]

        # BUG FIX: persistent buffer that survives checkpoint save/load
        self.register_buffer("bias_active", torch.tensor(False))

    def activate(self) -> None:
        """Call this when warmup is complete. Sets bias_active=True (saved to checkpoint)."""
        self.bias_active.fill_(True)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Add per-gene bias to logits.

        Args:
            logits: [B, 3, G]
        Returns:
            biased_logits: [B, 3, G]
        """
        if not self.bias_active.item():
            return logits
        # self.bias: [G, 3] → permute to [3, G] → unsqueeze → [1, 3, G]
        bias = self.bias.permute(1, 0).unsqueeze(0)  # [1, 3, G]
        return logits + bias


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
            item["labels"] = self.labels[idx]  # [G] in {0,1,2}
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
class StringGNNWithPriorBias(pl.LightningModule):
    """Frozen STRING_GNN + Neighborhood Attention (K=16) + GenePriorBias (fixed) + Bilinear head.

    Architecture:
        pert_id
            |
        STRING_GNN (FROZEN, pre-computed buffer [18870, 256])
        NeighborhoodAttentionAggregator (K=16, attn_dim=64)
            |
        [B, 256]
            |
        LayerNorm(256) → Dropout(dropout)
            |
        Bilinear: h[B,256] dot gene_class_emb[3,6640,256] → logits[B,3,6640]
            |
        GenePriorBias (active after warmup_epochs, BUG-FIXED)  [B,3,6640]
            |
        Output: [B, 3, 6640] → softmax probabilities

    Key fix: GenePriorBias uses a persistent buffer `bias_active` (saved in state_dict)
    instead of `current_epoch` (transient Python int reset on checkpoint load).
    """

    def __init__(
        self,
        bilinear_dim: int = 256,
        attn_dim: int = 64,
        K: int = 16,
        dropout: float = 0.35,
        lr: float = 3e-4,
        weight_decay: float = 3e-2,
        warmup_epochs: int = 20,
        t_max: int = 100,
        eta_min: float = 5e-6,
        label_smoothing: float = 0.05,
        gene_prior_warmup: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Collect all pert_ids ----
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

        # ---- STRING_GNN: precompute embeddings ----
        self.print("Precomputing STRING_GNN embeddings...")
        string_emb, pert_to_gnn_idx = precompute_string_gnn_embeddings()
        # string_emb: [18870, 256]

        # Register as buffer for automatic GPU transfer
        self.register_buffer("node_embeddings", string_emb)  # [18870, 256]

        # Build pert_id → STRING_GNN node index mapping
        gnn_idx_tensor = torch.tensor(
            [pert_to_gnn_idx.get(pid, -1) for pid in unique_sorted], dtype=torch.long
        )
        self.register_buffer("pert_gnn_idx", gnn_idx_tensor)  # [M]

        # Precompute neighborhood tables
        self.print(f"Precomputing PPI neighborhood tables (K={hp.K})...")
        nb_indices, nb_weights = precompute_neighborhood(string_emb, K=hp.K)
        self.register_buffer("neighbor_indices", nb_indices)  # [18870, K]
        self.register_buffer("neighbor_weights", nb_weights)  # [18870, K]

        # Fallback embedding for pert_ids not in STRING
        self.fallback_emb = nn.Parameter(torch.zeros(1, 256))

        # ---- Trainable modules ----
        self.neighborhood_attn = NeighborhoodAttentionAggregator(
            emb_dim=256, attn_dim=hp.attn_dim
        )

        # Normalization before bilinear projection
        self.proj_norm = nn.LayerNorm(256)
        self.proj_dropout = nn.Dropout(hp.dropout)

        # Bilinear gene-class embedding: logits[b,c,g] = h[b] · gene_class_emb[c,g]
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # GenePriorBias: per-gene per-class calibration module (BUG-FIXED)
        self.gene_prior_bias = GenePriorBias(
            n_genes=N_GENES,
            n_classes=N_CLASSES,
            warmup_epochs=hp.gene_prior_warmup,
        )

        self.register_buffer("class_weights", get_class_weights())

        # Cast trainable parameters to float32 for stable optimization
        for k, v in self.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor] = []
        self._val_tgts:  List[torch.Tensor] = []
        self._val_idx:   List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    def on_train_epoch_start(self) -> None:
        """Activate GenePriorBias once warmup period is complete."""
        if self.current_epoch >= self.hparams.gene_prior_warmup:
            if not self.gene_prior_bias.bias_active.item():
                self.gene_prior_bias.activate()
                self.print(f"[Epoch {self.current_epoch}] GenePriorBias activated.")

    # BUG FIX: ensure bias_active is synced before test inference
    def on_test_epoch_start(self) -> None:
        """Ensure GenePriorBias is active at test time if warmup was completed."""
        # After checkpoint load, bias_active is restored from state_dict.
        # This hook serves as a safety net in case the bias_active buffer was
        # not saved (e.g., debug runs), but normally the buffer handles this.
        if self.current_epoch >= self.hparams.gene_prior_warmup:
            if not self.gene_prior_bias.bias_active.item():
                self.gene_prior_bias.activate()

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

        safe_nb_idx = nb_idx.clamp(min=0)                # [B, K]
        nb_emb = self.node_embeddings[safe_nb_idx].float()  # [B, K, 256]
        nb_emb = nb_emb * valid_mask.unsqueeze(-1).float()

        aggregated = self.neighborhood_attn(
            center_emb, nb_emb, nb_wts, valid_mask
        )  # [B, 256]
        return aggregated

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        h = self._get_neighborhood_emb(pert_ids)  # [B, 256] float32
        h = self.proj_norm(h)
        h = self.proj_dropout(h)
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)  # [B, 3, G]
        logits = self.gene_prior_bias(logits)                          # [B, 3, G]
        return logits

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        loss = F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, 3]
            targets.reshape(-1),                       # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )
        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["pert_id"])
        loss = self._loss(logits, batch["labels"])
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
        local_preds = torch.cat(self._val_preds, dim=0)  # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  dim=0)  # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   dim=0)  # [N_local]
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds)  # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)   # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)    # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        order  = torch.argsort(idx_flat)
        s_idx  = idx_flat[order]
        s_pred = preds_flat[order]
        s_tgt  = tgts_flat[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device),
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
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)  # [N_local]
        all_preds   = self.all_gather(local_preds)          # [W, N_local, 3, G]
        all_idx     = self.all_gather(local_idx)            # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            order  = torch.argsort(idx_flat)
            s_idx  = idx_flat[order]
            s_pred = preds_flat[order]
            mask   = torch.cat([torch.ones(1, dtype=torch.bool, device=s_idx.device),
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
            print(f"[Node1-2] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
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

    # ---- Optimizer (single LR — matches node1-2-2-1 proven recipe) ----
    def configure_optimizers(self):
        hp = self.hparams

        # Single LR for all trainable parameters (proven in node1-2-2-1: F1=0.4829)
        # The parent's 3.3× discriminative LR caused convergence to inferior local minima.
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )

        # Linear warmup for warmup_epochs, then CosineAnnealingLR for T_max epochs
        warmup_sch = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=hp.t_max,
            eta_min=hp.eta_min,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_sch, cosine_sch],
            milestones=[hp.warmup_epochs],
        )
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
        description="Node1-2: STRING_GNN + Neighborhood Attention + GenePriorBias (Bug-Fixed)"
    )
    parser.add_argument("--micro_batch_size",  type=int,   default=32)
    parser.add_argument("--global_batch_size", type=int,   default=256)
    parser.add_argument("--max_epochs",        type=int,   default=180)
    parser.add_argument("--lr",                type=float, default=3e-4)
    parser.add_argument("--weight_decay",      type=float, default=3e-2)
    parser.add_argument("--bilinear_dim",      type=int,   default=256)
    parser.add_argument("--attn_dim",          type=int,   default=64)
    parser.add_argument("--k_neighbors",       type=int,   default=16)
    parser.add_argument("--dropout",           type=float, default=0.35)
    parser.add_argument("--warmup_epochs",     type=int,   default=20)
    parser.add_argument("--t_max",             type=int,   default=100)
    parser.add_argument("--eta_min",           type=float, default=5e-6)
    parser.add_argument("--label_smoothing",   type=float, default=0.05)
    parser.add_argument("--gene_prior_warmup", type=int,   default=30)
    parser.add_argument("--patience",          type=int,   default=10)
    parser.add_argument("--num_workers",       type=int,   default=4)
    parser.add_argument("--val_check_interval",type=float, default=1.0)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--debug_max_step",    type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast_dev_run",      action="store_true",
                        dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # val_check_interval: use argument value in normal mode, force 1.0 in debug mode
    if args.debug_max_step is not None or fast_dev_run:
        val_check_interval = 1.0
    else:
        val_check_interval = args.val_check_interval

    # DataModule + Model
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    model = StringGNNWithPriorBias(
        bilinear_dim      = args.bilinear_dim,
        attn_dim          = args.attn_dim,
        K                 = args.k_neighbors,
        dropout           = args.dropout,
        lr                = args.lr,
        weight_decay      = args.weight_decay,
        warmup_epochs     = args.warmup_epochs,
        t_max             = args.t_max,
        eta_min           = args.eta_min,
        label_smoothing   = args.label_smoothing,
        gene_prior_warmup = args.gene_prior_warmup,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
        auto_insert_metric_name = False,
    )
    es_cb = EarlyStopping(
        monitor="val/f1", mode="max", patience=args.patience, min_delta=5e-4
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy
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
        gradient_clip_val       = args.gradient_clip_val,
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
    print(f"[Node1-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
