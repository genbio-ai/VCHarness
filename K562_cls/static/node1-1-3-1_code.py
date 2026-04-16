"""Node 1-1-3-1 – Fine-tuned STRING_GNN + Neighborhood Attention (K=16) + Improved GenePriorBias

Strategy: Directly fixes the catastrophic cold-start failure in parent node (node1-1-3).

Root cause of parent failure (F1=0.4093):
  - Frozen STRING_GNN backbone: no adaptive signal for K562 DEG prediction
  - Strong GenePriorBias init locked for 50 epochs → all-neutral prediction trap (16 epochs wasted)
  - 20-epoch LR warmup: too slow to escape cold-start

Core fixes in this node:
  1. UNFREEZE STRING_GNN backbone with differential LR:
     - backbone LR = 3e-5 (10x lower than head, matching STRING_GNN skill recommendation)
     - head LR = 3e-4
     - Same approach as proven node1-1-1-1-1 (F1=0.4846, best STRING_GNN-only result)
  2. Reduce warmup_epochs from 20 → 5 (escape cold-start faster)
  3. Reduce bias_warmup_epochs from 50 → 10 (allow bias adaptation early)
  4. Reduce initial GenePriorBias magnitude (0.3 × log_freq → reduces neutral preference from 25x to ~3.5x)
  5. Reduce weight_decay from 2e-2 → 1e-2

Retained from proven lineage (node1-1-1-1-1):
  - Neighborhood attention K=16, attn_dim=64: proven +0.010 F1 in fine-tuned context
  - Simple 2-layer MLP (no ResBlocks): proven best on 1,388 samples
  - Scaled bilinear output: sqrt(bilinear_dim) scaling for stable logits
  - Weighted CE + label smoothing 0.05: best calibration

Expected target F1: >0.485 (node1-1-1-1-1 base 0.4846 + improved GenePriorBias +0.003-0.005)
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

    Uses a double stable sort: first by weight descending, then stable sort
    by source node ascending. O(E log E) — avoids Python loops over edges.

    Returns:
        neighbor_idx : [n_nodes, K] long tensor  (-1 = no neighbor)
        neighbor_wt  : [n_nodes, K] float tensor (edge confidence weights)
    """
    src = edge_index[0].cpu()
    dst = edge_index[1].cpu()
    wt  = edge_weight.cpu().float()

    # Step 1: sort by weight descending (secondary key)
    order1 = torch.argsort(-wt, stable=True)
    src1 = src[order1]; dst1 = dst[order1]; wt1 = wt[order1]

    # Step 2: stable sort by source ascending (primary key)
    order2 = torch.argsort(src1, stable=True)
    src_sorted = src1[order2]; dst_sorted = dst1[order2]; wt_sorted = wt1[order2]

    neighbor_idx = torch.full((n_nodes, K), -1, dtype=torch.long)
    neighbor_wt  = torch.zeros(n_nodes, K,      dtype=torch.float32)

    if len(src_sorted) == 0:
        return neighbor_idx, neighbor_wt

    change_mask = torch.cat([
        torch.tensor([True]),
        src_sorted[1:] != src_sorted[:-1],
    ])
    start_positions = change_mask.nonzero(as_tuple=True)[0]
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
            item["labels"] = self.labels[idx]
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

    Proven to add +0.010 F1 improvement in node1-1-1-1-1 (fine-tuned backbone context).
    Architecture: K=16 neighbors, attn_dim=64, edge-confidence log-bias, sigmoid gating.

    IMPORTANT: This module works correctly only when the backbone is being fine-tuned.
    The parent node1-1-3 froze the backbone, which prevented the attention from learning
    meaningful PPI-neighborhood patterns. This node fixes that by unfreezing the backbone.
    """

    def __init__(self, emb_dim: int = 256, attn_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.attn_dim = attn_dim

        self.W_q    = nn.Linear(emb_dim, attn_dim, bias=False)
        self.W_k    = nn.Linear(emb_dim, attn_dim, bias=False)
        self.W_v    = nn.Linear(emb_dim, emb_dim,  bias=False)
        self.W_gate = nn.Linear(emb_dim * 2, emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(emb_dim)

        # Conservative gate init: start with near-zero gate contribution
        nn.init.normal_(self.W_gate.weight, std=0.01)
        nn.init.zeros_(self.W_gate.bias)

    def forward(
        self,
        query_emb:    torch.Tensor,  # [B, D]
        neighbor_emb: torch.Tensor,  # [B, K, D]
        neighbor_wt:  torch.Tensor,  # [B, K]
        valid_mask:   torch.Tensor,  # [B, K] bool
    ) -> torch.Tensor:
        q = self.W_q(query_emb).unsqueeze(1)   # [B, 1, attn_dim]
        k = self.W_k(neighbor_emb)              # [B, K, attn_dim]
        v = self.W_v(neighbor_emb)              # [B, K, D]

        scores = (q * k).sum(-1) / (self.attn_dim ** 0.5)
        scores = scores + neighbor_wt.clamp(min=1e-6).log()
        scores = scores.masked_fill(~valid_mask, float('-inf'))
        attn_w = torch.softmax(scores, dim=-1)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)
        attn_w = self.dropout(attn_w)

        aggregated = (attn_w.unsqueeze(-1) * v).sum(1)   # [B, D]
        gate       = torch.sigmoid(self.W_gate(torch.cat([query_emb, aggregated], dim=-1)))
        enriched   = query_emb + gate * aggregated

        return self.norm(enriched)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class StringGNNFinetuneNeighborBiasModel(pl.LightningModule):
    """Fine-tuned STRING_GNN + Neighborhood Attention (K=16) + Improved GenePriorBias.

    Architecture:
        1. Trainable STRING_GNN backbone — runs per forward pass to produce [18870, 256] embeddings
        2. Lookup + Neighborhood Attention (K=16, attn_dim=64) → enriched [B, 256]
        3. Simple 2-layer MLP: LN→Linear(256→256)→GELU→Dropout→LN→Linear(256→256)→GELU→Dropout
        4. Scaled bilinear: logits[b,c,g] = h[b] · gene_class_emb[c,g] / sqrt(bilinear_dim)
                            + gene_prior_bias[c,g]

    Key difference from parent node1-1-3:
        - Backbone is TRAINABLE (not frozen) with backbone_lr=3e-5 (10x lower than head)
        - GenePriorBias init = 0.3 * log_freq (reduced neutral dominance: 25x → ~3.5x)
        - bias_warmup_epochs = 10 (not 50): bias adapts after 10 epochs
        - warmup_epochs = 5 (not 20): faster escape from cold-start
    """

    def __init__(
        self,
        hidden_dim:         int   = 256,
        bilinear_dim:       int   = 256,
        dropout:            float = 0.35,
        lr:                 float = 3e-4,
        backbone_lr:        float = 3e-5,
        weight_decay:       float = 1e-2,
        backbone_wd:        float = 0.0,
        label_smoothing:    float = 0.05,
        warmup_epochs:      int   = 5,
        t_max:              int   = 220,
        bias_warmup_epochs: int   = 10,
        bias_init_scale:    float = 0.3,
        K:                  int   = 16,
        attn_dim:           int   = 64,
    ) -> None:
        super().__init__()
        self._setup_done = False
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if self._setup_done:
            return
        self._setup_done = True

        hp = self.hparams

        # Load STRING_GNN: rank 0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # KEY FIX: Load backbone as TRAINABLE module (not frozen!)
        # This is the single most critical fix over parent node1-1-3.
        # node1-1-1-1-1 proved that fine-tuned backbone + attention = F1=0.4846.
        self.backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        self.backbone.train()

        # Load graph data and register as buffers (needed for per-step backbone forward)
        graph_data  = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph_data["edge_index"].long()
        edge_weight = graph_data["edge_weight"].float()

        self.register_buffer("edge_index",  edge_index)   # [2, 786012]
        self.register_buffer("edge_weight", edge_weight)  # [786012]

        # Build top-K neighbor lookup table (based on original graph weights; computed once)
        n_nodes = 18870
        neighbor_idx, neighbor_wt = build_neighbor_table(
            edge_index, edge_weight, n_nodes=n_nodes, K=hp.K
        )
        self.register_buffer("neighbor_idx", neighbor_idx)  # [18870, K]
        self.register_buffer("neighbor_wt",  neighbor_wt)   # [18870, K]

        del graph_data

        # Fallback embedding for pert_ids not in STRING (~6.4%)
        self.fallback_emb = nn.Embedding(1, 256)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # Neighborhood attention (K=16, attn_dim=64) — proven +0.010 F1 with fine-tuned backbone
        self.nbr_attn = NeighborhoodAttention(
            emb_dim=256, attn_dim=hp.attn_dim, dropout=0.1
        )

        # 2-layer MLP (no ResBlocks — proven best on 1,388 samples)
        STRING_DIM = 256
        self.mlp = nn.Sequential(
            nn.LayerNorm(STRING_DIM),
            nn.Linear(STRING_DIM,    hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
            nn.LayerNorm(hp.hidden_dim),
            nn.Linear(hp.hidden_dim, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # Bilinear gene-class embedding matrix [C, G, bilinear_dim]
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # Improved GenePriorBias: scaled down to reduce neutral-class trap
        # bias_init_scale=0.3: neutral init = 0.3 * (-0.078) = -0.023 (vs -0.078 in parent)
        # DEG class inits = 0.3 * (-3.15, -3.44) = (-0.945, -1.032)
        # Neutral preference ratio: exp((-0.023) - (-0.945)) ≈ exp(0.922) ≈ 2.5x (vs 25x in parent)
        log_freq  = torch.tensor(CLASS_FREQ).log()   # [-3.15, -0.08, -3.44]
        init_bias = (log_freq * hp.bias_init_scale).unsqueeze(1).expand(N_CLASSES, N_GENES).clone()
        self.gene_prior_bias = nn.Parameter(init_bias)  # [3, G]

        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        for _, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Accumulators for val/test
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    # ── Forward helpers ───────────────────────────────────────────────────────

    def _get_pert_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Get neighborhood-attended perturbation embeddings.

        Runs fine-tuned backbone forward to get all 18870 node embeddings,
        then looks up the perturbed gene and its top-K neighbors.

        Args:
            string_node_idx: [B] long tensor, -1 for unknowns
        Returns:
            [B, 256] float enriched perturbation embeddings
        """
        B      = string_node_idx.shape[0]
        K      = int(self.hparams.K)
        device = string_node_idx.device

        # Run full backbone GNN forward to get all node embeddings [18870, 256]
        # This provides adaptive signal — the key fix over frozen parent node1-1-3
        node_emb = self.backbone(
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
        ).last_hidden_state.float()   # [18870, 256]

        known_mask   = string_node_idx >= 0
        unknown_mask = ~known_mask

        # Center embeddings [B, 256]
        base_emb = torch.zeros(B, 256, dtype=torch.float32, device=device)
        if known_mask.any():
            base_emb[known_mask] = node_emb[string_node_idx[known_mask]]
        if unknown_mask.any():
            n_unk = int(unknown_mask.sum().item())
            base_emb[unknown_mask] = self.fallback_emb(
                torch.zeros(n_unk, dtype=torch.long, device=device)
            ).float()

        # Neighbor indices and weights
        nbr_idx = torch.full((B, K), -1, dtype=torch.long,    device=device)
        nbr_wt  = torch.zeros(B, K,      dtype=torch.float32, device=device)
        if known_mask.any():
            known_node_idx      = string_node_idx[known_mask]
            nbr_idx[known_mask] = self.neighbor_idx[known_node_idx]
            nbr_wt[known_mask]  = self.neighbor_wt[known_node_idx]

        valid_mask   = nbr_idx >= 0                        # [B, K]
        safe_nbr_idx = nbr_idx.clamp(min=0)               # safe indexing
        nbr_emb      = node_emb[safe_nbr_idx]              # [B, K, 256]

        # Neighborhood attention for all samples
        # Gradient flows through nbr_emb → node_emb → backbone (fine-tuning signal)
        enriched = self.nbr_attn(
            query_emb    = base_emb,
            neighbor_emb = nbr_emb,
            neighbor_wt  = nbr_wt,
            valid_mask   = valid_mask,
        )   # [B, 256]

        return enriched.float()

    def forward(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        pert_emb = self._get_pert_embeddings(string_node_idx)        # [B, 256]
        h        = self.mlp(pert_emb)                                 # [B, bilinear_dim]

        # Scaled bilinear (sqrt scaling for stable logit magnitudes)
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)  # [B, 3, G]
        logits = logits / (self.hparams.bilinear_dim ** 0.5)

        # Add per-gene-class prior bias (reduced init to avoid neutral trap)
        logits = logits + self.gene_prior_bias.unsqueeze(0)            # [B, 3, G]

        return logits

    # ── GenePriorBias gradient warmup (10 epochs) ─────────────────────────────

    def on_after_backward(self) -> None:
        """Zero GenePriorBias gradient during warmup (only 10 epochs, not 50 like parent).

        Short warmup lets the bias adapt once the bilinear head has basic convergence,
        without locking it for 91% of training as in the failed parent node.
        """
        if self.current_epoch < self.hparams.bias_warmup_epochs:
            if self.gene_prior_bias.grad is not None:
                self.gene_prior_bias.grad.zero_()

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
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

        local_preds = torch.cat(self._val_preds, dim=0)
        local_tgts  = torch.cat(self._val_tgts,  dim=0)
        local_idx   = torch.cat(self._val_idx,   dim=0)
        self._val_preds.clear()
        self._val_tgts.clear()
        self._val_idx.clear()

        all_preds = self.all_gather(local_preds)
        all_tgts  = self.all_gather(local_tgts)
        all_idx   = self.all_gather(local_idx)

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

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

        local_preds = torch.cat(self._test_preds, dim=0)
        local_idx   = torch.cat(self._test_idx,   dim=0)
        all_preds   = self.all_gather(local_preds)
        all_idx     = self.all_gather(local_idx)

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            order  = torch.argsort(idx_flat)
            s_idx  = idx_flat[order]
            s_pred = preds_flat[order]
            mask   = torch.cat([
                torch.ones(1, dtype=torch.bool, device=s_idx.device),
                s_idx[1:] != s_idx[:-1],
            ])
            preds_dedup = s_pred[mask]
            unique_sid  = s_idx[mask].tolist()

            test_df     = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {
                i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                for i in range(len(test_df))
            }

            rows = []
            for i, sid in enumerate(unique_sid):
                pid, sym  = idx_to_meta[int(sid)]
                pred_list = preds_dedup[i].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-1-3-1] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save trainable parameters and persistent buffers."""
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

    # ── Optimizer & scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self):
        hp = self.hparams

        # Differential LR: backbone (low LR) vs head (high LR)
        # STRING_GNN skill recommends weight_decay=0 for backbone (pretraining used 0)
        backbone_params = list(self.backbone.parameters())
        head_params = (
            list(self.fallback_emb.parameters()) +
            list(self.nbr_attn.parameters()) +
            list(self.mlp.parameters()) +
            [self.gene_class_emb, self.gene_prior_bias]
        )

        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": hp.backbone_lr, "weight_decay": hp.backbone_wd},
            {"params": head_params,     "lr": hp.lr,          "weight_decay": hp.weight_decay},
        ])

        # Short linear warmup (5 epochs) — prevents cold-start without long delay
        warmup_sch = LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=hp.warmup_epochs
        )
        # Cosine annealing after warmup
        cosine_sch = CosineAnnealingLR(opt, T_max=hp.t_max, eta_min=1e-6)

        sch = SequentialLR(
            opt, schedulers=[warmup_sch, cosine_sch], milestones=[hp.warmup_epochs]
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
        description="Node1-1-3-1 – Fine-tuned STRING_GNN + Neighborhood Attention + Improved GenePriorBias",
        allow_abbrev=False,
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=32)
    parser.add_argument("--global-batch-size",   type=int,   default=256)
    parser.add_argument("--max-epochs",          type=int,   default=250)
    parser.add_argument("--lr",                  type=float, default=3e-4)
    parser.add_argument("--backbone-lr",         type=float, default=3e-5,
                        dest="backbone_lr")
    parser.add_argument("--weight-decay",        type=float, default=1e-2)
    parser.add_argument("--backbone-wd",         type=float, default=0.0,
                        dest="backbone_wd")
    parser.add_argument("--hidden-dim",          type=int,   default=256)
    parser.add_argument("--bilinear-dim",        type=int,   default=256)
    parser.add_argument("--dropout",             type=float, default=0.35)
    parser.add_argument("--label-smoothing",     type=float, default=0.05)
    parser.add_argument("--warmup-epochs",       type=int,   default=5)
    parser.add_argument("--t-max",               type=int,   default=220)
    parser.add_argument("--bias-warmup-epochs",  type=int,   default=10,
                        dest="bias_warmup_epochs")
    parser.add_argument("--bias-init-scale",     type=float, default=0.3,
                        dest="bias_init_scale")
    parser.add_argument("--K",                   type=int,   default=16)
    parser.add_argument("--attn-dim",            type=int,   default=64)
    parser.add_argument("--num-workers",         type=int,   default=4)
    parser.add_argument("--patience",            type=int,   default=20)
    parser.add_argument("--min-epochs",          type=int,   default=15,
                        dest="min_epochs")
    parser.add_argument("--debug-max-step",      type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",        action="store_true",
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

    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    model = StringGNNFinetuneNeighborBiasModel(
        hidden_dim         = args.hidden_dim,
        bilinear_dim       = args.bilinear_dim,
        dropout            = args.dropout,
        lr                 = args.lr,
        backbone_lr        = args.backbone_lr,
        weight_decay       = args.weight_decay,
        backbone_wd        = args.backbone_wd,
        label_smoothing    = args.label_smoothing,
        warmup_epochs      = args.warmup_epochs,
        t_max              = args.t_max,
        bias_warmup_epochs = args.bias_warmup_epochs,
        bias_init_scale    = args.bias_init_scale,
        K                  = args.K,
        attn_dim           = args.attn_dim,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-val_f1={val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
        auto_insert_metric_name = False,
    )
    es_cb = EarlyStopping(
        monitor="val/f1", mode="max",
        patience=args.patience,
        min_delta=1e-4,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # find_unused_parameters=True: gene_prior_bias gradients are zeroed during warmup in on_after_backward
    # This is safe since gene_prior_bias IS used in forward; DDP just can't always detect it through autograd
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
        min_epochs              = args.min_epochs,
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

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1-1-3-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
