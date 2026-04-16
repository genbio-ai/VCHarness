"""Node 4-2-2-1-1-1 – AIDO.Cell-100M (LoRA r=16) + STRING_GNN (frozen, cached)
             + NeighborhoodAttention (K=16, attn_dim=64) + Simple Concat Fusion
             + GenePriorBias (warmup=30) — NO SWA

Strategy: Fix the primary bottleneck from the parent node (node4-2-2-1-1, F1=0.4738):
  - GatedFusion is INCOMPATIBLE with LoRA-adapted AIDO.Cell embeddings
  - Replace GatedFusion with simple concatenation (640+256=896-dim)
  - This directly mirrors node2-1-1-1 (F1=0.5059, tree best) which also used concat
  - Add GenePriorBias (warmup=30) as the only novel component over node2-1-1-1
  - Increase LoRA rank r=8→r=16 for higher adaptation capacity

Key changes from parent (node4-2-2-1-1, F1=0.4738):
1. Remove GatedFusion → Simple concatenation (640+256=896-dim) [CRITICAL FIX]
   - Parent's feedback: "GatedFusion is incompatible with LoRA-adapted AIDO.Cell embeddings"
   - node2-1-1-1 (F1=0.5059) used simple concat and achieved tree best
   - LoRA shifts the summary token distribution in ways GatedFusion cannot correctly handle
2. Increase LoRA rank: r=8 → r=16 (~1.1M trainable LoRA params, 2x capacity)
   - r=16 gives more adaptation capacity for the 640-dim AIDO.Cell summary token
   - Combined with concat (not GatedFusion), allows fuller exploration of the representation
3. Update classification head: Linear(896, 256) → GELU → Dropout(0.5) → Linear(256, 19920)
   - Matches node2-1-1-1's proven 256-dim head architecture with 896-dim input
   - Head capacity appropriate for the fusion dimension
4. Reduce max_epochs: 200 → 300 (parent ran all 200 epochs; more budget needed)
5. Tighten patience: 40 → 20 (parent's ES never fired; tighter is appropriate)
6. Reduce weight_decay: 3e-2 → 2e-2 (matching node2-1-1-1's proven recipe)
7. Keep warmup: 10 (same as parent; appropriate for LoRA)
8. Retain GenePriorBias(warmup=30) as incremental addition over node2-1-1-1

Architecture:
  AIDO.Cell-100M (LoRA r=16, all 18 layers) → summary_token[19264] → [B, 640]
  STRING_GNN (FROZEN, cached) → lookup [B, 256]
  NeighborhoodAttentionModule: aggregate top-16 PPI neighbors → enriched [B, 256]
  Concat: [B, 640] || [B, 256] → [B, 896]
  [Mixup alpha=0.2 during training on fused embedding]
  Head: Dropout(0.5) → Linear(896→256) → GELU → Dropout(0.25) → Linear(256→19920)
  GenePriorBias([3, 6640], gradient zeroed for epochs 0-30)
  → [B, 3, 6640] softmax probabilities
"""

from __future__ import annotations

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES     = 6640
N_CLASSES   = 3
AIDO_HIDDEN = 640    # AIDO.Cell-100M hidden size
GNN_HIDDEN  = 256    # STRING_GNN hidden size
CONCAT_DIM  = AIDO_HIDDEN + GNN_HIDDEN  # 896 — simple concat dimension
HEAD_HIDDEN = 256    # Two-layer head intermediate dimension (896 → 256 → 19920)

AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-100M"
GNN_MODEL_DIR  = "/home/Models/STRING_GNN"

CLASS_FREQ = [0.0429, 0.9251, 0.0320]   # down(-1→0), neutral(0→1), up(1→2)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Inverse-sqrt-frequency class weights to handle 92.5% neutral class dominance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float32 softmax probabilities
        targets: [N, G]   int64 class indices in {0, 1, 2}
    Returns:
        Scalar F1 averaged over genes.
    """
    y_hat = preds.argmax(dim=1)  # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)
    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0).float()
        tp  = (is_pred & is_true).float().sum(0)
        fp  = (is_pred & ~is_true).float().sum(0)
        fn  = (~is_pred & is_true).float().sum(0)
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(
            prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-8), torch.zeros_like(prec)
        )
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# NeighborhoodAttentionModule
# ---------------------------------------------------------------------------
class NeighborhoodAttentionModule(nn.Module):
    """Lightweight PPI neighborhood aggregation via attention over top-K neighbors.

    For each perturbed gene, retrieves the K highest-confidence PPI neighbors
    and aggregates their embeddings with an attention-weighted mean, using a
    center-context gating mechanism.

    Proven to provide +0.0035 F1 improvement in node4-2-1 and +0.010 in node1-1-1-1-1.
    Also proven effective in node2-1-1-1 (F1=0.5059, tree best).

    Architecture:
        - center (query): [B, d] → linear query projection → [B, attn_dim]
        - neighbors (keys/vals): [B, K, d] → linear key projection → [B, K, attn_dim]
        - attention: softmax(query · keys^T / sqrt(attn_dim)) → [B, K]
        - weighted_neighbor = sum_k attn_k * neighbor_k → [B, d]
        - gate = sigmoid(linear(center || weighted_neighbor)) → [B, d]
        - output = gate * center + (1 - gate) * weighted_neighbor → [B, d]
    """

    def __init__(
        self,
        d_gnn: int = GNN_HIDDEN,
        K: int = 16,
        attn_dim: int = 64,
    ) -> None:
        super().__init__()
        self.K = K
        self.scale = attn_dim ** -0.5
        self.query_proj  = nn.Linear(d_gnn, attn_dim, bias=False)
        self.key_proj    = nn.Linear(d_gnn, attn_dim, bias=False)
        self.gate_linear = nn.Linear(d_gnn * 2, d_gnn)

    def forward(
        self,
        center_emb: torch.Tensor,       # [B, d_gnn]
        all_embs: torch.Tensor,         # [N_nodes, d_gnn]
        neighbor_indices: torch.Tensor, # [B, K] int64
        neighbor_weights: torch.Tensor, # [B, K] float (STRING confidence)
    ) -> torch.Tensor:
        """Aggregate neighborhood context into each center gene's embedding.

        Returns:
            [B, d_gnn] enriched embedding
        """
        B, d = center_emb.shape
        K = neighbor_indices.shape[1]

        # Gather neighbor embeddings: [B, K, d]
        neigh_embs = all_embs[neighbor_indices.view(-1)].view(B, K, d)

        # Attention: query from center, keys from neighbors
        q = self.query_proj(center_emb).unsqueeze(1)          # [B, 1, attn_dim]
        k = self.key_proj(neigh_embs)                          # [B, K, attn_dim]
        scores = (q * k).sum(-1) * self.scale                  # [B, K]

        # Incorporate STRING confidence weights as prior (log-space addition)
        scores = scores + torch.log(neighbor_weights.clamp(min=1e-6))

        # Handle padding: zero-confidence neighbors get -inf score
        pad_mask = (neighbor_weights < 1e-6)
        scores = scores.masked_fill(pad_mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)                   # [B, K]
        attn = torch.nan_to_num(attn, nan=0.0)                 # handle all-inf rows

        # Weighted neighbor aggregation: [B, d]
        weighted_neigh = (attn.unsqueeze(-1) * neigh_embs).sum(1)

        # Center-context gating
        gate = torch.sigmoid(
            self.gate_linear(torch.cat([center_emb, weighted_neigh], dim=-1))
        )                                                       # [B, d]
        return gate * center_emb + (1.0 - gate) * weighted_neigh


# ---------------------------------------------------------------------------
# GenePriorBias Module
# ---------------------------------------------------------------------------
class GenePriorBias(nn.Module):
    """Per-gene, per-class log-prior bias.

    Initialized from training class frequency statistics and activated only
    after bias_warmup_epochs epochs (gradient zeroed before that point).

    Proven to yield +0.0035 F1 improvement in node4-2-1-1.
    Using warmup=30 (same as parent) for 270 effective learning epochs.
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        n_genes: int = N_GENES,
        bias_warmup_epochs: int = 30,
    ) -> None:
        super().__init__()
        self.bias_warmup_epochs = bias_warmup_epochs
        # Learnable per-gene, per-class bias initialized to log class frequencies
        log_prior = torch.tensor([
            [math.log(CLASS_FREQ[c] + 1e-9) for _ in range(n_genes)]
            for c in range(n_classes)
        ], dtype=torch.float32)  # [3, 6640]
        self.bias = nn.Parameter(log_prior)  # [n_classes, n_genes]

    def forward(
        self,
        logits: torch.Tensor,
        current_epoch: int,
    ) -> torch.Tensor:
        """Add per-gene class bias to logits.

        Args:
            logits: [B, 3, G] raw logits from the head
            current_epoch: current training epoch (0-indexed)
        Returns:
            [B, 3, G] logits + bias
        """
        return logits + self.bias.unsqueeze(0)  # [1, 3, G] broadcast over batch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List[torch.Tensor]] = (
            [
                torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
                for row in df["label"].tolist()
            ]
            if has_label else None
        )

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx": idx,
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_collate_aido(tokenizer):
    """Collate function that tokenizes inputs for AIDO.Cell.

    AIDO.Cell takes gene expression dict: missing genes get -1.0 (filled by tokenizer).
    We pass a single gene with expression=1.0 to represent perturbation identity.
    """

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        # AIDO.Cell: provide the perturbed gene's Ensembl ID with expression=1.0
        # All other 19263 genes will be filled with -1.0 (missing) by the tokenizer
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      tokenized["input_ids"],       # [B, 19264] float32
            "attention_mask": tokenized["attention_mask"],  # [B, 19264] int64
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out

    return collate_fn


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer   = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Rank-0 downloads/loads tokenizer first, others wait
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        self.train_ds = DEGDataset(pd.read_csv(TRAIN_TSV, sep="\t"))
        self.val_ds   = DEGDataset(pd.read_csv(VAL_TSV,   sep="\t"))
        self.test_ds  = DEGDataset(pd.read_csv(TEST_TSV,  sep="\t"))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate_aido(self.tokenizer),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate_aido(self.tokenizer),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate_aido(self.tokenizer),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FusionDEGModel(pl.LightningModule):
    """AIDO.Cell-100M (LoRA r=16) + STRING_GNN (frozen, cached)
       + NeighborhoodAttention(K=16) + Simple Concatenation + GenePriorBias.

    Key improvements over parent (node4-2-2-1-1, F1=0.4738):
    1. Replace GatedFusion with simple concatenation (640+256=896-dim) [PRIMARY FIX]:
       - Parent's feedback explicitly identified GatedFusion as the root cause of failure
       - node2-1-1-1 (F1=0.5059, tree best) used simple concat and achieved tree best
       - GatedFusion learns gates on LoRA-shifted AIDO.Cell embeddings, which are
         partially adapted but still largely determined by pretraining — gates cannot
         compensate for this misalignment while concat handles heterogeneity robustly
    2. Increase LoRA rank: r=8 → r=16 (higher adaptation capacity):
       - ~1.1M trainable LoRA params vs 0.55M with r=8
       - Combined with concat (not GatedFusion), r=16 provides richer perturbation signal
       - lora_alpha=32 (2x rank scaling, consistent ratio)
    3. Adapted classification head: Linear(896, 256) → GELU → Dropout(0.25) → Linear(256, 19920)
       - 896-dim (concat) input matching the architecture to the new fusion scheme
       - 256-dim hidden matches node2-1-1-1's proven capacity
    4. Tighter patience: 40 → 20 (parent ran all 200 epochs without ES; tighter is better)
    5. Extended max_epochs: 200 → 300 (model may still be improving slowly at 200)
    6. Reduce weight_decay: 3e-2 → 2e-2 (matching node2-1-1-1's proven recipe)
    7. Retain GenePriorBias(warmup=30) as incremental addition over node2-1-1-1
    """

    def __init__(
        self,
        lora_r: int              = 16,
        lora_alpha: int          = 32,
        lora_dropout: float      = 0.05,
        head_dropout: float      = 0.5,
        lr: float                = 1e-4,
        weight_decay: float      = 2e-2,
        warmup_epochs: int       = 10,
        max_epochs: int          = 300,
        min_lr_ratio: float      = 0.05,
        mixup_alpha: float       = 0.2,
        label_smoothing: float   = 0.1,
        bias_warmup_epochs: int  = 30,
        nb_attn_k: int           = 16,
        nb_attn_dim: int         = 64,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        gnn_dir = Path(GNN_MODEL_DIR)

        # ----------------------------------------------------------------
        # AIDO.Cell-100M backbone with LoRA fine-tuning (r=16)
        # ----------------------------------------------------------------
        print("[Node4-2-2-1-1-1] Loading AIDO.Cell-100M for LoRA fine-tuning (r=16)...")
        aido_base = AutoModel.from_pretrained(
            AIDO_MODEL_DIR,
            trust_remote_code=True,
        ).to(torch.bfloat16)
        aido_base.config.use_cache = False

        # Apply LoRA to Q/K/V projections in all 18 transformer layers
        # r=16 (vs parent's r=8): higher adaptation capacity for AIDO.Cell summary token
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,  # Apply LoRA to all 18 layers
        )
        self.aido = get_peft_model(aido_base, lora_cfg)
        self.aido.config.use_cache = False
        self.aido.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA (trainable) params to float32 for stable optimization
        for name, param in self.aido.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        aido_train = sum(p.numel() for p in self.aido.parameters() if p.requires_grad)
        aido_total = sum(p.numel() for p in self.aido.parameters())
        print(f"[Node4-2-2-1-1-1] AIDO.Cell-100M LoRA r={hp.lora_r}: {aido_train:,}/{aido_total:,} trainable params "
              f"({100*aido_train/aido_total:.2f}%)")

        # ----------------------------------------------------------------
        # STRING_GNN: fully frozen, embeddings precomputed and cached
        # ----------------------------------------------------------------
        print("[Node4-2-2-1-1-1] Precomputing STRING_GNN embeddings (frozen)...")
        gnn_temp = AutoModel.from_pretrained(str(gnn_dir), trust_remote_code=True).float()
        gnn_temp.eval()
        graph_data  = torch.load(gnn_dir / "graph_data.pt", map_location="cpu")
        edge_index  = graph_data["edge_index"].long()
        edge_weight = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.float()
        with torch.no_grad():
            gnn_out  = gnn_temp(edge_index=edge_index, edge_weight=edge_weight)
            gnn_embs = gnn_out.last_hidden_state.float().detach()   # [18870, 256]
        # Register as a buffer → auto-moved to GPU by Lightning
        self.register_buffer("gnn_embs_cached", gnn_embs)
        del gnn_temp   # free memory
        print(f"[Node4-2-2-1-1-1] GNN embeddings cached: {gnn_embs.shape}, 0 trainable GNN params")

        # Build Ensembl ID → node index lookup
        node_names = json.loads((gnn_dir / "node_names.json").read_text())
        self._ensembl_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(node_names)
        }

        # ----------------------------------------------------------------
        # Precompute top-K PPI neighbors from edge_index and edge_weight
        # Build neighbor lookup table for NeighborhoodAttentionModule
        # ----------------------------------------------------------------
        print(f"[Node4-2-2-1-1-1] Precomputing top-{hp.nb_attn_k} PPI neighbors...")
        N_nodes = gnn_embs.shape[0]
        K = hp.nb_attn_k

        # Build adjacency: for each node, collect (neighbor_idx, weight) pairs
        src_nodes = edge_index[0].tolist()
        dst_nodes = edge_index[1].tolist()
        ew = edge_weight.tolist() if edge_weight is not None else [1.0] * len(src_nodes)

        adj: Dict[int, List] = defaultdict(list)
        for s, d, w in zip(src_nodes, dst_nodes, ew):
            adj[s].append((d, w))

        # Build top-K neighbor index tensor [N_nodes, K] and weight tensor [N_nodes, K]
        nb_indices = torch.zeros(N_nodes, K, dtype=torch.long)
        nb_weights = torch.zeros(N_nodes, K, dtype=torch.float32)
        for node_i in range(N_nodes):
            neighbors = adj.get(node_i, [])
            if len(neighbors) == 0:
                # No neighbors: self-loop with weight 1
                nb_indices[node_i] = node_i
                nb_weights[node_i] = 1.0
            else:
                # Sort by weight descending, take top-K
                neighbors_sorted = sorted(neighbors, key=lambda x: x[1], reverse=True)[:K]
                for j, (nidx, nw) in enumerate(neighbors_sorted):
                    nb_indices[node_i, j] = nidx
                    nb_weights[node_i, j] = nw
                # Pad remaining with self-loop (zero weight)
                for j in range(len(neighbors_sorted), K):
                    nb_indices[node_i, j] = node_i
                    nb_weights[node_i, j] = 0.0

        self.register_buffer("nb_indices_cached", nb_indices)   # [N_nodes, K]
        self.register_buffer("nb_weights_cached", nb_weights)   # [N_nodes, K]
        print(f"[Node4-2-2-1-1-1] Neighbor lookup built: {nb_indices.shape}")

        # ----------------------------------------------------------------
        # NeighborhoodAttentionModule (K=16, attn_dim=64)
        # Same proven configuration from node2-1-1-1 (F1=0.5059)
        # ----------------------------------------------------------------
        self.nb_attn = NeighborhoodAttentionModule(
            d_gnn=GNN_HIDDEN,
            K=hp.nb_attn_k,
            attn_dim=hp.nb_attn_dim,
        )

        # ----------------------------------------------------------------
        # Simple Concatenation Fusion: AIDO(640) || GNN(256) → 896-dim
        # KEY CHANGE: Replace GatedFusion with simple concatenation
        # Rationale: GatedFusion fails with LoRA-adapted AIDO.Cell embeddings
        # node2-1-1-1 (F1=0.5059) used this exact approach and achieved tree best
        # No learnable parameters needed for fusion — concat is sufficient
        # ----------------------------------------------------------------
        # No fusion module; concatenation is done in get_fused_emb()

        # ----------------------------------------------------------------
        # Two-layer Classification Head: 896 → 256 → 3*6640
        # Matches node2-1-1-1's proven architecture with 896-dim (640+256) input
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Dropout(hp.head_dropout),
            nn.Linear(CONCAT_DIM, HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(hp.head_dropout * 0.5),   # lighter second dropout
            nn.Linear(HEAD_HIDDEN, N_CLASSES * N_GENES),
        )

        # ----------------------------------------------------------------
        # GenePriorBias: per-gene, per-class bias initialized from log-priors
        # Gradient is zeroed during the first bias_warmup_epochs epochs.
        # Novel addition over node2-1-1-1: provides +0.0035 F1 in node4-2-1-1.
        # warmup=30 retained from parent for 270 effective learning epochs.
        # ----------------------------------------------------------------
        self.gene_prior_bias = GenePriorBias(
            n_classes=N_CLASSES,
            n_genes=N_GENES,
            bias_warmup_epochs=hp.bias_warmup_epochs,
        )

        # Class weights for weighted CE loss
        self.register_buffer("class_weights", get_class_weights())

        # Accumulators for validation / test
        self._val_preds:     List[torch.Tensor] = []
        self._val_tgts:      List[torch.Tensor] = []
        self._val_idx:       List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []

    # ---- GNN index lookup ----
    def _get_gnn_indices(self, pert_ids: List[str], device: torch.device) -> torch.Tensor:
        """Look up STRING_GNN node indices for a batch of Ensembl gene IDs."""
        indices = [self._ensembl_to_idx.get(pid, 0) for pid in pert_ids]
        return torch.tensor(indices, dtype=torch.long, device=device)

    # ---- Embedding computation ----
    def get_fused_emb(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids:       List[str],
    ) -> torch.Tensor:
        """Compute fused embedding: AIDO.Cell + cached GNN (with NbAttn) → simple concat."""
        device = input_ids.device

        # 1. AIDO.Cell-100M → summary token at position 19264
        #    input_ids: [B, 19264] float32 (AIDO.Cell-style)
        #    last_hidden_state: [B, 19266, 640]
        #    summary token is at position 19264 (the first appended summary token)
        aido_out = self.aido(input_ids=input_ids, attention_mask=attention_mask)
        # Use summary token at position 19264 as perturbation embedding
        aido_emb = aido_out.last_hidden_state.float()[:, 19264, :]  # [B, 640]

        # 2. STRING_GNN cached embeddings → lookup perturbed gene embedding
        node_indices = self._get_gnn_indices(pert_ids, device)
        gnn_emb = self.gnn_embs_cached[node_indices]               # [B, 256]

        # 3. Neighborhood attention: enrich GNN embedding with K=16 PPI context
        node_nb_indices = self.nb_indices_cached[node_indices]   # [B, K]
        node_nb_weights = self.nb_weights_cached[node_indices]   # [B, K]
        gnn_emb_enriched = self.nb_attn(
            center_emb=gnn_emb,
            all_embs=self.gnn_embs_cached,
            neighbor_indices=node_nb_indices,
            neighbor_weights=node_nb_weights,
        )                                                          # [B, 256]

        # 4. Simple concatenation → [B, 896]
        # KEY CHANGE: Replace GatedFusion with simple concat
        # This directly mirrors node2-1-1-1 (F1=0.5059) which also used concat
        return torch.cat([aido_emb, gnn_emb_enriched], dim=-1)    # [B, 896]

    # ---- Forward ----
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids:       List[str],
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        fused  = self.get_fused_emb(input_ids, attention_mask, pert_ids)
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        logits = self.gene_prior_bias(logits, self.current_epoch)
        return logits

    # ---- Loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted CE with label smoothing."""
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, C]
            targets.reshape(-1),                       # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ---- Gradient control for bias warmup ----
    def on_before_optimizer_step(self, optimizer) -> None:
        """Zero out the GenePriorBias gradient for the first bias_warmup_epochs epochs."""
        if self.current_epoch < self.hparams.bias_warmup_epochs:
            if self.gene_prior_bias.bias.grad is not None:
                self.gene_prior_bias.bias.grad.zero_()

    # ---- Training ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pert_ids       = batch["pert_id"]
        labels         = batch["labels"]
        B = input_ids.shape[0]

        # Compute fused embeddings
        fused = self.get_fused_emb(input_ids, attention_mask, pert_ids)

        # Mixup augmentation on fused embeddings
        if self.hparams.mixup_alpha > 0.0 and B > 1 and self.training:
            lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
            perm = torch.randperm(B, device=fused.device)
            fused_mix = lam * fused + (1 - lam) * fused[perm]
            logits_raw = self.head(fused_mix).view(B, N_CLASSES, N_GENES)
            logits = self.gene_prior_bias(logits_raw, self.current_epoch)
            loss = lam * self._loss(logits, labels) + (1 - lam) * self._loss(logits, labels[perm])
        else:
            logits_raw = self.head(fused).view(B, N_CLASSES, N_GENES)
            logits = self.gene_prior_bias(logits_raw, self.current_epoch)
            loss = self._loss(logits, labels)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    # ---- Validation ----
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits.float(), dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        local_idx   = torch.cat(self._val_idx,   0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        # Deduplicate (DDP may duplicate samples at epoch boundaries)
        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([
            torch.ones(1, dtype=torch.bool, device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    # ---- Test ----
    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)

        # Gather predictions from all ranks
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        if is_dist:
            all_preds    = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
            all_pert_ids = [None] * self.trainer.world_size
            all_symbols  = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(all_pert_ids, self._test_pert_ids)
            torch.distributed.all_gather_object(all_symbols,  self._test_symbols)
            flat_pids = [p for rank_pids in all_pert_ids for p in rank_pids]
            flat_syms = [s for rank_syms in all_symbols  for s in rank_syms]
        else:
            all_preds = local_preds
            flat_pids = self._test_pert_ids
            flat_syms = self._test_symbols

        if self.trainer.is_global_zero:
            # Deduplicate predictions (DDP may produce duplicates)
            seen = {}
            for i in range(all_preds.shape[0]):
                pid = flat_pids[i]
                if pid not in seen:
                    seen[pid] = (flat_syms[i], all_preds[i])

            rows = []
            for pid, (sym, pred) in seen.items():
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(pred.float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node4-2-2-1-1-1] Saved {len(rows)} test predictions → {out_dir}/test_predictions.tsv")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    # ---- Checkpoint: save only trainable params + all buffers ----
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
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = (100 * trained / total) if total > 0 else 0.0
        print(f"[Node4-2-2-1-1-1] Checkpoint: {trained:,}/{total:,} params ({pct:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer with WarmupCosine LR scheduler ----
    def configure_optimizers(self):
        hp = self.hparams
        # Only optimize parameters with requires_grad (frozen GNN excluded)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable_params, lr=hp.lr, weight_decay=hp.weight_decay
        )

        def lr_lambda(epoch: int) -> float:
            """Linear warmup then cosine decay with tighter LR floor (min_lr_ratio=0.05)."""
            if epoch < hp.warmup_epochs:
                # Linear warmup from 0 to 1
                return max(1e-8, epoch / max(1, hp.warmup_epochs))
            # Cosine decay from 1 down to min_lr_ratio
            progress = (epoch - hp.warmup_epochs) / max(1, hp.max_epochs - hp.warmup_epochs)
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return hp.min_lr_ratio + (1.0 - hp.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node4-2-2-1-1-1 – AIDO.Cell-100M (LoRA r=16) + STRING_GNN (frozen) + NbAttn(K=16) + SimpleCat + GenePriorBias"
    )
    parser.add_argument("--micro-batch-size",      type=int,   default=4)
    parser.add_argument("--global-batch-size",     type=int,   default=32)
    parser.add_argument("--max-epochs",            type=int,   default=300)
    parser.add_argument("--lr",                    type=float, default=1e-4)
    parser.add_argument("--weight-decay",          type=float, default=2e-2)
    parser.add_argument("--lora-r",                type=int,   default=16,
                        dest="lora_r")
    parser.add_argument("--lora-alpha",            type=int,   default=32,
                        dest="lora_alpha")
    parser.add_argument("--lora-dropout",          type=float, default=0.05,
                        dest="lora_dropout")
    parser.add_argument("--head-dropout",          type=float, default=0.5)
    parser.add_argument("--warmup-epochs",         type=int,   default=10)
    parser.add_argument("--min-lr-ratio",          type=float, default=0.05)
    parser.add_argument("--mixup-alpha",           type=float, default=0.2)
    parser.add_argument("--label-smoothing",       type=float, default=0.1)
    parser.add_argument("--bias-warmup-epochs",    type=int,   default=30,
                        dest="bias_warmup_epochs")
    parser.add_argument("--nb-attn-k",             type=int,   default=16,
                        dest="nb_attn_k")
    parser.add_argument("--nb-attn-dim",           type=int,   default=64,
                        dest="nb_attn_dim")
    parser.add_argument("--num-workers",           type=int,   default=4)
    parser.add_argument("--val-check-interval",    type=float, default=1.0,
                        dest="val_check_interval")
    parser.add_argument("--debug-max-step",        type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",          action="store_true", dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = lim_val = lim_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = FusionDEGModel(
        lora_r              = args.lora_r,
        lora_alpha          = args.lora_alpha,
        lora_dropout        = args.lora_dropout,
        head_dropout        = args.head_dropout,
        lr                  = args.lr,
        weight_decay        = args.weight_decay,
        warmup_epochs       = args.warmup_epochs,
        max_epochs          = args.max_epochs,
        min_lr_ratio        = args.min_lr_ratio,
        mixup_alpha         = args.mixup_alpha,
        label_smoothing     = args.label_smoothing,
        bias_warmup_epochs  = args.bias_warmup_epochs,
        nb_attn_k           = args.nb_attn_k,
        nb_attn_dim         = args.nb_attn_dim,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
        auto_insert_metric_name=False,  # avoids "/" in filename issue
    )
    # Patience=20: tighter than parent's 40 (parent ran all 200 epochs without firing)
    es_cb  = EarlyStopping(monitor="val/f1", mode="max", patience=20, min_delta=1e-4)
    lr_cb  = LearningRateMonitor(logging_interval="epoch")
    pg_cb  = TQDMProgressBar(refresh_rate=10)

    # No SWA — use best checkpoint directly to avoid EarlyStopping interaction
    callbacks_list = [ckpt_cb, es_cb, lr_cb, pg_cb]

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

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
        val_check_interval      = args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
        num_sanity_val_steps    = 2,
        callbacks               = callbacks_list,
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    # Use best checkpoint for test evaluation (no SWA averaging)
    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node4-2-2-1-1-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
