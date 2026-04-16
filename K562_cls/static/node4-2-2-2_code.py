"""Node 4-2-2-2: AIDO.Cell-100M (LoRA r=8) + STRING_GNN (frozen, cached, 2-head NbAttn K=16)
             + Simple Concatenation + GenePriorBias (warmup=20) + 2-layer head

Strategy (distinct from sibling node4-2-2-1 which uses scFoundation + GatedFusion):
- AIDO.Cell-100M: LoRA r=8 (0.55M trainable params) → summary token → [B, 640]
  The proven best architecture in the entire MCTS tree uses AIDO.Cell (node2-1-1-1-1-1, F1=0.5128)
  rather than scFoundation, breaking the ~0.487 scFoundation ceiling.
- STRING_GNN: FULLY FROZEN. Embeddings precomputed once in setup() and cached as a buffer.
  2-head NeighborhoodAttentionModule (K=16, attn_dim=64) enriches center gene embedding with
  top-16 PPI neighbors, proven in node2-1-1-1-1-1 (+0.0524 F1 gain over parent).
- Simple Concatenation: [B, 640] concat [B, 256] → [B, 896]
  CRITICAL: GatedFusion fails with AIDO.Cell (node4-2-2-1-1: 0.4738 vs simple concat: 0.5128).
  Simple concatenation handles the heterogeneous embedding spaces better.
- GenePriorBias: per-gene, per-class log-prior bias [3, 6640] initialized from training
  class frequencies with 20-epoch gradient warmup. Shorter warmup than siblings (warmup=30)
  because AIDO.Cell converges faster and needs fewer epochs for backbone stabilization.
- Loss: Weighted CrossEntropyLoss + label_smoothing=0.05 (from proven node2 lineage)
- Mixup augmentation on concatenated embeddings (alpha=0.2)
- LR: WarmupCosine schedule, warmup_epochs=5, min_lr_ratio=0.05
- No SWA: Consistently problematic due to EarlyStopping interaction in this lineage
- max_epochs=200, patience=20, weight_decay=2e-2

Key differences from sibling node4-2-2-1:
  node4-2-2-1: scFoundation(6L) + GatedFusion + GenePriorBias(warmup=30) + NbAttn(K=16)
              Test F1=0.4867 (scFoundation fusion ceiling ~0.487)
  node4-2-2-2: AIDO.Cell-100M LoRA + Simple Concat + GenePriorBias(warmup=20) + 2-head NbAttn(K=16)
              Target F1 > 0.500 (AIDO.Cell+STRING ceiling ~0.513+)

Architecture inspiration: node2-1-1-1-1-1 (F1=0.5128) + GenePriorBias as novel addition.
"""

from __future__ import annotations

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES     = 6640
N_CLASSES   = 3
AIDO_HIDDEN = 640    # AIDO.Cell-100M hidden size
GNN_HIDDEN  = 256    # STRING_GNN hidden size
CONCAT_DIM  = AIDO_HIDDEN + GNN_HIDDEN  # = 896
HEAD_HIDDEN = 256    # Two-layer head intermediate dimension

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
# GenePriorBias Module
# ---------------------------------------------------------------------------
class GenePriorBias(nn.Module):
    """Per-gene, per-class log-prior bias.

    Initialized from training class frequency statistics and activated only
    after bias_warmup_epochs epochs (gradient zeroed before that point).

    With AIDO.Cell converging faster, a shorter warmup (20 epochs) is used
    to give the bias more effective learning time throughout training.

    Proven: +0.0035 F1 in node4-2-1-1 and +0.0068 in node1-1-1-1-2.
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        n_genes: int = N_GENES,
        bias_warmup_epochs: int = 20,
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
# Neighborhood Attention Module (2-head)
# Proven in node2-1-1-1-1-1 (F1=0.5128) via STRING_GNN lineage
# ---------------------------------------------------------------------------
class NeighborhoodAttentionModule(nn.Module):
    """Multi-head PPI neighborhood attention aggregator.

    Enriches the center gene's embedding by attending over its top-K PPI
    neighbors, using STRING confidence scores as attention priors.

    The 2-head configuration is proven in node2-1-1-1-1-1 (+0.0524 F1 over
    single-node STRING_GNN, producing the tree's best F1=0.5128).

    Args:
        embed_dim: dimension of GNN embeddings (256)
        n_heads:   number of attention heads (2)
        attn_dim:  per-head projection dimension (64, total 128)
        K:         top-K PPI neighbors to aggregate (16)
    """

    def __init__(
        self,
        embed_dim: int = GNN_HIDDEN,
        n_heads:   int = 2,
        attn_dim:  int = 64,
        K:         int = 16,
    ) -> None:
        super().__init__()
        self.n_heads  = n_heads
        self.attn_dim = attn_dim
        self.K        = K
        total_dim     = n_heads * attn_dim  # 128

        # Multi-head query and key projections
        self.query_proj = nn.Linear(embed_dim, total_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, total_dim, bias=False)
        # Gate: blend center vs. neighborhood-aggregated embedding
        self.gate_linear = nn.Linear(embed_dim * 2, embed_dim, bias=True)

    def forward(
        self,
        center_emb:  torch.Tensor,  # [B, D]
        neighbor_emb: torch.Tensor, # [B, K, D]
        neighbor_wts: torch.Tensor, # [B, K]  STRING confidence priors
    ) -> torch.Tensor:
        """
        Returns:
            [B, D] enriched gene embedding.
        """
        B, D = center_emb.shape
        K    = neighbor_emb.shape[1]

        # Project query (center) and keys (neighbors) for multi-head attention
        q = self.query_proj(center_emb).view(B, self.n_heads, self.attn_dim)    # [B, H, d]
        k = self.key_proj(neighbor_emb).view(B, K, self.n_heads, self.attn_dim) # [B, K, H, d]

        # Attention scores: [B, H, K]
        # q: [B, H, d] → [B, H, 1, d]; k: [B, K, H, d] → [B, H, K, d]
        q = q.unsqueeze(2)                   # [B, H, 1, d]
        k = k.permute(0, 2, 1, 3)           # [B, H, K, d]
        scores = (q * k).sum(-1) / math.sqrt(self.attn_dim)  # [B, H, K]

        # Incorporate STRING confidence prior (log-softmax scaled)
        # neighbor_wts: [B, K] → add as log-prior to attention scores
        prior = torch.log(neighbor_wts.unsqueeze(1).clamp(min=1e-8))  # [B, 1, K]
        scores = scores + prior                                         # [B, H, K]

        # Replace zero-weight neighbors with -inf so they contribute nothing
        zero_mask = (neighbor_wts == 0).unsqueeze(1)  # [B, 1, K]
        scores = scores.masked_fill(zero_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, K]

        # Weighted sum over neighbors: multi-head → concat → mean over heads
        # [B, H, K] × [B, 1, K, D] → sum over K → [B, H, D]
        neighbor_emb_expanded = neighbor_emb.unsqueeze(1)  # [B, 1, K, D]
        attn_weights_expanded = attn_weights.unsqueeze(-1) # [B, H, K, 1]
        agg = (attn_weights_expanded * neighbor_emb_expanded).sum(dim=2)  # [B, H, D]
        agg = agg.mean(dim=1)  # [B, D] (average across heads)

        # Gated combination: gate controls blend of center vs. aggregated
        gate_input = torch.cat([center_emb, agg], dim=-1)  # [B, 2D]
        gate = torch.sigmoid(self.gate_linear(gate_input))  # [B, D]
        enriched = gate * center_emb + (1.0 - gate) * agg  # [B, D]
        return enriched


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

    AIDO.Cell expects expression dicts with gene_ids (Ensembl IDs) and
    expression values. For perturbation identity, we provide the perturbed
    gene at expression=1.0 (all other genes are -1.0/missing).
    """

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        # AIDO.Cell: provide perturbed gene at expression=1.0;
        # all 19,264 - 1 other genes filled with -1.0 (missing) by tokenizer
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
    def __init__(self, batch_size: int = 8, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer   = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Rank-0 downloads tokenizer first, others wait
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        self.train_ds = DEGDataset(pd.read_csv(TRAIN_TSV, sep="\t"))
        self.val_ds   = DEGDataset(pd.read_csv(VAL_TSV,   sep="\t"))
        self.test_ds  = DEGDataset(pd.read_csv(TEST_TSV,  sep="\t"))

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate_aido(self.tokenizer),
        )

    def train_dataloader(self) -> DataLoader: return self._loader(self.train_ds, True)
    def val_dataloader(self)   -> DataLoader: return self._loader(self.val_ds,   False)
    def test_dataloader(self)  -> DataLoader: return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCellDEGModel(pl.LightningModule):
    """AIDO.Cell-100M (LoRA r=8) + STRING_GNN (frozen, 2-head NbAttn K=16)
       + Simple Concatenation + GenePriorBias (warmup=20) + 2-layer head

    Improvements over scFoundation lineage (parent/sibling):
    1. AIDO.Cell-100M: breaks the ~0.487 scFoundation ceiling. The tree's best
       node (node2-1-1-1-1-1, F1=0.5128) uses AIDO.Cell + STRING_GNN with
       simple concatenation.
    2. 2-head NeighborhoodAttention (K=16): enriches STRING_GNN embedding with
       PPI neighbor context. Proven in node2-1-1-1-1-1 as crucial for
       breaking the STRING_GNN-only ceiling.
    3. Simple concatenation (NOT GatedFusion): GatedFusion fails with AIDO.Cell
       (node4-2-2-1-1: 0.4738). Simple concat handles heterogeneous embedding
       spaces better.
    4. GenePriorBias (warmup=20): per-gene class bias initialized from log-priors.
       Proven +0.0035-0.0068 F1 in multiple scFoundation nodes. warmup=20 vs 30
       in the sibling because AIDO.Cell stabilizes faster.
    5. No SWA: consistently problematic due to EarlyStopping interaction.
    """

    def __init__(
        self,
        lora_r: int            = 8,
        head_dropout: float    = 0.5,
        lr: float              = 1e-4,
        weight_decay: float    = 2e-2,
        warmup_epochs: int     = 5,
        max_epochs: int        = 200,
        min_lr_ratio: float    = 0.05,
        mixup_alpha: float     = 0.2,
        label_smoothing: float = 0.05,
        bias_warmup_epochs: int = 20,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        gnn_dir = Path(GNN_MODEL_DIR)

        # ----------------------------------------------------------------
        # AIDO.Cell-100M backbone with LoRA r=8
        # Proven configuration from node2-1-1-1-1-1 (F1=0.5128)
        # ----------------------------------------------------------------
        aido_base = AutoModel.from_pretrained(
            AIDO_MODEL_DIR,
            trust_remote_code=True,
        ).to(torch.bfloat16)
        aido_base.config.use_cache = False
        aido_base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Fix AIDO.Cell + PEFT compatibility:
        # PEFT's get_peft_model() calls model.enable_input_require_grads() which
        # internally calls model.get_input_embeddings(). AIDO.Cell's GeneEmbedding
        # raises NotImplementedError("Not Implemented Yet") for this method.
        # Patch both methods before applying LoRA so PEFT can set up correctly.
        _orig_get_ie = getattr(aido_base, "get_input_embeddings", None)
        _orig_enable = getattr(aido_base, "enable_input_require_grads", None)
        aido_base.get_input_embeddings = lambda: aido_base.bert.gene_embedding
        aido_base.enable_input_require_grads = lambda: None
        if _orig_get_ie is not None:
            aido_base._orig_get_input_embeddings = _orig_get_ie
        if _orig_enable is not None:
            aido_base._orig_enable_input_require_grads = _orig_enable

        # Apply LoRA r=8 to Q/K/V attention weights
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_r * 2,   # alpha = 2×r (standard setting)
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
        )
        self.aido = get_peft_model(aido_base, lora_cfg)

        # Cast LoRA params to float32 for stable optimization
        for name, param in self.aido.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        n_trainable_aido = sum(p.numel() for p in self.aido.parameters() if p.requires_grad)
        n_total_aido     = sum(p.numel() for p in self.aido.parameters())
        print(f"[Node4-2-2-2] AIDO.Cell-100M: {n_trainable_aido:,}/{n_total_aido:,} trainable params")

        # ----------------------------------------------------------------
        # STRING_GNN: fully frozen, embeddings precomputed and cached
        # ----------------------------------------------------------------
        print("[Node4-2-2-2] Precomputing STRING_GNN embeddings (frozen)...")
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
        del gnn_temp   # free memory — GNN is no longer needed
        print(f"[Node4-2-2-2] GNN embeddings cached: {gnn_embs.shape}, 0 trainable GNN params")

        # Build Ensembl ID → node index lookup
        node_names = json.loads((gnn_dir / "node_names.json").read_text())
        self._ensembl_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(node_names)
        }

        # ----------------------------------------------------------------
        # Build neighbor lookup table for NeighborhoodAttentionModule
        # Precompute top-K neighbors and their STRING confidence weights
        # ----------------------------------------------------------------
        print("[Node4-2-2-2] Building PPI neighbor lookup table (K=16)...")
        K = 16
        n_nodes = gnn_embs.shape[0]
        # Build adjacency: node → list of (neighbor_idx, weight)
        adj: List[List] = [[] for _ in range(n_nodes)]
        ei = edge_index  # [2, E]
        ew = edge_weight  # [E]
        for i in range(ei.shape[1]):
            src, dst = int(ei[0, i]), int(ei[1, i])
            w = float(ew[i]) if ew is not None else 1.0
            adj[src].append((dst, w))

        # Precompute top-K neighbor indices and weights as buffers
        nb_indices = torch.zeros(n_nodes, K, dtype=torch.long)
        nb_weights = torch.zeros(n_nodes, K, dtype=torch.float32)
        for node_i in range(n_nodes):
            neighbors = sorted(adj[node_i], key=lambda x: -x[1])[:K]
            for j, (nb_idx, nb_w) in enumerate(neighbors):
                nb_indices[node_i, j] = nb_idx
                nb_weights[node_i, j] = nb_w
        self.register_buffer("nb_indices", nb_indices)  # [18870, K]
        self.register_buffer("nb_weights", nb_weights)  # [18870, K]
        print(f"[Node4-2-2-2] Neighbor lookup: {nb_indices.shape}")

        # ----------------------------------------------------------------
        # 2-head Neighborhood Attention Module
        # Proven in node2-1-1-1-1-1 (F1=0.5128) - 2-head is better than 1-head
        # ----------------------------------------------------------------
        self.nb_attn = NeighborhoodAttentionModule(
            embed_dim=GNN_HIDDEN, n_heads=2, attn_dim=64, K=16
        )

        # ----------------------------------------------------------------
        # Two-layer Classification Head: 896 → 256 → 3*6640
        # Follows proven architecture from node2-1-1-1-1-1
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Dropout(hp.head_dropout),
            nn.Linear(CONCAT_DIM, HEAD_HIDDEN),
            nn.LayerNorm(HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(hp.head_dropout * 0.5),   # lighter second dropout
            nn.Linear(HEAD_HIDDEN, N_CLASSES * N_GENES),
        )

        # ----------------------------------------------------------------
        # GenePriorBias: per-gene, per-class bias initialized from log-priors
        # warmup=20 epochs (shorter than sibling's 30 because AIDO.Cell
        # converges faster; provides more effective bias learning time)
        # Proven: +0.0035-0.0068 F1 across multiple lineages
        # ----------------------------------------------------------------
        self.gene_prior_bias = GenePriorBias(
            n_classes=N_CLASSES,
            n_genes=N_GENES,
            bias_warmup_epochs=hp.bias_warmup_epochs,
        )

        # Cast all trainable head/attn/bias params to float32
        for m in [self.nb_attn, self.head, self.gene_prior_bias]:
            for p in m.parameters():
                if p.is_floating_point():
                    p.data = p.data.float()

        # Class weights for weighted CE loss
        self.register_buffer("class_weights", get_class_weights())

        # Accumulators for validation / test
        self._val_preds:     List[torch.Tensor] = []
        self._val_tgts:      List[torch.Tensor] = []
        self._val_idx:       List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []

    # ---- GNN neighbor lookup ----
    def _get_neighbor_embs(
        self, pert_ids: List[str], device: torch.device
    ):
        """Look up top-K STRING_GNN PPI neighbors for a batch of Ensembl gene IDs.

        Returns:
            center_emb:   [B, D] center gene embedding
            neighbor_emb: [B, K, D] neighbor embeddings
            neighbor_wts: [B, K] STRING confidence weights
        """
        indices = [self._ensembl_to_idx.get(pid, 0) for pid in pert_ids]
        idx_t   = torch.tensor(indices, dtype=torch.long, device=device)

        center_emb   = self.gnn_embs_cached[idx_t]           # [B, D]
        nb_idx       = self.nb_indices[idx_t]                 # [B, K]
        nb_wts       = self.nb_weights[idx_t]                 # [B, K]

        # Flatten for batch lookup: [B*K] → [B, K, D]
        B, K = nb_idx.shape
        flat_nb_idx  = nb_idx.view(-1)                        # [B*K]
        neighbor_emb = self.gnn_embs_cached[flat_nb_idx].view(B, K, -1)  # [B, K, D]

        return center_emb, neighbor_emb, nb_wts

    # ---- Embedding computation ----
    def get_concat_emb(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids:       List[str],
    ) -> torch.Tensor:
        """Compute concatenated embedding: AIDO.Cell summary token + enriched STRING_GNN.

        The summary token is at position 19264 (the first of two summary tokens
        appended by AIDO.Cell's _prepare_inputs). This is the proven approach
        from node2-1 and the node2-1-1-1-1-1 best-in-tree architecture.
        """
        device = input_ids.device

        # 1. AIDO.Cell → summary token at position 19264
        # input_ids are float32 (expression values, not token indices)
        aido_out = self.aido(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: [B, 19266, 640] — positions 19264-19265 are summary tokens
        aido_emb = aido_out.last_hidden_state[:, 19264, :].float()  # [B, 640] summary token

        # 2. STRING_GNN cached embeddings + 2-head neighborhood attention
        center_emb, neighbor_emb, neighbor_wts = self._get_neighbor_embs(pert_ids, device)
        # Cast to float32 for attention computation
        center_emb   = center_emb.float()
        neighbor_emb = neighbor_emb.float()
        neighbor_wts = neighbor_wts.float()
        enriched_gnn = self.nb_attn(center_emb, neighbor_emb, neighbor_wts)  # [B, 256]

        # 3. Simple concatenation: [B, 640] || [B, 256] → [B, 896]
        return torch.cat([aido_emb, enriched_gnn], dim=-1)

    # ---- Forward ----
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids:       List[str],
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        concat_emb = self.get_concat_emb(input_ids, attention_mask, pert_ids)
        logits = self.head(concat_emb).view(B, N_CLASSES, N_GENES)
        # Apply per-gene bias (active at all times during forward;
        # gradient is controlled in on_before_optimizer_step)
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
        """Zero out the GenePriorBias gradient for the first bias_warmup_epochs epochs.

        Prevents the bias from learning noisy signal before the backbone
        has had time to stabilize. AIDO.Cell converges faster, so warmup=20
        is sufficient (vs 30 in scFoundation lineage).
        """
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

        # Compute concatenated embeddings
        concat_emb = self.get_concat_emb(input_ids, attention_mask, pert_ids)

        # Mixup augmentation on concatenated embeddings
        if self.hparams.mixup_alpha > 0.0 and B > 1 and self.training:
            lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
            perm = torch.randperm(B, device=concat_emb.device)
            concat_mix = lam * concat_emb + (1 - lam) * concat_emb[perm]
            logits_raw = self.head(concat_mix).view(B, N_CLASSES, N_GENES)
            logits = self.gene_prior_bias(logits_raw, self.current_epoch)
            loss = lam * self._loss(logits, labels) + (1 - lam) * self._loss(logits, labels[perm])
        else:
            logits_raw = self.head(concat_emb).view(B, N_CLASSES, N_GENES)
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
        # Build mask on the same device as s_idx to avoid CPU/GPU mismatch in cat()
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
            print(f"[Node4-2-2-2] Saved {len(rows)} test predictions → {out_dir}/test_predictions.tsv")

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
        print(f"[Node4-2-2-2] Checkpoint: {trained:,}/{total:,} params ({pct:.2f}%)")
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
            """Linear warmup then cosine decay with min_lr_ratio floor."""
            if epoch < hp.warmup_epochs:
                return max(1e-8, epoch / max(1, hp.warmup_epochs))
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
        description="Node4-2-2-2: AIDO.Cell-100M LoRA + STRING_GNN 2-head NbAttn + GenePriorBias"
    )
    parser.add_argument("--micro-batch-size",      type=int,   default=8)
    parser.add_argument("--global-batch-size",     type=int,   default=64)
    parser.add_argument("--max-epochs",            type=int,   default=200)
    parser.add_argument("--lr",                    type=float, default=1e-4)
    parser.add_argument("--weight-decay",          type=float, default=2e-2)
    parser.add_argument("--lora-r",                type=int,   default=8,
                        dest="lora_r")
    parser.add_argument("--head-dropout",          type=float, default=0.5)
    parser.add_argument("--warmup-epochs",         type=int,   default=5)
    parser.add_argument("--min-lr-ratio",          type=float, default=0.05)
    parser.add_argument("--mixup-alpha",           type=float, default=0.2)
    parser.add_argument("--label-smoothing",       type=float, default=0.05)
    parser.add_argument("--bias-warmup-epochs",    type=int,   default=20,
                        dest="bias_warmup_epochs")
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
    model = AIDOCellDEGModel(
        lora_r             = args.lora_r,
        head_dropout       = args.head_dropout,
        lr                 = args.lr,
        weight_decay       = args.weight_decay,
        warmup_epochs      = args.warmup_epochs,
        max_epochs         = args.max_epochs,
        min_lr_ratio       = args.min_lr_ratio,
        mixup_alpha        = args.mixup_alpha,
        label_smoothing    = args.label_smoothing,
        bias_warmup_epochs = args.bias_warmup_epochs,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
        auto_insert_metric_name=False,  # avoids "/" in filename issue
    )
    # patience=20: node2-1-1-1-1-1 recommended 15-20; captures the late-improvement spikes
    # that AIDO.Cell is known for (best at epoch 77 in node2-1-1-1-1-1)
    es_cb  = EarlyStopping(monitor="val/f1", mode="max", patience=20, min_delta=1e-4)
    lr_cb  = LearningRateMonitor(logging_interval="epoch")
    pg_cb  = TQDMProgressBar(refresh_rate=10)

    callbacks_list = [ckpt_cb, es_cb, lr_cb, pg_cb]

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Use DDP with find_unused_parameters=True because LoRA + gradient checkpointing
    # may create unused parameters in AIDO.Cell during distributed training
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

    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node4-2-2-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
