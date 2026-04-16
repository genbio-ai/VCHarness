"""Node 3-1-2-1: STRING_GNN K=16 Neighborhood Attention + AIDO.Cell-10M Hybrid.

Key changes from parent node3-1-2 (test F1=0.4407):
1. Replace naive STRING_GNN embedding lookup with K=16 PPI neighborhood attention aggregation
   (the proven innovation from node1-1-1-1-1 that achieved F1=0.4846 in STRING-only lineage)
2. Replace SGDR warm restarts with standard warmup + cosine annealing (no disruptive restarts)
3. Reduce head_hidden from 512 to 256 with dropout increased from 0.3 to 0.4
4. Keep AIDO.Cell-10M QKV-only fine-tuning with Muon optimizer (proven)
5. Keep label-smoothed CE + class weights (proven safe, focal loss was catastrophic)

Architecture:
    AIDO.Cell-10M (QKV-only, Muon) → last 4 layer embeds at pert gene pos → concat → [B, 1024]
    STRING_GNN (frozen, pre-computed) → K=16 neighborhood attention → [B, 256]
    Fused: concat([AIDO 4-layer, STRING K=16]) = [B, 1280]
    Head: Linear(1280→256) + LN + GELU + Dropout(0.4) + Linear(256→19920)
    Loss: label-smoothed CE (eps=0.1) + sqrt-inverse-frequency class weights
    Schedule: linear warmup(5ep) + CosineAnnealingLR(T_max=100)

Parent node3-1-2:
    - STRING: naive lookup → F1=0.4407
This node:
    - STRING: K=16 neighborhood attention → target F1 > 0.45
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
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES    = 6640
N_CLASSES  = 3
AIDO_GENES = 19264
AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
HIDDEN_DIM = 256      # AIDO.Cell-10M hidden size
N_LAYERS   = 8        # AIDO.Cell-10M transformer layers
STRING_DIM = 256      # STRING_GNN embedding dimension

CLASS_FREQ = [0.0429, 0.9251, 0.0320]  # down, neutral, up (remapped 0,1,2)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights for class imbalance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute per-gene macro F1, matching the calc_metric.py evaluation logic."""
    y_hat       = preds.argmax(dim=1)
    G           = targets.shape[1]
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
        f1_c = torch.where(prec + rec > 0, 2*prec*rec/(prec+rec+1e-8), torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


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

    # Stable-sort by src (preserves descending weight order within each src)
    sort_by_src = torch.argsort(src_sorted, stable=True)
    src_final = src_sorted[sort_by_src]
    dst_final = dst_sorted[sort_by_src]
    wgt_final = wgt_sorted[sort_by_src]

    # Count edges per source node
    counts = torch.bincount(src_final, minlength=n_nodes)  # [n_nodes]

    neighbor_indices = torch.full((n_nodes, K), -1, dtype=torch.long)
    neighbor_weights = torch.zeros(n_nodes, K, dtype=torch.float32)

    start = 0
    for node_i in range(n_nodes):
        c = int(counts[node_i].item())
        if c > 0:
            n_k = min(K, c)
            neighbor_indices[node_i, :n_k] = dst_final[start:start + n_k]
            neighbor_weights[node_i, :n_k] = wgt_final[start:start + n_k]
        start += c

    return neighbor_indices, neighbor_weights


def load_string_gnn_embeddings_and_neighbors(K: int = 16):
    """Load frozen STRING_GNN embeddings and pre-compute K-nearest neighbors.

    Returns:
        emb_matrix: [18870, 256] float32 tensor with per-gene PPI embeddings
        name_to_idx: dict mapping Ensembl gene ID -> row index in emb_matrix
        neighbor_indices: [18870, K] long tensor with top-K neighbor indices
        neighbor_weights: [18870, K] float tensor with neighbor STRING confidence scores
    """
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(node_names)}

    # Load STRING_GNN and compute embeddings (run once, frozen)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True).to(device)
    gnn_model.eval()

    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location=device)
    edge_index  = graph["edge_index"].long()
    edge_weight = graph.get("edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.float().to(device)

    with torch.no_grad():
        outputs = gnn_model(edge_index=edge_index, edge_weight=edge_weight)
        emb_matrix = outputs.last_hidden_state.float().cpu()  # [18870, 256]

    n_nodes = emb_matrix.shape[0]

    # Pre-compute K-NN neighborhood
    print(f"[Node3-1-2-1] Pre-computing top-{K} PPI neighbors for {n_nodes} nodes...")
    neighbor_indices, neighbor_weights = precompute_neighborhood(
        edge_index.cpu(), edge_weight.cpu() if edge_weight is not None else torch.ones(edge_index.shape[1]),
        n_nodes, K=K
    )

    del gnn_model
    torch.cuda.empty_cache()

    return emb_matrix, name_to_idx, neighbor_indices, neighbor_weights


# ---------------------------------------------------------------------------
# Neighborhood Attention Module
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Center-context gated attention over top-K PPI neighbors.

    Key innovation from node1-1-1-1-1 (F1=0.4846) that pushed past the naive
    embedding lookup ceiling. For each perturbed gene, aggregates top-K STRING
    PPI neighbors using learned attention scores gated by edge confidence.

    Architecture:
        attn_proj: [center(256) + neighbor(256)] -> attn_dim(64) -> score(1)
        attention = softmax(edge_weight + attn_proj_score)   # [B, K]
        aggregated = attention @ neighbor_emb                # [B, 256]
        gate = sigmoid(gate_proj(center_emb))                # [B, 256]
        output = center_emb + gate * aggregated              # [B, 256]

    Proven hyperparameters: K=16, attn_dim=64
    - K=32 caused regression (information bottleneck, F1=0.4743 vs 0.4846)
    - attn_dim=32 was suboptimal (F1=0.4743 vs 0.4846)
    """

    def __init__(self, embed_dim: int = 256, attn_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim  = attn_dim

        # Attention projection: [center(256) + neighbor(256)] -> attn_dim -> 1
        self.attn_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, 1),
        )
        # Gating: center embedding -> gate vector
        self.gate_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        center_emb: torch.Tensor,         # [B, D]
        neighbor_emb: torch.Tensor,        # [B, K, D]
        neighbor_weights: torch.Tensor,    # [B, K] STRING edge confidence (0-1)
        neighbor_mask: torch.Tensor,       # [B, K] bool: True = valid neighbor
    ) -> torch.Tensor:
        """Returns aggregated embedding [B, D]."""
        B, K, D = neighbor_emb.shape

        # Expand center for pair-wise attention projection
        center_expanded = center_emb.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]
        pair_features = torch.cat([center_expanded, neighbor_emb], dim=-1)  # [B, K, 2D]

        # Learned attention scores + STRING edge confidence prior
        attn_scores = self.attn_proj(pair_features).squeeze(-1)  # [B, K]
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
# Dataset / DataModule
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        name_to_idx: Dict[str, int],
    ) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        # STRING_GNN node index for each sample (-1 means not in STRING)
        self.string_node_indices = torch.tensor(
            [name_to_idx.get(p, -1) for p in self.pert_ids], dtype=torch.long
        )

        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List[torch.Tensor]] = (
            [torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
             for row in df["label"].tolist()]
            if has_label else None
        )

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


def make_collate(tokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        input_ids  = tokenized["input_ids"]  # [B, 19264] float32
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),
            torch.zeros(len(batch), dtype=torch.long),
        )
        out: Dict[str, Any] = {
            "sample_idx":      torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":         pert_ids,
            "symbol":          symbols,
            "input_ids":       input_ids,
            "attention_mask":  tokenized["attention_mask"],
            "gene_positions":  gene_positions,
            "string_node_idx": torch.stack([b["string_node_idx"] for b in batch]),  # [B]
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return collate_fn


class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer   = None
        self.name_to_idx: Optional[Dict[str, int]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Load tokenizer (rank-0 first)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        # Load STRING_GNN mapping only (embeddings pre-computed in model.setup)
        if self.name_to_idx is None:
            node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
            self.name_to_idx = {name: i for i, name in enumerate(node_names)}

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        self.train_ds = DEGDataset(train_df, self.name_to_idx)
        self.val_ds   = DEGDataset(val_df,   self.name_to_idx)
        self.test_ds  = DEGDataset(test_df,  self.name_to_idx)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=make_collate(self.tokenizer))

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HybridNeighborhoodStringAIDOModel(pl.LightningModule):
    """Hybrid STRING_GNN K=16 Neighborhood Attention + AIDO.Cell-10M QKV model.

    Architecture:
    - AIDO.Cell-10M: QKV-only fine-tuning with Muon optimizer
      Extract last 4 transformer layer embeddings at perturbed gene position
      → [B, 4*256=1024]
    - STRING_GNN: frozen, pre-computed K=16 neighborhood attention aggregation
      → [B, 256] (richer than naive lookup due to PPI context aggregation)
    - Fusion: concat → [B, 1280]
    - Head: Linear(1280→256) + LN + GELU + Dropout(0.4) + Linear(256→N_CLASSES*N_GENES)
    - Loss: label-smoothed CE + sqrt-inverse-frequency class weights
    - Schedule: linear warmup(5ep) + CosineAnnealingLR(T_max=100)

    Key innovation over parent node3-1-2:
    STRING_GNN K=16 neighborhood attention replaces naive direct embedding lookup.
    Proven in node1-2 (F1=0.4769) and node1-1-1-1-1 (F1=0.4846) to significantly
    improve over direct lookup in the STRING-only lineage.
    """

    def __init__(
        self,
        fusion_layers: int  = 4,       # last N AIDO.Cell transformer layers to concatenate
        K: int              = 16,      # neighborhood size (proven best: K=16)
        attn_dim: int       = 64,      # attention dimension (proven best: 64)
        head_hidden: int    = 256,     # MLP head hidden dim (reduced from parent's 512)
        head_dropout: float = 0.4,     # dropout in head (increased from parent's 0.3)
        lr_muon: float      = 0.02,    # Muon LR for QKV weight matrices
        lr_adamw: float     = 3e-4,    # AdamW LR for head (slightly higher for smaller head)
        weight_decay: float = 1.5e-2,  # slightly lower than parent's 2e-2 for smaller head
        warmup_epochs: int  = 5,       # linear warmup epochs
        t_max: int          = 100,     # CosineAnnealingLR T_max
        label_smoothing: float = 0.1,  # label smoothing epsilon (same as parent)
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        hp = self.hparams

        # ---- Load AIDO.Cell-10M backbone ----
        self.aido_backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        self.aido_backbone = self.aido_backbone.to(torch.bfloat16)
        self.aido_backbone.config.use_cache = False
        self.aido_backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Enable FlashAttention to avoid OOM from full 19266x19266 attention matrix
        self.aido_backbone.config._use_flash_attention_2 = True

        # Share QKV weight tensors between flash_self and self.self
        for layer in self.aido_backbone.bert.encoder.layer:
            ss = layer.attention.flash_self  # BertSelfFlashAttention
            mm = layer.attention.self       # CellFoundationSelfAttention (regular)
            ss.query.weight = mm.query.weight
            ss.key.weight   = mm.key.weight
            ss.value.weight = mm.value.weight
            ss.query.bias   = mm.query.bias
            ss.key.bias     = mm.key.bias
            ss.value.bias   = mm.value.bias

        # Freeze all AIDO.Cell layers, then unfreeze only QKV weights
        for param in self.aido_backbone.parameters():
            param.requires_grad = False

        qkv_patterns = (
            "attention.self.query.weight",
            "attention.self.key.weight",
            "attention.self.value.weight",
        )
        for name, param in self.aido_backbone.named_parameters():
            if any(name.endswith(p) for p in qkv_patterns):
                param.requires_grad = True

        qkv_count = sum(p.numel() for p in self.aido_backbone.parameters() if p.requires_grad)
        total      = sum(p.numel() for p in self.aido_backbone.parameters())
        print(f"[Node3-1-2-1] AIDO trainable QKV params: {qkv_count:,} / {total:,}")

        # ---- Load STRING_GNN and pre-compute neighborhood ----
        # Embeddings stored as frozen buffers; neighborhood pre-computed once
        gnn_model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        gnn_model.eval()
        for p in gnn_model.parameters():
            p.requires_grad = False

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph["edge_index"].long()
        edge_weight = graph.get("edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.float()
        else:
            edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

        with torch.no_grad():
            gnn_out = gnn_model(edge_index=edge_index, edge_weight=edge_weight)
            node_emb = gnn_out.last_hidden_state.float().detach().cpu()  # [18870, 256]

        self.register_buffer("node_embeddings", node_emb)
        n_nodes = node_emb.shape[0]

        # Pre-compute top-K neighbors
        print(f"[Node3-1-2-1] Pre-computing top-{hp.K} PPI neighbors for {n_nodes} nodes...")
        nbr_idx, nbr_wgt = precompute_neighborhood(
            edge_index, edge_weight, n_nodes, K=hp.K
        )
        self.register_buffer("neighbor_indices", nbr_idx)  # [n_nodes, K]
        self.register_buffer("neighbor_weights", nbr_wgt)  # [n_nodes, K]

        del gnn_model, graph, edge_index, edge_weight, gnn_out, node_emb

        # ---- Learnable fallback for unknown pert_ids ----
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ---- Neighborhood Attention Module ----
        # K=16, attn_dim=64 — proven best config from node1-2 and node1-1-1-1-1
        self.neighborhood_attn = NeighborhoodAttentionAggregator(
            embed_dim=STRING_DIM,
            attn_dim=hp.attn_dim,
            dropout=0.0,
        )

        # ---- Head ----
        # Input: 4 * HIDDEN_DIM (AIDO concat) + STRING_DIM (GNN neighborhood)
        # = 4 * 256 + 256 = 1280-dim
        aido_fused_dim  = hp.fusion_layers * HIDDEN_DIM  # 4 * 256 = 1024
        total_fused_dim = aido_fused_dim + STRING_DIM    # 1024 + 256 = 1280

        self.head = nn.Sequential(
            nn.Linear(total_fused_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # Cast head parameters to float32 for stable optimization
        for p in self.head.parameters():
            p.data = p.data.float()

        # Cast neighborhood attention parameters to float32
        for p in self.neighborhood_attn.parameters():
            p.data = p.data.float()

        # Cast fallback embedding to float32
        for p in self.fallback_emb.parameters():
            p.data = p.data.float()

        # ---- Loss with class weights ----
        class_weights = get_class_weights()
        self.register_buffer("class_weights", class_weights)

        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_meta:  List[Tuple]        = []

    # ---- STRING_GNN neighborhood embedding lookup ----
    def _get_string_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Lookup pre-computed embeddings with K=16 PPI neighborhood attention.

        For known pert_ids: apply neighborhood attention aggregation.
        For unknown pert_ids: use learnable fallback embedding.

        Args:
            string_node_idx: [B] long tensor, -1 for pert_ids not in STRING.
        Returns:
            [B, STRING_DIM] float32 perturbation embeddings.
        """
        B = string_node_idx.shape[0]
        emb = torch.zeros(B, STRING_DIM,
                          dtype=torch.float32,
                          device=self.node_embeddings.device)
        known   = string_node_idx >= 0
        unknown = ~known

        if known.any():
            known_idx = string_node_idx[known]   # [K_known]

            # Center embeddings
            center = self.node_embeddings[known_idx].float()  # [K_known, 256]

            # Get pre-computed neighbor indices and weights
            nbr_idx = self.neighbor_indices[known_idx]   # [K_known, K]
            nbr_wgt = self.neighbor_weights[known_idx]   # [K_known, K]

            # Validity mask (non-padding)
            nbr_mask = nbr_idx >= 0  # [K_known, K]

            # Clamp to valid range for embedding lookup
            nbr_idx_clamped = nbr_idx.clamp(min=0)  # [K_known, K]

            # Lookup neighbor embeddings
            n_known = int(known.sum().item())
            K_neighbors = nbr_idx.shape[1]
            flat_nbr_idx = nbr_idx_clamped.view(-1)            # [K_known * K]
            flat_nbr_emb = self.node_embeddings[flat_nbr_idx].float()  # [K_known * K, 256]
            neighbor_emb = flat_nbr_emb.view(n_known, K_neighbors, STRING_DIM)  # [K_known, K, 256]

            # Zero out padding neighbor embeddings
            neighbor_emb = neighbor_emb * nbr_mask.unsqueeze(-1).float()

            # Apply neighborhood attention aggregation
            aggregated = self.neighborhood_attn(
                center_emb=center,
                neighbor_emb=neighbor_emb,
                neighbor_weights=nbr_wgt.float(),
                neighbor_mask=nbr_mask,
            )  # [K_known, 256]

            emb[known] = aggregated

        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(int(unknown.sum()), dtype=torch.long,
                            device=self.node_embeddings.device)
            ).float()
            emb[unknown] = fb

        return emb  # [B, 256]

    # ---- forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_positions: torch.Tensor,
        string_node_idx: torch.Tensor,
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        # AIDO.Cell forward pass with hidden states
        out = self.aido_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states: tuple of (N_LAYERS+1) tensors, each [B, AIDO_GENES+2, 256]
        hidden_states = out.hidden_states  # len = N_LAYERS + 1 = 9

        n = self.hparams.fusion_layers
        # Collect per-layer embeddings at the perturbed gene position (last 4 layers)
        layer_embs = []
        for i in range(n):
            hs = hidden_states[-(i + 1)]   # [B, AIDO_GENES+2, 256]
            ge = hs[torch.arange(B, device=hs.device), gene_positions, :].float()  # [B, 256]
            layer_embs.append(ge)

        # Concatenate last 4 layers: [B, 4*256=1024]
        aido_features = torch.cat(layer_embs, dim=-1)

        # STRING_GNN K=16 neighborhood-aggregated embedding: [B, 256]
        string_features = self._get_string_embeddings(string_node_idx)

        # Fused representation: [B, 1280]
        fused = torch.cat([aido_features, string_features], dim=-1)

        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return logits

    # ---- loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        flat_logits  = logits.permute(0, 2, 1).reshape(-1, C)
        flat_targets = targets.reshape(-1)
        return F.cross_entropy(
            flat_logits, flat_targets,
            weight=self.class_weights.to(flat_logits.device),
            label_smoothing=self.hparams.label_smoothing,
        )

    # ---- steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"],
                      batch["gene_positions"], batch["string_node_idx"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"],
                      batch["gene_positions"], batch["string_node_idx"])
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

        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"],
                      batch["gene_positions"], batch["string_node_idx"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        for i, (pid, sym) in enumerate(zip(batch["pert_id"], batch["symbol"])):
            self._test_meta.append((pid, sym, batch["sample_idx"][i].item()))
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = torch.cat(self._test_preds, 0)   # [local_N, 3, 6640]
        self._test_preds.clear()

        # Determine if DDP is active (WORLD_SIZE > 1)
        is_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
        is_rank_zero = int(os.environ.get("RANK", "0")) == 0

        out_dir = Path(__file__).parent / "run"
        out_dir.mkdir(parents=True, exist_ok=True)

        if is_ddp:
            # ---- DDP MODE ----
            my_rank = int(os.environ.get("RANK", "0"))
            tmp_file = out_dir / f".test_preds_rank{my_rank}.tmp"

            # Build local metadata (pid, sym, gidx) from _test_meta which has tuples
            local_meta: List[tuple] = [(pid, sym, sidx) for pid, sym, sidx in self._test_meta]
            self._test_meta.clear()

            # Encode predictions as base64 to avoid TSV/CSV delimiter issues
            import base64 as b64_mod
            local_rows = []
            for i, meta in enumerate(local_meta):
                pid, sym, gidx = meta
                pred_bytes = local_preds[i].cpu().numpy().tobytes()
                pred_b64 = b64_mod.b64encode(pred_bytes).decode("ascii")
                local_rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": pred_b64,
                    "_gidx":      gidx,
                })
            pd.DataFrame(local_rows).to_csv(tmp_file, sep="\t", index=False)

            # Barrier: synchronize so both ranks finish writing before rank 0 reads
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                dummy = torch.zeros(world_size, dtype=torch.long, device=local_preds.device)
                dummy[my_rank] = 1
                torch.distributed.all_reduce(dummy, op=torch.distributed.ReduceOp.SUM)

            if not is_rank_zero:
                return

            # Rank 0: read all temp files, decode base64, deduplicate, write final TSV
            all_rows = []
            if torch.distributed.is_initialized():
                world_size_m = torch.distributed.get_world_size()
            else:
                world_size_m = 1

            for r in range(world_size_m):
                tf = out_dir / f".test_preds_rank{r}.tmp"
                if tf.exists():
                    df_r = pd.read_csv(tf, sep="\t", dtype={"_gidx": int})
                    all_rows.append(df_r)
                    try:
                        tf.unlink()
                    except OSError:
                        pass

            if all_rows:
                merged = pd.concat(all_rows, ignore_index=True)
                merged = merged.sort_values("_gidx").drop_duplicates(subset=["_gidx"], keep="first")
                merged = merged.drop(columns=["_gidx"])
                # Decode base64 back to nested list JSON for calc_metric.py
                import base64 as b64_mod
                final_rows = []
                for _, row in merged.iterrows():
                    pred_bytes = b64_mod.b64decode(row["prediction"])
                    pred_arr = np.frombuffer(pred_bytes, dtype=np.float32).reshape(3, 6640)
                    final_rows.append({
                        "idx":        row["idx"],
                        "input":      row["input"],
                        "prediction": json.dumps(pred_arr.tolist()),
                    })
            else:
                final_rows = []

            pd.DataFrame(final_rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node3-1-2-1] Saved {len(final_rows)} test predictions.")
        else:
            # ---- SINGLE-DEVICE MODE ----
            rows = []
            for i, meta in enumerate(self._test_meta):
                pid, sym, _ = meta
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(local_preds[i].cpu().numpy().tolist()),
                })
            self._test_meta.clear()
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node3-1-2-1] Saved {len(rows)} test predictions.")

    # ---- checkpoint: save only trainable parameters ----
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
        self.print(f"Checkpoint: {trained}/{total} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- optimizer: Muon for QKV weight matrices, AdamW for everything else ----
    def configure_optimizers(self):
        hp = self.hparams

        # QKV weight matrices for Muon (ndim >= 2)
        qkv_weights = [
            p for name, p in self.aido_backbone.named_parameters()
            if p.requires_grad and p.ndim >= 2
        ]
        # Head + neighborhood attention + fallback embedding for AdamW
        head_and_string_params = (
            list(self.head.parameters()) +
            list(self.neighborhood_attn.parameters()) +
            list(self.fallback_emb.parameters())
        )

        param_groups = [
            dict(
                params       = qkv_weights,
                use_muon     = True,
                lr           = hp.lr_muon,
                weight_decay = hp.weight_decay,
                momentum     = 0.95,
            ),
            dict(
                params       = head_and_string_params,
                use_muon     = False,
                lr           = hp.lr_adamw,
                betas        = (0.9, 0.95),
                weight_decay = hp.weight_decay,
            ),
        ]
        use_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        opt_cls   = MuonWithAuxAdam if use_distributed else SingleDeviceMuonWithAuxAdam
        optimizer = opt_cls(param_groups)

        # Phase 1: linear warmup (5 epochs)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        # Phase 2: CosineAnnealingLR (monotonic cosine decay, no warm restarts)
        # Proven stable in STRING lineage (node1-2 F1=0.4769, node1-1-1-1-1 F1=0.4846)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=hp.t_max,
            eta_min=hp.lr_muon * 0.01,  # 1% of peak LR
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[hp.warmup_epochs],
        )
        return {
            "optimizer":    optimizer,
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
        description="Node3-1-2-1: STRING_GNN K=16 Neighborhood Attention + AIDO.Cell-10M"
    )
    parser.add_argument("--micro-batch-size",  type=int,   default=16)
    parser.add_argument("--global-batch-size", type=int,   default=128)
    parser.add_argument("--max-epochs",        type=int,   default=150)
    parser.add_argument("--lr-muon",           type=float, default=0.02)
    parser.add_argument("--lr-adamw",          type=float, default=3e-4)
    parser.add_argument("--weight-decay",      type=float, default=1.5e-2)
    parser.add_argument("--fusion-layers",     type=int,   default=4)
    parser.add_argument("--K",                 type=int,   default=16,  dest="K")
    parser.add_argument("--attn-dim",          type=int,   default=64,  dest="attn_dim")
    parser.add_argument("--head-hidden",       type=int,   default=256)
    parser.add_argument("--head-dropout",      type=float, default=0.4)
    parser.add_argument("--warmup-epochs",     type=int,   default=5,   dest="warmup_epochs")
    parser.add_argument("--t-max",             type=int,   default=100, dest="t_max")
    parser.add_argument("--label-smoothing",   type=float, default=0.1, dest="label_smoothing")
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--debug_max_step",    type=int,   default=None)
    parser.add_argument("--fast_dev_run",      action="store_true")
    parser.add_argument("--val-check-interval", type=float, default=1.0)
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

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = HybridNeighborhoodStringAIDOModel(
        fusion_layers   = args.fusion_layers,
        K               = args.K,
        attn_dim        = args.attn_dim,
        head_hidden     = args.head_hidden,
        head_dropout    = args.head_dropout,
        lr_muon         = args.lr_muon,
        lr_adamw        = args.lr_adamw,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        t_max           = args.t_max,
        label_smoothing = args.label_smoothing,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
    )
    # Early stopping: patience=12, min_delta=0.001
    # More patient than parent (10) to allow neighborhood attention to converge
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=12, min_delta=0.001)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Use GLOO backend instead of NCCL to avoid network config hangs in this environment.
    # find_unused_parameters=True because fallback_emb may not be used every batch.
    # WORLD_SIZE check ensures single-GPU torchrun (WORLD_SIZE=1) uses single-device strategy.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        strategy = DDPStrategy(
            process_group_backend="gloo",
            find_unused_parameters=True,
            timeout=timedelta(seconds=600),
        )
    else:
        strategy = "auto"

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
        num_sanity_val_steps    = 0,  # Disable sanity check to avoid NCCL hangs
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

    # Compute real F1 from saved test predictions using calc_metric.py
    score_path = Path(__file__).parent / "test_score.txt"
    pred_path  = Path(__file__).parent / "run" / "test_predictions.tsv"
    if pred_path.exists() and Path(TEST_TSV).exists():
        import subprocess
        try:
            result = subprocess.run(
                ["python", str(DATA_ROOT / "calc_metric.py"), str(pred_path), str(TEST_TSV)],
                capture_output=True, text=True, timeout=120
            )
            metrics = json.loads(result.stdout.strip().split("\n")[-1])
            f1_score = metrics.get("value", None)
            if f1_score is not None:
                with open(score_path, "w") as f:
                    f.write(f"f1_score: {f1_score}\n")
                    if "details" in metrics:
                        for k, v in metrics["details"].items():
                            f.write(f"  {k}: {v}\n")
                print(f"[Node3-1-2-1] test_f1={f1_score:.4f} — saved to {score_path}")
            else:
                with open(score_path, "w") as f:
                    f.write(f"error: {metrics.get('error', 'unknown')}\n")
        except Exception as e:
            with open(score_path, "w") as f:
                f.write(f"error: {e}\n")
            print(f"[Node3-1-2-1] calc_metric failed: {e}")
    else:
        with open(score_path, "w") as f:
            f.write(f"error: test_predictions.tsv not found\n")
    print(f"[Node3-1-2-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
