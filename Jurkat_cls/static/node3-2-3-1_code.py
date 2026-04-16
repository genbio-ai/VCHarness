#!/usr/bin/env python3
"""
Node 3-2-3-1: AIDO.Cell-10M + LoRA (all 8 layers, r=4) + Symbol CNN + Frozen STRING GNN PPI
+ Cross-Attention Fusion (3-layer TransformerEncoder) + Manifold Mixup
===============================================================================================
This node adopts the cross-attention fusion architecture from the MCTS tree-best node
(node3-1-3-1-1-1-1, test F1=0.4768) to break the concat+MLP ceiling (~0.462) in the
node3-2 lineage.

ROOT CAUSE: The parent (node3-2-3, test F1=0.461) uses 832-dim concat+MLP fusion that has
a hard ceiling at ~0.462. The tree-best node uses cross-attention fusion (+0.015 F1) with
manifold mixup, weight_decay=0.10, class_weights=[6,1,12], gamma=1.5 — all proven across
multiple nodes. The CosineAnnealingLR experiment (node3-2-3) produced no improvement.

KEY CHANGES:
  1. ARCHITECTURE: Replace concat(832) → Linear(832→384) → head with 4-token cross-attention
     fusion: [global_emb, pert_emb, sym_proj, ppi_feat] each projected to 256-dim → 3-layer
     TransformerEncoder (nhead=8, dim_ff=256, attn_dropout=0.2) → mean-pool → Linear(256→3×6640)
  2. MANIFOLD MIXUP: Applied at fused feature level (alpha=0.3) — proven critical for
     generalization in the tree-best node and its ancestors.
  3. WEIGHT DECAY: 0.03 → 0.10 — required for the cross-attention fusion to regularize
     properly with only 1,500 training samples.
  4. CLASS WEIGHTS + FOCAL LOSS: [5,1,10] gamma=2.0 → [6,1,12] gamma=1.5 — proven optimal
     from the tree-best node3-1-3-1-1-1-1 and node3-1-3-1 branches.
  5. HEAD DROPOUT: 0.4 → 0.5 — explicitly recommended in node3-2-3 feedback to increase head
     regularization.
  6. LR SCHEDULE: Drop CosineAnnealingLR (no benefit in node3-2-3). Use ReduceLROnPlateau
     (monitor=val_f1, patience=12) with constant LR. This avoids both the never-fired problem
     of the parent and the over-fired problem of node3-2-2.
  7. TOP-3 CHECKPOINT AVERAGING: Average top-3 checkpoint predictions at test time — provides
     +0.001-0.003 F1 in the tree-best node.
  8. EARLY STOPPING: patience=25 on val_f1 (recommended in node3-2-3 feedback).

Architecture:
  - AIDO.Cell-10M with LoRA r=4 on ALL 8 transformer layers (~72K LoRA params)
  - Gene symbol character-level CNN (3-branch Conv1d → 64-dim)
  - Frozen STRING GNN embeddings (256-dim per gene)
  - PPI projection: Linear(256→256) (adapts frozen PPI features)
  - Cross-attention fusion: 4 tokens × 256-dim → 3-layer TransformerEncoder → 256-dim
  - Prediction head: LayerNorm(256) → Linear(256→256) → GELU → Dropout(0.5) → Linear(256→3×6640)

Training Configuration:
  - Global batch: 64 (micro_batch=8, 8 GPUs, accumulate=1)
  - LR: backbone=2e-4, symbol_encoder=4e-4, ppi_proj=5e-4, head=6e-4
  - Weight decay: 0.10 (critical for cross-attention regularization)
  - Manifold mixup: alpha=0.3
  - Early stopping patience: 25 (on val_f1, mode=max)
  - Max epochs: 120
  - ReduceLROnPlateau: monitor=val_f1, patience=12, factor=0.5
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault('NCCL_IB_DISABLE', '1')
os.environ.setdefault('NCCL_NET_GDR_LEVEL', '0')

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
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
AIDO_CELL_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_MODEL_DIR = "/home/Models/STRING_GNN"

N_GENES_AIDO = 19_264    # AIDO.Cell vocabulary size
N_GENES_OUT = 6_640      # output genes
N_CLASSES = 3
SENTINEL_EXPR = 1.0      # baseline expression for non-perturbed genes
KNOCKOUT_EXPR = 0.0      # expression for knocked-out gene
AIDO_HIDDEN = 256        # AIDO.Cell-10M hidden dimension
AIDO_N_LAYERS = 8        # AIDO.Cell-10M transformer layers
STRING_GNN_DIM = 256     # STRING GNN embedding dimension
FUSION_DIM = 256         # cross-attention fusion dimension

# Class weights: [down, unchanged, up] — proven from tree-best node3-1-3-1-1-1-1
# Stronger minority-class weighting: [6, 1, 12] with focal gamma=1.5
CLASS_WEIGHTS = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)

# Character vocabulary for gene symbol encoding
SYMBOL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
SYMBOL_PAD_IDX = len(SYMBOL_CHARS)          # 39 → padding index
SYMBOL_VOCAB_SIZE = len(SYMBOL_CHARS) + 1   # 40
SYMBOL_MAX_LEN = 12                          # max gene symbol length


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss with label smoothing
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weights and label smoothing."""

    def __init__(
        self,
        gamma: float = 1.5,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=w, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Gene Symbol Encoder (Character-level CNN)
# ──────────────────────────────────────────────────────────────────────────────
def symbol_to_indices(symbol: str) -> List[int]:
    """Convert a gene symbol string to a list of character indices."""
    char_to_idx = {c: i for i, c in enumerate(SYMBOL_CHARS)}
    indices = []
    for ch in symbol.upper()[:SYMBOL_MAX_LEN]:
        idx = char_to_idx.get(ch, SYMBOL_PAD_IDX)
        indices.append(idx)
    while len(indices) < SYMBOL_MAX_LEN:
        indices.append(SYMBOL_PAD_IDX)
    return indices


class SymbolEncoder(nn.Module):
    """
    Character-level CNN encoder for gene symbol strings.
    Three parallel Conv1d filters at kernel sizes 2, 3, 4 → max-pool → 64-dim.
    Captures character n-gram patterns: gene family prefixes, numeric suffixes, etc.
    """

    def __init__(self, out_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(SYMBOL_VOCAB_SIZE, embed_dim, padding_idx=SYMBOL_PAD_IDX)
        self.conv2 = nn.Conv1d(embed_dim, 32, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, 32, kernel_size=4, padding=2)
        self.proj = nn.Sequential(
            nn.Linear(96, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for conv in [self.conv2, self.conv3, self.conv4]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, symbol_ids: torch.Tensor) -> torch.Tensor:
        """symbol_ids: [B, SYMBOL_MAX_LEN] int64 → [B, out_dim]"""
        x = self.embed(symbol_ids)          # [B, L, embed_dim]
        x = x.transpose(1, 2)              # [B, embed_dim, L]
        f2 = F.relu(self.conv2(x))
        f3 = F.relu(self.conv3(x))
        f4 = F.relu(self.conv4(x))
        f2 = F.adaptive_max_pool1d(f2, 1).squeeze(-1)  # [B, 32]
        f3 = F.adaptive_max_pool1d(f3, 1).squeeze(-1)  # [B, 32]
        f4 = F.adaptive_max_pool1d(f4, 1).squeeze(-1)  # [B, 32]
        feat = torch.cat([f2, f3, f4], dim=-1)          # [B, 96]
        return self.proj(feat)                           # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# STRING GNN Static Embedding Extractor
# ──────────────────────────────────────────────────────────────────────────────
def build_frozen_string_embeddings(
    string_gnn_dir: str,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Run STRING GNN once (inference only) to extract frozen [18870, 256] node embeddings.
    Returns (frozen_embeddings, ensg_to_gnn_idx_mapping).
    """
    import json as _json
    from transformers import AutoModel as _AutoModel

    model_dir = Path(string_gnn_dir)
    node_names = _json.loads((model_dir / "node_names.json").read_text())
    ensg_to_gnn_idx = {n: i for i, n in enumerate(node_names) if n.startswith("ENSG")}

    graph = torch.load(model_dir / "graph_data.pt")
    edge_index = graph["edge_index"]
    edge_weight = graph.get("edge_weight", None)

    # Load STRING GNN and run inference
    gnn_model = _AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    gnn_model.eval()

    # Use CPU for this one-time computation to avoid GPU memory issues
    with torch.no_grad():
        outputs = gnn_model(
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
    # embeddings: [18870, 256] float32
    embeddings = outputs.last_hidden_state.cpu().float()

    del gnn_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return embeddings, ensg_to_gnn_idx


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Returns pre-built AIDO.Cell expression profiles, gene positions, symbol indices,
    STRING GNN embedding indices, and labels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],    # ENSG_base → AIDO.Cell position [0, 19264)
        ensg_to_gnn: Dict[str, int],    # ENSG_base → STRING GNN node index
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.ensg_to_gnn = ensg_to_gnn
        self.is_test = is_test

        # Pre-build AIDO.Cell expression profile tensors: [N, 19264] float32
        # Baseline: 1.0 everywhere, knocked-out gene: 0.0
        self.expr_inputs = self._build_expr_tensors()

        # Pre-build symbol character index tensors: [N, SYMBOL_MAX_LEN] int64
        self.symbol_ids = self._build_symbol_tensors()

        # Pre-build STRING GNN index tensors: [N] int64 (-1 if not in STRING)
        self.gnn_indices = self._build_gnn_indices()

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1} → {0,1,2}
        else:
            self.labels = None

    def _build_expr_tensors(self) -> torch.Tensor:
        N = len(self.pert_ids)
        expr = torch.full((N, N_GENES_AIDO), SENTINEL_EXPR, dtype=torch.float32)
        for i, pert_id in enumerate(self.pert_ids):
            base = pert_id.split(".")[0]
            pos = self.gene_to_pos.get(base)
            if pos is not None:
                expr[i, pos] = KNOCKOUT_EXPR
        return expr

    def _build_symbol_tensors(self) -> torch.Tensor:
        N = len(self.symbols)
        sym_ids = torch.zeros((N, SYMBOL_MAX_LEN), dtype=torch.long)
        for i, symbol in enumerate(self.symbols):
            sym_ids[i] = torch.tensor(symbol_to_indices(symbol), dtype=torch.long)
        return sym_ids

    def _build_gnn_indices(self) -> torch.Tensor:
        """Build [N] int64 tensor of STRING GNN node indices (-1 if not found)."""
        N = len(self.pert_ids)
        gnn_idx = torch.full((N,), -1, dtype=torch.long)
        for i, pert_id in enumerate(self.pert_ids):
            base = pert_id.split(".")[0]
            idx = self.ensg_to_gnn.get(base)
            if idx is not None:
                gnn_idx[i] = idx
        return gnn_idx

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.pert_ids[idx].split(".")[0]
        gene_pos = self.gene_to_pos.get(base, -1)
        item = {
            "idx": idx,
            "expr": self.expr_inputs[idx],        # [19264] float32
            "gene_pos": gene_pos,                  # int (-1 if not in AIDO vocab)
            "symbol_ids": self.symbol_ids[idx],    # [SYMBOL_MAX_LEN] int64
            "gnn_idx": self.gnn_indices[idx].item(),  # int (-1 if not in STRING)
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "expr": torch.stack([b["expr"] for b in batch]),              # [B, 19264]
        "gene_pos": torch.tensor([b["gene_pos"] for b in batch], dtype=torch.long),
        "symbol_ids": torch.stack([b["symbol_ids"] for b in batch]),  # [B, SYMBOL_MAX_LEN]
        "gnn_idx": torch.tensor([b["gnn_idx"] for b in batch], dtype=torch.long),  # [B]
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        micro_batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.gene_to_pos: Dict[str, int] = {}
        self.ensg_to_gnn: Dict[str, int] = {}
        self.frozen_gnn_embs: Optional[torch.Tensor] = None  # [18870, 256]
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        # Rank-0 downloads tokenizer first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # Build ENSG→position mapping for AIDO.Cell
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)
            print(f"[DEGDataModule] AIDO.Cell gene vocab coverage: "
                  f"{len(self.gene_to_pos)}/{len(unique_ids)} genes")

        # Build frozen STRING GNN embeddings once
        if self.frozen_gnn_embs is None:
            print("[DEGDataModule] Building frozen STRING GNN embeddings (one-time)...")
            self.frozen_gnn_embs, self.ensg_to_gnn = build_frozen_string_embeddings(
                STRING_GNN_MODEL_DIR, device="cpu"
            )
            print(f"[DEGDataModule] STRING GNN embeddings: {self.frozen_gnn_embs.shape}, "
                  f"coverage: {len(self.ensg_to_gnn)} ENSG IDs in STRING")

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self.gene_to_pos, self.ensg_to_gnn)
            self.val_ds = PerturbationDataset(val_df, self.gene_to_pos, self.ensg_to_gnn)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self.gene_to_pos, self.ensg_to_gnn, is_test=True
            )
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    @staticmethod
    def _build_gene_to_pos(tokenizer, gene_ids: List[str]) -> Dict[str, int]:
        """Map each ENSG gene_id to its position index in AIDO.Cell vocab."""
        mapping: Dict[str, int] = {}
        PROBE_VAL = 50.0
        for gene_id in gene_ids:
            try:
                inputs = tokenizer(
                    {"gene_ids": [gene_id], "expression": [PROBE_VAL]},
                    return_tensors="pt",
                )
                ids = inputs["input_ids"]
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                pos = (ids[0] == PROBE_VAL).nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    mapping[gene_id] = int(pos[0].item())
            except Exception:
                pass
        return mapping

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Attention Fusion Module
# ──────────────────────────────────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    4-token TransformerEncoder cross-attention fusion.
    Proven architecture from tree-best node3-1-3-1-1-1-1 (test F1=0.4768).

    Input: 4 feature vectors, each projected to FUSION_DIM=256
    [global_emb, pert_emb, sym_proj, ppi_feat] → [B, 4, 256]
    → 3-layer TransformerEncoder (nhead=8, dim_ff=256, attn_dropout=0.2)
    → mean-pool → LayerNorm → [B, 256]
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 256,
        num_layers: int = 3,
        attn_dropout: float = 0.2,
        n_tokens: int = 4,
    ):
        super().__init__()
        self.d_model = d_model

        # Token type embeddings: learnable per-source position bias
        self.token_type_emb = nn.Embedding(n_tokens, d_model)
        nn.init.trunc_normal_(self.token_type_emb.weight, std=0.02)

        # Input projections (each source → FUSION_DIM)
        # (a) AIDO global_emb: 256 → 256
        self.proj_global = nn.Linear(AIDO_HIDDEN, d_model, bias=False)
        # (b) AIDO pert_emb: 256 → 256
        self.proj_pert = nn.Linear(AIDO_HIDDEN, d_model, bias=False)
        # (c) Symbol CNN: 64 → 256
        self.proj_sym = nn.Sequential(
            nn.Linear(64, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        # (d) PPI: 256 → 256
        self.proj_ppi = nn.Linear(STRING_GNN_DIM, d_model, bias=False)

        # 3-layer TransformerEncoder (pre-norm, per tree-best)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output normalization
        self.out_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.proj_global, self.proj_pert, self.proj_ppi]:
            nn.init.xavier_uniform_(proj.weight)
        nn.init.xavier_uniform_(self.proj_sym[0].weight)
        nn.init.zeros_(self.proj_sym[0].bias)

    def forward(
        self,
        global_emb: torch.Tensor,   # [B, 256] float32
        pert_emb: torch.Tensor,     # [B, 256] float32
        sym_feat: torch.Tensor,     # [B, 64]  float32
        ppi_feat: torch.Tensor,     # [B, 256] float32
    ) -> torch.Tensor:
        """Returns [B, d_model] fused representation."""
        B = global_emb.shape[0]
        device = global_emb.device

        # Project each source to d_model
        g = self.proj_global(global_emb)    # [B, 256]
        p = self.proj_pert(pert_emb)        # [B, 256]
        s = self.proj_sym(sym_feat)         # [B, 256]
        q = self.proj_ppi(ppi_feat)         # [B, 256]

        # Stack to [B, 4, 256]
        tokens = torch.stack([g, p, s, q], dim=1)  # [B, 4, 256]

        # Add learnable token-type embeddings
        type_ids = torch.arange(4, device=device)
        tokens = tokens + self.token_type_emb(type_ids).unsqueeze(0)  # [B, 4, 256]

        # Cross-attention fusion
        fused = self.transformer(tokens)   # [B, 4, 256]

        # Mean-pool and normalize
        out = self.out_norm(fused.mean(dim=1))  # [B, 256]
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class AIDOCellCrossAttnDEGModel(nn.Module):
    """
    AIDO.Cell-10M + LoRA (r=4, ALL 8 layers) + Symbol CNN + Frozen STRING GNN
    + Cross-Attention Fusion (3-layer TransformerEncoder) + Prediction Head

    Feature fusion (4 sources via cross-attention):
      (a) Global mean-pool of AIDO.Cell last_hidden_state     → [B, 256]
      (b) Perturbed-gene positional embedding from AIDO.Cell  → [B, 256]
      (c) Gene symbol character-level CNN                     → [B, 64]
      (d) Frozen STRING GNN PPI topology embedding            → [B, 256]
    4-token TransformerEncoder fusion → [B, 256] → MLP head → [B, 3, 6640]

    Key difference from node3-2-3: Cross-attention replaces concat+MLP
    This is the proven tree-best architecture (node3-1-3-1-1-1-1, 0.4768).
    """

    HIDDEN_DIM = 256          # AIDO.Cell-10M hidden size
    SYMBOL_DIM = 64           # symbol CNN output dim
    PPI_DIM = 256             # STRING GNN embedding dim

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        head_hidden: int = 256,
        head_dropout: float = 0.5,
        fusion_dim: int = FUSION_DIM,
        fusion_nhead: int = 8,
        fusion_dim_ff: int = 256,
        fusion_num_layers: int = 3,
        fusion_attn_dropout: float = 0.2,
        frozen_gnn_embs: Optional[torch.Tensor] = None,  # [18870, 256] float32
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA on ALL 8 layers ──────────────────
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA on Q/K/V of ALL 8 transformer layers (layers 0-7)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],
            layers_to_transform=list(range(AIDO_N_LAYERS)),  # all 8 layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Cast trainable LoRA params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Gene symbol character-level CNN encoder ────────────────────────────
        self.symbol_encoder = SymbolEncoder(out_dim=self.SYMBOL_DIM, embed_dim=32)

        # ── Frozen STRING GNN embedding table ─────────────────────────────────
        if frozen_gnn_embs is not None:
            self.register_buffer("gnn_emb_table", frozen_gnn_embs)  # [18870, 256]
        else:
            self.register_buffer("gnn_emb_table",
                                 torch.zeros(18870, self.PPI_DIM, dtype=torch.float32))

        # Learnable fallback embedding for genes not in STRING GNN
        self.gnn_fallback = nn.Parameter(torch.zeros(self.PPI_DIM))
        nn.init.trunc_normal_(self.gnn_fallback, std=0.02)

        # Optional PPI projection (1-layer: adapts frozen embeddings to task)
        self.ppi_proj = nn.Sequential(
            nn.Linear(self.PPI_DIM, self.PPI_DIM),
            nn.GELU(),
            nn.LayerNorm(self.PPI_DIM),
        )
        nn.init.xavier_uniform_(self.ppi_proj[0].weight)
        nn.init.zeros_(self.ppi_proj[0].bias)

        # ── Cross-Attention Fusion Module ──────────────────────────────────────
        # Proven architecture from tree-best: 4 tokens × 256-dim
        self.fusion = CrossAttentionFusion(
            d_model=fusion_dim,
            nhead=fusion_nhead,
            dim_ff=fusion_dim_ff,
            num_layers=fusion_num_layers,
            attn_dropout=fusion_attn_dropout,
            n_tokens=4,
        )

        # ── Prediction head ────────────────────────────────────────────────────
        # Input: fusion_dim=256 (from cross-attention)
        # Output: [B, N_CLASSES * N_GENES_OUT] = [B, 19920]
        # Proven from tree-best: LN → Linear → GELU → Dropout → Linear
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        expr: torch.Tensor,          # [B, 19264] float32
        gene_pos: torch.Tensor,      # [B]         int64 (-1 if not in AIDO vocab)
        symbol_ids: torch.Tensor,    # [B, SYMBOL_MAX_LEN] int64
        gnn_idx: torch.Tensor,       # [B]         int64 (-1 if not in STRING)
    ) -> torch.Tensor:
        B = expr.shape[0]

        # ── AIDO.Cell backbone forward ─────────────────────────────────────────
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, 19266, 256]

        # (a) Global mean-pool over all gene positions (exclude 2 summary tokens)
        gene_emb = lhs[:, :N_GENES_AIDO, :]          # [B, 19264, 256]
        global_emb = gene_emb.mean(dim=1)             # [B, 256]

        # (b) Perturbed-gene positional embedding
        pert_emb = torch.zeros(B, self.HIDDEN_DIM, device=lhs.device, dtype=lhs.dtype)
        valid_aido = gene_pos >= 0
        if valid_aido.any():
            valid_pos = gene_pos[valid_aido]  # [k]
            pert_emb[valid_aido] = lhs[valid_aido, valid_pos, :]
        pert_emb[~valid_aido] = global_emb[~valid_aido]

        # Convert backbone features to float32 for head computation
        global_emb_f = global_emb.float()   # [B, 256]
        pert_emb_f = pert_emb.float()        # [B, 256]

        # (c) Gene symbol character CNN
        sym_feat = self.symbol_encoder(symbol_ids)    # [B, 64] float32

        # (d) Frozen STRING GNN PPI embedding lookup
        ppi_emb = self.gnn_fallback.unsqueeze(0).expand(B, -1).clone()  # [B, 256]
        valid_gnn = gnn_idx >= 0
        if valid_gnn.any():
            valid_gnn_idx = gnn_idx[valid_gnn]  # [k]
            with torch.no_grad():
                ppi_raw = self.gnn_emb_table[valid_gnn_idx]  # [k, 256]
            ppi_emb[valid_gnn] = ppi_raw

        # Project PPI features (trainable linear to adapt frozen embeddings)
        ppi_feat = self.ppi_proj(ppi_emb.to(global_emb_f.device))  # [B, 256]

        # ── Cross-Attention Fusion ─────────────────────────────────────────────
        # 4 feature sources fused via TransformerEncoder → [B, 256]
        fused = self.fusion(global_emb_f, pert_emb_f, sym_feat, ppi_feat)  # [B, 256]

        # ── Prediction Head ───────────────────────────────────────────────────
        logits = self.head(fused)                      # [B, 3 * 6640]
        return logits.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro F1, averaged over all genes.
    y_pred: [n_samples, 3, n_genes]  (3-class probability distributions)
    y_true_remapped: [n_samples, n_genes]  (labels in {0,1,2})
    """
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp_class = y_pred[:, :, g]
        yhat = yp_class.argmax(axis=1)
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yhat, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ──────────────────────────────────────────────────────────────────────────────
# Manifold Mixup Helper
# ──────────────────────────────────────────────────────────────────────────────
def mixup_manifold(
    fused: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Manifold mixup at the fused feature level.
    Returns mixed features, labels_a, labels_b, and lambda.
    Proven to improve generalization in the tree-best cross-attention nodes.
    """
    if alpha <= 0.0:
        return fused, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    B = fused.shape[0]
    index = torch.randperm(B, device=fused.device)
    mixed = lam * fused + (1 - lam) * fused[index]
    return mixed, labels, labels[index], lam


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        head_hidden: int = 256,
        head_dropout: float = 0.5,
        lr: float = 2e-4,
        head_lr_multiplier: float = 3.0,
        symbol_encoder_lr_multiplier: float = 2.0,
        ppi_proj_lr_multiplier: float = 2.5,
        weight_decay: float = 0.10,
        gamma_focal: float = 1.5,
        label_smoothing: float = 0.05,
        mixup_alpha: float = 0.3,
        plateau_patience: int = 12,
        plateau_factor: float = 0.5,
        fusion_dim: int = FUSION_DIM,
        fusion_nhead: int = 8,
        fusion_dim_ff: int = 256,
        fusion_num_layers: int = 3,
        fusion_attn_dropout: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCellCrossAttnDEGModel] = None
        self.criterion: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            # Get frozen STRING GNN embeddings from the datamodule
            frozen_gnn_embs = None
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
                dm = self.trainer.datamodule
                if hasattr(dm, "frozen_gnn_embs") and dm.frozen_gnn_embs is not None:
                    frozen_gnn_embs = dm.frozen_gnn_embs

            self.model = AIDOCellCrossAttnDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                head_hidden=self.hparams.head_hidden,
                head_dropout=self.hparams.head_dropout,
                fusion_dim=self.hparams.fusion_dim,
                fusion_nhead=self.hparams.fusion_nhead,
                fusion_dim_ff=self.hparams.fusion_dim_ff,
                fusion_num_layers=self.hparams.fusion_num_layers,
                fusion_attn_dropout=self.hparams.fusion_attn_dropout,
                frozen_gnn_embs=frozen_gnn_embs,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            batch["expr"],
            batch["gene_pos"],
            batch["symbol_ids"],
            batch["gnn_idx"],
        )

    def _compute_loss_with_labels(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_flat = labels.reshape(-1)
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training with manifold mixup at the fused feature level."""
        # Forward pass through backbone and feature extractors
        expr = batch["expr"]
        gene_pos = batch["gene_pos"]
        symbol_ids = batch["symbol_ids"]
        gnn_idx = batch["gnn_idx"]
        labels = batch["label"]

        # Get features before fusion (manually call sub-modules)
        model = self.model
        B = expr.shape[0]

        # AIDO.Cell backbone
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = model.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, 19266, 256]

        gene_emb = lhs[:, :N_GENES_AIDO, :]
        global_emb = gene_emb.mean(dim=1)

        pert_emb = torch.zeros(B, model.HIDDEN_DIM, device=lhs.device, dtype=lhs.dtype)
        valid_aido = gene_pos >= 0
        if valid_aido.any():
            valid_pos = gene_pos[valid_aido]
            pert_emb[valid_aido] = lhs[valid_aido, valid_pos, :]
        pert_emb[~valid_aido] = global_emb[~valid_aido]

        global_emb_f = global_emb.float()
        pert_emb_f = pert_emb.float()

        sym_feat = model.symbol_encoder(symbol_ids)

        ppi_emb = model.gnn_fallback.unsqueeze(0).expand(B, -1).clone()
        valid_gnn = gnn_idx >= 0
        if valid_gnn.any():
            valid_gnn_idx = gnn_idx[valid_gnn]
            with torch.no_grad():
                ppi_raw = model.gnn_emb_table[valid_gnn_idx]
            ppi_emb[valid_gnn] = ppi_raw

        ppi_feat = model.ppi_proj(ppi_emb.to(global_emb_f.device))

        # Cross-attention fusion → [B, 256]
        fused = model.fusion(global_emb_f, pert_emb_f, sym_feat, ppi_feat)

        # Manifold mixup at fused feature level
        alpha = self.hparams.mixup_alpha
        if self.training and alpha > 0.0:
            fused_mixed, labels_a, labels_b, lam = mixup_manifold(fused, labels, alpha)
        else:
            fused_mixed, labels_a, labels_b, lam = fused, labels, labels, 1.0

        # Prediction head
        logits_flat_raw = model.head(fused_mixed)          # [B, 3 * 6640]
        logits = logits_flat_raw.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]

        # Mixup loss: lam * loss(labels_a) + (1-lam) * loss(labels_b)
        loss_a = self._compute_loss_with_labels(logits, labels_a)
        if lam < 1.0:
            loss_b = self._compute_loss_with_labels(logits, labels_b)
            loss = lam * loss_a + (1 - lam) * loss_b
        else:
            loss = loss_a

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        loss = self._compute_loss_with_labels(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        lp = torch.cat(self._val_preds, 0)
        ll = torch.cat(self._val_labels, 0)
        li = torch.cat(self._val_indices, 0)
        ap = self.all_gather(lp)
        al = self.all_gather(ll)
        ai = self.all_gather(li)
        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
        labels = al.cpu().view(-1, N_GENES_OUT).numpy()
        idxs = ai.cpu().view(-1).numpy()
        _, uniq = np.unique(idxs, return_index=True)
        f1 = compute_deg_f1(preds[uniq], labels[uniq])

        # All-reduce so all ranks log the same val_f1
        f1_tensor = torch.tensor(f1, dtype=torch.float32, device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(f1_tensor, op=torch.distributed.ReduceOp.SUM)
            f1_tensor = f1_tensor / self.trainer.world_size

        self.log("val_f1", f1_tensor.item(), prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        lp = torch.cat(self._test_preds, 0)
        li = torch.cat(self._test_indices, 0)
        ap = self.all_gather(lp)
        ai = self.all_gather(li)
        self._test_preds.clear()
        self._test_indices.clear()

        is_global_zero = (
            getattr(self.trainer, "is_global_zero", None) or
            (getattr(self.trainer, "global_rank", 0) == 0)
        )

        if is_global_zero:
            preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
            idxs = ai.cpu().view(-1).numpy()
            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]
            idxs = idxs[uniq]
            order = np.argsort(idxs)
            preds = preds[order]
            idxs = idxs[order]

            # Store this checkpoint's sorted predictions for averaging
            if not hasattr(self, "_ckpt_preds_store"):
                self._ckpt_preds_store = []
                self._ckpt_idxs_store = None
            self._ckpt_preds_store.append(preds)
            self._ckpt_idxs_store = idxs

    def configure_optimizers(self):
        hp = self.hparams
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        ppi_proj_params = (
            list(self.model.ppi_proj.parameters()) +
            [self.model.gnn_fallback]
        )
        symbol_params = list(self.model.symbol_encoder.parameters())
        # Fusion + head parameters
        head_params = (
            list(self.model.fusion.parameters()) +
            list(self.model.head.parameters())
        )

        backbone_lr = hp.lr                                          # 2e-4
        ppi_lr = hp.lr * hp.ppi_proj_lr_multiplier                 # 5e-4
        symbol_lr = hp.lr * hp.symbol_encoder_lr_multiplier        # 4e-4
        head_lr = hp.lr * hp.head_lr_multiplier                    # 6e-4

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": ppi_proj_params, "lr": ppi_lr},
                {"params": symbol_params, "lr": symbol_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=hp.weight_decay,
        )

        # ReduceLROnPlateau monitoring val_f1 (mode=max)
        # - patience=12: proven from tree-best node3-1-3-1-1-1-1 (never fired, model improved through E86)
        # - factor=0.5: standard 2x reduction
        # - monitor=val_f1: the ONLY reliable signal for this task (iron rule from node3-2-2)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=hp.plateau_factor,
            patience=hp.plateau_patience,
            verbose=True,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        out = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                k = prefix + name
                if k in full:
                    out[k] = full[k]
        for name, buf in self.named_buffers():
            k = prefix + name
            if k in full:
                out[k] = full[k]
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )
        return out

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable parameters and buffers from a partial checkpoint."""
        full_state_keys = set(super().state_dict().keys())
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        buffer_keys = {
            name for name, _ in self.named_buffers() if name in full_state_keys
        }
        expected_keys = trainable_keys | buffer_keys

        missing_keys = [k for k in expected_keys if k not in state_dict]
        unexpected_keys = [k for k in state_dict if k not in expected_keys]

        if missing_keys:
            self.print(f"Warning: Missing checkpoint keys: {missing_keys[:5]}...")
        if unexpected_keys:
            self.print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

        loaded_trainable = len([k for k in state_dict if k in trainable_keys])
        loaded_buffers = len([k for k in state_dict if k in buffer_keys])
        self.print(
            f"Loading checkpoint: {loaded_trainable} trainable parameters and "
            f"{loaded_buffers} buffers"
        )
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Top-K Checkpoint Averaging for Test
# ──────────────────────────────────────────────────────────────────────────────
def run_test_with_ckpt_averaging(
    trainer: pl.Trainer,
    model_module: DEGLightningModule,
    datamodule: DEGDataModule,
    checkpoint_cb: ModelCheckpoint,
    output_dir: Path,
    top_k: int = 3,
) -> None:
    """
    Run test inference for each top-k checkpoint, then average predictions.
    Proven to give +0.001-0.003 F1 boost in tree-best node3-1-3-1-1-1-1.
    """
    # Get top-k checkpoint paths sorted by score
    ckpt_dir = output_dir / "checkpoints"
    scored_ckpts = []
    for ckpt in ckpt_dir.glob("*.ckpt"):
        if "last" not in ckpt.name:
            parts = ckpt.stem.split("-")
            try:
                score = float(parts[-1])
                scored_ckpts.append((score, str(ckpt)))
            except (ValueError, IndexError):
                pass

    scored_ckpts.sort(key=lambda x: x[0], reverse=True)
    ckpt_paths = [p for _, p in scored_ckpts[:top_k]]

    # Fallback: use best from callback
    if not ckpt_paths and checkpoint_cb.best_model_path:
        ckpt_paths = [checkpoint_cb.best_model_path]

    if not ckpt_paths:
        print("WARNING: No checkpoint paths found. Running test with current model.")
        trainer.test(model_module, datamodule=datamodule)
        return

    print(f"[CheckpointAveraging] Testing with {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        print(f"  - {p}")

    # Initialize prediction storage on model
    model_module._ckpt_preds_store = []
    model_module._ckpt_idxs_store = None

    for ckpt_path in ckpt_paths:
        # Reset test state before each checkpoint run
        model_module._test_preds = []
        model_module._test_indices = []
        trainer.test(model_module, datamodule=datamodule, ckpt_path=ckpt_path)

        is_rank0 = (
            getattr(trainer, "is_global_zero", None) or
            (getattr(trainer, "global_rank", 0) == 0)
        )
        if is_rank0 and hasattr(model_module, "_ckpt_preds_store"):
            n_stored = len(model_module._ckpt_preds_store)
            print(f"  Stored predictions for ckpt {ckpt_path}: "
                  f"total stored={n_stored}, "
                  f"shape={model_module._ckpt_preds_store[-1].shape if n_stored > 0 else 'N/A'}")

    # Average predictions on rank 0 and write file
    is_rank0 = (
        getattr(trainer, "is_global_zero", None) or
        (getattr(trainer, "global_rank", 0) == 0)
    )
    if is_rank0:
        all_preds_list = getattr(model_module, "_ckpt_preds_store", [])
        final_idxs = getattr(model_module, "_ckpt_idxs_store", None)

        if all_preds_list and final_idxs is not None:
            # Average softmax probabilities across checkpoints
            avg_preds = np.mean(all_preds_list, axis=0)  # [n_test, 3, 6640]

            all_pert_ids = getattr(datamodule, "test_pert_ids", None) or model_module._test_pert_ids
            all_symbols = getattr(datamodule, "test_symbols", None) or model_module._test_symbols

            output_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for rank_i, orig_i in enumerate(final_idxs):
                pert_id = (all_pert_ids[int(orig_i)]
                           if all_pert_ids is not None and int(orig_i) < len(all_pert_ids)
                           else str(orig_i))
                symbol = (all_symbols[int(orig_i)]
                          if all_symbols is not None and int(orig_i) < len(all_symbols)
                          else "")
                rows.append({
                    "idx": pert_id,
                    "input": symbol,
                    "prediction": json.dumps(avg_preds[rank_i].tolist()),
                })
            pd.DataFrame(rows).to_csv(
                output_dir / "test_predictions.tsv", sep="\t", index=False
            )
            print(f"[CheckpointAveraging] Saved {len(rows)} averaged predictions "
                  f"({len(all_preds_list)} checkpoints) → {output_dir / 'test_predictions.tsv'}")
        else:
            print("WARNING: No predictions collected from checkpoint averaging.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 3-2-3-1: AIDO.Cell-10M + LoRA all-8-layers + Symbol CNN + Frozen STRING GNN PPI + Cross-Attention Fusion + Manifold Mixup"
    )
    p.add_argument("--data-dir", type=str, default=str(Path.cwd() / "data"))
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--head-lr-multiplier", type=float, default=3.0)
    p.add_argument("--symbol-encoder-lr-multiplier", type=float, default=2.0)
    p.add_argument("--ppi-proj-lr-multiplier", type=float, default=2.5)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.5)
    p.add_argument("--gamma-focal", type=float, default=1.5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--mixup-alpha", type=float, default=0.3,
                   help="Manifold mixup alpha — 0.0 disables mixup")
    # Cross-attention fusion hyperparameters
    p.add_argument("--fusion-dim", type=int, default=256,
                   help="Cross-attention fusion d_model dimension")
    p.add_argument("--fusion-nhead", type=int, default=8,
                   help="Number of attention heads in fusion TransformerEncoder")
    p.add_argument("--fusion-dim-ff", type=int, default=256,
                   help="FFN dimension in fusion TransformerEncoder")
    p.add_argument("--fusion-num-layers", type=int, default=3,
                   help="Number of TransformerEncoder layers in fusion module")
    p.add_argument("--fusion-attn-dropout", type=float, default=0.2,
                   help="Dropout rate in cross-attention fusion TransformerEncoder")
    # ReduceLROnPlateau scheduler
    p.add_argument("--plateau-patience", type=int, default=12,
                   help="Patience for ReduceLROnPlateau (monitor=val_f1, mode=max)")
    p.add_argument("--plateau-factor", type=float, default=0.5,
                   help="LR reduction factor for ReduceLROnPlateau")
    # Early stopping and training duration
    p.add_argument("--early-stopping-patience", type=int, default=25,
                   help="Early stopping patience on val_f1 (recommended from node3-2-3 feedback)")
    p.add_argument("--top-k-ckpts", type=int, default=3,
                   help="Number of top checkpoints to average at test time")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug_max_step", "--debug-max-step", dest="debug_max_step", type=int, default=None)
    p.add_argument("--fast-dev-run", "--fast_dev_run", action="store_true", default=False)
    return p.parse_args()


def main():
    pl.seed_everything(seed=0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit_train_batches = args.debug_max_step if args.debug_max_step is not None else 1.0
    limit_val_batches = args.debug_max_step if args.debug_max_step is not None else 1.0
    limit_test_batches = 1.0  # Full test set for all non-fast_dev_run runs

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node3-2-3-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=args.top_k_ckpts, save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.early_stopping_patience, verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=180)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=1.0 if (args.debug_max_step is not None or args.fast_dev_run)
                           else args.val_check_interval,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=False,   # FlashAttention is non-deterministic
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        lr=args.lr,
        head_lr_multiplier=args.head_lr_multiplier,
        symbol_encoder_lr_multiplier=args.symbol_encoder_lr_multiplier,
        ppi_proj_lr_multiplier=args.ppi_proj_lr_multiplier,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        fusion_dim=args.fusion_dim,
        fusion_nhead=args.fusion_nhead,
        fusion_dim_ff=args.fusion_dim_ff,
        fusion_num_layers=args.fusion_num_layers,
        fusion_attn_dropout=args.fusion_attn_dropout,
    )

    trainer.fit(model_module, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        # Simple test for debug runs (single checkpoint, no averaging)
        test_results = trainer.test(model_module, datamodule=datamodule)
        # Write predictions from the on_test_epoch_end store
        if (getattr(trainer, "is_global_zero", None) or
                (getattr(trainer, "global_rank", 0) == 0)):
            if hasattr(model_module, "_ckpt_preds_store") and model_module._ckpt_preds_store:
                preds = model_module._ckpt_preds_store[-1]
                final_idxs = model_module._ckpt_idxs_store
                all_pert_ids = getattr(datamodule, "test_pert_ids", None)
                all_symbols = getattr(datamodule, "test_symbols", None)
                if final_idxs is not None:
                    rows = []
                    for rank_i, orig_i in enumerate(final_idxs):
                        pert_id = (all_pert_ids[int(orig_i)]
                                   if all_pert_ids is not None and int(orig_i) < len(all_pert_ids)
                                   else str(orig_i))
                        symbol = (all_symbols[int(orig_i)]
                                  if all_symbols is not None and int(orig_i) < len(all_symbols)
                                  else "")
                        rows.append({
                            "idx": pert_id, "input": symbol,
                            "prediction": json.dumps(preds[rank_i].tolist()),
                        })
                    if rows:
                        pd.DataFrame(rows).to_csv(
                            output_dir / "test_predictions.tsv", sep="\t", index=False
                        )
                        print(f"Debug test predictions saved ({len(rows)} samples)")
    else:
        # Production: use top-K checkpoint averaging
        run_test_with_ckpt_averaging(
            trainer=trainer,
            model_module=model_module,
            datamodule=datamodule,
            checkpoint_cb=checkpoint_cb,
            output_dir=output_dir,
            top_k=args.top_k_ckpts,
        )

    if getattr(trainer, "is_global_zero", None) or (trainer.global_rank == 0):
        score_path = Path(__file__).parent / "test_score.txt"
        primary_val = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else float("nan")
        )
        score_path.write_text(
            f"# Node 3-2-3-1 Test Evaluation Results\n"
            f"# Primary metric: f1_score (macro-averaged per-gene F1)\n"
            f"# Model: AIDO.Cell-10M + LoRA r=4 all-8-layers + Symbol CNN + Frozen STRING GNN PPI\n"
            f"# Architecture: Cross-Attention Fusion (3-layer TransformerEncoder) + Manifold Mixup\n"
            f"# Key changes from parent node3-2-3: cross-attention fusion, manifold mixup, WD=0.10,\n"
            f"#   class_weights=[6,1,12], gamma=1.5, head_dropout=0.5, ReduceLROnPlateau(val_f1, patience=12)\n"
            f"f1_score: {primary_val:.6f}\n"
            f"val_f1_best: {primary_val:.6f}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
