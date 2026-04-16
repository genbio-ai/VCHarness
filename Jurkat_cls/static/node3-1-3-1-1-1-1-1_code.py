#!/usr/bin/env python3
"""
Node 3-1-3-1-1-1-1-1: 4-Layer Cross-Attention Fusion + Extended Training (150 epochs) + Top-5 Checkpoint Avg
===========================================================================================================
Building on the tree-best node3-1-3-1-1-1-1 (test F1=0.4768), implementing the top-priority
recommendations from its feedback report.

The parent's key findings:
  - Model completed all 100 epochs with val_f1 STILL SLOWLY IMPROVING at epoch 100 (~0.475)
  - LR stayed constant throughout (ReduceLROnPlateau never fired) — ideal behavior
  - Top-3 checkpoints spanned epochs 79, 86, 88 — tight plateau (spread 0.0017 F1)
  - Val-test gap +0.0018 — excellent generalization (test > val)
  - No overfitting detected at epoch 100 — room for more training

This node implements three targeted improvements:
  1. EXTEND training to 150 epochs (feedback Priority 1: model still improving at E100)
     - Increase early_stopping_patience: 20 → 30 (proportional to longer training)
     - LR scheduler patience stays at 12 (never fires — correct behavior)

  2. ADD 4th fusion transformer layer (feedback Priority 2: "increase to 4 layers")
     - 4-layer, dim_ff=256 (unchanged): 4×330K ≈ 1.32M fusion params
     - vs parent's 3-layer: 3×330K ≈ 990K fusion params (+330K)
     - Still BELOW the failed child's 1.578M (dim_ff=512) — safe capacity zone
     - More depth = more cross-modal re-contextualization passes
     - Strong regularization (wd=0.10, mixup=0.3, attn_dropout=0.2) controls added capacity

  3. EXPAND checkpoint averaging from top-3 to top-5
     - Parent's 3 checkpoints were in a 14-epoch span (E79, E86, E88, spread=0.0017)
     - With 150 epochs and 4-layer fusion, peak plateau may be wider
     - Top-5 captures more diversity while remaining in the high-quality region
     - Feedback: "Expanding to top-5 could capture more diversity in the plateau region,
       potentially gaining +0.001–0.002"

Unchanged from tree-best parent (node3-1-3-1-1-1-1):
  - AIDO.Cell-10M backbone with LoRA r=4 on all 8 layers (Q/K/V)
  - Character-level Symbol CNN (3-branch, 64-dim)
  - Frozen STRING GNN PPI embeddings (256-dim)
  - nhead=8, attn_dropout=0.2 (stronger fusion regularization)
  - Class weights [6.0, 1.0, 12.0] (proven effective)
  - AdamW differential LR: backbone=2e-4, head=6e-4
  - Focal gamma=1.5, label_smoothing=0.05
  - Weight decay=0.10, Mixup alpha=0.3, LR scheduler patience=12
  - Gradient clipping (norm=1.0)

Tree context:
  - node3-1-3-1-1-1-1 (parent, F1=0.4768, tree-best): 3L+dim_ff=256+nhead=8+wd=0.10+mixup=0.3
  - node3-1-3-1-1 (grandparent, F1=0.4739): 3L+dim_ff=384+wd=0.10+mixup=0.3
  - node3-1-3-1 (F1=0.4731): 2L+dim_ff=512+wd=0.08
  - This node: **4L+dim_ff=256+nhead=8+wd=0.10+mixup=0.3 (150 epochs, top-5 avg)**
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
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
AIDO_CELL_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_MODEL_DIR = Path("/home/Models/STRING_GNN")

N_GENES_AIDO = 19_264    # AIDO.Cell vocabulary size
N_GENES_OUT = 6_640      # output genes (DEG prediction targets)
N_CLASSES = 3            # {0:down-regulated, 1:unchanged, 2:up-regulated}
AIDO_N_LAYERS = 8        # AIDO.Cell-10M transformer layers
CROSS_ATTN_DIM = 256     # unified dimension for cross-attention tokens

SENTINEL_EXPR = 1.0      # expression for all non-perturbed genes
KNOCKOUT_EXPR = 0.0      # expression for the knocked-out gene

# Class weights: [6.0, 1.0, 12.0] — proven effective in parent and tree
# down ~3.56%, unchanged ~94.82%, up ~1.63%
CLASS_WEIGHTS = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)

# Symbol character encoding
CHAR_VOCAB = sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._/")) + ["<UNK>", "<PAD>"]
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}
PAD_IDX = CHAR_TO_IDX["<PAD>"]
UNK_IDX = CHAR_TO_IDX["<UNK>"]
SYMBOL_MAX_LEN = 16


def symbol_to_indices(symbol: str) -> List[int]:
    """Convert gene symbol string to padded character index list."""
    chars = list(symbol.upper())[:SYMBOL_MAX_LEN]
    idxs = [CHAR_TO_IDX.get(c, UNK_IDX) for c in chars]
    idxs += [PAD_IDX] * (SYMBOL_MAX_LEN - len(idxs))
    return idxs


def build_frozen_string_embeddings(model_dir: Path, device: str = "cpu"):
    """
    Pre-compute frozen STRING GNN embeddings once on CPU.
    Returns:
      embs: [18870, 256] float32 tensor
      ensg_to_gnn: dict mapping ENSG id → row index in embs
    """
    gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    gnn_model.eval()
    gnn_model = gnn_model.to(device)

    graph_data = torch.load(str(model_dir / "graph_data.pt"), map_location=device)
    edge_index = graph_data["edge_index"].long().to(device)
    edge_w = graph_data.get("edge_weight")
    if edge_w is not None:
        edge_w = edge_w.float().to(device)
    else:
        edge_w = torch.ones(edge_index.shape[1], device=device)

    with torch.no_grad():
        out = gnn_model(
            edge_index=edge_index,
            edge_weight=edge_w,
        ).last_hidden_state.float().cpu()  # [18870, 256]

    node_names = json.loads((model_dir / "node_names.json").read_text())
    ensg_to_gnn = {name: i for i, name in enumerate(node_names)}

    del gnn_model
    if device != "cpu":
        torch.cuda.empty_cache()

    return out, ensg_to_gnn


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Multi-class focal loss to down-weight easy examples (dominant class 1=unchanged)."""

    def __init__(
        self,
        gamma: float = 1.5,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [N, C], targets: [N]
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(
            logits, targets,
            weight=w,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Pre-builds AIDO.Cell expression profiles, symbol indices,
    STRING GNN indices, and labels for efficient batch loading.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],   # ENSG_base → AIDO.Cell position [0, 19264)
        ensg_to_gnn: Dict[str, int],   # ENSG_base → STRING GNN node index
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.ensg_to_gnn = ensg_to_gnn
        self.is_test = is_test

        # Pre-build expression profile tensors: [N, N_GENES_AIDO] float32
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
            "expr": self.expr_inputs[idx],              # [N_GENES_AIDO] float32
            "gene_pos": gene_pos,                        # int (-1 if not in AIDO vocab)
            "symbol_ids": self.symbol_ids[idx],          # [SYMBOL_MAX_LEN] int64
            "gnn_idx": self.gnn_indices[idx].item(),     # int (-1 if not in STRING)
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "expr": torch.stack([b["expr"] for b in batch]),              # [B, N_GENES_AIDO]
        "gene_pos": torch.tensor([b["gene_pos"] for b in batch], dtype=torch.long),  # [B]
        "symbol_ids": torch.stack([b["symbol_ids"] for b in batch]),  # [B, SYMBOL_MAX_LEN]
        "gnn_idx": torch.tensor([b["gnn_idx"] for b in batch], dtype=torch.long),  # [B]
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])  # [B, N_GENES_OUT]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        micro_batch_size: int = 8,
        num_workers: int = 0,
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
        # Rank-safe tokenizer loading: rank 0 downloads first
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # Build ENSG → AIDO.Cell position mapping (probe each gene)
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)
            print(
                f"[DEGDataModule] AIDO.Cell gene vocab coverage: "
                f"{len(self.gene_to_pos)}/{len(unique_ids)} genes"
            )

        # Build frozen STRING GNN embeddings once (CPU)
        if self.frozen_gnn_embs is None:
            print("[DEGDataModule] Pre-computing STRING GNN PPI embeddings (CPU)...")
            self.frozen_gnn_embs, self.ensg_to_gnn = build_frozen_string_embeddings(
                STRING_GNN_MODEL_DIR, device="cpu"
            )
            print(
                f"[DEGDataModule] GNN embs shape: {self.frozen_gnn_embs.shape}, "
                f"vocab size: {len(self.ensg_to_gnn)}"
            )

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self.gene_to_pos, self.ensg_to_gnn
            )
            self.val_ds = PerturbationDataset(
                val_df, self.gene_to_pos, self.ensg_to_gnn
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self.gene_to_pos, self.ensg_to_gnn, is_test=True
            )
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    @staticmethod
    def _build_gene_to_pos(tokenizer, gene_ids: List[str]) -> Dict[str, int]:
        """Map each ENSG gene_id to its position index in AIDO.Cell vocab via probing."""
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
            collate_fn=collate_fn, persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=collate_fn, persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=collate_fn, persistent_workers=self.num_workers > 0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Symbol Encoder (3-branch character-level CNN)
# ──────────────────────────────────────────────────────────────────────────────
class SymbolEncoder(nn.Module):
    """
    3-branch character-level CNN for gene symbol strings.
    Kernels [3, 5, 7] → max-pool → concat → Linear(192→out_dim).
    Outputs 64-dim features (unchanged from parent).
    """

    def __init__(self, out_dim: int = 64, embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.char_emb = nn.Embedding(len(CHAR_VOCAB), embed_dim, padding_idx=PAD_IDX)
        nn.init.normal_(self.char_emb.weight, std=0.02)
        self.char_emb.weight.data[PAD_IDX].zero_()

        branch_dim = out_dim
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, branch_dim, kernel_size=k, padding=k // 2),
                nn.GELU(),
            )
            for k in [3, 5, 7]
        ])

        self.fusion = nn.Sequential(
            nn.Linear(3 * branch_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, symbol_ids: torch.Tensor) -> torch.Tensor:
        # symbol_ids: [B, SYMBOL_MAX_LEN] int64
        x = self.char_emb(symbol_ids)        # [B, L, embed_dim]
        x = x.transpose(1, 2)                # [B, embed_dim, L]

        branch_outs = []
        for branch in self.branches:
            out = branch(x)                  # [B, branch_dim, L]
            out = out.max(dim=-1).values     # [B, branch_dim] global max-pool
            branch_outs.append(out)

        concat = torch.cat(branch_outs, dim=-1)  # [B, 3*branch_dim]
        return self.fusion(concat)               # [B, out_dim=64]


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Attention Feature Fusion Transformer (4-Layer)
# ──────────────────────────────────────────────────────────────────────────────
class FeatureFusionTransformer(nn.Module):
    """
    Cross-attention fusion via Transformer self-attention.

    Takes 4 feature sources and treats them as a sequence of tokens:
      Token 0: global_emb  — AIDO.Cell global mean-pool     [B, 256]
      Token 1: pert_emb    — AIDO.Cell perturbed gene pos   [B, 256]
      Token 2: sym_proj    — Symbol CNN projected            [B, 64→256]
      Token 3: ppi_feat    — STRING GNN PPI embedding        [B, 256]

    KEY CHANGE vs parent (node3-1-3-1-1-1-1):
      n_layers: 3 → 4 (DEEPER fusion for richer cross-modal re-contextualization)
              Rationale: parent's feedback: "Increase fusion layers to 4: If capacity
              is the bottleneck, adding a 4th fusion layer (while keeping dim_ff=256
              to control param count) could provide additional cross-modal refinement.
              Estimated fusion params: 4×330K ≈ 1.32M vs current 3×330K ≈ 990K —
              still below the problematic 1.578M of the failed parent."

    Unchanged from parent:
      dim_ff: 256 (width-constrained, matching d_model)
      attn_dropout: 0.2 (targeted fusion regularization)
      n_heads: 8 (fine-grained multi-head attention per head=32-dim)

    Applies n-layer pre-norm Transformer encoder.
    Mean pools the output tokens → [B, 256] fused representation.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,          # unchanged from parent (8 heads, 32-dim each)
        n_layers: int = 4,          # CHANGED: parent=3 → this=4 (more depth)
        dim_ff: int = 256,          # unchanged from parent (width-constrained)
        attn_dropout: float = 0.2,  # unchanged from parent (targeted regularization)
        sym_in_dim: int = 64,
    ):
        super().__init__()

        # Project symbol features (64-dim) to d_model (256-dim)
        self.sym_proj = nn.Sequential(
            nn.Linear(sym_in_dim, d_model),
            nn.LayerNorm(d_model),
        )
        nn.init.xavier_uniform_(self.sym_proj[0].weight)
        nn.init.zeros_(self.sym_proj[0].bias)

        # Learnable token-type positional embeddings (4 token positions)
        self.pos_emb = nn.Parameter(torch.zeros(4, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # n-layer Transformer Encoder with pre-norm (more stable)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,  # [B, seq_len, d_model]
            norm_first=True,   # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,  # Avoid potential issues
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        global_emb: torch.Tensor,   # [B, 256] float32
        pert_emb: torch.Tensor,     # [B, 256] float32
        sym_feat: torch.Tensor,     # [B, 64] float32
        ppi_feat: torch.Tensor,     # [B, 256] float32
    ) -> torch.Tensor:
        # Project symbol to 256-dim
        sym_proj = self.sym_proj(sym_feat)  # [B, 256]

        # Stack 4 tokens → [B, 4, 256]
        tokens = torch.stack([global_emb, pert_emb, sym_proj, ppi_feat], dim=1)

        # Add token-type positional embeddings
        tokens = tokens + self.pos_emb.unsqueeze(0)  # [B, 4, 256]

        # Transformer self-attention (all tokens attend to all others)
        fused = self.transformer(tokens)  # [B, 4, 256]

        # Mean pool over tokens + normalize
        fused = self.output_norm(fused.mean(dim=1))  # [B, 256]
        return fused


# ──────────────────────────────────────────────────────────────────────────────
# DEG Model (Cross-Attention Fusion — 4-Layer, Width-Constrained FFN + Dropout)
# ──────────────────────────────────────────────────────────────────────────────
class AIDOCellCrossAttnDEGModel(nn.Module):
    """
    4-layer cross-attention feature fusion DEG predictor (width-constrained FFN + dropout):
      (a) AIDO.Cell-10M global mean-pool → [B, 256]
      (b) AIDO.Cell-10M perturbed-gene positional embedding → [B, 256]
      (c) Character-level Symbol CNN (64-dim) → projected to [B, 256]
      (d) Frozen STRING GNN PPI embedding → [B, 256]
      Fusion: FeatureFusionTransformer([a,b,c,d], n_layers=4, dim_ff=256,
                                        nhead=8, attn_dropout=0.2) → [B, 256]
      Head: LayerNorm(256) → Linear(256→256) → GELU → Dropout(0.5)
            → Linear(256→19920) → reshape [B, 3, 6640]
    """

    HIDDEN_DIM = 256          # AIDO.Cell-10M hidden size
    SYMBOL_DIM = 64           # symbol CNN output dim
    PPI_DIM = 256             # STRING GNN embedding dim
    FUSED_DIM = 256           # cross-attention output dim

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        head_hidden: int = 256,
        head_dropout: float = 0.5,
        frozen_gnn_embs: Optional[torch.Tensor] = None,  # [18870, 256]
        attn_n_layers: int = 4,   # CHANGED: parent=3 → this=4 (deeper fusion)
        attn_n_heads: int = 8,    # unchanged from parent (8 heads)
        attn_dim_ff: int = 256,   # unchanged from parent (width-constrained)
        attn_dropout: float = 0.2,  # unchanged from parent (targeted reg)
    ):
        super().__init__()

        # ── AIDO.Cell-10M with LoRA on ALL 8 layers ──
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
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

        # ── Symbol CNN ──
        self.symbol_encoder = SymbolEncoder(out_dim=self.SYMBOL_DIM, embed_dim=32)

        # ── Frozen STRING GNN embedding table ──
        if frozen_gnn_embs is not None:
            self.register_buffer("gnn_emb_table", frozen_gnn_embs.float())
        else:
            self.register_buffer(
                "gnn_emb_table",
                torch.zeros(18870, self.PPI_DIM, dtype=torch.float32)
            )

        # Learnable fallback for genes not in STRING GNN
        self.gnn_fallback = nn.Parameter(torch.zeros(self.PPI_DIM))
        nn.init.trunc_normal_(self.gnn_fallback, std=0.02)

        # Trainable PPI projection (1-layer + GELU + LayerNorm)
        self.ppi_proj = nn.Sequential(
            nn.Linear(self.PPI_DIM, self.PPI_DIM),
            nn.GELU(),
            nn.LayerNorm(self.PPI_DIM),
        )
        nn.init.xavier_uniform_(self.ppi_proj[0].weight)
        nn.init.zeros_(self.ppi_proj[0].bias)

        # ── Cross-Attention Feature Fusion Transformer ──
        # KEY CHANGE: n_layers=4 (depth added vs parent's 3)
        # Unchanged: dim_ff=256 (width-constrained), attn_dropout=0.2, nhead=8
        self.fusion_transformer = FeatureFusionTransformer(
            d_model=CROSS_ATTN_DIM,
            n_heads=attn_n_heads,
            n_layers=attn_n_layers,
            dim_ff=attn_dim_ff,
            attn_dropout=attn_dropout,
            sym_in_dim=self.SYMBOL_DIM,
        )

        # ── Prediction head (256-dim, unchanged from parent) ──
        self.head = nn.Sequential(
            nn.LayerNorm(self.FUSED_DIM),
            nn.Linear(self.FUSED_DIM, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def encode(
        self,
        expr: torch.Tensor,          # [B, N_GENES_AIDO] float32
        gene_pos: torch.Tensor,      # [B] int64 (-1 if not in AIDO vocab)
        symbol_ids: torch.Tensor,    # [B, SYMBOL_MAX_LEN] int64
        gnn_idx: torch.Tensor,       # [B] int64 (-1 if not in STRING)
    ) -> torch.Tensor:
        """Returns fused features [B, 256]."""
        B = expr.shape[0]
        device = expr.device

        # ── AIDO.Cell backbone ──
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, N_GENES_AIDO+2, 256] (bfloat16)

        # (a) Global mean-pool over gene positions (exclude 2 summary tokens)
        gene_emb = lhs[:, :N_GENES_AIDO, :]       # [B, 19264, 256]
        global_emb = gene_emb.mean(dim=1)           # [B, 256]

        # (b) Perturbed-gene positional embedding (fallback to global_emb if not found)
        pert_emb = torch.zeros(B, self.HIDDEN_DIM, device=device, dtype=lhs.dtype)
        valid_aido = gene_pos >= 0
        if valid_aido.any():
            valid_pos = gene_pos[valid_aido]
            pert_emb[valid_aido] = lhs[valid_aido, valid_pos, :]
        pert_emb[~valid_aido] = global_emb[~valid_aido]

        # Cast backbone features to float32 for cross-attention
        global_emb = global_emb.float()   # [B, 256]
        pert_emb = pert_emb.float()        # [B, 256]

        # (c) Symbol CNN
        sym_feat = self.symbol_encoder(symbol_ids)  # [B, 64] float32

        # (d) Frozen STRING GNN PPI lookup + trainable projection
        ppi_emb = self.gnn_fallback.unsqueeze(0).expand(B, -1).clone()  # [B, 256]
        valid_gnn = gnn_idx >= 0
        if valid_gnn.any():
            with torch.no_grad():
                ppi_raw = self.gnn_emb_table[gnn_idx[valid_gnn]]  # [k, 256]
            ppi_emb[valid_gnn] = ppi_raw
        ppi_feat = self.ppi_proj(ppi_emb.to(device))  # [B, 256]

        # (e) Cross-attention fusion (4-layer, dim_ff=256, attn_dropout=0.2, nhead=8)
        fused = self.fusion_transformer(global_emb, pert_emb, sym_feat, ppi_feat)  # [B, 256]
        return fused

    def decode(self, fused: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, 3, 6640]."""
        B = fused.shape[0]
        logits = self.head(fused)               # [B, N_CLASSES * N_GENES_OUT]
        return logits.view(B, N_CLASSES, N_GENES_OUT)

    def forward(
        self,
        expr: torch.Tensor,
        gene_pos: torch.Tensor,
        symbol_ids: torch.Tensor,
        gnn_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(self.encode(expr, gene_pos, symbol_ids, gnn_idx))


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro-averaged F1 (matches calc_metric.py logic)."""
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


def save_predictions_to_file(
    preds: np.ndarray,
    idxs: np.ndarray,
    test_pert_ids: List[str],
    test_symbols: List[str],
    pred_path: Path,
    label: str = "",
) -> None:
    """Save [N, 3, 6640] predictions to TSV file."""
    rows = []
    for r, i in enumerate(idxs):
        rows.append({
            "idx": test_pert_ids[i],
            "input": test_symbols[i],
            "prediction": json.dumps(preds[r].tolist()),
        })
    pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
    print(f"[{label}] Test predictions saved → {pred_path} ({len(rows)} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        head_hidden: int = 256,
        head_dropout: float = 0.5,
        backbone_lr: float = 2e-4,
        head_lr: float = 6e-4,
        weight_decay: float = 0.10,        # unchanged from tree-best parent
        focal_gamma: float = 1.5,
        label_smoothing: float = 0.05,
        max_epochs: int = 150,             # CHANGED: parent=100 → this=150 (extended)
        plateau_patience: int = 12,        # unchanged from tree-best parent
        plateau_factor: float = 0.5,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        mixup_alpha: float = 0.3,          # unchanged from tree-best parent
        attn_n_layers: int = 4,            # CHANGED: parent=3 → this=4 (deeper fusion)
        attn_n_heads: int = 8,             # unchanged from parent (8 heads)
        attn_dim_ff: int = 256,            # unchanged from parent (width-constrained)
        attn_dropout: float = 0.2,         # unchanged from parent (targeted reg)
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCellCrossAttnDEGModel] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        # Accumulates (preds, idxs) tuples across multiple test runs for checkpoint averaging
        self._checkpoint_pred_list: List[Tuple[np.ndarray, np.ndarray]] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            # Get frozen GNN embeddings from datamodule (already computed in setup)
            frozen_gnn_embs = None
            if hasattr(self, "trainer") and self.trainer is not None:
                dm = getattr(self.trainer, "datamodule", None)
                if dm is not None and dm.frozen_gnn_embs is not None:
                    frozen_gnn_embs = dm.frozen_gnn_embs

            self.model = AIDOCellCrossAttnDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                head_hidden=self.hparams.head_hidden,
                head_dropout=self.hparams.head_dropout,
                frozen_gnn_embs=frozen_gnn_embs,
                attn_n_layers=self.hparams.attn_n_layers,
                attn_n_heads=self.hparams.attn_n_heads,
                attn_dim_ff=self.hparams.attn_dim_ff,
                attn_dropout=self.hparams.attn_dropout,
            )

            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print(
                f"[setup] Model: {trainable:,}/{total:,} params trainable "
                f"({100 * trainable / total:.2f}%)"
            )

            self.loss_fn = FocalLoss(
                gamma=self.hparams.focal_gamma,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        if stage in ("test", None):
            self._populate_test_metadata()

    def _populate_test_metadata(self) -> None:
        if hasattr(self, "trainer") and self.trainer is not None:
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "test_pert_ids") and dm.test_pert_ids:
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            expr=batch["expr"],
            gene_pos=batch["gene_pos"],
            symbol_ids=batch["symbol_ids"],
            gnn_idx=batch["gnn_idx"],
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, 3, 6640], labels: [B, 6640]
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                               # [B*6640]
        return self.loss_fn(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        labels = batch["label"]  # [B, 6640]

        # Get fused features before head (for manifold mixup)
        fused_feat = self.model.encode(
            expr=batch["expr"],
            gene_pos=batch["gene_pos"],
            symbol_ids=batch["symbol_ids"],
            gnn_idx=batch["gnn_idx"],
        )  # [B, 256]

        # Manifold Mixup on fused features (training only)
        # alpha=0.3 (unchanged from tree-best parent node3-1-3-1-1-1-1)
        if self.hparams.mixup_alpha > 0 and self.training and fused_feat.size(0) > 1:
            lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
            idx_shuffle = torch.randperm(fused_feat.size(0), device=fused_feat.device)
            mixed_feat = lam * fused_feat + (1.0 - lam) * fused_feat[idx_shuffle]
            labels_b = labels[idx_shuffle]

            logits = self.model.decode(mixed_feat)  # [B, 3, 6640]
            loss = (
                lam * self._compute_loss(logits, labels)
                + (1.0 - lam) * self._compute_loss(logits, labels_b)
            )
        else:
            logits = self.model.decode(fused_feat)  # [B, 3, 6640]
            loss = self._compute_loss(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, 6640]
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        lp = (torch.cat(self._val_preds, 0)
              if self._val_preds else torch.zeros(0, N_CLASSES, N_GENES_OUT))
        ll = (torch.cat(self._val_labels, 0)
              if self._val_labels else torch.zeros(0, N_GENES_OUT, dtype=torch.long))
        li = (torch.cat(self._val_indices, 0)
              if self._val_indices else torch.zeros(0, dtype=torch.long))

        ap = self.all_gather(lp)
        al = self.all_gather(ll)
        ai = self.all_gather(li)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
        labels = al.view(-1, N_GENES_OUT).cpu().numpy()
        idxs = ai.view(-1).cpu().numpy()
        _, uniq = np.unique(idxs, return_index=True)
        f1_val = compute_deg_f1(preds[uniq], labels[uniq])
        self.log("val_f1", f1_val, prog_bar=True, sync_dist=True)

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

        if self.trainer.is_global_zero:
            preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
            idxs = ai.view(-1).cpu().numpy()
            _, uniq = np.unique(idxs, return_index=True)
            preds_uniq = preds[uniq]
            idxs_uniq = idxs[uniq]
            order = np.argsort(idxs_uniq)
            preds_ordered = preds_uniq[order]
            idxs_ordered = idxs_uniq[order]

            # Accumulate for checkpoint averaging (file saved from main())
            self._checkpoint_pred_list.append((preds_ordered, idxs_ordered))
            self.print(
                f"[on_test_epoch_end] Collected predictions from "
                f"{len(self._checkpoint_pred_list)} checkpoint(s), "
                f"shape={preds_ordered.shape}"
            )

    def configure_optimizers(self):
        # Differential learning rates:
        #   backbone LoRA params: backbone_lr=2e-4 (proven)
        #   head + symbol encoder + ppi_proj + fusion_transformer: head_lr=6e-4 (proven)
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters()
            if p.requires_grad
        ]
        other_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not n.startswith("backbone.")
        ]

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
                {"params": other_params, "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        # ReduceLROnPlateau on val_f1 (mode=max)
        # patience=12 (unchanged from tree-best parent): should not fire during
        # 150-epoch training — this is desirable, letting the model converge naturally.
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=self.hparams.plateau_factor,
            patience=self.hparams.plateau_patience,
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
        full_state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_state_dict:
                    trainable_state_dict[key] = full_state_dict[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_state_dict:
                trainable_state_dict[key] = full_state_dict[key]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable:,}/{total:,} params "
            f"({100 * trainable / total:.2f}%), plus {buffers:,} buffer values"
        )
        return trainable_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable parameters from partial checkpoint."""
        full_state_keys = set(super().state_dict().keys())
        trainable_keys = {
            name for name, p in self.named_parameters() if p.requires_grad
        }
        buffer_keys = {
            name for name, _ in self.named_buffers() if name in full_state_keys
        }
        expected_keys = trainable_keys | buffer_keys
        missing = [k for k in expected_keys if k not in state_dict]
        unexpected = [k for k in state_dict if k not in expected_keys]
        if missing:
            self.print(f"Warning: Missing checkpoint keys: {missing[:5]}...")
        if unexpected:
            self.print(f"Warning: Unexpected keys: {unexpected[:5]}...")
        loaded_t = len([k for k in state_dict if k in trainable_keys])
        loaded_b = len([k for k in state_dict if k in buffer_keys])
        self.print(
            f"Loading checkpoint: {loaded_t} trainable params + {loaded_b} buffers"
        )
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 3-1-3-1-1-1-1-1: 4-Layer Cross-Attention, dim_ff=256, nhead=8, 150 epochs"
    )
    # Use cwd() instead of __file__ to avoid symlink resolution issues.
    # Running from working_node_{i}/, the data/ directory is directly accessible.
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(Path.cwd() / "data"),
    )
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=150)             # CHANGED: 100→150
    p.add_argument("--backbone-lr", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.5)
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--plateau-patience", type=int, default=12)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--early-stopping-patience", type=int, default=30)  # CHANGED: 20→30
    p.add_argument("--early-stopping-min-delta", type=float, default=0.002)
    p.add_argument("--mixup-alpha", type=float, default=0.3,
                   help="Manifold mixup alpha (0.0 to disable)")
    p.add_argument("--top-k-avg", type=int, default=5,               # CHANGED: 3→5
                   help="Number of top checkpoints to average for test predictions")
    p.add_argument("--attn-n-layers", type=int, default=4,           # CHANGED: 3→4
                   help="Number of Transformer encoder layers in cross-attention fusion")
    p.add_argument("--attn-n-heads", type=int, default=8,
                   help="Number of attention heads in cross-attention fusion")
    p.add_argument("--attn-dim-ff", type=int, default=256,
                   help="Feedforward dim in cross-attention fusion transformer")
    p.add_argument("--attn-dropout", type=float, default=0.2,
                   help="Dropout in cross-attention fusion transformer")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(0)
    args = parse_args()

    # Use cwd() to ensure output_dir points to the working directory, not through
    # the mcts/node symlink to the actual node path. This ensures all outputs
    # (checkpoints, logs, predictions) go to working_node_{i}/mcts/node/run/.
    output_dir = Path.cwd() / "mcts" / "node" / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = args.fast_dev_run  # bool: True/False from store_true action
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    # When --debug-max-step is given, limit all stages to that many batches.
    # When --fast-dev-run is given (without --debug-max-step), Lightning's
    # fast_dev_run feature handles unit-test mode (1 batch per stage).
    limit = args.debug_max_step if args.debug_max_step is not None else 1.0

    # save_top_k=5 for checkpoint averaging (expanded from parent's 3)
    top_k = max(1, args.top_k_avg)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node3-1-3-1-1-1-1-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=top_k,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stopping_patience,  # 30 (extended for 150-epoch training)
        min_delta=args.early_stopping_min_delta,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(seconds=120),
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit,
        limit_val_batches=limit if not fast_dev_run else 1,
        limit_test_batches=limit,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
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
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        mixup_alpha=args.mixup_alpha,
        attn_n_layers=args.attn_n_layers,
        attn_n_heads=args.attn_n_heads,
        attn_dim_ff=args.attn_dim_ff,
        attn_dropout=args.attn_dropout,
    )

    trainer.fit(model_module, datamodule=datamodule)

    # ── Test phase with optional top-k checkpoint averaging ──
    # Reset the accumulated prediction list before test runs
    model_module._checkpoint_pred_list = []

    is_debug = args.fast_dev_run or args.debug_max_step is not None

    if is_debug:
        # Debug mode: single test run, no averaging
        trainer.test(model_module, datamodule=datamodule)
    else:
        # Production mode: top-k checkpoint averaging (k=5 expanded from parent's 3)
        best_k_models = checkpoint_cb.best_k_models  # {path: score}
        if not best_k_models:
            # Fallback: use best checkpoint
            trainer.test(model_module, datamodule=datamodule, ckpt_path="best")
        else:
            # Sort checkpoints by score (descending) and take top-k
            sorted_ckpts = sorted(
                best_k_models.keys(),
                key=lambda p: best_k_models[p],
                reverse=True
            )
            selected_ckpts = sorted_ckpts[:top_k]

            print(
                f"[Test] Running top-{len(selected_ckpts)} checkpoint averaging "
                f"(of {len(best_k_models)} saved checkpoints):"
            )
            for i, ckpt_path in enumerate(selected_ckpts):
                score = best_k_models[ckpt_path]
                print(f"  [{i+1}] {Path(ckpt_path).name} (val_f1={score:.4f})")

            for ckpt_path in selected_ckpts:
                trainer.test(model_module, datamodule=datamodule, ckpt_path=ckpt_path)

    # ── Save final (averaged) predictions ──
    if trainer.is_global_zero and model_module._checkpoint_pred_list:
        n_ckpts = len(model_module._checkpoint_pred_list)
        pred_path = output_dir / "test_predictions.tsv"

        if n_ckpts == 1:
            preds, idxs = model_module._checkpoint_pred_list[0]
            save_predictions_to_file(
                preds, idxs,
                datamodule.test_pert_ids, datamodule.test_symbols,
                pred_path, label="SingleCkpt"
            )
        else:
            # Average softmax probabilities across checkpoints
            all_preds = np.stack([p for p, i in model_module._checkpoint_pred_list], axis=0)
            avg_preds = all_preds.mean(axis=0)  # [N, 3, 6640]
            # Use indices from first checkpoint (all should be identical after dedup+sort)
            _, idxs_ref = model_module._checkpoint_pred_list[0]
            save_predictions_to_file(
                avg_preds, idxs_ref,
                datamodule.test_pert_ids, datamodule.test_symbols,
                pred_path, label=f"Avg-{n_ckpts}Ckpts"
            )
            print(f"[Test] Checkpoint averaging: {n_ckpts} checkpoints merged → {pred_path}")

    # Synchronize all ranks after file saving
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
