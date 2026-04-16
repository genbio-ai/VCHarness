#!/usr/bin/env python3
"""
Node 2-1-2-1-1: Cross-Attention Fusion with Manifold Mixup
============================================================
This node represents a fundamental architectural departure from the parent (node2-1-2-1,
test F1=0.4335) and grandparent (node2-1-2, test F1=0.4453) concat+MLP paradigm.

Key design motivation:
The parent's feedback explicitly recommended reverting to proven configurations.
However, the deeper tree analysis reveals that the concat+MLP paradigm has a hard
ceiling at ~0.462 (node3-2), while the cross-attention + manifold mixup architecture
achieves 0.4768–0.4843 (node3-1-3-1-1-1-1, node2-2-1-1-2-1-1-1).

Architecture change: concat+MLP → 4-token cross-attention TransformerEncoder fusion
This directly addresses the ceiling by modeling inter-source feature interactions
(PPI connectivity modulating cell-state interpretation, gene symbol patterns
informing positional attention).

Key design choices:
  1. LoRA r=8, last 4 layers (vs parent's r=4 all-8)
     - Parent feedback: r=8 last-4 achieved 0.4453 (parent's parent); per-layer
       capacity matters more than layer coverage
     - Cross-attention nodes that used r=4 all-8 achieved 0.4768/0.4843

  2. 3-layer TransformerEncoder fusion (d_model=256, nhead=4, dim_ff=256)
     - node3-1-3-1-1-1-1 (0.4768): 3-layer, dim_ff=256, nhead=8
     - node2-2-1-1-2-1-1-1 (0.4692): 3-layer, dim_ff=384, nhead=8
     - node2-2-1-1-2-1 (0.4843): 3-layer, dim_ff=256, nhead=8
     - 4 input tokens: [global_emb, pert_emb, ppi_feat, sym_emb]
     - Mean-pool output → 256-dim representation

  3. Manifold mixup (alpha=0.2) on the 4-token fusion space
     - Proven in node2-2-1-1-2 (0.4724 vs no-mixup 0.4472, +0.028)
     - Critical ingredient for breaking the 0.447 concat+MLP ceiling
     - alpha=0.2 is more conservative than alpha=0.3 (less label noise)

  4. Focal loss: gamma=1.5, class_weights=[6,1,12], label_smoothing=0.05
     - node2-2-1-1-2-1-1-1 used [6,1,12] with gamma=1.5 for 0.4692
     - Stronger minority class emphasis than parent's [5,1,10] with gamma=2.0

  5. Weight decay = 0.10 (vs parent's 0.03)
     - Proven in cross-attention nodes: node3-1-3-1-1-1-1 (wd=0.10, 0.4768)
     - With manifold mixup already providing regularization, higher WD prevents
       overfitting without capping the feature learning ceiling

  6. ReduceLROnPlateau: patience=12, factor=0.5, monitor val_f1 (max)
     - Cross-attention lineage uses patience=12 for deeper exploration
     - val_f1 monitoring remains critical (val_loss monitoring causes premature
       reductions, confirmed by node3-2-2 regression)

  7. Top-3 checkpoint averaging at test time
     - Provides +0.002-0.005 F1 at negligible computational cost
     - Not used in parent, widely proven across tree
     - Note: In DDP, only rank 0 performs averaging

  8. Early stopping patience=25 on val_f1
     - Allows ~2 full ReduceLROnPlateau cycles with patience=12

  9. STRING GNN projection: 256→256 single-layer (preserved from parent)
     - Single-layer projection is proven (node3-2-1 showed 2-layer regresses)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
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
N_GENES_OUT = 6_640      # output genes (task target)
N_CLASSES = 3
SENTINEL_EXPR = 1.0      # baseline expression (non-perturbed genes)
KNOCKOUT_EXPR = 0.0      # expression for knocked-out gene
AIDO_HIDDEN = 256        # AIDO.Cell-10M hidden dimension
AIDO_N_LAYERS = 8        # AIDO.Cell-10M transformer layers

# Class weights following cross-attention best practice from node2-2-1-1-2-1 lineage
CLASS_WEIGHTS = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)

# Character-level encoding for gene symbols
SYMBOL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-." + "_"  # + padding
CHAR_VOCAB_SIZE = len(SYMBOL_CHARS) + 1  # +1 for unknown
SYMBOL_MAX_LEN = 30  # max symbol length (padded/truncated to this)

# STRING GNN constants
STRING_GNN_DIM = 256    # STRING GNN hidden dimension
STRING_PROJ_DIM = 256   # Full 256-dim (single-layer, no bottleneck)

# Cross-attention fusion constants
FUSION_D_MODEL = 256    # matches AIDO_HIDDEN
FUSION_N_HEADS = 4      # 4 heads × 64-dim per head = 256-dim
FUSION_N_LAYERS = 3     # 3-layer TransformerEncoder (proven in node3-1-3-1-1-1-1)
FUSION_DIM_FF = 256     # width-constrained FFN (node2-2-1-1-2-1 used 256)
FUSION_ATTN_DROPOUT = 0.1  # attention dropout


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
        ce = F.cross_entropy(
            logits, targets, weight=w,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # Compute pt WITHOUT label smoothing for focal weighting (standard practice)
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Character-level symbol encoder
# ──────────────────────────────────────────────────────────────────────────────
def encode_symbol(symbol: str, max_len: int = SYMBOL_MAX_LEN) -> List[int]:
    """Encode a gene symbol string to character indices."""
    char_to_idx = {c: i + 1 for i, c in enumerate(SYMBOL_CHARS)}  # 0 = unknown
    encoded = [char_to_idx.get(c.upper(), 0) for c in symbol[:max_len]]
    encoded = encoded + [0] * (max_len - len(encoded))
    return encoded


class SymbolCNN(nn.Module):
    """3-branch character-level CNN for gene symbol encoding.

    Input: [B, max_len] character indices
    Output: [B, out_dim] symbol embedding
    """

    def __init__(
        self,
        vocab_size: int = CHAR_VOCAB_SIZE,
        embed_dim: int = 32,
        num_filters: int = 64,
        kernel_sizes: Tuple[int, ...] = (2, 3, 4),
        max_len: int = SYMBOL_MAX_LEN,
        out_dim: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.2)
        fusion_dim = num_filters * len(kernel_sizes)
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, max_len] int
        emb = self.embedding(x)         # [B, max_len, embed_dim]
        emb = self.dropout(emb)
        emb = emb.permute(0, 2, 1)      # [B, embed_dim, max_len]
        branch_outputs = []
        for conv in self.convs:
            c = conv(emb)                # [B, num_filters, L']
            c = F.gelu(c)
            c, _ = c.max(dim=-1)        # [B, num_filters]  global max-pool
            branch_outputs.append(c)
        fused = torch.cat(branch_outputs, dim=-1)  # [B, num_filters * n_branches]
        return self.proj(fused)         # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Returns AIDO.Cell expression tensors, perturbed-gene position, symbol char indices,
    STRING GNN index, and labels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],
        symbol_to_chars: Dict[str, List[int]],
        pert_to_gnn_idx: Dict[str, int],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.symbol_to_chars = symbol_to_chars
        self.pert_to_gnn_idx = pert_to_gnn_idx
        self.is_test = is_test

        # Pre-build expression input tensors: [N, 19264] float32
        self.expr_inputs = self._build_expr_tensors()

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Remap {-1,0,1} → {0,1,2} per metric contract
            self.labels = np.array(raw_labels, dtype=np.int8) + 1
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

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pert_id = self.pert_ids[idx]
        symbol = self.symbols[idx]
        base = pert_id.split(".")[0]
        gene_pos = self.gene_to_pos.get(base, -1)
        sym_chars = self.symbol_to_chars.get(symbol, [0] * SYMBOL_MAX_LEN)
        gnn_idx = self.pert_to_gnn_idx.get(base, -1)
        item = {
            "idx": idx,
            "expr": self.expr_inputs[idx],               # [19264] float32
            "gene_pos": gene_pos,                         # int (-1 if not in vocab)
            "sym_chars": torch.tensor(sym_chars, dtype=torch.long),  # [max_len]
            "gnn_idx": gnn_idx,                           # int (-1 if not in STRING)
            "pert_id": pert_id,
            "symbol": symbol,
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "expr": torch.stack([b["expr"] for b in batch]),           # [B, 19264]
        "gene_pos": torch.tensor([b["gene_pos"] for b in batch], dtype=torch.long),
        "sym_chars": torch.stack([b["sym_chars"] for b in batch]), # [B, max_len]
        "gnn_idx": torch.tensor([b["gnn_idx"] for b in batch], dtype=torch.long),
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
        self.symbol_to_chars: Dict[str, List[int]] = {}
        self.pert_to_gnn_idx: Dict[str, int] = {}
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        # ── Tokenizer: rank 0 downloads first ──
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # ── Build ENSG→position mapping ──
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)

        # ── Build symbol → char indices mapping ──
        if not self.symbol_to_chars:
            all_symbols: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_symbols.extend(df["symbol"].tolist())
            for sym in set(all_symbols):
                self.symbol_to_chars[sym] = encode_symbol(sym)

        # ── Build ENSG → STRING GNN node index mapping ──
        if not self.pert_to_gnn_idx:
            self.pert_to_gnn_idx = self._build_gnn_idx_mapping()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self.gene_to_pos, self.symbol_to_chars,
                self.pert_to_gnn_idx,
            )
            self.val_ds = PerturbationDataset(
                val_df, self.gene_to_pos, self.symbol_to_chars,
                self.pert_to_gnn_idx,
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self.gene_to_pos, self.symbol_to_chars,
                self.pert_to_gnn_idx, is_test=True,
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

    @staticmethod
    def _build_gnn_idx_mapping() -> Dict[str, int]:
        """Build ENSG→STRING GNN node index from node_names.json."""
        node_names_path = Path(STRING_GNN_MODEL_DIR) / "node_names.json"
        if not node_names_path.exists():
            return {}
        node_names = json.loads(node_names_path.read_text())
        mapping: Dict[str, int] = {}
        for i, name in enumerate(node_names):
            base = name.split(".")[0]
            mapping[base] = i
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
# Cross-Attention Multi-Source DEG Model
# ──────────────────────────────────────────────────────────────────────────────
class CrossAttentionDEGModel(nn.Module):
    """AIDO.Cell-10M (LoRA r=8 last-4) + Symbol CNN + Frozen STRING GNN
    with 3-layer TransformerEncoder cross-attention fusion.

    Architecture:
    - 4 input tokens: [global_emb(256), pert_emb(256), ppi_proj(256), sym_proj(256)]
    - Each projected to d_model=256 via linear layers
    - 3-layer TransformerEncoder (nhead=4, dim_ff=256, attn_dropout=0.1)
    - Mean-pool over 4 tokens → 256-dim representation
    - Head: 256→256→19920 with dropout=0.4
    """

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        head_dropout: float = 0.4,
        string_gnn_emb: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA on last 4 layers ──
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA r=8 on QKV of LAST 4 LAYERS (layers 4-7)
        # Proven in parent's parent (node2-1-2): test F1=0.4453
        # Per-layer capacity (r=8) matters more than coverage for this 1500-sample task
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=[4, 5, 6, 7],  # last 4 of 8 layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # ── Input projections to d_model=256 ──
        # backbone features: [B, 512] → 2 tokens of [B, 256] (already 256-dim)
        # symbol CNN: [B, 64] → project to [B, 256]
        # STRING GNN: [B, 256] → project to [B, 256] (single-layer)
        self.sym_proj = nn.Linear(64, FUSION_D_MODEL)
        self.gnn_proj = nn.Sequential(
            nn.Linear(STRING_GNN_DIM, FUSION_D_MODEL),
            nn.GELU(),
        )

        # ── Symbol CNN ──
        self.symbol_cnn = SymbolCNN(
            vocab_size=CHAR_VOCAB_SIZE,
            embed_dim=32,
            num_filters=64,
            kernel_sizes=(2, 3, 4),
            max_len=SYMBOL_MAX_LEN,
            out_dim=64,
        )

        # ── STRING GNN frozen embeddings ──
        if string_gnn_emb is not None:
            self.register_buffer("gnn_emb", string_gnn_emb.float())
        else:
            self.register_buffer("gnn_emb", None)

        # ── 3-layer TransformerEncoder for cross-modal fusion ──
        # Input: sequence of 4 tokens, each [B, 256]
        # dim_ff=256 (width-constrained, avoids over-parameterization)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FUSION_D_MODEL,
            nhead=FUSION_N_HEADS,
            dim_feedforward=FUSION_DIM_FF,
            dropout=FUSION_ATTN_DROPOUT,
            activation="gelu",
            batch_first=True,  # [B, seq_len, d_model]
            norm_first=False,  # post-norm (standard)
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=FUSION_N_LAYERS,
            norm=nn.LayerNorm(FUSION_D_MODEL),
        )

        # ── Prediction head: 256→256→19920 ──
        # Simpler than parent's 832→384→19920 but after cross-attention fusion
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_D_MODEL),
            nn.Linear(FUSION_D_MODEL, FUSION_D_MODEL),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(FUSION_D_MODEL),
            nn.Linear(FUSION_D_MODEL, N_CLASSES * N_GENES_OUT),
        )
        # Initialize output layer conservatively
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        expr: torch.Tensor,       # [B, 19264] float32
        gene_pos: torch.Tensor,   # [B]        int64  (-1 if not in vocab)
        sym_chars: torch.Tensor,  # [B, max_len] int64
        gnn_idx: torch.Tensor,    # [B]          int64  (-1 if not in STRING)
    ) -> torch.Tensor:
        # ── AIDO.Cell backbone ──
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, 19266, 256]

        # (a) Global mean-pool over all 19264 gene positions
        gene_emb = lhs[:, :N_GENES_AIDO, :]       # [B, 19264, 256]
        global_emb = gene_emb.mean(dim=1).float()  # [B, 256]

        # (b) Perturbed-gene positional embedding
        B = expr.shape[0]
        pert_emb = torch.zeros(B, AIDO_HIDDEN, device=lhs.device, dtype=torch.float32)
        valid_mask = gene_pos >= 0
        if valid_mask.any():
            valid_pos = gene_pos[valid_mask]
            pert_emb[valid_mask] = lhs[valid_mask, valid_pos, :].float()
        pert_emb[~valid_mask] = global_emb[~valid_mask]

        # ── Symbol CNN → project to 256-dim ──
        sym_emb = self.symbol_cnn(sym_chars).float()  # [B, 64]
        sym_proj = self.sym_proj(sym_emb)              # [B, 256]

        # ── STRING GNN feature → project to 256-dim ──
        if self.gnn_emb is not None:
            gnn_feats = torch.zeros(B, STRING_GNN_DIM, device=expr.device, dtype=torch.float32)
            valid_gnn_mask = gnn_idx >= 0
            if valid_gnn_mask.any():
                valid_gnn_idx = gnn_idx[valid_gnn_mask]
                gnn_feats[valid_gnn_mask] = self.gnn_emb[valid_gnn_idx].float()
            ppi_feat = self.gnn_proj(gnn_feats)  # [B, 256]
        else:
            ppi_feat = torch.zeros(B, FUSION_D_MODEL, device=expr.device, dtype=torch.float32)

        # ── 4-token cross-attention fusion ──
        # Tokens: [global_emb, pert_emb, ppi_feat, sym_proj]
        # Each is [B, 256] → stack to [B, 4, 256]
        tokens = torch.stack([global_emb, pert_emb, ppi_feat, sym_proj], dim=1)  # [B, 4, 256]
        fused = self.fusion_transformer(tokens)  # [B, 4, 256]
        fused = fused.mean(dim=1)                # [B, 256] — mean pool over tokens

        # ── Prediction head ──
        logits = self.head(fused)                          # [B, 3*6640]
        return logits.view(B, N_CLASSES, N_GENES_OUT)      # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# STRING GNN frozen inference
# ──────────────────────────────────────────────────────────────────────────────
def compute_frozen_gnn_embeddings() -> Optional[torch.Tensor]:
    """Run frozen STRING GNN inference on the full graph once."""
    model_dir = Path(STRING_GNN_MODEL_DIR)
    if not (model_dir / "graph_data.pt").exists():
        print(f"WARNING: STRING GNN graph_data.pt not found at {model_dir}. Skipping STRING GNN feature.")
        return None

    try:
        gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
        gnn_model.eval()
        graph = torch.load(str(model_dir / "graph_data.pt"), map_location="cpu")
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)

        with torch.no_grad():
            outputs = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
        emb = outputs.last_hidden_state  # [N_nodes, 256]
        print(f"STRING GNN embeddings computed: shape {emb.shape}")
        return emb.cpu()
    except Exception as e:
        print(f"WARNING: STRING GNN inference failed: {e}. Skipping STRING GNN feature.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro F1, averaged over all genes.

    y_pred: [n_samples, 3, n_genes]   (3-class logits or probabilities)
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
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        head_dropout: float = 0.4,
        lr: float = 2e-4,
        weight_decay: float = 0.10,
        gamma_focal: float = 1.5,
        label_smoothing: float = 0.05,
        max_epochs: int = 120,
        rlop_patience: int = 12,
        rlop_factor: float = 0.5,
        mixup_alpha: float = 0.2,
        head_lr_multiplier: float = 3.0,
        top_k_checkpoint: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[CrossAttentionDEGModel] = None
        self.criterion: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        # GNN embeddings computed once at setup
        self._gnn_emb: Optional[torch.Tensor] = None
        # Top-K checkpoint tracking for averaging
        self._checkpoint_paths: List[str] = []
        self._checkpoint_val_f1s: List[float] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            # Compute STRING GNN embeddings once (frozen inference)
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

            if world_size == 1:
                gnn_emb = compute_frozen_gnn_embeddings()
            else:
                if local_rank == 0:
                    gnn_emb = compute_frozen_gnn_embeddings()
                else:
                    gnn_emb = None

                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    has_gnn = torch.tensor(
                        [1 if gnn_emb is not None else 0],
                        dtype=torch.long,
                        device=torch.cuda.current_device(),
                    )
                    torch.distributed.broadcast(has_gnn, src=0)
                    if has_gnn.item() == 1:
                        if local_rank == 0:
                            emb_shape = torch.tensor(
                                list(gnn_emb.shape),
                                dtype=torch.long,
                                device=torch.cuda.current_device(),
                            )
                        else:
                            emb_shape = torch.zeros(
                                2, dtype=torch.long, device=torch.cuda.current_device()
                            )
                        torch.distributed.broadcast(emb_shape, src=0)
                        if local_rank != 0:
                            gnn_emb = torch.zeros(
                                emb_shape[0].item(),
                                emb_shape[1].item(),
                                dtype=torch.float32,
                            )
                        gnn_emb_device = gnn_emb.to(torch.cuda.current_device())
                        torch.distributed.broadcast(gnn_emb_device, src=0)
                        gnn_emb = gnn_emb_device.cpu()
                    else:
                        gnn_emb = None

            self._gnn_emb = gnn_emb

            self.model = CrossAttentionDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                head_dropout=self.hparams.head_dropout,
                string_gnn_emb=self._gnn_emb,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )
            # Cast trainable params to float32 for stable AdamW optimization
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        if stage == "test" and hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            batch["expr"], batch["gene_pos"],
            batch["sym_chars"], batch["gnn_idx"],
        )

    def _manifold_mixup(
        self,
        tokens: torch.Tensor,  # [B, 4, 256] - the fused token sequence
        labels: torch.Tensor,  # [B, 6640] - integer labels in {0,1,2}
        alpha: float = 0.2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Manifold mixup in the token space.

        Interpolates between two samples in the 4-token fusion space.
        Returns mixed tokens, original labels, shuffled labels, and lambda.
        """
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        B = tokens.shape[0]
        perm = torch.randperm(B, device=tokens.device)
        mixed = lam * tokens + (1 - lam) * tokens[perm]
        return mixed, labels, labels[perm], lam

    def _compute_loss_mixed(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Compute mixup loss: lam * loss_a + (1-lam) * loss_b."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_a_flat = labels_a.reshape(-1)
        labels_b_flat = labels_b.reshape(-1)
        loss_a = self.criterion(logits_flat, labels_a_flat)
        loss_b = self.criterion(logits_flat, labels_b_flat)
        return lam * loss_a + (1 - lam) * loss_b

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_flat = labels.reshape(-1)
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Extract features
        expr = batch["expr"]
        gene_pos = batch["gene_pos"]
        sym_chars = batch["sym_chars"]
        gnn_idx = batch["gnn_idx"]
        labels = batch["label"]

        # Forward pass to get backbone embeddings and tokens
        # Run backbone and extract features, then apply mixup before fusion
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.model.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state

        gene_emb = lhs[:, :N_GENES_AIDO, :]
        global_emb = gene_emb.mean(dim=1).float()

        B = expr.shape[0]
        pert_emb = torch.zeros(B, AIDO_HIDDEN, device=lhs.device, dtype=torch.float32)
        valid_mask = gene_pos >= 0
        if valid_mask.any():
            valid_pos = gene_pos[valid_mask]
            pert_emb[valid_mask] = lhs[valid_mask, valid_pos, :].float()
        pert_emb[~valid_mask] = global_emb[~valid_mask]

        sym_emb = self.model.symbol_cnn(sym_chars).float()
        sym_proj = self.model.sym_proj(sym_emb)

        if self.model.gnn_emb is not None:
            gnn_feats = torch.zeros(B, STRING_GNN_DIM, device=expr.device, dtype=torch.float32)
            valid_gnn_mask = gnn_idx >= 0
            if valid_gnn_mask.any():
                valid_gnn_idx = gnn_idx[valid_gnn_mask]
                gnn_feats[valid_gnn_mask] = self.model.gnn_emb[valid_gnn_idx].float()
            ppi_feat = self.model.gnn_proj(gnn_feats)
        else:
            ppi_feat = torch.zeros(B, FUSION_D_MODEL, device=expr.device, dtype=torch.float32)

        # Stack tokens: [B, 4, 256]
        tokens = torch.stack([global_emb, pert_emb, ppi_feat, sym_proj], dim=1)

        # Apply manifold mixup in token space (only during training)
        alpha = self.hparams.mixup_alpha
        if alpha > 0 and B > 1:
            mixed_tokens, labels_a, labels_b, lam = self._manifold_mixup(tokens, labels, alpha)
        else:
            mixed_tokens, labels_a, labels_b, lam = tokens, labels, labels, 1.0

        # Complete forward pass through fusion transformer and head
        fused = self.model.fusion_transformer(mixed_tokens)
        fused = fused.mean(dim=1)
        logits = self.model.head(fused)
        logits = logits.view(B, N_CLASSES, N_GENES_OUT)

        if lam < 1.0:
            loss = self._compute_loss_mixed(logits, labels_a, labels_b, lam)
        else:
            loss = self._compute_loss(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
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

        # After all_gather + np.unique deduplication, all ranks have identical f1.
        # The all_reduce is harmless (SUM / world_size == original value) but unnecessary.
        # Keep it for proper sync_dist=True logging behavior in Lightning.
        f1_tensor = torch.tensor(f1, dtype=torch.float32, device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(f1_tensor, op=torch.distributed.ReduceOp.SUM)
            f1_tensor = f1_tensor / max(self.trainer.world_size, 1)

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

        if self.global_rank == 0:
            preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
            idxs = ai.cpu().view(-1).numpy()
            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]
            idxs = idxs[uniq]
            order = np.argsort(idxs)
            preds = preds[order]
            idxs = idxs[order]

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for rank_i, orig_i in enumerate(idxs):
                rows.append({
                    "idx": self._test_pert_ids[orig_i],
                    "input": self._test_symbols[orig_i],
                    "prediction": json.dumps(preds[rank_i].tolist()),
                })
            out_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path}")

    def configure_optimizers(self):
        # Separate parameter groups with differential LR:
        # LoRA backbone params: lower LR (pretrained)
        # Symbol CNN + GNN projection + fusion transformer + head: higher LR (randomly initialized)
        backbone_params = [p for n, p in self.model.backbone.named_parameters()
                           if p.requires_grad]
        other_params = (
            list(self.model.symbol_cnn.parameters()) +
            list(self.model.sym_proj.parameters()) +
            list(self.model.gnn_proj.parameters()) +
            list(self.model.fusion_transformer.parameters()) +
            list(self.model.head.parameters())
        )

        max_lr_backbone = self.hparams.lr
        max_lr_other = self.hparams.lr * self.hparams.head_lr_multiplier

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": max_lr_backbone},
                {"params": other_params, "lr": max_lr_other},
            ],
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # ReduceLROnPlateau monitoring val_f1 (max):
        # CRITICAL: Do NOT monitor val_loss — node3-2-2 showed val_loss monitoring
        # caused premature LR reductions and regression from 0.462 to 0.442.
        # patience=12 follows cross-attention lineage (node3-1-3-1-1-1-1 used patience=12)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            patience=self.hparams.rlop_patience,
            factor=self.hparams.rlop_factor,
            min_lr=1e-6,
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
        """Save only trainable parameters and buffers."""
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        out = {}
        trainable_keys = set()
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_keys.add(prefix + name)
        for name, buf in self.named_buffers():
            trainable_keys.add(prefix + name)
        for k in full:
            if k in trainable_keys:
                out[k] = full[k]
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.parameters())
        self.print(
            f"Saving checkpoint: {trainable_count}/{total_count} trainable params "
            f"({100 * trainable_count / max(total_count, 1):.2f}%)"
        )
        return out

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Top-K checkpoint averaging for test inference
# ──────────────────────────────────────────────────────────────────────────────
def average_top_k_checkpoints(
    model_module: DEGLightningModule,
    checkpoint_callback: ModelCheckpoint,
    top_k: int = 3,
) -> None:
    """Average parameters from top-K checkpoints into the model.

    This is a post-training operation performed on rank 0 before test inference.
    Proven to provide +0.002-0.005 F1 benefit across multiple nodes in the tree.
    """
    # Get top-K checkpoint paths from the callback
    if not hasattr(checkpoint_callback, 'best_k_models') or len(checkpoint_callback.best_k_models) == 0:
        print("No checkpoints available for averaging, skipping.")
        return

    ckpt_paths = list(checkpoint_callback.best_k_models.keys())
    # Sort by score (higher is better for val_f1)
    ckpt_scores = {p: checkpoint_callback.best_k_models[p] for p in ckpt_paths}
    sorted_paths = sorted(ckpt_scores.keys(), key=lambda p: ckpt_scores[p], reverse=True)
    selected = sorted_paths[:min(top_k, len(sorted_paths))]

    if len(selected) < 2:
        print(f"Only {len(selected)} checkpoint(s) available, skipping averaging.")
        return

    print(f"Averaging {len(selected)} checkpoints with val_f1 scores: "
          f"{[float(ckpt_scores[p]) for p in selected]}")

    # Load and average state dicts
    avg_state = None
    n = 0
    for path in selected:
        try:
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            if avg_state is None:
                avg_state = {k: v.float().clone() for k, v in state.items()}
            else:
                for k in avg_state:
                    if k in state:
                        avg_state[k] += state[k].float()
            n += 1
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {path}: {e}")

    if avg_state is None or n == 0:
        print("No valid checkpoints loaded for averaging.")
        return

    for k in avg_state:
        avg_state[k] /= n

    # Load averaged state into model
    model_module.load_state_dict(avg_state, strict=False)
    print(f"Loaded averaged parameters from {n} checkpoints.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 2-1-2-1-1: Cross-Attention DEG predictor with Manifold Mixup"
    )
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--head-dropout", type=float, default=0.4)
    p.add_argument("--gamma-focal", type=float, default=1.5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--rlop-patience", type=int, default=12)
    p.add_argument("--rlop-factor", type=float, default=0.5)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--head-lr-multiplier", type=float, default=3.0)
    p.add_argument("--top-k-checkpoint", type=int, default=3)
    p.add_argument("--early-stopping-patience", type=int, default=25)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def main():
    pl.seed_everything(0)
    args = parse_args()

    # Resolve data_dir relative to project root
    if args.data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    else:
        data_dir = Path(args.data_dir)
    args.data_dir = str(data_dir)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit = args.debug_max_step if args.debug_max_step is not None else 1.0

    # Save top-K checkpoints (for averaging)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node2-1-2-1-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=args.top_k_checkpoint,
        save_last=True,
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
        limit_train_batches=limit,
        limit_val_batches=limit,
        limit_test_batches=limit,
        val_check_interval=(
            1.0 if (args.debug_max_step is not None or args.fast_dev_run)
            else args.val_check_interval
        ),
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
        lora_dropout=args.lora_dropout,
        head_dropout=args.head_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        rlop_patience=args.rlop_patience,
        rlop_factor=args.rlop_factor,
        mixup_alpha=args.mixup_alpha,
        head_lr_multiplier=args.head_lr_multiplier,
        top_k_checkpoint=args.top_k_checkpoint,
    )

    trainer.fit(model_module, datamodule=datamodule)

    # Top-K checkpoint averaging (rank 0 only, before test inference)
    is_rank_zero = (
        not torch.distributed.is_available() or
        not torch.distributed.is_initialized() or
        torch.distributed.get_rank() == 0
    )

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        # Load best checkpoint first
        best_ckpt = checkpoint_cb.best_model_path
        if best_ckpt and is_rank_zero and args.top_k_checkpoint > 1:
            # Attempt top-K averaging on rank 0
            average_top_k_checkpoints(model_module, checkpoint_cb, top_k=args.top_k_checkpoint)
            # Test with averaged model (no ckpt_path to avoid re-loading)
            test_results = trainer.test(model_module, datamodule=datamodule)
        else:
            test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # Save test score
    if is_rank_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"test_results: {test_results}\n"
            f"val_f1_best: {checkpoint_cb.best_model_score}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
