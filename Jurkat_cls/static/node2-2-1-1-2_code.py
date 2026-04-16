#!/usr/bin/env python3
"""
Node 1-2: Cross-Attention Fusion with AIDO.Cell-10M + STRING GNN + Symbol CNN + Manifold Mixup
================================================================================================
Pivots away from the saturated node2-2-1 lineage (~0.447 ceiling) to a fundamentally
different multi-source architecture inspired by the tree-best node3-1-1-1-1-1-2 (F1=0.5049).

Architecture:
  - AIDO.Cell-10M with LoRA (r=4, all 8 Q/K/V layers) for backbone embeddings
  - Frozen STRING GNN PPI embeddings for each perturbed gene (256-dim)
  - Character-level Symbol CNN (3-branch Conv1d -> 64-dim)
  - 3-layer TransformerEncoder cross-attention fusion over 4 tokens (global_emb,
    pert_emb, ppi_feat, sym_feat) -> mean-pool -> 256-dim
  - Manifold mixup (alpha=0.3) applied during training
  - FocalLoss (gamma=1.5, class_weights=[6,1,12], label_smoothing=0.05)
  - ReduceLROnPlateau (patience=8, factor=0.5) scheduler
  - Top-3 checkpoint averaging at test time
  - weight_decay=0.10

This directly addresses the fundamental bottleneck of the parent's synthetic one-hot paradigm,
replacing it with multi-modal biological information (PPI network + gene naming patterns) that
breaks the ~0.447 ceiling established across 5 generations of the node2-2 lineage.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Fix NCCL segfault on certain clusters with InfiniBand: disable IB transport for NCCL
os.environ.setdefault('NCCL_IB_DISABLE', '1')
os.environ.setdefault('NCCL_NET_GDR_LEVEL', '0')

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
N_GENES_AIDO = 19_264    # AIDO.Cell vocabulary size (fixed for all model sizes)
N_GENES_OUT = 6_640      # output genes
N_CLASSES = 3
SENTINEL_EXPR = 1.0      # baseline expression (non-perturbed genes)
KNOCKOUT_EXPR = 0.0      # expression for knocked-out gene (perturbed)
AIDO_HIDDEN = 256        # AIDO.Cell-10M hidden dimension
AIDO_N_LAYERS = 8        # AIDO.Cell-10M transformer layers

# Class weights for focal loss - calibrated for extreme imbalance
# Train distribution: class 0 (down) ~3.4%, class 1 (unchanged) ~95.5%, class 2 (up) ~1.1%
# Weights [6, 1, 12] inspired by node3-1-1-1-1-1-2 (tree-best, F1=0.5049)
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
    3-branch parallel Conv1d with global max-pool -> [B, out_dim]
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
        x = self.embed(symbol_ids)          # [B, L, embed_dim]
        x = x.transpose(1, 2)              # [B, embed_dim, L] for Conv1d
        f2 = F.relu(self.conv2(x))
        f3 = F.relu(self.conv3(x))
        f4 = F.relu(self.conv4(x))
        f2 = F.adaptive_max_pool1d(f2, 1).squeeze(-1)  # [B, 32]
        f3 = F.adaptive_max_pool1d(f3, 1).squeeze(-1)  # [B, 32]
        f4 = F.adaptive_max_pool1d(f4, 1).squeeze(-1)  # [B, 32]
        feat = torch.cat([f2, f3, f4], dim=-1)          # [B, 96]
        return self.proj(feat)                           # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Returns pre-built AIDO.Cell expression profile tensors together with
    the perturbed gene position index, gene symbol character indices, and label.
    No augmentation is applied (gene dropout was confirmed counterproductive).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],    # ENSG_base -> position in [0, 19264)
        ppi_emb_index: Dict[str, int],  # ENSG_base -> index in STRING GNN node_names
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.ppi_emb_index = ppi_emb_index
        self.is_test = is_test

        # Pre-build the AIDO.Cell expression profile tensors
        self.expr_inputs = self._build_expr_tensors()

        # Pre-build perturbed gene position indices
        self.pert_positions = self._build_pert_positions()

        # Pre-build PPI embedding indices (index into STRING GNN node matrix)
        self.ppi_indices = self._build_ppi_indices()

        # Pre-build symbol character index tensors
        self.symbol_ids = self._build_symbol_tensors()

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1} -> {0,1,2}
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

    def _build_pert_positions(self) -> List[int]:
        positions = []
        for pert_id in self.pert_ids:
            base = pert_id.split(".")[0]
            pos = self.gene_to_pos.get(base, -1)
            positions.append(pos)
        return positions

    def _build_ppi_indices(self) -> List[int]:
        """Map each pert_id to its STRING GNN node index (-1 if not found)."""
        indices = []
        for pert_id in self.pert_ids:
            base = pert_id.split(".")[0]
            idx = self.ppi_emb_index.get(base, -1)
            indices.append(idx)
        return indices

    def _build_symbol_tensors(self) -> torch.Tensor:
        N = len(self.symbols)
        sym_ids = torch.zeros((N, SYMBOL_MAX_LEN), dtype=torch.long)
        for i, symbol in enumerate(self.symbols):
            indices = symbol_to_indices(symbol)
            sym_ids[i] = torch.tensor(indices, dtype=torch.long)
        return sym_ids

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "expr": self.expr_inputs[idx],               # [19264] float32
            "gene_pos": self.pert_positions[idx],        # int (-1 if not in vocab)
            "ppi_idx": self.ppi_indices[idx],            # int (-1 if not in STRING GNN)
            "symbol_ids": self.symbol_ids[idx],          # [SYMBOL_MAX_LEN] int64
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "expr": torch.stack([b["expr"] for b in batch]),             # [B, 19264]
        "gene_pos": torch.tensor([b["gene_pos"] for b in batch], dtype=torch.long),
        "ppi_idx": torch.tensor([b["ppi_idx"] for b in batch], dtype=torch.long),
        "symbol_ids": torch.stack([b["symbol_ids"] for b in batch]),  # [B, SYMBOL_MAX_LEN]
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
        self.ppi_emb_index: Dict[str, int] = {}
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        # Rank-0 downloads tokenizer first, then barrier
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # Build ENSG -> position mapping from all splits
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)

        # Build ENSG -> STRING GNN index mapping
        if not self.ppi_emb_index:
            self.ppi_emb_index = self._build_ppi_index()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self.gene_to_pos, self.ppi_emb_index,
            )
            self.val_ds = PerturbationDataset(
                val_df, self.gene_to_pos, self.ppi_emb_index,
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self.gene_to_pos, self.ppi_emb_index, is_test=True,
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
    def _build_ppi_index() -> Dict[str, int]:
        """Build mapping from ENSG gene ID to STRING GNN node index."""
        import json as _json
        node_names_path = Path(STRING_GNN_MODEL_DIR) / "node_names.json"
        node_names = _json.loads(node_names_path.read_text())
        # node_names[i] contains the Ensembl gene ID (format: ENSG00000000003)
        mapping: Dict[str, int] = {}
        for i, ensg_id in enumerate(node_names):
            # Strip version suffix if any (e.g., ENSG00000000003.1 -> ENSG00000000003)
            base = ensg_id.split(".")[0]
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
# Model: Cross-Attention Fusion
# ──────────────────────────────────────────────────────────────────────────────
class CrossAttnFusionDEGModel(nn.Module):
    """
    Multi-source cross-attention fusion DEG predictor.

    Four input sources fused via TransformerEncoder self-attention:
      (a) global_emb: global mean-pool of AIDO.Cell-10M last_hidden_state [B, 256]
      (b) pert_emb: perturbed gene positional embedding [B, 256]
      (c) ppi_feat: STRING GNN PPI embedding for perturbed gene [B, 256]
      (d) sym_feat: gene symbol character CNN embedding [B, 64 -> 256 via proj]

    Fusion: 4 tokens [B, 4, 256] -> 3-layer TransformerEncoder (nhead=8, dim_ff=384)
            -> mean-pool -> [B, 256]
    Head: 256 -> 256 -> N_CLASSES * N_GENES_OUT

    Architecture inspired by node3-1-1-1-1-1-2 (test F1=0.5049, tree-best).
    """

    HIDDEN_DIM = 256          # AIDO.Cell-10M hidden dimension / fusion dimension
    SYMBOL_DIM = 64           # symbol CNN output
    FUSION_DIM = 256          # unified fusion token dimension

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_layers_start: int = 0,  # all 8 layers: indices 0,1,...,7
        fusion_layers: int = 3,      # number of TransformerEncoder layers
        fusion_heads: int = 8,       # number of attention heads
        fusion_ff_dim: int = 384,    # feed-forward dim in TransformerEncoder
        fusion_attn_dropout: float = 0.1,
        head_hidden: int = 256,
        head_dropout: float = 0.4,
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA ──
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA on Q/K/V of all 8 layers
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],
            layers_to_transform=list(range(lora_layers_start, AIDO_N_LAYERS)),
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Cast trainable LoRA params to float32 for training stability
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Frozen STRING GNN backbone ──
        self.string_gnn = AutoModel.from_pretrained(
            STRING_GNN_MODEL_DIR, trust_remote_code=True,
        )
        # Freeze STRING GNN (use as static embedding extractor)
        for param in self.string_gnn.parameters():
            param.requires_grad = False

        # ── Gene symbol character-level CNN encoder ──
        self.symbol_encoder = SymbolEncoder(out_dim=self.SYMBOL_DIM, embed_dim=32)

        # ── Project symbol embedding to fusion dimension ──
        self.sym_proj = nn.Sequential(
            nn.Linear(self.SYMBOL_DIM, self.FUSION_DIM),
            nn.GELU(),
            nn.LayerNorm(self.FUSION_DIM),
        )

        # ── Cross-attention fusion: 4 tokens x fusion_dim ──
        # Using TransformerEncoder for self-attention over the 4 input tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.FUSION_DIM,
            nhead=fusion_heads,
            dim_feedforward=fusion_ff_dim,
            dropout=fusion_attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stable training
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=fusion_layers,
        )

        # ── Prediction head: FUSION_DIM -> head_hidden -> N_CLASSES * N_GENES_OUT ──
        self.head = nn.Sequential(
            nn.LayerNorm(self.FUSION_DIM),
            nn.Linear(self.FUSION_DIM, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        # Conservative initialization for output layer
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        # Store graph data for STRING GNN (precomputed at setup time)
        self._graph_edge_index: Optional[torch.Tensor] = None
        self._graph_edge_weight: Optional[torch.Tensor] = None
        self._ppi_embeddings: Optional[torch.Tensor] = None  # [N_STRING, 256]

    def compute_ppi_embeddings(self, device: torch.device) -> None:
        """Precompute frozen STRING GNN embeddings once, then cache them on CPU.
        The STRING GNN is run once on GPU then moved back to CPU to free memory.
        """
        if self._ppi_embeddings is not None:
            return
        graph_data = torch.load(
            Path(STRING_GNN_MODEL_DIR) / "graph_data.pt", map_location=device
        )
        edge_index = graph_data["edge_index"].to(device)
        edge_weight = graph_data.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        # Move STRING GNN to GPU temporarily for embedding computation
        self.string_gnn = self.string_gnn.to(device)
        with torch.no_grad():
            outputs = self.string_gnn(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
        # Cache embedding matrix on CPU to avoid GPU memory overhead during training
        self._ppi_embeddings = outputs.last_hidden_state.detach().cpu().float()
        # Move STRING GNN back to CPU after embedding extraction to free GPU memory
        self.string_gnn = self.string_gnn.cpu()
        del edge_index, edge_weight, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(
        self,
        expr: torch.Tensor,          # [B, 19264] float32
        gene_pos: torch.Tensor,      # [B]         int64  (-1 if not in vocab)
        ppi_idx: torch.Tensor,       # [B]         int64  (-1 if not in STRING GNN)
        symbol_ids: torch.Tensor,    # [B, SYMBOL_MAX_LEN] int64
    ) -> torch.Tensor:
        B = expr.shape[0]
        device = expr.device

        # ── (a) Backbone embeddings ──
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, 19266, 256]

        gene_emb = lhs[:, :N_GENES_AIDO, :]           # [B, 19264, 256]
        global_emb = gene_emb.mean(dim=1).float()     # [B, 256]

        # Perturbed gene positional embedding
        pert_emb = torch.zeros(B, self.HIDDEN_DIM, device=device, dtype=torch.float32)
        valid_mask = gene_pos >= 0
        if valid_mask.any():
            valid_pos = gene_pos[valid_mask]
            pert_emb[valid_mask] = lhs[valid_mask, valid_pos, :].float()
        pert_emb[~valid_mask] = global_emb[~valid_mask]

        # ── (b) PPI embeddings from precomputed STRING GNN ──
        if self._ppi_embeddings is None:
            self.compute_ppi_embeddings(device)

        ppi_emb_all = self._ppi_embeddings.to(device)  # [N_STRING, 256]
        ppi_feat = torch.zeros(B, self.FUSION_DIM, device=device, dtype=torch.float32)
        valid_ppi = ppi_idx >= 0
        if valid_ppi.any():
            ppi_feat[valid_ppi] = ppi_emb_all[ppi_idx[valid_ppi]]
        # For genes not in STRING GNN, use global_emb as fallback
        ppi_feat[~valid_ppi] = global_emb[~valid_ppi]

        # ── (c) Symbol CNN embedding ──
        sym_feat = self.symbol_encoder(symbol_ids)    # [B, 64] float32
        sym_projected = self.sym_proj(sym_feat)       # [B, 256]

        # ── Cross-attention fusion ──
        # Stack 4 tokens: [B, 4, 256]
        tokens = torch.stack([global_emb, pert_emb, ppi_feat, sym_projected], dim=1)
        fused = self.fusion_transformer(tokens)        # [B, 4, 256]
        fused_pool = fused.mean(dim=1)                # [B, 256]

        # ── Prediction head ──
        logits = self.head(fused_pool)                # [B, 3 * 6640]
        return logits.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# Manifold Mixup
# ──────────────────────────────────────────────────────────────────────────────
def manifold_mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Apply manifold mixup in the latent space.
    x: [B, ...] feature tensor
    y: [B, N_GENES_OUT] integer labels
    Returns: (mixed_x, mixed_y_onehot_or_soft, lam)
    """
    if alpha <= 0:
        return x, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    B = x.shape[0]
    idx = torch.randperm(B, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, idx, lam


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint Averaging Callback
# ──────────────────────────────────────────────────────────────────────────────
class CheckpointAveragingCallback(pl.Callback):
    """
    At test time, load the top-k checkpoints by val_f1 and average their
    trainable parameters for smoother predictions.

    Fixed to correctly handle DDP state_dict key prefixes and ensure
    all trainable parameters are loaded (fixing the 'Matched: 43' bug
    in the sibling node2-2-1-1-1).
    """

    def __init__(self, top_k: int = 3, checkpoint_dir: Optional[str] = None):
        super().__init__()
        self.top_k = top_k
        self.checkpoint_dir = checkpoint_dir
        self._avg_ckpt_path: Optional[str] = None

    def on_train_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        """After training, compute and save the averaged checkpoint."""
        if trainer.fast_dev_run:
            return

        # Get top-k checkpoint paths from ModelCheckpoint callback
        ckpt_callback = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                ckpt_callback = cb
                break

        if ckpt_callback is None:
            pl_module.print("[CkptAvg] No ModelCheckpoint callback found, skipping averaging.")
            return

        # Collect top-k checkpoints sorted by val_f1 score
        checkpoint_dir = Path(ckpt_callback.dirpath)
        ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
        ckpt_files = [f for f in ckpt_files if "last" not in f.name]

        if not ckpt_files:
            pl_module.print("[CkptAvg] No checkpoints found, skipping averaging.")
            return

        # Parse val_f1 from filenames (format: ...-val_f1=0.XXXX.ckpt)
        import re
        scored_ckpts = []
        for f in ckpt_files:
            m = re.search(r'val_f1=(\d+\.\d{4})', f.name)
            if m:
                scored_ckpts.append((float(m.group(1)), f))

        scored_ckpts.sort(key=lambda x: x[0], reverse=True)
        top_k_ckpts = [path for _, path in scored_ckpts[:self.top_k]]

        if len(top_k_ckpts) < 1:
            pl_module.print("[CkptAvg] Could not find val_f1 scores in checkpoint names, skipping.")
            return

        pl_module.print(
            f"[CkptAvg] Averaging {len(top_k_ckpts)} checkpoints: "
            + ", ".join(f.name for f in top_k_ckpts)
        )

        # Load and average trainable parameters from top-k checkpoints
        # We need to average based on the module's current state_dict structure
        avg_state = None
        loaded_count = 0

        for ckpt_path in top_k_ckpts:
            try:
                ckpt = torch.load(str(ckpt_path), map_location="cpu")
                # Lightning checkpoints have state_dict under 'state_dict' key
                if "state_dict" in ckpt:
                    sd = ckpt["state_dict"]
                else:
                    sd = ckpt

                if avg_state is None:
                    avg_state = {k: v.float().clone() for k, v in sd.items()}
                else:
                    for k in avg_state:
                        if k in sd:
                            avg_state[k] = avg_state[k] + sd[k].float()
                loaded_count += 1
            except Exception as e:
                pl_module.print(f"[CkptAvg] Failed to load {ckpt_path.name}: {e}")

        if avg_state is None or loaded_count == 0:
            pl_module.print("[CkptAvg] No checkpoints successfully loaded, skipping.")
            return

        # Divide to get mean
        for k in avg_state:
            avg_state[k] = avg_state[k] / loaded_count

        # Load the averaged state into the current model
        # Use strict=False to handle potential DDP prefix differences
        load_result = pl_module.load_state_dict(avg_state, strict=False)
        pl_module.print(
            f"[CkptAvg] Loaded averaged state from {loaded_count} checkpoints. "
            f"Missing: {len(load_result.missing_keys)}, "
            f"Unexpected: {len(load_result.unexpected_keys)}"
        )

        # Save the averaged checkpoint for reproducibility
        if trainer.is_global_zero:
            avg_ckpt_path = str(checkpoint_dir / "averaged_checkpoint.ckpt")
            torch.save({"state_dict": avg_state}, avg_ckpt_path)
            self._avg_ckpt_path = avg_ckpt_path
            pl_module.print(f"[CkptAvg] Averaged checkpoint saved -> {avg_ckpt_path}")

        # Barrier to ensure all ranks have the averaged model before test
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()


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
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_layers_start: int = 0,
        fusion_layers: int = 3,
        fusion_heads: int = 8,
        fusion_ff_dim: int = 384,
        fusion_attn_dropout: float = 0.1,
        head_hidden: int = 256,
        head_dropout: float = 0.4,
        lr: float = 2e-4,
        head_lr_multiplier: float = 3.0,
        symbol_encoder_lr_multiplier: float = 2.0,
        weight_decay: float = 0.10,
        gamma_focal: float = 1.5,
        label_smoothing: float = 0.05,
        max_epochs: int = 120,
        plateau_patience: int = 8,
        plateau_factor: float = 0.5,
        plateau_min_lr: float = 1e-7,
        mixup_alpha: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[CrossAttnFusionDEGModel] = None
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
            self.model = CrossAttnFusionDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_layers_start=self.hparams.lora_layers_start,
                fusion_layers=self.hparams.fusion_layers,
                fusion_heads=self.hparams.fusion_heads,
                fusion_ff_dim=self.hparams.fusion_ff_dim,
                fusion_attn_dropout=self.hparams.fusion_attn_dropout,
                head_hidden=self.hparams.head_hidden,
                head_dropout=self.hparams.head_dropout,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def on_train_start(self) -> None:
        """Ensure PPI embeddings are computed before training starts (after device placement)."""
        if self.model._ppi_embeddings is None:
            self.model.compute_ppi_embeddings(self.device)

    def on_validation_start(self) -> None:
        """Ensure PPI embeddings are available during validation (e.g., after sanity val)."""
        if self.model._ppi_embeddings is None:
            self.model.compute_ppi_embeddings(self.device)

    def on_test_start(self) -> None:
        """Ensure PPI embeddings are available for test inference."""
        if self.model._ppi_embeddings is None:
            self.model.compute_ppi_embeddings(self.device)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            batch["expr"], batch["gene_pos"], batch["ppi_idx"], batch["symbol_ids"]
        )

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lam: float = 1.0,
        labels_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_flat = labels.reshape(-1)
        if lam < 1.0 and labels_b is not None:
            labels_b_flat = labels_b.reshape(-1)
            loss_a = self.criterion(logits_flat, labels_flat)
            loss_b = self.criterion(logits_flat, labels_b_flat)
            return lam * loss_a + (1 - lam) * loss_b
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Get backbone features for manifold mixup
        # Apply mixup at the token level AFTER fusion, before head
        expr, gene_pos, ppi_idx, symbol_ids = (
            batch["expr"], batch["gene_pos"], batch["ppi_idx"], batch["symbol_ids"]
        )
        labels = batch["label"]
        B = expr.shape[0]
        alpha = self.hparams.mixup_alpha

        # Forward pass with manifold mixup
        if alpha > 0 and self.training:
            # Apply mixup in the feature space
            lam = float(np.random.beta(alpha, alpha))
            idx = torch.randperm(B, device=expr.device)

            # Run backbone on both original and shuffled
            # For efficiency, run once and mix features before head
            logits = self._forward_with_mixup(
                expr, gene_pos, ppi_idx, symbol_ids, labels, idx, lam
            )
        else:
            logits = self(batch)
            lam = 1.0
            idx = None

        if idx is not None and lam < 1.0:
            loss = self._compute_loss(logits, labels, lam, labels[idx])
        else:
            loss = self._compute_loss(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def _forward_with_mixup(
        self,
        expr: torch.Tensor,
        gene_pos: torch.Tensor,
        ppi_idx: torch.Tensor,
        symbol_ids: torch.Tensor,
        labels: torch.Tensor,
        idx: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Forward with manifold mixup applied at the fusion token level."""
        B = expr.shape[0]
        device = expr.device
        model = self.model

        # ── Backbone embeddings ──
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = model.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, 19266, 256]

        gene_emb = lhs[:, :N_GENES_AIDO, :]
        global_emb = gene_emb.mean(dim=1).float()  # [B, 256]

        pert_emb = torch.zeros(B, model.HIDDEN_DIM, device=device, dtype=torch.float32)
        valid_mask = gene_pos >= 0
        if valid_mask.any():
            valid_pos = gene_pos[valid_mask]
            pert_emb[valid_mask] = lhs[valid_mask, valid_pos, :].float()
        pert_emb[~valid_mask] = global_emb[~valid_mask]

        # ── PPI embeddings ──
        if model._ppi_embeddings is None:
            model.compute_ppi_embeddings(device)
        ppi_emb_all = model._ppi_embeddings.to(device)
        ppi_feat = torch.zeros(B, model.FUSION_DIM, device=device, dtype=torch.float32)
        valid_ppi = ppi_idx >= 0
        if valid_ppi.any():
            ppi_feat[valid_ppi] = ppi_emb_all[ppi_idx[valid_ppi]]
        ppi_feat[~valid_ppi] = global_emb[~valid_ppi]

        # ── Symbol CNN ──
        sym_feat = model.symbol_encoder(symbol_ids)
        sym_projected = model.sym_proj(sym_feat)  # [B, 256]

        # ── Stack 4 tokens ──
        tokens = torch.stack([global_emb, pert_emb, ppi_feat, sym_projected], dim=1)  # [B, 4, 256]

        # ── Apply manifold mixup at the token level ──
        mixed_tokens = lam * tokens + (1 - lam) * tokens[idx]

        # ── Fusion transformer ──
        fused = model.fusion_transformer(mixed_tokens)  # [B, 4, 256]
        fused_pool = fused.mean(dim=1)                  # [B, 256]

        # ── Head ──
        logits = model.head(fused_pool)
        return logits.view(B, N_CLASSES, N_GENES_OUT)

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

        # Compute F1 from gathered data on rank 0, then sync to all ranks.
        # ModelCheckpoint and EarlyStopping run on rank 0, so we compute
        # correctly there and broadcast the scalar to all ranks for consistency.
        if self.trainer.is_global_zero:
            preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
            labels = al.cpu().view(-1, N_GENES_OUT).numpy()
            idxs = ai.cpu().view(-1).numpy()
            _, uniq = np.unique(idxs, return_index=True)
            f1 = compute_deg_f1(preds[uniq], labels[uniq])
            # Broadcast the scalar F1 to all ranks so all processes log the same value
            f1_tensor = torch.tensor(f1, dtype=torch.float32, device=self.device)
        else:
            f1_tensor = torch.zeros(1, dtype=torch.float32, device=self.device)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(f1_tensor, src=0)

        self.log("val_f1", f1_tensor.item(), prog_bar=True, sync_dist=False)

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
            pd.DataFrame(rows).to_csv(output_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"Test predictions saved -> {output_dir / 'test_predictions.tsv'}")

    def configure_optimizers(self):
        hp = self.hparams
        # Three separate parameter groups with different learning rates
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        symbol_params = list(self.model.symbol_encoder.parameters()) + list(self.model.sym_proj.parameters())
        # Fusion transformer + head params (highest LR)
        head_params = (
            list(self.model.fusion_transformer.parameters()) +
            list(self.model.head.parameters())
        )

        backbone_lr = hp.lr                                        # 2e-4
        symbol_lr = hp.lr * hp.symbol_encoder_lr_multiplier       # 4e-4 (2× backbone)
        head_lr = hp.lr * hp.head_lr_multiplier                   # 6e-4 (3× backbone)

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": symbol_params, "lr": symbol_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=hp.weight_decay,
        )

        # ReduceLROnPlateau: proven effective in the best nodes (node3-1-1-1-1-1-2)
        # patience=8 to avoid excessive LR reductions (sibling node2-2-1-1-1 showed
        # patience=5 caused 7 reductions with LR starvation)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=hp.plateau_factor,
            patience=hp.plateau_patience,
            min_lr=hp.plateau_min_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
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
        """Load trainable parameters and persistent buffers from a partial checkpoint.

        Handles DDP 'module.' prefix: checkpoints saved without DDP wrapping have
        unprefixed keys, while LightningModule.named_parameters() under DDP returns
        'module.' prefixed keys. We strip that prefix for matching.
        """
        # Get the full state dict keys (what the module actually has)
        full_state = super().state_dict()
        full_state_keys = set(full_state.keys())

        # Build expected keys: trainable params + buffers, matching format of full state dict
        # (Lightning strips 'module.' prefix internally before calling this)
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        buffer_keys = {
            name for name, _ in self.named_buffers() if name in full_state_keys
        }
        expected_keys = trainable_keys | buffer_keys

        # Strip any DDP 'module.' prefix from checkpoint keys for comparison
        # LightningModule's named_parameters() under DDP returns 'module.xxx',
        # but checkpoints saved without DDP wrapper have unprefixed 'xxx'
        prefix = "module."
        cleaned_state = {}
        for k, v in state_dict.items():
            # Check if checkpoint key matches expected (with or without prefix)
            if k in expected_keys:
                cleaned_state[k] = v
            elif k.startswith(prefix) and k[len(prefix):] in expected_keys:
                cleaned_state[k[len(prefix):]] = v
            else:
                pass  # Will be handled by strict=False

        missing_keys = [k for k in expected_keys if k not in cleaned_state]
        unexpected_keys = [k for k in state_dict if k not in expected_keys]

        if missing_keys:
            self.print(f"[load_state_dict] Missing keys ({len(missing_keys)}): {missing_keys[:3]}...")
        if unexpected_keys:
            self.print(f"[load_state_dict] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:3]}...")

        loaded_count = len(cleaned_state)
        self.print(f"[load_state_dict] Loading {loaded_count}/{len(expected_keys)} expected keys")
        return super().load_state_dict(cleaned_state, strict=False)



# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 1-2: Cross-Attention Fusion DEG predictor with AIDO.Cell + STRING GNN + Symbol CNN"
    )
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--global_batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--head_lr_multiplier", type=float, default=3.0)
    p.add_argument("--symbol_encoder_lr_multiplier", type=float, default=2.0)
    p.add_argument("--weight_decay", type=float, default=0.10)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_layers_start", type=int, default=0)
    p.add_argument("--fusion_layers", type=int, default=3)
    p.add_argument("--fusion_heads", type=int, default=8)
    p.add_argument("--fusion_ff_dim", type=int, default=384)
    p.add_argument("--fusion_attn_dropout", type=float, default=0.1)
    p.add_argument("--head_hidden", type=int, default=256)
    p.add_argument("--head_dropout", type=float, default=0.4)
    p.add_argument("--gamma_focal", type=float, default=1.5)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--plateau_patience", type=int, default=8)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--plateau_min_lr", type=float, default=1e-7)
    p.add_argument("--mixup_alpha", type=float, default=0.3)
    p.add_argument("--early_stopping_patience", type=int, default=35)
    p.add_argument("--checkpoint_avg_top_k", type=int, default=3)
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
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
    limit_test_batches = 1.0

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=args.checkpoint_avg_top_k, save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.early_stopping_patience, verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Checkpoint averaging callback for test-time model smoothing
    ckpt_avg_cb = CheckpointAveragingCallback(
        top_k=args.checkpoint_avg_top_k,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=1.0 if (args.debug_max_step is not None or args.fast_dev_run) else args.val_check_interval,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar, ckpt_avg_cb],
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
        lora_layers_start=args.lora_layers_start,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        fusion_ff_dim=args.fusion_ff_dim,
        fusion_attn_dropout=args.fusion_attn_dropout,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        lr=args.lr,
        head_lr_multiplier=args.head_lr_multiplier,
        symbol_encoder_lr_multiplier=args.symbol_encoder_lr_multiplier,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        plateau_min_lr=args.plateau_min_lr,
        mixup_alpha=args.mixup_alpha,
    )

    trainer.fit(model_module, datamodule=datamodule)

    # Test with the checkpoint-averaged model (averaging applied in on_train_end callback)
    # Skip ckpt_path='best' to use the averaged weights loaded by CheckpointAveragingCallback
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        # The CheckpointAveragingCallback already loaded averaged weights in on_train_end
        # Test with the current (averaged) model state
        test_results = trainer.test(model_module, datamodule=datamodule)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        # best_model_score may be None if max_steps was reached without early stopping.
        # Fall back to parsing the best checkpoint's val_f1 from the filename.
        best_f1 = checkpoint_cb.best_model_score
        if best_f1 is None and checkpoint_cb.best_model_path:
            import re
            m = re.search(r'val_f1=(\d+\.\d{4})', checkpoint_cb.best_model_path)
            if m:
                best_f1 = float(m.group(1))
        score_path.write_text(
            f"test_results: {test_results}\n"
            f"val_f1_best: {best_f1}\n"
        )
        print(f"Test score saved -> {score_path}")


if __name__ == "__main__":
    main()
