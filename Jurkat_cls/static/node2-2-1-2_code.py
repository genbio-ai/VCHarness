#!/usr/bin/env python3
"""
Node 2-2-1-2: AIDO.Cell-10M + LoRA + Symbol CNN + STRING GNN PPI + ReduceLROnPlateau (val_loss)
================================================================================================
Improves on parent node (node2-2-1: AIDO.Cell-10M + LoRA r=8 last 4 + symbol CNN +
ReduceLROnPlateau val_f1, test F1=0.4472) by targeting the primary bottleneck:
architectural saturation (ceiling ~0.447 F1 without additional information channels).

The parent node's training dynamics showed:
  - Best val_f1 = 0.447 at epoch 18 (never improved after)
  - ReduceLROnPlateau (val_f1 monitoring) fired 3 times but produced no val_f1 recovery
  - Persistent severe overfitting: train_loss 0.086 vs val_loss 0.832 at epoch 53 (9.7x ratio)
  - Architecture saturated at the ~0.447 ceiling

Key differences from parent node (node2-2-1):
  1. PRIMARY: Add frozen STRING GNN PPI embeddings (256-dim) as THIRD information channel
     - STRING GNN precomputes [18870, 256] embeddings for the entire human PPI graph
     - Perturbed gene's PPI neighborhood embedding provides network topology context
     - Orthogonal to AIDO.Cell (expression patterns) and symbol CNN (naming conventions)
     - PROVEN: node3-2 (same AIDO.Cell backbone + STRING GNN) achieved F1=0.462 vs
       parent's 0.447, a +0.015 improvement by adding this exact channel
     - Input fusion: 512 (AIDO dual-pool) + 256 (STRING GNN) + 64 (symbol) = 832-dim total

  2. SECONDARY: ReduceLROnPlateau monitors val_loss instead of val_f1
     - node3-2 showed val_f1 oscillates ±0.005, preventing scheduler from firing
     - val_loss monotonically rises → always triggers reductions reliably
     - Insight from node3-3-1 (val_loss monitoring achieved stable F1=0.4513)

  3. SECONDARY: Small Gaussian noise augmentation on expression inputs during training
     - std=0.05 on non-KO gene positions (distinct from sibling's gene dropout)
     - Forces model to learn robustly from slightly noisy expression profiles
     - Creates input diversity without removing gene identity information

  4. SECONDARY: weight_decay=0.04 (from 0.03) for slightly stronger regularization
     - Addresses the 9.7× overfitting gap (val_loss/train_loss at epoch 53)

Key differences from sibling (node2-2-1-1):
  - Sibling adds SWA + gene dropout (optimization/regularization-side improvements)
  - This node adds STRING GNN (new orthogonal information channel)
  - Distinct directions covering complementary improvement strategies
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
from typing import Any, Dict, List, Optional

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
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

N_GENES_AIDO = 19_264    # AIDO.Cell vocabulary size (fixed for all model sizes)
N_GENES_OUT = 6_640      # output genes
N_CLASSES = 3
SENTINEL_EXPR = 1.0      # baseline expression (non-perturbed genes)
KNOCKOUT_EXPR = 0.0      # expression for knocked-out gene (perturbed)
AIDO_HIDDEN = 256        # AIDO.Cell-10M hidden dimension
AIDO_N_LAYERS = 8        # AIDO.Cell-10M transformer layers
STRING_GNN_DIM = 256     # STRING GNN output embedding dimension

# Moderate class weights — limits severe val_loss inflation seen with extreme weights
# Train distribution: class 0 (down) ~3.4%, class 1 (unchanged) ~95.5%, class 2 (up) ~1.1%
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)

# Character vocabulary for gene symbol encoding
# Gene symbols contain: uppercase letters, digits, hyphens, underscores, dots
# Characters: A-Z (26), 0-9 (10), special: -, _, . -> total 39 + 1 padding = 40
SYMBOL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
SYMBOL_PAD_IDX = len(SYMBOL_CHARS)          # 39 → padding index
SYMBOL_VOCAB_SIZE = len(SYMBOL_CHARS) + 1   # 40
SYMBOL_MAX_LEN = 12                          # max gene symbol length (actual max ~10)


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss with label smoothing
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weights and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        # Standard cross-entropy with label smoothing for the probability used in focal weight
        ce = F.cross_entropy(logits, targets, weight=w, reduction="none",
                             label_smoothing=self.label_smoothing)
        # Focal weight from hard targets (no label smoothing in pt calculation)
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
    # Pad to SYMBOL_MAX_LEN
    while len(indices) < SYMBOL_MAX_LEN:
        indices.append(SYMBOL_PAD_IDX)
    return indices


class SymbolEncoder(nn.Module):
    """
    Character-level CNN encoder for gene symbol strings.

    Architecture:
      1. Character embedding: [B, L] → [B, L, embed_dim] (embed_dim=32)
      2. 1D conv layers with max-pool for feature extraction
      3. Global max-pool → [B, out_dim]
    """

    def __init__(self, out_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(SYMBOL_VOCAB_SIZE, embed_dim, padding_idx=SYMBOL_PAD_IDX)
        # Three parallel conv filters at different kernel sizes
        self.conv2 = nn.Conv1d(embed_dim, 32, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, 32, kernel_size=4, padding=2)
        # Project concatenated features (96) to out_dim
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
        """
        symbol_ids: [B, SYMBOL_MAX_LEN] int64
        returns: [B, out_dim]
        """
        x = self.embed(symbol_ids)          # [B, L, embed_dim]
        x = x.transpose(1, 2)              # [B, embed_dim, L] for Conv1d
        # Parallel convolutions with ReLU
        f2 = F.relu(self.conv2(x))         # [B, 32, L+1] (with padding)
        f3 = F.relu(self.conv3(x))         # [B, 32, L]
        f4 = F.relu(self.conv4(x))         # [B, 32, L+1] (with padding)
        # Trim to same length L via adaptive max-pool
        f2 = F.adaptive_max_pool1d(f2, 1).squeeze(-1)  # [B, 32]
        f3 = F.adaptive_max_pool1d(f3, 1).squeeze(-1)  # [B, 32]
        f4 = F.adaptive_max_pool1d(f4, 1).squeeze(-1)  # [B, 32]
        feat = torch.cat([f2, f3, f4], dim=-1)          # [B, 96]
        return self.proj(feat)                           # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# STRING GNN helper: precompute all node embeddings once
# ──────────────────────────────────────────────────────────────────────────────
def compute_string_gnn_embeddings(device: torch.device) -> tuple:
    """
    Load the STRING GNN model and compute all 18870 node embeddings once.

    Returns:
        (all_emb, ensg_to_idx) where:
          - all_emb: [18870, 256] float32 tensor on CPU
          - ensg_to_idx: dict mapping ENSG ID → row index in all_emb
    """
    # Load node names: list of 18870 ENSG IDs
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    ensg_to_idx = {name: i for i, name in enumerate(node_names)}

    # Load STRING GNN model
    string_model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
    string_model.eval()
    string_model = string_model.to(device)

    # Load the fixed graph
    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location=device)
    edge_index = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"]
    edge_weight = edge_weight.to(device) if edge_weight is not None else None

    # Forward pass once: [18870, 256] embeddings
    with torch.no_grad():
        outputs = string_model(edge_index=edge_index, edge_weight=edge_weight)
        all_emb = outputs.last_hidden_state.float().cpu()  # [18870, 256] on CPU

    # Free GPU memory
    del string_model, edge_index, edge_weight
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return all_emb, ensg_to_idx


def lookup_ppi_embeddings(
    pert_ids: List[str],
    all_emb: torch.Tensor,
    ensg_to_idx: Dict[str, int],
) -> torch.Tensor:
    """
    Look up STRING GNN embeddings for a list of pert_ids.

    Returns [N, 256] float32 tensor. Uses zero vector for genes not in STRING graph.
    """
    N = len(pert_ids)
    emb = torch.zeros(N, STRING_GNN_DIM, dtype=torch.float32)
    for i, pert_id in enumerate(pert_ids):
        base = pert_id.split(".")[0]  # Strip version suffix (e.g. "ENSG00000001084.5" → "ENSG00000001084")
        idx = ensg_to_idx.get(base, -1)
        if idx >= 0:
            emb[i] = all_emb[idx]
    return emb  # [N, 256] float32 CPU tensor


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Returns pre-built AIDO.Cell expression profile tensors (float32) together
    with the perturbed gene position index, the gene symbol character indices,
    STRING GNN PPI embeddings, and the label.

    New in node2-2-1-2:
      - ppi_emb: [256] float32 STRING GNN embedding for the perturbed gene
      - augment: if True, adds Gaussian noise to non-KO expression positions
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],    # ENSG_base -> position in [0, 19264)
        ppi_emb: torch.Tensor,          # [N, 256] float32 STRING GNN embeddings
        is_test: bool = False,
        augment: bool = False,
        input_noise_std: float = 0.05,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.is_test = is_test
        self.augment = augment
        self.input_noise_std = input_noise_std

        # Pre-build the AIDO.Cell expression profile tensors
        # Shape: [N, 19264] float32 — baseline 1.0, knocked-out gene 0.0
        self.expr_inputs = self._build_expr_tensors()

        # Pre-build symbol character index tensors
        # Shape: [N, SYMBOL_MAX_LEN] int64
        self.symbol_ids = self._build_symbol_tensors()

        # STRING GNN PPI embeddings: [N, 256] float32
        assert ppi_emb.shape[0] == len(self.pert_ids), \
            f"ppi_emb shape mismatch: {ppi_emb.shape[0]} != {len(self.pert_ids)}"
        self.ppi_emb = ppi_emb  # [N, 256] float32

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1} → {0,1,2}
        else:
            self.labels = None

    def _build_expr_tensors(self) -> torch.Tensor:
        """Pre-compute [N, 19264] float32 expression inputs."""
        N = len(self.pert_ids)
        expr = torch.full((N, N_GENES_AIDO), SENTINEL_EXPR, dtype=torch.float32)
        for i, pert_id in enumerate(self.pert_ids):
            base = pert_id.split(".")[0]
            pos = self.gene_to_pos.get(base)
            if pos is not None:
                expr[i, pos] = KNOCKOUT_EXPR
        return expr

    def _build_symbol_tensors(self) -> torch.Tensor:
        """Pre-compute [N, SYMBOL_MAX_LEN] int64 character index tensors."""
        N = len(self.symbols)
        sym_ids = torch.zeros((N, SYMBOL_MAX_LEN), dtype=torch.long)
        for i, symbol in enumerate(self.symbols):
            indices = symbol_to_indices(symbol)
            sym_ids[i] = torch.tensor(indices, dtype=torch.long)
        return sym_ids

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.pert_ids[idx].split(".")[0]
        gene_pos = self.gene_to_pos.get(base, -1)

        # Get expression profile (with optional augmentation)
        expr = self.expr_inputs[idx].clone()
        if self.augment and self.input_noise_std > 0.0:
            # Add Gaussian noise only to non-KO (SENTINEL) positions
            non_ko_mask = (expr > 0.5).float()  # 1.0 for non-KO genes
            noise = torch.randn_like(expr) * self.input_noise_std
            expr = expr + noise * non_ko_mask
            expr = expr.clamp(0.0, 2.0)  # Keep values in reasonable range

        item = {
            "idx": idx,
            "expr": expr,                               # [19264] float32
            "gene_pos": gene_pos,                       # int (-1 if not in vocab)
            "symbol_ids": self.symbol_ids[idx],         # [SYMBOL_MAX_LEN] int64
            "ppi_emb": self.ppi_emb[idx],               # [256] float32
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
        "ppi_emb": torch.stack([b["ppi_emb"] for b in batch]),        # [B, 256]
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
        input_noise_std: float = 0.05,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.input_noise_std = input_noise_std
        self.gene_to_pos: Dict[str, int] = {}
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []
        # STRING GNN cached embeddings (shared across all splits)
        self._string_all_emb: Optional[torch.Tensor] = None  # [18870, 256] CPU float32
        self._string_ensg_to_idx: Optional[Dict[str, int]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # ── Step 1: Rank-0 downloads AIDO.Cell tokenizer first, then barrier ──
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # ── Step 2: Build ENSG → position mapping ──
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)

        # ── Step 3: Pre-compute STRING GNN embeddings (once, rank-0 first) ──
        if self._string_all_emb is None:
            if local_rank == 0:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._string_all_emb, self._string_ensg_to_idx = \
                    compute_string_gnn_embeddings(device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            # Non-rank-0 processes also compute STRING GNN embeddings
            if self._string_all_emb is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._string_all_emb, self._string_ensg_to_idx = \
                    compute_string_gnn_embeddings(device)

        # ── Step 4: Build dataset splits ──
        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")

            train_ppi_emb = lookup_ppi_embeddings(
                train_df["pert_id"].tolist(), self._string_all_emb, self._string_ensg_to_idx)
            val_ppi_emb = lookup_ppi_embeddings(
                val_df["pert_id"].tolist(), self._string_all_emb, self._string_ensg_to_idx)

            self.train_ds = PerturbationDataset(
                train_df, self.gene_to_pos, train_ppi_emb,
                is_test=False, augment=True, input_noise_std=self.input_noise_std)
            self.val_ds = PerturbationDataset(
                val_df, self.gene_to_pos, val_ppi_emb,
                is_test=False, augment=False, input_noise_std=0.0)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            test_ppi_emb = lookup_ppi_embeddings(
                test_df["pert_id"].tolist(), self._string_all_emb, self._string_ensg_to_idx)

            self.test_ds = PerturbationDataset(
                test_df, self.gene_to_pos, test_ppi_emb,
                is_test=True, augment=False, input_noise_std=0.0)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    @staticmethod
    def _build_gene_to_pos(tokenizer, gene_ids: List[str]) -> Dict[str, int]:
        """Map each ENSG gene_id to its position index in AIDO.Cell vocab."""
        mapping: Dict[str, int] = {}
        PROBE_VAL = 50.0  # distinctive non-(-1) float to detect gene position
        for gene_id in gene_ids:
            try:
                inputs = tokenizer(
                    {"gene_ids": [gene_id], "expression": [PROBE_VAL]},
                    return_tensors="pt",
                )
                ids = inputs["input_ids"]
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)  # [1, 19264]
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
# Model
# ──────────────────────────────────────────────────────────────────────────────
class AIDOCellStringDEGModel(nn.Module):
    """
    AIDO.Cell-10M backbone + LoRA (last 4 layers) + character-level symbol CNN +
    frozen STRING GNN PPI embeddings + dual-pooling prediction head.

    Input representations fused (all as float32):
      (a) global mean-pool of AIDO.Cell-10M last_hidden_state [B, 256]
      (b) perturbed-gene positional embedding [B, 256]
      (c) gene symbol character CNN embedding [B, 64]
      (d) frozen STRING GNN PPI embedding for the perturbed gene [B, 256]
    Concatenated → [B, 832] → MLP head → [B, 3, 6640]

    New vs parent (node2-2-1):
      - Added STRING GNN PPI channel (d), head input expanded 576 → 832
      - head_hidden=320 maintained (tighter bottleneck for 832-dim input)
    """

    HIDDEN_DIM = 256          # AIDO.Cell-10M hidden size
    SYMBOL_DIM = 64           # gene symbol embedding dimension
    PPI_DIM = 256             # STRING GNN embedding dimension
    HEAD_INPUT_DIM = 256 * 2 + 64 + 256  # = 832 (AIDO dual-pool + symbol + PPI)

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_layers_start: int = 4,  # last 4 of 8 layers: indices 4,5,6,7
        head_hidden: int = 320,
        head_dropout: float = 0.4,
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA ──────────────────────────────────
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA on Q/K/V of last (AIDO_N_LAYERS - lora_layers_start) layers
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.3,
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

        # ── Gene symbol character-level CNN encoder ────────────────────────────
        self.symbol_encoder = SymbolEncoder(out_dim=self.SYMBOL_DIM, embed_dim=32)

        # ── STRING GNN PPI projection ──────────────────────────────────────────
        # Small projection layer to align STRING GNN embeddings with the other features
        # Keeps dimensionality at 256, just normalizes and adds learnable transformation
        self.ppi_proj = nn.Sequential(
            nn.LayerNorm(self.PPI_DIM),
            nn.Linear(self.PPI_DIM, self.PPI_DIM),
            nn.GELU(),
        )
        # Initialize with near-identity for stable early training
        nn.init.eye_(self.ppi_proj[1].weight)
        nn.init.zeros_(self.ppi_proj[1].bias)

        # ── Prediction head ────────────────────────────────────────────────────
        # head_hidden=320: same as parent, but with larger 832-dim input
        # The tighter bottleneck forces cross-channel fusion learning
        head_in = self.HEAD_INPUT_DIM  # 832
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        # Conservative initialization for the output layer
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        expr: torch.Tensor,          # [B, 19264] float32
        gene_pos: torch.Tensor,      # [B]         int64  (-1 if not in vocab)
        symbol_ids: torch.Tensor,    # [B, SYMBOL_MAX_LEN] int64
        ppi_emb: torch.Tensor,       # [B, 256] float32  STRING GNN embedding
    ) -> torch.Tensor:
        # ── AIDO.Cell backbone forward ─────────────────────────────────────────
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, 19266, 256]

        # (a) Global mean-pool over all gene positions (exclude 2 summary tokens)
        gene_emb = lhs[:, :N_GENES_AIDO, :]          # [B, 19264, 256]
        global_emb = gene_emb.mean(dim=1)             # [B, 256]

        # (b) Perturbed-gene positional embedding
        B = expr.shape[0]
        pert_emb = torch.zeros(B, self.HIDDEN_DIM, device=lhs.device, dtype=lhs.dtype)
        valid_mask = gene_pos >= 0
        if valid_mask.any():
            valid_pos = gene_pos[valid_mask]  # [k]
            pert_emb[valid_mask] = lhs[valid_mask, valid_pos, :]
        # Fallback: use global_emb for genes not in vocabulary
        pert_emb[~valid_mask] = global_emb[~valid_mask]

        # Convert to float32 for head computation
        backbone_feat = torch.cat([global_emb, pert_emb], dim=-1).float()  # [B, 512]

        # (c) Gene symbol character CNN embedding
        sym_feat = self.symbol_encoder(symbol_ids)    # [B, 64] float32

        # (d) STRING GNN PPI embedding (pre-computed frozen, projected for alignment)
        ppi_emb_f = ppi_emb.to(expr.device).float()  # [B, 256] float32
        ppi_feat = self.ppi_proj(ppi_emb_f)           # [B, 256] float32

        # Concatenate all four features: AIDO dual-pool + symbol + PPI
        combined = torch.cat([backbone_feat, sym_feat, ppi_feat], dim=-1)  # [B, 832]

        logits = self.head(combined)                  # [B, 3 * 6640]
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
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_layers_start: int = 4,
        head_hidden: int = 320,
        head_dropout: float = 0.4,
        lr: float = 3e-4,
        head_lr_multiplier: float = 3.0,
        symbol_encoder_lr_multiplier: float = 2.0,
        ppi_proj_lr_multiplier: float = 2.0,
        weight_decay: float = 0.04,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 120,
        plateau_patience: int = 8,
        plateau_factor: float = 0.5,
        plateau_min_lr: float = 1e-7,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCellStringDEGModel] = None
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
            self.model = AIDOCellStringDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_layers_start=self.hparams.lora_layers_start,
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

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(batch["expr"], batch["gene_pos"], batch["symbol_ids"], batch["ppi_emb"])

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_flat = labels.reshape(-1)
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
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

        # Each rank independently computes val_f1 on the full gathered dataset
        preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
        labels = al.cpu().view(-1, N_GENES_OUT).numpy()
        idxs = ai.cpu().view(-1).numpy()
        _, uniq = np.unique(idxs, return_index=True)
        f1 = compute_deg_f1(preds[uniq], labels[uniq])

        # All-reduce across ranks so all ranks have the same val_f1
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
            self.print(f"Test predictions saved → {output_dir / 'test_predictions.tsv'}")

    def configure_optimizers(self):
        hp = self.hparams
        # FOUR separate parameter groups with different learning rates:
        #   1. backbone LoRA (lowest lr — most pretrained)
        #   2. symbol_encoder (medium lr — small 22K randomly-initialized CNN)
        #   3. ppi_proj (medium lr — small projection for STRING GNN features)
        #   4. head (highest lr — main randomly-initialized predictor)
        # STRING GNN backbone is FROZEN (no parameter group needed)
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        symbol_params = list(self.model.symbol_encoder.parameters())
        ppi_proj_params = list(self.model.ppi_proj.parameters())
        head_params = list(self.model.head.parameters())

        backbone_lr = hp.lr                                          # 3e-4
        symbol_lr = hp.lr * hp.symbol_encoder_lr_multiplier         # 6e-4 (2× backbone)
        ppi_proj_lr = hp.lr * hp.ppi_proj_lr_multiplier             # 6e-4 (2× backbone)
        head_lr = hp.lr * hp.head_lr_multiplier                     # 9e-4 (3× backbone)

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": symbol_params, "lr": symbol_lr},
                {"params": ppi_proj_params, "lr": ppi_proj_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=hp.weight_decay,
        )

        # ReduceLROnPlateau monitors val_loss (not val_f1):
        # - val_f1 oscillates ±0.005 even when training is stuck → scheduler never fires
        # - val_loss monotonically rises as overfitting increases → reliably triggers reductions
        # - Insight from node3-3-1: val_loss monitoring achieved stable convergence at 0.4513
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",                    # minimize val_loss
            factor=hp.plateau_factor,      # 0.5: halve LR each reduction
            patience=hp.plateau_patience,  # 8: wait 8 epochs before reducing
            min_lr=hp.plateau_min_lr,      # 1e-7: floor to avoid near-zero LR
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss",          # KEY CHANGE: monitor val_loss not val_f1
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
        """Load trainable parameters and persistent buffers from a partial checkpoint."""
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
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 2-2-1-2: AIDO.Cell-10M + LoRA + Symbol CNN + STRING GNN + ReduceLROnPlateau (val_loss)"
    )
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--global_batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--head_lr_multiplier", type=float, default=3.0)
    p.add_argument("--symbol_encoder_lr_multiplier", type=float, default=2.0)
    p.add_argument("--ppi_proj_lr_multiplier", type=float, default=2.0)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_layers_start", type=int, default=4)
    p.add_argument("--head_hidden", type=int, default=320)
    p.add_argument("--head_dropout", type=float, default=0.4)
    p.add_argument("--gamma_focal", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--plateau_patience", type=int, default=8)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--plateau_min_lr", type=float, default=1e-7)
    p.add_argument("--early_stopping_patience", type=int, default=35)
    p.add_argument("--input_noise_std", type=float, default=0.05)
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
    # During debug modes, limit train/val batches to the specified count (int).
    # Test ALWAYS runs on the full test set regardless of debug flags.
    limit_train_batches = args.debug_max_step if args.debug_max_step is not None else 1.0
    limit_val_batches = args.debug_max_step if args.debug_max_step is not None else 1.0
    limit_test_batches = 1.0  # Always run full test set (required for correct evaluation)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node2-2-1-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=3, save_last=True,
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
        input_noise_std=args.input_noise_std,
    )
    model_module = DEGLightningModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_layers_start=args.lora_layers_start,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        lr=args.lr,
        head_lr_multiplier=args.head_lr_multiplier,
        symbol_encoder_lr_multiplier=args.symbol_encoder_lr_multiplier,
        ppi_proj_lr_multiplier=args.ppi_proj_lr_multiplier,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        plateau_min_lr=args.plateau_min_lr,
    )

    trainer.fit(model_module, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"test_results: {test_results}\n"
            f"val_f1_best: {checkpoint_cb.best_model_score}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
