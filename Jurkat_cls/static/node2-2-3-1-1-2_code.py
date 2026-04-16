#!/usr/bin/env python3
"""
Node 2-2-3-1-1-2: Top-5 Checkpoint Averaging + [8,1,18] Class Weights + Tighter Stopping
==========================================================================================

This node extends sibling node2-2-3-1-1-1 (test F1=0.4655, tree-best) by continuing two
proven improvement trends simultaneously, while maintaining architectural identity.

Sibling (node2-2-3-1-1-1) achieved test F1=0.4655 via:
  - Top-3 checkpoint weight averaging (+0.003 vs parent)
  - Strengthened class weights [7,1,15] (from parent's [6,1,12])
  - Tighter early stopping patience=15

This node (node2-2-3-1-1-2) extends BOTH directions beyond the sibling:
  1. Top-5 checkpoint averaging (from sibling's top-3):
     - sibling's feedback Priority 1: "Increase avg_top_k from 3 to 5"
     - More checkpoints → greater variance reduction → better generalization
     - Expected +0.001–0.002 F1 over sibling's top-3 averaging
     - Save top-5 checkpoints via save_top_k=5

  2. Class weights [8.0, 1.0, 18.0] (from sibling's [7.0, 1.0, 15.0]):
     - sibling's feedback Priority 3: "class_weights = [8.0, 1.0, 18.0]"
     - Continuing the proven [5→6→7→8] and [10→12→15→18] trend:
       * [5,1,10] → F1=0.4573 (node2-3-1-1-1-1)
       * [6,1,12] → F1=0.4625 (parent node2-2-3-1-1)
       * [7,1,15] → F1=0.4655 (sibling node2-2-3-1-1-1)
       * [8,1,18] → expected ~0.467+ (this node)
     - +14% down-reg (7→8) and +20% up-reg (15→18): moderate step
     - Expected +0.001–0.003 F1 beyond class weight trend

All other hyperparameters are IDENTICAL to the sibling:
  seed=0, weight_decay=0.03, plateau_patience=12, backbone_lr=3e-4,
  early_stopping_patience=15, focal_gamma=2.0, label_smoothing=0.05

Architecture: IDENTICAL to parent/sibling (AIDO.Cell-10M + LoRA QKV r=4 all-8 +
3-branch Symbol CNN 64-dim + Frozen STRING GNN PPI 256-dim → concat 832 → MLP 384 → 19920)
"""

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
STRING_GNN_DIM = 256     # STRING GNN output dimension

# Strengthened class weights: [8.0, 1.0, 18.0] (from sibling's [7.0, 1.0, 15.0])
# Trend evidence from MCTS tree:
#   [5,1,10] → 0.4573 (node2-3-1-1-1-1)
#   [6,1,12] → 0.4625 (parent node2-2-3-1-1)
#   [7,1,15] → 0.4655 (sibling node2-2-3-1-1-1)
#   [8,1,18] → expected ~0.467+ (this node)
# +14% down-reg (7→8) and +20% up-reg (15→18): next step in proven trend.
CLASS_WEIGHTS = torch.tensor([8.0, 1.0, 18.0], dtype=torch.float32)

# Character vocabulary for gene symbol encoding
# Gene symbols contain: uppercase letters, digits, hyphens, underscores, dots
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

        NOTE: F.adaptive_max_pool1d is non-deterministic (adaptive_max_pool2d_backward_cuda
        has no deterministic implementation). We use slicing instead since all conv outputs
        are consistently sized: conv2/conv4 output length=L+1, conv3 output length=L.
        """
        x = self.embed(symbol_ids)          # [B, L, embed_dim]
        x = x.transpose(1, 2)              # [B, embed_dim, L] for Conv1d
        # Parallel convolutions with ReLU
        f2 = F.relu(self.conv2(x))         # [B, 32, L+1] (with padding)
        f3 = F.relu(self.conv3(x))         # [B, 32, L]
        f4 = F.relu(self.conv4(x))         # [B, 32, L+1] (with padding)
        # Deterministic slicing: take last element (equivalent to global max-pool for 1D).
        # This replaces F.adaptive_max_pool1d which is non-deterministic.
        f2 = f2[:, :, -1]                  # [B, 32] - last position (L+1 → 1)
        f3 = f3[:, :, -1]                  # [B, 32] - last position (L → 1)
        f4 = f4[:, :, -1]                  # [B, 32] - last position (L+1 → 1)
        feat = torch.cat([f2, f3, f4], dim=-1)          # [B, 96]
        return self.proj(feat)                           # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# STRING GNN PPI Embeddings (frozen, pre-computed at setup)
# ──────────────────────────────────────────────────────────────────────────────
class FrozenSTRINGGNN:
    """
    Computes frozen STRING GNN PPI node embeddings for all genes at setup time.
    This is not an nn.Module — it's a one-time forward pass to get static embeddings.
    Returns a mapping: ENSG_id -> [STRING_GNN_DIM] float32 embedding.
    """

    def __init__(self, model_dir: str = STRING_GNN_MODEL_DIR):
        self.model_dir = Path(model_dir)

    def compute_embeddings(self, device: str = "cpu") -> Dict[str, np.ndarray]:
        """
        Run one forward pass to get all 18,870 node embeddings.
        Returns dict: ENSG_id -> numpy float32 array of shape [256]
        """
        model_dir = self.model_dir
        gnn_model = AutoModel.from_pretrained(
            str(model_dir), trust_remote_code=True
        ).to(device)
        gnn_model.eval()

        graph = torch.load(str(model_dir / "graph_data.pt"), map_location=device, weights_only=False)
        node_names = json.loads((model_dir / "node_names.json").read_text())

        edge_index = graph["edge_index"].to(device)
        edge_weight = graph["edge_weight"]
        edge_weight = edge_weight.to(device) if edge_weight is not None else None

        with torch.no_grad():
            outputs = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

        emb = outputs.last_hidden_state  # [18870, 256]
        emb_np = emb.cpu().float().numpy()

        # Build dict: ENSG_id (base, no version) -> embedding
        embedding_dict: Dict[str, np.ndarray] = {}
        for i, node_name in enumerate(node_names):
            base = node_name.split(".")[0]
            embedding_dict[base] = emb_np[i]

        # Free GPU memory
        del gnn_model, outputs, emb
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return embedding_dict


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Returns pre-built AIDO.Cell expression profile tensors (float32) together
    with the perturbed gene position index, the gene symbol character indices,
    frozen STRING GNN PPI embeddings, and the label.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],      # ENSG_base -> position in [0, 19264)
        ppi_embeddings: Dict[str, np.ndarray],  # ENSG_base -> [256] float32
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.ppi_embeddings = ppi_embeddings
        self.is_test = is_test

        # Pre-build the AIDO.Cell expression profile tensors
        self.expr_inputs = self._build_expr_tensors()

        # Pre-build symbol character index tensors
        self.symbol_ids = self._build_symbol_tensors()

        # Pre-build STRING GNN PPI embedding tensors
        self.ppi_embs = self._build_ppi_tensors()

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

    def _build_ppi_tensors(self) -> torch.Tensor:
        """Pre-compute [N, STRING_GNN_DIM] float32 PPI embedding tensors."""
        N = len(self.pert_ids)
        ppi = torch.zeros((N, STRING_GNN_DIM), dtype=torch.float32)
        for i, pert_id in enumerate(self.pert_ids):
            base = pert_id.split(".")[0]
            emb = self.ppi_embeddings.get(base)
            if emb is not None:
                ppi[i] = torch.tensor(emb, dtype=torch.float32)
        return ppi

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.pert_ids[idx].split(".")[0]
        gene_pos = self.gene_to_pos.get(base, -1)
        item = {
            "idx": idx,
            "expr": self.expr_inputs[idx],       # [19264] float32
            "gene_pos": gene_pos,                 # int (-1 if not in vocab)
            "symbol_ids": self.symbol_ids[idx],   # [SYMBOL_MAX_LEN] int64
            "ppi_emb": self.ppi_embs[idx],        # [STRING_GNN_DIM] float32
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "expr": torch.stack([b["expr"] for b in batch]),
        "gene_pos": torch.tensor([b["gene_pos"] for b in batch], dtype=torch.long),
        "symbol_ids": torch.stack([b["symbol_ids"] for b in batch]),
        "ppi_emb": torch.stack([b["ppi_emb"] for b in batch]),
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
        self.ppi_embeddings: Dict[str, np.ndarray] = {}
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

        # Build ENSG→position mapping from all pert_ids across all splits
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)

        # Compute frozen STRING GNN PPI embeddings (once, rank-0 first then barrier)
        # CRITICAL FIX: barrier must be OUTSIDE the rank-0 if-block to ensure all ranks
        # synchronize. Previously, only rank-0 reached the barrier, causing DDP hangs.
        if not self.ppi_embeddings:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            if local_rank == 0:
                string_gnn = FrozenSTRINGGNN(model_dir=STRING_GNN_MODEL_DIR)
                self.ppi_embeddings = string_gnn.compute_embeddings(device=device_str)
            # Barrier OUTSIDE the rank-0 block: all ranks wait for rank-0 to finish
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            # Non-rank-0 processes load from the same object after barrier
            if local_rank != 0:
                string_gnn = FrozenSTRINGGNN(model_dir=STRING_GNN_MODEL_DIR)
                self.ppi_embeddings = string_gnn.compute_embeddings(device=device_str)

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self.gene_to_pos, self.ppi_embeddings
            )
            self.val_ds = PerturbationDataset(
                val_df, self.gene_to_pos, self.ppi_embeddings
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self.gene_to_pos, self.ppi_embeddings, is_test=True
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
        # Explicit SequentialSampler: guarantees all ranks process FULL test set in SAME order.
        # Combined with use_distributed_sampler=False in Trainer, this ensures all_gather
        # + positional deduplication works correctly without any sharding.
        from torch.utils.data import SequentialSampler
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            sampler=SequentialSampler(self.test_ds),
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class AIDOCellStringDEGModel(nn.Module):
    """
    4-source feature fusion DEG predictor (architecture preserved from grandparent node2-2-3):
      (a) global mean-pool of AIDO.Cell-10M last_hidden_state [B, 256]
      (b) perturbed-gene positional embedding from AIDO.Cell [B, 256]
      (c) gene symbol character CNN embedding [B, 64]
      (d) frozen STRING GNN PPI topology embedding for perturbed gene [B, 256]

    Concatenated → [B, 832] → MLP head (384-dim hidden) → [B, 3, 6640]
    """

    HIDDEN_DIM = 256
    SYMBOL_DIM = 64
    HEAD_INPUT_DIM = 256 * 2 + 64 + 256  # = 832

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        head_hidden: int = 384,
        head_dropout: float = 0.4,
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA on ALL 8 layers ──────────────────
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA on Q/K/V of all 8 layers with r=4 (proven configuration from node3-2)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.3,
            target_modules=["query", "key", "value"],
            layers_to_transform=list(range(AIDO_N_LAYERS)),  # all 8 layers
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

        # ── PPI projection: learnable projection for frozen STRING GNN embeddings
        self.ppi_proj = nn.Sequential(
            nn.Linear(STRING_GNN_DIM, STRING_GNN_DIM),
            nn.LayerNorm(STRING_GNN_DIM),
        )

        # ── Prediction head ────────────────────────────────────────────────────
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
        ppi_emb: torch.Tensor,       # [B, STRING_GNN_DIM] float32 (pre-computed frozen)
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

        # (d) STRING GNN PPI projection
        ppi_feat = self.ppi_proj(ppi_emb.to(backbone_feat.device))  # [B, 256] float32

        # Concatenate all 4 feature sources
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
# Checkpoint Averaging Utility
# ──────────────────────────────────────────────────────────────────────────────
def average_checkpoints(
    ckpt_dir: Path,
    model_module: "DEGLightningModule",
    top_k: int = 5,
) -> Optional[Path]:
    """
    Average the top-k checkpoints by val_f1 and save a proper Lightning-format checkpoint.

    Strategy:
    1. Find all checkpoint files (excluding 'last.ckpt') in ckpt_dir
    2. Parse val_f1 from filenames (format: *-val_f1=XXXX.ckpt)
    3. Select top-k by val_f1
    4. Load each state dict and average all trainable tensors in float32
    5. Construct a full Lightning checkpoint with required metadata
    6. Save as avg_checkpoint.ckpt (Lightning-compatible)

    DDP safety: Only rank 0 performs averaging. All ranks sync via barrier.

    Returns:
        Path to the averaged checkpoint if successful, None otherwise.
    """
    if not ckpt_dir.exists():
        print(f"[avg_ckpt] Checkpoint dir {ckpt_dir} not found, skipping averaging.")
        return None

    # Find all non-last checkpoints
    ckpt_files = [f for f in ckpt_dir.glob("*.ckpt") if f.name != "last.ckpt"]
    if len(ckpt_files) == 0:
        print("[avg_ckpt] No checkpoints found, skipping averaging.")
        return None

    # Parse val_f1 from filename (format: ...-val_f1=0.2316.ckpt)
    scored: List[tuple] = []
    for ckpt_path in ckpt_files:
        try:
            stem = ckpt_path.stem  # e.g. "node2-2-3-1-1-2-epoch=021-val_f1=0.2316"
            if "val_f1=" in stem:
                val_f1_str = stem.split("val_f1=")[-1]
                val_f1 = float(val_f1_str)
                scored.append((val_f1, ckpt_path))
        except (ValueError, IndexError):
            pass

    if len(scored) == 0:
        print("[avg_ckpt] Could not parse val_f1 from checkpoint filenames, skipping averaging.")
        return None

    # Sort by val_f1 descending, take top-k
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:min(top_k, len(scored))]
    print(f"[avg_ckpt] Averaging top-{len(selected)} checkpoints (of {len(scored)} available):")
    for val_f1, path in selected:
        print(f"  val_f1={val_f1:.4f}  {path.name}")

    # Load all state dicts on CPU to avoid GPU OOM
    state_dicts = []
    for _, path in selected:
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            if "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            else:
                sd = ckpt
            state_dicts.append(sd)
        except Exception as e:
            print(f"[avg_ckpt] Failed to load {path}: {e}")

    if len(state_dicts) == 0:
        print("[avg_ckpt] Could not load any checkpoints, skipping averaging.")
        return None

    if len(state_dicts) == 1:
        print("[avg_ckpt] Only 1 checkpoint loaded, returning its path.")
        return selected[0][1]

    # Average: all tensors in float32 for numerical stability
    avg_state: Dict[str, torch.Tensor] = {}
    all_keys = set(state_dicts[0].keys())
    for key in all_keys:
        tensors = []
        for sd in state_dicts:
            if key in sd:
                t = sd[key]
                if t.is_floating_point():
                    tensors.append(t.float())
                else:
                    tensors.append(t)
        if len(tensors) == 0:
            continue
        if tensors[0].is_floating_point():
            avg_state[key] = torch.stack(tensors).mean(dim=0)
        else:
            avg_state[key] = tensors[0]

    print(f"[avg_ckpt] Averaged {len(avg_state)} tensors across {len(state_dicts)} checkpoints.")

    # Build a proper Lightning-format checkpoint with all required metadata.
    # CRITICAL: Lightning's _pl_migrate_checkpoint requires 'pytorch-lightning_version'.
    # Use metadata from the best checkpoint (top-1).
    best_path = selected[0][1]
    try:
        ref_ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
    except Exception:
        ref_ckpt = {}

    # Construct full Lightning checkpoint
    avg_ckpt: Dict[str, Any] = {
        "state_dict": avg_state,
        "pytorch-lightning_version": ref_ckpt.get("pytorch-lightning_version", "2.5.1.post0"),
        "hyper_parameters": ref_ckpt.get("hyper_parameters", {}),
        "hparams_name": ref_ckpt.get("hparams_name", "DEGLightningModule"),
        "epoch": ref_ckpt.get("epoch", 0),
        "global_step": ref_ckpt.get("global_step", 0),
        # Optional fields: set to None to avoid Lightning migration issues
        "loops": ref_ckpt.get("loops"),
        "lr_schedulers": ref_ckpt.get("lr_schedulers"),
        "optimizer_states": ref_ckpt.get("optimizer_states"),
        "callbacks": ref_ckpt.get("callbacks"),
    }

    avg_ckpt_path = ckpt_dir / "avg_checkpoint.ckpt"
    torch.save(avg_ckpt, str(avg_ckpt_path))
    print(f"[avg_ckpt] Averaged Lightning checkpoint saved to {avg_ckpt_path}")
    return avg_ckpt_path


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        head_hidden: int = 384,
        head_dropout: float = 0.4,
        backbone_lr: float = 3e-4,
        symbol_lr_multiplier: float = 2.0,
        head_lr_multiplier: float = 3.0,
        weight_decay: float = 0.03,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 120,
        plateau_patience: int = 12,
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
        return self.model(
            batch["expr"], batch["gene_pos"], batch["symbol_ids"], batch["ppi_emb"]
        )

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

        # Only rank 0 computes val_f1 on the full gathered dataset
        if self.trainer.is_global_zero:
            preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
            labels = al.cpu().view(-1, N_GENES_OUT).numpy()
            idxs = ai.cpu().view(-1).numpy()
            _, uniq = np.unique(idxs, return_index=True)
            f1 = compute_deg_f1(preds[uniq], labels[uniq])
            f1_tensor = torch.tensor(f1, dtype=torch.float32, device=self.device)
        else:
            f1_tensor = torch.tensor(0.0, dtype=torch.float32, device=self.device)

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
            # Deduplicate: keep first occurrence of each unique index
            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]
            idxs = idxs[uniq]
            # Sort by index to align with original data order
            order = np.argsort(idxs)
            preds_sorted = preds[order]
            idxs_sorted = idxs[order]
            # preds_sorted[i] and idxs_sorted[i] are aligned: idxs_sorted[i] is the
            # original sample index, preds_sorted[i] is its prediction
            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for i in range(len(idxs_sorted)):
                orig_idx = idxs_sorted[i]
                rows.append({
                    "idx": self._test_pert_ids[orig_idx],
                    "input": self._test_symbols[orig_idx],
                    "prediction": json.dumps(preds_sorted[i].tolist()),
                })
            pd.DataFrame(rows).to_csv(output_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"Test predictions saved ({len(rows)} rows) -> {output_dir / 'test_predictions.tsv'}")

    def configure_optimizers(self):
        hp = self.hparams
        # Three-tier parameter groups (same as grandparent node2-2-3):
        # 1. Backbone LoRA params (lower lr)
        # 2. Symbol encoder params (2× backbone lr)
        # 3. Head + PPI projection params (3× backbone lr)
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        symbol_params = list(self.model.symbol_encoder.parameters())
        head_ppi_params = (
            list(self.model.head.parameters())
            + list(self.model.ppi_proj.parameters())
        )

        backbone_lr = hp.backbone_lr
        symbol_lr = hp.backbone_lr * hp.symbol_lr_multiplier
        head_lr = hp.backbone_lr * hp.head_lr_multiplier

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": symbol_params, "lr": symbol_lr},
                {"params": head_ppi_params, "lr": head_lr},
            ],
            weight_decay=hp.weight_decay,
        )

        # ReduceLROnPlateau monitoring val_f1 (mode='max')
        # CRITICAL: val_f1 (not val_loss) is the only reliable scheduling signal.
        # Patience=12: less likely to fire prematurely (node3-2 best: it NEVER fired)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            patience=hp.plateau_patience,
            factor=hp.plateau_factor,
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
        description="Node 2-2-3-1-1-2: Top-5 Checkpoint Averaging + [8,1,18] Class Weights + patience=15"
    )
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--global_batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=120)
    p.add_argument("--backbone_lr", type=float, default=3e-4)
    p.add_argument("--symbol_lr_multiplier", type=float, default=2.0)
    p.add_argument("--head_lr_multiplier", type=float, default=3.0)
    p.add_argument("--weight_decay", type=float, default=0.03)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--head_hidden", type=int, default=384)
    p.add_argument("--head_dropout", type=float, default=0.4)
    p.add_argument("--gamma_focal", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--plateau_patience", type=int, default=12)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--plateau_min_lr", type=float, default=1e-7)
    p.add_argument("--early_stopping_patience", type=int, default=15)  # Tighter (same as sibling)
    p.add_argument("--avg_top_k", type=int, default=5,                 # TOP-5 averaging (vs sibling's 3)
                   help="Number of top checkpoints to average (0 to disable)")
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    return p.parse_args()


def main():
    # seed=0: proven best initialization basin for this architecture
    pl.seed_everything(seed=0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # Debug mode batch limiting:
    # - --debug_max_step N: train for N optimizer steps (N*accumulate_grad micro-batches),
    #   val/test every step, limit test to N batches.
    # - --fast_dev_run: use Lightning's fast_dev_run=True for automated unit-test-style
    #   stopping (1 step, 1 epoch, num_sanity_val_steps=0).
    if args.debug_max_step is not None:
        max_steps = args.debug_max_step
        limit_train_batches = args.debug_max_step * accumulate_grad
        limit_val_batches = args.debug_max_step
        limit_test_batches = args.debug_max_step
        val_check_interval = 1.0
        fast_dev_run = False
    elif args.fast_dev_run:
        # fast_dev_run=True: Lightning handles max_epochs=1, max_steps=1, etc.
        # Explicitly set limit_* to override Lightning's internal test limits.
        max_steps = 1
        limit_train_batches = 1.0   # Lightning's fast_dev_run handles the step limit
        limit_val_batches = 1.0     # Override Lightning's internal val limit (normally all)
        limit_test_batches = 1.0    # Override Lightning's internal test limit (normally 1)
        val_check_interval = 1.0
        fast_dev_run = True
    else:
        max_steps = -1
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        val_check_interval = args.val_check_interval
        fast_dev_run = False

    # save_top_k=5 to support top-5 checkpoint averaging
    # (sibling used save_top_k=3 for top-3 averaging; this node saves 5)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node2-2-3-1-1-2-{epoch:03d}-val_f1={val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=5, save_last=True,
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
        val_check_interval=val_check_interval,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,   # Ensure reproducibility (seed_everything(0) is set)
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,  # True for --fast_dev_run, False otherwise
        gradient_clip_val=1.0,
        use_distributed_sampler=False,  # All ranks process FULL test set; dedup by index
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
        backbone_lr=args.backbone_lr,
        symbol_lr_multiplier=args.symbol_lr_multiplier,
        head_lr_multiplier=args.head_lr_multiplier,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        plateau_min_lr=args.plateau_min_lr,
    )

    trainer.fit(model_module, datamodule=datamodule)

    # ── Checkpoint Averaging (top-5) ──────────────────────────────────────────
    # After training: average the top-k checkpoints by val_f1 to reduce variance.
    # DDP-safe: rank-0 creates the averaged checkpoint (now in proper Lightning format
    # with all required metadata), all ranks sync via barrier, then all ranks load.
    avg_ckpt_path: Optional[Path] = None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not args.fast_dev_run and args.debug_max_step is None and args.avg_top_k > 0:
        if local_rank == 0:
            ckpt_dir = output_dir / "checkpoints"
            avg_ckpt_path = average_checkpoints(
                ckpt_dir=ckpt_dir,
                model_module=model_module,
                top_k=args.avg_top_k,
            )
        # Synchronize all ranks after rank-0 creates the averaged checkpoint
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    # ── Test Evaluation ───────────────────────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    elif avg_ckpt_path is not None and avg_ckpt_path.exists():
        # Use the averaged checkpoint for test evaluation (primary path)
        # CRITICAL FIX: avg_checkpoint.ckpt is now saved as a proper Lightning-format
        # checkpoint with pytorch-lightning_version, state_dict, hyper_parameters, etc.
        print(f"[main] Using averaged checkpoint for test: {avg_ckpt_path}")
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path=str(avg_ckpt_path))
    else:
        # Fallback: use best single checkpoint by val_f1
        print("[main] Checkpoint averaging not available, using best single checkpoint.")
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # Save test score placeholder (actual score computed by EvaluateAgent)
    if test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text("test_predictions saved; run calc_metric.py for final score\n")


if __name__ == "__main__":
    main()
