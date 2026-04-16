#!/usr/bin/env python3
"""
Node 2-2-3-1-1-1-1-1: LoRA Rank Reduction r=4→r=2 + Reverted Class Weights [7,1,15] + Top-3 Averaging
=======================================================================================================

Parent: node2-2-3-1-1-1-1 (test F1=0.4655, tied with parent node2-2-3-1-1-1)

This node implements two targeted improvements based on the parent's feedback:

1. LoRA Rank Reduction: r=4 → r=2 (PRIMARY RECOMMENDATION from parent feedback)
   - Parent feedback: "Reduce LoRA rank from r=4 to r=2 to combat overfitting
     (reducing trainable params from 8.1M to ~4M)"
   - Severe overfitting confirmed: train_loss 0.564→0.107, val_loss 0.718→1.450
   - val-test gap (~0.235) suggests excess model capacity for 1,500 training samples
   - Halving LoRA capacity (r=4 → r=2) reduces backbone trainable params by ~50%
     while preserving the proven architecture of frozen STRING GNN + Symbol CNN
   - lora_dropout maintained at 0.3 (appropriate for smaller r to maintain regularization)
   - lora_alpha stays at 8 (ratio: alpha/r = 8/2 = 4, providing stronger scaling)

2. Reverted Class Weights: [8,1,18] → [7,1,15] (CONFIRMED SATURATION)
   - Parent feedback: "The [7→8] step is at the saturation point. The optimal is
     at or near [7,1,15]. Do NOT continue strengthening beyond [8,1,18]."
   - The [5,1,10]→[6,1,12]→[7,1,15]→[8,1,18] trend clearly shows diminishing
     returns at the [7→8] step (0.4655 → 0.4655, zero gain)
   - Reverting to [7,1,15] restores the last proven improvement point

3. Top-3 Checkpoint Averaging + patience=15 (RESTORED from grandparent pattern)
   - Parent averaging top-5 over epochs 19-25 lost diversity vs grandparent's
     top-3 spanning epochs 17, 21, 35 (pre-peak, peak, post-LR phases)
   - Restoring save_top_k=3 + avg_top_k=3 + patience=15 recreates the beneficial
     epoch diversity that drove the grandparent's +0.003 gain

Architecture: IDENTICAL to parent node2-2-3-1-1-1-1 except LoRA r=2:
  AIDO.Cell-10M + LoRA QKV r=2 all-8-layers + 3-branch Symbol CNN (64-dim) +
  Frozen STRING GNN PPI (256-dim) → concat (832-dim) → MLP head (384→19920)

Training config: seed=0, weight_decay=0.03, plateau_patience=12 (all unchanged)
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

# Reverted class weights: [7.0, 1.0, 15.0] (from parent's [8.0, 1.0, 18.0])
# Evidence from MCTS tree progression:
#   [5,1,10] → 0.4573  (node2-3-1-1-1-1)
#   [6,1,12] → 0.4586  (node2-3-1-1-1-1-1-1), 0.4625 (node2-2-3-1-1)
#   [7,1,15] → 0.4655  (node2-2-3-1-1-1 - the first to achieve this score)
#   [8,1,18] → 0.4655  (node2-2-3-1-1-1-1 - zero gain, saturation confirmed)
# Reverting to [7,1,15]: the last step with proven positive impact.
CLASS_WEIGHTS = torch.tensor([7.0, 1.0, 15.0], dtype=torch.float32)

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
            # node_names are in Ensembl format (e.g., "ENSG00000000938")
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
        # Shape: [N, 19264] float32 — baseline 1.0, knocked-out gene 0.0
        self.expr_inputs = self._build_expr_tensors()

        # Pre-build symbol character index tensors
        # Shape: [N, SYMBOL_MAX_LEN] int64
        self.symbol_ids = self._build_symbol_tensors()

        # Pre-build STRING GNN PPI embedding tensors
        # Shape: [N, STRING_GNN_DIM] float32 (zero vector for genes not in STRING vocab)
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
        "expr": torch.stack([b["expr"] for b in batch]),            # [B, 19264]
        "gene_pos": torch.tensor([b["gene_pos"] for b in batch], dtype=torch.long),
        "symbol_ids": torch.stack([b["symbol_ids"] for b in batch]),  # [B, SYMBOL_MAX_LEN]
        "ppi_emb": torch.stack([b["ppi_emb"] for b in batch]),      # [B, STRING_GNN_DIM]
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
    4-source feature fusion DEG predictor:
      (a) global mean-pool of AIDO.Cell-10M last_hidden_state [B, 256]
      (b) perturbed-gene positional embedding from AIDO.Cell [B, 256]
      (c) gene symbol character CNN embedding [B, 64]
      (d) frozen STRING GNN PPI topology embedding for perturbed gene [B, 256]

    Concatenated → [B, 832] → MLP head (384-dim hidden) → [B, 3, 6640]

    Key change from parent: LoRA rank r=2 (from r=4)
    - Reduces backbone trainable params by ~50% (~4M vs ~8.1M)
    - lora_alpha=8, ratio alpha/r=4 (stronger scaling than parent's 8/4=2)
    - lora_dropout=0.3 maintained for regularization
    - All 8 transformer layers still get LoRA adapters (confirmed optimal config)
    """

    HIDDEN_DIM = 256           # AIDO.Cell-10M hidden size
    SYMBOL_DIM = 64            # gene symbol embedding dimension
    HEAD_INPUT_DIM = 256 * 2 + 64 + 256  # = 832 (backbone×2 + symbol + STRING)

    def __init__(
        self,
        lora_r: int = 2,
        lora_alpha: int = 8,
        head_hidden: int = 384,
        head_dropout: float = 0.4,
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA on ALL 8 layers ──────────────────
        # KEY CHANGE: r=2 (from r=4) to reduce overfitting
        # With r=2:
        #   - Each LoRA adapter: query (256×2 + 2×256 = ~1K), key, value similarly
        #   - Total LoRA params: ~3 × 8 × (256×2 + 2×256) = ~3.1M (vs ~6.1M for r=4)
        #   - Combined with head params (~5M): total trainable ~8M (vs ~13M for r=4)
        # lora_alpha=8 means scaling = alpha/r = 8/2 = 4.0 (vs 8/4=2.0 for r=4)
        # This stronger scaling compensates for reduced rank capacity
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA on Q/K/V of all 8 layers with r=2 (reduced from parent's r=4)
        # lora_dropout=0.3: maintained (appropriate for small rank to add regularization)
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

        # ── PPI projection: linear projection for frozen STRING GNN embeddings ─
        # A single linear layer to project pre-computed 256-dim STRING embeddings
        # into the fusion space (identity-like but learnable for domain adaptation)
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
    model_module: "DEGLightningModule",
    checkpoint_paths: List[str],
    output_path: str,
) -> None:
    """
    Average the trainable parameters from multiple checkpoints and save the
    result to output_path as a Lightning-compatible checkpoint.

    This function is called on rank-0 only; other ranks synchronize via DDP barrier.
    All tensors are averaged in float32 for numerical stability.

    Args:
        model_module: The LightningModule (used only to determine expected keys)
        checkpoint_paths: Sorted list of checkpoint file paths to average
        output_path: Path to write the averaged checkpoint
    """
    print(f"[CheckpointAveraging] Averaging {len(checkpoint_paths)} checkpoints:")
    for p in checkpoint_paths:
        print(f"  - {p}")

    avg_state_dict: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)  # handle both formats
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                v_f = v.float()
                if k not in avg_state_dict:
                    avg_state_dict[k] = v_f.clone()
                    counts[k] = 1
                else:
                    avg_state_dict[k] = avg_state_dict[k] + v_f
                    counts[k] += 1

    # Divide by count to get the mean
    for k in avg_state_dict:
        avg_state_dict[k] = avg_state_dict[k] / counts[k]

    # Save as a Lightning-compatible checkpoint with all required keys
    import lightning as pl_version
    ckpt = {
        "state_dict": avg_state_dict,
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": pl_version.__version__,
    }
    torch.save(ckpt, output_path)
    print(f"[CheckpointAveraging] Averaged checkpoint saved to: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 2,
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

        # All ranks gather complete predictions/labels and compute val_f1 to avoid
        # the PL warning about sync_dist=True on epoch-level logging.
        # After all_gather, each rank has a copy of the full dataset (since
        # DistributedSampler gives each rank a unique slice and all_gather
        # broadcasts all slices to all ranks). Deduplication ensures correctness.
        # sync_dist=True ensures all ranks log the identical computed value.
        preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
        labels = al.cpu().view(-1, N_GENES_OUT).numpy()
        idxs = ai.cpu().view(-1).numpy()
        _, uniq = np.unique(idxs, return_index=True)
        f1 = compute_deg_f1(preds[uniq], labels[uniq])
        self.log("val_f1", float(f1), prog_bar=True, sync_dist=True)

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
        # Three-tier parameter groups (same as proven parent configuration):
        # 1. Backbone LoRA params (lower lr)
        # 2. Symbol encoder params (2× backbone lr, separately from head)
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
        # plateau_patience=12: fires once at ~epoch 33, matching proven training dynamics.
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
        description="Node 2-2-3-1-1-1-1-1: LoRA r=2 + Class Weights [7,1,15] + Top-3 Checkpoint Averaging"
    )
    # Use __file__-relative path to data directory so it works regardless of
    # the working directory from which the script is launched via torchrun.
    # The mcts/node symlink resolves to an external path (../../mcts/nodeX),
    # so we go up 3 levels from the script location to reach the working directory.
    default_data_dir = str(Path(__file__).parent.parent.parent / "data")
    p.add_argument("--data_dir", type=str, default=default_data_dir)
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--global_batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=120)
    p.add_argument("--backbone_lr", type=float, default=3e-4)
    p.add_argument("--symbol_lr_multiplier", type=float, default=2.0)
    p.add_argument("--head_lr_multiplier", type=float, default=3.0)
    p.add_argument("--weight_decay", type=float, default=0.03)
    # KEY CHANGE: lora_r=2 (from r=4 in parent)
    # Rationale: reduce overfitting by halving LoRA capacity
    p.add_argument("--lora_r", type=int, default=2)
    # lora_alpha=8: alpha/r=4.0 (stronger scaling vs parent's alpha/r=2.0)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--head_hidden", type=int, default=384)
    p.add_argument("--head_dropout", type=float, default=0.4)
    p.add_argument("--gamma_focal", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--plateau_patience", type=int, default=12)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--plateau_min_lr", type=float, default=1e-7)
    # patience=15: restored from grandparent (parent2-2-3-1-1-1) proven configuration
    # Parent's patience=20 caused narrow checkpoint diversity (epochs 19-25)
    # vs grandparent's patience=15 which gave diverse checkpoints (epochs 17, 21, 35)
    p.add_argument("--early_stopping_patience", type=int, default=15)
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    # avg_top_k=3: restored from grandparent (parent2-2-3-1-1-1) proven configuration
    # Parent's top-5 over narrow band (19-25) was worse than grandparent's top-3
    # spanning pre-peak (17), peak (21), post-LR (35): more diverse → better averaging
    p.add_argument("--avg_top_k", type=int, default=3,
                   help="Number of top checkpoints to average for test evaluation (0=disabled)")
    return p.parse_args()


def main():
    # seed=0: proven best initialization for this architecture
    pl.seed_everything(seed=0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    if args.debug_max_step is not None:
        limit_train_batches = args.debug_max_step * accumulate_grad
        limit_val_batches = args.debug_max_step
        # Test must always run on the FULL test set to produce complete predictions.
        # debug_max_step only limits training/validation for rapid code verification.
        limit_test_batches = 1.0
    else:
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0

    # save_top_k=3: restored from grandparent's proven config (parent's save_top_k=5
    # produced narrow epoch diversity). With patience=15 and peak typically at epoch
    # ~20-21, saves diverse checkpoints covering pre-peak, peak, post-LR phases.
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node2-2-3-1-1-1-1-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=3, save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1", mode="max",
        # patience=15: restored from grandparent's proven configuration
        # Parent's patience=20 extended training but created narrow checkpoint diversity
        # (all within 6 epochs of peak). Grandparent's patience=15 stopped earlier
        # but captured broader diversity (pre-peak + peak + post-LR phases).
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

    # ── Checkpoint averaging for final test evaluation ─────────────────────────
    # Top-3 checkpoint averaging: restored from grandparent's proven configuration.
    # The grandparent (node2-2-3-1-1-1) achieved the first 0.4655 score using
    # top-3 averaging with checkpoints from epochs 17, 21, 35 — spanning three
    # distinct optimization phases (pre-peak, peak, post-LR-reduction). This
    # diversity was the key to the +0.003 gain over the non-averaging grandparent.
    # The parent's top-5 averaging failed because all 5 checkpoints landed in the
    # same narrow 6-epoch window (epochs 19-25), losing the epoch diversity that
    # made the grandparent's approach effective.
    # ──────────────────────────────────────────────────────────────────────────────

    is_full_run = not (args.fast_dev_run or args.debug_max_step is not None)

    if is_full_run and args.avg_top_k > 0:
        # Collect available top-k checkpoint paths from the callback
        # best_k_models is a dict {path: metric_value}
        best_k = checkpoint_cb.best_k_models  # dict[str, Tensor]
        if len(best_k) > 1:
            # Sort by metric value (descending, higher val_f1 first)
            top_k_paths = sorted(
                best_k.keys(),
                key=lambda p: float(best_k[p]),
                reverse=True,
            )[:args.avg_top_k]

            avg_ckpt_path = str(output_dir / "checkpoints" / "avg_checkpoint.ckpt")
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))

            # Rank-0 creates the averaged checkpoint
            if local_rank == 0:
                average_checkpoints(model_module, top_k_paths, avg_ckpt_path)

            # Synchronize all ranks so they all see the averaged checkpoint file
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

            # All ranks load the same averaged checkpoint for test evaluation
            if Path(avg_ckpt_path).exists():
                print(f"[Main] Using averaged checkpoint for test: {avg_ckpt_path}")
                test_results = trainer.test(model_module, datamodule=datamodule,
                                            ckpt_path=avg_ckpt_path)
            else:
                # Fallback to best single checkpoint if averaging failed
                print("[Main] Averaged checkpoint not found, falling back to best single checkpoint")
                test_results = trainer.test(model_module, datamodule=datamodule,
                                            ckpt_path="best")
        else:
            # Only one checkpoint saved (early training or debug) — use it directly
            print("[Main] Only one checkpoint available, using best single checkpoint")
            test_results = trainer.test(model_module, datamodule=datamodule,
                                        ckpt_path="best")
    elif is_full_run:
        # Checkpoint averaging disabled (--avg_top_k 0)
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")
    else:
        # Debug/fast_dev_run: skip checkpoint loading
        test_results = trainer.test(model_module, datamodule=datamodule)

    # Save test score placeholder for easy retrieval
    if test_results is not None:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text("test_predictions saved; run calc_metric.py for final score\n")


if __name__ == "__main__":
    main()
