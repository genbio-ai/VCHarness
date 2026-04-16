#!/usr/bin/env python3
"""
Node 2-2-1-2-1: Cross-Attention Fusion (AIDO.Cell-10M + LoRA r=4 all-8 + Symbol CNN + STRING GNN)
===================================================================================================
Improves on parent node (node2-2-1-2: AIDO.Cell-10M + LoRA r=8 last-4 + symbol CNN +
STRING GNN + concat+MLP, test F1=0.4511) by adopting the breakthrough cross-attention
fusion architecture validated by the tree-best node3-1-3-1-1-1-1 (test F1=0.4768).

Key differences from parent node (node2-2-1-2):
  1. PRIMARY: Replace concat+MLP head with 4-token TransformerEncoder cross-attention fusion
     - 4 tokens: global_emb, pert_emb, sym_proj, ppi_feat (all 256-dim)
     - 3-layer TransformerEncoder (nhead=8, dim_ff=256, attn_dropout=0.2)
     - Proven to break ~0.46 concat+MLP ceiling (+0.015 F1 in node3-1-3-1 branch)
     - Inter-source feature interactions (PPI ↔ expression ↔ symbol) are modeled

  2. LoRA: r=8 last-4-layers → r=4 all-8-layers
     - Broader but shallower adaptation proven to outperform narrow deep (node3-2 0.462 vs node3-3 0.451)
     - Matches exact tree-best LoRA configuration

  3. Remove expression noise augmentation (input_noise_std=0.0)
     - Parent feedback: augmentation diluted signal without benefit
     - Add manifold mixup (alpha=0.3) in fusion embedding space instead

  4. LR scheduler: ReduceLROnPlateau(val_loss) → ReduceLROnPlateau(val_f1, patience=12)
     - Critical insight: focal loss makes val_loss inversely correlated with val_f1
     - val_f1 is the only reliable scheduling signal for this task

  5. weight_decay: 0.04 → 0.10 (matches tree-best regularization)
     class_weights: [5,1,10] → [6,1,12] (matches tree-best loss configuration)
     backbone_lr: 3e-4 → 2e-4 (matches tree-best)

  6. Top-3 checkpoint averaging for test predictions (proven +0.003 F1 in tree)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Fix NCCL segfault on certain clusters with InfiniBand
os.environ.setdefault('NCCL_IB_DISABLE', '1')
os.environ.setdefault('NCCL_NET_GDR_LEVEL', '0')

import argparse
import json
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
FUSION_DIM = 256         # unified token dimension for cross-attention fusion
N_FUSION_TOKENS = 4      # number of source tokens (global, pert, symbol, ppi)

# Class weights: [6,1,12] from tree-best (node3-1-3-1, node3-1-3-1-1-1-1)
# Train distribution: class 0 (down) ~3.4%, class 1 (unchanged) ~95.5%, class 2 (up) ~1.1%
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
    """Character-level CNN encoder for gene symbol strings → [B, 64]."""

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
        x = self.embed(symbol_ids)
        x = x.transpose(1, 2)
        f2 = F.relu(self.conv2(x))
        f3 = F.relu(self.conv3(x))
        f4 = F.relu(self.conv4(x))
        f2 = F.adaptive_max_pool1d(f2, 1).squeeze(-1)
        f3 = F.adaptive_max_pool1d(f3, 1).squeeze(-1)
        f4 = F.adaptive_max_pool1d(f4, 1).squeeze(-1)
        feat = torch.cat([f2, f3, f4], dim=-1)
        return self.proj(feat)


# ──────────────────────────────────────────────────────────────────────────────
# STRING GNN helper: precompute all node embeddings once
# ──────────────────────────────────────────────────────────────────────────────
def compute_string_gnn_embeddings(device: torch.device) -> tuple:
    """Load STRING GNN and compute all 18870 node embeddings once."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    ensg_to_idx = {name: i for i, name in enumerate(node_names)}

    string_model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
    string_model.eval()
    string_model = string_model.to(device)

    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location=device)
    edge_index = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"]
    edge_weight = edge_weight.to(device) if edge_weight is not None else None

    with torch.no_grad():
        outputs = string_model(edge_index=edge_index, edge_weight=edge_weight)
        all_emb = outputs.last_hidden_state.float().cpu()

    del string_model, edge_index, edge_weight
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return all_emb, ensg_to_idx


def lookup_ppi_embeddings(
    pert_ids: List[str],
    all_emb: torch.Tensor,
    ensg_to_idx: Dict[str, int],
) -> torch.Tensor:
    """Look up STRING GNN embeddings for a list of pert_ids. Returns [N, 256]."""
    N = len(pert_ids)
    emb = torch.zeros(N, STRING_GNN_DIM, dtype=torch.float32)
    for i, pert_id in enumerate(pert_ids):
        base = pert_id.split(".")[0]
        idx = ensg_to_idx.get(base, -1)
        if idx >= 0:
            emb[i] = all_emb[idx]
    return emb


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Returns expression profile, gene position, symbol ids, PPI embeddings, and labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],
        ppi_emb: torch.Tensor,          # [N, 256] float32
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.is_test = is_test

        # Pre-build expression tensors: [N, 19264] float32
        self.expr_inputs = self._build_expr_tensors()

        # Pre-build symbol character index tensors: [N, SYMBOL_MAX_LEN] int64
        self.symbol_ids = self._build_symbol_tensors()

        # STRING GNN PPI embeddings: [N, 256] float32
        assert ppi_emb.shape[0] == len(self.pert_ids), \
            f"ppi_emb shape mismatch: {ppi_emb.shape[0]} != {len(self.pert_ids)}"
        self.ppi_emb = ppi_emb

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
            indices = symbol_to_indices(symbol)
            sym_ids[i] = torch.tensor(indices, dtype=torch.long)
        return sym_ids

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.pert_ids[idx].split(".")[0]
        gene_pos = self.gene_to_pos.get(base, -1)

        item = {
            "idx": idx,
            "expr": self.expr_inputs[idx].clone(),   # [19264] float32
            "gene_pos": gene_pos,                     # int
            "symbol_ids": self.symbol_ids[idx],       # [SYMBOL_MAX_LEN] int64
            "ppi_emb": self.ppi_emb[idx],             # [256] float32
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
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []
        self._string_all_emb: Optional[torch.Tensor] = None
        self._string_ensg_to_idx: Optional[Dict[str, int]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Step 1: Rank-0 downloads AIDO.Cell tokenizer first, then barrier
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # Step 2: Build ENSG → position mapping
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)

        # Step 3: Pre-compute STRING GNN embeddings (each rank computes locally, then sync)
        # FIX: All ranks compute independently to avoid barrier deadlock where non-zero ranks
        # skip computation and hang at the barrier waiting for rank 0 (which is still computing).
        if self._string_all_emb is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._string_all_emb, self._string_ensg_to_idx = compute_string_gnn_embeddings(device)
            # Sync after all ranks complete computation to ensure consistency
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

        # Step 4: Build dataset splits
        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")

            train_ppi_emb = lookup_ppi_embeddings(
                train_df["pert_id"].tolist(), self._string_all_emb, self._string_ensg_to_idx)
            val_ppi_emb = lookup_ppi_embeddings(
                val_df["pert_id"].tolist(), self._string_all_emb, self._string_ensg_to_idx)

            self.train_ds = PerturbationDataset(train_df, self.gene_to_pos, train_ppi_emb,
                                                is_test=False)
            self.val_ds = PerturbationDataset(val_df, self.gene_to_pos, val_ppi_emb,
                                              is_test=False)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            test_ppi_emb = lookup_ppi_embeddings(
                test_df["pert_id"].tolist(), self._string_all_emb, self._string_ensg_to_idx)

            self.test_ds = PerturbationDataset(test_df, self.gene_to_pos, test_ppi_emb,
                                               is_test=True)
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
# Model
# ──────────────────────────────────────────────────────────────────────────────
class CrossAttentionDEGModel(nn.Module):
    """
    AIDO.Cell-10M backbone (LoRA r=4 all-8 layers) + character-level symbol CNN +
    frozen STRING GNN PPI embeddings, fused via 4-token TransformerEncoder cross-attention
    + manifold mixup, → 256-dim → prediction head → [B, 3, 6640].

    Architecture:
      (a) global_emb: AIDO.Cell global mean-pool [B, 256]
      (b) pert_emb:   AIDO.Cell perturbed gene positional embedding [B, 256]
      (c) sym_feat:   symbol CNN output [B, 64] → projected to [B, 256]
      (d) ppi_feat:   STRING GNN PPI embedding [B, 256] → projected to [B, 256]

      → Stack as [B, 4, 256] tokens with learnable positional embeddings
      → 3-layer TransformerEncoder (nhead=8, dim_ff=256, attn_dropout=0.2)
      → mean-pool → [B, 256]
      → Manifold mixup (training only, alpha=0.3)
      → LayerNorm → Linear(256→256) → GELU → Dropout(0.5) → Linear(256→3×6640)
    """

    HIDDEN_DIM = 256     # AIDO.Cell-10M hidden size
    SYMBOL_DIM = 64      # gene symbol CNN output dimension

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.10,
        lora_layers_start: int = 0,   # 0 = all 8 layers
        attn_n_layers: int = 3,
        attn_nhead: int = 8,
        attn_dim_ff: int = 256,
        attn_dropout: float = 0.2,
        head_dropout: float = 0.5,
        mixup_alpha: float = 0.3,
    ):
        super().__init__()
        self.mixup_alpha = mixup_alpha

        # ── AIDO.Cell-10M backbone with LoRA ──────────────────────────────────
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

        # ── Source projection modules (all → FUSION_DIM=256) ──────────────────
        # Project each feature source to the fusion token dimension
        self.global_proj = nn.Sequential(
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.Linear(self.HIDDEN_DIM, FUSION_DIM),
            nn.GELU(),
        )
        self.pert_proj = nn.Sequential(
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.Linear(self.HIDDEN_DIM, FUSION_DIM),
            nn.GELU(),
        )
        # Symbol CNN output is 64-dim → up-project to 256
        self.sym_proj = nn.Sequential(
            nn.LayerNorm(self.SYMBOL_DIM),
            nn.Linear(self.SYMBOL_DIM, FUSION_DIM),
            nn.GELU(),
        )
        # STRING GNN PPI: 256-dim → 256-dim (near-identity init, high LR)
        self.ppi_proj = nn.Sequential(
            nn.LayerNorm(STRING_GNN_DIM),
            nn.Linear(STRING_GNN_DIM, FUSION_DIM),
            nn.GELU(),
        )
        # Initialize ppi_proj linear with near-identity
        nn.init.eye_(self.ppi_proj[1].weight)
        nn.init.zeros_(self.ppi_proj[1].bias)

        # ── Learnable positional embeddings for 4 tokens ───────────────────────
        self.token_pos_emb = nn.Embedding(N_FUSION_TOKENS, FUSION_DIM)
        nn.init.normal_(self.token_pos_emb.weight, std=0.02)

        # ── 3-layer TransformerEncoder cross-attention fusion ─────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FUSION_DIM,
            nhead=attn_nhead,
            dim_feedforward=attn_dim_ff,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-norm for stability
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=attn_n_layers,
            enable_nested_tensor=False,
        )

        # ── Prediction head: 256 → 256 → 3×6640 ─────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, FUSION_DIM),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(FUSION_DIM, N_CLASSES * N_GENES_OUT),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def _project_sources(
        self,
        global_emb: torch.Tensor,   # [B, 256] float32
        pert_emb: torch.Tensor,     # [B, 256] float32
        sym_feat: torch.Tensor,     # [B, 64] float32
        ppi_emb: torch.Tensor,      # [B, 256] float32
    ) -> torch.Tensor:
        """Project 4 sources to FUSION_DIM and stack as tokens [B, 4, 256]."""
        t0 = self.global_proj(global_emb)     # [B, 256]
        t1 = self.pert_proj(pert_emb)         # [B, 256]
        t2 = self.sym_proj(sym_feat)          # [B, 256]
        t3 = self.ppi_proj(ppi_emb)           # [B, 256]

        tokens = torch.stack([t0, t1, t2, t3], dim=1)  # [B, 4, 256]

        # Add learnable positional embeddings
        pos_ids = torch.arange(N_FUSION_TOKENS, device=tokens.device)
        tokens = tokens + self.token_pos_emb(pos_ids).unsqueeze(0)  # [B, 4, 256]
        return tokens

    def forward(
        self,
        expr: torch.Tensor,          # [B, 19264] float32
        gene_pos: torch.Tensor,      # [B] int64
        symbol_ids: torch.Tensor,    # [B, SYMBOL_MAX_LEN] int64
        ppi_emb: torch.Tensor,       # [B, 256] float32
        labels: Optional[torch.Tensor] = None,   # [B, 6640] int64, only during training
        mixup_alpha: Optional[float] = None,
    ):
        # ── AIDO.Cell backbone forward ─────────────────────────────────────────
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, 19266, 256]

        # (a) Global mean-pool over gene positions
        gene_emb = lhs[:, :N_GENES_AIDO, :]
        global_emb = gene_emb.mean(dim=1).float()  # [B, 256]

        # (b) Perturbed-gene positional embedding
        B = expr.shape[0]
        pert_emb_f = torch.zeros(B, self.HIDDEN_DIM, device=lhs.device, dtype=lhs.dtype)
        valid_mask = gene_pos >= 0
        if valid_mask.all():
            # All positions valid — direct indexing
            pert_emb_f = lhs[torch.arange(B, device=lhs.device), gene_pos, :].float()
        elif valid_mask.any():
            # Mixed batch — handle valid and invalid separately
            valid_pos = gene_pos[valid_mask]
            pert_emb_f[valid_mask] = lhs[valid_mask, valid_pos, :].float()
            # Invalid positions: use mean-pool over all gene embeddings
            pert_emb_f[~valid_mask] = lhs[~valid_mask, :N_GENES_AIDO, :].mean(dim=1).float()
        else:
            # No valid positions in batch — fallback to mean-pool over all gene embeddings
            gene_emb = lhs[:, :N_GENES_AIDO, :]
            pert_emb_f = gene_emb.mean(dim=1).float()
        # pert_emb_f is now [B, 256] float32

        # (c) Gene symbol CNN embedding
        sym_feat = self.symbol_encoder(symbol_ids)  # [B, 64] float32

        # (d) STRING GNN PPI embedding
        ppi_emb_f = ppi_emb.to(expr.device).float()  # [B, 256] float32

        # ── Project all sources to 256-dim tokens ─────────────────────────────
        tokens = self._project_sources(global_emb, pert_emb_f, sym_feat, ppi_emb_f)
        # [B, 4, 256]

        # ── 3-layer TransformerEncoder fusion ─────────────────────────────────
        fused = self.fusion_transformer(tokens)   # [B, 4, 256]
        fused_pooled = fused.mean(dim=1)          # [B, 256]  mean-pool over tokens

        # ── Manifold mixup (training only) ────────────────────────────────────
        if labels is not None and mixup_alpha is not None and mixup_alpha > 0:
            fused_pooled, labels = self._manifold_mixup(fused_pooled, labels)

        # ── Prediction head ────────────────────────────────────────────────────
        logits = self.head(fused_pooled)                   # [B, 3 * 6640]
        logits = logits.view(B, N_CLASSES, N_GENES_OUT)    # [B, 3, 6640]

        if labels is not None:
            return logits, labels
        return logits

    def _manifold_mixup(
        self,
        features: torch.Tensor,   # [B, 256]
        labels: torch.Tensor,     # [B, 6640] int64
    ) -> tuple:
        """Apply manifold mixup in the 256-dim fusion embedding space."""
        alpha = self.mixup_alpha
        if alpha <= 0:
            return features, labels

        lam = float(np.random.beta(alpha, alpha))
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 to preserve the primary sample

        B = features.size(0)
        idx = torch.randperm(B, device=features.device)

        mixed_features = lam * features + (1 - lam) * features[idx]
        return mixed_features, (lam, labels, labels[idx])


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro F1, averaged over all genes.
    y_pred: [n_samples, 3, n_genes]
    y_true_remapped: [n_samples, n_genes] (labels in {0,1,2})
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
        lora_dropout: float = 0.10,
        lora_layers_start: int = 0,
        attn_n_layers: int = 3,
        attn_nhead: int = 8,
        attn_dim_ff: int = 256,
        attn_dropout: float = 0.2,
        head_dropout: float = 0.5,
        mixup_alpha: float = 0.3,
        lr: float = 2e-4,
        head_lr_multiplier: float = 3.0,
        sym_proj_lr_multiplier: float = 3.0,
        ppi_proj_lr_multiplier: float = 5.0,
        fusion_lr_multiplier: float = 3.0,
        weight_decay: float = 0.10,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 120,
        plateau_patience: int = 12,
        plateau_factor: float = 0.5,
        plateau_min_lr: float = 1e-7,
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
        # For checkpoint averaging
        self._ckpt_preds_list: List[np.ndarray] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = CrossAttentionDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                lora_layers_start=self.hparams.lora_layers_start,
                attn_n_layers=self.hparams.attn_n_layers,
                attn_nhead=self.hparams.attn_nhead,
                attn_dim_ff=self.hparams.attn_dim_ff,
                attn_dropout=self.hparams.attn_dropout,
                head_dropout=self.hparams.head_dropout,
                mixup_alpha=self.hparams.mixup_alpha,
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
        """Compute loss — handles both standard labels and mixup (lam, labels_a, labels_b) tuple."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        if isinstance(labels, tuple):
            lam, labels_a, labels_b = labels
            labels_a_flat = labels_a.reshape(-1)
            labels_b_flat = labels_b.reshape(-1)
            loss = lam * self.criterion(logits_flat, labels_a_flat) + \
                   (1 - lam) * self.criterion(logits_flat, labels_b_flat)
        else:
            labels_flat = labels.reshape(-1)
            loss = self.criterion(logits_flat, labels_flat)
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Forward with manifold mixup
        result = self.model(
            batch["expr"], batch["gene_pos"], batch["symbol_ids"], batch["ppi_emb"],
            labels=batch["label"], mixup_alpha=self.hparams.mixup_alpha,
        )
        logits, mixed_labels = result
        loss = self._compute_loss(logits, mixed_labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        # Validation: no mixup
        logits = self.model(batch["expr"], batch["gene_pos"], batch["symbol_ids"], batch["ppi_emb"])
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

        f1_tensor = torch.tensor(f1, dtype=torch.float32, device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(f1_tensor, op=torch.distributed.ReduceOp.SUM)
            f1_tensor = f1_tensor / self.trainer.world_size

        self.log("val_f1", f1_tensor.item(), prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["expr"], batch["gene_pos"], batch["symbol_ids"], batch["ppi_emb"])
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
            # Store for checkpoint averaging
            self._ckpt_preds_list.append(preds)
            self._last_idxs = idxs
            # Always save predictions from the best checkpoint test run
            # (will be overwritten by checkpoint averaging if it succeeds)
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
        # FIVE separate parameter groups:
        # 1. backbone LoRA (lowest lr)
        # 2. symbol_encoder (3× backbone)
        # 3. ppi_proj (5× backbone - needs to learn STRING→AIDO alignment)
        # 4. fusion modules (global_proj, pert_proj, sym_proj, token_pos_emb, fusion_transformer) (3× backbone)
        # 5. head (3× backbone)
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        symbol_params = list(self.model.symbol_encoder.parameters())
        ppi_proj_params = list(self.model.ppi_proj.parameters())
        fusion_params = (
            list(self.model.global_proj.parameters()) +
            list(self.model.pert_proj.parameters()) +
            list(self.model.sym_proj.parameters()) +
            list(self.model.token_pos_emb.parameters()) +
            list(self.model.fusion_transformer.parameters())
        )
        head_params = list(self.model.head.parameters())

        backbone_lr = hp.lr                                          # 2e-4
        symbol_lr = hp.lr * hp.sym_proj_lr_multiplier               # 6e-4 (3×)
        ppi_proj_lr = hp.lr * hp.ppi_proj_lr_multiplier             # 1e-3 (5×)
        fusion_lr = hp.lr * hp.fusion_lr_multiplier                 # 6e-4 (3×)
        head_lr = hp.lr * hp.head_lr_multiplier                     # 6e-4 (3×)

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": symbol_params, "lr": symbol_lr},
                {"params": ppi_proj_params, "lr": ppi_proj_lr},
                {"params": fusion_params, "lr": fusion_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=hp.weight_decay,
        )

        # ReduceLROnPlateau monitors val_f1 (NOT val_loss):
        # Critical insight from node3-2-2: focal loss makes val_loss inversely correlated
        # with val_f1 during learning phase → premature LR reduction when monitoring val_loss.
        # val_f1 is the ONLY reliable scheduling signal for this task.
        # patience=12 prevents premature reduction (tree-best node3-1-3-1-1 had LR never fire).
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",                    # maximize val_f1
            factor=hp.plateau_factor,      # 0.5: halve LR each reduction
            patience=hp.plateau_patience,  # 12 epochs before reducing
            min_lr=hp.plateau_min_lr,      # floor
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_f1",            # KEY: monitor val_f1, not val_loss
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
# Checkpoint averaging helper
# ──────────────────────────────────────────────────────────────────────────────
def run_checkpoint_averaging(
    checkpoint_cb: ModelCheckpoint,
    datamodule: DEGDataModule,
    output_dir: Path,
    fast_dev_run: bool,
    debug_max_step: Optional[int],
) -> None:
    """
    Load top-K best checkpoints, run test on each, average logits, save final predictions.
    This is run on single GPU (rank 0) after the main training to avoid DDP complications.
    """
    top_ckpts = checkpoint_cb.best_k_models
    if not top_ckpts or fast_dev_run or debug_max_step is not None:
        return

    # Sort by score (descending for val_f1)
    sorted_ckpts = sorted(top_ckpts.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_ckpts) < 2:
        return  # Single checkpoint — no averaging benefit

    print(f"\nRunning top-{len(sorted_ckpts)} checkpoint averaging...")

    accumulated_preds = None
    n_averaged = 0
    test_pert_ids = None
    test_symbols = None

    # Ensure test dataset is ready
    if datamodule.test_ds is None:
        datamodule.setup(stage="test")
    test_pert_ids = datamodule.test_pert_ids
    test_symbols = datamodule.test_symbols
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ckpt_path, score in sorted_ckpts:
        print(f"  Loading checkpoint: {ckpt_path} (val_f1={score:.4f})")
        try:
            # Load checkpoint data
            ckpt_data = torch.load(ckpt_path, map_location="cpu")
            hparams = ckpt_data.get("hyper_parameters", {})

            # Directly instantiate the nn.Module (avoids DataModule setup overhead)
            nn_model = CrossAttentionDEGModel(
                lora_r=hparams.get("lora_r", 4),
                lora_alpha=hparams.get("lora_alpha", 8),
                lora_dropout=hparams.get("lora_dropout", 0.10),
                lora_layers_start=hparams.get("lora_layers_start", 0),
                attn_n_layers=hparams.get("attn_n_layers", 3),
                attn_nhead=hparams.get("attn_nhead", 8),
                attn_dim_ff=hparams.get("attn_dim_ff", 256),
                attn_dropout=hparams.get("attn_dropout", 0.2),
                head_dropout=hparams.get("head_dropout", 0.5),
                mixup_alpha=0.0,  # No mixup during inference
            )

            # Load saved trainable parameters
            state = ckpt_data.get("state_dict", {})
            # Strip the "model." prefix from keys (LightningModule wraps nn.Module)
            nn_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
            nn_model.load_state_dict(nn_state, strict=False)
            nn_model.eval()
            nn_model = nn_model.to(device)

            all_preds = []
            all_idxs = []
            test_loader = datamodule.test_dataloader()

            with torch.no_grad():
                for batch in test_loader:
                    batch_device = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    logits = nn_model(
                        batch_device["expr"],
                        batch_device["gene_pos"],
                        batch_device["symbol_ids"],
                        batch_device["ppi_emb"],
                    )
                    probs = F.softmax(logits.float(), dim=1).cpu().numpy()
                    all_preds.append(probs)
                    all_idxs.append(batch["idx"].numpy())

            preds = np.concatenate(all_preds, axis=0)  # [N, 3, 6640]
            idxs = np.concatenate(all_idxs, axis=0)

            # Deduplicate and sort
            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]
            idxs = idxs[uniq]
            order = np.argsort(idxs)
            preds = preds[order]
            idxs = idxs[order]

            if accumulated_preds is None:
                accumulated_preds = preds.copy()
            else:
                accumulated_preds += preds
            n_averaged += 1

            # Clean up
            del nn_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Warning: Failed to load checkpoint {ckpt_path}: {e}")
            continue

    if accumulated_preds is not None and n_averaged > 1:
        averaged_preds = accumulated_preds / n_averaged
        print(f"  Averaged {n_averaged} checkpoints.")

        # Save averaged predictions
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for rank_i, orig_i in enumerate(idxs):
            rows.append({
                "idx": test_pert_ids[orig_i],
                "input": test_symbols[orig_i],
                "prediction": json.dumps(averaged_preds[rank_i].tolist()),
            })
        out_path = output_dir / "test_predictions.tsv"
        pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
        print(f"  Checkpoint-averaged test predictions saved → {out_path}")
    else:
        print("  Checkpoint averaging skipped (insufficient valid checkpoints).")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 2-2-1-2-1: Cross-Attention Fusion DEG Predictor"
    )
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--global_batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--head_lr_multiplier", type=float, default=3.0)
    p.add_argument("--sym_proj_lr_multiplier", type=float, default=3.0)
    p.add_argument("--ppi_proj_lr_multiplier", type=float, default=5.0)
    p.add_argument("--fusion_lr_multiplier", type=float, default=3.0)
    p.add_argument("--weight_decay", type=float, default=0.10)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.10)
    p.add_argument("--lora_layers_start", type=int, default=0)  # 0 = all 8 layers
    p.add_argument("--attn_n_layers", type=int, default=3)
    p.add_argument("--attn_nhead", type=int, default=8)
    p.add_argument("--attn_dim_ff", type=int, default=256)
    p.add_argument("--attn_dropout", type=float, default=0.2)
    p.add_argument("--head_dropout", type=float, default=0.5)
    p.add_argument("--mixup_alpha", type=float, default=0.3)
    p.add_argument("--gamma_focal", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--plateau_patience", type=int, default=12)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--plateau_min_lr", type=float, default=1e-7)
    p.add_argument("--early_stopping_patience", type=int, default=35)
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
        filename="node2-2-1-2-1-{epoch:03d}-{val_f1:.4f}",
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
    )
    model_module = DEGLightningModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers_start=args.lora_layers_start,
        attn_n_layers=args.attn_n_layers,
        attn_nhead=args.attn_nhead,
        attn_dim_ff=args.attn_dim_ff,
        attn_dropout=args.attn_dropout,
        head_dropout=args.head_dropout,
        mixup_alpha=args.mixup_alpha,
        lr=args.lr,
        head_lr_multiplier=args.head_lr_multiplier,
        sym_proj_lr_multiplier=args.sym_proj_lr_multiplier,
        ppi_proj_lr_multiplier=args.ppi_proj_lr_multiplier,
        fusion_lr_multiplier=args.fusion_lr_multiplier,
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

    is_global_zero = trainer.is_global_zero

    if is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"test_results: {test_results}\n"
            f"val_f1_best: {checkpoint_cb.best_model_score}\n"
        )
        print(f"Test score saved → {score_path}")

    # Top-K checkpoint averaging (rank 0 only, after DDP training)
    # Free GPU memory first to allow loading multiple checkpoint copies
    if is_global_zero and not args.fast_dev_run and args.debug_max_step is None:
        del trainer, model_module
        torch.cuda.empty_cache()
        run_checkpoint_averaging(
            checkpoint_cb=checkpoint_cb,
            datamodule=datamodule,
            output_dir=output_dir,
            fast_dev_run=bool(args.fast_dev_run),
            debug_max_step=args.debug_max_step,
        )


if __name__ == "__main__":
    main()
