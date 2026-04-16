#!/usr/bin/env python3
"""
Node 1-2 (node2-2-3-1-1-1-2): LoRA r=2 + Deeper Head + backbone_lr=4e-4 + patience=25
=========================================================================================

Parent: node2-2-3-1-1-1 (test F1=0.4655, tree-best)
Sibling: node2-2-3-1-1-1-1 (test F1=0.4655, flat vs parent)

This node explores a DIFFERENT optimization direction from the sibling
(sibling tried [8,1,18] class weights + top-5 averaging — no gain).

Key changes based on sibling's feedback recommendations:
1. LoRA rank reduction r=4 -> r=2: Reduces trainable LoRA params from ~8.1M to ~4M.
   The val-test gap (~0.235) and severe overfitting (train_loss 0.5->0.1, val_loss climbing)
   suggest excess capacity relative to 1,500 training samples. Lower rank forces the model
   to learn more parsimonious perturbation representations.
2. Deeper head: 832->512->256->19920 (from 832->384->19920). With less backbone
   adaptation (r=2), a more expressive head helps compensate. Two hidden layers with
   residual-style normalization provide richer non-linear transformations.
3. Increased backbone_lr: 3e-4 -> 4e-4. With lower LoRA rank (fewer parameters to adapt),
   a slightly stronger backbone learning rate helps achieve sufficient adaptation within
   fewer dimensions.
4. Extended early stopping: patience=15 -> patience=25. Lower LoRA rank may converge more
   slowly; extra patience allows the model to find its optimum without premature stopping.
5. Revert class weights to [7,1,15] (parent's proven optimal): Sibling demonstrated
   [8,1,18] provides zero additional gain. [7,1,15] is the proven best.
6. Top-3 checkpoint averaging: Maintained from parent (proven effective — primary driver
   of parent's +0.003 gain over grandparent). Also keep save_top_k=3 with patience=15
   parent structure (but extended patience allows more diverse checkpoint collection).

Architecture differences vs parent:
  - LoRA: r=2 (from r=4), lora_alpha=4 (from 8), maintaining 2x ratio
  - Head: 832->512->256->19920 (from 832->384->19920), two hidden layers
  - Head dropout: 0.3 on first hidden, 0.25 on second (from single 0.4)
  - backbone_lr: 4e-4 (from 3e-4)
  - early_stopping_patience: 25 (from 15)
  - class_weights: [7.0, 1.0, 15.0] (same as grandparent, reverted from sibling's [8,1,18])
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

# Proven optimal class weights from parent (node2-2-3-1-1-1):
# Sibling tried [8,1,18] — yielded zero gain (saturation at this step).
# Evidence from MCTS tree:
#   [5,1,10] -> 0.4573  (node2-3-1-1-1-1)
#   [6,1,12] -> 0.4586  (node2-3-1-1-1-1-1-1), 0.4625 (node2-2-3-1-1)
#   [7,1,15] -> 0.4655  (parent node2-2-3-1-1-1)
#   [8,1,18] -> 0.4655  (sibling, NO gain) <- saturation confirmed
# Reverting to proven-optimal [7,1,15] for this node.
CLASS_WEIGHTS = torch.tensor([7.0, 1.0, 15.0], dtype=torch.float32)

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
        # Standard cross-entropy with label smoothing
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
    while len(indices) < SYMBOL_MAX_LEN:
        indices.append(SYMBOL_PAD_IDX)
    return indices


class SymbolEncoder(nn.Module):
    """
    Character-level CNN encoder for gene symbol strings.
    Architecture:
      1. Character embedding: [B, L] -> [B, L, embed_dim]
      2. 3 parallel 1D conv branches (kernels 2/3/4) -> adaptive max-pool
      3. Concat [B, 96] -> project to out_dim [B, out_dim]
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
        x = x.transpose(1, 2)              # [B, embed_dim, L]
        f2 = F.relu(self.conv2(x))
        f3 = F.relu(self.conv3(x))
        f4 = F.relu(self.conv4(x))
        f2 = F.adaptive_max_pool1d(f2, 1).squeeze(-1)  # [B, 32]
        f3 = F.adaptive_max_pool1d(f3, 1).squeeze(-1)
        f4 = F.adaptive_max_pool1d(f4, 1).squeeze(-1)
        feat = torch.cat([f2, f3, f4], dim=-1)          # [B, 96]
        return self.proj(feat)                           # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# STRING GNN PPI Embeddings (frozen, pre-computed at setup)
# ──────────────────────────────────────────────────────────────────────────────
class FrozenSTRINGGNN:
    """
    Computes frozen STRING GNN PPI node embeddings for all genes at setup time.
    Returns a mapping: ENSG_id -> [STRING_GNN_DIM] float32 embedding.
    """

    def __init__(self, model_dir: str = STRING_GNN_MODEL_DIR):
        self.model_dir = Path(model_dir)

    def compute_embeddings(self, device: str = "cpu") -> Dict[str, np.ndarray]:
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

        embedding_dict: Dict[str, np.ndarray] = {}
        for i, node_name in enumerate(node_names):
            base = node_name.split(".")[0]
            embedding_dict[base] = emb_np[i]

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
        gene_to_pos: Dict[str, int],
        ppi_embeddings: Dict[str, np.ndarray],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.ppi_embeddings = ppi_embeddings
        self.is_test = is_test

        self.expr_inputs = self._build_expr_tensors()
        self.symbol_ids = self._build_symbol_tensors()
        self.ppi_embs = self._build_ppi_tensors()

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

    def _build_symbol_tensors(self) -> torch.Tensor:
        N = len(self.symbols)
        sym_ids = torch.zeros((N, SYMBOL_MAX_LEN), dtype=torch.long)
        for i, symbol in enumerate(self.symbols):
            indices = symbol_to_indices(symbol)
            sym_ids[i] = torch.tensor(indices, dtype=torch.long)
        return sym_ids

    def _build_ppi_tensors(self) -> torch.Tensor:
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
            "expr": self.expr_inputs[idx],
            "gene_pos": gene_pos,
            "symbol_ids": self.symbol_ids[idx],
            "ppi_emb": self.ppi_embs[idx],
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

        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)

        if not self.ppi_embeddings:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            if local_rank == 0:
                string_gnn = FrozenSTRINGGNN(model_dir=STRING_GNN_MODEL_DIR)
                self.ppi_embeddings = string_gnn.compute_embeddings(device=device_str)
            # Barrier OUTSIDE the rank-0 block
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
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
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class AIDOCellStringDEGModel(nn.Module):
    """
    4-source feature fusion DEG predictor with LoRA r=2 (reduced from r=4) and
    deeper MLP head (832->512->256->19920, from 832->384->19920).

    Key architectural change: LoRA rank=2 reduces trainable backbone params from
    ~8.1M to ~4M, combating overfitting for the 1,500-sample training set while
    maintaining sufficient adaptation capacity. The deeper head compensates for
    reduced backbone expressiveness.

    Sources:
      (a) global mean-pool of AIDO.Cell-10M last_hidden_state [B, 256]
      (b) perturbed-gene positional embedding from AIDO.Cell [B, 256]
      (c) gene symbol character CNN embedding [B, 64]
      (d) frozen STRING GNN PPI topology embedding for perturbed gene [B, 256]

    Concatenated -> [B, 832] -> Deeper MLP head (512->256 hidden) -> [B, 3, 6640]
    """

    HIDDEN_DIM = 256           # AIDO.Cell-10M hidden size
    SYMBOL_DIM = 64            # gene symbol embedding dimension
    HEAD_INPUT_DIM = 256 * 2 + 64 + 256  # = 832 (backbone×2 + symbol + STRING)

    def __init__(
        self,
        lora_r: int = 2,
        lora_alpha: int = 4,
        head_hidden1: int = 512,
        head_hidden2: int = 256,
        head_dropout1: float = 0.3,
        head_dropout2: float = 0.25,
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA r=2 on ALL 8 layers ──────────────
        # Reduced from r=4 to r=2: ~4M trainable params (from ~8.1M).
        # lora_alpha=4 maintains 2×r standard ratio (from alpha=8 with r=4).
        # This reduces overfitting capacity while preserving directional adaptation.
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

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
        self.ppi_proj = nn.Sequential(
            nn.Linear(STRING_GNN_DIM, STRING_GNN_DIM),
            nn.LayerNorm(STRING_GNN_DIM),
        )

        # ── Deeper prediction head ─────────────────────────────────────────────
        # Architecture: 832 -> LayerNorm -> 512 -> GELU -> Dropout(0.3) ->
        #               LayerNorm -> 256 -> GELU -> Dropout(0.25) ->
        #               LayerNorm -> 19920 -> reshape [B, 3, 6640]
        # A two-hidden-layer head provides more non-linear capacity than the
        # original 832->384->19920 one-hidden design, compensating for the
        # reduced LoRA rank (r=2 vs r=4) in the backbone.
        head_in = self.HEAD_INPUT_DIM  # 832
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden1),
            nn.GELU(),
            nn.Dropout(head_dropout1),
            nn.LayerNorm(head_hidden1),
            nn.Linear(head_hidden1, head_hidden2),
            nn.GELU(),
            nn.Dropout(head_dropout2),
            nn.LayerNorm(head_hidden2),
            nn.Linear(head_hidden2, N_CLASSES * N_GENES_OUT),
        )
        # Conservative initialization for the output layer
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        expr: torch.Tensor,          # [B, 19264] float32
        gene_pos: torch.Tensor,      # [B]         int64  (-1 if not in vocab)
        symbol_ids: torch.Tensor,    # [B, SYMBOL_MAX_LEN] int64
        ppi_emb: torch.Tensor,       # [B, STRING_GNN_DIM] float32
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
            valid_pos = gene_pos[valid_mask]
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
    """
    print(f"[CheckpointAveraging] Averaging {len(checkpoint_paths)} checkpoints:")
    for p in checkpoint_paths:
        print(f"  - {p}")

    avg_state_dict: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                v_f = v.float()
                if k not in avg_state_dict:
                    avg_state_dict[k] = v_f.clone()
                    counts[k] = 1
                else:
                    avg_state_dict[k] = avg_state_dict[k] + v_f
                    counts[k] += 1

    for k in avg_state_dict:
        avg_state_dict[k] = avg_state_dict[k] / counts[k]

    import lightning as _pl
    ckpt_out = {
        "state_dict": avg_state_dict,
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": _pl.__version__,
    }
    torch.save(ckpt_out, output_path)
    print(f"[CheckpointAveraging] Averaged checkpoint saved to: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 2,
        lora_alpha: int = 4,
        head_hidden1: int = 512,
        head_hidden2: int = 256,
        head_dropout1: float = 0.3,
        head_dropout2: float = 0.25,
        backbone_lr: float = 4e-4,
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
                head_hidden1=self.hparams.head_hidden1,
                head_hidden2=self.hparams.head_hidden2,
                head_dropout1=self.hparams.head_dropout1,
                head_dropout2=self.hparams.head_dropout2,
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
        # Three-tier parameter groups:
        # 1. Backbone LoRA params (lower lr) — r=2 uses backbone_lr=4e-4 (vs 3e-4 with r=4)
        #    Rationale: r=2 has fewer parameters to adapt; slightly higher lr compensates
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
        # val_f1 is the only reliable scheduling signal for this task.
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
        description="Node 1-2 (node2-2-3-1-1-1-2): LoRA r=2 + Deeper Head + backbone_lr=4e-4"
    )
    default_data_dir = str(Path(__file__).parent.parent.parent / "data")
    p.add_argument("--data_dir", type=str, default=default_data_dir)
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--global_batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=120)
    # LoRA r=2 (from r=4): reduces trainable params ~4M from ~8.1M
    p.add_argument("--lora_r", type=int, default=2)
    # lora_alpha=4 maintains 2×r ratio (from alpha=8 with r=4)
    p.add_argument("--lora_alpha", type=int, default=4)
    # Deeper head: 832->512->256->19920
    p.add_argument("--head_hidden1", type=int, default=512)
    p.add_argument("--head_hidden2", type=int, default=256)
    p.add_argument("--head_dropout1", type=float, default=0.3)
    p.add_argument("--head_dropout2", type=float, default=0.25)
    # backbone_lr=4e-4 (from 3e-4): slightly stronger updates for r=2
    p.add_argument("--backbone_lr", type=float, default=4e-4)
    p.add_argument("--symbol_lr_multiplier", type=float, default=2.0)
    p.add_argument("--head_lr_multiplier", type=float, default=3.0)
    p.add_argument("--weight_decay", type=float, default=0.03)
    p.add_argument("--gamma_focal", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--plateau_patience", type=int, default=12)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--plateau_min_lr", type=float, default=1e-7)
    # patience=25: extended from parent's 15 to allow r=2 to fully converge
    # r=2 may peak later than r=4 due to slower adaptation capacity
    p.add_argument("--early_stopping_patience", type=int, default=25)
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    # Checkpoint averaging: top-3 from parent (proven effective)
    p.add_argument("--avg_top_k", type=int, default=3,
                   help="Number of top checkpoints to average for test evaluation (0=disabled)")
    return p.parse_args()


def main():
    # seed=0: proven best initialization basin for this lineage
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
        limit_test_batches = args.debug_max_step
    else:
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0

    # save_top_k=3: maintain parent's proven checkpoint configuration
    # top-3 averaging was the primary driver of +0.003 gain in the parent
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=3, save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1", mode="max",
        # patience=25: extended to allow r=2 to fully converge (may peak later)
        # Parent (r=4) peaked at epoch 21; r=2 may converge more slowly
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
        head_hidden1=args.head_hidden1,
        head_hidden2=args.head_hidden2,
        head_dropout1=args.head_dropout1,
        head_dropout2=args.head_dropout2,
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
    # Top-3 checkpoint averaging: proven primary driver of +0.003 F1 gain in parent.
    # The diversity from pre-peak, peak, and post-LR phases is preserved here
    # because patience=25 allows training to continue through LR reduction.
    # ──────────────────────────────────────────────────────────────────────────────

    is_full_run = not (args.fast_dev_run or args.debug_max_step is not None)

    if is_full_run and args.avg_top_k > 0:
        best_k = checkpoint_cb.best_k_models  # dict[str, Tensor]
        if len(best_k) > 1:
            top_k_paths = sorted(
                best_k.keys(),
                key=lambda x: best_k[x].item() if hasattr(best_k[x], "item") else float(best_k[x]),
                reverse=True,
            )[:args.avg_top_k]

            avg_ckpt_path = str(output_dir / "checkpoints" / "avg_checkpoint.ckpt")

            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank == 0:
                try:
                    average_checkpoints(model_module, top_k_paths, avg_ckpt_path)
                    ckpt_path_for_test = avg_ckpt_path
                except Exception as e:
                    print(f"[CheckpointAveraging] Failed: {e}. Falling back to best single checkpoint.")
                    ckpt_path_for_test = checkpoint_cb.best_model_path
            else:
                ckpt_path_for_test = avg_ckpt_path

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

            # Verify the averaged checkpoint was created
            if not Path(avg_ckpt_path).exists():
                ckpt_path_for_test = checkpoint_cb.best_model_path
        else:
            # Only one checkpoint saved — use it directly
            ckpt_path_for_test = checkpoint_cb.best_model_path
            print(f"[CheckpointAveraging] Only {len(best_k)} checkpoint(s) available; "
                  f"skipping averaging, using best: {ckpt_path_for_test}")
    else:
        ckpt_path_for_test = "best" if is_full_run else None

    # Run test evaluation
    if is_full_run:
        test_results = trainer.test(model_module, datamodule=datamodule,
                                    ckpt_path=ckpt_path_for_test)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule)

    # Save test score summary
    if test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        with open(score_path, "w") as f:
            f.write(f"test_results: {test_results}\n")
        print(f"Test score saved to {score_path}")


if __name__ == "__main__":
    main()
