#!/usr/bin/env python3
"""
Node 1-2 (node2-2-1-3-4): AIDO.Cell-10M + LoRA r=4 ALL 8 layers + Symbol CNN + Frozen STRING GNN
+ val_f1 monitoring + strengthened class weights [7,1,15] + reduced head capacity + top-3 checkpoint averaging
========================================================================================
Improves on parent node (node2-2-1-3: test F1=0.4495) via multiple targeted changes:

  1. Revert ReduceLROnPlateau to monitor val_f1 (mode=max) — CRITICAL FIX
     - Parent's val_loss monitoring triggered premature LR reduction at epoch 12
       (val_f1 dropped from 0.445 to 0.433 after the reduction — clear evidence of
       premature firing)
     - val_f1 monitoring correctly avoids reduction while the model is still improving
     - plateau_patience=10 (up from 8) to avoid premature triggers with val_f1 oscillation

  2. Strengthened class weights [7.0, 1.0, 15.0] (from parent's [5.0, 1.0, 10.0])
     - Tree-best nodes consistently use [6,1,12] to [7,1,15] class weights
     - node2-2-3-1-1-1 (F1=0.4655, tree-best) used [7,1,15] with top-3 averaging
     - node2-2-3-1-2 (F1=0.4638) used [7,1,15] with improvement over [6,1,12] sibling
     - Monotonic increase [5,1,10]→[6,1,12]→[7,1,15] consistently shown to improve F1

  3. Reduce head_hidden from 320 to 256 — reduce head-dominated overfitting
     - Parent feedback: "head (6.64M params) is disproportionately large relative to 1,500
       training samples (~4.4K params per sample)"
     - Reducing from 320→256 cuts head params from ~6.64M to ~5.3M (saves ~1.3M params)
     - Still provides sufficient head capacity as proven by node3-2 (384-dim, 0.4622)
     - 256-dim is a proven sweet spot across many successful nodes

  4. Increase head_dropout 0.4→0.5 — additional head regularization
     - Parent's 8× train/val loss ratio indicates severe overfitting
     - Increasing head dropout from 0.4 to 0.5 provides direct regularization on the
       largest trainable component (the prediction head)
     - Proven effective by multiple tree nodes with 0.45+ F1

  5. Increase lora_dropout 0.1→0.2 — LoRA adapter regularization
     - Parent reduced this from 0.3 (parent's parent) to 0.1, likely contributing
       to increased overfitting
     - 0.2 is a balanced value between the parent's 0.1 (too low) and grandparent's 0.3
     - All-8-layer LoRA benefits from moderate dropout to prevent adapter overfitting

  6. Increase weight_decay 0.04→0.06 — stronger L2 regularization
     - Parent feedback explicitly recommends "increase weight_decay to 0.05–0.08"
     - 0.06 is a balanced target in the recommended range
     - Addresses the severe 8× overfitting ratio observed in the parent

  7. Top-3 checkpoint averaging for test inference
     - Tree-best nodes consistently benefit from this (+0.003–0.005 F1)
     - node2-2-3-1-1-1 (F1=0.4655) used top-3 averaging as primary gain driver
     - node2-2-3-2-1 (+0.0053 over parent) attributed top-3 averaging as primary driver
     - Simple variance reduction that doesn't require re-training

  8. Reduce early stopping patience 35→20
     - Parent peaked at epoch 21 and wasted 35 more epochs oscillating
     - patience=20 saves compute while still allowing sufficient exploration
     - Prevents unnecessary oscillation from degrading checkpoint selection

  9. plateau_factor: 0.5 → 0.5 (preserved)
     - Stable LR reduction factor proven in parent lineage

  10. head_lr_multiplier: 3.0 (preserved)
      - Stable head LR ratio from parent

Distinct from any sibling nodes (none exist yet for this node in the tree).

Architecture: AIDO.Cell-10M + LoRA r=4 all 8 layers + Symbol CNN + STRING GNN PPI
Fusion: AIDO dual-pool (512) + symbol CNN (64) + STRING PPI (256) = 832-dim → 256-dim head → 19920

Key differences from parent (node2-2-1-3):
  - Monitor: val_loss → val_f1 (PRIMARY FIX)
  - Class weights: [5.0, 1.0, 10.0] → [7.0, 1.0, 15.0]
  - head_hidden: 320 → 256
  - head_dropout: 0.4 → 0.5
  - lora_dropout: 0.1 → 0.2
  - weight_decay: 0.04 → 0.06
  - plateau_patience: 8 → 10 (to avoid premature reduction with val_f1 oscillation)
  - early_stopping_patience: 35 → 20
  - Top-3 checkpoint averaging for test inference (NEW)
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
PPI_DIM = 256            # STRING GNN output dimension

# Class weights for focal loss: strengthened to [7.0, 1.0, 15.0]
# Improvement from parent's [5.0, 1.0, 10.0]:
# - train distribution ~3.4% down, ~95.5% unchanged, ~1.1% up
# - node2-2-3-1-1-1 (tree-best 0.4655) and node2-2-3-1-2 (0.4638) both use [7,1,15]
# - Monotonic improvement: [5,1,10]→[6,1,12]→[7,1,15] consistently observed
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
        ce = F.cross_entropy(logits, targets, weight=w, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Gene Symbol Encoder (Character-level CNN) — unchanged from parent
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
    Three parallel Conv1d branches → global max-pool → 64-dim output.
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
        f2 = F.adaptive_max_pool1d(f2, 1).squeeze(-1)
        f3 = F.adaptive_max_pool1d(f3, 1).squeeze(-1)
        f4 = F.adaptive_max_pool1d(f4, 1).squeeze(-1)
        feat = torch.cat([f2, f3, f4], dim=-1)  # [B, 96]
        return self.proj(feat)                   # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# STRING GNN embedding loading helper
# ──────────────────────────────────────────────────────────────────────────────
def load_string_gnn_embeddings() -> Dict[str, torch.Tensor]:
    """
    Load frozen STRING GNN, run one forward pass, return ENSG→embedding dict.
    All computations on CPU. Result is a dict: ENSG_id (str) → [256] float32 tensor.
    """
    gnn_model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
    gnn_model.eval()

    graph = torch.load(str(STRING_GNN_DIR / "graph_data.pt"), map_location="cpu")
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())

    edge_index = graph["edge_index"]
    edge_weight = graph.get("edge_weight", None)

    with torch.no_grad():
        outputs = gnn_model(edge_index=edge_index, edge_weight=edge_weight)

    all_emb = outputs.last_hidden_state.float().cpu()  # [18870, 256]

    # Build ENSG → embedding lookup dict
    emb_dict: Dict[str, torch.Tensor] = {}
    for i, ensg_id in enumerate(node_names):
        emb_dict[ensg_id] = all_emb[i]  # [256]

    del gnn_model  # release memory
    return emb_dict


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Returns pre-built expression tensors, gene position index, gene symbol char indices,
    STRING GNN PPI embeddings, and the label.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],
        ppi_emb_dict: Dict[str, torch.Tensor],  # ENSG_base -> [PPI_DIM] embedding
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.ppi_emb_dict = ppi_emb_dict
        self.is_test = is_test

        # Pre-build tensors for efficient __getitem__
        self.expr_inputs = self._build_expr_tensors()    # [N, 19264] float32
        self.symbol_ids = self._build_symbol_tensors()   # [N, SYMBOL_MAX_LEN] int64
        self.ppi_embs = self._build_ppi_tensors()        # [N, PPI_DIM] float32

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
        """Pre-compute [N, PPI_DIM] float32 STRING GNN PPI embeddings.
        Genes not in STRING graph get zero vectors (coverage ~90%+ of human genes).
        """
        N = len(self.pert_ids)
        ppi = torch.zeros((N, PPI_DIM), dtype=torch.float32)
        for i, pert_id in enumerate(self.pert_ids):
            base = pert_id.split(".")[0]
            emb = self.ppi_emb_dict.get(base, None)
            if emb is not None:
                ppi[i] = emb
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
            "ppi_emb": self.ppi_embs[idx],        # [PPI_DIM] float32
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
        "ppi_emb": torch.stack([b["ppi_emb"] for b in batch]),        # [B, PPI_DIM]
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
        self.ppi_emb_dict: Dict[str, torch.Tensor] = {}
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        # ── Tokenizer: rank-0 downloads first, then barrier ──────────────────
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # ── Build ENSG→position mapping from all pert_ids ────────────────────
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)

        # ── Load frozen STRING GNN embeddings (each rank independently) ──────
        # Each rank runs ONE frozen forward pass (~5.43M params, CPU only, ~19MB output)
        if not self.ppi_emb_dict:
            self.ppi_emb_dict = load_string_gnn_embeddings()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self.gene_to_pos, self.ppi_emb_dict)
            self.val_ds = PerturbationDataset(val_df, self.gene_to_pos, self.ppi_emb_dict)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self.gene_to_pos, self.ppi_emb_dict, is_test=True
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
class AIDOCellSymbolPPIDEGModel(nn.Module):
    """
    AIDO.Cell-10M backbone + LoRA r=4 ALL 8 layers + character-level symbol CNN +
    frozen STRING GNN PPI embeddings + MLP prediction head.

    Input representations fused:
      (a) global mean-pool of AIDO.Cell-10M last_hidden_state [B, 256]
      (b) perturbed-gene positional embedding [B, 256]
      (c) gene symbol character CNN embedding [B, 64]
      (d) STRING GNN PPI embedding → ppi_proj [B, 256]
    Concatenated → [B, 832] → MLP head → [B, 3, 6640]

    Changes from parent:
    - head_hidden: 320 → 256 (reduces overfitting-prone head capacity)
    - head_dropout: 0.4 → 0.5 (stronger regularization)
    - lora_dropout: 0.1 → 0.2 (balanced LoRA regularization)
    """

    HIDDEN_DIM = 256          # AIDO.Cell-10M hidden size
    SYMBOL_DIM = 64           # gene symbol embedding dimension
    PPI_PROJ_DIM = 256        # STRING GNN projection output
    HEAD_INPUT_DIM = 256 * 2 + 64 + 256  # 832

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_layers_start: int = 0,  # 0 → all 8 layers
        lora_dropout: float = 0.2,   # increased from parent's 0.1 for better regularization
        head_hidden: int = 256,      # reduced from parent's 320 to cut overfitting head params
        head_dropout: float = 0.5,   # increased from parent's 0.4
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA on ALL 8 layers ─────────────────
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA r=4 on Q/K/V of ALL 8 layers (lora_layers_start=0 → indices 0..7)
        # Inspired by node3-2's decisive success: r=4 all-8 layers + STRING GNN → F1=0.4622
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,           # 8 = 2×r (standard scaling)
            lora_dropout=lora_dropout,        # 0.2 (balanced: parent's 0.1 caused overfitting)
            target_modules=["query", "key", "value"],
            layers_to_transform=list(range(lora_layers_start, AIDO_N_LAYERS)),  # [0..7]
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

        # ── STRING GNN PPI projection ─────────────────────────────────────────
        # Aligns frozen STRING GNN embeddings with the rest of the feature space
        # Near-identity initialization: LayerNorm → Linear(256→256) → GELU
        self.ppi_proj = nn.Sequential(
            nn.LayerNorm(self.PPI_PROJ_DIM),
            nn.Linear(self.PPI_PROJ_DIM, self.PPI_PROJ_DIM),
            nn.GELU(),
        )
        # Near-identity init: linear weight close to identity, bias = 0
        nn.init.eye_(self.ppi_proj[1].weight)
        nn.init.zeros_(self.ppi_proj[1].bias)

        # ── Prediction head ────────────────────────────────────────────────────
        # 832-dim input: 512 (backbone) + 64 (symbol) + 256 (PPI) → 256 → 19920
        # Changed: 320 → 256 to reduce head-dominated overfitting
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
        ppi_emb: torch.Tensor,       # [B, 256]    float32 (STRING GNN embedding)
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
        # ppi_emb is float32 already; move to device if needed
        ppi_feat = self.ppi_proj(ppi_emb.to(backbone_feat.device))  # [B, 256] float32

        # Concatenate all four features → [B, 832]
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
# Top-K Checkpoint Averaging Helper
# ──────────────────────────────────────────────────────────────────────────────
def average_top_k_checkpoints(
    checkpoint_paths: List[str],
    model: "DEGLightningModule",
) -> None:
    """
    Average weights of top-K checkpoints and load into model.
    Uses explicit float32 casting for bfloat16 LoRA checkpoint compatibility.

    Top-3 averaging provides proven +0.003-0.005 F1 gains across the MCTS tree
    (node2-2-3-1-1-1 at 0.4655, node2-2-3-2-1 at 0.4650 both used this technique).
    """
    if not checkpoint_paths:
        return

    valid_paths = [p for p in checkpoint_paths if Path(p).exists()]
    if not valid_paths:
        print(f"Warning: No valid checkpoint paths found for averaging. Skipping.")
        return

    if len(valid_paths) < len(checkpoint_paths):
        print(f"Warning: Only {len(valid_paths)}/{len(checkpoint_paths)} checkpoints found.")

    print(f"Averaging {len(valid_paths)} checkpoints for test inference...")

    # Load all state dicts, casting to float32 for numerical stability
    # Critical: must cast bfloat16 tensors to float32 before averaging
    # to avoid dtype incompatibility errors (proven bug in some tree nodes)
    all_state_dicts = []
    for ckpt_path in valid_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # Cast all tensors to float32 for safe averaging
        state_f32 = {k: v.float() if torch.is_tensor(v) else v for k, v in state.items()}
        all_state_dicts.append(state_f32)

    # Average state dicts
    avg_state = {}
    for key in all_state_dicts[0].keys():
        tensors = [sd[key] for sd in all_state_dicts if key in sd]
        if tensors and torch.is_tensor(tensors[0]) and tensors[0].is_floating_point():
            # Average floating point tensors; ensure same device (all on CPU here)
            avg_state[key] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            avg_state[key] = all_state_dicts[0][key]

    model.load_state_dict(avg_state, strict=False)
    print(f"Successfully averaged {len(valid_paths)} checkpoints.")


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_layers_start: int = 0,
        lora_dropout: float = 0.2,      # increased from parent's 0.1
        head_hidden: int = 256,          # reduced from parent's 320
        head_dropout: float = 0.5,       # increased from parent's 0.4
        lr: float = 3e-4,
        head_lr_multiplier: float = 3.0,
        symbol_encoder_lr_multiplier: float = 2.0,
        ppi_proj_lr_multiplier: float = 2.0,
        weight_decay: float = 0.06,      # increased from parent's 0.04
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 120,
        plateau_patience: int = 10,      # increased from parent's 8 for val_f1 monitoring
        plateau_factor: float = 0.5,
        plateau_min_lr: float = 1e-7,
        top_k_averaging: int = 3,        # number of checkpoints to average for test
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCellSymbolPPIDEGModel] = None
        self.criterion: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        # Store checkpoint callback for top-k averaging
        self._checkpoint_callback: Optional[ModelCheckpoint] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = AIDOCellSymbolPPIDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_layers_start=self.hparams.lora_layers_start,
                lora_dropout=self.hparams.lora_dropout,
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
            batch["expr"],
            batch["gene_pos"],
            batch["symbol_ids"],
            batch["ppi_emb"],
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

        # Each rank independently computes val_f1 on the full gathered dataset
        preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
        labels = al.cpu().view(-1, N_GENES_OUT).numpy()
        idxs = ai.cpu().view(-1).numpy()
        _, uniq = np.unique(idxs, return_index=True)
        f1 = compute_deg_f1(preds[uniq], labels[uniq])

        # All-reduce val_f1 across ranks for consistency
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
            for orig_i, pred_row in zip(idxs, preds):
                rows.append({
                    "idx": self._test_pert_ids[orig_i],
                    "input": self._test_symbols[orig_i],
                    "prediction": json.dumps(pred_row.tolist()),
                })
            pd.DataFrame(rows).to_csv(output_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"Test predictions saved → {output_dir / 'test_predictions.tsv'}")

    def configure_optimizers(self):
        hp = self.hparams
        # FOUR separate parameter groups with different learning rates:
        #   1. backbone LoRA (lowest lr — pretrained, needs careful fine-tuning)
        #   2. symbol_encoder (medium lr — 2× backbone, randomly initialized 22K params)
        #   3. ppi_proj (medium lr — 2× backbone, newly initialized alignment projection)
        #   4. head (highest lr — randomly initialized, main learning target)
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

        # ReduceLROnPlateau monitoring val_f1 (mode=max) — REVERTED from parent's val_loss
        # Rationale:
        # - Parent's val_loss monitoring triggered premature LR reduction at epoch 12,
        #   causing val_f1 drop from 0.445 to 0.433 — directly confirmed counterproductive
        # - val_f1 correctly oscillates ±0.005 during the plateau phase without triggering
        #   reduction until val_f1 truly stops improving (plateau_patience=10)
        # - plateau_patience=10 (up from 8) to accommodate val_f1 oscillation amplitude
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",                    # maximize val_f1 — CRITICAL FIX from parent
            factor=hp.plateau_factor,      # 0.5: halve LR each reduction
            patience=hp.plateau_patience,  # 10: longer patience for val_f1 oscillation
            min_lr=hp.plateau_min_lr,      # 1e-7: floor to avoid near-zero LR
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_f1",   # REVERTED to val_f1 (parent used val_loss, caused regression)
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
        description="Node 1-2: AIDO.Cell-10M + LoRA r=4 All-8 layers + Symbol CNN + STRING GNN DEG predictor"
    )
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--global_batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--head_lr_multiplier", type=float, default=3.0)
    p.add_argument("--symbol_encoder_lr_multiplier", type=float, default=2.0)
    p.add_argument("--ppi_proj_lr_multiplier", type=float, default=2.0)
    p.add_argument("--weight_decay", type=float, default=0.06)      # increased from 0.04
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_layers_start", type=int, default=0)  # 0 = all 8 layers
    p.add_argument("--lora_dropout", type=float, default=0.2)   # increased from 0.1
    p.add_argument("--head_hidden", type=int, default=256)       # reduced from 320
    p.add_argument("--head_dropout", type=float, default=0.5)   # increased from 0.4
    p.add_argument("--gamma_focal", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--plateau_patience", type=int, default=10)   # increased from 8
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--plateau_min_lr", type=float, default=1e-7)
    p.add_argument("--early_stopping_patience", type=int, default=20)  # reduced from 35
    p.add_argument("--top_k_averaging", type=int, default=3)     # number of top checkpoints to avg
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
    limit_test_batches = 1.0  # Always run full test set

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=args.top_k_averaging,  # save top-k for averaging
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
        lora_layers_start=args.lora_layers_start,
        lora_dropout=args.lora_dropout,
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
        top_k_averaging=args.top_k_averaging,
    )

    trainer.fit(model_module, datamodule=datamodule)

    # ── Top-K Checkpoint Averaging for test inference ──────────────────────
    # Apply top-3 checkpoint averaging before running test evaluation.
    # This consistently provides +0.003-0.005 F1 improvement by reducing prediction
    # variance across the best training checkpoints.
    # Note: Only applied for full runs (not debug/fast_dev_run)
    use_ckpt_averaging = (
        not args.fast_dev_run
        and args.debug_max_step is None
        and args.top_k_averaging > 1
    )

    if use_ckpt_averaging and trainer.is_global_zero:
        # Get top-k checkpoint paths from callback
        top_k_paths = []
        if hasattr(checkpoint_cb, "best_k_models") and checkpoint_cb.best_k_models:
            # best_k_models is a dict: {path: score}
            top_k_paths = list(checkpoint_cb.best_k_models.keys())
            print(f"Top-{len(top_k_paths)} checkpoint paths for averaging: {top_k_paths}")
        elif hasattr(checkpoint_cb, "_top_k_best_checkpoints") and checkpoint_cb._top_k_best_checkpoints:
            top_k_paths = list(checkpoint_cb._top_k_best_checkpoints.keys())
            print(f"Top-{len(top_k_paths)} checkpoint paths for averaging: {top_k_paths}")

        if top_k_paths:
            # Load averaged weights into model_module
            average_top_k_checkpoints(top_k_paths, model_module)
            # Run test with averaged weights (no specific ckpt_path needed)
            test_results = trainer.test(model_module, datamodule=datamodule)
        else:
            # Fallback: use best single checkpoint
            print("Warning: Could not retrieve top-k checkpoint paths, using best single checkpoint.")
            test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")
    elif args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        # Single GPU/non-global-zero: use best checkpoint
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"test_results: {test_results}\n"
            f"val_f1_best: {checkpoint_cb.best_model_score}\n"
            f"top_k_averaging: {args.top_k_averaging}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
