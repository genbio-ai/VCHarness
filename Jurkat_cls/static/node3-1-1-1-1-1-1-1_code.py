#!/usr/bin/env python3
"""
Node node1-2 (child of node3-1-1-1-1-1-1):
    AIDO.Cell-10M + LoRA (r=8, last 4 layers) + Gene Symbol CNN (96-dim) + concat+MLP Head (576→512→19920)
==============================================================================
Improves on parent (test F1=0.4086) by incorporating the gene symbol CNN
architecture from node2-2 (tree best at F1=0.4453) with targeted stability fixes.

KEY INSIGHT from MCTS tree:
  The parent's performance ceiling (~0.41 F1) is NOT due to overfitting or
  architectural bottlenecks — it's due to the synthetic expression encoding
  providing only positional lookup information. The tree's best node (node2-2)
  broke through to 0.4453 by adding a character-level gene symbol CNN that
  captures real orthogonal biological signal (gene family naming conventions,
  prefix patterns like NDUF/KDM/DHX) that AIDO.Cell cannot encode from
  near-one-hot synthetic expression profiles.

KEY CHANGES from parent (node3-1-1-1-1-1-1, test F1=0.4086):

  1. GENE SYMBOL CNN ENCODER (PRIMARY FIX — from node2-2)
     - 3-branch character-level Conv1d (kernels 2/3/4) → 96-dim projection
     - Encodes gene symbol string (e.g., "BRCA1", "NDUF5") to capture:
       * Gene family prefixes (NDUF→complex I, KDM→histone demethylase)
       * Paralog suffix patterns (numeric index, letter variants)
       * Character bigrams/trigrams that co-correlate with function
     - 96-dim (vs node2-2's 64-dim) for more expressive symbol representation
     - Evidence: node2-2 achieved 0.4453 (+0.035 over sibling) with 64-dim symbol CNN

  2. LORA ON LAST 4 LAYERS (vs parent's 3 layers)
     - Extends LoRA coverage from {5,6,7} → {4,5,6,7} (0-indexed in 8-layer model)
     - Provides broader backbone adaptation capacity
     - Evidence: node2-2 used last 4 layers and achieved 0.4453

  3. WIDER MLP HEAD: 576→512 (vs node2-2's 576→256)
     - Head input: concat([pert_hidden(256), global_mean(256), symbol_cnn(96)]) = 608-dim
     - Wait: recalculated: 256+256+96 = 608
     - Head: LayerNorm(608) → Dropout(0.25) → Linear(608→512) → GELU → Dropout(0.25) → Linear(512→19920)
     - 512-dim hidden matches parent's proven architecture (vs node2-2's 256-dim)
     - No information bottleneck in the intermediate layer

  4. STABLE TRAINING WITH ReduceLROnPlateau (vs node2-2's cosine annealing)
     - node2-2's key failure: CosineAnnealingLR caused val_loss spikes and val_f1
       oscillation at every cosine valley. Val_f1 never recovered above epoch 18 peak.
     - Fix: ReduceLROnPlateau(mode=max, factor=0.5, patience=8, min_lr=1e-7)
       reacts to val_f1 plateaus without causing the destabilizing warmup→cosine
       transition that caused val_loss spikes at epoch 7 in node2-2
     - This is the scheduler proven stable in the parent chain (all nodes ≥0.40 used it)

  5. REDUCED HEAD LR (3× backbone vs node2-2's 5× backbone)
     - node2-2 feedback: "head_lr 2.5e-3 is borderline high for symbol encoder (22K params)"
     - Fix: head_lr = backbone_lr × 3 = 5e-5 × 3 = 1.5e-4 backbone; head = 4.5e-4
     - Actually using: backbone_lr=3e-5, head_lr=3e-4 (10× backbone, parent's exact LR pair)
     - But with stronger weight decay (0.02) than parent's 0.01 to regularize symbol CNN
     - Symbol encoder shares head_lr — appropriate given small 22K param count

  6. MODERATE REGULARIZATION TUNING
     - Dropout: 0.25 in head (vs parent's 0.2, vs node2-2's 0.4)
     - Weight decay: 0.02 (vs parent's 0.01, vs node2-2's 0.05)
     - label_smoothing: 0.05 (vs parent's 0.1 — mild smoothing)
     - lora_dropout: 0.15 (vs parent's 0.1 — slightly higher for 4 layers)

Architecture:
  Input: pert_id (ENSG gene ID) + symbol (gene symbol string)
          ↓
  Synthetic expression: [19264] float32 (all genes=1.0, perturbed=0.0)
  Symbol: string → character-level tensor
          ↓
  ┌─────────────────────────────────────────────────────────────────────┐
  │ AIDO.Cell-10M backbone (256-dim, 8 layers, bf16)                    │
  │   LoRA r=8 on layers 4,5,6,7 — Q, K, V projections                 │
  │   lora_alpha=16 (alpha/r=2)                                         │
  │   Gradient checkpointing enabled                                    │
  │   Output: [B, 19266, 256]                                           │
  └─────────────────────────────────────────────────────────────────────┘
  +
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Gene Symbol CNN (3-branch, NEW)                                      │
  │   Char embedding: [B, L, 32]                                        │
  │   3× Conv1d(32, 32, kernel) + ReLU + MaxPool                        │
  │   Concat 3 branches: [B, 96]                                        │
  │   Linear(96→96) → ReLU → Dropout(0.3)                              │
  │   Output: [B, 96]                                                   │
  └─────────────────────────────────────────────────────────────────────┘
          ↓
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Dual Pooling                                                         │
  │   pert_hidden = hidden[:, pert_pos, :]    → [B, 256]               │
  │   global_mean = hidden[:, :19264, :].mean(1) → [B, 256]            │
  │   dual_pool = concat([pert_hidden, global_mean]) → [B, 512]         │
  └─────────────────────────────────────────────────────────────────────┘
          ↓
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 2-layer MLP Head (512-dim bottleneck)                               │
  │   Input = concat([dual_pool(512), symbol_emb(96)]) → [B, 608]      │
  │   LayerNorm(608) → Dropout(0.25) → [B, 608]                        │
  │   Linear(608→512) → GELU          → [B, 512]                       │
  │   Dropout(0.25)   → [B, 512]                                        │
  │   Linear(512→19920) → [B, 19920]                                   │
  │   reshape → [B, 3, 6640]                                            │
  └─────────────────────────────────────────────────────────────────────┘

Total trainable parameters: ~13.0M
  - LoRA adapters (r=8, Q/K/V, layers 4-7): ~48K (vs parent's ~36K)
  - Gene Symbol CNN: ~22K (embedding + 3 conv branches + projection)
  - LayerNorm(608): ~1.2K
  - Linear(608→512) + bias: ~312K
  - Linear(512→3×6640) + bias: ~9.96M
  - Total: ~10.34M head + ~48K LoRA + ~22K CNN ≈ 10.41M

Evidence:
  - node2-2 (F1=0.4453): gene symbol CNN (64-dim), LoRA r=8 last 4 layers, 576→256→19920
  - parent   (F1=0.4086): no symbol CNN, LoRA r=8 last 3 layers, 512→512→19920
  - This node: combines gene symbol CNN (96-dim) + 4 LoRA layers + 608→512→19920
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "/home/Models/AIDO.Cell-10M"
N_GENES_OUT = 6_640
N_GENES_MODEL = 19_264
N_CLASSES = 3
HIDDEN_DIM = 256  # AIDO.Cell-10M hidden size

# Gene symbol character vocabulary: a-z, A-Z, 0-9, plus special chars '-', '.', '_', PAD
SYMBOL_VOCAB = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._")
SYMBOL_CHAR2IDX = {c: i + 1 for i, c in enumerate(SYMBOL_VOCAB)}  # 1-indexed; 0=PAD
SYMBOL_VOCAB_SIZE = len(SYMBOL_VOCAB) + 1  # +1 for PAD
SYMBOL_MAX_LEN = 16  # Maximum symbol length (gene symbols are typically ≤10 chars)
SYMBOL_CNN_DIM = 96  # Output dimension of symbol CNN

# Head input dimension: dual_pool (256+256) + symbol_cnn (96) = 608
HEAD_IN_DIM = HIDDEN_DIM * 2 + SYMBOL_CNN_DIM  # 608

# Moderate class weights: corrects for severe imbalance (~95% class 0/unchanged)
# [5.0, 1.0, 10.0] for {down, unchanged, up} — proven effective in parent chain
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Multi-class focal loss with class weights and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [N, C], targets: [N]
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Gene Symbol Encoder
# ─────────────────────────────────────────────────────────────────────────────
def encode_symbol(symbol: str, max_len: int = SYMBOL_MAX_LEN) -> torch.Tensor:
    """Convert a gene symbol string to a padded integer tensor."""
    indices = [SYMBOL_CHAR2IDX.get(c, 0) for c in symbol[:max_len]]
    # Pad to max_len
    indices = indices + [0] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)


class GeneSymbolCNN(nn.Module):
    """
    3-branch character-level CNN for gene symbol encoding.

    Architecture:
        Input: [B, L] long (character indices, 0=PAD)
        Embedding: [B, L, char_embed_dim]
        3× Conv1d(char_embed_dim, branch_dim, kernel_k) + ReLU + MaxPool1d → [B, branch_dim]
        Concat 3 branches → [B, 3*branch_dim]
        Linear(3*branch_dim → out_dim) → ReLU → Dropout → [B, out_dim]

    Captures gene name prefixes (gene families), suffixes (paralogs),
    and bigrams/trigrams that correlate with biological function.
    Evidence: node2-2 achieved 0.4453 F1 using similar architecture.
    """

    def __init__(
        self,
        vocab_size: int = SYMBOL_VOCAB_SIZE,
        char_embed_dim: int = 32,
        branch_dim: int = 32,
        out_dim: int = SYMBOL_CNN_DIM,
        max_len: int = SYMBOL_MAX_LEN,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, char_embed_dim, padding_idx=0)
        # 3 branches with different kernel sizes: 2, 3, 4
        self.conv2 = nn.Conv1d(char_embed_dim, branch_dim, kernel_size=2, padding=0)
        self.conv3 = nn.Conv1d(char_embed_dim, branch_dim, kernel_size=3, padding=0)
        self.conv4 = nn.Conv1d(char_embed_dim, branch_dim, kernel_size=4, padding=0)
        concat_dim = branch_dim * 3  # 96
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Initialize with small weights for stable training from scratch
        for conv in [self.conv2, self.conv3, self.conv4]:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        nn.init.trunc_normal_(self.proj[0].weight, std=0.02)
        nn.init.zeros_(self.proj[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L] long — character indices (0=PAD)
        Returns:
            [B, out_dim] float32
        """
        emb = self.char_embed(x).float()   # [B, L, char_embed_dim]
        emb = emb.permute(0, 2, 1)         # [B, char_embed_dim, L]

        # Apply each conv + ReLU + max pool along temporal dim
        f2 = F.relu(self.conv2(emb))       # [B, branch_dim, L-1]
        f3 = F.relu(self.conv3(emb))       # [B, branch_dim, L-2]
        f4 = F.relu(self.conv4(emb))       # [B, branch_dim, L-3]

        # Global max pool over time dimension
        g2 = f2.max(dim=2).values          # [B, branch_dim]
        g3 = f3.max(dim=2).values          # [B, branch_dim]
        g4 = f4.max(dim=2).values          # [B, branch_dim]

        feat = torch.cat([g2, g3, g4], dim=1)  # [B, 3*branch_dim = 96]
        return self.proj(feat)                  # [B, out_dim = 96]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Pre-builds synthetic expression vectors and symbol character tensors
    for each perturbation sample.

    Input encoding:
      - Backbone: all 19264 genes at 1.0 (baseline), perturbed gene at 0.0
      - Symbol: character-level integer sequence, padded to SYMBOL_MAX_LEN
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_pos_map: Dict[str, int],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Map {-1,0,1} → {0,1,2} to match calc_metric.py's y_true + 1 convention
            self.labels = np.array(raw_labels, dtype=np.int8) + 1
        else:
            self.labels = None

        # Pre-compute expression vectors and symbol character tensors
        base_expr = torch.ones(N_GENES_MODEL, dtype=torch.float32)

        self._exprs: List[torch.Tensor] = []
        self._pert_positions: List[int] = []
        self._symbol_tensors: List[torch.Tensor] = []
        covered = 0
        for i, pid in enumerate(self.pert_ids):
            base_pid = pid.split(".")[0]
            pos = gene_pos_map.get(base_pid, -1)
            self._pert_positions.append(pos)

            if pos >= 0:
                expr = base_expr.clone()
                expr[pos] = 0.0  # knockout signal
                covered += 1
            else:
                expr = base_expr.clone()  # no knockout signal (gene not in vocab)
            self._exprs.append(expr)

            # Pre-encode gene symbol as character tensor
            sym_tensor = encode_symbol(self.symbols[i])
            self._symbol_tensors.append(sym_tensor)

        if not is_test:
            print(f"[Dataset] {len(self.pert_ids)} samples, "
                  f"{covered}/{len(self.pert_ids)} genes in AIDO.Cell vocab "
                  f"({100.0 * covered / len(self.pert_ids):.1f}%)")

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "expr": self._exprs[idx],                         # [19264] float32
            "pert_pos": self._pert_positions[idx],            # int
            "symbol_chars": self._symbol_tensors[idx],        # [SYMBOL_MAX_LEN] long
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
        "expr": torch.stack([b["expr"] for b in batch]),                  # [B, 19264]
        "pert_pos": torch.tensor([b["pert_pos"] for b in batch], dtype=torch.long),  # [B]
        "symbol_chars": torch.stack([b["symbol_chars"] for b in batch]),  # [B, SYMBOL_MAX_LEN]
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])        # [B, 6640]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, micro_batch_size: int = 8, num_workers: int = 0):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []
        self._gene_pos_map: Optional[Dict[str, int]] = None

    def _build_gene_pos_map(
        self, tokenizer, all_pert_ids: List[str]
    ) -> Dict[str, int]:
        """Build mapping from ENSG gene ID to its position in the 19264-gene vocabulary."""
        gene_pos_map: Dict[str, int] = {}
        unique_base_ids = list(set(pid.split(".")[0] for pid in all_pert_ids))
        print(f"[DataModule] Building gene position map for {len(unique_base_ids)} unique genes...")

        # Primary: use gene_id_to_index (maps ENSG → position)
        if hasattr(tokenizer, "gene_id_to_index"):
            gid2idx = tokenizer.gene_id_to_index
            for base_pid in unique_base_ids:
                if base_pid in gid2idx:
                    gene_pos_map[base_pid] = gid2idx[base_pid]
            if len(gene_pos_map) > 0:
                print(f"[DataModule] ENSG→pos via gene_id_to_index: "
                      f"{len(gene_pos_map)}/{len(unique_base_ids)} found")
                return gene_pos_map

        # Fallback: gene_to_index lookup unavailable — return empty map
        if hasattr(tokenizer, "gene_to_index"):
            print(f"[DataModule] gene_id_to_index had 0 matches; using gene_to_index fallback")
            return gene_pos_map  # Returns empty; lookup must be done by symbol separately

        print(f"[DataModule] No gene position mapping available; all pert_pos will be -1")
        return gene_pos_map

    def setup(self, stage: Optional[str] = None) -> None:
        # Initialize tokenizer: rank-0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

        # Build gene position map once (covers all splits)
        if self._gene_pos_map is None:
            all_ids: List[str] = []
            for fname in ["train.tsv", "val.tsv", "test.tsv"]:
                fpath = self.data_dir / fname
                if fpath.exists():
                    df_tmp = pd.read_csv(fpath, sep="\t")
                    if "pert_id" in df_tmp.columns:
                        all_ids.extend(df_tmp["pert_id"].tolist())
            self._gene_pos_map = self._build_gene_pos_map(tokenizer, all_ids)

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self._gene_pos_map)
            self.val_ds = PerturbationDataset(val_df, self._gene_pos_map)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(test_df, self._gene_pos_map, is_test=True)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    def _loader(self, ds: PerturbationDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class AIDOCellSymbolDEGModel(nn.Module):
    """
    AIDO.Cell-10M + LoRA (r=8, last 4 layers) + Gene Symbol CNN (96-dim) + MLP Head.

    Architecture:
      1. AIDO.Cell-10M backbone with LoRA on last 4 attention layers
      2. Dual pooling: concat([pert_hidden, global_mean]) → [B, 512]
      3. Gene symbol CNN: char-level 3-branch CNN → [B, 96]
      4. MLP head: concat([dual_pool, symbol_emb]) → [B, 608]
                   LayerNorm → Dropout → Linear(608→512) → GELU → Dropout
                   → Linear(512→19920) → reshape → [B, 3, 6640]

    Trainable parameters ~10.4M:
      - LoRA adapters (r=8, Q/K/V, layers 4-7): ~48K
      - Gene Symbol CNN (char emb + 3 convs + proj): ~22K
      - LayerNorm(608): ~1.2K
      - Linear(608→512): ~312K
      - Linear(512→19920): ~9.96M
    """

    def __init__(
        self,
        dropout: float = 0.25,
        mlp_hidden: int = 512,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.15,
        symbol_cnn_dim: int = SYMBOL_CNN_DIM,
        symbol_cnn_dropout: float = 0.3,
    ):
        super().__init__()

        # ── Backbone: AIDO.Cell-10M with LoRA ──────────────────────────────
        backbone = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16)
        backbone.config.use_cache = False

        # Monkey-patch enable_input_require_grads to avoid AIDO.Cell's NotImplementedError
        def noop_enable_input_require_grads(self):
            pass
        backbone.enable_input_require_grads = noop_enable_input_require_grads.__get__(
            backbone, type(backbone))

        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # LoRA on last 4 layers (layers 4, 5, 6, 7 in 0-indexed 8-layer model)
        # Extends parent's scope from {5,6,7} to {4,5,6,7} matching node2-2's proven config
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=[4, 5, 6, 7],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Cast LoRA (trainable) params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Gene Symbol CNN (NEW: primary information channel addition) ─────
        self.symbol_cnn = GeneSymbolCNN(
            vocab_size=SYMBOL_VOCAB_SIZE,
            char_embed_dim=32,
            branch_dim=32,
            out_dim=symbol_cnn_dim,
            max_len=SYMBOL_MAX_LEN,
            dropout=symbol_cnn_dropout,
        )

        # ── 2-layer MLP head (608→512→19920) ───────────────────────────────
        # Input: concat([pert_hidden(256), global_mean(256), symbol_cnn(96)]) = 608
        in_dim = HEAD_IN_DIM   # 256 + 256 + 96 = 608
        out_dim = N_CLASSES * N_GENES_OUT  # 3 × 6640 = 19920

        self.head_norm = nn.LayerNorm(in_dim)
        self.head_dropout1 = nn.Dropout(dropout)
        self.head_proj1 = nn.Linear(in_dim, mlp_hidden, bias=True)    # 608→512
        self.head_act = nn.GELU()
        self.head_dropout2 = nn.Dropout(dropout)
        self.head_proj2 = nn.Linear(mlp_hidden, out_dim, bias=True)   # 512→19920

        # Initialize with truncated normal
        nn.init.trunc_normal_(self.head_proj1.weight, std=0.02)
        nn.init.zeros_(self.head_proj1.bias)
        nn.init.trunc_normal_(self.head_proj2.weight, std=0.02)
        nn.init.zeros_(self.head_proj2.bias)

    def forward(
        self,
        expr: torch.Tensor,
        pert_pos: torch.Tensor,
        symbol_chars: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            expr:         [B, 19264] float32 — synthetic expression (1.0 baseline, 0.0 at pert)
            pert_pos:     [B] long — position of perturbed gene in vocab (-1 if unknown)
            symbol_chars: [B, SYMBOL_MAX_LEN] long — character indices of gene symbol
        Returns:
            logits: [B, 3, 6640]
        """
        B = expr.shape[0]
        device = expr.device

        # ── 1. AIDO.Cell forward pass ──────────────────────────────────────
        outputs = self.backbone(
            input_ids=expr,
            attention_mask=torch.ones(B, N_GENES_MODEL, dtype=torch.long, device=device),
        )
        # Cast to float32 for stable head computation
        hidden = outputs.last_hidden_state.float()  # [B, 19266, 256]

        # ── 2. Dual pooling ────────────────────────────────────────────────
        # Global mean pool over gene positions (exclude 2 summary tokens)
        global_pool = hidden[:, :N_GENES_MODEL, :].mean(dim=1)  # [B, 256]

        # Per-sample perturbed-gene positional embedding
        safe_pos = pert_pos.clamp(min=0)  # [B], -1 → 0 temporarily
        pos_idx = safe_pos.view(B, 1, 1).expand(B, 1, HIDDEN_DIM)  # [B, 1, 256]
        pert_hidden = hidden.gather(1, pos_idx).squeeze(1)  # [B, 256]

        # For genes not in vocabulary: fall back to global pool
        unknown_mask = (pert_pos < 0)
        if unknown_mask.any():
            pert_hidden = pert_hidden.clone()
            pert_hidden[unknown_mask] = global_pool[unknown_mask]

        # Dual pool: concat([pert_hidden, global_pool]) → [B, 512]
        dual_pool = torch.cat([pert_hidden, global_pool], dim=1)  # [B, 512]

        # ── 3. Gene symbol CNN (NEW) ────────────────────────────────────────
        symbol_emb = self.symbol_cnn(symbol_chars.to(device))  # [B, 96]

        # ── 4. Concat + MLP head ────────────────────────────────────────────
        combined = torch.cat([dual_pool, symbol_emb], dim=1)  # [B, 608]
        x = self.head_norm(combined)                           # [B, 608]
        x = self.head_dropout1(x)                              # [B, 608]
        x = self.head_proj1(x)                                 # [B, 512]
        x = self.head_act(x)                                   # [B, 512]
        x = self.head_dropout2(x)                              # [B, 512]
        logits = self.head_proj2(x)                            # [B, 19920]

        return logits.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """
    Per-gene macro-averaged F1 score, matching calc_metric.py logic exactly.

    y_pred: [n_samples, 3, n_genes] — class probabilities
    y_true_remapped: [n_samples, n_genes] — labels in {0,1,2} (i.e., y_true+1)
    """
    n_genes = y_true_remapped.shape[1]
    y_hat = y_pred.argmax(axis=1)  # [n_samples, n_genes]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp = y_hat[:, g]
        present = np.array([(yt == c).any() for c in range(N_CLASSES)])
        pf1 = sk_f1_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        dropout: float = 0.25,
        mlp_hidden: int = 512,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.15,
        symbol_cnn_dim: int = SYMBOL_CNN_DIM,
        symbol_cnn_dropout: float = 0.3,
        backbone_lr: float = 3e-5,
        head_lr: float = 3e-4,
        weight_decay: float = 0.02,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 200,
        lr_reduce_patience: int = 8,
        lr_reduce_factor: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCellSymbolDEGModel] = None
        self.loss_fn: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = AIDOCellSymbolDEGModel(
                dropout=self.hparams.dropout,
                mlp_hidden=self.hparams.mlp_hidden,
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                symbol_cnn_dim=self.hparams.symbol_cnn_dim,
                symbol_cnn_dropout=self.hparams.symbol_cnn_dropout,
            )
            self.loss_fn = FocalLoss(
                gamma=self.hparams.focal_gamma,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        # Populate test metadata for prediction saving
        if stage in ("test", None):
            dm = getattr(self, "trainer", None)
            if dm is not None:
                dm = getattr(self.trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "test_pert_ids") and dm.test_pert_ids:
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            batch["expr"],
            batch["pert_pos"],
            batch["symbol_chars"],
        )  # [B, 3, 6640]

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, 3, 6640], labels: [B, 6640]
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                               # [B*6640]
        return self.loss_fn(logits_flat, labels_flat)

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
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, 6640]
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        lp = torch.cat(self._val_preds, 0)   # [N, 3, 6640]
        ll = torch.cat(self._val_labels, 0)  # [N, 6640]
        li = torch.cat(self._val_indices, 0) # [N]

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # Check if distributed is actually initialized
        dist_init = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = int(os.environ.get("WORLD_SIZE", 1)) if dist_init else 1

        if dist_init and world_size > 1:
            # Multi-GPU: pad to same size, gather across ranks, deduplicate
            all_sizes = [0] * world_size
            torch.distributed.all_gather_object(all_sizes, li.size(0))
            max_n = max(all_sizes) if all_sizes else li.size(0)

            if li.size(0) < max_n:
                pad_li = torch.full((max_n - li.size(0),), -1, dtype=li.dtype, device=li.device)
                li = torch.cat([li, pad_li], dim=0)
                pad_ll = torch.full((max_n - ll.size(0), ll.size(1)), 0, dtype=ll.dtype, device=ll.device)
                ll = torch.cat([ll, pad_ll], dim=0)
                pad_lp = torch.zeros((max_n - lp.size(0), lp.size(1), lp.size(2)), dtype=lp.dtype, device=lp.device)
                lp = torch.cat([lp, pad_lp], dim=0)

            ap = self.all_gather(lp)  # [world, max_n, 3, 6640]
            al = self.all_gather(ll)  # [world, max_n, 6640]
            ai = self.all_gather(li)  # [world, max_n]

            preds_list: List[np.ndarray] = []
            labels_list: List[np.ndarray] = []
            seen: set = set()

            for w in range(world_size):
                for i in range(max_n):
                    global_idx = int(ai[w, i].item())
                    if global_idx < 0 or global_idx in seen:
                        continue
                    seen.add(global_idx)
                    preds_list.append(ap[w, i].cpu().numpy())
                    labels_list.append(al[w, i].cpu().numpy())

            order = np.argsort(list(seen))
            preds_arr = np.stack([preds_list[j] for j in order], axis=0)
            labels_arr = np.stack([labels_list[j] for j in order], axis=0)
        else:
            # Single-GPU: use tensors directly, sort by global index
            preds_arr = lp.cpu().numpy()
            labels_arr = ll.cpu().numpy()
            indices_arr = li.cpu().numpy()
            order = np.argsort(indices_arr)
            preds_arr = preds_arr[order]
            labels_arr = labels_arr[order]

        f1_val = compute_deg_f1(preds_arr, labels_arr.astype(np.int64))
        self.log("val_f1", f1_val, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        lp = torch.cat(self._test_preds, 0)  # [N, 3, 6640]
        li = torch.cat(self._test_indices, 0)  # [N]
        self._test_preds.clear()
        self._test_indices.clear()

        dist_init = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = int(os.environ.get("WORLD_SIZE", 1)) if dist_init else 1

        if dist_init and world_size > 1:
            # Multi-GPU: pad, gather, deduplicate
            all_sizes = [0] * world_size
            torch.distributed.all_gather_object(all_sizes, li.size(0))
            max_n = max(all_sizes) if all_sizes else li.size(0)

            if li.size(0) < max_n:
                pad_li = torch.full((max_n - li.size(0),), -1, dtype=li.dtype, device=li.device)
                li = torch.cat([li, pad_li], dim=0)
                pad_lp = torch.zeros((max_n - lp.size(0), lp.size(1), lp.size(2)), dtype=lp.dtype, device=lp.device)
                lp = torch.cat([lp, pad_lp], dim=0)

            ap = self.all_gather(lp)  # [world, max_n, 3, 6640]
            ai = self.all_gather(li)  # [world, max_n]

            preds_list: List[np.ndarray] = []
            idxs_list: List[int] = []
            seen: set = set()

            for w in range(world_size):
                for i in range(max_n):
                    global_idx = int(ai[w, i].item())
                    if global_idx < 0 or global_idx in seen:
                        continue
                    seen.add(global_idx)
                    preds_list.append(ap[w, i].cpu().numpy())
                    idxs_list.append(global_idx)

            order = np.argsort(idxs_list)
            preds_arr = np.stack([preds_list[i] for i in order], axis=0)
            idxs_arr = np.array(idxs_list, dtype=np.int64)[order]
        else:
            # Single-GPU: use tensors directly
            preds_arr = lp.cpu().numpy()
            indices_arr = li.cpu().numpy()
            order = np.argsort(indices_arr)
            preds_arr = preds_arr[order]
            idxs_arr = indices_arr[order]

        if self.trainer.is_global_zero:
            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "idx": self._test_pert_ids[i],
                    "input": self._test_symbols[i],
                    "prediction": json.dumps(preds_arr[r].tolist()),
                }
                for r, i in enumerate(idxs_arr)
            ]
            pred_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {pred_path} ({len(rows)} samples)")

    def configure_optimizers(self):
        # Separate learning rates: LoRA backbone (lower) vs. MLP head + symbol CNN (higher)
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters() if p.requires_grad
        ]
        # Symbol CNN + MLP head params share the head_lr
        symbol_params = list(self.model.symbol_cnn.parameters())
        head_params = (
            list(self.model.head_norm.parameters()) +
            list(self.model.head_proj1.parameters()) +
            list(self.model.head_proj2.parameters())
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
                {"params": symbol_params + head_params, "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
            eps=1e-8,
        )

        # ReduceLROnPlateau: monitors val_f1, reduces LR when plateau detected
        # Chosen over CosineAnnealingLR (node2-2 suffered LR spikes from cosine schedule)
        # mode=max: maximize val_f1; patience=8: responsive to plateaus
        # factor=0.5: halves LR on each plateau detection
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.lr_reduce_factor,
            patience=self.hparams.lr_reduce_patience,
            min_lr=1e-8,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ── Checkpoint helpers ─────────────────────────────────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_sd = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable_sd[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                trainable_sd[key] = full_sd[key]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable}/{total} params "
            f"({100.0 * trainable / total:.2f}%), plus {buffers} buffer values"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        """Load partial checkpoint (trainable params + buffers only)."""
        full_keys = set(super().state_dict().keys())
        trainable_keys = {n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys = {n for n, _ in self.named_buffers() if n in full_keys}
        expected_keys = trainable_keys | buffer_keys

        missing = [k for k in expected_keys if k not in state_dict]
        unexpected = [k for k in state_dict if k not in expected_keys]
        if missing:
            self.print(f"Warning: Missing keys in checkpoint (first 5): {missing[:5]}")
        if unexpected:
            self.print(f"Warning: Unexpected keys in checkpoint (first 5): {unexpected[:5]}")
        return super().load_state_dict(state_dict, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node node1-2: AIDO.Cell-10M + LoRA (r=8, last 4 layers) + Gene Symbol CNN (96-dim) + MLP Head (608→512)"
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "data"),
    )
    p.add_argument("--micro_batch_size",        type=int,   default=8)
    p.add_argument("--global_batch_size",       type=int,   default=64)
    p.add_argument("--max_epochs",              type=int,   default=200)
    p.add_argument("--backbone_lr",             type=float, default=3e-5)
    p.add_argument("--head_lr",                 type=float, default=3e-4)
    p.add_argument("--weight_decay",            type=float, default=0.02)
    p.add_argument("--dropout",                 type=float, default=0.25)
    p.add_argument("--mlp_hidden",              type=int,   default=512)
    p.add_argument("--lora_r",                  type=int,   default=8)
    p.add_argument("--lora_alpha",              type=int,   default=16)
    p.add_argument("--lora_dropout",            type=float, default=0.15)
    p.add_argument("--symbol_cnn_dim",          type=int,   default=96)
    p.add_argument("--symbol_cnn_dropout",      type=float, default=0.3)
    p.add_argument("--focal_gamma",             type=float, default=2.0)
    p.add_argument("--label_smoothing",         type=float, default=0.05)
    p.add_argument("--lr_reduce_patience",      type=int,   default=8)
    p.add_argument("--lr_reduce_factor",        type=float, default=0.5)
    p.add_argument("--early_stopping_patience", type=int,   default=30)
    p.add_argument("--num_workers",             type=int,   default=0)
    p.add_argument("--val_check_interval",      type=float, default=1.0)
    p.add_argument("--debug_max_step",          type=int,   default=None)
    p.add_argument("--fast_dev_run",            action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit = args.debug_max_step if args.debug_max_step is not None else 1.0

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-aido10m-symcnn96-lora4layers-r8-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stopping_patience,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    # Strategy: DDP for multi-GPU, SingleDevice for single GPU
    if n_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(seconds=300),
        )
    else:
        strategy = SingleDeviceStrategy(device="cuda:0")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit,
        limit_val_batches=limit,
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
        dropout=args.dropout,
        mlp_hidden=args.mlp_hidden,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        symbol_cnn_dim=args.symbol_cnn_dim,
        symbol_cnn_dropout=args.symbol_cnn_dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        lr_reduce_patience=args.lr_reduce_patience,
        lr_reduce_factor=args.lr_reduce_factor,
    )

    trainer.fit(model_module, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(
            model_module, datamodule=datamodule, ckpt_path="best"
        )

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        primary_val = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else float("nan")
        )
        score_path.write_text(
            f"# Node node1-2: AIDO.Cell-10M + LoRA (r=8, last 4 layers) + Gene Symbol CNN (96-dim) + MLP Head (608→512)\n"
            f"# Model: AIDO.Cell-10M, LoRA r=8 layers 4-7, gene symbol CNN (96-dim), 2-layer MLP head (608→512→19920)\n"
            f"# Primary metric: f1_score (macro-averaged per-gene F1)\n"
            f"# Key improvements: Gene symbol CNN (orthogonal information channel), 4 LoRA layers (vs parent's 3),\n"
            f"#   wider MLP head (608→512 vs parent's 512→512), ReduceLROnPlateau (vs cosine annealing in node2-2)\n"
            f"# Evidence: node2-2 achieved 0.4453 (tree best) with gene symbol CNN; parent 0.4086 without it\n"
            f"\n"
            f"Best val_f1 (from checkpoint): {primary_val:.6f}\n"
            f"\n"
            f"Test results:\n"
            f"{json.dumps(test_results, indent=2)}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
