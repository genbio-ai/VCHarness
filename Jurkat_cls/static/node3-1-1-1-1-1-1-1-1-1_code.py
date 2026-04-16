#!/usr/bin/env python3
"""
Node node1-2-2 (child of node3-1-1-1-1-1-1-1-1 / parent_node):
    AIDO.Cell-10M + LoRA (r=8, last 4 layers) + Gene Symbol CNN (64-dim) + MLP Head (576→320→19920)
    + CosineAnnealingLR with warmup + Checkpoint Averaging (top-3)
==============================================================================
Improves on parent (test F1=0.4323) by:

1. REDUCED MLP HEAD HIDDEN: 512 → 320 (PRIMARY FIX)
   - node2-2-1 achieved F1=0.4472 with 320-dim head (vs parent's 0.4323 with 512-dim)
   - 512-dim provides 192 extra hidden units that consume capacity fitting noise
   - 320-dim reduces total trainable params from 10.3M to ~9.5M

2. SWITCH TO CosineAnnealingLR WITH WARMUP (SECONDARY FIX)
   - Parent's ReduceLROnPlateau (patience=5, factor=0.3) triggered 4 LR reductions
     across 25 epochs, with LR dropping to near-zero (2.43e-7) without improvement
   - node2-2-1 used smooth cosine decay and achieved better F1 (0.4472)
   - CosineAnnealing with T_max=80 + warmup=5 epochs avoids destabilizing plateau jumps
   - Min LR = 1e-7 to prevent full decay to zero

3. REDUCED HEAD LR: 5e-4 → 3e-4
   - node2-2-1 used head_lr=3e-4 and achieved 0.4472
   - Parent's 5e-4 combined with wide 512-dim head creates larger weight updates
     that don't generalize as well under architectural saturation
   - Lower head_lr with narrower head produces a more stable convergence

4. INCREASED SYMBOL CNN DROPOUT: 0.2 → 0.3
   - The 64-dim symbol CNN with 0.2 dropout may still capture noise during long training
   - 0.3 provides slightly more regularization, consistent with node2-2's 0.4 dropout
   - Middle ground between parent's 0.2 and node2-2's 0.4

5. RETAINED: Checkpoint averaging (top-3) from parent (marginal +0.0003 but cost-free)

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
  │ Gene Symbol CNN (3-branch, 64-dim)                                  │
  │   Char embedding: [B, L, 32]                                        │
  │   3× Conv1d(32, 32, kernel) + ReLU + MaxPool                        │
  │   Concat 3 branches: [B, 96]                                        │
  │   Linear(96→64) → ReLU → Dropout(0.3)  ← INCREASED from 0.2       │
  │   Output: [B, 64]                                                   │
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
  │ 2-layer MLP Head (576→320→19920)  ← 320-dim (was 512)              │
  │   Input = concat([dual_pool(512), symbol_emb(64)]) → [B, 576]      │
  │   LayerNorm(576) → Dropout(0.25) → [B, 576]                        │
  │   Linear(576→320) → GELU          → [B, 320]  ← NARROWER           │
  │   Dropout(0.25)   → [B, 320]                                        │
  │   Linear(320→19920) → [B, 19920]                                   │
  │   reshape → [B, 3, 6640]                                            │
  └─────────────────────────────────────────────────────────────────────┘

Total trainable parameters: ~9.5M
  - LoRA adapters (r=8, Q/K/V, layers 4-7): ~48K
  - Gene Symbol CNN: ~18K (embedding + 3 conv branches + projection to 64-dim)
  - LayerNorm(576): ~1.2K
  - Linear(576→320) + bias: ~185K (vs parent's 296K for 576→512)
  - Linear(320→3×6640) + bias: ~6.35M (vs parent's 9.96M for 512→19920)

Evidence:
  - node2-2-1 (F1=0.4472): 64-dim symbol CNN, 576→320→19920 head, cosine LR, head_lr=3e-4
  - node2-2   (F1=0.4453): 64-dim symbol CNN, 576→256→19920 head, cosine LR
  - parent    (F1=0.4323): 64-dim symbol CNN, 576→512→19920 head, ReduceLROnPlateau, head_lr=5e-4
  - This node converges toward node2-2-1's exact winning configuration
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
SYMBOL_CHAR2IDX_reverse = {v: k for k, v in SYMBOL_CHAR2IDX.items()}  # 1-indexed; 0=PAD → ""
SYMBOL_VOCAB_SIZE = len(SYMBOL_VOCAB) + 1  # +1 for PAD
SYMBOL_MAX_LEN = 16  # Maximum symbol length (gene symbols are typically ≤10 chars)
SYMBOL_CNN_DIM = 64  # Output dimension of symbol CNN — proven optimal at node2-2 (F1=0.4453)

# Head input dimension: dual_pool (256+256) + symbol_cnn (64) = 576
HEAD_IN_DIM = HIDDEN_DIM * 2 + SYMBOL_CNN_DIM  # 576

# Moderate class weights: corrects for severe imbalance (~95% class 0/unchanged)
# [5.0, 1.0, 10.0] for {down, unchanged, up} — proven effective across all branches
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
    Evidence: node2-2 achieved 0.4453 F1 using 64-dim architecture.
    """

    def __init__(
        self,
        vocab_size: int = SYMBOL_VOCAB_SIZE,
        char_embed_dim: int = 32,
        branch_dim: int = 32,
        out_dim: int = SYMBOL_CNN_DIM,  # 64 (proven optimal)
        max_len: int = SYMBOL_MAX_LEN,
        dropout: float = 0.3,  # INCREASED from 0.2 for better regularization
    ):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, char_embed_dim, padding_idx=0)
        # 3 branches with different kernel sizes: 2, 3, 4
        self.conv2 = nn.Conv1d(char_embed_dim, branch_dim, kernel_size=2, padding=0)
        self.conv3 = nn.Conv1d(char_embed_dim, branch_dim, kernel_size=3, padding=0)
        self.conv4 = nn.Conv1d(char_embed_dim, branch_dim, kernel_size=4, padding=0)
        concat_dim = branch_dim * 3  # 96 (3 branches × 32)
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, out_dim),  # 96 → 64
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
        return self.proj(feat)                  # [B, out_dim = 64]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Averaging Utility
# ─────────────────────────────────────────────────────────────────────────────
def average_checkpoints(checkpoint_paths: List[Path]) -> Dict[str, torch.Tensor]:
    """
    Average model weights across multiple saved checkpoints.

    Loads each checkpoint's state_dict and computes a parameter-wise mean.
    This smooths the val_f1 plateau oscillation.

    Args:
        checkpoint_paths: List of paths to .ckpt files (top-k by val_f1)
    Returns:
        Averaged state_dict (same key format as Lightning checkpoints)
    """
    state_dicts = []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt)
        state_dicts.append(sd)

    if not state_dicts:
        return {}

    # Average all matching keys
    avg_sd = {}
    for key in state_dicts[0]:
        tensors = [sd[key].float() for sd in state_dicts if key in sd]
        if tensors:
            avg_sd[key] = torch.stack(tensors).mean(0)

    print(f"[CheckpointAvg] Averaged {len(state_dicts)} checkpoints "
          f"({', '.join(p.name for p in checkpoint_paths)})")
    return avg_sd


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
        loader = self._loader(self.val_ds, shuffle=False)
        dist_init = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = int(os.environ.get("WORLD_SIZE", 1)) if dist_init else 1
        rank = int(os.environ.get("RANK", "0")) if dist_init else 0
        if world_size > 1:
            sampler = torch.utils.data.DistributedSampler(
                self.val_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=0,
            )
            loader = DataLoader(
                self.val_ds,
                batch_size=self.micro_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
                sampler=sampler,
            )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = self._loader(self.test_ds, shuffle=False)
        # Use DistributedSampler so each rank processes its own subset in DDP mode.
        # Without this, each rank iterates over the full dataset independently,
        # leading to duplicate predictions and wasted computation.
        dist_init = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = int(os.environ.get("WORLD_SIZE", 1)) if dist_init else 1
        rank = int(os.environ.get("RANK", "0")) if dist_init else 0
        if world_size > 1:
            sampler = torch.utils.data.DistributedSampler(
                self.test_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=0,
            )
            loader = DataLoader(
                self.test_ds,
                batch_size=self.micro_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
                sampler=sampler,
            )
        return loader


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class AIDOCellSymbolDEGModel(nn.Module):
    """
    AIDO.Cell-10M + LoRA (r=8, last 4 layers) + Gene Symbol CNN (64-dim) + MLP Head.

    Architecture:
      1. AIDO.Cell-10M backbone with LoRA on last 4 attention layers
      2. Dual pooling: concat([pert_hidden, global_mean]) → [B, 512]
      3. Gene symbol CNN: char-level 3-branch CNN → [B, 64]
      4. MLP head: concat([dual_pool, symbol_emb]) → [B, 576]
                   LayerNorm → Dropout → Linear(576→320) → GELU → Dropout  ← NARROWER HEAD
                   → Linear(320→19920) → reshape → [B, 3, 6640]

    Trainable parameters ~9.5M:
      - LoRA adapters (r=8, Q/K/V, layers 4-7): ~48K
      - Gene Symbol CNN (char emb + 3 convs + proj): ~18K
      - LayerNorm(576): ~1.2K
      - Linear(576→320): ~185K (REDUCED from 576→512 = 296K)
      - Linear(320→19920): ~6.35M (REDUCED from 512→19920 = 9.96M)
    """

    def __init__(
        self,
        dropout: float = 0.25,
        mlp_hidden: int = 320,          # REDUCED from 512 — matching node2-2-1's proven 320-dim
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.15,
        symbol_cnn_dim: int = SYMBOL_CNN_DIM,
        symbol_cnn_dropout: float = 0.3,  # INCREASED from 0.2
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
        # Proven configuration from node2-2 (best at 0.4453)
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

        # ── Gene Symbol CNN (64-dim) ────────────────────────────────────────
        self.symbol_cnn = GeneSymbolCNN(
            vocab_size=SYMBOL_VOCAB_SIZE,
            char_embed_dim=32,
            branch_dim=32,
            out_dim=symbol_cnn_dim,
            max_len=SYMBOL_MAX_LEN,
            dropout=symbol_cnn_dropout,
        )

        # ── 2-layer MLP head (576→320→19920) — narrower for better generalization ─
        # Input: concat([pert_hidden(256), global_mean(256), symbol_cnn(64)]) = 576
        in_dim = HIDDEN_DIM * 2 + symbol_cnn_dim  # 512 + 64 = 576
        out_dim = N_CLASSES * N_GENES_OUT  # 3 × 6640 = 19920

        self.head_norm = nn.LayerNorm(in_dim)
        self.head_dropout1 = nn.Dropout(dropout)
        self.head_proj1 = nn.Linear(in_dim, mlp_hidden, bias=True)    # 576→320
        self.head_act = nn.GELU()
        self.head_dropout2 = nn.Dropout(dropout)
        self.head_proj2 = nn.Linear(mlp_hidden, out_dim, bias=True)   # 320→19920

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

        # ── 3. Gene symbol CNN (64-dim) ────────────────────────────────────
        symbol_emb = self.symbol_cnn(symbol_chars.to(device))  # [B, 64]

        # ── 4. Concat + MLP head ────────────────────────────────────────────
        combined = torch.cat([dual_pool, symbol_emb], dim=1)  # [B, 576]
        x = self.head_norm(combined)                           # [B, 576]
        x = self.head_dropout1(x)                              # [B, 576]
        x = self.head_proj1(x)                                 # [B, 320] ← NARROWER
        x = self.head_act(x)                                   # [B, 320]
        x = self.head_dropout2(x)                              # [B, 320]
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
        mlp_hidden: int = 320,           # REDUCED from 512 — PRIMARY FIX
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.15,
        symbol_cnn_dim: int = SYMBOL_CNN_DIM,
        symbol_cnn_dropout: float = 0.3, # INCREASED from 0.2
        backbone_lr: float = 3e-5,
        head_lr: float = 3e-4,           # REDUCED from 5e-4 — matching node2-2-1's proven config
        weight_decay: float = 0.02,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 120,           # Reduced: CosineAnnealing converges faster
        cosine_t_max: int = 80,          # CosineAnnealing period (80 epochs = good coverage)
        cosine_eta_min: float = 1e-7,    # Minimum LR — prevent full decay to zero
        warmup_epochs: int = 5,          # Warmup for the first 5 epochs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCellSymbolDEGModel] = None
        self.loss_fn: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []  # dataset indices (global, not local batch)
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []   # Full ordered list from datamodule (for single-GPU)
        self._test_symbols: List[str] = []    # Full ordered list from datamodule (for single-GPU)
        # Per-rank metadata lists for DDP multi-GPU test (tracks what THIS rank processed)
        self._test_pert_ids_local: List[str] = []  # pert_id strings per sample
        self._test_symbols_local: List[torch.Tensor] = []  # [B, 16] char tensors per batch

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

        # Populate test metadata for prediction saving.
        # Also reset local DDP lists in case of multiple test() calls.
        if stage in ("test", None):
            self._test_pert_ids_local.clear()
            self._test_symbols_local.clear()
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
        # Encode symbols as character tensors for all_gather compatibility
        self._test_symbols_local.append(torch.stack([encode_symbol(s) for s in batch["symbols"]]))

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        lp = torch.cat(self._test_preds, 0)  # [N, 3, 6640]
        li = torch.cat(self._test_indices, 0)  # [N] — global dataset indices
        ls = torch.cat(self._test_symbols_local, 0)  # [N, SYMBOL_MAX_LEN] char tensor
        self._test_preds.clear()
        self._test_indices.clear()
        self._test_symbols_local.clear()

        dist_init = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = int(os.environ.get("WORLD_SIZE", 1)) if dist_init else 1

        if dist_init and world_size > 1:
            # Multi-GPU: each rank processes its own subset via DistributedSampler.
            # Gather all predictions, indices, and symbols from every rank.
            all_sizes = [0] * world_size
            torch.distributed.all_gather_object(all_sizes, li.size(0))
            max_n = max(all_sizes) if all_sizes else li.size(0)

            # Pad to max_n for all_gather compatibility
            if li.size(0) < max_n:
                pad_li = torch.full((max_n - li.size(0),), -1, dtype=li.dtype, device=li.device)
                li = torch.cat([li, pad_li], dim=0)
                pad_lp = torch.zeros((max_n - lp.size(0), lp.size(1), lp.size(2)), dtype=lp.dtype, device=lp.device)
                lp = torch.cat([lp, pad_lp], dim=0)
                pad_ls = torch.zeros((max_n - ls.size(0), ls.size(1)), dtype=ls.dtype, device=ls.device)
                ls = torch.cat([ls, pad_ls], dim=0)

            ap = self.all_gather(lp)  # [world, max_n, 3, 6640]
            ai = self.all_gather(li)  # [world, max_n] — global dataset indices
            as_t = self.all_gather(ls)  # [world, max_n, 16] — symbol char tensors

            # Collect all predictions with their global dataset indices
            all_preds: List[np.ndarray] = []
            all_indices: List[int] = []
            all_symbol_chars: List[np.ndarray] = []  # [16] chars per sample
            seen: set = set()

            for w in range(world_size):
                for i in range(max_n):
                    global_idx = int(ai[w, i].item())
                    if global_idx < 0 or global_idx in seen:
                        continue
                    seen.add(global_idx)
                    all_preds.append(ap[w, i].cpu().numpy())
                    all_indices.append(global_idx)
                    all_symbol_chars.append(as_t[w, i].cpu().numpy())

            order = np.argsort(all_indices)
            preds_arr = np.stack([all_preds[j] for j in order], axis=0)
            sorted_symbol_chars = [all_symbol_chars[j] for j in order]
            # Lookup pert_ids using global indices → self._test_pert_ids has full 167-item list
            sorted_pert_ids = [self._test_pert_ids[i] for i in order]
        else:
            # Single-GPU: use tensors directly
            preds_arr = lp.cpu().numpy()
            indices_arr = li.cpu().numpy()
            symbols_arr = ls.cpu().numpy()  # [N, 16]
            order = np.argsort(indices_arr)
            preds_arr = preds_arr[order]
            sorted_symbol_chars = [symbols_arr[i] for i in order]
            sorted_pert_ids = [self._test_pert_ids[i] for i in order]

        # Decode character tensors back to symbol strings
        sorted_symbols = []
        for chars in sorted_symbol_chars:
            symbol = "".join(SYMBOL_CHAR2IDX_reverse.get(int(c), "") for c in chars)
            sorted_symbols.append(symbol.rstrip())

        if self.trainer.is_global_zero:
            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "idx": sorted_pert_ids[r],
                    "input": sorted_symbols[r],
                    "prediction": json.dumps(preds_arr[r].tolist()),
                }
                for r in range(len(sorted_pert_ids))
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

        # CosineAnnealingLR with warmup: smooth LR decay without destabilizing plateau jumps
        # T_max=80 gives sufficient coverage without decaying too fast
        # Warmup linearly increases LR over first 5 epochs for stable early training
        def lr_lambda(epoch: int) -> float:
            """Linear warmup then cosine annealing."""
            warmup = self.hparams.warmup_epochs
            if epoch < warmup:
                # Linear warmup: scale from 0 to 1 over warmup epochs
                return float(epoch + 1) / float(max(1, warmup))
            # After warmup: cosine annealing
            progress = (epoch - warmup) / max(1, self.hparams.cosine_t_max - warmup)
            progress = min(progress, 1.0)
            cosine_factor = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())
            # Scale between eta_min/head_lr and 1.0
            eta_min_ratio = self.hparams.cosine_eta_min / self.hparams.head_lr
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
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
        description="Node: AIDO.Cell-10M + LoRA (r=8, last 4 layers) + Gene Symbol CNN (64-dim) "
                    "+ MLP Head (576→320) + CosineAnnealingLR with warmup + Checkpoint Averaging"
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "data"),
    )
    p.add_argument("--micro_batch_size",        type=int,   default=8)
    p.add_argument("--global_batch_size",       type=int,   default=64)
    p.add_argument("--max_epochs",              type=int,   default=120)
    p.add_argument("--backbone_lr",             type=float, default=3e-5)
    p.add_argument("--head_lr",                 type=float, default=3e-4)    # REDUCED from 5e-4
    p.add_argument("--weight_decay",            type=float, default=0.02)
    p.add_argument("--dropout",                 type=float, default=0.25)
    p.add_argument("--mlp_hidden",              type=int,   default=320)     # REDUCED from 512
    p.add_argument("--lora_r",                  type=int,   default=8)
    p.add_argument("--lora_alpha",              type=int,   default=16)
    p.add_argument("--lora_dropout",            type=float, default=0.15)
    p.add_argument("--symbol_cnn_dim",          type=int,   default=64)
    p.add_argument("--symbol_cnn_dropout",      type=float, default=0.3)    # INCREASED from 0.2
    p.add_argument("--focal_gamma",             type=float, default=2.0)
    p.add_argument("--label_smoothing",         type=float, default=0.05)
    p.add_argument("--cosine_t_max",            type=int,   default=80,
                   help="CosineAnnealingLR T_max (annealing period in epochs)")
    p.add_argument("--cosine_eta_min",          type=float, default=1e-7,
                   help="CosineAnnealingLR minimum LR")
    p.add_argument("--warmup_epochs",           type=int,   default=5,
                   help="Linear warmup epochs before cosine annealing")
    p.add_argument("--early_stopping_patience", type=int,   default=30)
    p.add_argument("--avg_top_k",               type=int,   default=3,
                   help="Number of top checkpoints to average for test inference (0 disables)")
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

    # Save top-3 checkpoints to enable post-training weight averaging
    save_top_k = args.avg_top_k if args.avg_top_k > 0 else 1
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node-aido10m-symcnn64-320head-cosine-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=save_top_k,  # Save top-3 for checkpoint averaging
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
        cosine_t_max=args.cosine_t_max,
        cosine_eta_min=args.cosine_eta_min,
        warmup_epochs=args.warmup_epochs,
    )

    trainer.fit(model_module, datamodule=datamodule)

    # ── Checkpoint Averaging (top-k weight averaging for test inference) ───
    # After training, average the top-k checkpoints by val_f1 to smooth
    # the weight-space noise from the val_f1 plateau oscillation.
    # Falls back to best single checkpoint if averaging is disabled or unavailable.
    use_averaged = False
    if (
        not args.fast_dev_run
        and args.debug_max_step is None
        and args.avg_top_k >= 2
        and trainer.is_global_zero
    ):
        ckpt_dir = Path(output_dir / "checkpoints")
        # Find all non-last checkpoints sorted by val_f1 (descending)
        ckpt_files = [
            f for f in sorted(ckpt_dir.glob("*.ckpt"))
            if "last" not in f.name and "val_f1=" in f.name
        ]
        # Sort by val_f1 value extracted from filename
        def extract_f1(p: Path) -> float:
            try:
                return float(p.stem.split("val_f1=")[-1])
            except ValueError:
                return 0.0

        ckpt_files_sorted = sorted(ckpt_files, key=extract_f1, reverse=True)
        top_k_ckpts = ckpt_files_sorted[:args.avg_top_k]

        if len(top_k_ckpts) >= 2:
            print(f"\n[CheckpointAvg] Averaging top-{len(top_k_ckpts)} checkpoints:")
            for p in top_k_ckpts:
                print(f"  - {p.name}")
            avg_sd = average_checkpoints(top_k_ckpts)
            if avg_sd:
                model_module.load_state_dict(avg_sd)
                # Save averaged checkpoint for reproducibility
                avg_ckpt_path = ckpt_dir / "averaged_top_k.ckpt"
                torch.save({"state_dict": avg_sd}, str(avg_ckpt_path))
                print(f"[CheckpointAvg] Averaged checkpoint saved → {avg_ckpt_path}")
                use_averaged = True
        else:
            print(f"[CheckpointAvg] Only {len(top_k_ckpts)} checkpoint(s) found; "
                  f"skipping averaging (need ≥2)")

    # ── Test inference ─────────────────────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    elif use_averaged:
        # Use the averaged model (already loaded into model_module)
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(
            model_module, datamodule=datamodule, ckpt_path="best"
        )

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        if checkpoint_cb.best_model_score is not None:
            primary_val = float(checkpoint_cb.best_model_score)
            score_line = f"best_val_f1={primary_val:.6f}"
        elif args.fast_dev_run or args.debug_max_step is not None:
            score_line = "best_val_f1=nan (debug mode - no checkpoint saved)"
        else:
            score_line = "best_val_f1=nan (no checkpoint available)"
        avg_note = " (checkpoint-averaged)" if use_averaged else " (best single checkpoint)"
        score_path.write_text(
            f"{score_line}\n"
            f"inference_mode={avg_note.strip()}\n"
            f"test_results={json.dumps(test_results)}\n"
        )
        print(f"[Score] Saved test score → {score_path}")


if __name__ == "__main__":
    main()
