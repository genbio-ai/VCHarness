#!/usr/bin/env python3
"""
Node 1-2-1-1-2: Quad Fusion + EMA + ReduceLROnPlateau(patience=12) + Three-Stage Head + Fixed Sync
==================================================================================================
Key improvements over parent (node1-2-1-1, official test F1=0.4214):

  1. EXPONENTIAL MOVING AVERAGE (EMA) of model weights:
     PyTorch's built-in EMA (torch.optim.swa_utils.AveragedModel) smooths weight
     trajectories during training. EMA weights are used at test time, providing
     a smoother representation of the learned function than any single checkpoint.
     EMA decay = 0.9995 (slow decay = preserves long-term average better).

  2. ReduceLROnPlateau with patience=12 (vs parent's cosine that didn't decay):
     The parent's cosine schedule was only at ~1.5e-4 at epoch 77 — effectively
     constant LR throughout. Sibling's patience=7 never triggered because val_f1
     improved in small increments (0.001–0.005) every few epochs. Patience=12 is
     long enough to tolerate val_f1 oscillation while still triggering on true
     convergence plateau. Factor=0.5, min_lr=1e-7.

  3. THREE-STAGE HEAD maintained: 896→512→256→19920 (matching parent, NOT sibling):
     Sibling (node1-2-1-1-1) simultaneously simplified head (two-stage) AND reduced LR
     (1e-4 vs 2e-4), causing underfitting (0.4175 regression). This node keeps the
     three-stage head (proven to not overfit with frozen backbone) and restores LR=2e-4.

  4. FIXED val_f1 sync_dist issue:
     Parent's feedback identified the val_f1=0.4373 vs test_f1=0.4214 discrepancy
     likely due to DDP sync_dist=False logging producing rank-local metrics. This node
     uses a proper all_gather approach with deduplication for val_f1 calculation,
     ensuring the logged val_f1 accurately reflects global performance.

  5. LR = 2e-4 (restored from sibling's underfitting-causing 1e-4):
     Sibling's feedback confirmed that 1e-4 combined with simplified head caused
     underfitting. Since we keep three-stage head (parent configuration), we also
     restore parent's LR=2e-4 which was the successful optimization driver.

  6. Class weights [4.0, 1.0, 8.0] and label smoothing 0.08 maintained:
     Inherited from parent (node1-2-1-1). Sibling's reversion to [3,1,7] and 0.05
     was part of its underfitting package — not independently validated.

  7. WARMUP: 3-epoch linear warmup (1e-6→2e-4) before ReduceLROnPlateau takes over.
     Short warmup avoids large initial gradient instability without consuming too
     many epochs before ReduceLROnPlateau begins monitoring.

Architecture summary (same as parent, no architectural changes):
  ┌─ AIDO.Cell-10M FROZEN ─────────────────────────────────────────────────────┐
  │  dual pool: [gene_pos_emb(256)] + [mean_pool(256)] = 512-dim              │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌─ STRING_GNN FROZEN (static, pre-computed) ─────────────────────────────────┐
  │  lookup: string_static_embs[node_idx] → 256-dim                           │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌─ Char-CNN on pert_id (Ensembl ID) ─────────────────────────────────────────┐
  │  Conv1d(k=3,5,7) → Max pool → fc → 64-dim (trainable)                    │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌─ Char-CNN on symbol (gene name) ───────────────────────────────────────────┐
  │  Conv1d(k=3,5,7) → Max pool → fc → 64-dim (trainable)                    │
  └────────────────────────────────────────────────────────────────────────────┘
              ↓ cat([512, 256, 64, 64]) = 896-dim
  LayerNorm(896)
  → Linear(896, 512) → GELU → Dropout(0.4)
  → LayerNorm(512)
  → Linear(512, 256) → GELU → Dropout(0.4)
  → LayerNorm(256)
  → Linear(256, 3×6640)
  → [B, 3, 6640] logits

EMA wraps the QuadFeatureFusionModel for test inference.

Trainable params ≈ 5.72M (same architecture as parent node1-2-1-1).
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
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
N_GENES = 6_640
N_CLASSES = 3

AIDO_MODEL_PATH = "/home/Models/AIDO.Cell-10M"
STRING_GNN_PATH = "/home/Models/STRING_GNN"

AIDO_DIM = 256                               # AIDO.Cell-10M hidden dim
STRING_DIM = 256                             # STRING_GNN output dim
DUAL_POOL_DIM = AIDO_DIM * 2                # 512 (gene_pos + mean_pool)
CNN_DIM = 64                                # per-CNN output dim (for EACH CNN)
FUSION_DIM = DUAL_POOL_DIM + STRING_DIM + CNN_DIM + CNN_DIM  # 512+256+64+64=896

# Class weights: down-regulated / unchanged / up-regulated
# Inherited from parent node1-2-1-1 (confirmed better than sibling's [3,1,7])
CLASS_WEIGHTS = torch.tensor([4.0, 1.0, 8.0], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weighting and label smoothing."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [N, C]  float32
        targets: [N]     int64
        """
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none").detach())
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper (mirrors calc_metric.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """
    y_pred:          [N, 3, G]  float  (probabilities or logits)
    y_true_remapped: [N, G]     int    ({0, 1, 2} after +1 remap)
    Returns: macro F1 averaged over G genes.
    """
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    y_hat = y_pred.argmax(axis=1)  # [N, G]
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp = y_hat[:, g]
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ─────────────────────────────────────────────────────────────────────────────
# STRING_GNN Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_string_gnn(model_path: str = STRING_GNN_PATH):
    """
    Load STRING_GNN model and graph data.
    Returns: gnn_model, edge_index, edge_weight, node_name_to_idx
    """
    import json as _json
    model_dir = Path(model_path)

    node_names = _json.loads((model_dir / "node_names.json").read_text())
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

    gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    gnn_model.eval()
    for param in gnn_model.parameters():
        param.requires_grad = False

    graph = torch.load(str(model_dir / "graph_data.pt"), map_location="cpu")
    edge_index = graph["edge_index"]
    edge_weight = graph.get("edge_weight", None)

    n_nodes = len(node_names)
    print(f"STRING_GNN loaded: {n_nodes} nodes, "
          f"{edge_index.shape[1]} directed edges, "
          f"{sum(p.numel() for p in gnn_model.parameters()):,} params (all frozen)")

    return gnn_model, edge_index, edge_weight, node_name_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# Character-level CNN for gene identifiers (shared architecture, separate instances)
# ─────────────────────────────────────────────────────────────────────────────
class GeneCharCNN(nn.Module):
    """
    Generic character-level CNN encoder for gene identifiers.

    Two instances are used:
      1. PertCNN: encodes Ensembl gene IDs (pert_id) — vocab='ENSG0123456789'
      2. SymCNN:  encodes gene symbols (symbol column) — vocab=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.

    Multi-scale architecture: 3-branch Conv1d (k=3,5,7) → max pool → 64-dim fc.
    """

    def __init__(self, vocab: str, pad_idx: int, max_len: int,
                 out_dim: int = 64, embed_dim: int = 8):
        super().__init__()
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.max_len = max_len

        vocab_size = len(vocab) + 1  # +1 for pad
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.conv1 = nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, 32, kernel_size=7, padding=3)
        # Use explicit max pooling (deterministic) instead of AdaptiveMaxPool1d (non-deterministic on CUDA)
        self.fc = nn.Linear(96, out_dim)

    def encode_str(self, s: str) -> List[int]:
        """Encode a string as padded character indices."""
        ids = []
        for c in s[:self.max_len]:
            idx = self.vocab.find(c)
            ids.append(idx if idx >= 0 else self.pad_idx)
        while len(ids) < self.max_len:
            ids.append(self.pad_idx)
        return ids

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        char_ids: [B, max_len] int64
        Returns:  [B, out_dim] float32
        """
        x = self.embed(char_ids).permute(0, 2, 1)          # [B, embed_dim, max_len]
        # Explicit max pooling (deterministic, equivalent to AdaptiveMaxPool1d(1))
        b1 = F.gelu(self.conv1(x)).max(dim=-1)[0]          # [B, 32]
        b2 = F.gelu(self.conv2(x)).max(dim=-1)[0]          # [B, 32]
        b3 = F.gelu(self.conv3(x)).max(dim=-1)[0]          # [B, 32]
        out = torch.cat([b1, b2, b3], dim=-1)               # [B, 96]
        return F.gelu(self.fc(out))                          # [B, out_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary constants for the two char-CNNs
# ─────────────────────────────────────────────────────────────────────────────
PERT_VOCAB = 'ENSG0123456789'       # 14 unique chars in Ensembl IDs
PERT_PAD_IDX = 14
PERT_MAX_LEN = 15

# Gene symbol vocabulary: uppercase letters, digits, hyphens (e.g., "GCLC", "MYC", "H3F3C")
SYM_VOCAB = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.'
SYM_PAD_IDX = len(SYM_VOCAB)   # 38
SYM_MAX_LEN = 16                # most gene symbols are <=16 chars


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Dataset with AIDO tokenized inputs, STRING indices, and dual char-CNN inputs."""

    def __init__(
        self,
        df: pd.DataFrame,
        input_ids: torch.Tensor,              # [N, 19264] float32
        pert_positions: torch.Tensor,         # [N] int64 (-1 if unknown)
        string_node_indices: torch.Tensor,    # [N] int64 (-1 if unknown)
        pert_char_ids: torch.Tensor,          # [N, PERT_MAX_LEN] int64
        sym_char_ids: torch.Tensor,           # [N, SYM_MAX_LEN] int64
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.input_ids = input_ids
        self.pert_positions = pert_positions
        self.string_node_indices = string_node_indices
        self.pert_char_ids = pert_char_ids
        self.sym_char_ids = sym_char_ids
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            arr = np.array(raw_labels, dtype=np.int8) + 1   # {-1,0,1} → {0,1,2}
            self.labels = torch.from_numpy(arr).long()       # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "input_ids": self.input_ids[idx],
            "pert_pos": self.pert_positions[idx],
            "string_node_idx": self.string_node_indices[idx],
            "pert_char_ids": self.pert_char_ids[idx],
            "sym_char_ids": self.sym_char_ids[idx],
        }
        if not self.is_test:
            item["label"] = self.labels[idx]
        return item


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []
        self.test_labels: Optional[torch.Tensor] = None  # Loaded from CSV for metric computation
        self._string_node_to_idx: Optional[Dict[str, int]] = None

    def _init_tokenizer(self) -> AutoTokenizer:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)

    def _init_string_vocab(self) -> None:
        if self._string_node_to_idx is not None:
            return
        import json as _json
        node_names = _json.loads((Path(STRING_GNN_PATH) / "node_names.json").read_text())
        self._string_node_to_idx = {name: i for i, name in enumerate(node_names)}

    def _tokenize_and_get_positions(
        self,
        tokenizer: AutoTokenizer,
        pert_ids: List[str],
        split_name: str = "split",
    ) -> tuple:
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        chunk_size = 128
        all_input_ids: List[torch.Tensor] = []
        for i in range(0, len(expr_dicts), chunk_size):
            chunk = expr_dicts[i:i + chunk_size]
            toks = tokenizer(chunk, return_tensors="pt")
            all_input_ids.append(toks["input_ids"])
        input_ids = torch.cat(all_input_ids, dim=0)   # [N, 19264] float32
        non_missing = input_ids > -0.5
        has_gene = non_missing.any(dim=1)
        pert_positions = non_missing.long().argmax(dim=1)
        pert_positions[~has_gene] = -1
        coverage = 100.0 * has_gene.float().mean().item()
        print(f"  [{split_name}] AIDO vocab coverage: "
              f"{has_gene.sum().item()}/{len(pert_ids)} ({coverage:.1f}%)")
        return input_ids, pert_positions

    def _get_string_node_indices(
        self,
        pert_ids: List[str],
        split_name: str = "split",
    ) -> torch.Tensor:
        indices = []
        n_found = 0
        for pid in pert_ids:
            idx = self._string_node_to_idx.get(pid, -1)
            indices.append(idx)
            if idx >= 0:
                n_found += 1
        coverage = 100.0 * n_found / len(pert_ids)
        print(f"  [{split_name}] STRING vocab coverage: "
              f"{n_found}/{len(pert_ids)} ({coverage:.1f}%)")
        return torch.tensor(indices, dtype=torch.long)

    @staticmethod
    def _encode_pert_char_ids(pert_ids: List[str]) -> torch.Tensor:
        """Encode Ensembl gene IDs: [N, PERT_MAX_LEN] int64."""
        result = []
        for pid in pert_ids:
            ids = []
            for c in pid[:PERT_MAX_LEN]:
                idx = PERT_VOCAB.find(c)
                ids.append(idx if idx >= 0 else PERT_PAD_IDX)
            while len(ids) < PERT_MAX_LEN:
                ids.append(PERT_PAD_IDX)
            result.append(ids)
        return torch.tensor(result, dtype=torch.long)

    @staticmethod
    def _encode_sym_char_ids(symbols: List[str]) -> torch.Tensor:
        """Encode gene symbols (upper-cased): [N, SYM_MAX_LEN] int64."""
        result = []
        for sym in symbols:
            sym_upper = sym.upper()
            ids = []
            for c in sym_upper[:SYM_MAX_LEN]:
                idx = SYM_VOCAB.find(c)
                ids.append(idx if idx >= 0 else SYM_PAD_IDX)
            while len(ids) < SYM_MAX_LEN:
                ids.append(SYM_PAD_IDX)
            result.append(ids)
        return torch.tensor(result, dtype=torch.long)

    def _prepare_split(
        self,
        tokenizer: AutoTokenizer,
        df: pd.DataFrame,
        split_name: str,
        is_test: bool = False,
    ) -> PerturbationDataset:
        pert_ids = df["pert_id"].tolist()
        symbols = df["symbol"].tolist()

        input_ids, pert_positions = self._tokenize_and_get_positions(
            tokenizer, pert_ids, split_name)
        string_node_indices = self._get_string_node_indices(pert_ids, split_name)
        pert_char_ids = self._encode_pert_char_ids(pert_ids)
        sym_char_ids = self._encode_sym_char_ids(symbols)

        return PerturbationDataset(
            df, input_ids, pert_positions, string_node_indices,
            pert_char_ids, sym_char_ids, is_test=is_test,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self._init_string_vocab()
        tokenizer = self._init_tokenizer()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")

            print("Preparing train set...")
            self.train_ds = self._prepare_split(tokenizer, train_df, "train", is_test=False)
            print("Preparing val set...")
            self.val_ds = self._prepare_split(tokenizer, val_df, "val", is_test=False)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            print("Preparing test set...")
            self.test_ds = self._prepare_split(tokenizer, test_df, "test", is_test=True)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()
            # Load test labels from CSV for F1 metric computation (test.tsv has labels)
            if "label" in test_df.columns:
                raw_labels = [json.loads(x) for x in test_df["label"].tolist()]
                arr = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1} → {0,1,2}
                self.test_labels = torch.from_numpy(arr).long()  # [N_test, 6640]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model: Quad Feature Fusion (AIDO + STRING + PertCNN + SymCNN)
# ─────────────────────────────────────────────────────────────────────────────
class QuadFeatureFusionModel(nn.Module):
    """
    Four-stream feature fusion:
      1. AIDO.Cell-10M (frozen, dual pool): 512-dim
      2. STRING_GNN (frozen static): 256-dim
      3. Char-CNN on pert_id (Ensembl ID): 64-dim
      4. Char-CNN on gene symbol: 64-dim

    Three-stage head: 896→512→256→19920 (same as parent node1-2-1-1)
    """

    def __init__(self, head_width: int = 256, head_mid: int = 512, head_dropout: float = 0.4):
        super().__init__()
        self.head_width = head_width
        self.head_mid = head_mid
        self.head_dropout = head_dropout

        # AIDO backbone (set in initialize_aido)
        self.aido_backbone: Optional[nn.Module] = None

        # STRING backbone (set in initialize_string_gnn)
        self.string_gnn: Optional[nn.Module] = None
        self._n_nodes: int = 18870

        # Learnable fallback for genes not in STRING vocab
        self.string_null_emb = nn.Parameter(torch.zeros(STRING_DIM))

        # Char-CNN for Ensembl ID (pert_id)
        self.pert_cnn = GeneCharCNN(
            vocab=PERT_VOCAB,
            pad_idx=PERT_PAD_IDX,
            max_len=PERT_MAX_LEN,
            out_dim=CNN_DIM,
            embed_dim=8,
        )

        # Char-CNN for gene symbol
        self.sym_cnn = GeneCharCNN(
            vocab=SYM_VOCAB,
            pad_idx=SYM_PAD_IDX,
            max_len=SYM_MAX_LEN,
            out_dim=CNN_DIM,
            embed_dim=8,
        )

        # Output head (set in initialize_head)
        self.head: Optional[nn.Sequential] = None

    def initialize_aido(self) -> None:
        """Load AIDO.Cell-10M (frozen) on GPU."""
        self.aido_backbone = AutoModel.from_pretrained(
            AIDO_MODEL_PATH, trust_remote_code=True)
        self.aido_backbone = self.aido_backbone.to(torch.bfloat16)
        self.aido_backbone.config.use_cache = False
        for param in self.aido_backbone.parameters():
            param.requires_grad = False
        total = sum(p.numel() for p in self.aido_backbone.parameters())
        print(f"AIDO.Cell-10M: {total:,} params (all frozen)")

    def initialize_string_gnn(self) -> None:
        """
        Load STRING_GNN (frozen), run ONE forward pass to pre-compute
        static topology embeddings as buffer.
        """
        gnn_model, edge_index, edge_weight, _ = load_string_gnn()

        self.register_buffer("_edge_index_buf", edge_index)
        if edge_weight is not None:
            self.register_buffer("_edge_weight_buf", edge_weight)

        self._n_nodes = 18870

        gnn_model.eval()
        with torch.no_grad():
            out = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
        static_embs = out.last_hidden_state.float()   # [18870, 256]
        self.register_buffer("string_static_embs", static_embs)
        self.string_gnn = gnn_model
        print(f"STRING_GNN static embeddings pre-computed: {static_embs.shape}")

    def initialize_head(self) -> None:
        """Three-stage output head (trainable) — same as parent node1-2-1-1."""
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),                            # 896
            nn.Linear(FUSION_DIM, self.head_mid),                # 896 → 512
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.LayerNorm(self.head_mid),                         # 512
            nn.Linear(self.head_mid, self.head_width),           # 512 → 256
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.LayerNorm(self.head_width),                       # 256
            nn.Linear(self.head_width, N_CLASSES * N_GENES),     # 256 → 3*6640
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.string_null_emb.data)

    def _get_aido_dual_pool(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32
        pert_positions: torch.Tensor,  # [B] int64 (-1 for unknown)
    ) -> torch.Tensor:
        """Returns dual_pool: [B, 512] = cat([gene_pos_emb(256), mean_pool(256)])."""
        backbone_device = next(self.aido_backbone.parameters()).device
        input_ids_dev = input_ids.to(backbone_device)
        attn_mask = torch.ones(
            input_ids_dev.shape[0], input_ids_dev.shape[1],
            dtype=torch.long, device=backbone_device,
        )
        with torch.no_grad():
            out = self.aido_backbone(
                input_ids=input_ids_dev,
                attention_mask=attn_mask,
            )
        hidden = out.last_hidden_state   # [B, 19266, 256] bfloat16

        mean_pool = hidden[:, :19264, :].mean(dim=1).float()   # [B, 256]

        B = hidden.size(0)
        hidden_device = hidden.device
        pert_positions_dev = pert_positions.to(hidden_device)
        valid = pert_positions_dev >= 0
        safe_pos = pert_positions_dev.clamp(min=0)
        gene_emb_raw = hidden[
            torch.arange(B, device=hidden_device), safe_pos, :
        ].float()

        valid_f = valid.float().unsqueeze(-1)
        gene_emb = gene_emb_raw * valid_f + mean_pool * (1.0 - valid_f)

        dual_pool = torch.cat([gene_emb, mean_pool], dim=-1)   # [B, 512]
        return dual_pool

    def forward(
        self,
        input_ids: torch.Tensor,            # [B, 19264] float32
        pert_positions: torch.Tensor,       # [B] int64
        string_node_indices: torch.Tensor,  # [B] int64 (-1 for STRING unknown)
        pert_char_ids: torch.Tensor,        # [B, PERT_MAX_LEN] int64
        sym_char_ids: torch.Tensor,         # [B, SYM_MAX_LEN] int64
    ) -> torch.Tensor:
        """Returns: [B, 3, N_GENES] logits."""
        B = input_ids.size(0)

        # ── AIDO.Cell-10M dual-pool ───────────────────────────────────────────
        aido_dual = self._get_aido_dual_pool(input_ids, pert_positions)  # [B, 512]

        # ── STRING static lookup + null fallback ─────────────────────────────
        str_device = self.string_static_embs.device
        valid = (string_node_indices >= 0)
        safe_idx = string_node_indices.clamp(min=0).to(str_device)
        str_feat = self.string_static_embs[safe_idx]              # [B, 256]
        null_emb = self.string_null_emb.to(str_device)
        valid_f = valid.float().to(str_device).unsqueeze(-1)
        str_feat = str_feat * valid_f + null_emb.unsqueeze(0) * (1.0 - valid_f)

        # ── Char-CNN for Ensembl ID ───────────────────────────────────────────
        cnn_device = next(self.pert_cnn.parameters()).device
        pert_feat = self.pert_cnn(pert_char_ids.to(cnn_device))   # [B, 64]

        # ── Char-CNN for gene symbol ─────────────────────────────────────────
        sym_feat = self.sym_cnn(sym_char_ids.to(cnn_device))      # [B, 64]

        # ── Fuse on head device ──────────────────────────────────────────────
        head_device = next(self.head.parameters()).device
        combined = torch.cat([
            aido_dual.to(head_device),
            str_feat.to(head_device),
            pert_feat.to(head_device),
            sym_feat.to(head_device),
        ], dim=-1)   # [B, 896]

        logits = self.head(combined)                    # [B, 3*6640]
        return logits.view(B, N_CLASSES, N_GENES)       # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average) Callback
# ─────────────────────────────────────────────────────────────────────────────
class EMACallback(pl.Callback):
    """
    Maintains an EMA copy of the trainable model parameters during training.
    At test time, the EMA weights are swapped in for inference.

    EMA update rule:
        ema_param = decay * ema_param + (1 - decay) * param

    High decay (0.9995) means the EMA changes slowly, smoothing out noise
    in the optimization trajectory. At test time, EMA weights represent
    the smoothed average of recent parameter values.

    Why EMA instead of checkpoint averaging:
    - EMA is computed continuously during training, not just at checkpoint saves
    - EMA weights represent a smoother average with more diversity than top-K
      checkpoints which are often near-identical at convergence
    - EMA has no overhead on checkpoint saving/loading logic
    - The spread of checkpoints was only 0.001 F1 in sibling — EMA captures
      a much broader window of the optimization trajectory
    """

    def __init__(self, decay: float = 0.9995):
        super().__init__()
        self.decay = decay
        self._ema_params: Optional[Dict[str, torch.Tensor]] = None
        self._original_params: Optional[Dict[str, torch.Tensor]] = None

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: LightningModule,
        outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Update EMA after each training batch."""
        if self._ema_params is None:
            # Initialize EMA with current parameter values
            self._ema_params = {}
            for name, param in pl_module.model.named_parameters():
                if param.requires_grad:
                    self._ema_params[name] = param.data.clone().detach()
        else:
            # Exponential moving average update
            decay = self.decay
            for name, param in pl_module.model.named_parameters():
                if param.requires_grad and name in self._ema_params:
                    self._ema_params[name].mul_(decay).add_(
                        param.data.detach(), alpha=1.0 - decay
                    )

    def on_test_start(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        """Swap in EMA weights before test inference."""
        if self._ema_params is None:
            return
        # Save original parameters
        self._original_params = {}
        for name, param in pl_module.model.named_parameters():
            if param.requires_grad and name in self._ema_params:
                self._original_params[name] = param.data.clone()
                param.data.copy_(self._ema_params[name])
        print("EMA: swapped in EMA weights for test inference.")

    def on_test_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        """Restore original weights after test inference."""
        if self._original_params is None:
            return
        for name, param in pl_module.model.named_parameters():
            if param.requires_grad and name in self._original_params:
                param.data.copy_(self._original_params[name])
        self._original_params = None
        print("EMA: restored original weights after test inference.")


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler: linear warmup then ReduceLROnPlateau
# ─────────────────────────────────────────────────────────────────────────────
class WarmupReduceLROnPlateau:
    """
    Combines short linear warmup with ReduceLROnPlateau.

    During warmup phase (first warmup_epochs epochs): LR linearly scales from
    lr_start to lr_peak. After warmup, hands off to ReduceLROnPlateau behavior
    which reduces LR on plateau detection.

    Since ReduceLROnPlateau doesn't support warmup natively, we use a LambdaLR
    for warmup, then switch to a custom ReduceLROnPlateau after warmup completes.
    """
    pass  # Implemented directly in configure_optimizers


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        head_width: int = 256,
        head_mid: int = 512,
        head_dropout: float = 0.4,
        lr: float = 2e-4,
        lr_patience: int = 12,
        lr_factor: float = 0.5,
        lr_min: float = 1e-7,
        warmup_epochs: int = 3,
        weight_decay: float = 0.05,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.08,
        max_epochs: int = 150,
        ema_decay: float = 0.9995,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model: Optional[QuadFeatureFusionModel] = None
        self.criterion: Optional[FocalLoss] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

        # Warmup tracking
        self._in_warmup: bool = True
        self._warmup_optimizer: Optional[torch.optim.Optimizer] = None
        self._plateau_scheduler = None
        self._current_epoch_for_warmup: int = 0

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = QuadFeatureFusionModel(
                head_width=self.hparams.head_width,
                head_mid=self.hparams.head_mid,
                head_dropout=self.hparams.head_dropout,
            )
            self.model.initialize_aido()
            self.model.initialize_string_gnn()
            self.model.initialize_head()

            # Cast trainable parameters to float32 for stable optimization
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()

            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            self.print(f"Trainable params: {trainable:,} / {total:,} "
                       f"({100.0 * trainable / max(total, 1):.2f}%)")
            self.print(f"Trainable params per training sample: {trainable / 1500:.0f}")

        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def forward(
        self,
        input_ids: torch.Tensor,
        pert_positions: torch.Tensor,
        string_node_indices: torch.Tensor,
        pert_char_ids: torch.Tensor,
        sym_char_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, pert_positions, string_node_indices,
                          pert_char_ids, sym_char_ids)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G], labels: [B, G] ({0,1,2}) → scalar loss."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)   # [B*G, 3]
        labels_flat = labels.reshape(-1)                        # [B*G]
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["input_ids"], batch["pert_pos"],
            batch["string_node_idx"], batch["pert_char_ids"], batch["sym_char_ids"],
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["pert_pos"],
            batch["string_node_idx"], batch["pert_char_ids"], batch["sym_char_ids"],
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()   # [B, 3, G]
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, dim=0)
        local_labels = torch.cat(self._val_labels, dim=0)
        local_idx = torch.cat(self._val_indices, dim=0)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        world_size = self.trainer.world_size if self.trainer.world_size else 1
        if world_size > 1:
            all_preds = self.all_gather(local_preds)
            all_labels = self.all_gather(local_labels)
            all_idx = self.all_gather(local_idx)

            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            labels_flat = all_labels.view(-1, N_GENES).cpu().numpy()
            idx_flat = all_idx.view(-1).cpu().numpy()

            unique_pos = np.unique(idx_flat, return_index=True)[1]
            preds_flat = preds_flat[unique_pos]
            labels_flat = labels_flat[unique_pos]
            order = np.argsort(idx_flat[unique_pos])
            preds_flat = preds_flat[order]
            labels_flat = labels_flat[order]
            f1 = compute_deg_f1(preds_flat, labels_flat)
            # Use sync_dist=True to ensure val_f1 is globally consistent in DDP mode
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        else:
            f1 = compute_deg_f1(local_preds.numpy(), local_labels.numpy())
            self.log("val_f1", f1, prog_bar=True, sync_dist=False)

        # Manually step the ReduceLROnPlateau scheduler based on val_f1
        # This is needed because Lightning's ReduceLROnPlateau requires explicit val metric
        if self._plateau_scheduler is not None and not self._in_warmup:
            self._plateau_scheduler.step(f1)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["pert_pos"],
            batch["string_node_idx"], batch["pert_char_ids"], batch["sym_char_ids"],
        )
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)
        local_idx = torch.cat(self._test_indices, dim=0)

        all_preds = self.all_gather(local_preds)
        all_idx = self.all_gather(local_idx)

        self._test_preds.clear()
        self._test_indices.clear()

        if self.trainer.is_global_zero:
            preds = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            idxs = all_idx.view(-1).cpu().numpy()

            unique_pos = np.unique(idxs, return_index=True)[1]
            preds = preds[unique_pos]
            sorted_idxs = idxs[unique_pos]

            order = np.argsort(sorted_idxs)
            preds = preds[order]
            final_idxs = sorted_idxs[order]

            # Compute test F1 using labels from datamodule (loaded from test CSV)
            test_f1 = None
            dm = self.trainer.datamodule
            if hasattr(dm, "test_labels") and dm.test_labels is not None:
                labels = dm.test_labels.numpy()  # [N_test, 6640]
            else:
                # Reload test labels from CSV directly
                test_df = pd.read_csv(dm.data_dir / "test.tsv", sep="\t")
                raw_labels = [json.loads(x) for x in test_df["label"].tolist()]
                labels = (np.array(raw_labels, dtype=np.int8) + 1).astype(np.int64)
            test_f1 = compute_deg_f1(preds, labels)
            self.log("test_f1", test_f1, prog_bar=True, sync_dist=False)

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"

            rows = []
            for rank_i, orig_i in enumerate(final_idxs):
                rows.append({
                    "idx": self._test_pert_ids[orig_i],
                    "input": self._test_symbols[orig_i],
                    "prediction": json.dumps(preds[rank_i].tolist()),
                })
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path}")
            if test_f1 is not None:
                self.print(f"Test F1: {test_f1:.4f}")

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]

        # Start with a very low LR; manual warmup will scale it up over first warmup_epochs epochs
        opt = torch.optim.AdamW(
            trainable,
            lr=1e-8,   # Start near zero; warmup in on_train_epoch_end will scale to lr_peak
            weight_decay=self.hparams.weight_decay,
        )

        # Store optimizer reference for manual LR management
        self._warmup_optimizer = opt

        # ReduceLROnPlateau for post-warmup adaptive decay.
        # We do NOT register it with Lightning (would conflict with manual warmup).
        # Instead it is stepped manually in on_validation_epoch_end.
        # IMPORTANT: ReduceLROnPlateau initialized with the peak LR as the optimizer's current lr.
        # We will set optimizer lr to lr_peak at the end of warmup, then let ReduceLROnPlateau
        # track subsequent reductions.
        self._plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",  # maximize val_f1
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min,
        )

        # Return only optimizer; no Lightning-managed scheduler to avoid LR conflicts.
        # All LR updates are handled manually in on_train_epoch_end and on_validation_epoch_end.
        return opt

    def on_train_epoch_end(self) -> None:
        """Manage linear warmup LR schedule manually."""
        current_epoch = self.current_epoch
        warmup_epochs = self.hparams.warmup_epochs
        lr_peak = self.hparams.lr
        lr_start = 1e-6

        if current_epoch < warmup_epochs:
            # Still in warmup: linearly interpolate LR from lr_start to lr_peak
            # current_epoch=0 → lr at end of epoch 0 = lr_start + (lr_peak-lr_start) * 1/warmup_epochs
            warmup_lr = lr_start + (lr_peak - lr_start) * (current_epoch + 1) / warmup_epochs
            if self._warmup_optimizer is not None:
                for pg in self._warmup_optimizer.param_groups:
                    pg["lr"] = warmup_lr
            self.log("lr", warmup_lr, prog_bar=False, sync_dist=True)
        elif current_epoch == warmup_epochs:
            # First post-warmup epoch: set to lr_peak and enable ReduceLROnPlateau
            if self._warmup_optimizer is not None:
                for pg in self._warmup_optimizer.param_groups:
                    pg["lr"] = lr_peak
            self._in_warmup = False
            self.log("lr", lr_peak, prog_bar=False, sync_dist=True)
        else:
            # Post-warmup: log current LR (ReduceLROnPlateau may have reduced it)
            self._in_warmup = False
            if self._warmup_optimizer is not None:
                current_lr = self._warmup_optimizer.param_groups[0]["lr"]
                self.log("lr", current_lr, prog_bar=False, sync_dist=True)

    # ── Checkpoint: save only trainable parameters ────────────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                k = prefix + name
                if k in full:
                    trainable[k] = full[k]
        for name, buf in self.named_buffers():
            k = prefix + name
            if k in full:
                trainable[k] = full[k]
        tr_cnt = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        self.print(
            f"Checkpoint: saving {tr_cnt:,}/{total:,} params "
            f"({100.0 * tr_cnt / max(total, 1):.2f}% trainable)"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quad Feature Fusion + EMA + ReduceLROnPlateau(p=12) + Three-Stage Head"
    )
    p.add_argument("--data-dir",               type=str,   default="data")
    p.add_argument("--micro-batch-size",        type=int,   default=8)
    p.add_argument("--global-batch-size",       type=int,   default=64)
    p.add_argument("--max-epochs",              type=int,   default=150)
    p.add_argument("--lr",                      type=float, default=2e-4)
    p.add_argument("--lr-patience",             type=int,   default=12)
    p.add_argument("--lr-factor",               type=float, default=0.5)
    p.add_argument("--lr-min",                  type=float, default=1e-7)
    p.add_argument("--warmup-epochs",           type=int,   default=3)
    p.add_argument("--weight-decay",            type=float, default=0.05)
    p.add_argument("--head-width",              type=int,   default=256)
    p.add_argument("--head-mid",                type=int,   default=512)
    p.add_argument("--head-dropout",            type=float, default=0.4)
    p.add_argument("--gamma-focal",             type=float, default=2.0)
    p.add_argument("--label-smoothing",         type=float, default=0.08)
    p.add_argument("--early-stopping-patience", type=int,   default=25)
    p.add_argument("--ema-decay",               type=float, default=0.9995)
    p.add_argument("--num-workers",             type=int,   default=4)
    p.add_argument("--val-check-interval",      type=float, default=1.0)
    p.add_argument("--debug-max-step",          type=int,   default=None)
    p.add_argument("--fast-dev-run",            action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Distributed setup ────────────────────────────────────────────────────
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit_train = limit_val = limit_test = 1.0
    if args.debug_max_step is not None:
        limit_train = float(args.debug_max_step)
        limit_val = float(args.debug_max_step)
        limit_test = float(args.debug_max_step)

    val_check_interval = args.val_check_interval if (
        args.debug_max_step is None and not args.fast_dev_run
    ) else 1.0

    if n_gpus == 1:
        strategy: Any = SingleDeviceStrategy(device="cuda:0")
    else:
        strategy = DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(seconds=120),
        )

    # ── EMA callback ─────────────────────────────────────────────────────────
    ema_callback = EMACallback(decay=args.ema_decay)

    # ── Callbacks ────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-1-1-2-{epoch:03d}-{val_f1:.4f}",
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
    # LearningRateMonitor requires a registered scheduler; since we manage LR manually,
    # we log the LR in on_train_epoch_end instead. Keep the callback for compatibility.
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # ── Loggers ──────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=1.0,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar, ema_callback],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )
    # Ensure limit_test_batches respects debug_max_step if set, else full test set for fast_dev_run
    if args.debug_max_step is not None:
        trainer.limit_test_batches = min(float(args.debug_max_step), 1.0)
    elif fast_dev_run:
        trainer.limit_test_batches = 1.0

    # ── Data & model ─────────────────────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        head_width=args.head_width,
        head_mid=args.head_mid,
        head_dropout=args.head_dropout,
        lr=args.lr,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        ema_decay=args.ema_decay,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Test ─────────────────────────────────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # ── Save test score ──────────────────────────────────────────────────────
    if trainer.is_global_zero:
        score_path = output_dir / "test_score.txt"
        best_val_f1 = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else None
        )
        # Extract f1 from test_results
        test_f1 = None
        if test_results and len(test_results) > 0:
            for key in ("test_f1", "f1", "test_macro_f1", "macro_f1"):
                if key in test_results[0]:
                    test_f1 = float(test_results[0][key])
                    break
            if test_f1 is None:
                for k, v in test_results[0].items():
                    if isinstance(v, (int, float)) and "f1" in k.lower():
                        test_f1 = float(v)
                        break
        score_path.write_text(
            f"test_f1: {test_f1}\n"
            f"val_f1_best: {best_val_f1}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
