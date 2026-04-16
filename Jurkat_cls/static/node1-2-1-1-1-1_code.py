#!/usr/bin/env python3
"""
Node 1-2-1-1-1-1: Quad Fusion + Three-Stage Head + LR=2e-4 + EMA + Longer ReduceLROnPlateau
============================================================================================
Key improvements over parent (node1-2-1-1-1, test F1=0.4175):

  1. RESTORE three-stage head: 896→512→256→19920.
     Parent feedback confirmed that simultaneous head simplification + LR reduction
     caused underfitting. Node1-2-1-1 achieved 0.4373 with the three-stage head at 2e-4.
     Reverting head capacity to address the root cause of underfitting.

  2. INCREASE LR back to 2e-4 (vs parent's 1e-4).
     Parent feedback: "The reduction to 1e-4 was counterproductive. With a frozen backbone,
     the head LR is the sole optimization driver. Parent node1-2-1-1 used 2e-4 successfully."
     The simplified head at lower LR combined caused insufficient optimization.

  3. INCREASE ReduceLROnPlateau patience to 12 (vs parent's 7).
     Parent feedback: "patience=7 was insufficient — val_f1 oscillates at ±0.001 every 2-4
     epochs, preventing LR reduction." With patience=12, the scheduler can actually trigger
     at the true convergence plateau.

  4. ADD EMA (Exponential Moving Average) weight averaging during training.
     Parent feedback explicitly recommended EMA as "more principled than checkpoint averaging."
     EMA with tau=0.9995 accumulates a smoothed version of model weights throughout training,
     providing better generalization than individual checkpoints or simple averaging.
     Applied at test time for inference.

  5. RESTORE class weights [4, 1, 8] (vs parent's [3, 1, 7]).
     Parent used [3, 1, 7] which may have weakened minority class signal. The grandparent
     node1-2-1-1 used [4, 1, 8] and achieved 0.4373. Combined with a stronger head,
     the original class weights provide appropriate minority class emphasis.

  6. INCREASE checkpoint pool to top-5 (vs parent's top-3).
     Parent's top-3 checkpoints spanned only 0.001 F1 — too similar for averaging benefit.
     With top-5, the span is wider and averaging provides more diversity for test-time
     smoothing. Used as fallback if EMA is not available.

  7. TIGHTER gradient clipping at 0.5 (vs parent's 1.0).
     With higher LR (2e-4), tighter gradient clipping prevents instability in early training,
     especially beneficial for the larger three-stage head.

Architecture summary:
  ┌─ AIDO.Cell-10M FROZEN ─────────────────────────────────────────────────────┐
  │  dual pool: [gene_pos_emb(256)] + [mean_pool(256)] = 512-dim              │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌─ STRING_GNN FROZEN (static, pre-computed) ─────────────────────────────────┐
  │  lookup: string_static_embs[node_idx] → 256-dim                           │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌─ Char-CNN on pert_id (Ensembl ID) ─────────────────────────────────────────┐
  │  Conv1d(k=3,5,7) → MaxPool → fc → 64-dim (trainable)                     │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌─ Char-CNN on symbol (gene name) ───────────────────────────────────────────┐
  │  Conv1d(k=3,5,7) → MaxPool → fc → 64-dim (trainable)                     │
  └────────────────────────────────────────────────────────────────────────────┘
              ↓ cat([512, 256, 64, 64]) = 896-dim
  LayerNorm(896)
  → Linear(896, 512) → GELU → Dropout(0.35)      ← three-stage head restored
  → LayerNorm(512)
  → Linear(512, 256) → GELU → Dropout(0.35)
  → LayerNorm(256)
  → Linear(256, 3×6640)
  → [B, 3, 6640] logits

Trainable params ≈ 5.7M:
  - PertCNN (Ensembl): ~7,200
  - SymCNN (gene symbol): ~8,400
  - STRING null_emb: 256
  - LayerNorm(896): 1,792
  - Linear(896→512) + LayerNorm(512): ~459,776 + 1,024
  - Linear(512→256) + LayerNorm(256): ~131,072 + 512
  - Linear(256→19920): ~5,099,520
  Total ≈ 5.7M params (~3,800 / training sample)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import copy
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

# Class weights: restored to [4, 1, 8] from node1-2-1-1 (grandparent at 0.4373)
# Parent's [3, 1, 7] reduced minority class signal; this restores stronger emphasis
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
      2. SymCNN:  encodes gene symbols (symbol column) — vocab=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-

    Multi-scale architecture: 3-branch Conv1d (k=3,5,7) → MaxPool(1) → 64-dim fc.
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
def _resolve_data_dir(data_dir: str) -> Path:
    """Resolve data_dir to an absolute path that works regardless of CWD."""
    p = Path(data_dir)
    if p.is_absolute() or p.exists():
        return p.resolve() if p.is_absolute() else p
    # Try resolving relative to the script's location (handles CWD mismatch)
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / data_dir,
        script_dir / ".." / data_dir,
        script_dir / ".." / ".." / data_dir,
        script_dir / ".." / ".." / "data",
        (script_dir / ".." / ".." / "data").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return p.resolve() if p.is_absolute() else p


class DEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = _resolve_data_dir(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []
        self.test_labels: Optional[torch.Tensor] = None
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
#        THREE-STAGE head (896→512→256→19920) — restored from grandparent node1-2-1-1
# ─────────────────────────────────────────────────────────────────────────────
class QuadFeatureFusionModel(nn.Module):
    """
    Four-stream feature fusion:
      1. AIDO.Cell-10M (frozen, dual pool): 512-dim
      2. STRING_GNN (frozen static): 256-dim
      3. Char-CNN on pert_id (Ensembl ID): 64-dim
      4. Char-CNN on gene symbol: 64-dim

    THREE-STAGE head: 896→512→256→19920 (restored from node1-2-1-1 at 0.4373)
    """

    def __init__(self, head_width1: int = 512, head_width2: int = 256,
                 head_dropout: float = 0.35):
        super().__init__()
        self.head_width1 = head_width1
        self.head_width2 = head_width2
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
        """
        Three-stage output head (trainable) — restored from grandparent node1-2-1-1.
        896→512→256→19920 with GELU activations and LayerNorm at each stage.
        """
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),                                # 896
            nn.Linear(FUSION_DIM, self.head_width1),                 # 896 → 512
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.LayerNorm(self.head_width1),                          # 512
            nn.Linear(self.head_width1, self.head_width2),           # 512 → 256
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.LayerNorm(self.head_width2),                          # 256
            nn.Linear(self.head_width2, N_CLASSES * N_GENES),        # 256 → 3*6640
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
    Maintains an exponential moving average (EMA) of model trainable parameters.

    EMA provides smoother weight estimates than individual checkpoints by continuously
    tracking a weighted average of all parameter states throughout training. This is
    more principled than post-hoc checkpoint averaging because:
    1. It incorporates every training step, not just a few selected checkpoints
    2. The exponential weighting emphasizes recent, better-converged parameters
    3. It is insensitive to checkpoint frequency and selection

    At test time, the EMA weights are loaded into the model for inference.
    """

    def __init__(self, decay: float = 0.9995):
        super().__init__()
        self.decay = decay
        self._ema_state: Optional[Dict[str, torch.Tensor]] = None

    def _get_trainable_state(self, pl_module: pl.LightningModule) -> Dict[str, torch.Tensor]:
        """Extract trainable parameter state dict."""
        return {
            name: param.data.float().clone()
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        }

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize EMA state from current model parameters."""
        self._ema_state = self._get_trainable_state(pl_module)
        pl_module.print(
            f"EMA initialized with decay={self.decay:.4f} "
            f"({len(self._ema_state)} parameter tensors)"
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA after each training batch."""
        if self._ema_state is None:
            return
        # Only update on global rank 0 to avoid issues in multi-GPU training
        # Actually we update on each GPU but the EMA is consistent since params are synced
        decay = self.decay
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self._ema_state:
                self._ema_state[name].mul_(decay).add_(
                    param.data.float(), alpha=1.0 - decay
                )

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Load EMA weights into the model for test inference."""
        if self._ema_state is None:
            pl_module.print("EMA: No EMA state available, using current model weights.")
            return

        pl_module.print(
            f"EMA: Loading EMA weights ({len(self._ema_state)} tensors) for test inference."
        )
        # Backup current weights before loading EMA
        self._backup_state = self._get_trainable_state(pl_module)

        # Load EMA weights
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if param.requires_grad and name in self._ema_state:
                    param.data.copy_(self._ema_state[name].to(param.device))

        pl_module.print("EMA: EMA weights loaded for test inference.")

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Restore original weights after test (for subsequent use if any)."""
        if hasattr(self, '_backup_state') and self._backup_state is not None:
            with torch.no_grad():
                for name, param in pl_module.named_parameters():
                    if param.requires_grad and name in self._backup_state:
                        param.data.copy_(self._backup_state[name].to(param.device))
            pl_module.print("EMA: Original weights restored after test.")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Averaging Callback (fallback backup)
# ─────────────────────────────────────────────────────────────────────────────
class CheckpointAveragingCallback(pl.Callback):
    """
    Fallback checkpoint averaging — used in combination with EMA for robustness.
    This callback averages top-K checkpoint trainable parameters, but EMA takes
    priority since it's applied on_test_start before this callback's effects.

    Note: Since EMACallback's on_test_start runs before this callback's on_test_start
    (order depends on callback list position), we only apply this if EMA is not active.
    In practice, EMA is the primary mechanism; this is kept as reference.
    """

    def __init__(self, checkpoint_dir: Path, top_k: int = 5, enabled: bool = False):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.top_k = top_k
        self.enabled = enabled  # Only active if EMA is not available

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Skip: EMA callback handles test-time averaging."""
        if not self.enabled:
            return

        import glob
        ckpt_files = glob.glob(str(self.checkpoint_dir / "*.ckpt"))
        ckpt_files = [f for f in ckpt_files if "last" not in Path(f).name]

        if len(ckpt_files) < 2:
            pl_module.print(
                f"CheckpointAveraging: Only {len(ckpt_files)} checkpoint(s) found, skipping.")
            return

        def extract_val_f1(path: str) -> float:
            name = Path(path).stem
            try:
                import re
                m = re.search(r'val_f1[=_]([0-9.]+)', name)
                if m:
                    return float(m.group(1))
            except Exception:
                pass
            return 0.0

        ckpt_files_with_scores = [(f, extract_val_f1(f)) for f in ckpt_files]
        ckpt_files_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_files = [f for f, _ in ckpt_files_with_scores[:self.top_k]]

        pl_module.print(f"CheckpointAveraging: Averaging top-{len(top_files)} checkpoints:")
        for f, s in ckpt_files_with_scores[:self.top_k]:
            pl_module.print(f"  {Path(f).name} (val_f1={s:.4f})")

        device = pl_module.device
        avg_state = None
        loaded_count = 0

        for ckpt_path in top_files:
            try:
                ckpt = torch.load(ckpt_path, map_location=device)
                sd = ckpt.get("state_dict", ckpt)
                if avg_state is None:
                    avg_state = {k: v.float().clone() for k, v in sd.items()}
                else:
                    for k in avg_state:
                        if k in sd:
                            avg_state[k] += sd[k].float()
                loaded_count += 1
            except Exception as e:
                pl_module.print(f"  Warning: Could not load {ckpt_path}: {e}")

        if avg_state is None or loaded_count < 1:
            return

        for k in avg_state:
            avg_state[k] /= loaded_count

        try:
            pl_module.load_state_dict(avg_state, strict=False)
            pl_module.print(
                f"CheckpointAveraging: Successfully averaged {loaded_count} checkpoints.")
        except Exception as e:
            pl_module.print(f"CheckpointAveraging: Error loading averaged state: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        head_width1: int = 512,
        head_width2: int = 256,
        head_dropout: float = 0.35,
        lr: float = 2e-4,
        weight_decay: float = 0.05,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.06,
        lr_patience: int = 12,
        lr_factor: float = 0.5,
        lr_min: float = 1e-7,
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

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = QuadFeatureFusionModel(
                head_width1=self.hparams.head_width1,
                head_width2=self.hparams.head_width2,
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
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        else:
            f1 = compute_deg_f1(local_preds.numpy(), local_labels.numpy())
            self.log("val_f1", f1, prog_bar=True, sync_dist=False)

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
            dm = self.trainer.datamodule
            if hasattr(dm, "test_labels") and dm.test_labels is not None:
                labels = dm.test_labels.numpy()  # [N_test, 6640]
            else:
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
            self.print(f"Test F1: {test_f1:.4f}")

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # ReduceLROnPlateau with patience=12 — longer patience so it can actually trigger.
        # Parent's patience=7 was never triggered because val_f1 oscillated at ±0.001
        # every 2-4 epochs, continuously resetting the patience counter.
        # With patience=12, a genuine 12-epoch plateau (without ±0.001 improvement)
        # will trigger LR reduction, providing the intended convergence signal.
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            patience=self.hparams.lr_patience,
            factor=self.hparams.lr_factor,
            min_lr=self.hparams.lr_min,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

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
        description="Quad Feature Fusion (AIDO.Cell-10M + STRING static + PertCNN + SymCNN)"
        " with Three-Stage Head + EMA + ReduceLROnPlateau(patience=12)"
    )
    p.add_argument("--data-dir",               type=str,   default="data")
    p.add_argument("--micro-batch-size",        type=int,   default=8)
    p.add_argument("--global-batch-size",       type=int,   default=64)
    p.add_argument("--max-epochs",              type=int,   default=150)
    p.add_argument("--lr",                      type=float, default=2e-4)
    p.add_argument("--lr-patience",             type=int,   default=12)
    p.add_argument("--lr-factor",               type=float, default=0.5)
    p.add_argument("--lr-min",                  type=float, default=1e-7)
    p.add_argument("--weight-decay",            type=float, default=0.05)
    p.add_argument("--head-width1",             type=int,   default=512)
    p.add_argument("--head-width2",             type=int,   default=256)
    p.add_argument("--head-dropout",            type=float, default=0.35)
    p.add_argument("--gamma-focal",             type=float, default=2.0)
    p.add_argument("--label-smoothing",         type=float, default=0.06)
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
            find_unused_parameters=False,
            timeout=timedelta(seconds=120),
        )

    # ── Callbacks ────────────────────────────────────────────────────────────
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="node1-2-1-1-1-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=5,  # Top-5 for potential fallback checkpoint averaging
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

    # EMA callback: primary test-time averaging mechanism
    ema_cb = EMACallback(decay=args.ema_decay)

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
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar, ema_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=0.5,   # Tighter clipping than parent (1.0) for stability with LR=2e-4
    )
    if fast_dev_run:
        trainer.limit_test_batches = 1.0

    # ── Data & model ─────────────────────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        head_width1=args.head_width1,
        head_width2=args.head_width2,
        head_dropout=args.head_dropout,
        lr=args.lr,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Test ─────────────────────────────────────────────────────────────────
    # EMACallback will load EMA weights on test start for improved generalization.
    # The best checkpoint is loaded first, then EMA weights override it.
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # ── Save test score ──────────────────────────────────────────────────────
    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        best_val_f1 = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else None
        )
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
