#!/usr/bin/env python3
"""
Node 1-2-1: Triple Feature Fusion — AIDO.Cell-10M (frozen) + STRING_GNN (static) + Char-CNN
============================================================================================
Key design decisions (improving on node1-2):

  1. AIDO.Cell-10M (FROZEN) dual pool: gene_pos_emb (256) + mean_pool (256) = 512-dim.
     No LoRA, no QKV adaptation — frozen backbone provides stable representations.

  2. STRING_GNN (FROZEN, pre-computed once at setup) — static topology-based embeddings.
     Critically, NO dynamic cond_proj conditioning (source of overfitting in node1-2).
     Pre-compute [18870, 256] embeddings once; use lookup per sample → 256-dim.
     This mirrors node1-1-1-1 (test F1=0.410) which worked stably.

  3. Character-level CNN on pert_id (Ensembl gene ID) → 64-dim.
     Proven orthogonal signal from node2-2 lineage (pushed +0.041 improvement).
     3-branch multi-scale Conv1d (k=3,5,7) → AdaptiveMaxPool → 64-dim fc.

  4. Triple fusion: cat([512, 256, 64]) = 832-dim.

  5. Wider head: LayerNorm(832) → Linear(832, 256) → GELU → Dropout(0.5)
     → LayerNorm(256) → Linear(256, 3×6640).
     (256-dim vs parent's 128-dim; wider head confirmed beneficial in feedback.)

  6. ReduceLROnPlateau(mode='max', patience=7, factor=0.5) — stable LR decay
     vs parent's CosineAnnealingWarmRestarts (no meaningful benefit confirmed).

  7. Stronger regularization: dropout=0.5, weight_decay=0.05.

  8. Softer focal: gamma=2.0, label_smoothing=0.05.

  9. Shorter early stopping patience: 20 (vs parent's 30).

Root cause fixes vs node1-2:
  - Remove cond_proj (~66K params) → eliminates the primary overfitting source
  - Pre-compute STRING embeddings → no per-batch GNN forward (stable, fast)
  - Add char-CNN → proven +0.041 orthogonal signal from node2-2 experiments
  - Wider head (256) → confirmed beneficial in feedback
  - ReduceLROnPlateau → stable training without oscillation
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

AIDO_MODEL_PATH = "/home/Models/AIDO.Cell-10M"   # 10M, 256-dim hidden
STRING_GNN_PATH = "/home/Models/STRING_GNN"

AIDO_DIM = 256                              # AIDO.Cell-10M hidden dim
STRING_DIM = 256                            # STRING_GNN output dim
DUAL_POOL_DIM = AIDO_DIM * 2               # 512 (gene_pos + mean_pool)
CNN_DIM = 64                               # character-CNN output dim
FUSION_DIM = STRING_DIM + DUAL_POOL_DIM + CNN_DIM   # 832

# Moderate class weights: down-regulated / unchanged / up-regulated
CLASS_WEIGHTS = torch.tensor([3.0, 1.0, 7.0], dtype=torch.float32)


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

    Returns:
        gnn_model:        StringGNNModel (eval mode, FROZEN)
        edge_index:       [2, E] int64 CPU tensor
        edge_weight:      [E] float32 CPU tensor (or None)
        node_name_to_idx: dict mapping Ensembl ID -> node index in [0, 18869]
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
    edge_index = graph["edge_index"]               # [2, E] int64
    edge_weight = graph.get("edge_weight", None)   # [E] float32 or None

    n_nodes = len(node_names)
    print(f"STRING_GNN loaded: {n_nodes} nodes, "
          f"{edge_index.shape[1]} directed edges, "
          f"{sum(p.numel() for p in gnn_model.parameters()):,} params (all frozen)")

    return gnn_model, edge_index, edge_weight, node_name_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# Character-level CNN for Ensembl gene IDs
# ─────────────────────────────────────────────────────────────────────────────
class GeneSymbolCNN(nn.Module):
    """
    Encode Ensembl gene IDs (e.g. "ENSG00000001084") as 64-dim embeddings
    using a 3-branch multi-scale character-level CNN.

    Proven orthogonal signal: node2-2 showed +0.041 F1 improvement over
    node2 by adding this CNN to AIDO.Cell-10M LoRA. Here we apply the same
    trick to the frozen-backbone lineage without LoRA.

    Character vocabulary: 'ENSG0123456789' (14 unique chars in Ensembl IDs).
    Pad index = 14.  Max length = 15.
    """
    CHARS = 'ENSG0123456789'   # 14 unique chars
    PAD_IDX = 14
    MAX_LEN = 15

    def __init__(self, out_dim: int = 64, embed_dim: int = 8):
        super().__init__()
        vocab_size = len(self.CHARS) + 1   # 15 (indices 0-13 = chars, 14 = pad)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=self.PAD_IDX)
        self.conv1 = nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, 32, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(96, out_dim)

    @classmethod
    def encode_pert_id(cls, pert_id: str) -> List[int]:
        """Encode an Ensembl gene ID as padded character indices."""
        ids = []
        for c in pert_id[:cls.MAX_LEN]:
            idx = cls.CHARS.find(c)
            ids.append(idx if idx >= 0 else cls.PAD_IDX)
        while len(ids) < cls.MAX_LEN:
            ids.append(cls.PAD_IDX)
        return ids

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        char_ids: [B, MAX_LEN] int64
        Returns:  [B, out_dim] float32
        """
        x = self.embed(char_ids).permute(0, 2, 1)        # [B, embed_dim, MAX_LEN]
        b1 = self.pool(F.gelu(self.conv1(x))).squeeze(-1)  # [B, 32]
        b2 = self.pool(F.gelu(self.conv2(x))).squeeze(-1)  # [B, 32]
        b3 = self.pool(F.gelu(self.conv3(x))).squeeze(-1)  # [B, 32]
        out = torch.cat([b1, b2, b3], dim=-1)              # [B, 96]
        return F.gelu(self.fc(out))                        # [B, out_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Dataset that pre-tokenizes AIDO.Cell inputs, stores STRING node indices,
    and pre-encodes pert_id character sequences for the char-CNN.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        input_ids: torch.Tensor,             # [N, 19264] float32 (AIDO.Cell input)
        pert_positions: torch.Tensor,        # [N] int64 (-1 if gene not in AIDO vocab)
        string_node_indices: torch.Tensor,   # [N] int64 (-1 if gene not in STRING vocab)
        pert_char_ids: torch.Tensor,         # [N, MAX_LEN] int64 char indices
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.input_ids = input_ids
        self.pert_positions = pert_positions
        self.string_node_indices = string_node_indices
        self.pert_char_ids = pert_char_ids
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
            "input_ids": self.input_ids[idx],                # [19264] float32
            "pert_pos": self.pert_positions[idx],             # int64 (-1 if unknown)
            "string_node_idx": self.string_node_indices[idx], # int64 (-1 if unknown)
            "pert_char_ids": self.pert_char_ids[idx],         # [MAX_LEN] int64
        }
        if not self.is_test:
            item["label"] = self.labels[idx]   # [6640] int64
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

        # STRING vocabulary (set in setup)
        self._string_node_to_idx: Optional[Dict[str, int]] = None

    def _init_tokenizer(self) -> AutoTokenizer:
        """Rank-safe tokenizer initialization (rank 0 downloads first if distributed)."""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)

    def _init_string_vocab(self) -> None:
        """Load STRING node vocabulary (fast, JSON only, no model weights)."""
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
        """Tokenize pert_ids for AIDO.Cell and get positional indices."""
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
        """Look up STRING node indices for pert_ids. Returns -1 for unknown genes."""
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
        """Encode Ensembl IDs as character index tensors: [N, MAX_LEN] int64."""
        encoded = [GeneSymbolCNN.encode_pert_id(pid) for pid in pert_ids]
        return torch.tensor(encoded, dtype=torch.long)   # [N, MAX_LEN]

    def _prepare_split(
        self,
        tokenizer: AutoTokenizer,
        df: pd.DataFrame,
        split_name: str,
        is_test: bool = False,
    ) -> PerturbationDataset:
        """Prepare a single data split."""
        pert_ids = df["pert_id"].tolist()

        input_ids, pert_positions = self._tokenize_and_get_positions(
            tokenizer, pert_ids, split_name)
        string_node_indices = self._get_string_node_indices(pert_ids, split_name)
        pert_char_ids = self._encode_pert_char_ids(pert_ids)

        return PerturbationDataset(
            df, input_ids, pert_positions, string_node_indices, pert_char_ids,
            is_test=is_test,
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
# Model: Triple Feature Fusion Encoder
# ─────────────────────────────────────────────────────────────────────────────
class TripleFeatureFusionModel(nn.Module):
    """
    Architecture:
      ┌─ AIDO.Cell-10M FROZEN (256-dim) ──────────────────────────────────────┐
      │  dual pooling: [gene_pos_emb (256)] + [mean_pool (256)] = 512-dim     │
      └────────────────────────────────────────────────────────────────────────┘
                  ↓
      ┌─ STRING_GNN FROZEN, pre-computed static embeddings ─────────────────────┐
      │  One GNN forward at setup, no dynamic conditioning                      │
      │  Static lookup: string_static_embs[node_idx] → [B, 256]                │
      │  Null fallback for genes not in STRING vocab                            │
      └────────────────────────────────────────────────────────────────────────┘
                  ↓
      ┌─ Character-level CNN on Ensembl ID ──────────────────────────────────────┐
      │  Multi-scale Conv1d(k=3,5,7) → AdaptiveMaxPool1d(1) → [B, 96] → [B,64] │
      └────────────────────────────────────────────────────────────────────────┘
                  ↓ cat([512, 256, 64]) = 832-dim
      LayerNorm(832)
      → Linear(832, 256) → GELU → Dropout(0.5)
      → LayerNorm(256)
      → Linear(256, 3×6640)
      → [B, 3, 6640] logits

    Trainable parameters (head_width=256):
      gene_cnn:         ~7,200 params
      string_null_emb:  256 params
      LayerNorm(832):   1,664
      Linear(832→256):  213,248
      LayerNorm(256):   512
      Linear(256→19920): 5,099,520
      biases:           ~20,432
      Total ≈ 5.34M params (~3,560/training sample)
    """

    def __init__(self, head_width: int = 256, head_dropout: float = 0.5):
        super().__init__()
        self.head_width = head_width
        self.head_dropout = head_dropout

        # AIDO.Cell-10M backbone (initialized in initialize_aido())
        self.aido_backbone: Optional[nn.Module] = None

        # STRING_GNN backbone (initialized in initialize_string_gnn())
        # string_static_embs: registered buffer [18870, 256] once pre-computed
        self.string_gnn: Optional[nn.Module] = None
        self._n_nodes: int = 18870

        # Learnable fallback for genes not in STRING vocabulary
        self.string_null_emb = nn.Parameter(torch.zeros(STRING_DIM))

        # Character-level CNN encoder for Ensembl IDs (trainable)
        self.gene_cnn = GeneSymbolCNN(out_dim=CNN_DIM, embed_dim=8)

        # Output head (trainable)
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
        Load STRING_GNN (frozen), run ONE forward pass to pre-compute static
        topology embeddings, then store as buffer. No dynamic conditioning.
        """
        gnn_model, edge_index, edge_weight, _ = load_string_gnn()

        # Register graph tensors as buffers (auto-moved to correct device)
        self.register_buffer("_edge_index_buf", edge_index)
        if edge_weight is not None:
            self.register_buffer("_edge_weight_buf", edge_weight)

        self._n_nodes = 18870

        # Pre-compute static embeddings once (pure topology, no cond_emb)
        gnn_model.eval()
        with torch.no_grad():
            out = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
        static_embs = out.last_hidden_state.float()   # [18870, 256]
        self.register_buffer("string_static_embs", static_embs)

        # Keep reference for completeness but only static_embs are used
        self.string_gnn = gnn_model
        print(f"STRING_GNN static embeddings pre-computed: {static_embs.shape}")

    def initialize_head(self) -> None:
        """Create the output head (trainable)."""
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),                             # 832
            nn.Linear(FUSION_DIM, self.head_width),               # 832 → 256
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.LayerNorm(self.head_width),                        # 256
            nn.Linear(self.head_width, N_CLASSES * N_GENES),      # 256 → 3*6640
        )
        # Truncated-normal init for stable early training
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize null_emb near zero
        nn.init.zeros_(self.string_null_emb.data)

    def _get_aido_dual_pool(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32
        pert_positions: torch.Tensor,  # [B] int64 (-1 for unknown)
    ) -> tuple:
        """
        Run AIDO.Cell-10M (frozen) and return:
          - gene_pos_emb: [B, 256] float32
          - mean_pool:    [B, 256] float32
          - dual_pool:    [B, 512] float32 = cat([gene_pos_emb, mean_pool])
        """
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

        # Global mean-pool over gene positions (exclude 2 summary tokens)
        mean_pool = hidden[:, :19264, :].mean(dim=1).float()   # [B, 256]

        # Per-gene positional extraction
        B = hidden.size(0)
        hidden_device = hidden.device
        pert_positions_dev = pert_positions.to(hidden_device)
        valid = pert_positions_dev >= 0
        safe_pos = pert_positions_dev.clamp(min=0)
        gene_emb_raw = hidden[
            torch.arange(B, device=hidden_device), safe_pos, :
        ].float()   # [B, 256]

        # Fallback to mean_pool for unknown genes
        valid_f = valid.float().unsqueeze(-1)
        gene_emb = gene_emb_raw * valid_f + mean_pool * (1.0 - valid_f)

        dual_pool = torch.cat([gene_emb, mean_pool], dim=-1)   # [B, 512]
        return gene_emb, mean_pool, dual_pool

    def forward(
        self,
        input_ids: torch.Tensor,            # [B, 19264] float32
        pert_positions: torch.Tensor,       # [B] int64 (-1 for AIDO unknown)
        string_node_indices: torch.Tensor,  # [B] int64 (-1 for STRING unknown)
        pert_char_ids: torch.Tensor,        # [B, MAX_LEN] int64
    ) -> torch.Tensor:
        """Returns: [B, 3, N_GENES] logits."""
        B = input_ids.size(0)

        # ── AIDO.Cell-10M: frozen dual-pool ──────────────────────────────────
        gene_pos_emb, mean_pool, aido_dual = self._get_aido_dual_pool(
            input_ids, pert_positions)   # [B, 256], [B, 256], [B, 512]

        # ── STRING static lookup + null fallback ──────────────────────────────
        str_device = self.string_static_embs.device
        valid = (string_node_indices >= 0)
        safe_idx = string_node_indices.clamp(min=0).to(str_device)
        str_feat = self.string_static_embs[safe_idx]   # [B, 256]
        null_emb = self.string_null_emb.to(str_device)
        valid_f = valid.float().to(str_device).unsqueeze(-1)
        str_feat = str_feat * valid_f + null_emb.unsqueeze(0) * (1.0 - valid_f)

        # ── Character-level CNN ───────────────────────────────────────────────
        cnn_device = next(self.gene_cnn.parameters()).device
        cnn_feat = self.gene_cnn(pert_char_ids.to(cnn_device))   # [B, 64]

        # ── Fuse all features on head device ─────────────────────────────────
        head_device = next(self.head.parameters()).device
        combined = torch.cat([
            str_feat.to(head_device),
            aido_dual.to(head_device),
            cnn_feat.to(head_device),
        ], dim=-1)   # [B, 832]

        # ── Output head ───────────────────────────────────────────────────────
        logits = self.head(combined)                    # [B, 3*6640]
        return logits.view(B, N_CLASSES, N_GENES)       # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        head_width: int = 256,
        head_dropout: float = 0.5,
        lr_head: float = 5e-4,
        lr_patience: int = 7,
        lr_factor: float = 0.5,
        weight_decay: float = 0.05,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 150,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialized in setup()
        self.model: Optional[TripleFeatureFusionModel] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators (cleared each epoch)
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = TripleFeatureFusionModel(
                head_width=self.hparams.head_width,
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

            # Print trainable parameter count
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            self.print(f"Trainable params: {trainable:,} / {total:,} "
                       f"({100.0 * trainable / max(total, 1):.2f}%)")
            self.print(f"Trainable params per training sample: "
                       f"{trainable / 1500:.0f}")

        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def forward(
        self,
        input_ids: torch.Tensor,
        pert_positions: torch.Tensor,
        string_node_indices: torch.Tensor,
        pert_char_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, pert_positions, string_node_indices, pert_char_ids)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G], labels: [B, G] ({0,1,2}) → scalar loss."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)   # [B*G, 3]
        labels_flat = labels.reshape(-1)                        # [B*G]
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["input_ids"], batch["pert_pos"],
            batch["string_node_idx"], batch["pert_char_ids"],
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["pert_pos"],
            batch["string_node_idx"], batch["pert_char_ids"],
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
            batch["string_node_idx"], batch["pert_char_ids"],
        )
        probs = F.softmax(logits.detach().float(), dim=1).cpu()   # [B, 3, G]
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

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable,
            lr=self.hparams.lr_head,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            patience=self.hparams.lr_patience,
            factor=self.hparams.lr_factor,
            min_lr=1e-7,
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
        description="Triple Feature Fusion (AIDO.Cell-10M + STRING_GNN static + Char-CNN)"
    )
    p.add_argument("--data-dir",               type=str,   default="data")
    p.add_argument("--micro-batch-size",        type=int,   default=8)
    p.add_argument("--global-batch-size",       type=int,   default=64)
    p.add_argument("--max-epochs",              type=int,   default=150)
    p.add_argument("--lr-head",                 type=float, default=5e-4)
    p.add_argument("--lr-patience",             type=int,   default=7)
    p.add_argument("--lr-factor",               type=float, default=0.5)
    p.add_argument("--weight-decay",            type=float, default=0.05)
    p.add_argument("--head-width",              type=int,   default=256)
    p.add_argument("--head-dropout",            type=float, default=0.5)
    p.add_argument("--gamma-focal",             type=float, default=2.0)
    p.add_argument("--label-smoothing",         type=float, default=0.05)
    p.add_argument("--early-stopping-patience", type=int,   default=20)
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
            find_unused_parameters=True,   # Frozen backbone params are not in grad graph
            timeout=timedelta(seconds=120),
        )

    # ── Callbacks ────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-1-{epoch:03d}-{val_f1:.4f}",
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
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic="warn_only",
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    # ── Data & model ─────────────────────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        head_width=args.head_width,
        head_dropout=args.head_dropout,
        lr_head=args.lr_head,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Test ─────────────────────────────────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # ── Save test score ───────────────────────────────────────────────────────
    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        best_val_f1 = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else None
        )
        score_path.write_text(
            f"test_results: {test_results}\n"
            f"val_f1_best: {best_val_f1}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
