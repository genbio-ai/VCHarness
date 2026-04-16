#!/usr/bin/env python3
"""
Node 1-2: STRING_GNN + AIDO.Cell-10M Frozen Dual Encoder with Low-Rank Head
============================================================================
Key design decisions (distinct from siblings):
  1. STRING_GNN (PPI graph) provides frozen 256-dim embeddings for perturbed genes
     → captures gene regulatory network position, the most relevant biological signal
  2. AIDO.Cell-10M (10M params, 256-dim hidden) provides frozen single-cell context
     → dual pooling: gene_pos_emb + mean_pool → 512-dim transcriptomic representation
  3. Feature fusion: STRING_GNN (256) + AIDO.Cell dual-pool (512) → 768-dim total
  4. Low-rank output decomposition to minimize head parameters:
     Linear(768, rank=64) → GELU → Dropout(0.5) → Linear(64, 3×6640)
     → total head params: 768*64 + 64*19920 ≈ 1.33M (vs 11.6M in parent)
  5. BOTH backbones fully FROZEN: only the head is trainable (~1.33M trainable params)
     → ~887 trainable params per training sample → dramatically reduces overfitting
  6. Fallback for STRING_GNN-unknown genes: learnable null embedding (256-dim)
  7. Fallback for AIDO.Cell-unknown genes: mean_pool only (fully differentiable)
  8. Strong regularization: Dropout(0.5) in head, label_smoothing=0.15, focal_gamma=3.0
  9. Soft class weights [2.0, 1.0, 5.0] (softer than parent's [5.0, 1.0, 10.0])
  10. Higher LR (1e-3) for the head since only the head trains
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

AIDO_MODEL_PATH = "/home/Models/AIDO.Cell-10M"  # 10M variant, 256-dim hidden
STRING_GNN_PATH = "/home/Models/STRING_GNN"

AIDO_DIM = 256          # AIDO.Cell-10M hidden dim
STRING_DIM = 256        # STRING_GNN output dim
DUAL_POOL_DIM = AIDO_DIM * 2   # 512 (gene_pos + mean_pool)
FUSION_DIM = STRING_DIM + DUAL_POOL_DIM  # 768

# Soft class weights: moderate minority boost (softer than parent's [5.0, 1.0, 10.0])
# Train distribution: ~3.41% down, ~95.48% unchanged, ~1.10% up → remapped {0,1,2}
CLASS_WEIGHTS = torch.tensor([2.0, 1.0, 5.0], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weighting and label smoothing."""

    def __init__(self, gamma: float = 3.0, weight: Optional[torch.Tensor] = None,
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
# Metric helper (mirrors calc_metric.py)
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
# STRING_GNN Embedding Extraction (pre-computed frozen embeddings)
# ─────────────────────────────────────────────────────────────────────────────
def extract_string_gnn_embeddings(model_path: str = STRING_GNN_PATH) -> tuple:
    """
    Run STRING_GNN once to extract frozen node embeddings.
    Returns:
        emb_matrix: [18870, 256] float32 tensor on CPU
        node_name_to_idx: dict mapping Ensembl ID → row index in emb_matrix
    """
    import json
    from pathlib import Path

    model_dir = Path(model_path)
    node_names = json.loads((model_dir / "node_names.json").read_text())
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

    gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    gnn_model.eval()

    graph = torch.load(str(model_dir / "graph_data.pt"), map_location="cpu")
    edge_index = graph["edge_index"]
    edge_weight = graph["edge_weight"]

    with torch.no_grad():
        outputs = gnn_model(edge_index=edge_index, edge_weight=edge_weight)

    emb_matrix = outputs.last_hidden_state.detach().cpu().float()  # [18870, 256]
    del gnn_model
    torch.cuda.empty_cache()

    return emb_matrix, node_name_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Dataset with pre-computed STRING_GNN embeddings + AIDO.Cell tokenized inputs."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_embs: torch.Tensor,      # [N, 256] float32 (STRING GNN embeddings per sample)
        string_valid: torch.Tensor,     # [N] bool (True if gene found in STRING)
        input_ids: torch.Tensor,        # [N, 19264] float32 (AIDO.Cell input)
        pert_positions: torch.Tensor,   # [N] int64 (-1 if gene not in AIDO vocab)
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.string_embs = string_embs
        self.string_valid = string_valid
        self.input_ids = input_ids
        self.pert_positions = pert_positions
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            arr = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1} → {0,1,2}
            self.labels = torch.from_numpy(arr).long()      # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "string_emb": self.string_embs[idx],       # [256] float32
            "string_valid": self.string_valid[idx],     # bool
            "input_ids": self.input_ids[idx],           # [19264] float32
            "pert_pos": self.pert_positions[idx],        # int64 (-1 if unknown)
        }
        if not self.is_test:
            item["label"] = self.labels[idx]  # [6640] int64
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

        # Pre-computed STRING_GNN embeddings (set in setup)
        self._string_emb_matrix: Optional[torch.Tensor] = None
        self._string_node_to_idx: Optional[Dict[str, int]] = None
        # Pre-computed null embedding for genes not in STRING_GNN (mean of all embeddings)
        self._string_null_emb: Optional[torch.Tensor] = None

    def _init_string_gnn(self) -> None:
        """Extract STRING_GNN embeddings once (rank-0 first, then all ranks load from CPU)."""
        if self._string_emb_matrix is not None:
            return

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            print("Extracting STRING_GNN embeddings (once)...")
            emb, node_to_idx = extract_string_gnn_embeddings()
            self._string_emb_matrix = emb
            self._string_node_to_idx = node_to_idx
            self._string_null_emb = emb.mean(dim=0)  # mean embedding as null fallback
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if local_rank != 0:
            # Other ranks also load (all from same local model path)
            emb, node_to_idx = extract_string_gnn_embeddings()
            self._string_emb_matrix = emb
            self._string_node_to_idx = node_to_idx
            self._string_null_emb = emb.mean(dim=0)

    def _init_tokenizer(self) -> AutoTokenizer:
        """Rank-safe tokenizer initialization (rank 0 first if distributed)."""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)

    def _get_string_embs(self, pert_ids: List[str]) -> tuple:
        """Look up STRING_GNN embeddings for a list of pert_ids.

        Returns:
            embs:  [N, 256] float32 (null_emb for unknown genes)
            valid: [N] bool (True if gene found in STRING_GNN)
        """
        null = self._string_null_emb  # [256]
        embs = []
        valid_flags = []
        for pid in pert_ids:
            if pid in self._string_node_to_idx:
                idx = self._string_node_to_idx[pid]
                embs.append(self._string_emb_matrix[idx])
                valid_flags.append(True)
            else:
                embs.append(null)
                valid_flags.append(False)
        return torch.stack(embs, dim=0), torch.tensor(valid_flags, dtype=torch.bool)

    def _tokenize_and_get_positions(
        self,
        tokenizer: AutoTokenizer,
        pert_ids: List[str],
        split_name: str = "split",
    ) -> tuple:
        """Tokenize all pert_ids and find each gene's position in the 19264-gene vocabulary."""
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        chunk_size = 128
        all_input_ids: List[torch.Tensor] = []
        for i in range(0, len(expr_dicts), chunk_size):
            chunk = expr_dicts[i:i + chunk_size]
            toks = tokenizer(chunk, return_tensors="pt")
            all_input_ids.append(toks["input_ids"])

        input_ids = torch.cat(all_input_ids, dim=0)  # [N, 19264] float32
        non_missing = input_ids > -0.5
        has_gene = non_missing.any(dim=1)
        pert_positions = non_missing.long().argmax(dim=1)
        pert_positions[~has_gene] = -1

        coverage = 100.0 * has_gene.float().mean().item()
        print(f"  [{split_name}] AIDO vocab coverage: "
              f"{has_gene.sum().item()}/{len(pert_ids)} ({coverage:.1f}%)")
        return input_ids, pert_positions

    def setup(self, stage: Optional[str] = None) -> None:
        # Initialize STRING_GNN embeddings
        self._init_string_gnn()
        tokenizer = self._init_tokenizer()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")

            print("Preparing train set...")
            tr_str_embs, tr_str_valid = self._get_string_embs(train_df["pert_id"].tolist())
            tr_ids, tr_pos = self._tokenize_and_get_positions(
                tokenizer, train_df["pert_id"].tolist(), "train")

            print("Preparing val set...")
            va_str_embs, va_str_valid = self._get_string_embs(val_df["pert_id"].tolist())
            va_ids, va_pos = self._tokenize_and_get_positions(
                tokenizer, val_df["pert_id"].tolist(), "val")

            self.train_ds = PerturbationDataset(
                train_df, tr_str_embs, tr_str_valid, tr_ids, tr_pos, is_test=False)
            self.val_ds = PerturbationDataset(
                val_df, va_str_embs, va_str_valid, va_ids, va_pos, is_test=False)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            print("Preparing test set...")
            te_str_embs, te_str_valid = self._get_string_embs(test_df["pert_id"].tolist())
            te_ids, te_pos = self._tokenize_and_get_positions(
                tokenizer, test_df["pert_id"].tolist(), "test")

            self.test_ds = PerturbationDataset(
                test_df, te_str_embs, te_str_valid, te_ids, te_pos, is_test=True)
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
# Model: Frozen Dual Encoder (STRING_GNN + AIDO.Cell-10M) + Low-Rank Head
# ─────────────────────────────────────────────────────────────────────────────
class FrozenDualEncoderModel(nn.Module):
    """
    Architecture:
      ┌─ STRING_GNN embedding (256-dim, FROZEN, pre-computed) ─────────────────┐
      │  for each perturbed gene: look up its PPI graph embedding              │
      └──────────────────────────────────────────────────────────────────────┘
                  + null_emb (256-dim learnable) for unknown genes
      ┌─ AIDO.Cell-10M FROZEN (256-dim hidden) ──────────────────────────────┐
      │  dual pooling: [gene_pos_emb (256)] + [mean_pool (256)] = 512-dim    │
      └──────────────────────────────────────────────────────────────────────┘
             ↓ cat
      768-dim fusion feature
             ↓
      LayerNorm(768)
      → Linear(768, head_rank) → GELU → Dropout(0.5) → Linear(head_rank, 3×6640)
             ↓
      [B, 3, 6640] logits

    Total trainable params (head_rank=64):
      null_emb:    256
      head W1:     768 × 64 = 49,152
      head W2:     64 × 19,920 = 1,274,880
      head biases: 64 + 19,920 = 19,984
      LayerNorm:   768 × 2 = 1,536
      Total ≈ 1.35M trainable params

    (vs parent node's 11.6M trainable params)
    """

    def __init__(self, head_rank: int = 64, head_dropout: float = 0.5):
        super().__init__()
        self.head_rank = head_rank
        self.head_dropout = head_dropout

        # Backbone (initialized in initialize_backbones)
        self.aido_backbone: Optional[nn.Module] = None

        # Learnable fallback embedding for genes not in STRING_GNN vocabulary
        # Initialized to zero; adapts during training to represent "unknown gene"
        self.string_null_emb = nn.Parameter(torch.zeros(STRING_DIM))

        # Low-rank output head
        self.head: Optional[nn.Sequential] = None

    def initialize_backbones(self) -> None:
        """Load AIDO.Cell-10M (frozen) and enable gradient checkpointing."""
        self.aido_backbone = AutoModel.from_pretrained(
            AIDO_MODEL_PATH, trust_remote_code=True)
        self.aido_backbone = self.aido_backbone.to(torch.bfloat16)
        self.aido_backbone.config.use_cache = False

        # Freeze ALL backbone parameters
        for param in self.aido_backbone.parameters():
            param.requires_grad = False

        print("AIDO.Cell-10M backbone: ALL parameters FROZEN.")
        total = sum(p.numel() for p in self.aido_backbone.parameters())
        print(f"  AIDO.Cell-10M total params (all frozen): {total:,}")

    def initialize_head(self) -> None:
        """Create low-rank output head."""
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),         # 768
            nn.Linear(FUSION_DIM, self.head_rank),   # 768 → 64
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.head_rank, N_CLASSES * N_GENES),  # 64 → 3*6640
        )
        # Truncated-normal init for stable early training
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_aido_dual_pool(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32
        pert_positions: torch.Tensor,  # [B] int64 (-1 for unknown)
    ) -> torch.Tensor:
        """
        Run AIDO.Cell-10M (frozen) and return dual-pooled representation.
        Returns: [B, 512] float32 (gene_pos_emb ++ mean_pool)
        """
        # Determine backbone device from its parameters
        backbone_device = next(self.aido_backbone.parameters()).device
        input_ids_dev = input_ids.to(backbone_device)
        attn_mask = torch.ones(input_ids_dev.shape[0], input_ids_dev.shape[1],
                               dtype=torch.long, device=backbone_device)
        with torch.no_grad():
            # Forward through frozen backbone
            out = self.aido_backbone(
                input_ids=input_ids_dev,
                attention_mask=attn_mask,
            )
        hidden = out.last_hidden_state  # [B, 19266, 256] bfloat16

        # Global mean-pool over gene positions (exclude the 2 summary tokens)
        mean_pool = hidden[:, :19264, :].mean(dim=1).float()  # [B, 256]

        # Per-gene positional extraction
        B = hidden.size(0)
        hidden_device = hidden.device
        # Move pert_positions to backbone device for consistent indexing
        pert_positions_dev = pert_positions.to(hidden_device)
        valid = pert_positions_dev >= 0
        safe_pos = pert_positions_dev.clamp(min=0)

        gene_emb_raw = hidden[
            torch.arange(B, device=hidden_device), safe_pos, :
        ].float()  # [B, 256]

        # Differentiable fallback to mean_pool for unknown genes
        valid_f = valid.float().unsqueeze(-1)
        gene_emb = gene_emb_raw * valid_f + mean_pool * (1.0 - valid_f)  # [B, 256]

        return torch.cat([gene_emb, mean_pool], dim=-1)  # [B, 512]

    def forward(
        self,
        string_emb: torch.Tensor,      # [B, 256] float32 (pre-computed, STRING_GNN)
        string_valid: torch.Tensor,    # [B] bool (True if gene in STRING_GNN)
        input_ids: torch.Tensor,       # [B, 19264] float32 (AIDO.Cell input)
        pert_positions: torch.Tensor,  # [B] int64 (-1 for unknown)
    ) -> torch.Tensor:
        """Returns: [B, 3, N_GENES] logits."""
        B = string_emb.size(0)

        # ── STRING_GNN feature ───────────────────────────────────────────────
        # Use pre-computed frozen embeddings; fall back to learnable null_emb
        # string_emb and string_valid are already on the correct device (moved by Lightning DataLoader)
        valid_f = string_valid.float().unsqueeze(-1)  # [B, 1]
        null_exp = self.string_null_emb.float().unsqueeze(0).expand(B, -1)  # [B, 256] on model device
        str_feat = string_emb.float() * valid_f + null_exp * (1.0 - valid_f)
        # [B, 256]

        # ── AIDO.Cell-10M dual-pool feature (frozen inference) ───────────────
        aido_feat = self._get_aido_dual_pool(input_ids, pert_positions)  # [B, 512]
        # Move aido_feat to same device as str_feat (which is on the head's device)
        aido_feat = aido_feat.to(str_feat.device)

        # ── Fusion ───────────────────────────────────────────────────────────
        combined = torch.cat([str_feat, aido_feat], dim=-1)  # [B, 768]

        # ── Low-rank head ─────────────────────────────────────────────────────
        logits = self.head(combined)                  # [B, 3*6640]
        return logits.view(B, N_CLASSES, N_GENES)     # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        head_rank: int = 64,
        head_dropout: float = 0.5,
        lr_head: float = 1e-3,
        weight_decay: float = 5e-2,
        gamma_focal: float = 3.0,
        label_smoothing: float = 0.15,
        max_epochs: int = 150,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialized in setup()
        self.model: Optional[FrozenDualEncoderModel] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators (cleared each epoch)
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._test_labels: Optional[torch.Tensor] = None  # loaded in setup(stage="test")
        self._test_f1_computed: Optional[float] = None  # set after test epoch ends

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = FrozenDualEncoderModel(
                head_rank=self.hparams.head_rank,
                head_dropout=self.hparams.head_dropout,
            )
            self.model.initialize_backbones()
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

        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols
            # Load test ground-truth labels for metric computation
            test_df = pd.read_csv(
                Path(self.trainer.datamodule.data_dir) / "test.tsv", sep="\t")
            raw_test_labels = [json.loads(x) for x in test_df["label"].tolist()]
            test_labels_arr = np.array(raw_test_labels, dtype=np.int8) + 1  # {-1,0,1}→{0,1,2}
            self._test_labels = torch.from_numpy(test_labels_arr).long()     # [N_test, 6640]

    def forward(
        self,
        string_emb: torch.Tensor,
        string_valid: torch.Tensor,
        input_ids: torch.Tensor,
        pert_positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(string_emb, string_valid, input_ids, pert_positions)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G], labels: [B, G] ({0,1,2}) → scalar loss."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        labels_flat = labels.reshape(-1)                       # [B*G]
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["string_emb"], batch["string_valid"],
            batch["input_ids"], batch["pert_pos"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(
            batch["string_emb"], batch["string_valid"],
            batch["input_ids"], batch["pert_pos"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
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
            batch["string_emb"], batch["string_valid"],
            batch["input_ids"], batch["pert_pos"])
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
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

            # Compute test F1 if ground-truth labels are available
            test_f1 = None
            if hasattr(self, "_test_labels") and self._test_labels is not None:
                gt_labels = self._test_labels.numpy()  # [N_test, 6640]
                if len(preds) == len(gt_labels):
                    test_f1 = compute_deg_f1(preds, gt_labels)
                    self.log("test_f1", test_f1, prog_bar=True, sync_dist=False)
                    self.print(f"Test F1: {test_f1:.4f}")

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
            self._test_f1_computed = test_f1

    def configure_optimizers(self):
        # Only the head parameters are trainable
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.lr_head,
            weight_decay=self.hparams.weight_decay,
        )
        # OneCycleLR provides warming up + annealing for head-only training
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    # ── Checkpoint: save only trainable parameters ────────────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
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
        total = sum(p.numel() for p in self.parameters())
        tr_cnt = sum(p.numel() for p in self.parameters() if p.requires_grad)
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
        description="STRING_GNN + AIDO.Cell-10M Frozen Dual Encoder DEG predictor"
    )
    p.add_argument("--data-dir",               type=str,   default="data")
    p.add_argument("--micro-batch-size",        type=int,   default=16)
    p.add_argument("--global-batch-size",       type=int,   default=64)
    p.add_argument("--max-epochs",              type=int,   default=150)
    p.add_argument("--lr-head",                 type=float, default=1e-3)
    p.add_argument("--weight-decay",            type=float, default=5e-2)
    p.add_argument("--head-rank",               type=int,   default=64)
    p.add_argument("--head-dropout",            type=float, default=0.5)
    p.add_argument("--gamma-focal",             type=float, default=3.0)
    p.add_argument("--label-smoothing",         type=float, default=0.15)
    p.add_argument("--early-stopping-patience", type=int,   default=25)
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
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = args.debug_max_step

    val_check_interval = args.val_check_interval if (
        args.debug_max_step is None and not args.fast_dev_run
    ) else 1.0

    strategy: Any
    if n_gpus == 1:
        strategy = SingleDeviceStrategy(device="cuda:0")
    else:
        strategy = DDPStrategy(
            find_unused_parameters=False,  # Head-only training: no unused params
            timeout=timedelta(seconds=300),
        )

    # ── Callbacks ────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-{epoch:03d}-{val_f1:.4f}",
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
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

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
        deterministic=True,
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
        head_rank=args.head_rank,
        head_dropout=args.head_dropout,
        lr_head=args.lr_head,
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
        # Only write meaningful results (skip fast_dev_run / debug_max_step limited runs)
        test_f1 = getattr(model_module, "_test_f1_computed", None)
        best_score = checkpoint_cb.best_model_score
        if test_f1 is not None:
            # Full run: write computed test F1 + best val F1
            val_f1_str = f"{best_score:.6f}" if best_score is not None else "N/A"
            score_path.write_text(
                f"test_f1: {test_f1:.6f}\n"
                f"val_f1_best: {val_f1_str}\n"
            )
            print(f"Test score saved → {score_path}")
        elif test_results and test_results[0]:
            # Fallback: write trainer.test() results
            score_path.write_text(f"test_results: {test_results}\n")
            print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
