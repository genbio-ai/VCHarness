#!/usr/bin/env python3
"""
Node 2-3-1: AIDO.Cell-10M + LoRA (Last 4 Layers) + Gene Symbol CNN
           + Multi-Pooling (Mean+Max) + LinearWarmup + CosineAnnealing
           + Expression Noise Augmentation + Restored Regularization
================================================================
Key improvements over parent node2-3 (test F1=0.4375):

1. Restored proven regularization from node2-2 (F1=0.4453):
   - weight_decay=0.05 (was 0.03 → too weak, caused overfitting)
   - focal_gamma=2.0 (was 1.5 → restored minority class emphasis)
   - head_hidden=256 (was 320 → 256 is proven optimal for 1,500 samples)

2. Multi-pooling: mean + max over gene sequence (ARCHITECTURAL INNOVATION)
   - global_mean_emb [B, 256] + global_max_emb [B, 256] captures complementary
     statistics: mean reflects average perturbation response, max captures
     the most salient activated gene feature
   - Total head input: 832-dim (vs 576 in parent)

3. LinearWarmup + CosineAnnealingLR scheduler
   - node2-2 showed CosineAnnealing outperforms ReduceLROnPlateau for this task
   - LinearLR warmup (5 epochs) avoids the abrupt warmup→cosine transition
     instability identified in node2-2's feedback
   - Smooth LR decay over remaining epochs

4. Expression noise augmentation (new regularization strategy)
   - Small Gaussian noise N(0, expr_noise_std=0.02) added to non-knockout
     expression positions during training only
   - Acts as implicit data augmentation, reducing memorization

5. Tighter gradient clipping (0.5 vs 1.0)
   - Feedback identified gradient spikes in large head as contributing to instability
   - 0.5 clips more aggressively to stabilize head training

6. Earlier early stopping (patience=20 vs 35)
   - node2-3's 35-epoch patience allowed 35 epochs of pure overfitting
   - With CosineAnnealing, the model converges more smoothly so tighter patience
     is appropriate

7. head_lr_multiplier=3.0 (retained from node2-3)
   - node2-2's 5× was shown to be too aggressive for symbol CNN stability

Differentiation from parent node2-3:
- Architecture: adds max-pooling (832-dim vs 576-dim head input)
- Regularization: fully restored to node2-2's proven values
- Scheduler: LinearWarmup+Cosine instead of ReduceLROnPlateau
- Augmentation: expression noise (new)
- Training stability: tighter gradient clip, earlier stopping
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
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
AIDO_CELL_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
N_GENES_AIDO = 19_264     # AIDO.Cell vocabulary size
N_GENES_OUT = 6_640       # output genes (DEG prediction target panel)
N_CLASSES = 3
SENTINEL_EXPR = 1.0       # baseline expression for non-perturbed genes
KNOCKOUT_EXPR = 0.0       # expression value for knocked-out gene
# Restored to node2-2 proven values: [5, 1, 10] with gamma=2.0
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)

# Character vocabulary for gene symbol CNN
# Index 0: padding, 1: unknown, 2-27: A-Z, 28-37: 0-9, 38: '-', 39: '.', 40: '_'
CHAR_VOCAB: Dict[str, int] = {}
for _i, _c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    CHAR_VOCAB[_c] = _i + 2
for _i, _c in enumerate("0123456789"):
    CHAR_VOCAB[_c] = _i + 28
CHAR_VOCAB['-'] = 38
CHAR_VOCAB['.'] = 39
CHAR_VOCAB['_'] = 40
CHAR_VOCAB_SIZE = 41   # 0..40
SYMBOL_MAX_LEN = 12    # max gene symbol length to encode


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss with Label Smoothing
# ─────────────────────────────────────────────────────────────────────────────
class FocalLossWithSmoothing(nn.Module):
    """Focal loss with class weighting and label smoothing.

    Focal weight is computed from hard targets (no smoothing) to preserve the
    "hard example emphasis" property. The cross-entropy term uses smoothed
    targets to prevent overconfident training predictions.
    """

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
        """
        logits:  [N, C]  float32 — raw class scores
        targets: [N]     int64   — hard class indices
        """
        w = self.weight.to(logits.device) if self.weight is not None else None

        # (1) Focal weight: based on hard-target CE (no smoothing)
        with torch.no_grad():
            ce_hard = F.cross_entropy(logits.float(), targets, reduction="none")
            pt = torch.exp(-ce_hard)
            focal_weight = (1.0 - pt) ** self.gamma

        # (2) Smoothed CE with class weighting for the actual gradient signal
        ce_smooth = F.cross_entropy(
            logits.float(),
            targets,
            weight=w,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        return (focal_weight * ce_smooth).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Gene Symbol CNN Encoder
# ─────────────────────────────────────────────────────────────────────────────
def encode_symbol(symbol: str, max_len: int = SYMBOL_MAX_LEN) -> torch.Tensor:
    """Convert gene symbol string to a character index tensor of length max_len."""
    sym_upper = symbol.upper()
    indices = [CHAR_VOCAB.get(c, 1) for c in sym_upper[:max_len]]
    indices += [0] * (max_len - len(indices))  # zero-pad
    return torch.tensor(indices, dtype=torch.long)


class SymbolCNNEncoder(nn.Module):
    """Character-level CNN encoder for gene symbol strings.

    Three parallel Conv1d branches (bigrams, trigrams, 4-grams) capture
    prefix/suffix patterns that encode gene family membership, matching
    the approach proven in node2-2 to provide genuine orthogonal signal.
    """

    def __init__(
        self,
        vocab_size: int = CHAR_VOCAB_SIZE,
        embed_dim: int = 32,
        n_channels: int = 32,
        out_dim: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Three parallel Conv1d branches
        self.conv2 = nn.Conv1d(embed_dim, n_channels, kernel_size=2)
        self.conv3 = nn.Conv1d(embed_dim, n_channels, kernel_size=3)
        self.conv4 = nn.Conv1d(embed_dim, n_channels, kernel_size=4)
        # Project concatenated branch outputs to out_dim
        self.proj = nn.Sequential(
            nn.Linear(n_channels * 3, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, symbol_ids: torch.Tensor) -> torch.Tensor:
        """
        symbol_ids: [B, SYMBOL_MAX_LEN]  int64
        returns:    [B, out_dim]         float32
        """
        emb = self.embedding(symbol_ids)        # [B, max_len, embed_dim]
        emb = emb.transpose(1, 2)               # [B, embed_dim, max_len] for Conv1d

        # Global max-pool after each conv branch
        f2 = F.relu(self.conv2(emb)).max(dim=-1)[0]   # [B, n_channels]
        f3 = F.relu(self.conv3(emb)).max(dim=-1)[0]   # [B, n_channels]
        f4 = F.relu(self.conv4(emb)).max(dim=-1)[0]   # [B, n_channels]

        cat = torch.cat([f2, f3, f4], dim=-1)          # [B, n_channels * 3]
        return self.proj(cat)                           # [B, out_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """DEG prediction dataset.

    Pre-builds:
      - AIDO.Cell expression profile tensors [N, 19264]: all genes=1.0 except
        the knocked-out gene=0.0.
      - Symbol character index tensors [N, SYMBOL_MAX_LEN].
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.is_test = is_test

        # Pre-compute expression tensors
        self.expr_inputs = self._build_expr_tensors()
        # Pre-encode symbol strings
        self.symbol_ids = torch.stack([encode_symbol(s) for s in self.symbols])

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Remap {-1, 0, 1} → {0, 1, 2}
            self.labels = np.array(raw_labels, dtype=np.int8) + 1
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

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.pert_ids[idx].split(".")[0]
        gene_pos = self.gene_to_pos.get(base, -1)
        item = {
            "idx": idx,
            "expr": self.expr_inputs[idx],          # [19264] float32
            "symbol_ids": self.symbol_ids[idx],     # [SYMBOL_MAX_LEN] int64
            "gene_pos": gene_pos,                   # int (-1 if not in vocab)
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
        "symbol_ids": torch.stack([b["symbol_ids"] for b in batch]),
        "gene_pos": torch.tensor([b["gene_pos"] for b in batch], dtype=torch.long),
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────
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

    def setup(self, stage: Optional[str] = None) -> None:
        # Rank-0 downloads tokenizer first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # Build ENSG→position mapping from all splits
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self.gene_to_pos)
            self.val_ds = PerturbationDataset(val_df, self.gene_to_pos)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(test_df, self.gene_to_pos, is_test=True)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    @staticmethod
    def _build_gene_to_pos(tokenizer, gene_ids: List[str]) -> Dict[str, int]:
        """Map each ENSG gene_id to its position index in AIDO.Cell vocabulary."""
        mapping: Dict[str, int] = {}
        PROBE_VAL = 50.0  # distinctive float to locate gene position in tokenizer output
        for gene_id in gene_ids:
            try:
                inputs = tokenizer(
                    {"gene_ids": [gene_id], "expression": [PROBE_VAL]},
                    return_tensors="pt",
                )
                ids = inputs["input_ids"]
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)   # [1, 19264]
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


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class AIDOCell10MDEGModel(nn.Module):
    """AIDO.Cell-10M + LoRA (last 4 layers) + Symbol CNN + MULTI-pooling head.

    FOUR orthogonal signals:
      (a) Global MEAN-pool of AIDO.Cell-10M hidden states → 256-dim
          Captures the average cell-state perturbation response context
      (b) Global MAX-pool of AIDO.Cell-10M hidden states → 256-dim  [NEW]
          Captures the most strongly activated gene features
      (c) Perturbed gene positional embedding → 256-dim
          Gene-in-context representation at the knockout position
      (d) Symbol CNN → 64-dim
          Gene family / naming convention features

    Combined: 832-dim → 2-layer MLP (256-dim hidden) → [B, 3, 6640]

    Key improvement over node2-3 (576-dim → 320-dim → output):
    - More expressive global representation (mean+max capture different statistics)
    - head_hidden=256 (proven optimal by node2-2 on 1,500 samples)
    - Net head params: ~5.31M (vs 6.56M in node2-3 → LESS overparameterized)
    """

    HIDDEN_DIM = 256   # AIDO.Cell-10M hidden dimension
    N_LAYERS = 8       # AIDO.Cell-10M total transformer layers
    SYMBOL_DIM = 64    # symbol CNN output dimension

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.3,
        lora_n_layers: int = 4,     # apply LoRA to last N transformer layers
        head_hidden: int = 256,
        head_dropout: float = 0.4,
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone ──────────────────────────────────────────
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA on Q/K/V of last lora_n_layers layers only
        lora_layers = list(range(self.N_LAYERS - lora_n_layers, self.N_LAYERS))
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=lora_layers,
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast trainable (LoRA) params to float32 for optimization stability
        for _name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Gene Symbol CNN encoder ─────────────────────────────────────────
        self.symbol_encoder = SymbolCNNEncoder(
            vocab_size=CHAR_VOCAB_SIZE,
            embed_dim=32,
            n_channels=32,
            out_dim=self.SYMBOL_DIM,
        )

        # ── Prediction head ─────────────────────────────────────────────────
        # Input: mean_emb (256) + max_emb (256) + pert_emb (256) + sym_feat (64) = 832
        head_in = self.HIDDEN_DIM * 3 + self.SYMBOL_DIM  # 832
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        # Conservative initialization to stabilize early training
        nn.init.trunc_normal_(self.head[1].weight, std=0.02)
        nn.init.zeros_(self.head[1].bias)
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        expr: torch.Tensor,        # [B, 19264]  float32
        gene_pos: torch.Tensor,    # [B]          int64 (-1 if not in vocab)
        symbol_ids: torch.Tensor,  # [B, max_len] int64
    ) -> torch.Tensor:
        # ── AIDO.Cell forward pass ──────────────────────────────────────────
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state   # [B, 19266, 256] (includes 2 summary tokens)

        # Extract gene positions only (exclude 2 summary tokens)
        gene_emb = lhs[:, :N_GENES_AIDO, :]          # [B, 19264, 256]

        # (a) Global MEAN-pool: captures average perturbation context
        global_mean_emb = gene_emb.mean(dim=1)         # [B, 256]

        # (b) Global MAX-pool: captures most strongly activated features [NEW]
        global_max_emb = gene_emb.max(dim=1)[0]        # [B, 256]

        # (c) Perturbed-gene positional embedding
        B = expr.shape[0]
        pert_emb = torch.zeros(B, self.HIDDEN_DIM, device=lhs.device, dtype=lhs.dtype)
        valid_mask = gene_pos >= 0
        if valid_mask.any():
            valid_pos = gene_pos[valid_mask]
            pert_emb[valid_mask] = lhs[valid_mask, valid_pos, :]
        # Fallback for genes not in AIDO.Cell vocab
        pert_emb[~valid_mask] = global_mean_emb[~valid_mask]

        # Cast bf16 backbone outputs to float32 for head computation
        global_mean_emb = global_mean_emb.float()
        global_max_emb = global_max_emb.float()
        pert_emb = pert_emb.float()

        # (d) Gene symbol CNN features
        sym_feat = self.symbol_encoder(symbol_ids)     # [B, 64] float32

        # Concatenate all four signals → [B, 832]
        combined = torch.cat([global_mean_emb, global_max_emb, pert_emb, sym_feat], dim=-1)

        logits = self.head(combined)                   # [B, 3 * 6640]
        return logits.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro F1, averaged over all N_GENES_OUT genes.

    Matches the evaluation logic in data/calc_metric.py:
      - For each gene, compute F1 over only the classes actually present.
      - Average per-gene F1 scores over all genes.

    y_pred:          [n_samples, 3, n_genes]  (3-class probability distributions)
    y_true_remapped: [n_samples, n_genes]     (labels in {0, 1, 2})
    """
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp_class = y_pred[:, :, g]          # [n_samples, 3]
        yhat = yp_class.argmax(axis=1)      # [n_samples]
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yhat, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.3,
        lora_n_layers: int = 4,
        head_hidden: int = 256,
        head_dropout: float = 0.4,
        backbone_lr: float = 5e-4,
        head_lr_multiplier: float = 3.0,
        weight_decay: float = 0.05,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 120,
        expr_noise_std: float = 0.02,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCell10MDEGModel] = None
        self.criterion: Optional[FocalLossWithSmoothing] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = AIDOCell10MDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                lora_n_layers=self.hparams.lora_n_layers,
                head_hidden=self.hparams.head_hidden,
                head_dropout=self.hparams.head_dropout,
            )
            self.criterion = FocalLossWithSmoothing(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )
        if stage == "test":
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None:
                # Ensure test data is loaded on all ranks (handles test-only DDP mode)
                if not hasattr(dm, "test_pert_ids") or not dm.test_pert_ids:
                    dm.setup("test")
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(batch["expr"], batch["gene_pos"], batch["symbol_ids"])

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_flat = labels.reshape(-1)
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Expression noise augmentation during training only
        if self.hparams.expr_noise_std > 0 and self.training:
            expr = batch["expr"].clone()
            noise = torch.randn_like(expr) * self.hparams.expr_noise_std
            # Only add noise to non-knockout positions (expr > 0.5 = SENTINEL_EXPR)
            noise_mask = (expr > 0.5).float()
            batch = dict(batch)  # shallow copy to avoid modifying original
            batch["expr"] = (expr + noise * noise_mask).clamp(0.0, 2.0)

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

        # Gather across all DDP ranks
        ap = self.all_gather(lp)
        al = self.all_gather(ll)
        ai = self.all_gather(li)
        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # Compute F1 only on global rank 0 (F1 is NOT additive, cannot be summed)
        is_global_zero = getattr(self.trainer, "is_global_zero", True)
        if is_global_zero:
            # Deduplicate (samples may appear on multiple ranks in DDP)
            preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
            labels = al.cpu().view(-1, N_GENES_OUT).numpy()
            idxs = ai.cpu().view(-1).numpy()
            _, uniq = np.unique(idxs, return_index=True)
            f1 = compute_deg_f1(preds[uniq], labels[uniq])
            f1_tensor = torch.tensor(f1, dtype=torch.float32, device=self.device)
        else:
            f1_tensor = torch.zeros(1, dtype=torch.float32, device=self.device)

        # Broadcast scalar F1 from rank 0 so all ranks have identical val_f1 (required for EarlyStopping)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(f1_tensor, src=0)
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
            out_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path}")

    def configure_optimizers(self):
        # Separate backbone LoRA params (lower LR) from head + symbol encoder (higher LR)
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        other_params = (
            list(self.model.symbol_encoder.parameters())
            + list(self.model.head.parameters())
        )
        opt = torch.optim.AdamW(
            [
                {"params": backbone_params,
                 "lr": self.hparams.backbone_lr},
                {"params": other_params,
                 "lr": self.hparams.backbone_lr * self.hparams.head_lr_multiplier},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        # LinearWarmup + CosineAnnealingLR:
        # - warmup_epochs: linear ramp from 10% to 100% of target LR
        # - remaining epochs: smooth cosine decay to eta_min
        warmup_epochs = self.hparams.warmup_epochs
        total_epochs = self.hparams.max_epochs
        cosine_epochs = max(1, total_epochs - warmup_epochs)

        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=cosine_epochs,
            eta_min=1e-7,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ── Checkpoint helpers ───────────────────────────────────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        out = {}
        trainable_keys = {name for name, p in self.named_parameters() if p.requires_grad}
        buffer_keys = {name for name, _ in self.named_buffers()}
        expected_keys = trainable_keys | buffer_keys

        for k, v in full.items():
            rel_key = k[len(prefix):] if k.startswith(prefix) else k
            if rel_key in expected_keys:
                out[k] = v

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%), "
            f"plus {total_buffers} buffer values"
        )
        return out

    def load_state_dict(self, state_dict, strict=True):
        """Load checkpoint (strict=False to handle partial checkpoints gracefully)."""
        return super().load_state_dict(state_dict, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint averaging utilities
# ─────────────────────────────────────────────────────────────────────────────
def average_checkpoints(checkpoint_paths: List[Path]) -> Dict[str, torch.Tensor]:
    """Average state_dicts from multiple Lightning checkpoint files."""
    avg_state: Dict[str, torch.Tensor] = {}
    n = len(checkpoint_paths)
    for path in checkpoint_paths:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        for k, v in state_dict.items():
            if torch.is_tensor(v):
                if k not in avg_state:
                    avg_state[k] = v.float() / n
                else:
                    avg_state[k] = avg_state[k] + v.float() / n
    return avg_state


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 2-3-1: AIDO.Cell-10M + Multi-Pooling + LinearWarmup+Cosine"
    )
    # Data
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--num-workers", type=int, default=4)
    # Batch / training
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    # LoRA
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.3)
    p.add_argument("--lora-n-layers", type=int, default=4,
                   help="Apply LoRA to last N transformer layers")
    # Head
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.4)
    # Optimizer
    p.add_argument("--backbone-lr", type=float, default=5e-4)
    p.add_argument("--head-lr-multiplier", type=float, default=3.0)
    p.add_argument("--weight-decay", type=float, default=0.05)
    # Loss
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    # LR scheduler
    p.add_argument("--warmup-epochs", type=int, default=5)
    # Augmentation
    p.add_argument("--expr-noise-std", type=float, default=0.02,
                   help="Std of Gaussian noise added to non-knockout expression during training")
    # Checkpoint averaging
    p.add_argument("--avg-top-k", type=int, default=3,
                   help="Number of top checkpoints to average for test inference "
                        "(1 = use single best checkpoint, no averaging)")
    # Debug
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
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

    if args.debug_max_step is not None:
        max_epochs_eff = 2
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        val_check_interval = 1.0
    else:
        max_epochs_eff = None  # use args.max_epochs
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        val_check_interval = args.val_check_interval

    # ── Callbacks ────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node2-3-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=args.avg_top_k,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    if args.debug_max_step is None:
        early_stop_cb = EarlyStopping(
            monitor="val_f1", mode="max",
            patience=args.early_stopping_patience,
            verbose=True,
        )
        callbacks = [checkpoint_cb, early_stop_cb, lr_monitor, progress_bar]
    else:
        callbacks = [checkpoint_cb, lr_monitor, progress_bar]

    # ── Loggers ──────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=max_epochs_eff if max_epochs_eff is not None else args.max_epochs,
        max_steps=-1,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=2,
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=False,   # FlashAttention is non-deterministic
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=0.5,  # Tighter than parent's 1.0 to stabilize head training
    )

    # ── Datamodule & LightningModule ──────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_n_layers=args.lora_n_layers,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        backbone_lr=args.backbone_lr,
        head_lr_multiplier=args.head_lr_multiplier,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        max_epochs=max_epochs_eff if max_epochs_eff is not None else args.max_epochs,
        expr_noise_std=args.expr_noise_std,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Testing (with optional top-k checkpoint averaging) ────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)

    elif args.avg_top_k > 1 and checkpoint_cb.best_k_models:
        # Sort checkpoint paths by their monitored score (val_f1) descending
        sorted_ckpts = sorted(
            checkpoint_cb.best_k_models.keys(),
            key=lambda p: checkpoint_cb.best_k_models[p],
            reverse=(checkpoint_cb.mode == "max"),
        )
        top_ckpts = [Path(p) for p in sorted_ckpts[:args.avg_top_k]]

        if len(top_ckpts) >= 2 and trainer.is_global_zero:
            avg_state = average_checkpoints(top_ckpts)
            avg_ckpt_path = output_dir / "checkpoints" / "averaged.ckpt"
            torch.save(
                {
                    "state_dict": avg_state,
                    "pytorch-lightning_version": pl.__version__,
                },
                str(avg_ckpt_path),
            )
            print(
                f"Checkpoint averaging: averaged {len(top_ckpts)} checkpoints "
                f"(val_f1 range: {min(checkpoint_cb.best_k_models.values()):.4f}"
                f"–{max(checkpoint_cb.best_k_models.values()):.4f}) "
                f"→ {avg_ckpt_path}"
            )

        # Synchronize across ranks before loading
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        avg_ckpt_path = output_dir / "checkpoints" / "averaged.ckpt"
        if avg_ckpt_path.exists():
            test_results = trainer.test(
                model_module, datamodule=datamodule, ckpt_path=str(avg_ckpt_path)
            )
        else:
            test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # ── Save test score ───────────────────────────────────────────────────────
    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_value = None
        if test_results and isinstance(test_results, list) and len(test_results) > 0:
            first_result = test_results[0]
            for key in ['test_f1', 'test_loss', 'f1', 'f1_score']:
                if key in first_result:
                    score_value = float(first_result[key])
                    break
            if score_value is None:
                for v in first_result.values():
                    if isinstance(v, (int, float)):
                        score_value = float(v)
                        break

        val_f1_best = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else None
        )

        score_path.write_text(
            f"val_f1_best: {val_f1_best}\n"
            f"test_metric: {score_value}\n"
            f"checkpoint_averaging: top-{args.avg_top_k} checkpoints\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
