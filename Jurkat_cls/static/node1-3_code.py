#!/usr/bin/env python3
"""
Node 1-3: AIDO.Cell-10M + LoRA Fine-Tuning + Dual Pooling + Character-Level CNN
===============================================================================
Strategy:
  - AIDO.Cell-10M backbone fine-tuned with LoRA (r=8, last 6 layers, Q/K/V)
  - Dual-pooling: perturbed gene positional embedding + global mean-pool over gene positions
  - Character-level CNN on gene symbol as auxiliary input
  - Wide MLP head (hidden_dim=256) with strong regularization
  - Focal loss (gamma=2.0) + class weights + label smoothing
  - AdamW + ReduceLROnPlateau (monitoring val_f1)
  - ModelCheckpoint + EarlyStopping both monitor val_f1 (critical fix from node1)
  - Gradient checkpointing for memory efficiency

Key improvements over parent node1 and siblings node1-1, node1-2:
  1. LoRA fine-tuning instead of frozen backbone (tree-best node2-2 confirmed +0.05 gain)
  2. Dual pooling: gene-specific positional + global mean (vs broken mean-pool in node1-1)
  3. Character-level CNN on gene symbol (auxiliary input)
  4. val_f1 checkpoint monitoring (critical fix from node1 feedback)
  5. Stronger regularization (dropout=0.5, weight_decay=5e-2)
  6. Wider head (256-dim) vs 128-dim bottleneck that failed in node1-2
  7. ReduceLROnPlateau on val_f1 (more stable than cosine for this task)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from peft import LoraConfig, get_peft_model, TaskType

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
N_GENES = 6_640          # output genes
N_CLASSES = 3            # {0=down, 1=unchanged, 2=up}

# AIDO.Cell-10M model path
MODEL_DIR = "/home/Models/AIDO.Cell-10M"

# Class weights: moderate (balance between node1 extreme weights and equal weights)
# Computed from inverse frequency but capped: down ~3.56%, unchanged ~94.82%, up ~1.63%
# Moderate: [3.0, 1.0, 5.0] — avoids extreme focal-loss instability seen in prior nodes
CLASS_WEIGHTS = torch.tensor([3.0, 1.0, 5.0], dtype=torch.float32)

# Character vocabulary for gene symbol CNN
CHAR_VOCAB = {c: i + 1 for i, c in enumerate(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.abcdefghijklmnopqrstuvwxyz"
)}
CHAR_VOCAB["<pad>"] = 0
MAX_SYM_LEN = 16  # longest gene symbol we expect


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weighting and label smoothing."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [N, C]
        targets: [N]  (int64, class indices)
        """
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(
            logits, targets,
            weight=w,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def encode_symbol(symbol: str, max_len: int = MAX_SYM_LEN) -> torch.Tensor:
    """Encode a gene symbol as a padded integer sequence for CNN."""
    sym = symbol.upper()[:max_len]
    ids = [CHAR_VOCAB.get(c, 0) for c in sym]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Single-fold perturbation→DEG dataset with AIDO tokenization."""

    def __init__(self, df: pd.DataFrame, tokenizer, is_test: bool = False):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.tokenizer = tokenizer
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Remap: {-1→0, 0→1, 1→2}
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pert_id = self.pert_ids[idx]
        symbol = self.symbols[idx]

        # Tokenize: set perturbed gene to 1.0 (active), all others fill as -1.0 (missing)
        # This matches AIDO.Cell's expected input distribution better than 10.0
        # NOTE: single-sample tokenizer already returns [19264] (no batch dim), not [1, 19264]
        token_inputs = self.tokenizer(
            {"gene_ids": [pert_id], "expression": [1.0]},
            return_tensors="pt",
        )
        # token_inputs["input_ids"]: [19264] float32 (no batch dim for single sample)
        input_ids = token_inputs["input_ids"]
        attention_mask = token_inputs["attention_mask"]
        # Ensure 1D shape [19264] (squeeze if batch dim accidentally added)
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)

        item = {
            "idx": idx,
            "pert_id": pert_id,
            "input_ids": input_ids,          # [19264] float32
            "attention_mask": attention_mask, # [19264] int64
            "sym_ids": encode_symbol(symbol), # [MAX_SYM_LEN]
        }

        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)  # [6640]

        return item


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, micro_batch_size: int = 4, num_workers: int = 4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.tokenizer = None
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        # Load tokenizer: rank 0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self.tokenizer)
            self.val_ds = PerturbationDataset(val_df, self.tokenizer)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(test_df, self.tokenizer, is_test=True)
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


# ──────────────────────────────────────────────────────────────────────────────
# Character-Level CNN Encoder for Gene Symbol
# ──────────────────────────────────────────────────────────────────────────────
class SymbolCNN(nn.Module):
    """Character-level CNN for gene symbol encoding."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Multi-scale convolutions
        self.conv3 = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, out_dim, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(out_dim * 2)
        self.out_dim = out_dim * 2  # 128

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, L] → [B, out_dim*2]"""
        x = self.embedding(ids)      # [B, L, embed_dim]
        x = x.transpose(1, 2)       # [B, embed_dim, L]
        f3 = F.gelu(self.conv3(x))  # [B, out_dim, L]
        f5 = F.gelu(self.conv5(x))  # [B, out_dim, L]
        # Global max pool
        f3 = f3.max(dim=-1).values  # [B, out_dim]
        f5 = f5.max(dim=-1).values  # [B, out_dim]
        out = torch.cat([f3, f5], dim=-1)  # [B, out_dim*2]
        return self.norm(out)


# ──────────────────────────────────────────────────────────────────────────────
# Main Model: AIDO.Cell-10M + LoRA + Dual Pooling + Symbol CNN + MLP Head
# ──────────────────────────────────────────────────────────────────────────────
class DEGModel(nn.Module):
    """
    AIDO.Cell-10M + LoRA fine-tuning + dual pooling + character CNN → MLP head.
    Dual pooling: [gene_pos_emb (256), global_mean_emb (256)] = 512-dim AIDO features
    Symbol CNN: 128-dim from character-level gene symbol
    Combined: 640-dim → LayerNorm → Linear(640, 256) → GELU → Dropout → LayerNorm → Linear(256, 3*6640)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_layers: Optional[List[int]] = None,  # which backbone layers get LoRA
    ):
        super().__init__()
        self.lora_layers = lora_layers  # stored for setup

        # These are set in setup() to avoid issues with distributed init
        self.backbone = None
        self.tokenizer_ref = None  # set externally after setup
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._lora_dropout = lora_dropout
        self._hidden_dim = hidden_dim
        self._dropout = dropout

        # Symbol CNN
        char_vocab_size = len(CHAR_VOCAB)
        self.symbol_cnn = SymbolCNN(char_vocab_size, embed_dim=32, out_dim=64)  # 128-dim out

        # Fused input: AIDO dual-pool (256+256=512) + symbol CNN (128) = 640
        in_dim = 512 + self.symbol_cnn.out_dim  # 640

        # Wide MLP head
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N_CLASSES * N_GENES),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def build_backbone(self):
        """Build and configure backbone with LoRA. Must be called after instantiation."""
        backbone = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)

        # CRITICAL: Cast to bfloat16 BEFORE applying LoRA.
        # The AIDO.Cell FlashAttention selection requires ln_outputs.dtype in (fp16, bf16).
        # LayerNorm always outputs float32 regardless of input dtype.
        # The model has a fix: if ln_outputs is float32, cast to query.weight.dtype.
        # So query.weight MUST be bfloat16 to activate FlashAttention.
        # Without bfloat16, the full 19264×19264 attention matrix materializes (~6 GB/layer).
        backbone = backbone.to(torch.bfloat16)

        # Configure LoRA
        lora_layers = self.lora_layers
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self._lora_r,
            lora_alpha=self._lora_alpha,
            lora_dropout=self._lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=lora_layers,  # None = all layers
        )
        backbone = get_peft_model(backbone, lora_cfg)

        # Enable gradient checkpointing
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA adapter params to float32 for stable gradient updates.
        # LoRA A/B matrices are small (r=8 rank), so float32 for optimizer states
        # provides better numerical precision for gradient accumulation.
        # The base model weights remain bfloat16 (ensures FlashAttention activation).
        for name, param in backbone.named_parameters():
            if param.requires_grad and "lora_" in name:
                param.data = param.data.float()

        self.backbone = backbone

    def get_pert_gene_pos(self, pert_id: str, tokenizer) -> Optional[int]:
        """Get the position (0-indexed) of a perturbed gene in the 19264-gene sequence.

        Uses AIDO.Cell tokenizer's gene_id_to_index mapping, which maps Ensembl IDs
        to their position in the model's 19264-gene vocabulary.
        """
        # Strip version suffix from Ensembl ID (e.g., "ENSG00000001084.5" -> "ENSG00000001084")
        eid = pert_id.split(".")[0]
        if hasattr(tokenizer, "gene_id_to_index"):
            pos = tokenizer.gene_id_to_index.get(eid)
            if pos is not None:
                return int(pos)
        if hasattr(tokenizer, "gene_to_index"):
            pos = tokenizer.gene_to_index.get(eid)
            if pos is not None:
                return int(pos)
        return None

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, 19264] float32
        attention_mask: torch.Tensor, # [B, 19264] int64
        sym_ids: torch.Tensor,        # [B, MAX_SYM_LEN] int64
        pert_positions: Optional[torch.Tensor] = None,  # [B] int64 or None
    ) -> torch.Tensor:
        """Returns logits [B, 3, N_GENES]"""
        # Run backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = outputs.last_hidden_state  # [B, 19266, 256]

        # Gene positions only (exclude the 2 appended summary tokens)
        gene_hidden = last_hidden[:, :19264, :]  # [B, 19264, 256]

        # Global mean-pool over gene positions
        mean_emb = gene_hidden.mean(dim=1)  # [B, 256]

        # Perturbed gene positional embedding
        if pert_positions is not None:
            B = input_ids.shape[0]
            # Gather the embedding at each sample's perturbed gene position
            pos_indices = pert_positions.clamp(0, 19263)  # safety clamp
            pos_idx = pos_indices.unsqueeze(1).unsqueeze(2).expand(B, 1, 256)  # [B, 1, 256]
            gene_pos_emb = gene_hidden.gather(1, pos_idx).squeeze(1)  # [B, 256]
        else:
            # Fallback: use mean-pool as substitute (for unknown genes)
            gene_pos_emb = mean_emb

        # Dual-pool fusion: [gene_pos_emb, mean_emb] = [B, 512]
        aido_features = torch.cat([gene_pos_emb, mean_emb], dim=-1)  # [B, 512]
        # Cast to float32 for stable head computation
        aido_features = aido_features.float()

        # Character CNN features
        sym_features = self.symbol_cnn(sym_ids)  # [B, 128]

        # Concatenate all features
        fused = torch.cat([aido_features, sym_features], dim=-1)  # [B, 640]

        # Head
        logits = self.head(fused)                          # [B, 3 * N_GENES]
        return logits.view(-1, N_CLASSES, N_GENES)         # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper (mirrors calc_metric.py)
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        lr: float = 5e-4,
        weight_decay: float = 5e-2,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.10,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_layers_last_n: int = 6,  # Fine-tune last N backbone layers
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[DEGModel] = None
        self.criterion: Optional[FocalLoss] = None
        self._tokenizer = None

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._pert_pos_cache: Dict[str, Optional[int]] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            # Determine which layers to apply LoRA to (last N layers of backbone)
            # AIDO.Cell-10M has 8 transformer layers (0-7)
            n_backbone_layers = 8
            lora_last_n = self.hparams.lora_layers_last_n
            if lora_last_n >= n_backbone_layers:
                lora_layers = None  # all layers
            else:
                lora_layers = list(range(n_backbone_layers - lora_last_n, n_backbone_layers))

            self.model = DEGModel(
                hidden_dim=self.hparams.hidden_dim,
                dropout=self.hparams.dropout,
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                lora_layers=lora_layers,
            )
            self.model.build_backbone()

            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

            # Cast non-backbone (head + symbol CNN) trainable params to float32 for stable optimization
            # NOTE: LoRA backbone params remain in bfloat16 to preserve FlashAttention usage
            # (FlashAttention requires bf16/fp16 input dtype to activate)
            for name, param in self.model.named_parameters():
                if param.requires_grad and "backbone" not in name:
                    param.data = param.data.float()

        # Store test metadata from DataModule
        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

        # Store tokenizer reference for position lookups
        if self.trainer is not None and hasattr(self.trainer, "datamodule"):
            dm = self.trainer.datamodule
            if dm is not None and hasattr(dm, "tokenizer") and dm.tokenizer is not None:
                self._tokenizer = dm.tokenizer

    def _get_pert_positions(self, pert_ids: List[str]) -> Optional[torch.Tensor]:
        """Get gene positions for a batch of pert_ids. Returns None if not available."""
        if self._tokenizer is None:
            return None

        positions = []
        has_any = False
        for pid in pert_ids:
            if pid not in self._pert_pos_cache:
                pos = self.model.get_pert_gene_pos(pid, self._tokenizer)
                self._pert_pos_cache[pid] = pos
            pos = self._pert_pos_cache[pid]
            if pos is not None:
                positions.append(pos)
                has_any = True
            else:
                positions.append(0)  # fallback to position 0

        if not has_any:
            return None
        return torch.tensor(positions, dtype=torch.long)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Batch stores pert_id as a list of strings under the "pert_id" key (not "pert_id_str")
        pert_ids = batch.get("pert_id", None)
        if pert_ids is None and self._test_pert_ids:
            # Fallback: during test/val, recover pert_ids from stored metadata
            pert_ids = [self._test_pert_ids[i] for i in batch["idx"].tolist()]
        if pert_ids is None:
            pert_ids_list = None
        else:
            pert_ids_list = list(pert_ids) if not isinstance(pert_ids, list) else pert_ids

        pert_positions = None
        if pert_ids_list is not None:
            pert_positions_cpu = self._get_pert_positions(pert_ids_list)
            if pert_positions_cpu is not None:
                pert_positions = pert_positions_cpu.to(batch["input_ids"].device)

        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            sym_ids=batch["sym_ids"],
            pert_positions=pert_positions,
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G], labels: [B, G] (0/1/2)"""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        labels_flat = labels.reshape(-1)                       # [B*G]
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Extract pert_ids as strings from batch
        pert_ids = batch.get("pert_id")
        if isinstance(pert_ids, (list, tuple)):
            pert_ids_list = list(pert_ids)
        else:
            pert_ids_list = None

        pert_positions = None
        if pert_ids_list is not None:
            pert_positions_cpu = self._get_pert_positions(pert_ids_list)
            if pert_positions_cpu is not None:
                pert_positions = pert_positions_cpu.to(batch["input_ids"].device)

        logits = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            sym_ids=batch["sym_ids"],
            pert_positions=pert_positions,
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        pert_ids = batch.get("pert_id")
        if isinstance(pert_ids, (list, tuple)):
            pert_ids_list = list(pert_ids)
        else:
            pert_ids_list = None

        pert_positions = None
        if pert_ids_list is not None:
            pert_positions_cpu = self._get_pert_positions(pert_ids_list)
            if pert_positions_cpu is not None:
                pert_positions = pert_positions_cpu.to(batch["input_ids"].device)

        logits = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            sym_ids=batch["sym_ids"],
            pert_positions=pert_positions,
        )
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
        local_preds = torch.cat(self._val_preds, dim=0)    # [N_local, 3, G]
        local_labels = torch.cat(self._val_labels, dim=0)  # [N_local, G]

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # Compute local F1 on each rank
        local_preds_np = local_preds.numpy()
        local_f1 = compute_deg_f1(local_preds_np, local_labels.numpy())

        # Reduce F1 across all ranks (scalar all-reduce)
        world_size = self.trainer.world_size if self.trainer.world_size else 1
        if world_size > 1:
            import torch.distributed as dist
            local_f1_t = torch.tensor(local_f1, dtype=torch.float32, device="cuda")
            dist.all_reduce(local_f1_t, op=dist.ReduceOp.SUM)
            f1 = (local_f1_t / world_size).item()
        else:
            f1 = local_f1

        self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        pert_ids = batch.get("pert_id")
        if isinstance(pert_ids, (list, tuple)):
            pert_ids_list = list(pert_ids)
        else:
            pert_ids_list = None

        pert_positions = None
        if pert_ids_list is not None:
            pert_positions_cpu = self._get_pert_positions(pert_ids_list)
            if pert_positions_cpu is not None:
                pert_positions = pert_positions_cpu.to(batch["input_ids"].device)

        logits = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            sym_ids=batch["sym_ids"],
            pert_positions=pert_positions,
        )
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

            # Reorder by original index
            order = np.argsort(sorted_idxs)
            preds = preds[order]
            final_idxs = sorted_idxs[order]

            # Write test_predictions.tsv
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
        # Separate param groups: LoRA params (higher lr), head params (standard lr)
        lora_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                lora_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": lora_params, "lr": self.hparams.lr, "weight_decay": self.hparams.weight_decay},
            {"params": head_params, "lr": self.hparams.lr * 2, "weight_decay": self.hparams.weight_decay},
        ]
        opt = torch.optim.AdamW(param_groups)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5, min_lr=1e-6,
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

    # ── Checkpoint: save only trainable params ────────────────────────────────
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
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Custom collate to preserve pert_id strings
# ──────────────────────────────────────────────────────────────────────────────
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that keeps pert_id as a list of strings."""
    result = {}
    for key in batch[0]:
        if key == "pert_id":
            result[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], int):
            result[key] = torch.tensor([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# DataModule with custom collate_fn
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModuleV2(DEGDataModule):
    """Extends DEGDataModule to use custom collate_fn that preserves pert_id strings."""

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Node 1-3: AIDO.Cell-10M + LoRA DEG predictor")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.10)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-layers-last-n", type=int, default=6,
                   help="Apply LoRA to last N backbone layers")
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Trainer settings ──────────────────────────────────────────────────────
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit_train = 1.0
    limit_val = 1.0
    limit_test = 1.0
    if args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = args.debug_max_step

    val_check_interval = args.val_check_interval if (
        args.debug_max_step is None and not args.fast_dev_run
    ) else 1.0

    if n_gpus == 1:
        strategy = SingleDeviceStrategy(device="cuda:0")
    else:
        strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-3-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
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

    # ── Data & model ──────────────────────────────────────────────────────────
    datamodule = DEGDataModuleV2(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers_last_n=args.lora_layers_last_n,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Test ──────────────────────────────────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # ── Save test score ───────────────────────────────────────────────────────
    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"test_results: {test_results}\n"
            f"val_f1_best: {checkpoint_cb.best_model_score}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
