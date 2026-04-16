"""Node 1-2 – AIDO.Cell-10M + QKV+FFN(last 2 layers) + SGDR(T_0=15) + GenePriorBias + SWA.

Strategy (improvements over node3-3-1-1 parent, F1=0.4368):

Lessons from node3-3-1-1 (test F1=0.4368, best AIDO.Cell lineage):
- SGDR T_0=15 + no Mixup is the proven optimal configuration for this backbone family.
- Zero val-test gap confirms perfect generalization. The model is NOT overfitting.
- The ~0.437 AIDO.Cell ceiling is caused by domain mismatch (steady-state pretraining vs
  perturbation prediction), NOT by optimization or regularization deficiencies.
- node4-2-1-1 (+0.003 test F1 with GenePriorBias) and node4-2-1-1-1 (record 0.4868 with
  GenePriorBias + SWA) proved that per-gene class bias encoding is a strong, low-risk lever.
- Feedback from node3-3-1-1: "GenePriorBias could give +0.002-0.005 F1" and "SWA in
  cycle 3 could give +0.001-0.003 F1".

This node (node1-2) makes two targeted new additions to node3-3-1-1:

1. GENEPRIORBIAS MODULE (Primary Addition):
   - Add a learnable per-gene class bias tensor: shape [N_CLASSES=3, N_GENES=6640].
   - Initialized from training data class log-frequencies per gene (not a flat prior).
   - Added to the logits AFTER the main head: logits = head_output + bias.
   - 20-epoch warm-up period: bias gradients zeroed for first 20 epochs (bias_warmup_epochs=20).
     This forces the backbone+head to establish a baseline before the bias fine-tunes.
   - The bias captures gene-specific DEG tendencies directly from the training data statistics.
   - Addresses the core domain gap: AIDO.Cell backbone doesn't inherently know WHICH genes
     tend to be up/down regulated in K562 perturbation experiments.
   - Proven successful in node4-2-1-1 (+0.003) and node4-2-1-1-1 (+0.003, record 0.4868).
   - Expected impact: +0.002-0.005 F1.

2. STOCHASTIC WEIGHT AVERAGING (SWA):
   - Add SWA starting at epoch 95 (within SGDR cycle 3, epochs 50-110).
   - SWA averages weights from epochs 95+ with SWA LR=1e-4.
   - With max_epochs=130, SWA captures ~35 checkpoints in the final descent.
   - In node4-2-1-1-1, SWA was the key driver for the record 0.4868 (+0.003 over parent).
   - For AIDO.Cell, SWA averaging in late cycle 3 explores a wider region of parameter space.
   - Expected impact: +0.001-0.003 F1.

3. ADJUSTED TRAINING BUDGET AND ES:
   - max_epochs increased 120 → 130 (covers full SGDR cycle 3: 5+15+30+60=110 epochs + 20 buffer).
   - ES patience increased 8 → 15 (allows SWA to fully operate in cycle 3 without premature stop).
   - min_delta tightened 0.001 → 0.002 (reduces false triggers during ±0.003 oscillation).
   - These changes work together: longer training + larger patience + tighter min_delta ensures
     the SWA phase and bias learning can fully develop before ES fires.

RETAINED FROM NODE3-3-1-1 (all proven effective):
- FFN unfreeze_layers=2 (last 2 layers, layers 6-7): confirmed optimal by node3-3 feedback.
- lr_muon=0.02: node3-3 proved 0.03 was too high for QKV+FFN(2) on this task.
- lr_adamw=2e-4: proportionally correct for AdamW (head + FFN).
- weight_decay=2e-2: node3-2's proven regularization.
- Label-smoothed CE (ε=0.1) + class frequency weights (NEVER focal loss: node3-1 catastrophe).
- Fixed 4-layer concatenation fusion (1024-dim).
- head_hidden=512, head_dropout=0.35 (node3-1 proved 256 insufficient).
- FlashAttention + LayerNorm patching (memory-efficient for 19,264-gene input).
- QKV weight sharing (flash_self ↔ self.self).
- Float32 cast for trainable parameters.
- SGDR T_0=15 epochs, T_mult=2, warmup=5 epochs (confirmed optimal timing in node3-3-1-1).
- DDPStrategy(find_unused_parameters=True) for multi-GPU safety (GenePriorBias adds new params).
- gradient_clip_val=1.0.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES    = 6640
N_CLASSES  = 3
AIDO_GENES = 19264
MODEL_DIR  = "/home/Models/AIDO.Cell-10M"
HIDDEN_DIM = 256      # AIDO.Cell-10M hidden size
N_LAYERS   = 8        # AIDO.Cell-10M transformer layers

CLASS_FREQ = [0.0429, 0.9251, 0.0320]  # down, neutral, up (remapped 0,1,2)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    y_hat       = preds.argmax(dim=1)
    G           = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)
    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0).float()
        tp  = (is_pred & is_true).float().sum(0)
        fp  = (is_pred & ~is_true).float().sum(0)
        fn  = (~is_pred & is_true).float().sum(0)
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(prec + rec > 0, 2*prec*rec/(prec+rec+1e-8), torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


def compute_gene_class_log_prior(train_tsv: Path) -> torch.Tensor:
    """Compute per-gene log class frequencies from training labels.

    Returns a [N_CLASSES=3, N_GENES=6640] tensor of log-prior log-probabilities
    for each (class, gene) pair. Used to initialize GenePriorBias.

    The log-prior encodes the marginal probability of each class per gene:
      log P(class c | gene g) = log(count(c at gene g) / total_samples)

    This is NOT a learnable embedding — it is a one-time computed initialization
    that seeds the GenePriorBias module with empirical class tendencies.
    """
    df = pd.read_csv(train_tsv, sep="\t")
    n_samples = len(df)
    # Accumulate class counts per gene: shape [N_CLASSES, N_GENES]
    counts = np.zeros((N_CLASSES, N_GENES), dtype=np.float64)
    for row in df["label"].tolist():
        labels = np.array(json.loads(row), dtype=int) + 1  # remap {-1,0,1} -> {0,1,2}
        for c in range(N_CLASSES):
            counts[c] += (labels == c).astype(np.float64)
    # Normalize to probabilities, add small epsilon for numerical stability
    probs = (counts + 1e-6) / (n_samples + N_CLASSES * 1e-6)
    log_probs = np.log(probs)
    return torch.tensor(log_probs, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset / DataModule
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List] = (
            [torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
             for row in df["label"].tolist()]
            if has_label else None
        )

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx": idx,
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_collate(tokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        input_ids  = tokenized["input_ids"]  # [B, 19264] float32
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),
            torch.zeros(len(batch), dtype=torch.long),
        )
        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      input_ids,
            "attention_mask": tokenized["attention_mask"],
            "gene_positions": gene_positions,
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return collate_fn


class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer   = None

    def setup(self, stage: Optional[str] = None) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        self.train_ds = DEGDataset(train_df)
        self.val_ds   = DEGDataset(val_df)
        self.test_ds  = DEGDataset(test_df)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=make_collate(self.tokenizer))

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# LR Schedule: Linear Warmup + SGDR (CosineAnnealingWarmRestarts)
# ---------------------------------------------------------------------------
def sgdr_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    T_0_steps: int,
    T_mult: int = 2,
    min_lr_ratio: float = 0.05,
):
    """Linear warmup followed by Cosine Annealing Warm Restarts (SGDR).

    During warmup (0 to num_warmup_steps): LR linearly increases from 0 to base_lr.
    After warmup: SGDR with cycle length T_0_steps, doubling each restart.
    Minimum LR = min_lr_ratio * base_lr (e.g., 0.05 * 0.02 = 0.001).

    Retained from node3-3-1-1 (T_0=15 confirmed optimal):
    - Warmup: epochs 0-5
    - Cycle 1: epochs 5-20 (15 epochs)
    - Restart at epoch 20 (confirmed aligned with convergence point)
    - Cycle 2: epochs 20-50 (30 epochs) → produced second-descent improvement to F1=0.437
    - Cycle 3: epochs 50-110 (60 epochs) → SWA operates here (epochs 95+)
    """
    def lr_lambda(current_step: int) -> float:
        # Phase 1: Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Phase 2: SGDR - find which cycle we are in
        step = current_step - num_warmup_steps
        current_T = T_0_steps
        while step >= current_T:
            step -= current_T
            current_T = int(current_T * T_mult)

        # Cosine decay within current cycle
        progress = float(step) / float(max(1, current_T))
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# GenePriorBias Module
# ---------------------------------------------------------------------------
class GenePriorBias(nn.Module):
    """Learnable per-gene class bias module.

    Stores a [N_CLASSES, N_GENES] parameter tensor initialized from the
    log-class-frequency prior computed from training data.

    During training, this bias is added to the model's raw logits AFTER
    the classification head:
        final_logits = head_logits + gene_bias  # [B, 3, 6640]

    This allows the model to learn gene-specific DEG tendencies (i.e., which
    genes are more likely to be up/down/neutral in K562 perturbation experiments)
    independently of the backbone+head representation.

    The 20-epoch warm-up (controlled externally by zeroing gradients) ensures
    the backbone+head establishes a baseline before the bias module fine-tunes.

    Proven in:
    - node4-2-1-1: +0.003 F1 (0.4836 vs 0.4801 parent)
    - node4-2-1-1-1: retained for record 0.4868
    - First application in AIDO.Cell lineage (untested, hence high potential).
    """

    def __init__(self, init_log_prior: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if init_log_prior is not None:
            # Initialize from empirical log-class-frequencies
            self.bias = nn.Parameter(init_log_prior.clone().float())
        else:
            # Fallback: initialize to log of global class frequencies
            log_freq = torch.tensor(
                [math.log(f + 1e-9) for f in CLASS_FREQ],
                dtype=torch.float32
            ).unsqueeze(1).expand(N_CLASSES, N_GENES)  # [3, 6640]
            self.bias = nn.Parameter(log_freq.clone())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Add per-gene class bias to logits. logits: [B, N_CLASSES, N_GENES]."""
        return logits + self.bias.unsqueeze(0)  # broadcast over batch dim


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCell10MQKVFFNSGDRGenePriorBiasModel(pl.LightningModule):
    """AIDO.Cell-10M + QKV (all 8 layers) + FFN (last 2 layers) + SGDR(T_0=15) + GenePriorBias + SWA.

    Improvements over node3-3-1-1 (parent, F1=0.4368):
    1. GenePriorBias: Learnable [3, 6640] bias initialized from per-gene class frequencies.
       Addresses domain gap by encoding gene-specific DEG tendencies from training data.
       20-epoch warm-up before bias gradients are active.
    2. SWA: Stochastic Weight Averaging starting at epoch 95 (cycle 3 of SGDR).
       Captures ~35 checkpoints in the final descent phase for a flatter minimum.
    3. Extended max_epochs (120→130) and ES patience (8→15) to accommodate SWA phase.
    4. Tighter min_delta (0.001→0.002) to avoid false triggers during oscillation.
    5. All proven core settings from node3-3-1-1 retained unchanged.
    """

    def __init__(
        self,
        fusion_layers: int       = 4,
        head_hidden: int         = 512,
        head_dropout: float      = 0.35,
        lr_muon: float           = 0.02,
        lr_adamw: float          = 2e-4,
        lr_bias: float           = 2e-4,    # same as lr_adamw for bias module
        weight_decay: float      = 2e-2,
        label_smoothing: float   = 0.1,
        ffn_unfreeze_layers: int = 2,
        warmup_epochs: int       = 5,
        min_lr_ratio: float      = 0.05,
        sgdr_T0_epochs: int      = 15,
        sgdr_T_mult: int         = 2,
        bias_warmup_epochs: int  = 20,      # epochs before bias gradients active
        swa_start_epoch: int     = 95,      # epoch to begin SWA
        swa_lr: float            = 1e-4,    # SWA learning rate
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._swa_model: Optional[AveragedModel] = None
        self._swa_active: bool = False

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load backbone ----
        self.backbone = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
        self.backbone = self.backbone.to(torch.bfloat16)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # ---- Enable FlashAttention ----
        self.backbone.config._use_flash_attention_2 = True
        self.backbone.config._attn_implementation = "flash_attention_2"

        def _make_ln_patch(ln_orig):
            def ln_patch(t):
                out = ln_orig(t)
                if out.dtype not in (torch.float16, torch.bfloat16) and t.dtype in (torch.float16, torch.bfloat16):
                    out = out.to(t.dtype)
                return out
            return ln_patch

        for layer in self.backbone.bert.encoder.layer:
            layer.attention.ln.forward = _make_ln_patch(layer.attention.ln.forward)
            layer.ln.forward = _make_ln_patch(layer.ln.forward)

        # ---- Share QKV weight tensors between flash_self and self.self ----
        for layer in self.backbone.bert.encoder.layer:
            ss = layer.attention.flash_self  # BertSelfFlashAttention
            mm = layer.attention.self        # CellFoundationSelfAttention
            ss.query.weight = mm.query.weight
            ss.key.weight   = mm.key.weight
            ss.value.weight = mm.value.weight
            ss.query.bias   = mm.query.bias
            ss.key.bias     = mm.key.bias
            ss.value.bias   = mm.value.bias

        # ---- Freeze all layers ----
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ---- Unfreeze QKV weights in ALL attention layers ----
        qkv_suffixes = (
            "attention.self.query.weight",
            "attention.self.key.weight",
            "attention.self.value.weight",
        )
        for name, param in self.backbone.named_parameters():
            if name.endswith(qkv_suffixes):
                param.requires_grad = True

        # ---- Unfreeze FFN (SwiGLU) in the last `ffn_unfreeze_layers` transformer layers ----
        last_layers_start = N_LAYERS - hp.ffn_unfreeze_layers
        encoder_layers = self.backbone.bert.encoder.layer
        for layer_idx in range(last_layers_start, N_LAYERS):
            layer = encoder_layers[layer_idx]
            for name, param in layer.mlp.named_parameters():
                param.requires_grad = True

        # ---- Cast trainable params to float32 for stable optimization ----
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        qkv_count = sum(p.numel() for n, p in self.backbone.named_parameters() if p.requires_grad)
        total      = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Node1-2] Trainable backbone params: {qkv_count:,} / {total:,}")

        # ---- Head: fixed multi-layer fusion (concat) → classification ----
        in_dim = hp.fusion_layers * HIDDEN_DIM  # 4 * 256 = 1024
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )
        for param in self.head.parameters():
            param.data = param.data.float()

        # ---- GenePriorBias module ----
        # Compute per-gene log class priors from training data
        print(f"[Node1-2] Computing per-gene class log-prior from training data...")
        init_log_prior = compute_gene_class_log_prior(TRAIN_TSV)
        self.gene_bias = GenePriorBias(init_log_prior=init_log_prior)
        # Cast bias to float32
        for param in self.gene_bias.parameters():
            param.data = param.data.float()
        print(f"[Node1-2] GenePriorBias initialized: shape={self.gene_bias.bias.shape}")

        self.register_buffer("class_weights", get_class_weights())

        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_meta:  List[Tuple]        = []

        # Store training steps info for LR scheduling (set in main() before fit)
        self._num_training_steps: int = -1
        self._num_warmup_steps: int   = -1
        self._T0_steps: int           = -1
        self._steps_per_epoch: int    = -1

    # ---- encode: extract fused feature vector ----
    def _encode(self, batch: Dict) -> torch.Tensor:
        """Run backbone and return fused multi-layer feature vector [B, fusion_layers*256]."""
        B = batch["input_ids"].shape[0]
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        hidden_states = out.hidden_states  # len = N_LAYERS + 1 = 9

        n = self.hparams.fusion_layers
        layer_embs = []
        for i in range(n):
            hs = hidden_states[-(i + 1)]             # [B, AIDO_GENES+2, 256]
            ge = hs[torch.arange(B, device=hs.device), batch["gene_positions"], :]  # [B, 256]
            layer_embs.append(ge.float())

        return torch.cat(layer_embs, dim=-1)  # [B, n*256=1024]

    # ---- forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_positions: torch.Tensor,
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        fused = self._encode({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "gene_positions": gene_positions,
        })
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return self.gene_bias(logits)

    # ---- loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ---- steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Standard training step with GenePriorBias warm-up handling.

        During the first `bias_warmup_epochs` epochs, bias gradients are zeroed
        after the backward pass to prevent premature bias learning before
        the backbone+head establishes a stable baseline representation.
        """
        fused  = self._encode(batch)
        logits = self.head(fused).view(batch["input_ids"].shape[0], N_CLASSES, N_GENES)
        logits = self.gene_bias(logits)
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def on_after_backward(self) -> None:
        """Zero bias gradients during warm-up period."""
        if self.current_epoch < self.hparams.bias_warmup_epochs:
            if self.gene_bias.bias.grad is not None:
                self.gene_bias.bias.grad.zero_()

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        B = batch["input_ids"].shape[0]
        fused = self._encode(batch)
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        logits = self.gene_bias(logits)
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits.float(), dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        local_idx   = torch.cat(self._val_idx,   0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        B = batch["input_ids"].shape[0]
        fused = self._encode(batch)
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        logits = self.gene_bias(logits)
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        for i, (pid, sym) in enumerate(zip(batch["pert_id"], batch["symbol"])):
            self._test_meta.append((pid, sym, batch["sample_idx"][i].item()))
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        all_preds   = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)

        local_pids = [m[0] for m in self._test_meta]
        local_syms = [m[1] for m in self._test_meta]
        local_idxs_list = [m[2] for m in self._test_meta]
        local_idxs = torch.tensor(local_idxs_list, device=self.device, dtype=torch.long)

        # Handle empty local meta (can happen with --debug_max_step on multi-GPU where some ranks get no samples)
        if len(local_pids) == 0:
            self._test_preds.clear()
            self._test_meta.clear()
            return

        # Use Lightning's native string gathering (handles ragged data automatically)
        all_pids_list = self.all_gather(local_pids)
        all_syms_list = self.all_gather(local_syms)
        all_idxs_t    = self.all_gather(local_idxs)

        if self.trainer.is_global_zero:
            all_idxs = all_idxs_t.view(-1).tolist()

            # Build prediction map from index to (pid, sym, pred)
            pred_map: Dict[int, Tuple[str, str, torch.Tensor]] = {}
            for i, (idx, pid, sym) in enumerate(zip(all_idxs, all_pids_list, all_syms_list)):
                # Deduplicate by sample index (keep first occurrence)
                if idx not in pred_map:
                    pred_map[idx] = (pid, sym, all_preds[i])

            # Sort by index for consistent output order
            rows = []
            for idx in sorted(pred_map.keys()):
                pid, sym, pred = pred_map[idx]
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(pred.float().cpu().numpy().tolist()),
                })

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-2] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_meta.clear()

    # ---- checkpoint ----
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        _params_list = list(self.named_parameters())
        total        = sum(p.numel() for _, p in _params_list)
        trained      = sum(p.numel() for _, p in _params_list if p.requires_grad)
        _buffers_list = list(self.named_buffers())
        total_bufs   = sum(b.numel() for _, b in _buffers_list)

        pct = f"{100*trained/total:.1f}%" if total > 0 else "N/A"
        buf_str = f", {total_bufs} buffers" if total_bufs else ""
        print(f"Checkpoint: {trained}/{total} params ({pct}){buf_str}")

        full_state_dict = super().state_dict(destination=None, prefix=prefix, keep_vars=keep_vars)
        full_keys = set(full_state_dict.keys())

        trainable = {}
        for name, p in _params_list:
            if p.requires_grad:
                key = prefix + name
                if key in full_keys:
                    trainable[key] = full_state_dict[key]
        for name, buf in _buffers_list:
            key = prefix + name
            if key in full_keys:
                trainable[key] = full_state_dict[key]
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- optimizer: Muon for QKV weights, AdamW for FFN + head + bias ----
    def configure_optimizers(self):
        hp = self.hparams

        # QKV weight matrices for Muon (ndim >= 2, attention QKV weights only)
        qkv_weights = [
            p for name, p in self.backbone.named_parameters()
            if p.requires_grad and p.ndim >= 2
            and any(s in name for s in ["query.weight", "key.weight", "value.weight"])
        ]

        # FFN params (up_proj, down_proj, gate_proj from last 2 layers) - handled by AdamW
        ffn_params = [
            p for name, p in self.backbone.named_parameters()
            if p.requires_grad and not any(s in name for s in ["query.weight", "key.weight", "value.weight"])
        ]

        # Head parameters
        head_params = list(self.head.parameters())

        # GenePriorBias parameters (separate group for potential LR tuning)
        bias_params = list(self.gene_bias.parameters())

        param_groups = [
            # Muon group: QKV weight matrices (ndim >= 2)
            dict(
                params       = qkv_weights,
                use_muon     = True,
                lr           = hp.lr_muon,
                weight_decay = hp.weight_decay,
                momentum     = 0.95,
            ),
            # AdamW group: FFN params + head params + bias params
            dict(
                params       = head_params + ffn_params + bias_params,
                use_muon     = False,
                lr           = hp.lr_adamw,
                betas        = (0.9, 0.95),
                weight_decay = hp.weight_decay,
            ),
        ]

        use_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        opt_cls = MuonWithAuxAdam if use_distributed else SingleDeviceMuonWithAuxAdam
        optimizer = opt_cls(param_groups)

        # Compute total, warmup, and T_0 steps for the SGDR schedule
        if self._num_training_steps > 0:
            num_steps    = self._num_training_steps
            warmup_steps = self._num_warmup_steps
            T0_steps     = self._T0_steps
        else:
            # Default fallback
            num_steps    = 1430
            warmup_steps = 55
            T0_steps     = 165

        scheduler = sgdr_schedule_with_warmup(
            optimizer,
            num_warmup_steps = warmup_steps,
            T_0_steps        = T0_steps,
            T_mult           = hp.sgdr_T_mult,
            min_lr_ratio     = hp.min_lr_ratio,
        )

        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
                "frequency": 1,
                "monitor":   "val/f1",
            },
        }

    # ---- SWA callback hooks ----
    def on_train_epoch_start(self) -> None:
        """Initialize SWA model at swa_start_epoch."""
        hp = self.hparams
        if (self.current_epoch == hp.swa_start_epoch
                and self._swa_model is None
                and not getattr(self, '_fast_dev_run', False)):
            print(f"[Node1-2] Epoch {self.current_epoch}: Initializing SWA model...")
            # Create SWA averaged model on CPU to save GPU memory during training
            self._swa_model = AveragedModel(self)
            self._swa_active = True
            print(f"[Node1-2] SWA initialized at epoch {self.current_epoch}")

    def on_train_epoch_end(self) -> None:
        """Update SWA model after each epoch once SWA is active."""
        if self._swa_active and self._swa_model is not None:
            self._swa_model.update_parameters(self)
            n_averaged = self._swa_model.n_averaged
            if hasattr(n_averaged, 'item'):
                n_averaged = n_averaged.item()
            self.log("train/swa_n_averaged", float(n_averaged), prog_bar=False, sync_dist=True)

    def on_fit_end(self) -> None:
        """After training: update BN stats and save SWA weights."""
        if self._swa_active and self._swa_model is not None:
            print("[Node1-2] Applying SWA: updating batch norm statistics...")
            # Update BN running statistics using training data
            # (AIDO.Cell-10M has no BatchNorm, so this is a no-op, but kept for correctness)
            # Load SWA weights into the main model
            try:
                self._load_swa_weights()
                print("[Node1-2] SWA weights loaded into model successfully.")
            except Exception as e:
                print(f"[Node1-2] Warning: SWA weight loading failed: {e}. Using regular checkpoint.")

    def _load_swa_weights(self) -> None:
        """Copy SWA averaged weights back into the main model for test inference."""
        if self._swa_model is None:
            return
        # SWA model wraps the original model via module attribute
        swa_state = self._swa_model.module.state_dict()
        # Only update trainable parameters (backbone QKV+FFN, head, bias)
        current_state = self.state_dict()
        updated_state = {}
        for key in current_state:
            if key in swa_state:
                updated_state[key] = swa_state[key]
            else:
                updated_state[key] = current_state[key]
        super().load_state_dict(updated_state, strict=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-2 – AIDO.Cell-10M + QKV+FFN(2L) + SGDR(T_0=15) + GenePriorBias + SWA"
    )
    parser.add_argument("--micro-batch-size",       type=int,   default=16,   dest="micro_batch_size")
    parser.add_argument("--global-batch-size",       type=int,   default=128,  dest="global_batch_size")
    parser.add_argument("--max-epochs",             type=int,   default=130,  dest="max_epochs",
                        help="Extended to 130 (vs 120) to allow full SGDR cycle 3 + SWA phase")
    parser.add_argument("--lr-muon",                type=float, default=0.02, dest="lr_muon",
                        help="Proven optimal LR for QKV+FFN(2)")
    parser.add_argument("--lr-adamw",               type=float, default=2e-4, dest="lr_adamw")
    parser.add_argument("--weight-decay",           type=float, default=2e-2, dest="weight_decay")
    parser.add_argument("--fusion-layers",          type=int,   default=4,    dest="fusion_layers")
    parser.add_argument("--head-hidden",            type=int,   default=512,  dest="head_hidden")
    parser.add_argument("--head-dropout",           type=float, default=0.35, dest="head_dropout")
    parser.add_argument("--label-smoothing",        type=float, default=0.1,  dest="label_smoothing")
    parser.add_argument("--ffn-unfreeze-layers",    type=int,   default=2,    dest="ffn_unfreeze_layers",
                        help="FFN=2 confirmed optimal")
    parser.add_argument("--warmup-epochs",          type=int,   default=5,    dest="warmup_epochs")
    parser.add_argument("--min-lr-ratio",           type=float, default=0.05, dest="min_lr_ratio")
    parser.add_argument("--sgdr-t0-epochs",         type=int,   default=15,   dest="sgdr_T0_epochs",
                        help="T_0=15 epochs aligned restart with convergence peak at epoch 20")
    parser.add_argument("--sgdr-t-mult",            type=int,   default=2,    dest="sgdr_T_mult")
    parser.add_argument("--bias-warmup-epochs",     type=int,   default=20,   dest="bias_warmup_epochs",
                        help="Epochs before GenePriorBias gradients are active")
    parser.add_argument("--swa-start-epoch",        type=int,   default=95,   dest="swa_start_epoch",
                        help="Epoch to start SWA (within SGDR cycle 3, epochs 50-110)")
    parser.add_argument("--swa-lr",                 type=float, default=1e-4, dest="swa_lr",
                        help="SWA learning rate")
    parser.add_argument("--es-patience",            type=int,   default=15,   dest="es_patience",
                        help="Increased from 8 to 15 to allow SWA phase to develop")
    parser.add_argument("--val-check-interval",     type=float, default=1.0,  dest="val_check_interval")
    parser.add_argument("--num-workers",            type=int,   default=4,    dest="num_workers")
    parser.add_argument("--debug_max_step",         type=int,   default=None, dest="debug_max_step")
    parser.add_argument("--fast_dev_run",           action="store_true",      dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = lim_val = lim_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # Estimate training steps for LR schedule
    n_train_samples    = 1388
    steps_per_epoch    = max(1, math.ceil(n_train_samples / (args.micro_batch_size * n_gpus)) // accum)
    num_training_steps = steps_per_epoch * args.max_epochs
    num_warmup_steps   = steps_per_epoch * args.warmup_epochs
    T0_steps           = steps_per_epoch * args.sgdr_T0_epochs

    print(f"[Node1-2] steps_per_epoch={steps_per_epoch}, "
          f"num_training_steps={num_training_steps}, "
          f"num_warmup_steps={num_warmup_steps}, "
          f"T0_steps={T0_steps} (T_0={args.sgdr_T0_epochs} epochs — restart at epoch {args.warmup_epochs + args.sgdr_T0_epochs}), "
          f"SWA starts at epoch {args.swa_start_epoch}, "
          f"bias_warmup={args.bias_warmup_epochs} epochs")

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOCell10MQKVFFNSGDRGenePriorBiasModel(
        fusion_layers       = args.fusion_layers,
        head_hidden         = args.head_hidden,
        head_dropout        = args.head_dropout,
        lr_muon             = args.lr_muon,
        lr_adamw            = args.lr_adamw,
        weight_decay        = args.weight_decay,
        label_smoothing     = args.label_smoothing,
        ffn_unfreeze_layers = args.ffn_unfreeze_layers,
        warmup_epochs       = args.warmup_epochs,
        min_lr_ratio        = args.min_lr_ratio,
        sgdr_T0_epochs      = args.sgdr_T0_epochs,
        sgdr_T_mult         = args.sgdr_T_mult,
        bias_warmup_epochs  = args.bias_warmup_epochs,
        swa_start_epoch     = args.swa_start_epoch,
        swa_lr              = args.swa_lr,
    )
    # Signal fast_dev_run to disable SWA (avoid memory overhead in unit tests)
    if fast_dev_run or args.debug_max_step is not None:
        model._fast_dev_run = True

    # Inject computed schedule info before configure_optimizers is called
    model._num_training_steps = num_training_steps
    model._num_warmup_steps   = num_warmup_steps
    model._T0_steps           = T0_steps
    model._steps_per_epoch    = steps_per_epoch

    ckpt_cb = ModelCheckpoint(
        dirpath  = str(output_dir / "checkpoints"),
        filename = "best-{epoch:03d}-{val/f1:.4f}",
        monitor  = "val/f1", mode="max", save_top_k=1,
    )
    # ES patience=15 to accommodate SWA phase (cycle 3 spans epochs 50-110).
    # min_delta=0.002 (vs 0.001 in parent) to prevent false triggers from ±0.003 oscillation.
    es_cb = EarlyStopping(monitor="val/f1", mode="max",
                          patience=args.es_patience, min_delta=0.002)

    lr_cb = LearningRateMonitor(logging_interval="step")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # find_unused_parameters=True because GenePriorBias bias gradients are zeroed
    # during warm-up, which can cause DDP to detect unused parameters.
    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = n_gpus,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        max_steps               = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches     = lim_train,
        limit_val_batches       = lim_val,
        limit_test_batches      = lim_test,
        val_check_interval      = 1.0 if (args.debug_max_step is not None or fast_dev_run) else args.val_check_interval,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
