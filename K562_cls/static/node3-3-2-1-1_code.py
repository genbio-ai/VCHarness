"""Node 3-3-2-1-1 – AIDO.Cell-10M + QKV+FFN(2L) + T_0=18 SGDR (Restored) + Higher Dropout.

Strategy: Revert all failed T_0=12 changes from parent (node3-3-2-1, F1=0.4251),
restore the proven node3-3-2 configuration (F1=0.4496), and explore increased dropout
as an alternative regularization approach.

Parent (node3-3-2-1, F1=0.4251) FAILED because:
  - T_0=12 produced only 12-epoch C0 — too short for initial convergence
    (reached F1=0.379 at C0 boundary vs grandparent's 0.415)
  - min_lr_ratio=0.10 narrowed LR range (10x vs grandparent's 20x), limiting exploration
  - head_hidden=384 may have been slightly under-capacity for the 1024→19920 mapping
  - Only 3 epochs of C2 before ES fired at epoch 44

Grandparent (node3-3-2, F1=0.4496) SUCCEEDED because:
  - T_0=18: 18-epoch C0 was sufficient for initial convergence to F1=0.415
  - 36-epoch C1 produced sustained climb from 0.413 to 0.450 (global peak at epoch 42)
  - min_lr_ratio=0.05 (LR floor 0.001): full 20x LR range within each cycle
  - head_hidden=512: sufficient capacity for complex 1024→19920 mapping

This node (node3-3-2-1-1) restores all proven settings from node3-3-2 and explores:

1. RESTORED T_0=18 (from grandparent, proven best):
   Schedule (warmup=5):
   - Warmup:  epochs 0-5
   - Cycle 0: epochs 5-23  (18 epochs, restart at epoch 23)
   - Cycle 1: epochs 23-59 (36 epochs, restart at epoch 59)
   - Cycle 2: epochs 59-131 (72 epochs, restart at epoch 131)
   This is the exact schedule that produced F1=0.4496 at epoch 42.

2. RESTORED min_lr_ratio=0.05 (from grandparent, proven best):
   LR floor = 0.001, providing 20x LR range within each cycle.
   The grandparent's 20x range was more effective than the parent's 10x range.

3. RESTORED head_hidden=512 (from grandparent, proven best):
   ~10.7M head params (vs parent's 8.0M). Sufficient capacity proven at 0.4496.

4. INCREASED head_dropout=0.42 (vs 0.35 in grandparent — KEY NOVEL CHANGE):
   The grandparent (F1=0.4496) showed a train-val loss gap of ~0.216 at best epoch (42),
   suggesting moderate overfitting remained. Increasing dropout from 0.35→0.42:
   - Provides additional structural regularization on the large 10.7M-param head
   - Complements the existing 3% label noise in a different regularization axis
   - Head dropout 0.42 has 2 fewer percentage points than the "confirmed catastrophic" 0.5
     (head_dropout=0.5 was catastrophic in some lineages but 0.42 is more moderate)
   - Expected to reduce the ~0.216 train-val gap and improve generalization
   From the memory, node3-3-2's test F1 equaled val F1 (0.4496 = 0.4496), indicating
   moderate but controlled overfitting. Increased dropout targets this residual gap.

5. LABEL NOISE RATE = 2% (vs 3% in grandparent):
   With increased structural regularization from dropout 0.42, slightly reduced label noise
   from 3%→2% balances the total regularization. The parent failed with 2% noise + reduced
   head, but that failure was primarily due to T_0=12 and min_lr_ratio=0.10. Here, 2% noise
   combined with the proven T_0=18 and head_hidden=512 (full capacity) provides:
   - More preservation of DEG signal than 3% noise
   - Still meaningful head regularization at 2%
   - Combined with dropout 0.42 as the dominant structural regularization
   (If this hurts, 3% label noise is the fallback)

6. ES PATIENCE=15, max_epochs=160:
   The grandparent (T_0=18) peaked at epoch 42 and ES fired at epoch 53 (patience=10).
   Cycle 2 started at epoch 59 — just 6 epochs after ES fired. If patience had been 15:
   - ES would have fired at epoch 57 (15 epochs after peak 42)
   - Still 2 epochs before cycle 2 started at epoch 59
   - Not sufficient to reach C2 even with patience=15
   BUT: With max_epochs=160 and patience=15:
   - The model might peak later in C1 (potentially 2-4 epochs later with dropout 0.42)
   - A later C1 peak means ES fires later, possibly allowing C2 entry
   - max_epochs=160 ensures full C2 coverage even if restart fires at epoch 59
   (C2 goes from 59 to 131, and max_epochs=160 > 131 allowing partial C3 too)

RETAINED FROM GRANDPARENT (all proven effective):
- FFN=2 layers (last 2 of 8 transformer layers, layers 6-7)
- lr_muon=0.02 (optimal for QKV+FFN(2) config)
- lr_adamw=2e-4
- weight_decay=2e-2
- Label-smoothed CE (eps=0.1) + class frequency weights (NEVER focal loss)
- Fixed 4-layer concatenation fusion (1024-dim)
- FlashAttention + LayerNorm patching
- QKV weight sharing (flash_self <-> self.self)
- Float32 cast for trainable parameters
- warmup_epochs=5
- sgdr_T_mult=2

KEY DIFFERENCE SUMMARY vs node3-3-2 (grandparent, F1=0.4496):
- head_dropout: 0.35 → 0.42 (increased, main novel change)
- label_noise_rate: 3% → 2% (slightly reduced)
- ES patience: 10 → 15 (increased for longer exploration)
- max_epochs: 120 → 160 (extended to cover C2)

KEY DIFFERENCE SUMMARY vs node3-3-2-1 (parent, F1=0.4251):
- SGDR T_0: 12 → 18 (RESTORED, primary fix)
- min_lr_ratio: 0.10 → 0.05 (RESTORED, secondary fix)
- head_hidden: 384 → 512 (RESTORED, tertiary fix)
- head_dropout: 0.35 → 0.42 (INCREASED, new exploration)
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
    """Compute per-gene macro-averaged F1 matching calc_metric.py logic."""
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

    With T_0=18 epochs and warmup=5 epochs:
    - Warmup: epochs 0-5 (LR ramp up)
    - Cycle 0: epochs 5-23  (18 epochs, restart at epoch ~23)
    - Cycle 1: epochs 23-59 (36 epochs, restart at epoch ~59)  <-- grandparent peaked here at epoch 42
    - Cycle 2: epochs 59-131 (72 epochs, restart at epoch ~131)
    - Cycle 3: epochs 131-195 (64 epochs ... incomplete in 160-epoch run)
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
# Label Noise Injection
# ---------------------------------------------------------------------------
def inject_label_noise(
    labels: torch.Tensor,
    noise_rate: float,
    training: bool = True,
) -> torch.Tensor:
    """Inject label noise into training labels by flipping ~noise_rate fraction
    of DEG labels (class 0 = down-reg and class 2 = up-reg) to neutral (class 1).

    This forces the model to avoid over-memorizing exact DEG patterns for specific
    perturbations. Only applied during training; validation/test use clean labels.

    Args:
        labels: [B, N_GENES] integer tensor with values in {0, 1, 2}
        noise_rate: fraction of DEG labels to flip to neutral (e.g., 0.02 = 2%)
        training: whether to apply noise (only during training)

    Returns:
        Noisy labels tensor (same shape and dtype as input).
    """
    if not training or noise_rate <= 0.0:
        return labels

    noisy = labels.clone()
    # Identify DEG positions: class 0 (down-reg) and class 2 (up-reg)
    is_deg = (labels != 1)   # True for down-reg (0) and up-reg (2)
    # Sample noise mask: noise_rate fraction of DEG positions
    noise_mask = is_deg & (torch.rand_like(labels.float()) < noise_rate)
    # Flip DEG labels to neutral (class 1)
    noisy[noise_mask] = 1
    return noisy


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCell10MQKVFFNSGDRRestoredModel(pl.LightningModule):
    """AIDO.Cell-10M + QKV (all 8 layers) + FFN (last 2 layers) + Restored T_0=18 SGDR + Higher Dropout.

    This node restores the proven grandparent (node3-3-2, F1=0.4496) configuration
    and explores increased head dropout (0.35→0.42) as alternative regularization.

    Key differences from node3-3-2 (grandparent, F1=0.4496):
    1. head_dropout: 0.35 → 0.42 (increased for stronger head regularization)
    2. label_noise_rate: 3% → 2% (slightly reduced, balanced by higher dropout)
    3. ES patience: 10 → 15 (extended for better cycle coverage)
    4. max_epochs: 120 → 160 (extended to cover C2 = 72 epochs, starting at epoch 59)

    Restored from failed parent (node3-3-2-1, F1=0.4251):
    - T_0: 12 → 18 (critical: longer C0 allows proper initial convergence)
    - min_lr_ratio: 0.10 → 0.05 (restored 20x LR range)
    - head_hidden: 384 → 512 (restored proven capacity)
    """

    def __init__(
        self,
        fusion_layers: int       = 4,
        head_hidden: int         = 512,    # restored from 384 (grandparent proven optimal)
        head_dropout: float      = 0.42,   # INCREASED from 0.35 to 0.42 (key novel change)
        lr_muon: float           = 0.02,   # proven optimal
        lr_adamw: float          = 2e-4,   # proven optimal
        weight_decay: float      = 2e-2,   # proven optimal
        label_smoothing: float   = 0.1,
        ffn_unfreeze_layers: int = 2,      # proven optimal
        warmup_epochs: int       = 5,      # proven optimal
        min_lr_ratio: float      = 0.05,   # restored from 0.10 (grandparent proven 20x range)
        sgdr_T0_epochs: int      = 18,     # restored from 12 (grandparent proven C0=18)
        sgdr_T_mult: int         = 2,      # SGDR cycle length multiplier
        label_noise_rate: float  = 0.02,   # slightly reduced from 3%; balanced by dropout 0.42
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load backbone ----
        self.backbone = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
        self.backbone = self.backbone.to(torch.bfloat16)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # ---- Enable FlashAttention to avoid OOM from full-attention matmul ----
        # Lightning's bf16-mixed autocast causes LayerNorm outputs to be float32
        # (LayerNorm not on PyTorch autocast whitelist), breaking the attention's
        # dtype check and falling back to standard matmul (OOMs on 19266 seq len).
        # We patch every LayerNorm in the encoder to always output bf16/fp16.
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
        # After enabling flash_attn, each attention layer has a separate
        # flash_self (BertSelfFlashAttention). For QKV-only fine-tuning we only
        # train self.self's QKV weights, so we make flash_self's QKV tensors
        # alias self.self's QKV tensors so both paths always see the same weights.
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
            "attention.self.query.bias",
            "attention.self.key.bias",
            "attention.self.value.bias",
        )
        for name, param in self.backbone.named_parameters():
            if name.endswith(qkv_suffixes):
                param.requires_grad = True

        # ---- Unfreeze FFN (SwiGLU) in the last `ffn_unfreeze_layers` transformer layers ----
        # Retained at 2 layers (layers 6-7) — node3-3 proved FFN=4 hurts performance.
        last_layers_start = N_LAYERS - hp.ffn_unfreeze_layers
        encoder_layers = self.backbone.bert.encoder.layer
        for layer_idx in range(last_layers_start, N_LAYERS):
            layer = encoder_layers[layer_idx]
            # Unfreeze all MLP (SwiGLU FFN) params: up_proj, down_proj, gate_proj
            for name, param in layer.mlp.named_parameters():
                param.requires_grad = True

        # ---- Cast trainable params to float32 for stable optimization ----
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        qkv_count = sum(p.numel() for n, p in self.backbone.named_parameters() if p.requires_grad)
        total      = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Node3-3-2-1-1] Trainable backbone params: {qkv_count:,} / {total:,}")

        # ---- Head: fixed multi-layer fusion (concat) → classification ----
        # Restored head_hidden=512 (vs parent's 384) for proven capacity.
        # INCREASED head_dropout=0.42 (vs grandparent's 0.35) for additional regularization.
        # Head param count: 512 hidden → ~10.7M params (same as grandparent proven value).
        in_dim = hp.fusion_layers * HIDDEN_DIM  # 4 * 256 = 1024
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # Cast head params to float32 for stable optimization
        for param in self.head.parameters():
            param.data = param.data.float()

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

    # ---- encode: extract fused feature vector ----
    def _encode(self, batch: Dict) -> torch.Tensor:
        """Run backbone and return fused multi-layer feature vector [B, fusion_layers*256]."""
        B = batch["input_ids"].shape[0]
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        # hidden_states: tuple of (N_LAYERS+1) tensors, each [B, AIDO_GENES+2, 256]
        hidden_states = out.hidden_states  # len = N_LAYERS + 1 = 9

        n = self.hparams.fusion_layers
        layer_embs = []
        for i in range(n):
            hs = hidden_states[-(i + 1)]             # [B, AIDO_GENES+2, 256]
            # Extract at perturbed gene position
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
        return self.head(fused).view(B, N_CLASSES, N_GENES)

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
        B = batch["input_ids"].shape[0]
        fused = self._encode(batch)
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)

        # Apply label noise injection during training only
        # Flip ~noise_rate fraction of DEG labels (class 0 or 2) to neutral (class 1)
        # to prevent the model from memorizing exact DEG patterns for each perturbation.
        if "labels" not in batch:
            # Should not happen in training, but guard defensively
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
        noisy_labels = inject_label_noise(
            batch["labels"],
            noise_rate=self.hparams.label_noise_rate,
            training=True,
        )
        loss = self._loss(logits, noisy_labels)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        # No label noise during validation — use clean labels
        B = batch["input_ids"].shape[0]
        fused = self._encode(batch)
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
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
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        B = batch["input_ids"].shape[0]
        fused = self._encode(batch)
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
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

        # Gather pert_id and symbol strings via all_gather_object
        local_pids = [m[0] for m in self._test_meta]
        local_syms = [m[1] for m in self._test_meta]
        local_idxs = torch.tensor([m[2] for m in self._test_meta], device=self.device, dtype=torch.long)

        all_pids_gathered: List[str] = []
        all_syms_gathered: List[str] = []
        if torch.distributed.is_initialized():
            _world_size = torch.distributed.get_world_size()
            _pids_list: List[List[str]] = [local_pids] + [[] for _ in range(_world_size - 1)]
            _syms_list: List[List[str]] = [local_syms] + [[] for _ in range(_world_size - 1)]
            torch.distributed.all_gather_object(_pids_list, local_pids)
            torch.distributed.all_gather_object(_syms_list, local_syms)
            for bucket in _pids_list:
                all_pids_gathered.extend(bucket)
            for bucket in _syms_list:
                all_syms_gathered.extend(bucket)
        else:
            all_pids_gathered = local_pids
            all_syms_gathered = local_syms
        all_idxs_t = self.all_gather(local_idxs)

        if self.trainer.is_global_zero:
            all_idxs = all_idxs_t.view(-1).tolist()

            # Sort by sample_idx and deduplicate
            sorted_order = sorted(range(len(all_idxs)), key=lambda i: all_idxs[i])
            sorted_idxs  = [all_idxs[i] for i in sorted_order]
            sorted_pids  = [all_pids_gathered[i] for i in sorted_order]
            sorted_syms  = [all_syms_gathered[i] for i in sorted_order]
            sorted_preds = all_preds[sorted_order]  # [total_samples, 3, 6640]

            seen = set()
            rows = []
            for j, (idx, pid, sym) in enumerate(zip(sorted_idxs, sorted_pids, sorted_syms)):
                if idx not in seen:
                    seen.add(idx)
                    rows.append({
                        "idx":        pid,
                        "input":      sym,
                        "prediction": json.dumps(sorted_preds[j].float().cpu().numpy().tolist()),
                    })

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node3-3-2-1-1] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_meta.clear()

    # ---- checkpoint: save only trainable params ----
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

    # ---- optimizer: Muon for QKV weights, AdamW for FFN + head ----
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

        param_groups = [
            # Muon group: QKV weight matrices (ndim >= 2)
            dict(
                params       = qkv_weights,
                use_muon     = True,
                lr           = hp.lr_muon,
                weight_decay = hp.weight_decay,
                momentum     = 0.95,
            ),
            # AdamW group: FFN params + head params
            dict(
                params       = head_params + ffn_params,
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
            # Default fallback: 160 epochs * ~11 steps/epoch = 1760; warmup=5 epochs; T0=18 epochs
            num_steps    = 1760
            warmup_steps = 55
            T0_steps     = 198

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
                "interval":  "step",   # update every step not epoch
                "frequency": 1,
                "monitor":   "val/f1",
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node3-3-2-1-1 – AIDO.Cell-10M + QKV+FFN(2L) + Restored T_0=18 SGDR + Higher Dropout"
    )
    parser.add_argument("--micro-batch-size",       type=int,   default=16,   dest="micro_batch_size")
    parser.add_argument("--global-batch-size",       type=int,   default=128,  dest="global_batch_size")
    parser.add_argument("--max-epochs",             type=int,   default=160,  dest="max_epochs",
                        help="Covers warmup(5)+C0(18)+C1(36)+C2(72)+partial C3; 160 epochs total")
    parser.add_argument("--lr-muon",                type=float, default=0.02, dest="lr_muon",
                        help="Proven optimal Muon LR for QKV+FFN(2) config")
    parser.add_argument("--lr-adamw",               type=float, default=2e-4, dest="lr_adamw",
                        help="Proven optimal AdamW LR for FFN+head")
    parser.add_argument("--weight-decay",           type=float, default=2e-2, dest="weight_decay",
                        help="Proven optimal weight decay")
    parser.add_argument("--fusion-layers",          type=int,   default=4,    dest="fusion_layers")
    parser.add_argument("--head-hidden",            type=int,   default=512,  dest="head_hidden",
                        help="Restored from 384 to 512 (grandparent proven optimal)")
    parser.add_argument("--head-dropout",           type=float, default=0.42, dest="head_dropout",
                        help="Increased from 0.35 to 0.42 for additional regularization")
    parser.add_argument("--label-smoothing",        type=float, default=0.1,  dest="label_smoothing")
    parser.add_argument("--ffn-unfreeze-layers",    type=int,   default=2,    dest="ffn_unfreeze_layers",
                        help="Number of last transformer layers to unfreeze FFN in (proven: 2)")
    parser.add_argument("--warmup-epochs",          type=int,   default=5,    dest="warmup_epochs",
                        help="Proven optimal 5 epochs warmup")
    parser.add_argument("--min-lr-ratio",           type=float, default=0.05, dest="min_lr_ratio",
                        help="SGDR minimum LR ratio (eta_min = min_lr_ratio * base_lr), restored 0.05")
    parser.add_argument("--sgdr-t0-epochs",         type=int,   default=18,   dest="sgdr_T0_epochs",
                        help="SGDR first cycle length in epochs (T0=18 restored from grandparent)")
    parser.add_argument("--sgdr-t-mult",            type=int,   default=2,    dest="sgdr_T_mult",
                        help="SGDR cycle length multiplier (each cycle T_mult times longer)")
    parser.add_argument("--label-noise-rate",       type=float, default=0.02, dest="label_noise_rate",
                        help="Fraction of DEG training labels flipped to neutral (2% with dropout 0.42)")
    parser.add_argument("--es-patience",            type=int,   default=15,   dest="es_patience",
                        help="Early stopping patience (15, increased from grandparent's 10)")
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
    # Train dataset has 1388 samples; with micro_batch_size=16 and accumulation,
    # effective steps_per_epoch depends on n_gpus and accum.
    # For 2 GPUs: 1388/(16*2)=44 batches per GPU, accum=4, steps_per_epoch=11 steps
    # For 1 GPU:  1388/16=87 batches, accum=8, steps_per_epoch=11 steps
    n_train_samples    = 1388
    steps_per_epoch    = max(1, math.ceil(n_train_samples / (args.micro_batch_size * n_gpus)) // accum)
    num_training_steps = steps_per_epoch * args.max_epochs
    num_warmup_steps   = steps_per_epoch * args.warmup_epochs
    T0_steps           = steps_per_epoch * args.sgdr_T0_epochs

    print(f"[Node3-3-2-1-1] n_gpus={n_gpus}, steps_per_epoch={steps_per_epoch}, "
          f"num_training_steps={num_training_steps}, "
          f"num_warmup_steps={num_warmup_steps}, T0_steps={T0_steps}")
    print(f"[Node3-3-2-1-1] SGDR restart schedule (epochs from start):")
    print(f"  Warmup:  0 → {args.warmup_epochs}")
    print(f"  Cycle 0: {args.warmup_epochs} → {args.warmup_epochs + args.sgdr_T0_epochs} (restart at epoch {args.warmup_epochs + args.sgdr_T0_epochs})")
    cycle1_len = args.sgdr_T0_epochs * args.sgdr_T_mult
    cycle1_end = args.warmup_epochs + args.sgdr_T0_epochs + cycle1_len
    cycle2_len = cycle1_len * args.sgdr_T_mult
    cycle2_end = cycle1_end + cycle2_len
    print(f"  Cycle 1: {args.warmup_epochs + args.sgdr_T0_epochs} → {cycle1_end} (restart at epoch {cycle1_end}, {cycle1_len} epochs)")
    print(f"  Cycle 2: {cycle1_end} → {cycle2_end} (restart at epoch {cycle2_end}, {cycle2_len} epochs)")
    print(f"  → Key: C1 has {cycle1_len} epochs (matches grandparent's proven 36-epoch C1)")
    print(f"  → Key: C2 has {cycle2_len} epochs (extends beyond grandparent's unused C2)")
    print(f"[Node3-3-2-1-1] Key changes vs grandparent (F1=0.4496):")
    print(f"  head_dropout: 0.35 → {args.head_dropout} (increased for regularization)")
    print(f"  label_noise_rate: 3% → {args.label_noise_rate*100:.0f}% (slightly reduced)")
    print(f"  ES patience: 10 → {args.es_patience} (extended)")
    print(f"  max_epochs: 120 → {args.max_epochs} (C2 coverage)")

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOCell10MQKVFFNSGDRRestoredModel(
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
        label_noise_rate    = args.label_noise_rate,
    )
    # Inject computed schedule info before configure_optimizers is called
    model._num_training_steps = num_training_steps
    model._num_warmup_steps   = num_warmup_steps
    model._T0_steps           = T0_steps

    ckpt_cb = ModelCheckpoint(
        dirpath  = str(output_dir / "checkpoints"),
        filename = "best-{epoch:03d}-{val/f1:.4f}",
        monitor  = "val/f1", mode="max", save_top_k=1,
    )
    # ES patience=15 (vs grandparent's 10, vs parent's 12).
    # Rationale: With T_0=18 (same as grandparent), the model's best was at epoch 42 and
    # ES fired at epoch 53. With dropout 0.42, the peak may arrive 2-4 epochs later (epoch 44-46),
    # and ES patience=15 gives more room to explore beyond the C1 peak.
    # min_delta=0.0005 retained from grandparent for sensitivity to small improvements.
    es_cb = EarlyStopping(monitor="val/f1", mode="max",
                          patience=args.es_patience, min_delta=0.0005)

    lr_cb = LearningRateMonitor(logging_interval="step")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    strategy = (
        DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120))
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
    print(f"[Node3-3-2-1-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
