"""Node 3-3-1 – AIDO.Cell-10M + QKV+FFN(last 2 layers) + SGDR + Feature Mixup.

Strategy (improvements over node3-3 parent, reverting to node3-2 proven core + novel additions):

Lessons from node3-3 (test F1=0.4175, regression vs node3-2 F1=0.4295):
- FFN=4 layers + Muon LR=0.03 pushed optimizer into worse local minimum
- SWA scheduled at epoch 220 was never reached (early stop at epoch 45)
- Extended patience=20 allowed 20 wasted epochs after peak

This node (node3-3-1) reverts to node3-2's proven architecture and adds two novel improvements:

1. REVERT TO NODE3-2 CORE:
   - FFN unfreeze_layers: 4 → 2 (last 2 of 8 layers, layers 6-7)
   - lr_muon: 0.03 → 0.02 (proven optimal for QKV+FFN(2) config)
   - lr_adamw: 3e-4 → 2e-4 (proportional reduction)
   - weight_decay: 3e-2 → 2e-2 (node3-2's proven regularization)

2. NOVEL: SGDR (CosineAnnealingWarmRestarts) [T_0=25 epochs, T_mult=2]:
   - Replaces monotone cosine decay with warm restart cycles
   - Cycle lengths: 25, 50, 100 epochs (3 restarts in 175 epochs)
   - With max_epochs=120, completes 2 full cycles + part of cycle 3
   - After early stopping (~epoch 30-40 based on node3-2 history), model has seen
     at least one complete LR descent (epoch 0-25) + beginning of cycle 2 (LR restart)
   - Warm restarts help escape local optima that monotone cosine gets stuck in
   - Recommended explicitly in node3-3 feedback: "cosine annealing with restart"

3. NOVEL: FEATURE-LEVEL MIXUP (alpha=0.2):
   - Mixes fused 1024-dim features before the classification head
   - Mixed loss = lam * CE(logits, label_a) + (1-lam) * CE(logits, label_b)
   - Addresses the core overfitting problem: 1,388 training samples for a 13M-param model
   - Only applied during training; validation/test use standard forward pass
   - alpha=0.2 is conservative and proven in related small-dataset tasks (node1-3 tried it)
   - Recommended in node3-3 feedback: "Mixup/CutMix augmentation on perturbation embeddings"

4. TIGHTER EARLY STOPPING: patience=8, min_delta=0.001
   - node3-3's patience=20 allowed 20 wasted epochs after peak
   - node3-2's patience=6 was too tight (stopped at epoch 31 at 73% LR)
   - patience=8 balances: sufficient buffer without wasting compute
   - min_delta=0.001 (vs 0.003 in node3-3): slightly more sensitive to small improvements

5. SHORTER MAX_EPOCHS: 250 → 120
   - Captures 2 complete SGDR cycles (25 + 50 = 75 epochs) + part of cycle 3
   - With SGDR, extended training is less needed (multiple descent phases)
   - Prevents wasted compute on plateau

6. INCREASED WARMUP: 3 → 4 epochs (for stability before SGDR begins)

RETAINED FROM NODE3-2 (all proven effective):
- Label-smoothed CE (ε=0.1) + class frequency weights (NEVER focal loss)
- Fixed 4-layer concatenation fusion (1024-dim; learnable sum proved inferior in node3-1)
- head_hidden=512, head_dropout=0.35 (node3-1 proved 256 insufficient)
- FlashAttention + LayerNorm patching (memory-efficient for 19,264-gene input)
- QKV weight sharing (flash_self ↔ self.self)
- Float32 cast for trainable parameters
- DDPStrategy(find_unused_parameters=True) for multi-GPU safety
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

    Cycles (after warmup):
    - Cycle 0: T_0_steps steps (e.g., 25 epochs * 11 steps = 275 steps)
    - Cycle 1: T_0_steps * T_mult steps (e.g., 550 steps = 50 epochs)
    - Cycle 2: T_0_steps * T_mult^2 steps (e.g., 1100 steps = 100 epochs)
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
# Model
# ---------------------------------------------------------------------------
class AIDOCell10MQKVFFNSGDRMixupModel(pl.LightningModule):
    """AIDO.Cell-10M + QKV (all 8 layers) + FFN (last 2 layers) + SGDR + Feature Mixup.

    Key improvements over node3-3 (parent, F1=0.4175):
    1. FFN unfreeze_layers reverted: 4 → 2 (node3-2's proven config, F1=0.4295).
    2. Muon LR reverted: 0.03 → 0.02 (node3-3's 0.03 pushed into worse local min).
    3. SGDR schedule: warm restarts (T_0=25 epochs, T_mult=2) for multiple descent chances.
    4. Feature-level Mixup (alpha=0.2): regularize 1,388-sample overfitting.
    5. Tighter ES patience: 20 → 8 (node3-3's 20 allowed 20 useless post-peak epochs).
    6. Shorter max_epochs: 250 → 120 (covers 2 complete SGDR cycles).
    7. weight_decay: 3e-2 → 2e-2, lr_adamw: 3e-4 → 2e-4 (back to node3-2 proven values).
    8. Remove SWA (planned for epoch 220, never reached in node3-3 or node3-2).
    """

    def __init__(
        self,
        fusion_layers: int       = 4,
        head_hidden: int         = 512,
        head_dropout: float      = 0.35,
        lr_muon: float           = 0.02,    # reverted from node3-3's 0.03
        lr_adamw: float          = 2e-4,    # reverted from node3-3's 3e-4
        weight_decay: float      = 2e-2,    # reverted from node3-3's 3e-2
        label_smoothing: float   = 0.1,
        ffn_unfreeze_layers: int = 2,       # reverted from node3-3's 4
        warmup_epochs: int       = 4,       # slightly more warmup for stability
        min_lr_ratio: float      = 0.05,    # cosine floor as fraction of initial LR
        sgdr_T0_epochs: int      = 25,      # SGDR first cycle = 25 epochs
        sgdr_T_mult: int         = 2,       # SGDR cycle length multiplier
        mixup_alpha: float       = 0.2,     # Feature Mixup concentration parameter
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
        )
        for name, param in self.backbone.named_parameters():
            if name.endswith(qkv_suffixes):
                param.requires_grad = True

        # ---- Unfreeze FFN (SwiGLU) in the last `ffn_unfreeze_layers` transformer layers ----
        # Reverted to 2 layers (layers 6-7) per node3-3 feedback: FFN=4 hurt performance.
        # Node3-2 proved that FFN=2 is optimal for AIDO.Cell-10M on this 1,388-sample task.
        last_layers_start = N_LAYERS - hp.ffn_unfreeze_layers
        encoder_layers = self.backbone.bert.encoder.layer
        for layer_idx in range(last_layers_start, N_LAYERS):
            layer = encoder_layers[layer_idx]
            # Unfreeze all MLP (SwiGLU FFN) params: up_proj, down_proj, gate_proj
            for name, param in layer.mlp.named_parameters():
                param.requires_grad = True

        # ---- Cast trainable params to float32 for stable optimization ----
        # Important for both Muon's Newton-Schulz orthogonalization and AdamW stability.
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        qkv_count = sum(p.numel() for n, p in self.backbone.named_parameters() if p.requires_grad)
        total      = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Node3-3-1] Trainable backbone params: {qkv_count:,} / {total:,}")

        # ---- Head: fixed multi-layer fusion (concat) → classification ----
        # Sibling node3-1 proved learnable 256-dim weighted sum loses information.
        # Fixed 1024-dim concatenation is retained as proven effective.
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

        alpha = self.hparams.mixup_alpha
        if alpha > 0 and B >= 2 and "labels" in batch:
            # Feature-level Mixup: mix fused 1024-dim embeddings before the head.
            # A single lambda per batch (standard Mixup formulation):
            #   feat_mix = lam * feat_i + (1-lam) * feat_perm
            #   loss = lam * CE(logits, labels_i) + (1-lam) * CE(logits, labels_perm)
            lam = float(np.random.beta(alpha, alpha))
            perm = torch.randperm(B, device=fused.device)
            fused_mix = lam * fused + (1.0 - lam) * fused[perm]
            logits = self.head(fused_mix).view(B, N_CLASSES, N_GENES)
            loss = lam * self._loss(logits, batch["labels"]) + (1.0 - lam) * self._loss(logits, batch["labels"][perm])
        else:
            logits = self.head(fused).view(B, N_CLASSES, N_GENES)
            loss = self._loss(logits, batch["labels"])

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        # No Mixup during validation — use standard forward pass
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
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

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

        # Gather pert_id and symbol strings via all_gather using padded tensors.
        local_pids = [m[0] for m in self._test_meta]
        local_syms = [m[1] for m in self._test_meta]
        local_idxs = torch.tensor([m[2] for m in self._test_meta], device=self.device, dtype=torch.long)

        # Determine max string length for padding
        max_pid_len = max(len(s) for s in local_pids) if local_pids else 1
        max_sym_len = max(len(s) for s in local_syms) if local_syms else 1

        pid_char_tensor = torch.zeros(len(local_pids), max_pid_len, dtype=torch.long, device=self.device)
        for i, s in enumerate(local_pids):
            pid_char_tensor[i, :len(s)] = torch.tensor([ord(c) for c in s], device=self.device, dtype=torch.long)

        sym_char_tensor = torch.zeros(len(local_syms), max_sym_len, dtype=torch.long, device=self.device)
        for i, s in enumerate(local_syms):
            sym_char_tensor[i, :len(s)] = torch.tensor([ord(c) for c in s], device=self.device, dtype=torch.long)

        all_pids_t = self.all_gather(pid_char_tensor)
        all_syms_t = self.all_gather(sym_char_tensor)
        all_idxs_t = self.all_gather(local_idxs)

        if self.trainer.is_global_zero:
            all_pids_flat = []
            for row in all_pids_t:
                for char_vals in row:
                    chars = [chr(c.item()) for c in char_vals if c.item() > 0]
                    all_pids_flat.append("".join(chars))

            all_syms_flat = []
            for row in all_syms_t:
                for char_vals in row:
                    chars = [chr(c.item()) for c in char_vals if c.item() > 0]
                    all_syms_flat.append("".join(chars))

            all_idxs = all_idxs_t.view(-1).tolist()

            # Sort by sample_idx and deduplicate
            combined = list(zip(all_idxs, all_pids_flat, all_syms_flat,
                               [all_preds[i] for i in range(all_preds.shape[0])]))
            combined.sort(key=lambda x: x[0])
            # Remove duplicates (keep first)
            seen = set()
            rows = []
            for idx, pid, sym, pred in combined:
                if idx not in seen:
                    seen.add(idx)
                    rows.append({
                        "idx":        pid,
                        "input":      sym,
                        "prediction": json.dumps(pred.float().cpu().numpy().tolist()),
                    })

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node3-3-1] Saved {len(rows)} test predictions.")
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
            # Default fallback: 120 epochs * ~11 steps/epoch = 1320; warmup=4 epochs; T0=25 epochs
            num_steps    = 1320
            warmup_steps = 44
            T0_steps     = 275

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
        description="Node3-3-1 – AIDO.Cell-10M + QKV+FFN(2L) + SGDR + Feature Mixup"
    )
    parser.add_argument("--micro-batch-size",       type=int,   default=16,   dest="micro_batch_size")
    parser.add_argument("--global-batch-size",       type=int,   default=128,  dest="global_batch_size")
    parser.add_argument("--max-epochs",             type=int,   default=120,  dest="max_epochs",
                        help="Covers 2+ SGDR cycles (25+50=75 epochs) without over-extending")
    parser.add_argument("--lr-muon",                type=float, default=0.02, dest="lr_muon",
                        help="Reverted from node3-3's 0.03: node3-2's proven optimal LR")
    parser.add_argument("--lr-adamw",               type=float, default=2e-4, dest="lr_adamw",
                        help="Reverted from node3-3's 3e-4")
    parser.add_argument("--weight-decay",           type=float, default=2e-2, dest="weight_decay",
                        help="Reverted from node3-3's 3e-2: node3-2's proven regularization")
    parser.add_argument("--fusion-layers",          type=int,   default=4,    dest="fusion_layers")
    parser.add_argument("--head-hidden",            type=int,   default=512,  dest="head_hidden")
    parser.add_argument("--head-dropout",           type=float, default=0.35, dest="head_dropout")
    parser.add_argument("--label-smoothing",        type=float, default=0.1,  dest="label_smoothing")
    parser.add_argument("--ffn-unfreeze-layers",    type=int,   default=2,    dest="ffn_unfreeze_layers",
                        help="Reverted from node3-3's 4: node3-3 feedback confirmed FFN=2 optimal")
    parser.add_argument("--warmup-epochs",          type=int,   default=4,    dest="warmup_epochs",
                        help="Slightly extended from node3-3's 3 for better SGDR stability")
    parser.add_argument("--min-lr-ratio",           type=float, default=0.05, dest="min_lr_ratio",
                        help="SGDR minimum LR ratio (eta_min = min_lr_ratio * base_lr)")
    parser.add_argument("--sgdr-t0-epochs",         type=int,   default=25,   dest="sgdr_T0_epochs",
                        help="SGDR first cycle length in epochs (restart at epoch 25, 75, 175)")
    parser.add_argument("--sgdr-t-mult",            type=int,   default=2,    dest="sgdr_T_mult",
                        help="SGDR cycle length multiplier (each cycle T_mult times longer)")
    parser.add_argument("--mixup-alpha",            type=float, default=0.2,  dest="mixup_alpha",
                        help="Feature Mixup Beta distribution parameter (0=disabled)")
    parser.add_argument("--es-patience",            type=int,   default=8,    dest="es_patience",
                        help="Early stopping patience (reverted from 20 to 8 per node3-3 feedback)")
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

    print(f"[Node3-3-1] steps_per_epoch={steps_per_epoch}, "
          f"num_training_steps={num_training_steps}, "
          f"num_warmup_steps={num_warmup_steps}, "
          f"T0_steps={T0_steps}")

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOCell10MQKVFFNSGDRMixupModel(
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
        mixup_alpha         = args.mixup_alpha,
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
    # Early stopping: patience=8 per node3-3 feedback.
    # node3-3's patience=20 allowed 20 useless post-peak epochs.
    # node3-2's patience=6 was slightly too tight (stopped at epoch 31 at 73% LR).
    # patience=8 balances: prevents early cutoff while avoiding compute waste.
    # min_delta=0.001: slightly more sensitive than node3-3's 0.003 to catch
    # small improvements in SGDR's secondary descent phases.
    es_cb = EarlyStopping(monitor="val/f1", mode="max",
                          patience=args.es_patience, min_delta=0.001)

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
    print(f"[Node3-3-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
