"""Node 3-2-2 – AIDO.Cell-10M + QKV+FFN(2L) + SGDR + Label Noise.

Strategy (improvements over node3-2 parent):
- Retain proven AIDO.Cell-10M backbone: QKV fine-tuning (all 8 layers) + FFN last 2 layers
  (sibling node3-2-1 proved that expanding to 4 FFN layers + lr=0.03 causes regression)
- Replace broken cosine annealing → SGDR (CosineAnnealingWarmRestarts, T_0=18, T_mult=2)
  The fundamental issue: cosine decay never activates because ES fires at epoch 30-31.
  SGDR creates a restart at epoch ~23 (warmup=5 + T_0=18) that falls within the actual
  training budget and provides a second opportunity to improve from a high-LR reset.
  This directly replicates node3-3-2's winning recipe (+0.020 F1 from SGDR+label noise).
- Add label noise (3% DEG flip, training only) for data-level head regularization
  Label smoothing operates at loss level; label noise at data level. Together they provide
  complementary regularization without reducing head capacity or introducing focal loss
  (which caused catastrophic collapse in node3-1).
- Retain label-smoothed CE (NOT focal loss - sibling node3-1 proved focal loss causes collapse)
- Retain head_hidden=512 - proven sufficient; 256 insufficient (from node3-1 lessons)
- Retain head_dropout=0.35 and weight_decay=2e-2 - validated in parent
- Retain lr_muon=0.02 - higher LR (0.03) caused regression in both node3-2-1 and node3-3
- Retain fixed 4-layer concatenation fusion (1024-dim) - proven information-rich representation
- ES patience=10, min_delta=0.0005 - matching node3-3-2's proven settings
  (parent's patience=6 was appropriate but too tight for SGDR multi-cycle training)
- max_epochs=200 to accommodate SGDR C0+C1+C2 cycles (18+36+72 = 126 epoch cycles + warmup)
- Cast trainable params to float32 for stable Muon/AdamW optimization

SGDR Cycle Schedule (with warmup=5 + T_0=18, T_mult=2):
  Epoch  0- 4: warmup (LR 0→peak)
  Epoch  5-22: SGDR C0 (cosine decay from peak to min over 18 epochs)
  Epoch 23    : C0 restart → LR resets to peak (second opportunity!)
  Epoch 23-58: SGDR C1 (cosine decay from peak to min over 36 epochs)
  Epoch 59    : C1 restart
  Epoch 59-130: SGDR C2 (72 epochs)

Expected: Best checkpoint at epoch ~42 (within C1), similar to node3-3-2 (0.4496).
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
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
# SGDR (Cosine Annealing Warm Restarts) schedule with linear warmup
# ---------------------------------------------------------------------------
def sgdr_schedule_with_warmup(optimizer, warmup_steps: int, T_0_steps: int,
                               T_mult: int = 2, min_lr_ratio: float = 0.05):
    """Linear warmup then SGDR (CosineAnnealingWarmRestarts) at step level.

    After warmup:
      - Cycle 0 spans T_0_steps steps (cosine decay from 1.0 to min_lr_ratio)
      - Cycle 1 spans T_0_steps * T_mult steps
      - Cycle k spans T_0_steps * T_mult^k steps
      - At each cycle boundary, LR resets to peak (1.0 before multiplier)

    The restart at epoch (warmup_epochs + T_0) gives the model a second
    chance at high LR — a key mechanism that enabled node3-3-2's +0.020 gain.
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear ramp from 0 to 1
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # SGDR phase: determine position within current cycle
        s = current_step - warmup_steps
        T_cur = T_0_steps
        while s >= T_cur:
            s -= T_cur
            T_cur = int(T_cur * T_mult)

        progress = float(s) / float(max(1, T_cur))
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCell10MSGDR(pl.LightningModule):
    """AIDO.Cell-10M + QKV + last-2-layer FFN + SGDR + Label Noise.

    Key improvements over node3-2 parent:
    1. SGDR (CosineAnnealingWarmRestarts, T_0=18 epochs, T_mult=2) replacing
       the broken cosine decay that never activated before early stopping.
       A restart at epoch ~23 (warmup=5 + T_0=18) falls within the actual
       training budget and provides a fresh high-LR phase within which the
       model can improve. This replicates node3-3-2's winning recipe (+0.020 F1).
    2. Label noise: 3% DEG flip applied to training labels only.
       For each DEG-class label (down=0 or up=2), flip to a random class
       with 3% probability. Complements the existing label-smoothed CE by
       providing data-level head regularization without reducing model capacity.
    3. Retain parent's proven architecture: QKV all 8 layers + FFN last 2 layers.
       Sibling node3-2-1 showed that expanding to 4 FFN layers with lr=0.03
       caused regression (-3.6%). We stay with the proven 2-layer FFN setting.
    4. ES patience=10, min_delta=0.0005 to accommodate SGDR multi-cycle training.
    """

    def __init__(
        self,
        fusion_layers: int       = 4,      # last N transformer layers to fuse
        head_hidden: int         = 512,
        head_dropout: float      = 0.35,
        lr_muon: float           = 0.02,   # Muon lr for QKV weight matrices
        lr_adamw: float          = 2e-4,   # AdamW lr for head + FFN + scalar params
        weight_decay: float      = 2e-2,   # retaining parent's proven value
        label_smoothing: float   = 0.1,
        ffn_unfreeze_layers: int = 2,      # last 2 FFN layers (proven optimal)
        warmup_epochs: int       = 5,      # linear warmup before SGDR
        T_0_epochs: int          = 18,     # SGDR initial cycle length in epochs
        T_mult: int              = 2,      # SGDR cycle multiplier
        min_lr_ratio: float      = 0.05,   # SGDR cosine floor fraction
        label_noise_rate: float  = 0.03,   # DEG flip probability for label noise
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
        # This ensures both attention paths use the same trained weights.
        for layer in self.backbone.bert.encoder.layer:
            ss = layer.attention.flash_self
            mm = layer.attention.self
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
        # AIDO.Cell-10M has 8 transformer layers (0-indexed: 0..7).
        # We unfreeze the FFN modules in the last 2 layers (same as parent node3-2).
        # Sibling node3-2-1 showed 4 layers causes regression — keep 2 layers.
        last_layers_start = N_LAYERS - hp.ffn_unfreeze_layers
        encoder_layers = self.backbone.bert.encoder.layer
        for layer_idx in range(last_layers_start, N_LAYERS):
            layer = encoder_layers[layer_idx]
            for name, param in layer.mlp.named_parameters():
                param.requires_grad = True

        # ---- Cast trainable params to float32 for stable optimization ----
        # Critical for Muon's Newton-Schulz orthogonalization and AdamW stability
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        qkv_count = sum(p.numel() for n, p in self.backbone.named_parameters() if p.requires_grad)
        total      = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Node3-2-2] Trainable backbone params: {qkv_count:,} / {total:,}")

        # ---- Head: fixed multi-layer fusion (concat) → classification ----
        # Proven superior to learnable weighted sum (node3-1 lesson).
        # 4 layers × 256 = 1024 concat dim → 512 hidden → 3 × 6640 outputs.
        in_dim = hp.fusion_layers * HIDDEN_DIM  # 4 * 256 = 1024
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # Cast head params to float32
        for param in self.head.parameters():
            param.data = param.data.float()

        self.register_buffer("class_weights", get_class_weights())

        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_meta:  List[Tuple]        = []

        # Store training steps info for SGDR scheduling (set in main() before fit)
        self._num_warmup_steps: int = -1
        self._T_0_steps: int        = -1

    # ---- label noise ----
    def _apply_label_noise(self, labels: torch.Tensor) -> torch.Tensor:
        """Apply DEG-targeted label noise during training.

        For each label position where the true class is DEG (class 0=down or
        class 2=up), flip it to a randomly sampled class {0,1,2} with
        probability label_noise_rate.
        Neutral labels (class 1) are left untouched to preserve the dominant
        signal while introducing stochasticity in the minority DEG classes.
        This is data-level regularization complementing the loss-level label smoothing.
        """
        if self.hparams.label_noise_rate <= 0:
            return labels
        is_deg = (labels == 0) | (labels == 2)        # DEG positions only
        noise_mask = (torch.rand_like(labels.float()) < self.hparams.label_noise_rate) & is_deg
        random_classes = torch.randint_like(labels, 0, N_CLASSES)
        return torch.where(noise_mask, random_classes, labels)

    # ---- forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_positions: torch.Tensor,
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states: tuple of (N_LAYERS+1) tensors, each [B, AIDO_GENES+2, 256]
        hidden_states = out.hidden_states  # len = N_LAYERS + 1 = 9

        n = self.hparams.fusion_layers
        # Take the last `fusion_layers` hidden states (layers 5,6,7,8 for n=4)
        layer_embs = []
        for i in range(n):
            hs = hidden_states[-(i + 1)]             # [B, AIDO_GENES+2, 256]
            ge = hs[torch.arange(B, device=hs.device), gene_positions, :]  # [B, 256]
            layer_embs.append(ge.float())

        fused = torch.cat(layer_embs, dim=-1)        # [B, n*256=1024]
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return logits

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
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_positions"])
        labels = batch["labels"]
        # Apply label noise to DEG entries (training only, not validation)
        if self.training and self.hparams.label_noise_rate > 0:
            labels = self._apply_label_noise(labels)
        loss   = self._loss(logits, labels)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_positions"])
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
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_positions"])
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

        # Gather test metadata from all ranks so rank 0 has complete information.
        local_meta = [(pid, sym, idx) for (pid, sym, idx) in self._test_meta]
        all_meta_list = self.all_gather(local_meta)

        if all_meta_list and isinstance(all_meta_list[0], tuple):
            all_meta = list(all_meta_list)
        else:
            all_meta = []
            for rank_meta in all_meta_list:
                all_meta.extend(rank_meta)

        if self.trainer.is_global_zero:
            rows = []
            for i, meta in enumerate(all_meta):
                if i >= all_preds.shape[0]:
                    break
                pid, sym, _ = meta
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(all_preds[i].float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node3-2-2] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_meta.clear()

    # ---- checkpoint (save trainable params + buffers only) ----
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full:
                    trainable[key] = full[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full:
                trainable[key] = full[key]
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {trained}/{total} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- optimizer: Muon for QKV weights, AdamW for FFN + head ----
    def configure_optimizers(self):
        hp = self.hparams

        # QKV weight matrices for Muon (ndim >= 2, square-ish matrices)
        qkv_weights = [
            p for name, p in self.backbone.named_parameters()
            if p.requires_grad and p.ndim >= 2
            and any(s in name for s in ["query.weight", "key.weight", "value.weight"])
        ]

        # FFN params (up_proj, down_proj, gate_proj) - handled by AdamW
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
            # AdamW group: FFN params + head params + scalar backbone params
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

        # SGDR schedule with warmup
        # _num_warmup_steps and _T_0_steps are injected by main() before fit()
        if self._num_warmup_steps > 0 and self._T_0_steps > 0:
            warmup_steps = self._num_warmup_steps
            T_0_steps    = self._T_0_steps
        else:
            # Fallback for fast_dev_run / debug: assume 11 steps/epoch
            warmup_steps = hp.warmup_epochs * 11
            T_0_steps    = hp.T_0_epochs * 11

        scheduler = sgdr_schedule_with_warmup(
            optimizer,
            warmup_steps = warmup_steps,
            T_0_steps    = T_0_steps,
            T_mult       = hp.T_mult,
            min_lr_ratio = hp.min_lr_ratio,
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",   # update every step for smooth SGDR decay
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
        description="Node3-2-2 – AIDO.Cell-10M + QKV+FFN(2L) + SGDR + Label Noise"
    )
    parser.add_argument("--micro_batch_size",    type=int,   default=16)
    parser.add_argument("--global_batch_size",   type=int,   default=128)
    parser.add_argument("--max_epochs",          type=int,   default=200)
    parser.add_argument("--lr_muon",             type=float, default=0.02)
    parser.add_argument("--lr_adamw",            type=float, default=2e-4)
    parser.add_argument("--weight_decay",        type=float, default=2e-2)
    parser.add_argument("--fusion_layers",       type=int,   default=4)
    parser.add_argument("--head_hidden",         type=int,   default=512)
    parser.add_argument("--head_dropout",        type=float, default=0.35)
    parser.add_argument("--label_smoothing",     type=float, default=0.1)
    parser.add_argument("--ffn_unfreeze_layers", type=int,   default=2,
                        help="Number of last transformer layers to unfreeze FFN in")
    parser.add_argument("--warmup_epochs",       type=int,   default=5)
    parser.add_argument("--T_0_epochs",          type=int,   default=18,
                        help="SGDR initial cycle length in epochs")
    parser.add_argument("--T_mult",              type=int,   default=2,
                        help="SGDR cycle length multiplier")
    parser.add_argument("--min_lr_ratio",        type=float, default=0.05)
    parser.add_argument("--label_noise_rate",    type=float, default=0.03,
                        help="Probability of flipping DEG labels during training (0=off)")
    parser.add_argument("--val_check_interval",  type=float, default=1.0)
    parser.add_argument("--num_workers",         type=int,   default=4)
    parser.add_argument("--debug_max_step",      type=int,   default=None)
    parser.add_argument("--fast_dev_run",        action="store_true")
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

    # Compute steps per epoch for SGDR schedule
    # Train dataset: 1388 samples; with micro_batch_size=16, n_gpus GPUs, accum grad.
    # steps_per_epoch = ceil(1388 / (micro_batch_size * n_gpus)) // accum
    n_train_samples = 1388
    steps_per_epoch = max(1, math.ceil(n_train_samples / (args.micro_batch_size * n_gpus)) // accum)
    num_warmup_steps = steps_per_epoch * args.warmup_epochs
    T_0_steps        = steps_per_epoch * args.T_0_epochs

    print(f"[Node3-2-2] n_gpus={n_gpus}, accum={accum}, steps_per_epoch={steps_per_epoch}")
    print(f"[Node3-2-2] warmup_steps={num_warmup_steps} ({args.warmup_epochs} epochs)")
    print(f"[Node3-2-2] T_0_steps={T_0_steps} ({args.T_0_epochs} epochs)")
    print(f"[Node3-2-2] SGDR restart at epoch ~{args.warmup_epochs + args.T_0_epochs} "
          f"({args.warmup_epochs} warmup + {args.T_0_epochs} C0)")

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOCell10MSGDR(
        fusion_layers       = args.fusion_layers,
        head_hidden         = args.head_hidden,
        head_dropout        = args.head_dropout,
        lr_muon             = args.lr_muon,
        lr_adamw            = args.lr_adamw,
        weight_decay        = args.weight_decay,
        label_smoothing     = args.label_smoothing,
        ffn_unfreeze_layers = args.ffn_unfreeze_layers,
        warmup_epochs       = args.warmup_epochs,
        T_0_epochs          = args.T_0_epochs,
        T_mult              = args.T_mult,
        min_lr_ratio        = args.min_lr_ratio,
        label_noise_rate    = args.label_noise_rate,
    )
    # Inject computed schedule info before configure_optimizers is called
    model._num_warmup_steps = num_warmup_steps
    model._T_0_steps        = T_0_steps

    ckpt_cb = ModelCheckpoint(
        dirpath  = str(output_dir / "checkpoints"),
        filename = "best-{epoch:03d}-{val/f1:.4f}",
        monitor  = "val/f1", mode="max", save_top_k=1,
    )
    # ES patience=10 to accommodate SGDR multi-cycle training.
    # min_delta=0.0005 requires genuine improvement — matches node3-3-2's proven settings.
    # Larger patience than parent (6) is needed because the SGDR restart at C0→C1 boundary
    # provides a second improvement opportunity that pure cosine never had.
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=10, min_delta=0.0005)
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
    print(f"[Node3-2-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
