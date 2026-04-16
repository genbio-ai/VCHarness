"""Node 3-2-1-2 – AIDO.Cell-10M + QKV+FFN(2 layers, reverted) + Calibrated Cosine LR + Higher Weight Decay.

Strategy (improvements over parent node3-2-1, differentiated from sibling node3-2-1-1):

Reverts from parent node3-2-1:
- FFN fine-tuning: 4 -> 2 layers (reverted to grandparent's proven value; 4 layers caused overfitting)
- Muon base LR: 0.03 -> 0.02 (reverted; 0.03 destabilized optimization)
- Weight decay: 3e-2 -> 5e-2 (increased further; node1-1-1-1-1 achieved F1=0.4846 with wd=5e-2)

Key innovation vs ALL prior nodes:
- Cosine schedule T_max calibrated to actual training horizon (50 epochs) rather than max_epochs (150).
  This ensures meaningful LR decay during the final training epochs (LR at ~64% of base at epoch 24,
  ~48% at epoch 30) without ReduceLROnPlateau's aggressive step-wise reductions (which caused
  LR to collapse to 0.005 = 25% in sibling node3-2-1-1).

Differences from sibling node3-2-1-1:
- Cosine LR with T_max=50 (not ReduceLROnPlateau)
- Weight decay 5e-2 (not 2e-2)
- Head dropout 0.35 (not 0.45)
- No Mixup augmentation

Retained from grandparent node3-2 (proven effective):
- QKV fine-tuning all 8 layers
- FFN fine-tuning last 2 layers
- Muon LR=0.02 for QKV
- AdamW LR=2e-4 for FFN+head
- Warmup epochs=5
- Label-smoothed CE (ε=0.1) + sqrt-inverse-frequency class weights
- Fixed 4-layer concatenation fusion (1024-dim)
- head_hidden=512, head_dropout=0.35
- FlashAttention + LayerNorm patching
- Gradient checkpointing
- float32 cast for trainable params
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
# Calibrated Cosine LR schedule with linear warmup
#
# KEY INNOVATION: num_training_steps is set to cosine_T_max_epochs * steps_per_epoch
# (e.g. 50 epochs) rather than max_epochs (150). This ensures the cosine decay
# actually operates during training (~30 epochs observed) instead of being a
# near-constant schedule that never reaches its decay phase.
#
# Progress is clamped to [0, 1] so LR stays at min_lr_ratio after T_max,
# rather than oscillating back upward.
# ---------------------------------------------------------------------------
def calibrated_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,   # = cosine_T_max_epochs * steps_per_epoch
    min_lr_ratio: float = 0.05,
):
    """Linear warmup then cosine decay to min_lr_ratio, clamped at min_lr after T_max."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # CRITICAL: clamp to prevent LR from rising back up after T_max steps
        progress = min(1.0, progress)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCell10MCalibratedModel(pl.LightningModule):
    """AIDO.Cell-10M + QKV + 2-layer FFN fine-tuning + Muon + Calibrated Cosine LR.

    Key improvements over parent node3-2-1:
    1. FFN fine-tuning reverted from 4 to 2 last layers — 4 layers caused overfitting on 1388 samples.
    2. Muon base LR reverted from 0.03 to 0.02 — higher LR destabilized the optimization.
    3. Weight decay increased from 3e-2 to 5e-2 — node1-1-1-1-1 achieved F1=0.4846 with wd=5e-2;
       stronger L2 regularization improves generalization with limited training data.
    4. Cosine schedule T_max calibrated to 50 epochs rather than max_epochs (150).
       This is the PRIMARY innovation differentiating this node from both the grandparent
       and the sibling. The cosine decay now meaningfully reduces LR during the actual training
       window (~30 epochs), providing gradual convergence assistance without ReduceLROnPlateau's
       aggressive step-wise reductions.

    Differences from sibling node3-2-1-1:
    - Cosine LR with calibrated T_max=50 (not ReduceLROnPlateau which was too aggressive)
    - Weight decay 5e-2 (not 2e-2)
    - Head dropout 0.35 (not 0.45; sibling showed 0.45 hurts)
    - No Mixup augmentation (harmful in node3 lineage)
    """

    def __init__(
        self,
        fusion_layers: int         = 4,      # last N transformer layers to fuse
        head_hidden: int           = 512,
        head_dropout: float        = 0.35,
        lr_muon: float             = 0.02,   # Reverted from 0.03 to proven 0.02
        lr_adamw: float            = 2e-4,   # AdamW lr for head + FFN + non-QKV params
        weight_decay: float        = 5e-2,   # Increased from 3e-2; proven in node1 lineage
        label_smoothing: float     = 0.1,
        ffn_unfreeze_layers: int   = 2,      # Reverted from 4 to proven 2 layers
        warmup_epochs: int         = 5,      # Reverted from 3 to grandparent's proven 5
        cosine_T_max_epochs: int   = 50,     # KEY: calibrated T_max for cosine schedule
        min_lr_ratio: float        = 0.05,   # cosine floor as fraction of initial LR
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
        # AIDO.Cell-10M has 8 transformer layers (0-indexed: 0..7)
        # Using 2 layers (proven optimal): unfreezing layers 6, 7
        # This is reverted from parent's 4 layers which caused overfitting
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
        print(f"[Node3-2-1-2] Trainable backbone params: {qkv_count:,} / {total:,}")

        # ---- Head: fixed multi-layer fusion (concat) → classification ----
        # Proven: 1024-dim input (4 layers × 256-dim) as fixed concatenation
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
        self._test_idx:   List[torch.Tensor] = []  # Tensor-based tracking for DDP-safe all_gather

        # Store training step counts for LR scheduling (set in main() before training)
        self._num_cosine_steps: int = -1   # cosine_T_max_epochs * steps_per_epoch
        self._num_warmup_steps: int = -1   # warmup_epochs * steps_per_epoch

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
        # Indices: -1 (last), -2, -3, -4 → transformer layers 8,7,6,5
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
        loss   = self._loss(logits, batch["labels"])
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
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_positions"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        # Track sample indices as tensors (DDP-safe all_gather approach)
        self._test_idx.append(batch["sample_idx"].clone())
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        local_idx   = torch.cat(self._test_idx,   0)

        # DDP-safe all_gather on tensors only (not Python lists)
        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        # Sort by sample index and deduplicate
        order = torch.argsort(all_idx)
        s_idx  = all_idx[order]
        s_pred = all_preds[order]
        unique_mask = torch.ones(s_idx.numel(), dtype=torch.bool, device=s_idx.device)
        if s_idx.numel() > 1:
            unique_mask[1:] = s_idx[:-1] != s_idx[1:]
        unique_preds = s_pred[unique_mask]
        unique_idx   = s_idx[unique_mask]

        if self.trainer.is_global_zero:
            # Re-read test_df on rank 0 to retrieve metadata by sample index
            test_df = pd.read_csv(TEST_TSV, sep="\t")
            rows = []
            for rank_i in range(unique_preds.shape[0]):
                row_idx = unique_idx[rank_i].item()
                pert_id = test_df.iloc[row_idx]["pert_id"]
                symbol  = test_df.iloc[row_idx]["symbol"]
                rows.append({
                    "idx":        pert_id,
                    "input":      symbol,
                    "prediction": json.dumps(unique_preds[rank_i].float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node3-2-1-2] Saved {len(rows)} test predictions (deduplicated from {all_preds.shape[0]} gathered).")
        self._test_preds.clear()
        self._test_idx.clear()

    # ---- checkpoint: save only trainable params + buffers ----
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

        # QKV weight matrices for Muon (ndim >= 2, all-attention trainable params)
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

        # Calibrated cosine schedule:
        # - _num_cosine_steps = cosine_T_max_epochs * steps_per_epoch (e.g. 50 * 11 = 550)
        # - This is SHORTER than total training steps (max_epochs=150), ensuring the
        #   cosine decay actually operates within the ~30-epoch training window.
        # - After T_max steps, LR stays at min_lr_ratio (clamped by the lambda).
        if self._num_cosine_steps > 0:
            cosine_steps = self._num_cosine_steps
            warmup_steps = self._num_warmup_steps
        else:
            # Fallback: 50 epochs * ~11 steps/epoch = 550 cosine steps, 55 warmup
            cosine_steps = 550
            warmup_steps = 55

        scheduler = calibrated_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = warmup_steps,
            num_training_steps = cosine_steps,
            min_lr_ratio       = hp.min_lr_ratio,
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",   # update every step
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
        description="Node3-2-1-2 – AIDO.Cell-10M + QKV+FFN(2 layers, reverted) + Calibrated Cosine LR + Higher WD"
    )
    parser.add_argument("--micro_batch_size",     type=int,   default=16)
    parser.add_argument("--global_batch_size",    type=int,   default=128)
    parser.add_argument("--max_epochs",           type=int,   default=150)
    parser.add_argument("--lr_muon",              type=float, default=0.02,
                        help="Reverted from parent's 0.03 to grandparent's proven 0.02")
    parser.add_argument("--lr_adamw",             type=float, default=2e-4)
    parser.add_argument("--weight_decay",         type=float, default=5e-2,
                        help="Increased from 2e-2; node1-1-1-1-1 achieved F1=0.4846 with wd=5e-2")
    parser.add_argument("--fusion_layers",        type=int,   default=4)
    parser.add_argument("--head_hidden",          type=int,   default=512)
    parser.add_argument("--head_dropout",         type=float, default=0.35,
                        help="Reverted to grandparent's 0.35; sibling showed 0.45 hurts")
    parser.add_argument("--label_smoothing",      type=float, default=0.1)
    parser.add_argument("--ffn_unfreeze_layers",  type=int,   default=2,
                        help="Reverted from parent's 4 to proven 2 layers (last 2 of 8)")
    parser.add_argument("--warmup_epochs",        type=int,   default=5,
                        help="Reverted from parent's 3 to grandparent's proven 5")
    parser.add_argument("--cosine_T_max_epochs",  type=int,   default=50,
                        help="KEY: cosine schedule period in epochs (shorter than max_epochs to ensure decay activates)")
    parser.add_argument("--min_lr_ratio",         type=float, default=0.05)
    parser.add_argument("--val_check_interval",   type=float, default=1.0)
    parser.add_argument("--num_workers",          type=int,   default=4)
    parser.add_argument("--es_patience",          type=int,   default=8,
                        help="Early stopping patience; calibrated for ~30-epoch training with LR decay")
    parser.add_argument("--es_min_delta",         type=float, default=0.001)
    parser.add_argument("--debug_max_step",       type=int,   default=None)
    parser.add_argument("--fast_dev_run",         action="store_true")
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

    # Estimate training steps for LR schedule.
    # Train dataset has 1388 samples; with micro_batch_size=16 and n_gpus GPUs:
    #   batches per GPU per epoch = ceil(1388 / (16 * n_gpus))
    #   effective steps per epoch = ceil(batches_per_gpu / accum)
    # For 2 GPUs: 1388/(16*2)=44 batches, accum=4, steps_per_epoch=11
    # For 8 GPUs: 1388/(16*8)=11 batches, accum=1, steps_per_epoch=11
    n_train_samples   = 1388
    steps_per_epoch   = max(1, math.ceil(n_train_samples / (args.micro_batch_size * n_gpus)) // accum)

    # Calibrated cosine schedule: use cosine_T_max_epochs (=50) * steps_per_epoch
    # This is the KEY difference from prior nodes:
    # - grandparent: T_max = 150 * 11 = 1650 steps, but training stops at ~341 steps (epoch 31)
    #   → cosine only 17% through its decay at best checkpoint
    # - this node: T_max = 50 * 11 = 550 steps, training stops at ~341 steps (epoch 31)
    #   → cosine 57% through its decay at epoch 31 (meaningful LR reduction to ~42% of base)
    num_cosine_steps  = steps_per_epoch * args.cosine_T_max_epochs
    num_warmup_steps  = steps_per_epoch * args.warmup_epochs

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOCell10MCalibratedModel(
        fusion_layers       = args.fusion_layers,
        head_hidden         = args.head_hidden,
        head_dropout        = args.head_dropout,
        lr_muon             = args.lr_muon,
        lr_adamw            = args.lr_adamw,
        weight_decay        = args.weight_decay,
        label_smoothing     = args.label_smoothing,
        ffn_unfreeze_layers = args.ffn_unfreeze_layers,
        warmup_epochs       = args.warmup_epochs,
        cosine_T_max_epochs = args.cosine_T_max_epochs,
        min_lr_ratio        = args.min_lr_ratio,
    )
    # Inject computed schedule info before configure_optimizers is called
    model._num_cosine_steps = num_cosine_steps
    model._num_warmup_steps = num_warmup_steps

    ckpt_cb = ModelCheckpoint(
        dirpath  = str(output_dir / "checkpoints"),
        filename = "best-{epoch:03d}-{val/f1:.4f}",
        monitor  = "val/f1", mode="max", save_top_k=1,
    )
    es_cb = EarlyStopping(
        monitor   = "val/f1",
        mode      = "max",
        patience  = args.es_patience,
        min_delta = args.es_min_delta,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Always use DDPStrategy for consistency with torchrun (even with 1 GPU).
    # find_unused_parameters=True required because some backbone params may not
    # receive gradients in every forward pass (frozen layers, weight sharing).
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

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
    print(f"[Node3-2-1-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
