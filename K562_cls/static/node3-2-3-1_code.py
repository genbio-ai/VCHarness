"""Node 3-2-3-1 – AIDO.Cell-10M + SGDR (T_0=15) + Label Noise + Rebalanced Regularization.

Strategy (improvements over node3-2-3 parent):
- Retain proven AIDO.Cell-10M backbone with QKV (all 8 layers) + FFN (last 2 layers) fine-tuning
- Retain SGDR schedule but reduce T_0 from 18 to 15 epochs:
    * Earlier C1 restart (epoch 20 vs epoch 23 in parent)
    * node3-3-1-1 demonstrated T_0=15 achieved F1=0.4368 (vs T_0=18 in this lineage)
    * Shorter C0 phase means model reaches cosine minimum sooner and gets an earlier
      fresh high-LR opportunity, potentially improving before val F1 plateaus
- Reduce head_weight_decay from 5e-2 to 3.5e-2 (the "Goldilocks" middle ground):
    * 5e-2 moderated post-restart overfitting (parent's 0.005 drop vs sibling's 0.038)
      but still prevented the model from climbing during C1 (parent feedback diagnosis)
    * 2e-2 (sibling node3-2-2) was too weak — catastrophic 0.038 drop at restart
    * 3.5e-2 allows the head to learn more aggressively during C1's high-LR phase
      while still providing meaningful regularization against catastrophic memorization
- Reduce head_dropout from 0.40 to 0.38:
    * Combined with lower head_wd (3.5e-2 vs 5e-2), maintains total regularization level
    * Allows slightly more gradient signal through the head during C1 high-LR phase
- Add StochasticDepth (DropPath) at rate 0.10 on backbone transformer layers:
    * Provides structural regularization complementary to weight decay
    * Reduces backbone co-adaptation from small dataset (1,388 samples)
    * Proven effective for small-data regimes in the original stochastic depth paper
    * Complements rather than competes with head-level regularization
- Extend ES patience from 12 to 15 for more room during C1 exploration
- Retain: label_noise=0.03, label_smoothing=0.1, lr_muon=0.02, lr_adamw=2e-4,
  backbone_wd=2e-2, FFN 2 layers, 4-layer concat fusion, head_hidden=512
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
import torch.distributed as dist
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
# StochasticDepth (DropPath) wrapper for transformer layers
# ---------------------------------------------------------------------------
class StochasticDepthResidualWrapper(nn.Module):
    """Wraps a transformer layer to apply stochastic depth (DropPath).

    During training, each layer is dropped with probability `drop_prob`.
    During evaluation, the layer always executes (deterministic).
    This provides structural regularization complementary to weight decay.
    """

    def __init__(self, layer: nn.Module, drop_prob: float = 0.1) -> None:
        super().__init__()
        self.layer = layer
        self.drop_prob = drop_prob

    def forward(self, *args, **kwargs):
        """Apply stochastic depth: with probability drop_prob, skip this layer.

        Accepts *args and **kwargs so it is transparent to any calling signature
        (the CellFoundationEncoder passes up to 9 positional arguments).
        The first positional argument is always `hidden_states`.
        """
        hidden_states = args[0]
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1.0 - self.drop_prob
            # Sample batch-wise mask: [B, 1, 1] for broadcasting
            batch_size = hidden_states.shape[0]
            mask = torch.rand(batch_size, 1, 1, device=hidden_states.device) < keep_prob
            # Cast mask to match hidden_states dtype to avoid float32 promotion.
            # If mask stays float32 and hidden_states is bfloat16, the multiplication
            # promotes hidden_states to float32, which breaks the FlashAttention
            # dtype check (requires fp16/bf16) in all subsequent transformer layers.
            mask = mask.to(hidden_states.dtype)
            # Apply layer and mix with residual via stochastic depth
            # We do this by: output = hidden_states + mask * (layer(hidden_states) - hidden_states)
            # Which simplifies to: if mask=1, output = layer(hidden_states); if mask=0, output = hidden_states
            layer_out = self.layer(*args, **kwargs)
            if isinstance(layer_out, tuple):
                # Some layers return (hidden_states, ...) tuples
                layer_hidden = layer_out[0]
                layer_hidden = hidden_states + mask * (layer_hidden - hidden_states)
                return (layer_hidden,) + layer_out[1:]
            else:
                return hidden_states + mask * (layer_out - hidden_states)
        else:
            return self.layer(*args, **kwargs)


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
# SGDR schedule with linear warmup
# ---------------------------------------------------------------------------
def sgdr_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    steps_per_epoch: int,
    T_0_epochs: int,
    T_mult: int = 2,
    min_lr_ratio: float = 0.05,
):
    """Linear warmup followed by SGDR (CosineAnnealingWarmRestarts).

    The warmup ramps LR from 0 to 1.0 over num_warmup_steps steps.
    After warmup, SGDR cosine decay with warm restarts every T_0_epochs * steps_per_epoch
    steps (with T_mult geometric growth of cycle length).

    - C0: T_0 epochs → decays from 1.0 to min_lr_ratio
    - C1: T_0 * T_mult epochs → restart at 1.0, decay to min_lr_ratio
    - C2: T_0 * T_mult^2 epochs → etc.
    """
    T_0_steps = T_0_epochs * steps_per_epoch

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Post-warmup: SGDR
        step_after_warmup = current_step - num_warmup_steps

        # Find which SGDR cycle we're in
        if T_mult == 1:
            # All cycles same length
            cycle_idx = step_after_warmup // T_0_steps
            cycle_step = step_after_warmup % T_0_steps
            cycle_len = T_0_steps
        else:
            # Cycle lengths: T_0, T_0*T_mult, T_0*T_mult^2, ...
            # Find cycle boundary
            cumulative = 0
            cycle_idx = 0
            cycle_len = T_0_steps
            while cumulative + cycle_len <= step_after_warmup:
                cumulative += cycle_len
                cycle_idx += 1
                cycle_len = int(T_0_steps * (T_mult ** cycle_idx))
            cycle_step = step_after_warmup - cumulative

        progress = cycle_step / max(1, cycle_len)
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCell10MRebalancedModel(pl.LightningModule):
    """AIDO.Cell-10M + QKV + FFN (last 2 layers) + Muon + SGDR(T_0=15) + Rebalanced Regularization.

    Key improvements over parent node3-2-3 (head_wd=5e-2, T_0=18, dropout=0.40):
    1. T_0=15 (vs 18): Earlier SGDR C1 restart → faster cycle completion
    2. head_weight_decay=3.5e-2 (vs 5e-2): Middle ground to prevent memorization
       while still allowing the head to learn during C1's high-LR phase
    3. head_dropout=0.38 (vs 0.40): Slightly reduced to allow more gradient signal
       through the head during C1, complementing the lower wd
    4. StochasticDepth (DropPath) at drop_prob=0.10 on backbone layers:
       Provides structural regularization complementary to head-level regularization
    5. ES patience=15 (vs 12): More room for C1 exploration after the restart
    """

    def __init__(
        self,
        fusion_layers: int           = 4,
        head_hidden: int             = 512,
        head_dropout: float          = 0.38,     # slightly reduced from 0.40 for more C1 signal
        lr_muon: float               = 0.02,     # unchanged - proven safe
        lr_adamw: float              = 2e-4,     # unchanged
        backbone_weight_decay: float = 2e-2,     # backbone stays at 2e-2
        head_weight_decay: float     = 3.5e-2,   # reduced from 5e-2 to find sweet spot
        label_smoothing: float       = 0.1,
        ffn_unfreeze_layers: int     = 2,        # 4 layers caused regression in siblings
        warmup_epochs: int           = 5,
        min_lr_ratio: float          = 0.05,
        sgdr_T0_epochs: int          = 15,       # reduced from 18 for earlier C1 restart
        sgdr_T_mult: int             = 2,
        label_noise_rate: float      = 0.03,     # 3% DEG flip - proven effective
        drop_path_rate: float        = 0.10,     # stochastic depth on backbone layers
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
            ss = layer.attention.flash_self
            mm = layer.attention.self
            ss.query.weight = mm.query.weight
            ss.key.weight   = mm.key.weight
            ss.value.weight = mm.value.weight
            ss.query.bias   = mm.query.bias
            ss.key.bias     = mm.key.bias
            ss.value.bias   = mm.value.bias

        # ---- Freeze all backbone layers ----
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
        print(f"[Node3-2-3-1] Trainable backbone params: {qkv_count:,} / {total:,}")

        # ---- StochasticDepth: wrap backbone layers with DropPath ----
        # Applied to all N_LAYERS transformer layers with linearly increasing drop rate
        # Layer 0 gets drop_prob=0, Layer N_LAYERS-1 gets drop_prob=drop_path_rate
        # This is the standard stochastic depth schedule (linear decay from input to output)
        if hp.drop_path_rate > 0.0:
            encoder_layers = self.backbone.bert.encoder.layer
            n_total = len(encoder_layers)
            for layer_idx, layer in enumerate(encoder_layers):
                # Linear schedule: earlier layers have lower drop prob
                layer_drop_prob = hp.drop_path_rate * (layer_idx + 1) / n_total
                encoder_layers[layer_idx] = StochasticDepthResidualWrapper(
                    layer, drop_prob=layer_drop_prob
                )
            print(f"[Node3-2-3-1] StochasticDepth applied: max_rate={hp.drop_path_rate}, "
                  f"linear schedule over {n_total} layers")

        # ---- Head: fixed multi-layer fusion (concat) → classification ----
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

        # These will be set in main() before fit
        self._steps_per_epoch: int = -1

    # ---- Label noise: flip a fraction of DEG labels during training ----
    def _apply_label_noise(self, labels: torch.Tensor) -> torch.Tensor:
        """Randomly flip 3% of DEG (non-neutral) labels during training.

        Only modifies class 0 (down-reg) and class 2 (up-reg) entries.
        Never flips the dominant neutral class (class 1) to avoid distorting signal.
        This provides data-level regularization complementing label smoothing.
        """
        noise_rate = self.hparams.label_noise_rate
        if noise_rate <= 0.0:
            return labels
        is_deg = (labels == 0) | (labels == 2)
        noise_mask = (torch.rand_like(labels.float()) < noise_rate) & is_deg
        random_classes = torch.randint_like(labels, 0, 3)
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
        hidden_states = out.hidden_states  # len = N_LAYERS + 1 = 9

        n = self.hparams.fusion_layers
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
        labels = batch["labels"]
        # Apply label noise during training only
        labels_noisy = self._apply_label_noise(labels)
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_positions"])
        loss   = self._loss(logits, labels_noisy)
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
        rank = self.global_rank if hasattr(self, 'global_rank') and self.global_rank >= 0 else 0
        for i, (pid, sym) in enumerate(zip(batch["pert_id"], batch["symbol"])):
            # Use global_unique_idx = rank * max_batch_size + local_idx to ensure uniqueness across GPUs
            global_idx = rank * 10000 + batch["sample_idx"][i].item()
            self._test_meta.append((pid, sym, global_idx))
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        all_preds   = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)

        local_meta = [(pid, sym, idx) for (pid, sym, idx) in self._test_meta]
        # Use torch.distributed.all_gather_object directly to gather metadata from all ranks
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        world_size = max(world_size, 1)
        all_meta_list = [local_meta]
        if world_size > 1:
            # all_gather_object concatenates results from all ranks into a list
            all_meta_list = [None] * world_size
            dist.all_gather_object(all_meta_list, local_meta)
        # Flatten: list of lists of tuples -> list of tuples
        all_meta = []
        for sublist in all_meta_list:
            if sublist:
                all_meta.extend(sublist)

        if self.trainer.is_global_zero:
            # Deduplicate by global_unique_idx (rank * 10000 + local_idx)
            seen = set()
            rows = []
            for i, meta in enumerate(all_meta):
                if i >= all_preds.shape[0]:
                    break
                pid, sym, global_idx = meta
                if global_idx in seen:
                    continue
                seen.add(global_idx)
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(all_preds[i].float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node3-2-3-1] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_meta.clear()

    # ---- checkpoint ----
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

    # ---- optimizer: Muon for QKV weights, AdamW for FFN + head with separate wd ----
    def configure_optimizers(self):
        hp = self.hparams

        # QKV weight matrices for Muon (ndim >= 2)
        # Note: StochasticDepthResidualWrapper wraps layers but doesn't change parameter names
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

        # Head parameters — use REDUCED weight decay from parent's 5e-2 to 3.5e-2
        # This is the "sweet spot" that prevents catastrophic C1 memorization (>2e-2)
        # while allowing enough head learning during C1's high-LR phase (<5e-2)
        head_params = list(self.head.parameters())

        param_groups = [
            # Muon group: QKV weight matrices
            dict(
                params       = qkv_weights,
                use_muon     = True,
                lr           = hp.lr_muon,
                weight_decay = hp.backbone_weight_decay,
                momentum     = 0.95,
            ),
            # AdamW group for backbone FFN params
            dict(
                params       = ffn_params,
                use_muon     = False,
                lr           = hp.lr_adamw,
                betas        = (0.9, 0.95),
                weight_decay = hp.backbone_weight_decay,
            ),
            # AdamW group for head — REDUCED from 5e-2 (parent) to 3.5e-2
            # The parent's 5e-2 moderated but didn't prevent C1 overfitting, and
            # simultaneously prevented beneficial C1 learning by over-constraining.
            # 3.5e-2 is the middle ground: still significantly above the 2e-2 that
            # caused catastrophic overfitting (sibling node3-2-2), but below 5e-2
            # which prevented the head from climbing during C1's high-LR phase.
            dict(
                params       = head_params,
                use_muon     = False,
                lr           = hp.lr_adamw,
                betas        = (0.9, 0.95),
                weight_decay = hp.head_weight_decay,
            ),
        ]

        use_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        opt_cls = MuonWithAuxAdam if use_distributed else SingleDeviceMuonWithAuxAdam
        optimizer = opt_cls(param_groups)

        # SGDR schedule with linear warmup
        # steps_per_epoch is set in main() before fit
        steps_per_epoch = max(1, self._steps_per_epoch)
        num_warmup_steps = hp.warmup_epochs * steps_per_epoch

        scheduler = sgdr_schedule_with_warmup(
            optimizer,
            num_warmup_steps  = num_warmup_steps,
            steps_per_epoch   = steps_per_epoch,
            T_0_epochs        = hp.sgdr_T0_epochs,
            T_mult            = hp.sgdr_T_mult,
            min_lr_ratio      = hp.min_lr_ratio,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(description="Node3-2-3-1 – AIDO.Cell-10M + SGDR(T_0=15) + Rebalanced Regularization")
    parser.add_argument("--micro_batch_size",       type=int,   default=16)
    parser.add_argument("--global_batch_size",      type=int,   default=128)
    parser.add_argument("--max_epochs",             type=int,   default=200)
    parser.add_argument("--lr_muon",                type=float, default=0.02)
    parser.add_argument("--lr_adamw",               type=float, default=2e-4)
    parser.add_argument("--backbone_weight_decay",  type=float, default=2e-2)
    parser.add_argument("--head_weight_decay",      type=float, default=3.5e-2,
                        help="Head WD reduced from parent's 5e-2 to 3.5e-2 for balanced C1 learning")
    parser.add_argument("--fusion_layers",          type=int,   default=4)
    parser.add_argument("--head_hidden",            type=int,   default=512)
    parser.add_argument("--head_dropout",           type=float, default=0.38,
                        help="Head dropout reduced from parent's 0.40 to complement lower head_wd")
    parser.add_argument("--label_smoothing",        type=float, default=0.1)
    parser.add_argument("--ffn_unfreeze_layers",    type=int,   default=2)
    parser.add_argument("--warmup_epochs",          type=int,   default=5)
    parser.add_argument("--min_lr_ratio",           type=float, default=0.05)
    parser.add_argument("--sgdr_T0_epochs",         type=int,   default=15,
                        help="SGDR T_0 in epochs (C0 duration). Reduced from 18 to 15 for earlier C1 restart.")
    parser.add_argument("--sgdr_T_mult",            type=int,   default=2,
                        help="SGDR T_mult for cycle length growth.")
    parser.add_argument("--label_noise_rate",       type=float, default=0.03,
                        help="Fraction of DEG labels to flip during training. 0 to disable.")
    parser.add_argument("--drop_path_rate",         type=float, default=0.10,
                        help="StochasticDepth (DropPath) rate for backbone layers. 0 to disable.")
    parser.add_argument("--es_patience",            type=int,   default=15,
                        help="EarlyStopping patience (increased from 12 for more C1 exploration)")
    parser.add_argument("--val_check_interval",     type=float, default=1.0)
    parser.add_argument("--num_workers",            type=int,   default=4)
    parser.add_argument("--debug_max_step",         type=int,   default=None)
    parser.add_argument("--fast_dev_run",           action="store_true")
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

    # Estimate steps per epoch for SGDR schedule
    # 1388 training samples, micro_batch_size=16, n_gpus GPUs, accumulation steps
    # steps_per_epoch = ceil(1388 / (micro_batch_size * n_gpus)) / accum
    n_train_samples = 1388
    batches_per_gpu_per_epoch = math.ceil(n_train_samples / (args.micro_batch_size * n_gpus))
    steps_per_epoch = max(1, batches_per_gpu_per_epoch // accum)

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOCell10MRebalancedModel(
        fusion_layers          = args.fusion_layers,
        head_hidden            = args.head_hidden,
        head_dropout           = args.head_dropout,
        lr_muon                = args.lr_muon,
        lr_adamw               = args.lr_adamw,
        backbone_weight_decay  = args.backbone_weight_decay,
        head_weight_decay      = args.head_weight_decay,
        label_smoothing        = args.label_smoothing,
        ffn_unfreeze_layers    = args.ffn_unfreeze_layers,
        warmup_epochs          = args.warmup_epochs,
        min_lr_ratio           = args.min_lr_ratio,
        sgdr_T0_epochs         = args.sgdr_T0_epochs,
        sgdr_T_mult            = args.sgdr_T_mult,
        label_noise_rate       = args.label_noise_rate,
        drop_path_rate         = args.drop_path_rate,
    )
    # Set steps_per_epoch before configure_optimizers is called
    model._steps_per_epoch = steps_per_epoch

    print(f"[Node3-2-3-1] Setup: n_gpus={n_gpus}, accum={accum}, steps_per_epoch={steps_per_epoch}")
    print(f"[Node3-2-3-1] SGDR: T_0={args.sgdr_T0_epochs} epochs (reduced from parent's 18), T_mult={args.sgdr_T_mult}")
    print(f"[Node3-2-3-1] Weight decay: backbone={args.backbone_weight_decay}, head={args.head_weight_decay} (reduced from parent's 5e-2)")
    print(f"[Node3-2-3-1] Head dropout={args.head_dropout} (reduced from 0.40), DropPath={args.drop_path_rate}")
    print(f"[Node3-2-3-1] Label noise: {args.label_noise_rate*100:.1f}% DEG flip rate")
    print(f"[Node3-2-3-1] ES patience={args.es_patience} (increased from parent's 12)")

    ckpt_cb = ModelCheckpoint(
        dirpath  = str(output_dir / "checkpoints"),
        filename = "best-{epoch:03d}-{val/f1:.4f}",
        monitor  = "val/f1", mode="max", save_top_k=1,
    )
    # patience=15 allows more room for SGDR C1 exploration (increased from parent's 12)
    # min_delta=0.0005 unchanged from parent's proven setting
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=args.es_patience, min_delta=0.0005)
    lr_cb = LearningRateMonitor(logging_interval="step")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

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
    print(f"[Node3-2-3-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
