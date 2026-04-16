"""Node 3-3-3 – AIDO.Cell-10M + QKV+FFN(2L) + SGDR(T_0=12) + GenePriorBias + Label Noise.

Strategy (improvements over node3-3 parent and siblings node3-3-1, node3-3-2):
- Sibling2 (node3-3-2) achieved F1=0.4496 (NEW BEST) with SGDR T_0=18 + label noise 3%.
  Key: the 36-epoch cycle 1 (from T_0=18) enabled a sustained 0.037 F1 climb.
- This node explores T_0=12 (shorter first cycle, longer subsequent cycles):
  - Three restart opportunities: epochs 17, 41, 89 (vs sibling2's two: 23, 59)
  - Cycle 0: 12 epochs, Cycle 1: 24 epochs, Cycle 2: 48 epochs
  - The 48-epoch cycle 2 gives more exploration than sibling2's 36-epoch cycle 1
- Novel 1: GenePriorBias (learnable [N_CLASSES=3, N_GENES=6640] per-gene class bias)
  - Decomposes prediction into: gene-level prior + perturbation-specific adjustment
  - Inspired by node1 STRING_GNN family where gene-level priors improved generalization
  - 19,920 additional parameters (negligible vs 13M total)
  - Separate AdamW group (lr_bias=1e-3, weight_decay=0) for fast calibration
  - NOT tried in any prior AIDO.Cell lineage node
- Novel 2: head_hidden=384 (reduced from 512)
  - Reduces head from ~10.7M to ~8.0M parameters (as recommended in sibling2 feedback)
  - Additional regularization for the 1388-sample training set
- Label noise 2% (slightly less than sibling2's 3%, finding optimal level)
- Retain all proven node3-2 settings: FFN=2, lr_muon=0.02, warmup=5, wd=2e-2
- Remove SWA (was impractical — early stopping fired before SWA phase)
- Patience=15, min_delta=0.0005 (extended from sibling2's 10/0.0005)

DO NOT use: Focal loss (catastrophic in node3-1), Feature Mixup (negative in node3-3-1),
FFN=4 layers (hurt in node3-3), lr_muon>0.02 (hurt in node3-3).
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
        tp   = (is_pred & is_true).float().sum(0)
        fp   = (is_pred & ~is_true).float().sum(0)
        fn   = (~is_pred & is_true).float().sum(0)
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(prec + rec > 0, 2*prec*rec/(prec+rec+1e-8), torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


def apply_label_noise(labels: torch.Tensor, noise_rate: float) -> torch.Tensor:
    """Flip DEG labels (class 0 or class 2) to neutral (class 1) with probability noise_rate.

    Only applied to DEG positions (where label != 1). Validation and test use clean labels.
    This provides head-level regularization without distorting backbone activations.
    At noise_rate=0.02: ~2% of DEG entries are flipped, affecting 0.15% of all labels.
    """
    if noise_rate <= 0.0:
        return labels
    noisy_labels = labels.clone()
    deg_mask  = (labels != 1)                                    # [B, N_GENES] — DEG positions
    flip_mask = (torch.rand_like(labels.float()) < noise_rate) & deg_mask
    noisy_labels[flip_mask] = 1                                  # Flip to neutral
    return noisy_labels


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
        pert_ids   = [b["pert_id"] for b in batch]
        symbols    = [b["symbol"]  for b in batch]
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
# SGDR LR schedule with linear warmup
# ---------------------------------------------------------------------------
def sgdr_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    T0_steps: int,
    T_mult: int = 2,
    min_lr_ratio: float = 0.05,
):
    """Linear warmup followed by cosine annealing with warm restarts (SGDR).

    After warmup, applies SGDR with cycle lengths:
      T_0, T_0*T_mult, T_0*T_mult^2, ...

    At each restart, LR resets to 1.0 × base_lr. Within each cycle, LR
    decays from 1.0 to min_lr_ratio via cosine annealing.

    Returns a LambdaLR scheduler (multiplier applied to each param group's base LR).
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # SGDR phase: find current cycle and position within it
        step_after_warmup = current_step - num_warmup_steps
        T_cur = T0_steps
        cumulative = 0
        while cumulative + T_cur <= step_after_warmup:
            cumulative += T_cur
            T_cur = T_cur * T_mult

        # Position within current cycle [0, T_cur)
        t = step_after_warmup - cumulative
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * t / max(1, T_cur)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCell10MGenePriorBiasModel(pl.LightningModule):
    """AIDO.Cell-10M + QKV(all 8L) + FFN(last 2L) + SGDR(T_0=12) + GenePriorBias.

    Two novel contributions vs parent (node3-3) and siblings (node3-3-1, node3-3-2):
    1. GenePriorBias: Learnable [N_CLASSES, N_GENES] gene-level class prior added to head logits.
       - Decouples "gene-level class distribution" from "perturbation-specific adjustment".
       - Initialized to zeros (starts neutral, learns gene-specific biases during training).
       - Separate AdamW group (lr_bias=1e-3, weight_decay=0) for fast prior calibration.
       - Inspired by node1 STRING_GNN lineage where gene priors improved generalization.
       - NOT previously tried in any AIDO.Cell lineage node.

    2. SGDR T_0=12 (shorter first cycle than sibling2's T_0=18):
       - Three restart opportunities at epochs ~17, ~41, ~89 (vs sibling2's ~23, ~59).
       - Cycle lengths: 12 → 24 → 48 epochs (vs sibling2's 18 → 36 epochs).
       - The 48-epoch cycle 2 provides more exploration than sibling2's 36-epoch cycle 1.
       - Multiple restarts give more escape opportunities from local optima.

    Additional change: head_hidden=384 (vs 512) for additional regularization,
    and label_noise_rate=0.02 (vs sibling2's 0.03) — slightly less aggressive.
    """

    def __init__(
        self,
        fusion_layers: int       = 4,
        head_hidden: int         = 384,    # Reduced from 512 for additional regularization
        head_dropout: float      = 0.35,
        lr_muon: float           = 0.02,   # Proven optimal (0.03 hurt in node3-3)
        lr_adamw: float          = 2e-4,   # Proven optimal
        lr_bias: float           = 1e-3,   # Higher LR for gene_class_bias fast calibration
        weight_decay: float      = 2e-2,   # Proven optimal (3e-2 hurt in node3-3)
        label_smoothing: float   = 0.1,
        label_noise_rate: float  = 0.02,   # 2% DEG flip (slightly less than sibling2's 3%)
        ffn_unfreeze_layers: int = 2,      # Proven optimal (4 hurt in node3-3)
        warmup_epochs: int       = 5,      # Proven optimal (node3-2 warmup=5)
        T0_epochs: int           = 12,     # SGDR first cycle length (shorter than sibling2's 18)
        T_mult: int              = 2,      # SGDR cycle multiplier
        min_lr_ratio: float      = 0.05,   # LR floor as fraction of base LR
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
        # Lightning's bf16-mixed autocast causes LayerNorm outputs to be float32,
        # breaking the attention's dtype check. Patching LN to output bf16.
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

        # ---- Freeze all backbone layers ----
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ---- Unfreeze QKV weights in ALL 8 attention layers ----
        qkv_suffixes = (
            "attention.self.query.weight",
            "attention.self.key.weight",
            "attention.self.value.weight",
        )
        for name, param in self.backbone.named_parameters():
            if name.endswith(qkv_suffixes):
                param.requires_grad = True

        # ---- Unfreeze FFN (SwiGLU) in the last 2 transformer layers ----
        # Node3-3 proved that FFN=4 layers hurts; FFN=2 (layers 6-7) is optimal.
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

        trainable_count = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_count     = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Node3-3-3] Trainable backbone params: {trainable_count:,} / {total_count:,}")

        # ---- Classification head (reduced to head_hidden=384) ----
        # Sibling2's feedback recommends reducing head_hidden for additional regularization.
        # Linear(1024→384) + LN(384) + GELU + Dropout(0.35) + Linear(384→3×6640)
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

        # ---- GenePriorBias: learnable per-gene per-class bias ----
        # Initialized to zeros: starts as a "no prior" state, then learns gene-specific
        # class distributions during training. This decouples:
        #   logits = head(fused) + gene_class_bias
        #           = perturbation_response + gene_prior
        # The bias captures which genes are consistently up/down/neutral across perturbations,
        # while the head captures perturbation-specific adjustments.
        # 19,920 additional parameters (3 × 6640) — negligible vs 13M total.
        self.gene_class_bias = nn.Parameter(torch.zeros(N_CLASSES, N_GENES))

        self.register_buffer("class_weights", get_class_weights())

        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_meta:  List[Tuple]        = []

        # Injected from main() before fit
        self._T0_steps:        int = -1
        self._num_warmup_steps: int = -1

    # ---- forward ----
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        gene_positions: torch.Tensor,
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Extract perturbed gene embedding from the last `fusion_layers` transformer layers
        hidden_states = out.hidden_states  # len = N_LAYERS + 1 = 9
        n = self.hparams.fusion_layers
        layer_embs = []
        for i in range(n):
            hs = hidden_states[-(i + 1)]             # [B, AIDO_GENES+2, 256]
            ge = hs[torch.arange(B, device=hs.device), gene_positions, :]  # [B, 256]
            layer_embs.append(ge.float())
        fused = torch.cat(layer_embs, dim=-1)        # [B, 1024]

        # Standard head output + GenePriorBias
        # gene_class_bias: [N_CLASSES, N_GENES] → broadcast to [B, N_CLASSES, N_GENES]
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        logits = logits + self.gene_class_bias.unsqueeze(0)
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
        # Apply label noise: flip 2% of DEG labels to neutral (training only)
        if self.hparams.label_noise_rate > 0:
            labels = apply_label_noise(labels, self.hparams.label_noise_rate)
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_positions"])
        loss   = self._loss(logits, labels)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_positions"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])  # Clean labels for validation
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

        local_pids = [m[0] for m in self._test_meta]
        local_syms = [m[1] for m in self._test_meta]
        local_idxs = torch.tensor([m[2] for m in self._test_meta], device=self.device, dtype=torch.long)

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

            combined = list(zip(all_idxs, all_pids_flat, all_syms_flat,
                               [all_preds[i] for i in range(all_preds.shape[0])]))
            combined.sort(key=lambda x: x[0])
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
            print(f"[Node3-3-3] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_meta.clear()

    # ---- checkpoint: save only trainable params + buffers ----
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        _params_list  = list(self.named_parameters())
        total         = sum(p.numel() for _, p in _params_list)
        trained       = sum(p.numel() for _, p in _params_list if p.requires_grad)
        _buffers_list = list(self.named_buffers())
        total_bufs    = sum(b.numel() for _, b in _buffers_list)

        pct     = f"{100*trained/total:.1f}%" if total > 0 else "N/A"
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

    # ---- optimizer: Muon for QKV, AdamW for FFN + head, AdamW for gene bias ----
    def configure_optimizers(self):
        hp = self.hparams

        # Muon group: QKV weight matrices (ndim >=2, attention QKV weights only)
        qkv_weights = [
            p for name, p in self.backbone.named_parameters()
            if p.requires_grad and p.ndim >= 2
            and any(s in name for s in ["query.weight", "key.weight", "value.weight"])
        ]

        # AdamW group 1: FFN params + head params (combined with AdamW + weight decay)
        ffn_params  = [
            p for name, p in self.backbone.named_parameters()
            if p.requires_grad
            and not any(s in name for s in ["query.weight", "key.weight", "value.weight"])
        ]
        head_params = list(self.head.parameters())

        # AdamW group 2: GenePriorBias (separate group — high LR, no weight decay)
        # Higher LR (1e-3 vs 2e-4) lets the bias calibrate quickly to gene-level priors.
        # No weight decay: the bias should freely represent gene distributions without
        # being pulled toward zero.
        bias_params = [self.gene_class_bias]

        param_groups = [
            # Muon: QKV weight matrices (2D+, hidden weights only)
            dict(
                params       = qkv_weights,
                use_muon     = True,
                lr           = hp.lr_muon,
                weight_decay = hp.weight_decay,
                momentum     = 0.95,
            ),
            # AdamW: FFN layers + classification head
            dict(
                params       = head_params + ffn_params,
                use_muon     = False,
                lr           = hp.lr_adamw,
                betas        = (0.9, 0.95),
                weight_decay = hp.weight_decay,
            ),
            # AdamW: GenePriorBias (fast calibration, no regularization)
            dict(
                params       = bias_params,
                use_muon     = False,
                lr           = hp.lr_bias,
                betas        = (0.9, 0.95),
                weight_decay = 0.0,
            ),
        ]

        use_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        opt_cls   = MuonWithAuxAdam if use_distributed else SingleDeviceMuonWithAuxAdam
        optimizer = opt_cls(param_groups)

        # SGDR schedule: warmup → cosine annealing with warm restarts
        # T0_steps: steps per SGDR cycle 0 (T0_epochs * steps_per_epoch)
        # Each subsequent cycle is T_mult times longer.
        if self._T0_steps > 0:
            T0_steps    = self._T0_steps
            warmup_steps = self._num_warmup_steps
        else:
            # Fallback: ~12 epochs * 11 steps/epoch = 132; warmup = 5 * 11 = 55
            T0_steps     = 132
            warmup_steps = 55

        scheduler = sgdr_schedule_with_warmup(
            optimizer,
            num_warmup_steps = warmup_steps,
            T0_steps         = T0_steps,
            T_mult           = hp.T_mult,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node3-3-3 – AIDO.Cell-10M + QKV+FFN(2L) + SGDR(T_0=12) + GenePriorBias"
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=16,   dest="micro_batch_size")
    parser.add_argument("--global-batch-size",   type=int,   default=128,  dest="global_batch_size")
    parser.add_argument("--max-epochs",          type=int,   default=150,  dest="max_epochs",
                        help="Extended budget to cover 3 SGDR cycles (restarts at ~17, ~41, ~89)")
    parser.add_argument("--lr-muon",             type=float, default=0.02, dest="lr_muon",
                        help="Proven optimal Muon LR (0.03 hurt in node3-3)")
    parser.add_argument("--lr-adamw",            type=float, default=2e-4, dest="lr_adamw")
    parser.add_argument("--lr-bias",             type=float, default=1e-3, dest="lr_bias",
                        help="LR for GenePriorBias (higher for fast calibration)")
    parser.add_argument("--weight-decay",        type=float, default=2e-2, dest="weight_decay",
                        help="Proven optimal (3e-2 hurt in node3-3)")
    parser.add_argument("--fusion-layers",       type=int,   default=4,    dest="fusion_layers")
    parser.add_argument("--head-hidden",         type=int,   default=384,  dest="head_hidden",
                        help="Reduced from 512 for additional regularization (sibling2 feedback)")
    parser.add_argument("--head-dropout",        type=float, default=0.35, dest="head_dropout")
    parser.add_argument("--label-smoothing",     type=float, default=0.1,  dest="label_smoothing")
    parser.add_argument("--label-noise-rate",    type=float, default=0.02, dest="label_noise_rate",
                        help="2% DEG flip noise (slightly less than sibling2's 3%)")
    parser.add_argument("--ffn-unfreeze-layers", type=int,   default=2,    dest="ffn_unfreeze_layers",
                        help="FFN=2 layers proven optimal (FFN=4 hurt in node3-3)")
    parser.add_argument("--warmup-epochs",       type=int,   default=5,    dest="warmup_epochs",
                        help="node3-2's proven warmup=5 (warmup=4 in node3-3-1 shifted peak earlier)")
    parser.add_argument("--T0-epochs",           type=int,   default=12,   dest="T0_epochs",
                        help="SGDR T_0 cycle length (shorter than sibling2's 18 → 3 restarts)")
    parser.add_argument("--T-mult",              type=int,   default=2,    dest="T_mult",
                        help="SGDR cycle multiplier (each cycle is T_mult times longer)")
    parser.add_argument("--min-lr-ratio",        type=float, default=0.05, dest="min_lr_ratio")
    parser.add_argument("--es-patience",         type=int,   default=15,   dest="es_patience",
                        help="Extended patience to allow cycle 2 (48 epochs) to play out")
    parser.add_argument("--es-min-delta",        type=float, default=5e-4, dest="es_min_delta")
    parser.add_argument("--val-check-interval",  type=float, default=1.0,  dest="val_check_interval")
    parser.add_argument("--num-workers",         type=int,   default=4,    dest="num_workers")
    parser.add_argument("--debug-max-step",      type=int,   default=None, dest="debug_max_step")
    parser.add_argument("--fast-dev-run",        action="store_true",      dest="fast_dev_run")
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

    # Estimate training steps per epoch (used for SGDR schedule alignment)
    # With n_gpus=2, micro_batch=16, global_batch=128:
    #   steps/epoch = ceil(1388 / (16*2)) // 4 = 44 // 4 = 11
    n_train_samples  = 1388
    steps_per_epoch  = max(1, math.ceil(n_train_samples / (args.micro_batch_size * n_gpus)) // accum)
    T0_steps         = steps_per_epoch * args.T0_epochs
    num_warmup_steps = steps_per_epoch * args.warmup_epochs

    print(f"[Node3-3-3] steps_per_epoch={steps_per_epoch}, T0_steps={T0_steps} "
          f"({args.T0_epochs} epochs), warmup={num_warmup_steps} ({args.warmup_epochs} epochs)")
    print(f"[Node3-3-3] SGDR restarts at approximately epochs: "
          f"{args.warmup_epochs + args.T0_epochs}, "
          f"{args.warmup_epochs + args.T0_epochs + args.T0_epochs * args.T_mult}, "
          f"{args.warmup_epochs + args.T0_epochs + args.T0_epochs * args.T_mult + args.T0_epochs * args.T_mult**2}")

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOCell10MGenePriorBiasModel(
        fusion_layers       = args.fusion_layers,
        head_hidden         = args.head_hidden,
        head_dropout        = args.head_dropout,
        lr_muon             = args.lr_muon,
        lr_adamw            = args.lr_adamw,
        lr_bias             = args.lr_bias,
        weight_decay        = args.weight_decay,
        label_smoothing     = args.label_smoothing,
        label_noise_rate    = args.label_noise_rate,
        ffn_unfreeze_layers = args.ffn_unfreeze_layers,
        warmup_epochs       = args.warmup_epochs,
        T0_epochs           = args.T0_epochs,
        T_mult              = args.T_mult,
        min_lr_ratio        = args.min_lr_ratio,
    )
    # Inject schedule info (needed by configure_optimizers before fit)
    model._T0_steps         = T0_steps
    model._num_warmup_steps = num_warmup_steps

    ckpt_cb = ModelCheckpoint(
        dirpath  = str(output_dir / "checkpoints"),
        filename = "best-{epoch:03d}-{val/f1:.4f}",
        monitor  = "val/f1", mode="max", save_top_k=1,
    )
    # Extended patience=15: allows model to survive SGDR oscillations and fully exploit
    # the 48-epoch cycle 2. Sibling2's patience=10 was sufficient for 36-epoch cycle 1;
    # patience=15 provides equivalent coverage for the longer 48-epoch cycle 2.
    es_cb = EarlyStopping(monitor="val/f1", mode="max",
                          patience=args.es_patience, min_delta=args.es_min_delta)

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
    print(f"[Node3-3-3] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
