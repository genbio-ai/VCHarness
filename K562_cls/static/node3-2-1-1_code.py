"""Node 3-2-1-1 – AIDO.Cell-10M: Reverted to Proven Config + ReduceLROnPlateau + Mixup + Higher Dropout.

Strategy (improvements over node3-2-1 parent, informed by node3-2 proven baseline):
- REVERT backbone capacity to node3-2's proven config: QKV (all 8 layers) + FFN (last 2 layers only)
- REVERT Muon LR from 0.03 to 0.02 (node3-2's proven value that achieved best val F1=0.4295)
- REVERT weight_decay from 3e-2 to 2e-2 (node3-2's proven value)
- REPLACE cosine schedule with ReduceLROnPlateau (schedule fundamentally misaligned with ~30-epoch training)
- INCREASE head dropout from 0.35 to 0.45 for stronger regularization (feedback recommendation)
- ADD Mixup augmentation (alpha=0.2, proven effective in node4-2 to reduce train-val gap)
- RETAIN FlashAttention + LayerNorm patching, QKV weight sharing, gradient checkpointing
- RETAIN label-smoothed CE (epsilon=0.1) + sqrt-inverse-frequency class weights (confirmed effective)
- RETAIN fixed 4-layer concatenation fusion (1024-dim), head_hidden=512
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
    """Compute mean per-gene macro F1 consistent with calc_metric.py."""
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

    def _loader(self, ds, shuffle, sampler=None):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle if sampler is None else False,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=make_collate(self.tokenizer),
                          sampler=sampler)

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self):
        # DDP val: each rank processes all samples; deduplication handles duplicates in on_validation_epoch_end.
        return self._loader(self.val_ds, False)

    def test_dataloader(self):
        # DDP test: each rank processes all samples; deduplication handles duplicates in on_test_epoch_end.
        return self._loader(self.test_ds, False)


# ---------------------------------------------------------------------------
# Mixup Augmentation Helpers
# ---------------------------------------------------------------------------
def mixup_data(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup to AIDO.Cell continuous expression inputs.

    Since input_ids are float32 continuous expression values (not discrete tokens),
    linear interpolation is semantically valid: it corresponds to mixing two gene
    expression profiles, which is biologically plausible.

    Returns:
        mixed_inputs: linearly interpolated inputs
        targets_a: original targets
        targets_b: shuffled targets
        lam: mixing coefficient
    """
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = input_ids.size(0)
    index = torch.randperm(batch_size, device=input_ids.device)

    mixed_inputs = lam * input_ids + (1 - lam) * input_ids[index]
    targets_a = labels
    targets_b = labels[index]
    return mixed_inputs, targets_a, targets_b, lam


def mixup_criterion(
    criterion_fn,
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute Mixup loss as convex combination of two CE losses."""
    return lam * criterion_fn(logits, targets_a) + (1 - lam) * criterion_fn(logits, targets_b)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCell10MRevertedModel(pl.LightningModule):
    """AIDO.Cell-10M with reverted proven config + ReduceLROnPlateau + Mixup + Higher Dropout.

    Key changes from node3-2-1 (parent, F1=0.4142) to recover and exceed node3-2 (F1=0.4295):

    1. REVERT FFN unfreezing from 4 to 2 layers (node3-2's proven value).
       Parent's 4-layer expansion created excess capacity that overfits on 1,388 samples.

    2. REVERT Muon LR from 0.03 to 0.02 (node3-2's proven value).
       The 50% LR increase in the parent destabilized optimization; the parent only needed
       to reach 73% of its intended LR peak for good convergence — this was actually beneficial
       as implicit regularization, not a bug.

    3. REVERT weight_decay from 3e-2 to 2e-2 (node3-2's proven value).
       The weight_decay increase failed to compensate for excess capacity in the parent.

    4. REPLACE cosine schedule with ReduceLROnPlateau.
       Cosine schedule is fundamentally mismatched to the ~30-epoch training budget:
       early stopping consistently fires before meaningful cosine decay occurs, making
       the schedule a no-op. ReduceLROnPlateau directly ties LR reduction to observed
       val performance, which is more appropriate.

    5. INCREASE head dropout from 0.35 to 0.45.
       Feedback explicitly recommended increasing dropout to 0.4-0.5 for better regularization.

    6. ADD Mixup augmentation (alpha=0.2).
       Mixup was proven effective in node4-2 (F1=0.4801), dramatically reducing the
       train-val loss gap from 12x to 1.25x. Since AIDO.Cell's input_ids are continuous
       float32 expression values (not discrete token indices), linear interpolation is
       semantically valid and corresponds to mixing two gene expression profiles.
    """

    def __init__(
        self,
        fusion_layers: int       = 4,      # last N transformer layers to fuse
        head_hidden: int         = 512,
        head_dropout: float      = 0.45,   # increased from 0.35 for stronger regularization
        lr_muon: float           = 0.02,   # reverted to node3-2 proven value
        lr_adamw: float          = 2e-4,   # AdamW lr for head + FFN + non-QKV params
        weight_decay: float      = 2e-2,   # reverted to node3-2 proven value
        label_smoothing: float   = 0.1,
        ffn_unfreeze_layers: int = 2,      # reverted to node3-2 proven value (2 layers)
        mixup_alpha: float       = 0.2,    # Mixup augmentation coefficient
        rlrop_patience: int      = 4,      # ReduceLROnPlateau patience
        rlrop_factor: float      = 0.5,    # ReduceLROnPlateau factor
        rlrop_min_lr: float      = 1e-5,   # minimum LR floor
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

        # ---- Unfreeze FFN (SwiGLU) in last `ffn_unfreeze_layers` transformer layers ----
        # REVERTED to 2 layers (matching node3-2's proven config F1=0.4295)
        # node3-2-1's expansion to 4 layers caused overfitting with only 1,388 training samples
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
        print(f"[Node3-2-1-1] Trainable backbone params: {qkv_count:,} / {total:,}")

        # ---- Head: fixed multi-layer fusion (concat) → classification ----
        # Keeps proven 1024-dim input (4 layers × 256) as fixed concatenation
        in_dim = hp.fusion_layers * HIDDEN_DIM  # 4 * 256 = 1024
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),  # INCREASED from 0.35 to 0.45
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
        self._test_idx:   List[torch.Tensor]  = []

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
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        gene_positions = batch["gene_positions"]
        labels         = batch["labels"]

        alpha = self.hparams.mixup_alpha
        if alpha > 0 and self.training:
            # Apply Mixup: linearly interpolate expression profiles
            # input_ids are float32 continuous values, so interpolation is valid
            mixed_inputs, targets_a, targets_b, lam = mixup_data(input_ids, labels, alpha=alpha)
            logits = self(mixed_inputs, attention_mask, gene_positions)
            loss = mixup_criterion(self._loss, logits, targets_a, targets_b, lam)
        else:
            logits = self(input_ids, attention_mask, gene_positions)
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
        # Robust deduplication: for sorted indices, keep True where the next index differs.
        # This marks the first occurrence of each unique sample.
        keep_mask = torch.ones(s_idx.numel(), dtype=torch.bool, device=s_idx.device)
        keep_mask[1:] = s_idx[:-1] != s_idx[1:]
        f1 = compute_per_gene_f1(s_pred[keep_mask], s_tgt[keep_mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_positions"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].clone())
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = torch.cat(self._test_preds, 0)
        local_idx   = torch.cat(self._test_idx,   0)

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        # Sort by sample index and deduplicate: for sorted indices, keep where next differs.
        order = torch.argsort(all_idx)
        s_idx  = all_idx[order]
        s_pred = all_preds[order]

        unique_mask = torch.ones(s_idx.numel(), dtype=torch.bool, device=s_idx.device)
        unique_mask[1:] = s_idx[:-1] != s_idx[1:]
        unique_preds = s_pred[unique_mask]
        unique_idx   = s_idx[unique_mask]

        if self.trainer.is_global_zero:
            # Build metadata on rank 0 from the test dataloader (all 154 samples in order).
            test_df = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_row = {str(r["pert_id"]): r for _, r in test_df.iterrows()}

            rows = []
            for rank0_idx in range(unique_preds.shape[0]):
                row_idx_in_test = unique_idx[rank0_idx].item()
                pert_id = test_df.iloc[row_idx_in_test]["pert_id"]
                symbol  = test_df.iloc[row_idx_in_test]["symbol"]
                rows.append({
                    "idx":        pert_id,
                    "input":      symbol,
                    "prediction": json.dumps(unique_preds[rank0_idx].float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node3-2-1-1] Saved {len(rows)} test predictions (deduplicated from {all_preds.shape[0]} gathered).")
        self._test_preds.clear()
        self._test_idx.clear()

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

    # ---- optimizer: Muon for QKV weights, AdamW for FFN + head ----
    # REPLACED cosine schedule with ReduceLROnPlateau
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
                lr           = hp.lr_muon,      # REVERTED to 0.02
                weight_decay = hp.weight_decay,  # REVERTED to 2e-2
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

        # REPLACED cosine schedule with ReduceLROnPlateau.
        # Rationale: cosine schedule is misaligned with ~30-epoch training: early stopping
        # fires before meaningful decay occurs. ReduceLROnPlateau directly ties LR reduction
        # to observed val performance, which is more appropriate for this regime.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode     = "max",
            factor   = hp.rlrop_factor,
            patience = hp.rlrop_patience,
            min_lr   = hp.rlrop_min_lr,
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor":   "val/f1",
                "interval":  "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node3-2-1-1 – AIDO.Cell-10M reverted to proven config + ReduceLROnPlateau + Mixup + Higher Dropout"
    )
    parser.add_argument("--micro_batch_size",    type=int,   default=16)
    parser.add_argument("--global_batch_size",   type=int,   default=128)
    parser.add_argument("--max_epochs",          type=int,   default=150,
                        help="Max epochs; ReduceLROnPlateau + early stopping will terminate appropriately")
    parser.add_argument("--lr_muon",             type=float, default=0.02,
                        help="Reverted to node3-2's proven Muon LR=0.02")
    parser.add_argument("--lr_adamw",            type=float, default=2e-4)
    parser.add_argument("--weight_decay",        type=float, default=2e-2,
                        help="Reverted to node3-2's proven wd=0.02")
    parser.add_argument("--fusion_layers",       type=int,   default=4)
    parser.add_argument("--head_hidden",         type=int,   default=512)
    parser.add_argument("--head_dropout",        type=float, default=0.45,
                        help="Increased from 0.35 to 0.45 for stronger regularization")
    parser.add_argument("--label_smoothing",     type=float, default=0.1)
    parser.add_argument("--ffn_unfreeze_layers", type=int,   default=2,
                        help="Reverted to node3-2's proven 2 FFN layers (from 4)")
    parser.add_argument("--mixup_alpha",         type=float, default=0.2,
                        help="Mixup augmentation alpha; 0 disables Mixup")
    parser.add_argument("--rlrop_patience",      type=int,   default=4,
                        help="ReduceLROnPlateau patience in epochs")
    parser.add_argument("--rlrop_factor",        type=float, default=0.5,
                        help="ReduceLROnPlateau LR reduction factor")
    parser.add_argument("--rlrop_min_lr",        type=float, default=1e-5,
                        help="ReduceLROnPlateau minimum LR")
    parser.add_argument("--val_check_interval",  type=float, default=1.0)
    parser.add_argument("--num_workers",         type=int,   default=4)
    parser.add_argument("--es_patience",         type=int,   default=12,
                        help="Early stopping patience in epochs")
    parser.add_argument("--es_min_delta",        type=float, default=0.001,
                        help="Early stopping minimum delta")
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

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOCell10MRevertedModel(
        fusion_layers       = args.fusion_layers,
        head_hidden         = args.head_hidden,
        head_dropout        = args.head_dropout,
        lr_muon             = args.lr_muon,
        lr_adamw            = args.lr_adamw,
        weight_decay        = args.weight_decay,
        label_smoothing     = args.label_smoothing,
        ffn_unfreeze_layers = args.ffn_unfreeze_layers,
        mixup_alpha         = args.mixup_alpha,
        rlrop_patience      = args.rlrop_patience,
        rlrop_factor        = args.rlrop_factor,
        rlrop_min_lr        = args.rlrop_min_lr,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath  = str(output_dir / "checkpoints"),
        filename = "best-{epoch:03d}-{val/f1:.4f}",
        monitor  = "val/f1", mode="max", save_top_k=1,
    )
    # Increased patience slightly to allow ReduceLROnPlateau to fire and help convergence
    es_cb = EarlyStopping(monitor="val/f1", mode="max",
                          patience=args.es_patience, min_delta=args.es_min_delta)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
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
    print(f"[Node3-2-1-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
