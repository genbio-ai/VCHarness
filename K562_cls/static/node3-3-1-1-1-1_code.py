"""Node 3-3-1-1-1-1 – AIDO.Cell-10M + QKV+FFN(last 2 layers) + SGDR(T_0=15) + Checkpoint Ensemble.

Strategy: Revert GenePriorBias+SWA regression and add checkpoint ensemble on top of node3-3-1-1.

Parent node3-3-1-1-1 (F1=0.4098) suffered a -0.027 regression from grandparent node3-3-1-1
(F1=0.4368) due to GenePriorBias module initialization distorting the early loss landscape.
The 20-epoch bias warmup was catastrophically misaligned (bias activated when LR had decayed to
η_min=0.001), and SWA never fired because ES stopped training at epoch 43 (vs planned epoch 95).

This node (node3-3-1-1-1-1) makes three targeted changes:

1. REMOVE GenePriorBias ENTIRELY (primary fix):
   - The per-gene log-prior bias initialization (log P(c|g) dominated by neutral class ~92.5%)
     shifted the initial logit baseline, forcing the model to fight a fixed initialization signal
     during the critical first 10 epochs where val F1 was stuck at 0.34–0.35.
   - The bias warmup_epochs=20 meant bias became learnable at epoch 20 when LR had decayed to
     η_min=0.001, leaving only ~9 useful epochs of bias learning vs. 40 for the parent's head.
   - GenePriorBias works in the scFoundation lineage (node4-2-1-1: +0.003) because scFoundation
     has perturbation-aware pretraining. In AIDO.Cell, the steady-state pretraining lacks this
     signal, so bias competes with rather than complements the learned representation.
   - Expected recovery: +0.027 F1 (back to grandparent node3-3-1-1 level of 0.4368)

2. REMOVE SWA (never fires in the AIDO.Cell training window):
   - With SGDR T_0=15 and the model peaking at epoch ~40, SWA at epoch 95 is unreachable.
   - SWA requires training through epoch 95+, but ES fires at epoch ~48 (peak epoch 40 + patience 8).
   - No benefit achievable; only adds code complexity.

3. REVERT hyperparameters to grandparent node3-3-1-1 proven config:
   - max_epochs: 130 → 120 (sufficient for SGDR through cycle 3)
   - ES patience: 15 → 8 (grandparent's patience correctly captured peak without overfitting)
   - min_delta: 0.002 → 0.001 (grandparent's threshold worked correctly)

4. ADD CHECKPOINT ENSEMBLE (top-3 by val F1):
   - Save top-3 checkpoints during training (save_top_k=3)
   - After training, run inference with each of the top-3 checkpoints
   - Average softmax predictions across 3 checkpoint runs
   - Write ensemble-averaged predictions to test_predictions.tsv
   - This is a safe, low-risk improvement validated across ML literature
   - Expected gain: +0.002-0.003 F1

RETAINED FROM NODE3-3-1-1 (all proven effective):
- FFN unfreeze_layers=2 (last 2 layers, layers 6-7)
- lr_muon=0.02, lr_adamw=2e-4, weight_decay=2e-2
- Label-smoothed CE (ε=0.1) + class frequency weights
- Fixed 4-layer concatenation fusion (1024-dim)
- head_hidden=512, head_dropout=0.35
- SGDR T_0=15 epochs, T_mult=2, warmup=5 epochs
- gradient_clip_val=1.0
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

# Class frequencies: down(-1→0), neutral(0→1), up(+1→2)
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Inverse-sqrt frequency class weights for weighted CE loss."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py definition.

    preds:   [N, 3, G] softmax probabilities
    targets: [N, G]    integer class labels in {0,1,2}
    Returns: mean per-gene macro-F1 (float)
    """
    y_hat       = preds.argmax(dim=1)   # [N, G]
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
        pert_ids   = [b["pert_id"] for b in batch]
        symbols    = [b["symbol"]  for b in batch]
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        input_ids  = tokenized["input_ids"]   # [B, 19264] float32

        # Locate the position of the perturbed gene in the full-genome vocabulary
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
        # Attach global_rank so validation/test dedup uses globally unique sample identifiers.
        # NOTE: sample_idx from dataset is the global dataset index (0..N-1). Since
        # DistributedSampler partitions the dataset (each rank gets different samples),
        # sample_idx values are already globally unique — no need for (rank, local_idx).
        # We keep pert_id-based dedup in on_test_epoch_end for extra safety.
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
        # Rank-0 downloads first, then all ranks load (avoids race condition)
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

    Retained from node3-3-1-1 (T_0=15 confirmed optimal):
    - Warmup: epochs 0-5
    - Cycle 1: epochs 5-20  (15 epochs, restart at epoch 20)
    - Cycle 2: epochs 20-50 (30 epochs, produced second-descent +0.009 F1 in node3-3-1-1)
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        step = current_step - num_warmup_steps
        current_T = T_0_steps
        while step >= current_T:
            step -= current_T
            current_T = int(current_T * T_mult)

        progress = float(step) / float(max(1, current_T))
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCell10MQKVFFNSGDREnsembleModel(pl.LightningModule):
    """AIDO.Cell-10M + QKV (all 8 layers) + FFN (last 2 layers) + SGDR(T_0=15) + Checkpoint Ensemble.

    Changes from parent node3-3-1-1-1 (F1=0.4098):
    1. REMOVED GenePriorBias — initialization distorted early loss landscape, -0.027 regression
    2. REMOVED SWA — never fires before ES at epoch ~48; SWA at epoch 95 is unreachable
    3. REVERTED hyperparameters to grandparent node3-3-1-1 (F1=0.4368):
       max_epochs 130→120, ES patience 15→8, min_delta 0.002→0.001
    4. ADDED checkpoint ensemble: top-3 checkpoints by val F1 are averaged during test
    5. find_unused_parameters=False (no more frozen-grad DDP issue without bias warmup)
    """

    def __init__(
        self,
        fusion_layers: int       = 4,
        head_hidden: int         = 512,
        head_dropout: float      = 0.35,
        lr_muon: float           = 0.02,
        lr_adamw: float          = 2e-4,
        weight_decay: float      = 2e-2,
        label_smoothing: float   = 0.1,
        ffn_unfreeze_layers: int = 2,
        warmup_epochs: int       = 5,
        min_lr_ratio: float      = 0.05,
        sgdr_T0_epochs: int      = 15,
        sgdr_T_mult: int         = 2,
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

        # Patch LayerNorm to preserve dtype in bfloat16 forward pass
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
        # This ensures Muon updates flash_self QKV when self.self QKV is the parameter
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
        print(f"[Node3-3-1-1-1-1] Trainable backbone params: {qkv_count:,} / {total:,}")

        # ---- Head: 4-layer concat fusion (1024-dim) → 2-layer MLP → [3, 6640] ----
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

        self.register_buffer("class_weights", get_class_weights())

        # Accumulation buffers for validation
        self._val_preds: List[torch.Tensor] = []
        self._val_tgts:  List[torch.Tensor] = []
        self._val_pids:   List[List[str]]    = []  # pert_ids per batch (strings)

        # Accumulation buffers for test (cleared at end of each test run)
        self._test_preds: List[torch.Tensor] = []
        self._test_meta:  List[Tuple]        = []

        # After each test epoch end, stores {sample_idx: (pert_id, symbol, pred_tensor)}
        # Used by ensemble logic in main() to collect predictions across multiple ckpt runs
        self._current_test_results: Optional[Dict[int, Tuple[str, str, torch.Tensor]]] = None

        # LR schedule info (injected before configure_optimizers is called)
        self._num_training_steps: int = -1
        self._num_warmup_steps: int   = -1
        self._T0_steps: int           = -1
        self._steps_per_epoch: int    = -1

    # ---- encode: extract fused multi-layer feature vector ----
    def _encode(self, batch: Dict) -> torch.Tensor:
        """Run backbone and return fused feature vector [B, fusion_layers*256]."""
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

        return torch.cat(layer_embs, dim=-1)  # [B, 4*256=1024]

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
        fused  = self._encode(batch)
        logits = self.head(fused).view(batch["input_ids"].shape[0], N_CLASSES, N_GENES)
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        B = batch["input_ids"].shape[0]
        fused  = self._encode(batch)
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits.float(), dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_pids.append(batch["pert_id"])

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank       = int(os.environ.get("LOCAL_RANK", 0))
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        self._val_preds.clear()
        self._val_tgts.clear()
        self._val_pids.clear()

        if local_preds.shape[0] == 0:
            return

        # Gather predictions and targets from all ranks
        all_preds  = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts   = self.all_gather(local_tgts).view(-1, N_GENES)
        N_local    = local_preds.shape[0]  # samples per rank
        N_total    = all_preds.shape[0]   # world_size * N_local

        # Use torch.cat mask trick: DistributedSampler assigns different samples to each
        # rank with no overlap. After all_gather, each rank has identical (all_preds, all_tgts).
        # We keep only the positions belonging to our rank: [rank*N_local, (rank+1)*N_local).
        # This gives us all unique samples across all ranks.
        keep_mask = torch.zeros(N_total, dtype=torch.bool, device=all_preds.device)
        keep_mask[rank * N_local:(rank + 1) * N_local] = True
        uniq_preds_t = all_preds[keep_mask]
        uniq_tgts_t  = all_tgts[keep_mask]
        f1 = compute_per_gene_f1(uniq_preds_t, uniq_tgts_t)
        # Log on ALL ranks (required for EarlyStopping to work on non-zero ranks)
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        B = batch["input_ids"].shape[0]
        fused  = self._encode(batch)
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        for i, (pid, sym) in enumerate(zip(batch["pert_id"], batch["symbol"])):
            self._test_meta.append((pid, sym))
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank       = int(os.environ.get("LOCAL_RANK", 0))
        local_preds = torch.cat(self._test_preds, 0)
        local_pids  = [m[0] for m in self._test_meta]
        local_syms  = [m[1] for m in self._test_meta]
        N_local     = local_preds.shape[0]

        if len(local_pids) == 0:
            self._test_preds.clear()
            self._test_meta.clear()
            return

        # Gather predictions from all ranks (tensors via Lightning's all_gather)
        all_preds_list = self.all_gather(local_preds)    # [world_size, N_local, 3, 6640]

        # Only gather string metadata when in true multi-GPU (world_size > 1).
        # In single-GPU mode, dist.group.WORLD is not initialized.
        if world_size > 1:
            import torch.distributed as dist
            gathered_pids: List[List[str]] = [[] for _ in range(world_size)]
            gathered_syms: List[List[str]] = [[] for _ in range(world_size)]
            dist.all_gather_object(gathered_pids, local_pids, group=dist.group.WORLD)
            dist.all_gather_object(gathered_syms, local_syms, group=dist.group.WORLD)
            flat_pids = [pid for sublist in gathered_pids for pid in sublist]
            flat_syms = [sym for sublist in gathered_syms for sym in sublist]
        else:
            # Single-GPU: no gathering needed, use local data directly
            flat_pids = local_pids
            flat_syms = local_syms

        if self.trainer.is_global_zero:
            # Reshape predictions: [world_size * N_local, 3, 6640]
            N_total = all_preds_list.shape[0] * all_preds_list.shape[1]
            all_preds = all_preds_list.reshape(-1, N_CLASSES, N_GENES)

            # Build pred_map: dedup by pert_id (globally unique) to get all test samples.
            pred_map: Dict[str, Tuple[str, str, torch.Tensor]] = {}
            for pos in range(len(flat_pids)):
                pid = flat_pids[pos]
                sym = flat_syms[pos]
                if pid not in pred_map:
                    pred_map[pid] = (pid, sym, all_preds[pos])

            # Store for ensemble aggregation in main()
            self._current_test_results = pred_map
            print(f"[Node3-3-1-1-1-1] Test run collected {len(pred_map)} unique predictions.")

        self._test_preds.clear()
        self._test_meta.clear()

    # ---- checkpoint: save only trainable params + buffers ----
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        _params_list  = list(self.named_parameters())
        _buffers_list = list(self.named_buffers())
        total        = sum(p.numel() for _, p in _params_list)
        trained      = sum(p.numel() for _, p in _params_list if p.requires_grad)
        total_bufs   = sum(b.numel() for _, b in _buffers_list)

        pct     = f"{100*trained/total:.1f}%" if total > 0 else "N/A"
        buf_str = f", {total_bufs} buffers" if total_bufs else ""
        print(f"[Node3-3-1-1-1-1] Checkpoint: {trained}/{total} params ({pct}){buf_str}")

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

        # Muon group: QKV weight matrices (ndim >= 2)
        qkv_weights = [
            p for name, p in self.backbone.named_parameters()
            if p.requires_grad and p.ndim >= 2
            and any(s in name for s in ["query.weight", "key.weight", "value.weight"])
        ]

        # AdamW group: FFN params (up_proj, down_proj, gate_proj from last 2 layers)
        ffn_params = [
            p for name, p in self.backbone.named_parameters()
            if p.requires_grad
            and not any(s in name for s in ["query.weight", "key.weight", "value.weight"])
        ]

        # Head parameters (all layers of the MLP head)
        head_params = list(self.head.parameters())

        param_groups = [
            # Muon group: QKV weight matrices only
            dict(
                params       = qkv_weights,
                use_muon     = True,
                lr           = hp.lr_muon,
                weight_decay = hp.weight_decay,
                momentum     = 0.95,
            ),
            # AdamW group: FFN + head params
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
        opt_cls  = MuonWithAuxAdam if use_distributed else SingleDeviceMuonWithAuxAdam
        optimizer = opt_cls(param_groups)

        if self._num_training_steps > 0:
            num_steps    = self._num_training_steps
            warmup_steps = self._num_warmup_steps
            T0_steps     = self._T0_steps
        else:
            # Default fallback (8 GPUs, micro_batch=16, global_batch=128, 120 epochs)
            num_steps    = 1320
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


# ---------------------------------------------------------------------------
# Ensemble utilities
# ---------------------------------------------------------------------------
def discover_top_k_checkpoints(
    ckpt_dir: Path,
    top_k: int = 3,
) -> List[Tuple[Path, float]]:
    """Read checkpoint files from disk and return top-k by val/f1 score.

    Checkpoints are saved in subdirectories: dirpath/best-epoch={epoch:03d}-val/f1={f1:.4f}.ckpt
    We use recursive glob (**/*.ckpt) to find files inside subdirectories, then extract
    the val/f1 score from the filename (f1=X.ckpt format).

    Returns:
        List of (checkpoint_path, f1_score) tuples sorted descending by F1.
    """
    import re
    if not ckpt_dir.exists():
        return []
    # Recursive glob: find all .ckpt files inside subdirectories
    ckpt_files = list(ckpt_dir.glob("**/*.ckpt"))
    results = []
    for fp in ckpt_files:
        # Filename format: f1=0.4976.ckpt
        m = re.search(r"f1=([0-9.]+)\.ckpt$", fp.name)
        if m:
            f1 = float(m.group(1))
            results.append((fp, f1))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def write_ensemble_predictions(
    pred_dicts: List[Dict[str, Tuple[str, str, torch.Tensor]]],
    output_dir: Path,
) -> None:
    """Average softmax predictions from multiple checkpoint runs and write to TSV.

    Args:
        pred_dicts: List of {sample_idx: (pert_id, symbol, probs [3,6640])} dicts,
                    one dict per checkpoint run.
        output_dir: Directory to write test_predictions.tsv.
    """
    if not pred_dicts:
        print("[Node3-3-1-1-1-1] Warning: No predictions collected for ensemble.")
        return

    all_idxs = sorted(pred_dicts[0].keys())
    n_ckpts  = len(pred_dicts)

    rows = []
    for idx in all_idxs:
        pid = pred_dicts[0][idx][0]
        sym = pred_dicts[0][idx][1]
        # Collect probability tensors from each checkpoint run
        probs_list = [d[idx][2] for d in pred_dicts if idx in d]
        avg_prob   = torch.stack(probs_list).mean(0)  # [3, 6640]
        rows.append({
            "idx":        pid,
            "input":      sym,
            "prediction": json.dumps(avg_prob.float().cpu().numpy().tolist()),
        })

    out_path = output_dir / "test_predictions.tsv"
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(
        f"[Node3-3-1-1-1-1] Saved {len(rows)} ensemble-averaged predictions "
        f"({n_ckpts} checkpoint{'s' if n_ckpts != 1 else ''}) to {out_path}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description=(
            "Node3-3-1-1-1-1 – AIDO.Cell-10M + QKV+FFN(2L) + SGDR(T_0=15) + "
            "Checkpoint Ensemble (top-3)"
        )
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=16,   dest="micro_batch_size")
    parser.add_argument("--global-batch-size",   type=int,   default=128,  dest="global_batch_size")
    parser.add_argument("--max-epochs",          type=int,   default=120,  dest="max_epochs",
                        help="Reverted to 120 (grandparent node3-3-1-1 value)")
    parser.add_argument("--lr-muon",             type=float, default=0.02, dest="lr_muon")
    parser.add_argument("--lr-adamw",            type=float, default=2e-4, dest="lr_adamw")
    parser.add_argument("--weight-decay",        type=float, default=2e-2, dest="weight_decay")
    parser.add_argument("--fusion-layers",       type=int,   default=4,    dest="fusion_layers")
    parser.add_argument("--head-hidden",         type=int,   default=512,  dest="head_hidden")
    parser.add_argument("--head-dropout",        type=float, default=0.35, dest="head_dropout")
    parser.add_argument("--label-smoothing",     type=float, default=0.1,  dest="label_smoothing")
    parser.add_argument("--ffn-unfreeze-layers", type=int,   default=2,    dest="ffn_unfreeze_layers",
                        help="FFN=2 confirmed optimal in node3-3-1-1")
    parser.add_argument("--warmup-epochs",       type=int,   default=5,    dest="warmup_epochs")
    parser.add_argument("--min-lr-ratio",        type=float, default=0.05, dest="min_lr_ratio")
    parser.add_argument("--sgdr-t0-epochs",      type=int,   default=15,   dest="sgdr_T0_epochs",
                        help="T_0=15: restart at epoch 20 aligned with convergence peak")
    parser.add_argument("--sgdr-t-mult",         type=int,   default=2,    dest="sgdr_T_mult")
    parser.add_argument("--es-patience",         type=int,   default=8,    dest="es_patience",
                        help="Reverted to 8 (grandparent value, correctly captured peak)")
    parser.add_argument("--top-k-ckpts",         type=int,   default=3,    dest="top_k_ckpts",
                        help="Number of top checkpoints to ensemble during test")
    parser.add_argument("--val-check-interval",  type=float, default=1.0,  dest="val_check_interval")
    parser.add_argument("--num-workers",         type=int,   default=4,    dest="num_workers")
    parser.add_argument("--debug_max_step",      type=int,   default=None, dest="debug_max_step")
    parser.add_argument("--fast_dev_run",        action="store_true",      dest="fast_dev_run")
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

    # Compute steps for LR schedule
    n_train_samples    = 1388
    steps_per_epoch    = max(1, math.ceil(n_train_samples / (args.micro_batch_size * n_gpus)) // accum)
    num_training_steps = steps_per_epoch * args.max_epochs
    num_warmup_steps   = steps_per_epoch * args.warmup_epochs
    T0_steps           = steps_per_epoch * args.sgdr_T0_epochs

    print(
        f"[Node3-3-1-1-1-1] steps_per_epoch={steps_per_epoch}, "
        f"num_training_steps={num_training_steps}, "
        f"num_warmup_steps={num_warmup_steps}, "
        f"T0_steps={T0_steps} (T_0={args.sgdr_T0_epochs} epochs, "
        f"restart at epoch {args.warmup_epochs + args.sgdr_T0_epochs}), "
        f"top_k_ckpts={args.top_k_ckpts}"
    )

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOCell10MQKVFFNSGDREnsembleModel(
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
    )

    # Inject computed schedule info before configure_optimizers is called
    model._num_training_steps = num_training_steps
    model._num_warmup_steps   = num_warmup_steps
    model._T0_steps           = T0_steps
    model._steps_per_epoch    = steps_per_epoch

    # Top-k checkpoints for ensemble (reverted to top-1 in debug mode)
    save_top_k = args.top_k_ckpts if (not fast_dev_run and args.debug_max_step is None) else 1

    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = save_top_k,
    )
    # Reverted to patience=8 and min_delta=0.001 (grandparent node3-3-1-1 values)
    es_cb  = EarlyStopping(monitor="val/f1", mode="max",
                           patience=args.es_patience, min_delta=0.001)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pg_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # find_unused_parameters=False: no frozen-grad DDP issue (GenePriorBias removed)
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

    # ---- Test phase ----
    if fast_dev_run or args.debug_max_step is not None:
        # Simple single-pass test for debugging
        test_results = trainer.test(model, datamodule=dm, ckpt_path=None)
        # Write predictions even in debug mode (for verification)
        if trainer.is_global_zero and model._current_test_results is not None:
            write_ensemble_predictions([model._current_test_results], output_dir)
    else:
        # Ensemble test: discover top-k checkpoints from disk (safe across ranks)
        # We read checkpoint files directly from disk using the filename-embedded
        # val/f1 score — this avoids any rank-dependent in-memory state issues
        # that can arise from stale best_k_models dicts (which may contain
        # checkpoints from a previous training run on a different rank).
        ckpt_dir = output_dir / "checkpoints"
        disk_ckpts = discover_top_k_checkpoints(ckpt_dir, top_k=args.top_k_ckpts)

        if disk_ckpts:
            top_k_paths = [str(path) for path, score in disk_ckpts]
            print(f"[Node3-3-1-1-1-1] Running ensemble test on {len(top_k_paths)} checkpoint(s):")
            for path, score in disk_ckpts:
                print(f"  {path.name}  (val/f1={score:.4f})")

            # Run test on each checkpoint, collecting results
            all_pred_dicts: List[Dict] = []
            test_results = None
            for ckpt_f in top_k_paths:
                tr = trainer.test(model, datamodule=dm, ckpt_path=ckpt_f)
                if test_results is None:
                    test_results = tr  # store first run's metrics for logging
                if trainer.is_global_zero and model._current_test_results is not None:
                    all_pred_dicts.append(model._current_test_results.copy())
                    model._current_test_results = None  # reset for next run

            # Average predictions and write final ensemble output
            if trainer.is_global_zero and all_pred_dicts:
                write_ensemble_predictions(all_pred_dicts, output_dir)
        else:
            # Fallback: no checkpoints on disk
            print("[Node3-3-1-1-1-1] Warning: No checkpoints found on disk. Running single test pass.")
            test_results = trainer.test(model, datamodule=dm, ckpt_path="best")
            if trainer.is_global_zero and model._current_test_results is not None:
                write_ensemble_predictions([model._current_test_results], output_dir)

    # Write test_score.txt with test metrics summary
    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"{k}: {v}\n")
        f.write(f"\ntest_predictions: {output_dir}/test_predictions.tsv\n")
    print(f"[Node3-3-1-1-1-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
