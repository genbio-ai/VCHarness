#!/usr/bin/env python3
"""
Node 1-2: AIDO.Cell-10M + LoRA – Reduced-Capacity Regularised DEG Predictor
============================================================================
Key design differences from parent node (node2 / AIDO.Cell-100M + LoRA r=16):
  1. Backbone: AIDO.Cell-10M (256-dim, 8 layers) instead of 100M – far fewer
     parameters for 1,500 training samples, reducing severe overfitting.
  2. LoRA applied to ALL 8 transformer layers (not just last 12) with r=8
     instead of r=16 — smaller rank reduces adapter capacity.
  3. Strongly increased regularisation:
       - lora_dropout: 0.05 → 0.25
       - weight_decay: 1e-4 → 1e-2
       - Head: two-layer MLP with Dropout(0.3) between layers
  4. Moderate class weights [5.0, 1.0, 10.0] replacing extreme [28.1, 1.05, 90.9]
     to prevent val_loss inflation while maintaining minority-class sensitivity.
  5. OneCycleLR scheduler with 5-epoch linear warmup instead of ReduceLROnPlateau
     for a more principled LR schedule decoupled from noisy val_f1 plateaus.
  6. Dual-pooling (global mean-pool + perturbed-gene positional embedding →
     concat → head) retained from parent node — proven effective.

Key differences from sibling node (node1-1 / Frozen AIDO.Cell-100M + MLP):
  - Backbone is fine-tuned via LoRA (not frozen) — critical for learning
    task-specific attention patterns from the synthetic perturbation profiles.
  - Uses perturbed-gene positional embedding (not just global mean-pool).
  - AIDO.Cell-10M instead of 100M as backbone.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
AIDO_CELL_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
N_GENES_AIDO = 19_264   # AIDO.Cell vocabulary size (fixed for all model sizes)
N_GENES_OUT = 6_640     # output genes
N_CLASSES = 3
SENTINEL_EXPR = 1.0     # baseline expression (non-perturbed genes)
KNOCKOUT_EXPR = 0.0     # expression for knocked-out gene (perturbed)
AIDO_HIDDEN = 256       # AIDO.Cell-10M hidden dimension
AIDO_N_LAYERS = 8       # AIDO.Cell-10M transformer layers

# Moderate class weights — inverse frequency but capped to avoid extreme values.
# Train distribution: class -1 (down) ~3.4%, class 0 (unchanged) ~95.5%, class 1 (up) ~1.1%
# Remapped: class 0 (down)=3.4%, class 1 (unchanged)=95.5%, class 2 (up)=1.1%
# Weights: [5.0, 1.0, 10.0] — severe overfitting in node2 was linked to [28.1, 1.05, 90.9]
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weights."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=w, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Returns pre-built AIDO.Cell expression profile tensors (float32) together
    with the perturbed gene position index and the label.

    Synthetic expression profile for each perturbation:
      - All 19,264 genes set to SENTINEL_EXPR (1.0) — "baseline expressed"
      - Knocked-out gene set to KNOCKOUT_EXPR (0.0) — "silenced"
    After AIDO.Cell's internal CP10K + log1p normalization, the knocked-out gene
    achieves ≈0 while others have ≈0.415, providing a clear distinguishing signal.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.is_test = is_test

        # Pre-build expression input tensors: [N, 19264] float32
        self.expr_inputs = self._build_expr_tensors()

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Remap {-1,0,1} → {0,1,2} per metric contract
            self.labels = np.array(raw_labels, dtype=np.int8) + 1
        else:
            self.labels = None

    def _build_expr_tensors(self) -> torch.Tensor:
        N = len(self.pert_ids)
        expr = torch.full((N, N_GENES_AIDO), SENTINEL_EXPR, dtype=torch.float32)
        for i, pert_id in enumerate(self.pert_ids):
            base = pert_id.split(".")[0]
            pos = self.gene_to_pos.get(base)
            if pos is not None:
                expr[i, pos] = KNOCKOUT_EXPR
        return expr

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.pert_ids[idx].split(".")[0]
        gene_pos = self.gene_to_pos.get(base, -1)
        item = {
            "idx": idx,
            "expr": self.expr_inputs[idx],   # [19264] float32
            "gene_pos": gene_pos,            # int (-1 if not in vocab)
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "expr": torch.stack([b["expr"] for b in batch]),   # [B, 19264]
        "gene_pos": torch.tensor([b["gene_pos"] for b in batch], dtype=torch.long),
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        micro_batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.gene_to_pos: Dict[str, int] = {}
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # Build ENSG→position mapping from all split pert_ids
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self.gene_to_pos)
            self.val_ds = PerturbationDataset(val_df, self.gene_to_pos)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(test_df, self.gene_to_pos, is_test=True)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    @staticmethod
    def _build_gene_to_pos(tokenizer, gene_ids: List[str]) -> Dict[str, int]:
        """Map each ENSG gene_id to its position index in AIDO.Cell vocab."""
        mapping: Dict[str, int] = {}
        PROBE_VAL = 50.0
        for gene_id in gene_ids:
            try:
                inputs = tokenizer(
                    {"gene_ids": [gene_id], "expression": [PROBE_VAL]},
                    return_tensors="pt",
                )
                ids = inputs["input_ids"]
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)  # [1, 19264]
                # Find position where the expression was injected
                pos = (ids[0] == PROBE_VAL).nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    mapping[gene_id] = int(pos[0].item())
            except Exception:
                pass
        return mapping

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class AIDOCell10MDEGModel(nn.Module):
    """AIDO.Cell-10M backbone + LoRA (all 8 layers) + dual-pooling head.

    Changes from parent node (node2):
    - AIDO.Cell-10M (256-dim, 8 layers) instead of 100M (640-dim, 18 layers)
    - LoRA applied to ALL 8 layers (not just last 12 of 18)
    - LoRA r=8 (vs r=16) — fewer adapter parameters
    - lora_dropout=0.25 (vs 0.05) — strong dropout regularisation
    - Head: two-layer MLP with LayerNorm + Dropout(0.3) between layers
    """

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.25,
        head_dropout: float = 0.3,
        head_hidden: int = 512,
    ):
        super().__init__()
        # Load AIDO.Cell-10M in bf16 to enable FlashAttention
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA on Q/K/V of ALL 8 layers (layers_to_transform=None means all layers)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,  # apply to all 8 layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Two-layer prediction head with dropout regularisation
        # Input: AIDO_HIDDEN * 2 = 512 (concat of global mean-pool + pert-gene emb)
        head_in = AIDO_HIDDEN * 2  # 512
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        # Initialise output layer conservatively
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        expr: torch.Tensor,      # [B, 19264] float32
        gene_pos: torch.Tensor,  # [B]        int64  (-1 if not in vocab)
    ) -> torch.Tensor:
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, 19266, 256]  (+2 summary tokens)

        # (a) Global mean-pool over all 19264 gene positions
        gene_emb = lhs[:, :N_GENES_AIDO, :]          # [B, 19264, 256]
        global_emb = gene_emb.mean(dim=1)             # [B, 256]

        # (b) Perturbed-gene positional embedding (AIDO.Cell-10M: 256-dim)
        B = expr.shape[0]
        pert_emb = torch.zeros(B, AIDO_HIDDEN, device=lhs.device, dtype=lhs.dtype)
        valid_mask = gene_pos >= 0
        if valid_mask.any():
            valid_pos = gene_pos[valid_mask]          # [k]
            pert_emb[valid_mask] = lhs[valid_mask, valid_pos, :]
        # Fallback for genes not in vocabulary: use global embedding
        pert_emb[~valid_mask] = global_emb[~valid_mask]

        # Concatenate → [B, 512] → head → [B, 3, 6640]
        combined = torch.cat([global_emb, pert_emb], dim=-1).float()  # [B, 512]
        logits = self.head(combined)                                   # [B, 3 * 6640]
        return logits.view(B, N_CLASSES, N_GENES_OUT)                  # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro F1, averaged over all genes.

    y_pred: [n_samples, 3, n_genes]   (3-class logits or probabilities)
    y_true_remapped: [n_samples, n_genes]  (labels in {0,1,2})
    """
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp_class = y_pred[:, :, g]           # [n_samples, 3]
        yhat = yp_class.argmax(axis=1)
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yhat, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.25,
        head_dropout: float = 0.3,
        head_hidden: int = 512,
        lr: float = 5e-4,
        weight_decay: float = 1e-2,
        gamma_focal: float = 2.0,
        max_epochs: int = 100,
        steps_per_epoch: int = 375,     # will be overridden at fit time
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCell10MDEGModel] = None
        self.criterion: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = AIDOCell10MDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                head_dropout=self.hparams.head_dropout,
                head_hidden=self.hparams.head_hidden,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
            )
            # Cast trainable (LoRA + head) params to float32 for stable AdamW optimization
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()
        if stage == "test" and hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(batch["expr"], batch["gene_pos"])

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_flat = labels.reshape(-1)
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        lp = torch.cat(self._val_preds, 0)
        ll = torch.cat(self._val_labels, 0)
        li = torch.cat(self._val_indices, 0)
        ap = self.all_gather(lp)
        al = self.all_gather(ll)
        ai = self.all_gather(li)
        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # Compute val_f1 on full gathered dataset, deduplicated by sample index
        preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
        labels = al.cpu().view(-1, N_GENES_OUT).numpy()
        idxs = ai.cpu().view(-1).numpy()
        _, uniq = np.unique(idxs, return_index=True)
        f1 = compute_deg_f1(preds[uniq], labels[uniq])

        # All-reduce so all DDP ranks share the same val_f1
        f1_tensor = torch.tensor(f1, dtype=torch.float32, device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(f1_tensor, op=torch.distributed.ReduceOp.SUM)
            f1_tensor = f1_tensor / self.trainer.world_size

        self.log("val_f1", f1_tensor.item(), prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        lp = torch.cat(self._test_preds, 0)
        li = torch.cat(self._test_indices, 0)
        ap = self.all_gather(lp)
        ai = self.all_gather(li)
        self._test_preds.clear()
        self._test_indices.clear()

        if self.global_rank == 0:
            preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
            idxs = ai.cpu().view(-1).numpy()
            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]
            idxs = idxs[uniq]
            order = np.argsort(idxs)
            preds = preds[order]
            idxs = idxs[order]

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for rank_i, orig_i in enumerate(idxs):
                rows.append({
                    "idx": self._test_pert_ids[orig_i],
                    "input": self._test_symbols[orig_i],
                    "prediction": json.dumps(preds[rank_i].tolist()),
                })
            out_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path}")

    def configure_optimizers(self):
        # Separate param groups: LoRA backbone params (lower lr) and head (higher lr)
        backbone_params = [p for n, p in self.model.backbone.named_parameters()
                           if p.requires_grad]
        head_params = list(self.model.head.parameters())

        # Head gets 5× higher lr to compensate for random init vs pretrained LoRA
        max_lr_backbone = self.hparams.lr
        max_lr_head = self.hparams.lr * 5

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": max_lr_backbone / 25,
                 "max_lr": max_lr_backbone},
                {"params": head_params, "lr": max_lr_head / 25,
                 "max_lr": max_lr_head},
            ],
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # OneCycleLR: starts low, ramps to max_lr, cosine decays to min
        # total_steps = steps_per_epoch * max_epochs (set dynamically in main)
        total_steps = self.hparams.steps_per_epoch * self.hparams.max_epochs
        warmup_pct = self.hparams.warmup_epochs / self.hparams.max_epochs

        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=[max_lr_backbone, max_lr_head],
            total_steps=total_steps,
            pct_start=warmup_pct,
            anneal_strategy="cos",
            div_factor=25,        # initial lr = max_lr / 25
            final_div_factor=1e4, # final lr = initial lr / 1e4
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        out = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                k = prefix + name
                if k in full:
                    out[k] = full[k]
        for name, buf in self.named_buffers():
            k = prefix + name
            if k in full:
                out[k] = full[k]
        return out

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 1-2: AIDO.Cell-10M LoRA DEG predictor (reduced capacity)"
    )
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.25)
    p.add_argument("--head-dropout", type=float, default=0.3)
    p.add_argument("--head-hidden", type=int, default=512)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def main():
    pl.seed_everything(0)
    args = parse_args()

    # Resolve data_dir relative to project root.
    # From mcts/node2-1/ we need 3 levels up to reach project root:
    # mcts/node2-1 -> mcts -> project_root -> parent of project root
    if args.data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    else:
        data_dir = Path(args.data_dir)
    args.data_dir = str(data_dir)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit = args.debug_max_step if args.debug_max_step is not None else 1.0

    # Estimate steps_per_epoch for OneCycleLR scheduler
    # 1500 train samples, drop_last=True; each GPU sees micro_batch_size samples per step
    n_train_samples = 1500
    steps_per_gpu_per_epoch = max(1, n_train_samples // (args.micro_batch_size * n_gpus))
    steps_per_epoch = max(1, steps_per_gpu_per_epoch // accumulate_grad)
    if args.debug_max_step is not None:
        steps_per_epoch = args.debug_max_step

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.early_stopping_patience, verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(refresh_rate=20)
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=180)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit,
        limit_val_batches=limit,
        limit_test_batches=limit,
        val_check_interval=(
            1.0 if (args.debug_max_step is not None or args.fast_dev_run)
            else args.val_check_interval
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=False,   # FlashAttention is non-deterministic
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        head_dropout=args.head_dropout,
        head_hidden=args.head_hidden,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        max_epochs=args.max_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )

    trainer.fit(model_module, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # Use torch.distributed to reliably check global rank (more robust than trainer.is_global_zero)
    is_rank_zero = (
        not torch.distributed.is_available() or
        not torch.distributed.is_initialized() or
        torch.distributed.get_rank() == 0
    )
    if is_rank_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"test_results: {test_results}\n"
            f"val_f1_best: {checkpoint_cb.best_model_score}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
