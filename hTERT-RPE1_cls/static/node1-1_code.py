"""
Node 1-2 – AIDO.Cell-10M + LoRA + Bilinear Interaction Head

Architecture:
  - AIDO.Cell-10M backbone with LoRA fine-tuning (r=8)
  - Perturbation profile: {perturbed_gene: expression=1.0, all others: missing}
  - Extract gene-specific embedding at vocabulary position of perturbed gene
  - Bilinear interaction head: pert_emb [B,256] → [B,3,rank] @ out_gene_emb[6640,rank].T → [B,3,6640]
  - Focal cross-entropy loss (gamma=2) with inverse-frequency class weights
  - Cosine annealing LR with warmup (no aggressive ReduceLROnPlateau)
  - Differential LR: backbone (LoRA) 1e-4, head 5e-4

Improvements over Node 1 (MLP baseline, F1=0.3762):
  1. Pretrained biological knowledge via AIDO.Cell-10M → addresses primary bottleneck
  2. Focal loss (gamma=2) → better handling of severe class imbalance (88.9% neutral)
  3. Cosine LR schedule with warmup → avoids premature LR decay seen in node1
  4. Bilinear interaction head → richer gene-gene interaction modeling
  5. Differential learning rates → backbone and head optimized independently
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
from functools import partial
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─── Constants ────────────────────────────────────────────────────────────────

N_GENES_OUT = 6640
N_CLASSES = 3
MODEL_DIR = "/home/Models/AIDO.Cell-10M"
AIDO_HIDDEN_DIM = 256  # AIDO.Cell-10M hidden size

# Inverse-frequency class weights from training distribution:
#   class 0 (down,  8.14%) → 1/0.0814 ≈ 12.28
#   class 1 (neutral, 88.86%) → 1/0.8886 ≈  1.12
#   class 2 (up,   3.00%) → 1/0.0300 ≈ 33.33
CLASS_WEIGHTS = torch.tensor([12.28, 1.12, 33.33], dtype=torch.float32)


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Exact per-gene macro F1 matching calc_metric.py logic.

    Args:
        pred_logits_np: [N, 3, G] float (logits or probabilities)
        labels_np:      [N, G]    int   (class indices 0/1/2)

    Returns:
        Mean per-gene F1 score (float).
    """
    pred_classes = pred_logits_np.argmax(axis=1)  # [N, G]
    n_genes = labels_np.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = labels_np[:, g]
        yh = pred_classes[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Loss ─────────────────────────────────────────────────────────────────────

def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal cross-entropy loss for multi-output classification.

    Args:
        logits:  [B, C, G] float32 – per-class logits
        targets: [B, G]    long    – class indices 0..C-1
        class_weights: [C] float32
        gamma:   focusing parameter (0 = standard CE)

    Returns:
        Scalar loss.
    """
    # Standard weighted CE (reduction='none' to get per-element loss)
    ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weights.to(logits.device),
        reduction="none",
    )  # [B, G]
    # Focal modulation
    pt = torch.exp(-ce)
    focal = (1.0 - pt) ** gamma * ce
    return focal.mean()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset.  Labels are optionally present."""

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        # Map gene symbol to AIDO.Cell vocabulary position; -1 for unknown
        self.gene_positions: List[int] = [
            gene_to_pos.get(sym, -1) for sym in self.symbols
        ]
        self.has_labels = has_labels
        if has_labels and "label" in df.columns:
            rows = []
            for lbl_str in df["label"]:
                rows.append([x + 1 for x in json.loads(lbl_str)])
            self.labels = torch.tensor(rows, dtype=torch.long)  # [N, G]
        else:
            self.has_labels = False

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int):
        item = {
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "gene_pos": self.gene_positions[idx],
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


def build_collate_fn(tokenizer):
    """Returns a collate_fn that tokenizes each sample's perturbation profile."""

    def collate_fn(batch):
        # Build perturbation expression profiles: {symbol: 1.0}, others missing
        expr_inputs = [
            {"gene_names": [item["symbol"]], "expression": [1.0]}
            for item in batch
        ]
        tokenized = tokenizer(expr_inputs, return_tensors="pt")

        gene_positions = torch.tensor(
            [item["gene_pos"] for item in batch], dtype=torch.long
        )
        pert_ids = [item["pert_id"] for item in batch]
        symbols = [item["symbol"] for item in batch]

        result = {
            "input_ids": tokenized["input_ids"],        # [B, 19264] float32
            "attention_mask": tokenized["attention_mask"],  # [B, 19264] int64
            "gene_pos": gene_positions,                 # [B] long; -1 = unknown
            "pert_id": pert_ids,
            "symbol": symbols,
        }
        if "label" in batch[0]:
            result["label"] = torch.stack(
                [item["label"] for item in batch], dim=0
            )
        return result

    return collate_fn


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule for perturbation DEG prediction with AIDO.Cell."""

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.gene_to_pos: Dict[str, int] = {}
        self.tokenizer = None

    def setup(self, stage: Optional[str] = None):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        # Rank-0 downloads tokenizer first; all ranks sync before loading
        if local_rank == 0:
            AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, trust_remote_code=True
        )
        # Build gene symbol → AIDO.Cell vocabulary position mapping
        self.gene_to_pos = {
            sym: pos for sym, pos in self.tokenizer.gene_to_index.items()
        }

        # Load splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        self.train_ds = PerturbationDataset(dfs["train"], self.gene_to_pos, True)
        self.val_ds = PerturbationDataset(dfs["val"], self.gene_to_pos, True)
        self.test_ds = PerturbationDataset(dfs["test"], self.gene_to_pos, True)

        # Pre-build collate function (tokenizer is serializable-safe in workers)
        self._collate = build_collate_fn(self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate,
            persistent_workers=self.num_workers > 0,
        )


# ─── Model ────────────────────────────────────────────────────────────────────

class AIDOCellPerturbationModel(nn.Module):
    """AIDO.Cell-10M + LoRA backbone + bilinear interaction DEG prediction head."""

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        rank: int = 128,
        head_dropout: float = 0.1,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()

        # Load backbone in bfloat16 so that hidden states are bf16, which triggers
        # the FlashAttention path (config._use_flash_attention_2=True is already set).
        # Without bf16 dtype the standard O(L²) attention would be used on 19266
        # tokens, requiring ~95 GB for attention scores alone → OOM.
        backbone = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                              dtype=torch.bfloat16)

        # Apply LoRA
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Bilinear interaction head
        # pert_emb [B, D] → proj_bilinear [B, n_classes * rank]
        # → reshape [B, n_classes, rank] @ out_gene_emb[n_genes, rank].T
        # → logits [B, n_classes, n_genes]
        self.dropout = nn.Dropout(head_dropout)
        self.proj_bilinear = nn.Linear(AIDO_HIDDEN_DIM, n_classes * rank, bias=True)
        self.out_gene_emb = nn.Embedding(n_genes_out, rank)
        nn.init.normal_(self.out_gene_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)

        self.n_classes = n_classes
        self.rank = rank

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, 19264] float32
        attention_mask: torch.Tensor,  # [B, 19264] int64
        gene_pos: torch.Tensor,        # [B] long; -1 = unknown
    ) -> torch.Tensor:
        """Returns logits [B, 3, 6640]."""
        # Forward through AIDO.Cell backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # [B, 19266, 256]

        # Extract perturbed-gene embedding at its vocabulary position
        # For out-of-vocab genes (gene_pos == -1): fall back to mean-pool
        B = input_ids.shape[0]
        valid_mask = gene_pos >= 0  # [B]

        # Safe index: clamp -1 to 0 to avoid out-of-bounds (will be overwritten)
        safe_pos = gene_pos.clone()
        safe_pos[~valid_mask] = 0

        # Gather gene-specific embeddings [B, 256]
        pert_emb = hidden[torch.arange(B, device=hidden.device), safe_pos]

        # Override unknown-gene entries with mean-pool of all gene positions
        if (~valid_mask).any():
            mean_emb = hidden[:, :19264, :].mean(dim=1)  # [B, 256]
            pert_emb = pert_emb.clone()
            pert_emb[~valid_mask] = mean_emb[~valid_mask]

        # Cast to float32 for head computation stability
        pert_emb = pert_emb.float()

        # Bilinear interaction
        pert_emb = self.dropout(pert_emb)          # [B, 256]
        proj = self.proj_bilinear(pert_emb)          # [B, n_classes*rank]
        proj = proj.view(B, self.n_classes, self.rank)  # [B, 3, rank]

        out_emb = self.out_gene_emb.weight           # [6640, rank]
        logits = torch.einsum("bcr,gr->bcg", proj, out_emb)  # [B, 3, 6640]
        return logits


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(
    local_preds: torch.Tensor,
    local_labels: torch.Tensor,
    device: torch.device,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather variable-length tensors from all DDP ranks with padding."""
    local_size = torch.tensor([local_preds.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = int(max(s.item() for s in all_sizes))

    pad = max_size - local_preds.shape[0]
    p = local_preds.to(device)
    l = local_labels.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], dim=0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], dim=0)

    g_preds  = [torch.zeros_like(p) for _ in range(world_size)]
    g_labels = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(g_preds, p)
    dist.all_gather(g_labels, l)

    real_preds  = torch.cat([g_preds[i][: all_sizes[i].item()].cpu()  for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][: all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """LightningModule for gene-perturbation DEG prediction (Node 1-2: AIDO.Cell)."""

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        rank: int = 128,
        head_dropout: float = 0.1,
        lr_backbone: float = 1e-4,
        lr_head: float = 5e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 100,
        focal_gamma: float = 2.0,
        max_steps_total: int = 10000,  # used for cosine schedule
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams
        self.model = AIDOCellPerturbationModel(
            lora_r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            rank=hp.rank,
            head_dropout=hp.head_dropout,
        )
        # Cast all trainable params to float32 for stable optimization
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_pos: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, attention_mask, gene_pos)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return focal_cross_entropy(
            logits,
            labels,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_pos"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_pos"])
        if "label" in batch:
            loss = self._compute_loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())
        return logits

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        local_p = torch.cat(self._val_preds, dim=0)
        local_l = torch.cat(self._val_labels, dim=0)

        if self.trainer.world_size > 1:
            all_p, all_l = _gather_tensors(local_p, local_l, self.device, self.trainer.world_size)
        else:
            all_p, all_l = local_p, local_l

        f1 = compute_per_gene_f1(all_p.numpy(), all_l.numpy())
        # All ranks compute the same f1 (identical after all_gather); sync_dist=True
        # satisfies Lightning's DDP logging requirement without extra communication.
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_pos"])
        probs = torch.softmax(logits, dim=1)  # [B, 3, 6640]
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

        if "label" in batch:
            if not hasattr(self, "_test_labels"):
                self._test_labels = []
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs = torch.cat(self._test_preds, dim=0)
        dummy_labels = torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        if hasattr(self, "_test_labels") and self._test_labels:
            dummy_labels = torch.cat(self._test_labels, dim=0)
            del self._test_labels

        if self.trainer.world_size > 1:
            all_probs, all_labels = _gather_tensors(
                local_probs, dummy_labels, self.device, self.trainer.world_size
            )
            all_pert = [None] * self.trainer.world_size
            all_syms = [None] * self.trainer.world_size
            dist.all_gather_object(all_pert, self._test_pert_ids)
            dist.all_gather_object(all_syms, self._test_symbols)
            all_pert = [p for sub in all_pert for p in sub]
            all_syms = [s for sub in all_syms for s in sub]
        else:
            all_probs  = local_probs
            all_labels = dummy_labels
            all_pert   = self._test_pert_ids
            all_syms   = self._test_symbols

        if self.trainer.is_global_zero:
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            seen_ids: set = set()
            dedup_probs: list = []
            dedup_labels: list = []
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for i, (pert_id, symbol, probs) in enumerate(
                    zip(all_pert, all_syms, all_probs.numpy())
                ):
                    if pert_id not in seen_ids:
                        seen_ids.add(pert_id)
                        fh.write(f"{pert_id}\t{symbol}\t{json.dumps(probs.tolist())}\n")
                        dedup_probs.append(probs)
                        dedup_labels.append(all_labels[i].numpy())
            self.print(
                f"[Node1-2] Saved test predictions → {pred_path} ({len(seen_ids)} unique samples)"
            )

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-2] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Separate parameter groups: LoRA backbone vs. prediction head
        backbone_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and "backbone" in n
        ]
        head_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and "backbone" not in n
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": hp.lr_backbone},
                {"params": head_params,    "lr": hp.lr_head},
            ],
            weight_decay=hp.weight_decay,
        )

        # Cosine annealing with linear warmup
        def lr_lambda(current_step: int):
            if current_step < hp.warmup_steps:
                return float(current_step) / max(1, hp.warmup_steps)
            progress = float(current_step - hp.warmup_steps) / max(
                1, hp.max_steps_total - hp.warmup_steps
            )
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable params ─────────────────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        trainable_sd = {
            k: v for k, v in full_sd.items()
            if k in trainable_keys or k in buffer_keys
        }
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Node 1-2 – AIDO.Cell-10M + LoRA + Bilinear Head")
    p.add_argument("--data-dir",         type=str,   default="data")
    p.add_argument("--lora-r",           type=int,   default=8)
    p.add_argument("--lora-alpha",       type=int,   default=16)
    p.add_argument("--lora-dropout",     type=float, default=0.05)
    p.add_argument("--rank",             type=int,   default=128)
    p.add_argument("--head-dropout",     type=float, default=0.1)
    p.add_argument("--lr-backbone",      type=float, default=1e-4)
    p.add_argument("--lr-head",          type=float, default=5e-4)
    p.add_argument("--weight-decay",     type=float, default=1e-4)
    p.add_argument("--warmup-steps",     type=int,   default=100)
    p.add_argument("--focal-gamma",      type=float, default=2.0)
    p.add_argument("--micro-batch-size", type=int,   default=8)
    p.add_argument("--global-batch-size",type=int,   default=64)
    p.add_argument("--max-epochs",       type=int,   default=80)
    p.add_argument("--patience",         type=int,   default=15)
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0,
                   help="How often to run validation (fraction of epoch). Default: 1.0 (every epoch).")
    p.add_argument("--debug-max-step",   type=int,   default=None)
    p.add_argument("--fast-dev-run",     action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataModule — do NOT call dm.setup() here; the Trainer will call it after
    # initialising DDP so that the distributed barrier in setup() is effective.
    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # Compute total training steps for LR scheduler by reading the file header
    # instead of calling dm.setup() early (which would bypass the DDP barrier).
    _train_df_size = pd.read_csv(Path(args.data_dir) / "train.tsv", sep="\t", usecols=["pert_id"]).shape[0]
    steps_per_epoch = _train_df_size // (args.micro_batch_size * n_gpus)
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)
    max_steps_total = effective_steps_per_epoch * args.max_epochs

    # LightningModule
    lit = PerturbationLitModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        rank=args.rank,
        head_dropout=args.head_dropout,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        focal_gamma=args.focal_gamma,
        max_steps_total=max(max_steps_total, 1),
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-4)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    # Debug / fast-dev-run settings
    max_steps: int          = -1
    limit_train_batches: float | int = 1.0
    limit_val_batches:   float | int = 1.0
    limit_test_batches:  float | int = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps           = args.debug_max_step
        limit_train_batches = args.debug_max_step
        limit_val_batches   = 2
        limit_test_batches  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"Node 1-2 – AIDO.Cell-10M + LoRA + Bilinear Head\n"
            f"Test results from trainer: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
