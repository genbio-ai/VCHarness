#!/usr/bin/env python3
"""
Node 1-2: Frozen AIDO.Cell-100M + MLP Head
===========================================
Strategy:
  - Frozen AIDO.Cell-100M as biological gene encoder (no gradient updates to backbone)
  - Pre-compute and cache 640-dim cell embeddings per perturbation gene
  - Train only a lightweight MLP head on cached embeddings (fast iterations)
  - Fix critical checkpoint bug: monitor val_f1 (max), not val_loss (min)
  - CosineAnnealingLR for stable convergence over 300 epochs
  - Focal loss (γ=2) + class weights + label smoothing for extreme imbalance
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import time
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
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
N_GENES = 6_640          # output genes
N_CLASSES = 3            # {0=down, 1=unchanged, 2=up}
EMBED_DIM = 640          # AIDO.Cell-100M hidden dimension

# Class weights: inverse frequency (train: ~3.56% down, ~94.82% unchanged, ~1.10% up)
CLASS_WEIGHTS = torch.tensor([1.0 / 0.0356, 1.0 / 0.9482, 1.0 / 0.0110], dtype=torch.float32)

AIDO_MODEL_PATH = "/home/Models/AIDO.Cell-100M"


# ──────────────────────────────────────────────────────────────────────────────
# AIDO.Cell Embedding Extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_aido_embeddings(pert_ids: List[str], device: torch.device, batch_size: int = 1) -> torch.Tensor:
    """Extract 640-dim cell embeddings from frozen AIDO.Cell-100M.

    For each perturbation gene, constructs a synthetic expression profile:
    - The perturbed gene is expressed at 10.0
    - All other genes are missing (-1.0, auto-filled by tokenizer)
    This creates a minimal but biologically-grounded input signal.

    AIDO.Cell processes the full 19,264-gene transcriptome. With batch_size=64,
    the input tensor would be [64, 19264] float32 = ~50MB per tensor, but the
    attention mechanism would need ~176GB of activation memory. We therefore use
    batch_size=1 to stay within the 80GB GPU limit (frozen 100M = ~3.4GiB per sample).
    """
    tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
    backbone = backbone.to(torch.bfloat16).to(device)
    backbone.eval()
    backbone.config.use_cache = False

    all_emb = []
    n = len(pert_ids)
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            batch_ids = pert_ids[i:i + batch_size]
            # Each gene: expressed at 10.0, all others missing (-1.0 auto-filled)
            # Tokenize per-sample (not all at once) to avoid massive intermediate tensors
            emb_list = []
            for pid in batch_ids:
                expr_dicts = [{"gene_ids": [pid], "expression": [10.0]}]
                inputs = tokenizer(expr_dicts, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = backbone(**inputs)
                # Mean-pool over gene positions (exclude 2 summary tokens at end)
                emb = out.last_hidden_state[:, :19264, :].mean(dim=1).float().cpu()  # [1, 640]
                emb_list.append(emb)
            all_emb.append(torch.cat(emb_list, dim=0))  # [B, 640]
            if (i // batch_size) % 50 == 0 or (i + batch_size) >= n:
                print(f"  Embedding extraction: {min(i + batch_size, n)}/{n} done")

    del backbone
    torch.cuda.empty_cache()
    return torch.cat(all_emb, dim=0)  # [N, 640]


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weighting and label smoothing."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [N, C]
        targets: [N]  (int64, class indices)
        """
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Perturbation dataset using pre-computed AIDO.Cell embeddings."""

    def __init__(self, df: pd.DataFrame, embeddings: torch.Tensor, is_test: bool = False):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.embeddings = embeddings  # [N, 640]
        self.is_test = is_test
        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Remap: {-1→0, 0→1, 1→2}
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "embedding": self.embeddings[idx],  # [640]
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)  # [6640]
        return item


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        micro_batch_size: int = 32,
        num_workers: int = 4,
        embed_batch_size: int = 1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.embed_batch_size = embed_batch_size
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None

    def _get_or_compute_embeddings(self, pert_ids: List[str], split: str) -> torch.Tensor:
        """DDP-safe embedding precomputation with disk caching.

        Uses a file-based synchronization approach:
        1. All ranks check if cache exists
        2. Rank 0 computes and saves if needed
        3. All ranks wait for the file to exist (file-based barrier)
        4. All ranks load the cached embeddings
        """
        cache_dir = Path(__file__).parent / "run" / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{split}_embeddings.pt"

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Step 1: All ranks check cache existence
        cache_exists = cache_path.exists()

        # Step 2: If not cached, rank 0 computes and saves
        if not cache_exists:
            if local_rank == 0:
                print(f"[Rank 0] Computing {split} embeddings for {len(pert_ids)} perturbations...")
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                embeddings = extract_aido_embeddings(pert_ids, device, batch_size=self.embed_batch_size)
                torch.save(embeddings, cache_path)
                print(f"[Rank 0] Saved {split} embeddings to {cache_path}")
            else:
                # Non-rank-0: wait for the file to appear
                wait_count = 0
                while not cache_path.exists():
                    time.sleep(5)
                    wait_count += 1
                    if wait_count % 12 == 0:  # print every minute
                        print(f"[Rank {local_rank}] Still waiting for {cache_path.name}...")

        # Step 3: Synchronized load - all ranks wait until file is ready
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        embeddings = torch.load(cache_path, weights_only=True)
        return embeddings  # [N, 640]

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            train_emb = self._get_or_compute_embeddings(train_df["pert_id"].tolist(), "train")
            val_emb = self._get_or_compute_embeddings(val_df["pert_id"].tolist(), "val")
            self.train_ds = PerturbationDataset(train_df, train_emb, is_test=False)
            self.val_ds = PerturbationDataset(val_df, val_emb, is_test=False)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            test_emb = self._get_or_compute_embeddings(test_df["pert_id"].tolist(), "test")
            self.test_ds = PerturbationDataset(test_df, test_emb, is_test=True)
            self.test_pert_ids: List[str] = test_df["pert_id"].tolist()
            self.test_symbols: List[str] = test_df["symbol"].tolist()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# MLP Head Model
# ──────────────────────────────────────────────────────────────────────────────
class AIDOCellMLPHead(nn.Module):
    """MLP head on top of frozen AIDO.Cell-100M embeddings.

    Architecture:
      LayerNorm(640) → Linear(640, hidden_dim) → GELU → Dropout → LayerNorm(hidden_dim)
      → Linear(hidden_dim, 3*6640) → reshape [B, 3, 6640]
    """

    def __init__(self, embed_dim: int = 640, hidden_dim: int = 512, dropout: float = 0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, N_CLASSES * N_GENES),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, 640]
        Returns:    [B, 3, N_GENES]
        """
        logits = self.head(embeddings)  # [B, 3 * N_GENES]
        return logits.view(-1, N_CLASSES, N_GENES)  # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper (mirrors calc_metric.py)
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """
    y_pred:          [N, 3, G]  float  (probabilities or logits)
    y_true_remapped: [N, G]     int    ({0, 1, 2} after +1 remap)
    Returns: macro F1 averaged over G genes.
    """
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    y_hat = y_pred.argmax(axis=1)  # [N, G]
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp = y_hat[:, g]
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 512,
        dropout: float = 0.4,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.10,
        max_epochs: int = 300,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCellMLPHead] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = AIDOCellMLPHead(
                embed_dim=EMBED_DIM,
                hidden_dim=self.hparams.hidden_dim,
                dropout=self.hparams.dropout,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )
        # Store test metadata from DataModule for prediction saving
        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.model(embeddings)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G], labels: [B, G] (0/1/2)"""
        B, C, G = logits.shape
        # Reshape for cross-entropy: [B*G, C] and [B*G]
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        labels_flat = labels.reshape(-1)                       # [B*G]
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["embedding"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["embedding"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, dim=0)    # [N_local, 3, G]
        local_labels = torch.cat(self._val_labels, dim=0)  # [N_local, G]

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # Compute local F1 (avoids large-tensor all_gather NCCL overhead)
        local_preds_np = local_preds.numpy()
        local_f1 = compute_deg_f1(local_preds_np, local_labels.numpy())

        # Reduce F1 across all ranks (scalar all-reduce)
        world_size = self.trainer.world_size if self.trainer.world_size else 1
        if world_size > 1:
            import torch.distributed as dist
            local_f1_t = torch.tensor(local_f1, dtype=torch.float32, device=self.device)
            dist.all_reduce(local_f1_t, op=dist.ReduceOp.SUM)
            f1 = (local_f1_t / world_size).item()
        else:
            f1 = local_f1

        self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["embedding"])
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)
        local_idx = torch.cat(self._test_indices, dim=0)

        all_preds = self.all_gather(local_preds)
        all_idx = self.all_gather(local_idx)

        self._test_preds.clear()
        self._test_indices.clear()

        if self.trainer.is_global_zero:
            preds = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            idxs = all_idx.view(-1).cpu().numpy()
            # Deduplicate (may overlap in DDP)
            unique_pos = np.unique(idxs, return_index=True)[1]
            preds = preds[unique_pos]
            sorted_idxs = idxs[unique_pos]

            # Reorder by original index
            order = np.argsort(sorted_idxs)
            preds = preds[order]
            final_idxs = sorted_idxs[order]

            # Write test_predictions.tsv
            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"

            rows = []
            for rank_i, orig_i in enumerate(final_idxs):
                rows.append({
                    "idx": self._test_pert_ids[orig_i],
                    "input": self._test_symbols[orig_i],
                    "prediction": json.dumps(preds[rank_i].tolist()),
                })
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path}")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    # ── Checkpoint: save only trainable params ────────────────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                k = prefix + name
                if k in full:
                    trainable[k] = full[k]
        for name, buf in self.named_buffers():
            k = prefix + name
            if k in full:
                trainable[k] = full[k]
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Node 1-2: Frozen AIDO.Cell-100M + MLP Head DEG predictor")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.10)
    p.add_argument("--early-stopping-patience", type=int, default=30)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--embed-batch-size", type=int, default=1,
                   help="Batch size for AIDO.Cell embedding extraction")
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Trainer settings ──────────────────────────────────────────────────────
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit_train = 1.0
    limit_val = 1.0
    limit_test = 1.0
    if args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = args.debug_max_step

    val_check_interval = args.val_check_interval if (
        args.debug_max_step is None and not args.fast_dev_run
    ) else 1.0

    # Strategy selection
    if n_gpus == 1:
        strategy = SingleDeviceStrategy(device="cuda:0")
    else:
        strategy = DDPStrategy(timeout=timedelta(seconds=120))

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",   # FIXED: was val_loss in node1
        mode="max",          # FIXED: was min in node1
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",    # FIXED: was val_loss in node1
        mode="max",          # FIXED: was min in node1
        patience=args.early_stopping_patience,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    # ── Data & model ──────────────────────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        embed_batch_size=args.embed_batch_size,
    )
    model_module = DEGLightningModule(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Test ──────────────────────────────────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # ── Save test score ───────────────────────────────────────────────────────
    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"test_results: {test_results}\n"
            f"val_f1_best: {checkpoint_cb.best_model_score}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
