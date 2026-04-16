#!/usr/bin/env python3
"""
Node 1: Baseline – Character-Level BiGRU Gene Encoder + MLP Head
=================================================================
No pretrained backbone. Establishes a performance floor for MCTS.

Strategy:
  - Encode gene symbol (e.g. "GCLC") via a character-level BiGRU
  - Additionally embed the last 5 hex-digits of the Ensembl ID for uniqueness
  - Concatenate and feed to a deep MLP that predicts [3, 6640] logits
  - Focal loss to handle extreme class imbalance (95% class 0)
  - Label smoothing (0.1) for regularisation on the tiny dataset
  - AdamW + ReduceLROnPlateau
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
N_GENES = 6_640          # output genes
N_CLASSES = 3            # {0=down, 1=unchanged, 2=up}
# Class weights: inverse frequency (train: ~3.41%, ~95.48%, ~1.10%)
CLASS_WEIGHTS = torch.tensor([1.0 / 0.0356, 1.0 / 0.9482, 1.0 / 0.0110], dtype=torch.float32)

# Character vocabulary: letters, digits, dash, underscore + <pad>
CHAR_VOCAB = {c: i + 1 for i, c in enumerate(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.abcdefghijklmnopqrstuvwxyz"
)}
CHAR_VOCAB["<pad>"] = 0
MAX_SYM_LEN = 16          # longest gene symbol we expect
MAX_ID_SUFFIX_LEN = 8     # last N characters of the numeric Ensembl ID suffix


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
def encode_symbol(symbol: str, max_len: int = MAX_SYM_LEN) -> torch.Tensor:
    """Encode a gene symbol as a padded integer sequence."""
    sym = symbol.upper()[:max_len]
    ids = [CHAR_VOCAB.get(c, CHAR_VOCAB.get(c.upper(), 0)) for c in sym]
    # Pad to max_len
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def encode_ensg_suffix(pert_id: str, max_len: int = MAX_ID_SUFFIX_LEN) -> torch.Tensor:
    """Encode the numeric suffix of an Ensembl ID (last max_len digits)."""
    # ENSG00000001084 → "00000001084" → take last max_len chars → encode as chars
    digits = re.sub(r"\D", "", pert_id)[-max_len:]
    digits = digits.zfill(max_len)          # left-pad with zeros
    ids = [int(c) + 1 for c in digits]     # 0-9 → 1-10, leaving 0 for pad
    return torch.tensor(ids, dtype=torch.long)


class PerturbationDataset(Dataset):
    """Single-fold perturbation→DEG dataset."""

    def __init__(self, df: pd.DataFrame, is_test: bool = False):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
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
            "sym_ids": encode_symbol(self.symbols[idx]),        # [MAX_SYM_LEN]
            "ensg_ids": encode_ensg_suffix(self.pert_ids[idx]), # [MAX_ID_SUFFIX_LEN]
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)  # [6640]
        return item


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, micro_batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df)
            self.val_ds = PerturbationDataset(val_df)
        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(test_df, is_test=True)
            # Store metadata for prediction output
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
# Model
# ──────────────────────────────────────────────────────────────────────────────
class GeneSymbolEncoder(nn.Module):
    """Character-level bidirectional GRU encoder for gene symbols."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.out_dim = hidden_dim * 2  # bidirectional

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, L] → embedding: [B, out_dim]"""
        x = self.embedding(ids)              # [B, L, embed_dim]
        _, h = self.gru(x)                   # h: [num_layers*2, B, hidden_dim]
        # Concatenate last forward and backward hidden states
        h_fwd = h[-2]                        # [B, hidden_dim]
        h_bwd = h[-1]                        # [B, hidden_dim]
        return torch.cat([h_fwd, h_bwd], dim=-1)  # [B, out_dim]


class BaselineDEGModel(nn.Module):
    """Baseline DEG prediction model without pretrained backbone."""

    def __init__(self, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        char_vocab_size = len(CHAR_VOCAB)
        digit_vocab_size = 11  # 0=pad, 1-10 for digits 0-9

        # Symbol encoder (BiGRU)
        self.sym_encoder = GeneSymbolEncoder(char_vocab_size, embed_dim=32, hidden_dim=128, num_layers=2)
        # ENSG digit encoder (small BiGRU)
        self.id_encoder = GeneSymbolEncoder(digit_vocab_size, embed_dim=16, hidden_dim=64, num_layers=1)

        in_dim = self.sym_encoder.out_dim + self.id_encoder.out_dim  # 256 + 128 = 384

        # Deep MLP
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Output head: hidden_dim → [3 * N_GENES]
        self.head = nn.Linear(hidden_dim, N_CLASSES * N_GENES)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sym_ids: torch.Tensor, ensg_ids: torch.Tensor) -> torch.Tensor:
        """
        sym_ids:  [B, MAX_SYM_LEN]
        ensg_ids: [B, MAX_ID_SUFFIX_LEN]
        Returns:  [B, 3, N_GENES]
        """
        sym_emb = self.sym_encoder(sym_ids)    # [B, 256]
        id_emb = self.id_encoder(ensg_ids)     # [B, 128]
        x = torch.cat([sym_emb, id_emb], dim=-1)  # [B, 384]
        x = self.encoder(x)                    # [B, hidden_dim]
        logits = self.head(x)                  # [B, 3 * N_GENES]
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
        dropout: float = 0.3,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[BaselineDEGModel] = None
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
            self.model = BaselineDEGModel(
                hidden_dim=self.hparams.hidden_dim,
                dropout=self.hparams.dropout,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )
            # Cast trainable parameters to float32 for stable optimization
            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    v.data = v.data.float()
        # Store test metadata from DataModule for prediction saving
        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch["sym_ids"], batch["ensg_ids"])

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G], labels: [B, G] (0/1/2)"""
        B, C, G = logits.shape
        # Reshape for cross-entropy: [B*G, C] and [B*G]
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        labels_flat = labels.reshape(-1)                       # [B*G]
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

        # Compute local F1 on each rank (avoids large tensor all_gather NCCL timeout)
        local_preds_np = local_preds.numpy()
        local_f1 = compute_deg_f1(local_preds_np, local_labels.numpy())

        # Reduce F1 across all ranks (scalar all-reduce, no large tensor)
        world_size = self.trainer.world_size if self.trainer.world_size else 1
        if world_size > 1:
            import torch.distributed as dist
            # Must use GPU tensor for NCCL backend
            local_f1_t = torch.tensor(local_f1, dtype=torch.float32, device="cuda")
            dist.all_reduce(local_f1_t, op=dist.ReduceOp.SUM)
            f1 = (local_f1_t / world_size).item()
        else:
            f1 = local_f1

        self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
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
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5, min_lr=1e-6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_f1", "interval": "epoch"},
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
    p = argparse.ArgumentParser(description="Node 1: Baseline BiGRU DEG predictor")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.10)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
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

    val_check_interval = args.val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0

    # Use SingleDeviceStrategy for single GPU to avoid NCCL issues
    if n_gpus == 1:
        strategy = SingleDeviceStrategy(device="cuda:0")
    else:
        strategy = DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120))

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    # Use val_loss for EarlyStopping since val_f1 is only available on rank 0
    early_stop_cb = EarlyStopping(
        monitor="val_loss", mode="min",
        patience=args.early_stopping_patience, verbose=True,
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
    )
    model_module = DEGLightningModule(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
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
