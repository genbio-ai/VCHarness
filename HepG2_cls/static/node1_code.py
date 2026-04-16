"""Node 1: Baseline — Learned Gene Embedding + Deep MLP (no pretrained weights).

Architecture:
  - Input: perturbed gene identifier (ENSG ID or symbol)
  - Learned embedding table over all unique gene identifiers in train+val
  - Deep MLP with batch norm, residual connections, dropout
  - Output head: [B, embed_dim] → [B, 6640 * 3] → reshape to [B, 3, 6640]
  - Weighted cross-entropy loss (class weights inversely proportional to class frequency)

This is the performance-floor baseline without any pretrained knowledge.
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640          # number of response genes
N_CLASSES = 3           # {-1→0, 0→1, 1→2}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Dataset for gene perturbation → differential expression prediction."""

    def __init__(self, df: pd.DataFrame, gene2idx: Dict[str, int]) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        # Map to integer indices; fall back to UNK=0 for unseen genes
        self.gene_indices = torch.tensor(
            [gene2idx.get(sym, 0) for sym in self.symbols], dtype=torch.long
        )
        if "label" in df.columns:
            labels = np.array([json.loads(x) for x in df["label"].tolist()], dtype=np.int64)
            # Shift {-1,0,1} → {0,1,2}
            self.labels = torch.tensor(labels + 1, dtype=torch.long)  # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "gene_idx": self.gene_indices[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]  # [6640]
        return item


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class PerturbDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        micro_batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

        self.gene2idx: Dict[str, int] = {}
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        # Build gene vocabulary from train + val symbols (use index 0 for UNK)
        if not self.gene2idx:
            all_symbols = set(train_df["symbol"].tolist()) | set(val_df["symbol"].tolist())
            # index 0 = UNK
            self.gene2idx = {sym: i + 1 for i, sym in enumerate(sorted(all_symbols))}

        self.train_ds = PerturbDataset(train_df, self.gene2idx)
        self.val_ds = PerturbDataset(val_df, self.gene2idx)
        self.test_ds = PerturbDataset(test_df, self.gene2idx)

    @property
    def vocab_size(self) -> int:
        return len(self.gene2idx) + 1  # +1 for UNK at index 0

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """MLP residual block with batch norm."""
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.BatchNorm1d(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class BaselineMLP(nn.Module):
    """Deep MLP baseline for perturbation response prediction."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        n_blocks: int = 6,
        n_genes: int = N_GENES,
        n_classes: int = N_CLASSES,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([ResidualBlock(embed_dim, dropout) for _ in range(n_blocks)])
        # Project to n_genes * n_classes
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_genes * n_classes),
        )
        self.n_genes = n_genes
        self.n_classes = n_classes

    def forward(self, gene_idx: torch.Tensor) -> torch.Tensor:
        # gene_idx: [B]
        x = self.embedding(gene_idx)  # [B, embed_dim]
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)  # [B, n_genes * n_classes]
        return logits.view(-1, self.n_classes, self.n_genes)  # [B, 3, 6640]


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        n_blocks: int = 6,
        dropout: float = 0.15,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing

        self.model: Optional[BaselineMLP] = None

        # Validation / test buffers for metric computation
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Class weights: inversely proportional to frequency in training data
        # Approx: class0 (neutral) 92.82%, class1 (down) 4.77%, class2 (up) 2.41%
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq)
        class_weights = class_weights / class_weights.sum() * N_CLASSES  # normalize
        self.register_buffer("class_weights", class_weights)

        # Initialize model in setup() (not __init__) per Lightning best practices.
        # Guard: only init if not already set (handles re-entrant calls from test stage).
        if self.model is None:
            self.model = BaselineMLP(
                vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                n_blocks=self.n_blocks,
                dropout=self.dropout,
            )
            self.print(
                f"BaselineMLP | vocab={self.vocab_size} | embed={self.embed_dim} | "
                f"blocks={self.n_blocks} | params={sum(p.numel() for p in self.parameters()):,}"
            )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy with optional label smoothing.

        logits: [B, 3, 6640]
        labels: [B, 6640]  values in {0,1,2}
        """
        B = logits.size(0)
        # logits: [B, 3, 6640] → [B*6640, 3]
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_flat = labels.reshape(-1)  # [B*6640]
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self.model(batch["gene_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["gene_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # Accumulate for F1 computation
        self._val_preds.append(logits.detach().cpu())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds_local = torch.cat(self._val_preds, dim=0)   # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0) # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather validation data from all DDP ranks for accurate global F1
        # all_gather with world_size=1 returns same shape; with ws>1 prepends [world_size]
        all_preds = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640] or [N_local, 3, 6640]
        all_labels = self.all_gather(labels_local) # [world_size, N_local, 6640] or [N_local, 6640]
        ws = self.trainer.world_size
        if ws > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
            all_labels = all_labels.view(-1, N_GENES)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        # all_gather always prepends world_size dim; squeeze when ws==1
        if preds_np.shape[0] == 1:
            preds_np = preds_np[0]
            labels_np = labels_np[0]

        f1 = _compute_per_gene_f1(preds_np, labels_np)
        # sync_dist=True ensures only rank-0 metric is logged as the canonical value
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["gene_idx"])  # [B, 3, 6640]
        self._test_preds.append(logits.detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        # Gather predictions from all DDP ranks
        all_preds = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640] always
        ws = self.trainer.world_size
        if ws > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)  # [N_total, 3, 6640]
        else:
            all_preds = all_preds.squeeze(0)  # [1, N_local, 3, 6640] → [N_local, 3, 6640]

        # Gather pert_ids and symbols from all ranks.
        # Use gather_object to collect to rank 0 only (where saving happens).
        # all_gather_object requires pre-allocated list of world_size — use gather instead.
        all_pert_ids: List[str] = []
        all_symbols: List[str] = []
        if self.trainer.is_global_zero:
            # Pre-allocate lists for gather_object
            _pert_ids_gathered: List[List[str]] = [[] for _ in range(ws)]
            _symbols_gathered: List[List[str]] = [[] for _ in range(ws)]
            torch.distributed.gather_object(self._test_pert_ids, _pert_ids_gathered, dst=0)
            torch.distributed.gather_object(self._test_symbols, _symbols_gathered, dst=0)
            for p_list, s_list in zip(_pert_ids_gathered, _symbols_gathered):
                all_pert_ids.extend(p_list)
                all_symbols.extend(s_list)
        else:
            torch.distributed.gather_object(self._test_pert_ids, dst=0)
            torch.distributed.gather_object(self._test_symbols, dst=0)

        if self.trainer.is_global_zero:
            if all_preds.size(0) == 0:
                self.print("Warning: No test predictions gathered. Skipping save.")
            else:
                preds_np = all_preds.float().cpu().numpy()  # [N, 3, 6640]
                _save_test_predictions(
                    pert_ids=all_pert_ids,
                    symbols=all_symbols,
                    preds=preds_np,
                    out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
                )
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/f1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                trainable[key] = full_sd[key]
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute per-gene macro F1 averaged over genes (matches calc_metric.py logic).

    preds:  [N, 3, 6640] float — class logits
    labels: [N, 6640]    int   — class indices in {0,1,2}
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N, 6640]
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        f1_vals.append(float(per_class_f1[present].mean()))
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    """Save test predictions in required TSV format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
        pred_list = preds[i].tolist()  # [[3 values] × 6640] → shape [3][6640]
        rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Node1: Baseline MLP for HepG2 perturbation response")
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- DataModule ---
    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup("fit")

    # --- Model ---
    model = PerturbModule(
        vocab_size=datamodule.vocab_size,
        embed_dim=args.embed_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    # --- Trainer config ---
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        max_steps = -1

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.early_stop_patience,
        min_delta=1e-5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=1.0,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Save score
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")


if __name__ == "__main__":
    main()
