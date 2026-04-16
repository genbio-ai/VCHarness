"""Node 1 – Baseline Bilinear MLP (no pretrained backbone).

Strategy: Establish a performance floor with a scratch-trained gene-embedding model.
- Input: gene symbol → learnable embedding lookup
- Output: bilinear factored head for memory-efficient gene×class scoring
- Loss: class-weighted cross-entropy + label smoothing (handles 92.5% neutral imbalance)
- Optimizer: AdamW with ReduceLROnPlateau
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3
# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights; neutral class stays ≈ 1."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def build_vocab(dfs: List[pd.DataFrame]) -> Dict[str, int]:
    """Build symbol→index vocab from a list of DataFrames."""
    vocab: Dict[str, int] = {"<UNK>": 0}
    for df in dfs:
        for sym in df["symbol"].tolist():
            if sym not in vocab:
                vocab[sym] = len(vocab)
    return vocab


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0,1,2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)          # [N, G]
    G = targets.shape[1]
    f1_per_gene   = torch.zeros(G, device=preds.device)
    n_present     = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)          # [N, G]
        is_pred = (y_hat == c)            # [N, G]
        present = is_true.any(dim=0)      # [G]

        tp = (is_pred & is_true).float().sum(0)
        fp = (is_pred & ~is_true).float().sum(0)
        fn = (~is_pred & is_true).float().sum(0)

        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(
            prec + rec > 0,
            2 * prec * rec / (prec + rec + 1e-8),
            torch.zeros_like(prec),
        )
        f1_per_gene += f1_c * present.float()
        n_present   += present.float()

    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    """K562 DEG prediction dataset."""

    def __init__(self, df: pd.DataFrame, vocab: Dict[str, int]) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        self.gene_indices = torch.tensor(
            [vocab.get(s, 0) for s in self.symbols], dtype=torch.long
        )
        has_label = "label" in df.columns and df["label"].notna().all()
        if has_label:
            self.labels = [
                torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
                for row in df["label"].tolist()
            ]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.symbols)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx": idx,
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
            "gene_idx":   self.gene_indices[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]  # [G] in {0,1,2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx": torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":    [b["pert_id"]  for b in batch],
        "symbol":     [b["symbol"]   for b in batch],
        "gene_idx":   torch.stack([b["gene_idx"] for b in batch]),
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.vocab: Optional[Dict[str, int]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        if self.vocab is None:
            self.vocab = build_vocab([train_df, val_df, test_df])

        self.train_ds = DEGDataset(train_df, self.vocab)
        self.val_ds   = DEGDataset(val_df,   self.vocab)
        self.test_ds  = DEGDataset(test_df,  self.vocab)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BilinearDEGModel(pl.LightningModule):
    """Baseline: gene embedding + MLP + bilinear factored output head."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        bilinear_dim: int = 128,
        dropout: float = 0.3,
        lr: float = 5e-3,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Model initialized in setup()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        self.embedding = nn.Embedding(hp.vocab_size, hp.embed_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hp.embed_dim),
            nn.Linear(hp.embed_dim, hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
            nn.LayerNorm(hp.hidden_dim),
            nn.Linear(hp.hidden_dim, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )
        # Bilinear gene-class embedding matrix: [C, G, bilinear_dim]
        # logits[b, c, g] = pert_emb[b] · gene_class_emb[c, g]
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )
        self.register_buffer("class_weights", get_class_weights())

        # Cast trainable parameters to float32 for stable optimization
        for k, v in self.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []  # sample indices for proper DDP alignment

    # ---- forward ----
    def forward(self, gene_idx: torch.Tensor) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        emb    = self.embedding(gene_idx)           # [B, embed_dim]
        h      = self.mlp(emb)                      # [B, bilinear_dim]
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)  # [B,3,G]
        return logits

    # ---- loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),   # [B*G, 3]
            targets.reshape(-1),                        # [B*G]
            weight=self.class_weights,
            label_smoothing=0.1,
        )

    # ---- steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["gene_idx"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["gene_idx"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits, dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, dim=0)    # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  dim=0)    # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   dim=0)    # [N_local]
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        # Gather across all ranks
        all_preds = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)    # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)     # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        # De-duplicate (DDP padding may introduce repeated samples)
        order  = torch.argsort(idx_flat)
        s_idx  = idx_flat[order]
        s_pred = preds_flat[order]
        s_tgt  = tgts_flat[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device),
                            s_idx[1:] != s_idx[:-1]])
        preds_dedup = s_pred[mask]
        tgts_dedup  = s_tgt[mask]

        f1 = compute_per_gene_f1(preds_dedup, tgts_dedup)
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["gene_idx"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)   # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)   # [N_local]
        all_preds   = self.all_gather(local_preds)          # [W, N_local, 3, G]
        all_idx     = self.all_gather(local_idx)            # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)  # [W*N_local, 3, G]
            idx_flat   = all_idx.view(-1)                         # [W*N_local]

            # De-duplicate by sample index: sort, then keep first of each unique index
            order = torch.argsort(idx_flat)
            s_idx = idx_flat[order]
            s_pred = preds_flat[order]
            mask = torch.cat([torch.ones(1, dtype=torch.bool, device=s_idx.device),
                              s_idx[1:] != s_idx[:-1]])
            preds_dedup = s_pred[mask]   # [N_test, 3, G]
            unique_sid  = s_idx[mask].tolist()  # sorted unique sample indices [N_test]

            # Reload test.tsv on rank 0 to get pert_id/symbol mapping by sample index
            test_df = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                           for i in range(len(test_df))}

            rows = []
            for sid in unique_sid:
                pid, sym = idx_to_meta[int(sid)]
                dedup_pos = (s_idx == sid).nonzero(as_tuple=True)[0][0].item()
                pred_list = preds_dedup[dedup_pos].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_idx.clear()

    # ---- checkpoint helpers ----
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
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {train}/{total} params ({100*train/total:.1f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- optimizer ----
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=10, min_lr=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val/f1"}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(description="Node1 – Baseline Bilinear MLP")
    parser.add_argument("--micro-batch-size",  type=int,   default=32)
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument("--max-epochs",        type=int,   default=200)
    parser.add_argument("--lr",                type=float, default=5e-3)
    parser.add_argument("--weight-decay",      type=float, default=1e-2)
    parser.add_argument("--embed-dim",         type=int,   default=512)
    parser.add_argument("--hidden-dim",        type=int,   default=256)
    parser.add_argument("--bilinear-dim",      type=int,   default=128)
    parser.add_argument("--dropout",           type=float, default=0.3)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--debug-max-step",    type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",      action="store_true",
                        dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Limit logic
    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        lim_test  = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        lim_test  = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    # Model
    model = BilinearDEGModel(
        vocab_size   = len(dm.vocab),
        embed_dim    = args.embed_dim,
        hidden_dim   = args.hidden_dim,
        bilinear_dim = args.bilinear_dim,
        dropout      = args.dropout,
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath  = str(output_dir / "checkpoints"),
        filename = "best-{epoch:03d}-{val/f1:.4f}",
        monitor  = "val/f1",
        mode     = "max",
        save_top_k = 1,
    )
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=30, min_delta=1e-4)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy
    strategy = (
        DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator           = "gpu",
        devices               = n_gpus,
        num_nodes             = 1,
        strategy              = strategy,
        precision             = "bf16-mixed",
        max_epochs            = args.max_epochs,
        max_steps             = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches   = lim_train,
        limit_val_batches     = lim_val,
        limit_test_batches    = lim_test,
        val_check_interval    = 1.0 if (args.debug_max_step is None and not fast_dev_run) else 1.0,
        num_sanity_val_steps  = 2,
        callbacks             = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                = [csv_logger, tb_logger],
        log_every_n_steps     = 10,
        deterministic         = True,
        default_root_dir      = str(output_dir),
        fast_dev_run          = fast_dev_run,
        gradient_clip_val     = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    # Save test score
    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
