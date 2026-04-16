"""Node 3: AIDO.Cell-100M (LoRA) — Single-Cell Perturbation Encoding.

Architecture:
  - Input: synthetic single-cell expression vector where the PERTURBED gene is set to
    a low expression value (simulating KD/KO), all other genes set to -1.0 (missing).
    The model internally normalizes, so this encodes "only this gene is (minimally) expressed".
  - Backbone: AIDO.Cell-100M (18 transformer layers, hidden_size=640) with LoRA on Q/K/V
  - Pooling: mean pool over 19264 gene token positions → [B, 640]
  - Head: 3-layer MLP → [B, 6640 * 3] → reshape to [B, 3, 6640]
  - Loss: weighted cross-entropy + label smoothing (0.05)

AIDO.Cell processes entire human transcriptomes (19,264 genes). By setting only the
perturbed gene to a low value and masking all others, we leverage the model's
pre-learned gene-gene co-expression relationships to encode the perturbation context.
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
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = "/home/Models/AIDO.Cell-100M"
N_GENES = 6640
N_CLASSES = 3
AIDO_CELL_N_GENES = 19264    # AIDO.Cell vocabulary size
PERTURBATION_EXPR = 1.0      # Expression value assigned to the knocked-down gene
                              # (low to simulate KD; model applies log1p(x/total*10k))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbCellDataset(Dataset):
    """Each sample: a synthetic single-cell expression where only the perturbed gene
    has a small non-zero expression; all others are -1.0 (missing)."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        gene_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        if "label" in df.columns:
            labels = np.array([json.loads(x) for x in df["label"].tolist()], dtype=np.int64)
            self.labels: Optional[torch.Tensor] = torch.tensor(labels + 1, dtype=torch.long)
        else:
            self.labels = None

        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        symbol = self.symbols[idx]
        # Synthetic single-cell input: only perturbed gene present at PERTURBATION_EXPR
        # AIDO.Cell tokenizer fills missing genes with -1.0 automatically.
        expr_dict = {symbol: PERTURBATION_EXPR}
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": symbol,
            "expr_dict": expr_dict,   # passed to tokenizer in collate_fn
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
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
        micro_batch_size: int = 4,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

        self.tokenizer = None
        self.train_ds: Optional[PerturbCellDataset] = None
        self.val_ds: Optional[PerturbCellDataset] = None
        self.test_ds: Optional[PerturbCellDataset] = None

    def setup(self, stage: str = "fit") -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbCellDataset(train_df, self.tokenizer)
        self.val_ds = PerturbCellDataset(val_df, self.tokenizer)
        self.test_ds = PerturbCellDataset(test_df, self.tokenizer)

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        expr_dicts = [item["expr_dict"] for item in batch]
        # AIDO.Cell tokenizer accepts list of expression dicts
        tokenized = self.tokenizer(expr_dicts, return_tensors="pt")
        # tokenized["input_ids"]: [B, 19264] float32 — raw expression values
        result = {
            "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
            "pert_id": [item["pert_id"] for item in batch],
            "symbol": [item["symbol"] for item in batch],
            "input_ids": tokenized["input_ids"],          # [B, 19264] float32
            "attention_mask": tokenized["attention_mask"], # [B, 19264] int64 (all ones, ignored by model)
        }
        if "label" in batch[0]:
            result["label"] = torch.stack([item["label"] for item in batch])
        return result

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )


# ---------------------------------------------------------------------------
# Prediction Head
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """3-layer MLP: [B, hidden] → [B, 3, N_GENES]."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 1024,
        n_genes: int = N_GENES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_genes * N_CLASSES),
        )
        self.n_genes = n_genes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)   # [B, n_genes * 3]
        return out.view(-1, N_CLASSES, self.n_genes)  # [B, 3, 6640]


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        head_hidden_dim: int = 1024,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = None
        self.head: Optional[PerturbHead] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Load AIDO.Cell-100M
        backbone = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)

        # Apply LoRA on Q/K/V (target path: bert.encoder.layer.*.attention.self.{query,key,value})
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA params to float32 for stability
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        hidden_size = 640  # AIDO.Cell-100M hidden_size
        self.head = PerturbHead(
            in_dim=hidden_size,
            hidden_dim=self.hparams.head_hidden_dim,
        )

        # Class weights
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq)
        class_weights = class_weights / class_weights.mean()
        self.register_buffer("class_weights", class_weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"AIDO.Cell-100M+LoRA | trainable={trainable:,}/{total:,}")

    def _encode(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Run AIDO.Cell and mean-pool over 19264 gene positions → [B, 640]."""
        # input_ids: [B, 19264] float32 — do NOT cast to long
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=False,
        )
        last_hidden = out.last_hidden_state  # [B, 19266, 640] (19264 genes + 2 summary tokens)
        gene_emb = last_hidden[:, :AIDO_CELL_N_GENES, :].float()  # [B, 19264, 640]
        return gene_emb.mean(dim=1)  # [B, 640]

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        emb = self._encode(batch)
        logits = self.head(emb)
        loss = self._compute_loss(logits, batch["label"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        emb = self._encode(batch)
        logits = self.head(emb)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds, dim=0).numpy()
        labels = torch.cat(self._val_labels, dim=0).numpy()
        self._val_preds.clear()
        self._val_labels.clear()
        f1 = _compute_per_gene_f1(preds, labels)
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        emb = self._encode(batch)
        logits = self.head(emb)
        self._test_preds.append(logits.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        # Gather predictions from all ranks
        all_preds = self.all_gather(preds_local)
        if self.trainer.world_size > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        # Gather string metadata via torch.distributed.all_gather_object
        world_size = dist.get_world_size()
        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        # all_gather_object returns a list of lists, one per rank
        gathered_pert_ids_flat: List[List[str]] = [local_pert_ids]
        gathered_symbols_flat: List[List[str]] = [local_symbols]
        if world_size > 1:
            # Create objects list for all_gather_object
            pert_obj = local_pert_ids
            sym_obj = local_symbols
            object_list_pert = [None] * world_size
            object_list_sym = [None] * world_size
            dist.all_gather_object(object_list_pert, pert_obj)
            dist.all_gather_object(object_list_sym, sym_obj)
            gathered_pert_ids_flat = object_list_pert
            gathered_symbols_flat = object_list_sym

        if self.trainer.is_global_zero:
            # Flatten: each element is a list from one rank
            gathered_pert_ids = []
            gathered_symbols = []
            for rank_list in gathered_pert_ids_flat:
                gathered_pert_ids.extend(rank_list)
            for rank_list in gathered_symbols_flat:
                gathered_symbols.extend(rank_list)

            _save_test_predictions(
                pert_ids=gathered_pert_ids,
                symbols=gathered_symbols,
                preds=all_preds.float().cpu().numpy(),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/f1", "interval": "epoch"},
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        result = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    result[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                result[key] = full_sd[key]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving {trainable}/{total} params ({100*trainable/total:.2f}%)")
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import f1_score as sk_f1
    y_hat = preds.argmax(axis=1)
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert len(pert_ids) == len(preds), (
        f"Length mismatch: {len(pert_ids)} pert_ids vs {len(preds)} pred rows. "
        "Ensure all_gather correctly gathered metadata from all ranks."
    )
    rows = []
    for i in range(len(pert_ids)):
        rows.append({
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Node3: AIDO.Cell-100M LoRA for HepG2 DEG")
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--head-hidden-dim", type=int, default=1024)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    model = PerturbModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        head_hidden_dim=args.head_hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = limit_val = limit_test = args.debug_max_step
        max_steps = args.debug_max_step
        val_check_interval = 1.0
        num_sanity_val_steps = 0
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval
        num_sanity_val_steps = 2

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(monitor="val/f1", mode="max", patience=args.early_stop_patience, min_delta=1e-5)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    # Test with current model weights (from end of training).
    # Using ckpt_path="best" causes DDP checkpoint loading issues where rank 1
    # cannot find the same checkpoint path as rank 0, leading to NCCL timeout.
    # Since the model already has trained weights in memory, ckpt_path=None is safe.
    test_results = trainer.test(model, datamodule=datamodule)

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")


if __name__ == "__main__":
    main()
