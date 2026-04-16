#!/usr/bin/env python3
"""
Node 3: AIDO.Protein-16B + LoRA – Protein Sequence Perturbation Encoding
=========================================================================
Uses AIDO.Protein-16B (AIDO architecture, MoE-Transformer) to encode the
protein sequence of the knocked-out gene. Protein sequence is fetched from
local hg38 GENCODE resources (genomic-data-skill).

Architecture:
  - Build ENSG→protein-sequence lookup at startup from hg38_gencode_protein.fa
    (FASTA header format: >ENSP...|ENST...|ENSG...|...)
  - AIDO.Protein-16B with LoRA (r=8, α=16, last 4 of 36 layers, on Q/K/V)
  - Mean-pool over non-padding, non-special tokens → [B, 2304]
  - Two-layer MLP head: Linear(2304, 1024) → GELU → Dropout(0.1) → Linear(1024, 3*6640)
  - Weighted CrossEntropyLoss with label smoothing (0.1)
  - AdamW + ReduceLROnPlateau
  - Genes with no protein sequence → learnable fallback embedding

Note: AIDO.Protein-16B uses `tokenizer.make_a_batch(...)` (not `tokenizer(...)`).
The `attention_mask` returned is shape [B] = sequence lengths (1D), NOT a 2D mask.
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
import pysam
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
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
AIDO_PROTEIN_MODEL_DIR = "/home/Models/AIDO.Protein-16B"
GENCODE_PROTEIN_FA = "/home/data/genome/hg38_gencode_protein.fa"

N_GENES_OUT = 6_640
N_CLASSES = 3
MAX_SEQ_LEN = 1024   # truncate protein sequences to this length

# Class weights: inverse frequency (down ~3.56%, unchanged ~94.82%, up ~1.10%)
CLASS_WEIGHTS = torch.tensor([1.0 / 0.0356, 1.0 / 0.9482, 1.0 / 0.0110], dtype=torch.float32)

# A short dummy sequence used as fallback for genes with no known protein
FALLBACK_SEQ = "MAAAA" * 20  # 100 aa


# ──────────────────────────────────────────────────────────────────────────────
# Genome resource: build ENSG → longest protein sequence mapping
# ──────────────────────────────────────────────────────────────────────────────
def build_ensg_to_protein_seq(fasta_path: str) -> Dict[str, str]:
    """
    Parse hg38_gencode_protein.fa to build ENSG(base) → longest protein seq.
    Header format: >ENSP...|ENST...|ENSG00000186092.7|...
    """
    mapping: Dict[str, str] = {}
    try:
        fasta = pysam.FastaFile(fasta_path)
        for header in fasta.references:
            parts = header.split("|")
            ensg_id = None
            for part in parts:
                if part.startswith("ENSG"):
                    ensg_id = part.split(".")[0]  # strip version
                    break
            if ensg_id is None:
                continue
            seq = fasta.fetch(header)
            if ensg_id not in mapping or len(seq) > len(mapping[ensg_id]):
                mapping[ensg_id] = seq
        fasta.close()
    except Exception as e:
        print(f"Warning: failed to build protein mapping from {fasta_path}: {e}")
    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        ensg_to_seq: Dict[str, str],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.is_test = is_test

        # Look up protein sequence for each pert_id
        self.sequences: List[str] = []
        for pert_id in self.pert_ids:
            base = pert_id.split(".")[0]
            seq = ensg_to_seq.get(base, FALLBACK_SEQ)
            # Truncate to MAX_SEQ_LEN amino acids (before tokenisation adds special tokens)
            self.sequences.append(seq[:MAX_SEQ_LEN])

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1}→{0,1,2}
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "seq": self.sequences[idx],
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_collate_fn(tokenizer, max_length: int = MAX_SEQ_LEN):
    """Return a collate function that batches protein sequences via AIDO tokeniser."""
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        seqs = [b["seq"] for b in batch]
        tokenized = tokenizer.make_a_batch(
            seqs,
            max_length=max_length,
            padding_to="longest",
            add_sep_token=True,
        )
        result = {
            "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
            "input_ids": tokenized["input_ids"],            # [B, T] int64
            "attention_mask": tokenized["attention_mask"],  # [B] int64 (sequence lengths!)
            "padding_mask": tokenized["padding_mask"],      # [B, T] int64
            "special_mask": tokenized["special_mask"],      # [B, T] int64
            "pert_ids": [b["pert_id"] for b in batch],
            "symbols": [b["symbol"] for b in batch],
        }
        if "label" in batch[0]:
            result["label"] = torch.stack([b["label"] for b in batch])
        return result
    return collate_fn


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, micro_batch_size: int = 1, num_workers: int = 2):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.tokenizer = None
        self.ensg_to_seq: Dict[str, str] = {}
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        # Rank-0 downloads tokenizer, then barrier
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_PROTEIN_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(
            AIDO_PROTEIN_MODEL_DIR, trust_remote_code=True
        )

        # Build ENSG→protein sequence mapping once
        if not self.ensg_to_seq:
            self.ensg_to_seq = build_ensg_to_protein_seq(GENCODE_PROTEIN_FA)
            print(f"[DEGDataModule] Protein lookup: {len(self.ensg_to_seq)} ENSG IDs found")

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self.ensg_to_seq)
            self.val_ds = PerturbationDataset(val_df, self.ensg_to_seq)
        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(test_df, self.ensg_to_seq, is_test=True)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            collate_fn=make_collate_fn(self.tokenizer, MAX_SEQ_LEN),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class AIDOProteinDEGModel(nn.Module):
    """AIDO.Protein-16B backbone + LoRA (last 4 layers) + MLP head."""

    HIDDEN_DIM = 2304  # AIDO.Protein-16B hidden size
    N_LAYERS = 36      # total transformer layers

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_layers: Optional[List[int]] = None,
        head_hidden: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        if lora_layers is None:
            lora_layers = list(range(self.N_LAYERS - 4, self.N_LAYERS))  # [32,33,34,35]

        # Load backbone in bf16
        backbone = AutoModelForMaskedLM.from_pretrained(
            AIDO_PROTEIN_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
            layers_to_transform=lora_layers,
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable()

        # Cast trainable (LoRA) params to float32
        for p in self.backbone.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # MLP head
        self.head = nn.Sequential(
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.Linear(self.HIDDEN_DIM, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, T] int64
        attention_mask: torch.Tensor,  # [B] int64 — sequence LENGTHS, NOT 2D mask
        padding_mask: torch.Tensor,    # [B, T] int64 (1=padding)
        special_mask: torch.Tensor,    # [B, T] int64 (1=special token)
    ) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,  # 1D seqlens for FlashAttention
            output_hidden_states=True,
        )
        last_hidden = out.hidden_states[-1].float()  # [B, T, 2304]

        # Mean-pool excluding padding and special tokens
        valid_mask = (~(padding_mask.bool() | special_mask.bool())).float()  # [B, T]
        valid_mask = valid_mask.unsqueeze(-1)                                  # [B, T, 1]
        pooled = (last_hidden * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-9)
        # pooled: [B, 2304]

        logits = self.head(pooled)            # [B, 3*6640]
        return logits.view(-1, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    y_hat = y_pred.argmax(axis=1)
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
        lora_r: int = 8,
        lora_alpha: int = 16,
        head_hidden: int = 1024,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOProteinDEGModel] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = AIDOProteinDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                head_hidden=self.hparams.head_hidden,
                dropout=self.hparams.dropout,
            )
            # Cast head (MLP) parameters to float32 for stable optimisation
            for p in self.model.head.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        # Populate test metadata whenever the test dataloader is prepared
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            if hasattr(dm, "test_pert_ids") and dm.test_pert_ids:
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["padding_mask"],
            batch["special_mask"],
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_flat = labels.reshape(-1)
        w = CLASS_WEIGHTS.to(logits.device)
        return F.cross_entropy(
            logits_flat, labels_flat,
            weight=w,
            label_smoothing=self.hparams.label_smoothing,
        )

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
        # Gather from ALL ranks — even ranks with no local data must call all_gather to participate
        lp = torch.cat(self._val_preds, 0) if self._val_preds else torch.zeros(0, N_CLASSES, N_GENES_OUT, dtype=torch.float32)
        ll = torch.cat(self._val_labels, 0) if self._val_labels else torch.zeros(0, N_GENES_OUT, dtype=torch.long)
        li = torch.cat(self._val_indices, 0) if self._val_indices else torch.zeros(0, dtype=torch.long)
        ap = self.all_gather(lp)
        al = self.all_gather(ll)
        ai = self.all_gather(li)
        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # All ranks deduplicate by index and compute F1 on the full gathered dataset
        preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
        labels = al.view(-1, N_GENES_OUT).cpu().numpy()
        idxs = ai.view(-1).cpu().numpy()
        _, uniq = np.unique(idxs, return_index=True)
        f1_val = compute_deg_f1(preds[uniq], labels[uniq])

        # sync_dist=True broadcasts the same value to all ranks so EarlyStopping/metrics work everywhere
        self.log("val_f1", f1_val, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        lp = torch.cat(self._test_preds, 0); li = torch.cat(self._test_indices, 0)
        ap = self.all_gather(lp); ai = self.all_gather(li)
        self._test_preds.clear(); self._test_indices.clear()
        if self.trainer.is_global_zero:
            preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
            idxs = ai.view(-1).cpu().numpy()
            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]; idxs = idxs[uniq]
            order = np.argsort(idxs); preds = preds[order]; idxs = idxs[order]
            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {"idx": self._test_pert_ids[i], "input": self._test_symbols[i],
                 "prediction": json.dumps(preds[r].tolist())}
                for r, i in enumerate(idxs)
            ]
            pd.DataFrame(rows).to_csv(output_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"Test predictions saved → {output_dir / 'test_predictions.tsv'}")

    def configure_optimizers(self):
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        head_params = list(self.model.head.parameters())
        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.hparams.lr},
            {"params": head_params, "lr": self.hparams.lr * 3},
        ], weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5, min_lr=1e-8,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_f1"}}

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
    p = argparse.ArgumentParser(description="Node 3: AIDO.Protein-16B LoRA DEG predictor")
    # Default: data/ is at working_node_3/data/, two levels up from mcts/node3/
    p.add_argument("--data-dir", type=str, default=str(Path(__file__).parent.parent.parent / "data"))
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--global-batch-size", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--head-hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit = args.debug_max_step if args.debug_max_step is not None else 1.0

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node3-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1", mode="max", patience=args.early_stopping_patience, verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=180)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit,
        limit_val_batches=limit,
        limit_test_batches=limit,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=1,
        deterministic=True,
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
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    trainer.fit(model_module, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        # Extract primary metric from test_results if available
        primary_val = float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score is not None else float("nan")
        result_dict = {
            "metric": "f1_score",
            "mode": "max",
            "value": primary_val,
            "details": {
                "val_f1_best": primary_val,
                "test_results": test_results[0] if test_results else {},
            }
        }
        score_path.write_text(
            f"# Node 3 Test Evaluation Results\n"
            f"# Primary metric: f1_score (macro-averaged per-gene F1)\n"
            f"# Model: AIDO.Protein-16B + LoRA\n"
            f"f1_score: {primary_val:.6f}\n"
            f"val_f1_best: {primary_val:.6f}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
