"""Node 2: AIDO.Protein-16B (LoRA) for HepG2 Perturbation Response Prediction.

Architecture:
  - Input: protein sequence of the perturbed gene (fetched from hg38 GENCODE protein FASTA
    via Ensembl gene ID, using genomic-data-skill local files)
  - Backbone: AIDO.Protein-16B (MoE transformer, 16B params) with LoRA on Q/K/V
  - Pooling: mean pooling over non-padding, non-special tokens → [B, 2304]
  - Head: 2-layer MLP → [B, 6640 * 3] → reshape to [B, 3, 6640]
  - Loss: Focal loss with class weights to handle severe imbalance

Auxiliary data:
  - Protein sequences from /home/data/genome/hg38_gencode_protein.fa
    (genomic-data-skill: search FASTA by ENSG ID in header field 3)
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = "/home/Models/AIDO.Protein-16B"
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
N_GENES = 6640
N_CLASSES = 3
MAX_SEQ_LEN = 512  # AIDO.Protein-16B supports up to 2048; 512 fits in ~35GB for LoRA


# ---------------------------------------------------------------------------
# Protein sequence lookup helpers
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str) -> Dict[str, str]:
    """Parse hg38_gencode_protein.fa to build ENSG→longest protein sequence map.

    FASTA header format (from genomic-data-skill):
      >ENSP00000493376.2|ENST00000641515.2|ENSG00000186092.7|...
    We strip version suffixes and pick the longest sequence per ENSG ID.
    """
    ensg2seq: Dict[str, str] = {}
    current_ensg: Optional[str] = None
    current_seq_parts: List[str] = []

    def _flush():
        if current_ensg and current_seq_parts:
            seq = "".join(current_seq_parts)
            if current_ensg not in ensg2seq or len(seq) > len(ensg2seq[current_ensg]):
                ensg2seq[current_ensg] = seq

    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                _flush()
                current_seq_parts = []
                current_ensg = None
                # Parse ENSG from header — field index 2 (0-based) after splitting by '|'
                fields = line[1:].split("|")
                if len(fields) >= 3:
                    ensg_raw = fields[2]
                    # Strip version suffix (e.g., ENSG00000186092.7 → ENSG00000186092)
                    current_ensg = ensg_raw.split(".")[0]
            else:
                current_seq_parts.append(line)
    _flush()
    return ensg2seq


# Singleton cache so we only parse the FASTA once per process
_ENSG2SEQ_CACHE: Optional[Dict[str, str]] = None

def get_ensg2seq() -> Dict[str, str]:
    global _ENSG2SEQ_CACHE
    if _ENSG2SEQ_CACHE is None:
        _ENSG2SEQ_CACHE = _build_ensg_to_seq(PROTEIN_FASTA)
    return _ENSG2SEQ_CACHE


FALLBACK_SEQ = "M"  # minimal placeholder if no protein sequence found


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbProteinDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        ensg2seq: Dict[str, str],
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        # Resolve protein sequences (strip version from pert_id)
        self.sequences: List[str] = []
        for pid in self.pert_ids:
            ensg = pid.split(".")[0]
            seq = ensg2seq.get(ensg, FALLBACK_SEQ)
            self.sequences.append(seq)

        if "label" in df.columns:
            labels = np.array([json.loads(x) for x in df["label"].tolist()], dtype=np.int64)
            self.labels: Optional[torch.Tensor] = torch.tensor(labels + 1, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "seq": self.sequences[idx],
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
        micro_batch_size: int = 2,
        num_workers: int = 4,
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len

        self.tokenizer = None
        self.ensg2seq: Optional[Dict[str, str]] = None
        self.train_ds: Optional[PerturbProteinDataset] = None
        self.val_ds: Optional[PerturbProteinDataset] = None
        self.test_ds: Optional[PerturbProteinDataset] = None

    def setup(self, stage: str = "fit") -> None:
        # Rank-0 downloads / caches tokenizer; all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

        # Build protein sequence lookup once (all ranks do this independently)
        self.ensg2seq = get_ensg2seq()

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbProteinDataset(train_df, self.ensg2seq)
        self.val_ds = PerturbProteinDataset(val_df, self.ensg2seq)
        self.test_ds = PerturbProteinDataset(test_df, self.ensg2seq)

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # AIDO.Protein-16B tokenizer expects space-separated amino acid tokens.
        # Convert plain sequences (e.g., "MKP") to space-separated format ("M K P")
        # with a trailing space to satisfy the tokenizer's assertion that the string
        # must contain a space. " ".join(list("M")) gives "M" (no space) which fails;
        # adding a trailing space ensures "M " contains at least one space, while
        # split() naturally strips the trailing whitespace during tokenization.
        seqs = [" ".join(list(item["seq"])) + " " for item in batch]
        tokenized = self.tokenizer.make_a_batch(
            seqs,
            max_length=self.max_seq_len,
            padding_to="longest",
            add_sep_token=True,
        )
        result = {
            "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
            "pert_id": [item["pert_id"] for item in batch],
            "symbol": [item["symbol"] for item in batch],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],  # 1D seqlens
            "padding_mask": tokenized["padding_mask"],
            "special_mask": tokenized["special_mask"],
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
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: [N, C], target: [N]
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        # Gather log-prob for true class
        log_p_t = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)  # [N]
        p_t = prob.gather(1, target.unsqueeze(1)).squeeze(1)           # [N]
        focal_weight = (1 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t

        if self.weight is not None:
            w = self.weight.to(device=target.device, dtype=target.dtype)[target]
            loss = loss * w

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Prediction Head
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """MLP head: [B, hidden] → [B, 3, N_GENES]."""

    def __init__(self, in_dim: int, hidden_dim: int = 1024, n_genes: int = N_GENES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_genes * N_CLASSES),
        )
        self.n_genes = n_genes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        out = self.net(x)  # [B, n_genes * 3]
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
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = None
        self.head: Optional[PerturbHead] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Load AIDO.Protein-16B in bfloat16
        backbone = AutoModelForMaskedLM.from_pretrained(
            MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16
        )
        # Apply LoRA on Q/K/V
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable()
        # Disable KV cache during training (required for gradient checkpointing)
        if hasattr(self.backbone, "config"):
            self.backbone.config.use_cache = False

        # Cast trainable (LoRA) params to float32 for stability
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Prediction head (float32 by default)
        hidden_size = 2304  # AIDO.Protein-16B hidden size
        self.head = PerturbHead(
            in_dim=hidden_size,
            hidden_dim=self.hparams.head_hidden_dim,
        )

        # Class weights for focal loss (inversely proportional to frequency)
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq)
        class_weights = class_weights / class_weights.mean()
        self.focal_loss = FocalLoss(gamma=self.hparams.focal_gamma, weight=class_weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"AIDO.Protein-16B+LoRA | trainable={trainable:,}/{total:,} params")

    def _encode(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Run backbone and mean-pool over non-padding, non-special tokens.

        Returns: [B, 2304]
        """
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],  # 1D seqlens
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]  # [B, T, 2304]

        # Mask out padding and special tokens
        # padding_mask: 1=pad, special_mask: 1=special
        exclude_mask = (batch["padding_mask"].bool() | batch["special_mask"].bool())
        # Mean over valid positions
        valid_mask = ~exclude_mask  # [B, T], True=valid
        hidden = hidden.float()
        valid_mask_f = valid_mask.unsqueeze(-1).float()  # [B, T, 1]
        sum_emb = (hidden * valid_mask_f).sum(dim=1)     # [B, 2304]
        count = valid_mask_f.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        return sum_emb / count  # [B, 2304]

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, 3, 6640], labels: [B, 6640] in {0,1,2}
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return self.focal_loss(logits_flat, labels_flat)

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
        preds_local = torch.cat(self._val_preds, dim=0)
        labels_local = torch.cat(self._val_labels, dim=0)
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather all predictions and labels across ranks for global F1 computation
        all_preds = self.all_gather(preds_local)
        all_labels = self.all_gather(labels_local)
        if self.trainer.world_size > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
            all_labels = all_labels.view(-1, N_GENES)

        if self.trainer.is_global_zero:
            f1 = _compute_per_gene_f1(all_preds.float().cpu().numpy(), all_labels.cpu().numpy())
            self.log("val/f1", f1, prog_bar=True, sync_dist=False)
        else:
            self.log("val/f1", 0.0, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        emb = self._encode(batch)
        logits = self.head(emb)  # [B, 3, 6640]
        self._test_preds.append(logits.detach().cpu().float())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist
        import pickle

        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        # Gather all predictions across ranks
        all_preds = self.all_gather(preds_local)
        # Reshape to [total_samples, 3, 6640] regardless of world_size
        total_samples = all_preds.shape[0] * all_preds.shape[1]
        all_preds = all_preds.view(total_samples, N_CLASSES, N_GENES)

        # Gather all pert_ids and symbols using pickle + CUDA tensor all_gather
        world_size = self.trainer.world_size
        local_ids = list(self._test_pert_ids)
        local_syms = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        all_pert_ids: List[str] = []
        all_symbols: List[str] = []

        if world_size > 1:
            # Pickle local lists of strings
            local_ids_bytes = pickle.dumps(local_ids)
            local_syms_bytes = pickle.dumps(local_syms)
            local_len = len(local_ids_bytes)

            # Gather byte lengths using all_gather_object
            obj_gather = [0] * world_size
            dist.all_gather_object(obj_gather, local_len)
            all_lens = obj_gather

            max_len = max(all_lens) if all_lens else 0
            if max_len > 0:
                # Pack into CUDA tensor for NCCL all_gather (NCCL requires CUDA tensors)
                ids_np = np.frombuffer(local_ids_bytes, dtype=np.uint8).copy()
                ids_tensor = torch.from_numpy(ids_np).cuda()
                ids_tensor = torch.nn.functional.pad(ids_tensor, (0, max_len - local_len))
                gathered_ids = [torch.zeros(max_len, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
                dist.all_gather(gathered_ids, ids_tensor)
                for r, blen in enumerate(all_lens):
                    if blen > 0:
                        b = gathered_ids[r][:blen].cpu().numpy().tobytes()
                        all_pert_ids.extend(pickle.loads(b))

                # Same for symbols
                syms_np = np.frombuffer(local_syms_bytes, dtype=np.uint8).copy()
                syms_tensor = torch.from_numpy(syms_np).cuda()
                syms_tensor = torch.nn.functional.pad(syms_tensor, (0, max_len - len(local_syms_bytes)))
                gathered_syms = [torch.zeros(max_len, dtype=torch.uint8, device="cuda") for _ in range(world_size)]
                dist.all_gather(gathered_syms, syms_tensor)
                for r, blen in enumerate(all_lens):
                    if blen > 0:
                        b = gathered_syms[r][:blen].cpu().numpy().tobytes()
                        all_symbols.extend(pickle.loads(b))
        else:
            all_pert_ids = local_ids
            all_symbols = local_syms

        # Gather test labels if available (for F1 computation)
        has_labels = bool(self._test_labels)
        if has_labels:
            labels_local = torch.cat(self._test_labels, dim=0)
            all_labels = self.all_gather(labels_local)
            total_labels = all_labels.shape[0] * all_labels.shape[1]
            all_labels = all_labels.view(total_labels, N_GENES)
            self._test_labels.clear()
        else:
            all_labels = None

        if self.trainer.is_global_zero:
            n_preds = all_preds.shape[0]
            n_ids = len(all_pert_ids)
            min_len = min(n_preds, n_ids)
            saved_preds = all_preds[:min_len]
            # If preds have an extra batch dim from single-sample gather, squeeze it
            if saved_preds.shape[0] == 1 and saved_preds.ndim == 3:
                saved_preds = saved_preds.squeeze(0)
            _save_test_predictions(
                pert_ids=all_pert_ids[:min_len],
                symbols=all_symbols[:min_len],
                preds=saved_preds.float().cpu().numpy(),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

            # Compute and log test F1 if labels are available
            if has_labels and all_labels is not None:
                min_labels = min_len
                f1 = _compute_per_gene_f1(
                    saved_preds[:min_labels].float().cpu().numpy(),
                    all_labels[:min_labels].cpu().numpy(),
                )
                self.log("test/f1", f1, prog_bar=True, sync_dist=False)
                self.print(f"Test F1: {f1:.4f}")

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
# Helper functions
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute per-gene macro F1.

    Args:
        preds: shape [n_samples, 3, n_genes] — class logits/probabilities
        labels: shape [n_samples, n_genes] — integer class labels in {0, 1, 2}
    """
    from sklearn.metrics import f1_score as sk_f1
    # preds: [n_samples, 3, n_genes] -> argmax over class dim -> [n_samples, n_genes]
    y_hat = preds.argmax(axis=1)
    n_genes = labels.shape[1]
    n_samples = labels.shape[0]
    f1_vals = []
    for g in range(n_genes):
        # Flatten all samples for this gene
        yt = labels[:, g].flatten().astype(np.int32)
        yh = y_hat[:, g].flatten().astype(np.int32)
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
    rows = []
    n = min(len(pert_ids), len(preds))
    for i in range(n):
        p = preds[i]
        # Ensure shape is [3, 6640] — squeeze any extra batch dims
        if p.ndim == 3:
            p = p.squeeze(0)
        elif p.ndim == 4:
            p = p.squeeze(0).squeeze(0)
        rows.append({
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(p.tolist()),
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Node2: AIDO.Protein-16B LoRA for HepG2 DEG")
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--global-batch-size", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--head-hidden-dim", type=int, default=1024)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--early-stop-patience", type=int, default=15)
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

    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        max_seq_len=MAX_SEQ_LEN,
    )

    model = PerturbModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        head_hidden_dim=args.head_hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
    )

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = limit_val = limit_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1

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
        # Use "last" instead of "best" to avoid NCCL desync: "best" can resolve to
        # different paths on different ranks when f1=0.0000 is re-eval'd on rank1,
        # causing FileNotFoundError → NCCL timeout. "last.ckpt" is always present.
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="last")

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")


if __name__ == "__main__":
    main()
