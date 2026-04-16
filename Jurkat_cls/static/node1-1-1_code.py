#!/usr/bin/env python3
"""
Node 1-2 Improvement: AIDO.Cell-100M LoRA Fine-tuning with Dual Pooling
========================================================================
Key improvements over parent node (node1-2 — frozen backbone + global mean-pool):
  1. LoRA fine-tuning (r=16, Q/K/V in layers 6-17): backbone adapts to task
  2. Dual pooling: concat [perturbed-gene positional hidden state, global mean-pool]
     → 1280-dim representation that captures both gene-specific and global context
  3. Moderate class weights [5.0, 1.0, 10.0] replacing extreme [28.1, 1.05, 90.9]
  4. Online forward pass (no pre-cached embeddings) since LoRA parameters must update
  5. Reduced max_epochs=70 with early stopping patience=12 (LoRA converges faster)
  6. Two optimizer param groups: LoRA backbone lr=2e-4, head lr=5e-4
  7. Gradient checkpointing for memory-efficient training
  8. Label smoothing=0.05 (lighter regularization than parent's 0.10)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
N_GENES = 6_640
N_CLASSES = 3
BACKBONE_DIM = 640          # AIDO.Cell-100M hidden dim
DUAL_DIM = BACKBONE_DIM * 2  # 1280 = gene_pos_emb + mean_pool

AIDO_MODEL_PATH = "/home/Models/AIDO.Cell-100M"

# Moderate class weights — sqrt-approximate of inverse class frequency
# Train distribution: ~3.56% down (-1), ~95.48% unchanged (0), ~1.10% up (+1)
# Remapped to {0=down, 1=unchanged, 2=up}
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weighting and label smoothing."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [N, C]  float32
        targets: [N]     int64
        """
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # p_t for focal weighting (detached to avoid double-diff)
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none").detach())
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper (mirrors calc_metric.py)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Dataset with pre-tokenized AIDO.Cell inputs and gene-position index."""

    def __init__(
        self,
        df: pd.DataFrame,
        input_ids: torch.Tensor,       # [N, 19264] float32
        attention_mask: torch.Tensor,  # [N, 19264] int64
        pert_positions: torch.Tensor,  # [N] int64; -1 if gene not in vocabulary
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.pert_positions = pert_positions
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            arr = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1} → {0,1,2}
            self.labels = torch.from_numpy(arr).long()  # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "input_ids": self.input_ids[idx],          # [19264] float32
            "attention_mask": self.attention_mask[idx], # [19264] int64
            "pert_pos": self.pert_positions[idx],       # int64 (-1 if unknown)
        }
        if not self.is_test:
            item["label"] = self.labels[idx]  # [6640] int64
        return item


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def _init_tokenizer(self) -> AutoTokenizer:
        """Rank-safe tokenizer initialization (rank 0 first if distributed)."""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)

    def _tokenize_and_get_positions(
        self,
        tokenizer: AutoTokenizer,
        pert_ids: List[str],
        split_name: str = "split",
    ):
        """Tokenize all pert_ids and find each gene's position in the 19264-gene vocabulary.

        Returns:
          input_ids:      [N, 19264] float32  (raw expression values; -1.0 for missing)
          attention_mask: [N, 19264] int64    (all-ones; AIDO.Cell overrides internally)
          pert_positions: [N] int64           (-1 for genes not in vocabulary)
        """
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]

        chunk_size = 128
        all_input_ids: List[torch.Tensor] = []
        for i in range(0, len(expr_dicts), chunk_size):
            chunk = expr_dicts[i:i + chunk_size]
            toks = tokenizer(chunk, return_tensors="pt")
            all_input_ids.append(toks["input_ids"])  # [chunk, 19264] float32

        input_ids = torch.cat(all_input_ids, dim=0)  # [N, 19264] float32
        # AIDO.Cell's attention_mask is always all-ones (overridden internally)
        attention_mask = torch.ones(len(pert_ids), 19264, dtype=torch.long)

        # Find position of the perturbed gene: the single slot with input ≠ -1.0
        non_missing = input_ids > -0.5          # [N, 19264] bool
        has_gene = non_missing.any(dim=1)        # [N] bool
        pert_positions = non_missing.long().argmax(dim=1)  # [N] int64
        pert_positions[~has_gene] = -1          # Unknown genes → fallback to mean-pool

        coverage = 100.0 * has_gene.float().mean().item()
        print(f"  [{split_name}] Gene vocab coverage: "
              f"{has_gene.sum().item()}/{len(pert_ids)} ({coverage:.1f}%)")

        return input_ids, attention_mask, pert_positions

    def setup(self, stage: Optional[str] = None) -> None:
        tokenizer = self._init_tokenizer()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")

            print("Tokenizing train set...")
            tr_ids, tr_mask, tr_pos = self._tokenize_and_get_positions(
                tokenizer, train_df["pert_id"].tolist(), "train")
            print("Tokenizing val set...")
            va_ids, va_mask, va_pos = self._tokenize_and_get_positions(
                tokenizer, val_df["pert_id"].tolist(), "val")

            self.train_ds = PerturbationDataset(train_df, tr_ids, tr_mask, tr_pos, is_test=False)
            self.val_ds = PerturbationDataset(val_df, va_ids, va_mask, va_pos, is_test=False)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            print("Tokenizing test set...")
            te_ids, te_mask, te_pos = self._tokenize_and_get_positions(
                tokenizer, test_df["pert_id"].tolist(), "test")

            self.test_ds = PerturbationDataset(test_df, te_ids, te_mask, te_pos, is_test=True)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

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


# ─────────────────────────────────────────────────────────────────────────────
# Model: AIDO.Cell-100M + LoRA + Dual Pooling + MLP Head
# ─────────────────────────────────────────────────────────────────────────────
class AIDOCellLoRAModel(nn.Module):
    """AIDO.Cell-100M with LoRA adapters and dual-pooling MLP head.

    Dual pooling:
      - gene_emb:  hidden state at the perturbed gene's positional slot  → [B, 640]
      - mean_pool: mean over all 19264 gene positions                    → [B, 640]
      - combined:  cat([gene_emb, mean_pool])                            → [B, 1280]

    This gives the head access to both gene-specific context and the global cell state.
    """

    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_layers: Optional[List[int]] = None,
        hidden_dim: int = 512,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # Default: last 12 of 18 layers (recommended by feedback from node1-2)
        self.lora_layers = lora_layers if lora_layers is not None else list(range(6, 18))
        self.hidden_dim = hidden_dim
        self.head_dropout = head_dropout

        # Initialized separately in LightningModule.setup()
        self.backbone: Optional[nn.Module] = None
        self.head: Optional[nn.Sequential] = None

    def initialize_backbone(self) -> None:
        """Load AIDO.Cell-100M, apply LoRA, enable gradient checkpointing."""
        from peft import LoraConfig, get_peft_model, TaskType

        backbone = AutoModel.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        # AIDO.Cell uses GeneEmbedding (not a standard word embedding), so
        # get_input_embeddings() raises NotImplementedError.  PEFT's
        # get_peft_model() calls enable_input_require_grads() which invokes
        # get_input_embeddings().  We monkey-patch the method so PEFT can
        # proceed.  The gene_embedding has no trainable params here (we are
        # fine-tuning via LoRA on Q/K/V), so returning it is safe.
        _gene_emb = backbone.bert.gene_embedding
        backbone.get_input_embeddings = lambda: _gene_emb

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["query", "key", "value"],
            # layers_to_transform=None → all layers; list → specific layers
            layers_to_transform=self.lora_layers,
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Cast LoRA adapter weights to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Gradient checkpointing reduces activation memory at ~30% speed cost
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def initialize_head(self) -> None:
        """Create MLP head on top of dual-pooled 1280-dim representation."""
        self.head = nn.Sequential(
            nn.LayerNorm(DUAL_DIM),
            nn.Linear(DUAL_DIM, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, N_CLASSES * N_GENES),
        )
        # Truncated-normal initialization for stable early training
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, 19264] float32
        attention_mask: torch.Tensor,   # [B, 19264] int64
        pert_positions: torch.Tensor,   # [B] int64 (-1 for unknown gene)
    ) -> torch.Tensor:
        """
        Returns: [B, 3, N_GENES]  logits
        """
        # ── Backbone forward ─────────────────────────────────────────────────
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, 19266, 640] bfloat16

        # ── Dual pooling ─────────────────────────────────────────────────────
        # Global mean-pool (exclude the 2 appended summary tokens at positions 19264-19265)
        mean_pool = hidden[:, :19264, :].mean(dim=1).float()  # [B, 640]

        # Per-gene positional extraction
        B = hidden.size(0)
        valid = pert_positions >= 0                          # [B] bool
        safe_pos = pert_positions.clamp(min=0)               # [B] — avoid -1 index

        # Extract hidden state at the perturbed gene's position
        gene_emb_raw = hidden[
            torch.arange(B, device=hidden.device), safe_pos, :
        ].float()  # [B, 640]

        # For genes not in vocabulary (pos=-1): fall back to mean_pool
        # Use differentiable masking instead of in-place operations
        valid_f = valid.float().unsqueeze(-1)  # [B, 1]
        gene_emb = gene_emb_raw * valid_f + mean_pool * (1.0 - valid_f)  # [B, 640]

        # Concatenate for dual representation
        combined = torch.cat([gene_emb, mean_pool], dim=-1)  # [B, 1280]

        # ── MLP head ─────────────────────────────────────────────────────────
        logits = self.head(combined)              # [B, 3*6640]
        return logits.view(B, N_CLASSES, N_GENES)  # [B, 3, 6640]

    def get_parameter_groups(
        self, lr_backbone: float, lr_head: float, weight_decay: float
    ) -> List[Dict]:
        """Return separate param groups for backbone LoRA and head."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params,     "lr": lr_head,     "weight_decay": weight_decay},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_layers: Optional[List[int]] = None,
        hidden_dim: int = 512,
        head_dropout: float = 0.3,
        lr_backbone: float = 2e-4,
        lr_head: float = 5e-4,
        weight_decay: float = 1e-2,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 70,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialized in setup()
        self.model: Optional[AIDOCellLoRAModel] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators (cleared each epoch)
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = AIDOCellLoRAModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                lora_layers=self.hparams.lora_layers,
                hidden_dim=self.hparams.hidden_dim,
                head_dropout=self.hparams.head_dropout,
            )
            self.model.initialize_backbone()
            self.model.initialize_head()

            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pert_positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, attention_mask, pert_positions)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G], labels: [B, G] ({0,1,2}) → scalar loss."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        labels_flat = labels.reshape(-1)                       # [B*G]
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_pos"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_pos"])
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
        local_idx = torch.cat(self._val_indices, dim=0)    # [N_local]

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        world_size = self.trainer.world_size if self.trainer.world_size else 1
        if world_size > 1:
            # Gather all predictions, labels, and indices from all ranks
            all_preds = self.all_gather(local_preds)      # [world, N_local, 3, G]
            all_labels = self.all_gather(local_labels)     # [world, N_local, G]
            all_idx = self.all_gather(local_idx)          # [world]

            # All ranks flatten, deduplicate, and compute global F1
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            labels_flat = all_labels.view(-1, N_GENES).cpu().numpy()
            idx_flat = all_idx.view(-1).cpu().numpy()
            # De-duplicate by original index
            unique_pos = np.unique(idx_flat, return_index=True)[1]
            preds_flat = preds_flat[unique_pos]
            labels_flat = labels_flat[unique_pos]
            # Restore original dataset order
            order = np.argsort(idx_flat[unique_pos])
            preds_flat = preds_flat[order]
            labels_flat = labels_flat[order]
            f1 = compute_deg_f1(preds_flat, labels_flat)
            # Log on all ranks (sync_dist=True → Lightning keeps only rank-0 value)
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        else:
            f1 = compute_deg_f1(local_preds.numpy(), local_labels.numpy())
            self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_pos"])
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

            # De-duplicate (DDP may overlap last batch)
            unique_pos = np.unique(idxs, return_index=True)[1]
            preds = preds[unique_pos]
            sorted_idxs = idxs[unique_pos]

            # Restore original dataset order
            order = np.argsort(sorted_idxs)
            preds = preds[order]
            final_idxs = sorted_idxs[order]

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
        param_groups = self.model.get_parameter_groups(
            lr_backbone=self.hparams.lr_backbone,
            lr_head=self.hparams.lr_head,
            weight_decay=self.hparams.weight_decay,
        )
        opt = torch.optim.AdamW(param_groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    # ── Checkpoint: save only trainable parameters ────────────────────────────
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
        total = sum(p.numel() for p in self.parameters())
        tr_cnt = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Checkpoint: saving {tr_cnt:,}/{total:,} params "
            f"({100.0 * tr_cnt / max(total, 1):.2f}% trainable)"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AIDO.Cell-100M LoRA + Dual Pooling DEG predictor"
    )
    p.add_argument("--data-dir",               type=str,   default="data")
    p.add_argument("--micro-batch-size",        type=int,   default=4)
    p.add_argument("--global-batch-size",       type=int,   default=32)
    p.add_argument("--max-epochs",              type=int,   default=70)
    p.add_argument("--lr-backbone",             type=float, default=2e-4)
    p.add_argument("--lr-head",                 type=float, default=5e-4)
    p.add_argument("--weight-decay",            type=float, default=1e-2)
    p.add_argument("--lora-r",                  type=int,   default=16)
    p.add_argument("--lora-alpha",              type=int,   default=32)
    p.add_argument("--lora-dropout",            type=float, default=0.05)
    p.add_argument("--hidden-dim",              type=int,   default=512)
    p.add_argument("--head-dropout",            type=float, default=0.3)
    p.add_argument("--gamma-focal",             type=float, default=2.0)
    p.add_argument("--label-smoothing",         type=float, default=0.05)
    p.add_argument("--early-stopping-patience", type=int,   default=12)
    p.add_argument("--num-workers",             type=int,   default=4)
    p.add_argument("--val-check-interval",      type=float, default=1.0)
    p.add_argument("--debug-max-step",          type=int,   default=None)
    p.add_argument("--fast-dev-run",            action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Distributed setup ────────────────────────────────────────────────────
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit_train = limit_val = limit_test = 1.0
    if args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = args.debug_max_step

    val_check_interval = args.val_check_interval if (
        args.debug_max_step is None and not args.fast_dev_run
    ) else 1.0

    strategy: Any
    if n_gpus == 1:
        strategy = SingleDeviceStrategy(device="cuda:0")
    else:
        strategy = DDPStrategy(
            find_unused_parameters=True,   # LoRA leaves some backbone params unused
            timeout=timedelta(seconds=300),
        )

    # ── Callbacks ────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-imp-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stopping_patience,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # ── Loggers ──────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # ── Trainer ──────────────────────────────────────────────────────────────
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

    # ── Data & model ─────────────────────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=list(range(6, 18)),   # layers 6-17 (last 12 of 18)
        hidden_dim=args.hidden_dim,
        head_dropout=args.head_dropout,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Test ─────────────────────────────────────────────────────────────────
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
