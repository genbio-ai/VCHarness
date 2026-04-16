#!/usr/bin/env python3
"""
Node 3-1-1: AIDO.Cell-10M + LoRA (r=4, last 3 layers) + Additive Interaction Head
====================================================================================
Pivots from the failed STRING GNN approach (node3-1, test F1=0.2434) to the proven
AIDO.Cell LoRA fine-tuning path with a novel additive interaction head.

Key design principles:
  1. AIDO.Cell-10M backbone (256-dim, 8 layers) — lighter than 100M, less overfitting risk
  2. LoRA r=4 on last 3 layers (5,6,7), Q/K/V only — ~18K backbone trainable params
  3. Novel ADDITIVE interaction head (distinct from all existing nodes):
       - Dual pool: [pert_hidden || global_mean] → LayerNorm → Linear(512, 256) → GELU → Dropout
       - Gene query embeddings: Embedding(6640, 256)
       - Additive combination: pert_proj[B,1,256] + gene_q[1,6640,256] → [B, 6640, 256]
       - LayerNorm + Linear(256, 3) → [B, 6640, 3] → [B, 3, 6640]
  4. Strong regularization: dropout=0.3, weight_decay=0.01, lora_dropout=0.1
  5. Focal loss γ=2.0 with frequency-based class weights
  6. AdamW: backbone lr=3e-5, head lr=3e-4, cosine annealing with 5-epoch warmup

Total trainable params: ~1.85M (18K LoRA + 1.83M head) vs 11.6M in node1-1-1
This is ~1233 params per training sample — well-suited for 1500-sample dataset.

Architectural differentiation:
  - node3-1 (parent): STRING GNN — completely different, static output flaw
  - node2 (AIDO.Cell-100M LoRA r=16, all 12 layers + concat+MLP): test F1=0.404
  - node2-1 (AIDO.Cell-10M LoRA r=8, all 8 layers + concat+MLP): test F1=0.4101
  - node1-1-1 (AIDO.Cell-100M LoRA r=16 + dual-pool concat+MLP): test F1=0.411
  - THIS NODE: AIDO.Cell-10M, LoRA r=4 last 3 layers, ADDITIVE head (novel)
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
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "/home/Models/AIDO.Cell-10M"
N_GENES_OUT = 6_640
N_GENES_MODEL = 19_264
N_CLASSES = 3
HIDDEN_DIM = 256  # AIDO.Cell-10M hidden size

# Class weights: inverse frequency (from DATA_ABSTRACT.md val stats)
# down ~3.56%, unchanged ~94.82%, up ~1.63%
CLASS_WEIGHTS = torch.tensor([
    1.0 / 0.0356,   # class 0: down-regulated (-1 → 0)
    1.0 / 0.9482,   # class 1: unchanged      (0  → 1)
    1.0 / 0.0163,   # class 2: up-regulated   (+1 → 2)
], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Multi-class focal loss with class weights and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [N, C], targets: [N]
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Pre-builds synthetic expression vectors for each perturbation sample.

    Input encoding: all 19264 genes at 1.0 (baseline), perturbed gene at 0.0.
    This encodes the knockout signal in the expression space used by AIDO.Cell.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_pos_map: Dict[str, int],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Map {-1,0,1} → {0,1,2} to match calc_metric.py's y_true + 1 convention
            self.labels = np.array(raw_labels, dtype=np.int8) + 1
        else:
            self.labels = None

        # Pre-compute expression vectors for efficiency
        # base: all 19264 genes at 1.0; knock out perturbed gene to 0.0
        base_expr = torch.ones(N_GENES_MODEL, dtype=torch.float32)

        self._exprs: List[torch.Tensor] = []
        self._pert_positions: List[int] = []
        covered = 0
        for pid in self.pert_ids:
            base_pid = pid.split(".")[0]
            pos = gene_pos_map.get(base_pid, -1)
            self._pert_positions.append(pos)

            if pos >= 0:
                expr = base_expr.clone()
                expr[pos] = 0.0  # knockout signal
                covered += 1
            else:
                expr = base_expr.clone()  # no knockout signal (gene not in vocab)
            self._exprs.append(expr)

        if not is_test:
            print(f"[Dataset] {len(self.pert_ids)} samples, "
                  f"{covered}/{len(self.pert_ids)} genes in AIDO.Cell vocab "
                  f"({100.0 * covered / len(self.pert_ids):.1f}%)")

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "expr": self._exprs[idx],                         # [19264] float32
            "pert_pos": self._pert_positions[idx],            # int
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
        "expr": torch.stack([b["expr"] for b in batch]),                  # [B, 19264]
        "pert_pos": torch.tensor([b["pert_pos"] for b in batch], dtype=torch.long),  # [B]
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])        # [B, 6640]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, micro_batch_size: int = 8, num_workers: int = 0):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []
        self._gene_pos_map: Optional[Dict[str, int]] = None

    def _build_gene_pos_map(
        self, tokenizer, all_pert_ids: List[str]
    ) -> Dict[str, int]:
        """
        Build mapping from ENSG gene ID to its position in the 19264-gene vocabulary.

        Uses tokenizer.gene_id_to_index (maps ENSG without version suffix → position)
        as the primary lookup. Falls back to symbol-based lookup via gene_to_index
        if the ENSG ID is not found.
        """
        gene_pos_map: Dict[str, int] = {}
        unique_base_ids = list(set(pid.split(".")[0] for pid in all_pert_ids))
        print(f"[DataModule] Building gene position map for {len(unique_base_ids)} unique genes...")

        # Primary: use gene_id_to_index (maps ENSG → position)
        # e.g. "ENSG00000121410" → 0
        if hasattr(tokenizer, "gene_id_to_index"):
            gid2idx = tokenizer.gene_id_to_index
            for base_pid in unique_base_ids:
                if base_pid in gid2idx:
                    gene_pos_map[base_pid] = gid2idx[base_pid]
            if len(gene_pos_map) > 0:
                print(f"[DataModule] ENSG→pos via gene_id_to_index: "
                      f"{len(gene_pos_map)}/{len(unique_base_ids)} found")
                return gene_pos_map

        # Fallback: use gene_to_index (maps gene symbol → position)
        # Requires symbol column from dataset; lookup by symbol name
        if hasattr(tokenizer, "gene_to_index"):
            g2i = tokenizer.gene_to_index
            print(f"[DataModule] gene_id_to_index had 0 matches; using gene_to_index fallback")
            return gene_pos_map  # Returns empty; lookup must be done by symbol separately

        print(f"[DataModule] No gene position mapping available; all pert_pos will be -1")
        return gene_pos_map

    def setup(self, stage: Optional[str] = None) -> None:
        # Initialize tokenizer: rank-0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

        # Build gene position map once (covers all splits)
        if self._gene_pos_map is None:
            all_ids: List[str] = []
            for fname in ["train.tsv", "val.tsv", "test.tsv"]:
                fpath = self.data_dir / fname
                if fpath.exists():
                    df_tmp = pd.read_csv(fpath, sep="\t")
                    if "pert_id" in df_tmp.columns:
                        all_ids.extend(df_tmp["pert_id"].tolist())
            self._gene_pos_map = self._build_gene_pos_map(tokenizer, all_ids)

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self._gene_pos_map)
            self.val_ds = PerturbationDataset(val_df, self._gene_pos_map)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(test_df, self._gene_pos_map, is_test=True)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    def _loader(self, ds: PerturbationDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class AIDOCellDEGModel(nn.Module):
    """
    AIDO.Cell-10M + LoRA (r=4, last 3 layers) + Additive Interaction Head.

    Novel architecture: pert_proj [B,1,256] + gene_query [1,6640,256] → additive [B,6640,256]
    This is more parameter-efficient and captures pairwise interactions without the
    quadratic cost of cross-attention.

    Trainable params:
      LoRA:       ~18K  (r=4, Q/K/V in layers 5,6,7)
      pert_proj:  ~131K (LayerNorm + Linear(512,256))
      gene_query: ~1.70M (Embedding 6640×256)
      gene_ln:    ~0.5K
      classifier: ~771  (Linear 256×3 + bias)
      Total:      ~1.85M
    """

    def __init__(
        self,
        head_proj_dim: int = 256,
        dropout: float = 0.3,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        # ── Backbone: AIDO.Cell-10M with LoRA ──────────────────────────────
        backbone = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16)
        backbone.config.use_cache = False

        # PEFT's get_peft_model calls enable_input_require_grads() internally,
        # which internally calls get_input_embeddings() — but AIDO.Cell raises
        # NotImplementedError("Not Implemented Yet") for that method.
        # Monkey-patch it to be a no-op so PEFT can initialize.
        def noop_enable_input_require_grads(self):
            pass
        backbone.enable_input_require_grads = noop_enable_input_require_grads.__get__(
            backbone, type(backbone))

        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # LoRA on last 3 layers (layers 5, 6, 7 in 0-indexed 8-layer model)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=[5, 6, 7],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Cast LoRA (trainable) params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Perturbation projection (dual-pool → 256-dim) ──────────────────
        # Input: concat([pert_hidden, global_mean_pool]) = 512-dim
        in_dim = HIDDEN_DIM * 2  # 512
        self.pert_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, head_proj_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        nn.init.trunc_normal_(self.pert_proj[1].weight, std=0.02)

        # ── Gene query embeddings (output-side context, one per gene) ───────
        self.gene_query = nn.Embedding(N_GENES_OUT, head_proj_dim)
        nn.init.normal_(self.gene_query.weight, std=0.02)

        # ── Per-gene classifier ─────────────────────────────────────────────
        self.gene_ln = nn.LayerNorm(head_proj_dim)
        self.classifier = nn.Linear(head_proj_dim, N_CLASSES, bias=True)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, expr: torch.Tensor, pert_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expr:     [B, 19264] float32 — synthetic expression (1.0 baseline, 0.0 at pert)
            pert_pos: [B] long — position of perturbed gene in vocab (-1 if unknown)
        Returns:
            logits: [B, 3, 6640]
        """
        B = expr.shape[0]
        device = expr.device

        # ── 1. AIDO.Cell forward pass ──────────────────────────────────────
        # Lightning's bf16-mixed precision wraps this in autocast automatically
        outputs = self.backbone(
            input_ids=expr,
            attention_mask=torch.ones(B, N_GENES_MODEL, dtype=torch.long, device=device),
        )
        # Cast to float32 for stable head computation
        hidden = outputs.last_hidden_state.float()  # [B, 19266, 256]

        # ── 2. Dual pooling ────────────────────────────────────────────────
        # Global mean pool over gene positions (exclude 2 summary tokens appended by model)
        global_pool = hidden[:, :N_GENES_MODEL, :].mean(dim=1)  # [B, 256]

        # Per-sample perturbed-gene positional embedding
        # Use vectorized gather for efficiency; clamp -1 to 0 for valid indexing
        safe_pos = pert_pos.clamp(min=0)  # [B], -1 → 0 temporarily
        pos_idx = safe_pos.view(B, 1, 1).expand(B, 1, HIDDEN_DIM)  # [B, 1, 256]
        pert_hidden = hidden.gather(1, pos_idx).squeeze(1)  # [B, 256]

        # For genes not in vocabulary: fall back to global pool
        unknown_mask = (pert_pos < 0)
        if unknown_mask.any():
            pert_hidden = pert_hidden.clone()
            pert_hidden[unknown_mask] = global_pool[unknown_mask]

        # ── 3. Perturbation projection ─────────────────────────────────────
        dual_pool = torch.cat([pert_hidden, global_pool], dim=1)  # [B, 512]
        pert_proj = self.pert_proj(dual_pool)                      # [B, 256]

        # ── 4. Gene query embeddings ───────────────────────────────────────
        gene_idx = torch.arange(N_GENES_OUT, device=device)
        gene_q = self.gene_query(gene_idx).float()  # [6640, 256]

        # ── 5. Additive interaction: NOVEL HEAD ────────────────────────────
        # Instead of concat(pert, gene) → MLP (used in all other nodes),
        # we additively combine: pert[B,1,256] + gene[1,6640,256] → [B,6640,256]
        # This captures pairwise interactions with fewer parameters.
        per_gene = pert_proj.unsqueeze(1) + gene_q.unsqueeze(0)  # [B, 6640, 256]

        # ── 6. LayerNorm + classification ──────────────────────────────────
        per_gene = self.gene_ln(per_gene)        # [B, 6640, 256]
        logits = self.classifier(per_gene)        # [B, 6640, 3]

        return logits.permute(0, 2, 1)  # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """
    Per-gene macro-averaged F1 score, matching calc_metric.py logic exactly.

    y_pred: [n_samples, 3, n_genes] — class probabilities
    y_true_remapped: [n_samples, n_genes] — labels in {0,1,2} (i.e., y_true+1)
    """
    n_genes = y_true_remapped.shape[1]
    y_hat = y_pred.argmax(axis=1)  # [n_samples, n_genes]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp = y_hat[:, g]
        present = np.array([(yt == c).any() for c in range(N_CLASSES)])
        pf1 = sk_f1_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        head_proj_dim: int = 256,
        dropout: float = 0.3,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        backbone_lr: float = 3e-5,
        head_lr: float = 3e-4,
        weight_decay: float = 0.01,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 100,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCellDEGModel] = None
        self.loss_fn: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = AIDOCellDEGModel(
                head_proj_dim=self.hparams.head_proj_dim,
                dropout=self.hparams.dropout,
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
            )
            self.loss_fn = FocalLoss(
                gamma=self.hparams.focal_gamma,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        # Populate test metadata for prediction saving
        if stage in ("test", None):
            dm = getattr(self, "trainer", None)
            if dm is not None:
                dm = getattr(self.trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "test_pert_ids") and dm.test_pert_ids:
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(batch["expr"], batch["pert_pos"])  # [B, 3, 6640]

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, 3, 6640], labels: [B, 6640]
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                               # [B*6640]
        return self.loss_fn(logits_flat, labels_flat)

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
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, 6640]
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        lp = torch.cat(self._val_preds, 0)   # [N, 3, 6640]
        ll = torch.cat(self._val_labels, 0)  # [N, 6640]
        li = torch.cat(self._val_indices, 0) # [N]

        # Gather from all ranks and de-duplicate
        ap = self.all_gather(lp)  # [world, N, 3, 6640]
        al = self.all_gather(ll)
        ai = self.all_gather(li)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
        labels = al.view(-1, N_GENES_OUT).cpu().numpy()
        idxs = ai.view(-1).cpu().numpy()
        _, uniq = np.unique(idxs, return_index=True)
        f1_val = compute_deg_f1(preds[uniq], labels[uniq])
        self.log("val_f1", f1_val, prog_bar=True, sync_dist=True)

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

        if self.trainer.is_global_zero:
            preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
            idxs = ai.view(-1).cpu().numpy()
            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]
            idxs = idxs[uniq]
            order = np.argsort(idxs)
            preds = preds[order]
            idxs = idxs[order]

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "idx": self._test_pert_ids[i],
                    "input": self._test_symbols[i],
                    "prediction": json.dumps(preds[r].tolist()),
                }
                for r, i in enumerate(idxs)
            ]
            pred_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {pred_path}")

    def configure_optimizers(self):
        # Separate learning rates: LoRA backbone (lower) vs. custom head (higher)
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters() if p.requires_grad
        ]
        head_params = (
            list(self.model.pert_proj.parameters()) +
            list(self.model.gene_query.parameters()) +
            list(self.model.gene_ln.parameters()) +
            list(self.model.classifier.parameters())
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
                {"params": head_params,     "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
            eps=1e-8,
        )

        # Linear warmup for warmup_epochs, then cosine decay
        warmup_epochs = self.hparams.warmup_epochs
        max_epochs = self.hparams.max_epochs
        cosine_epochs = max(1, max_epochs - warmup_epochs)

        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=1e-7,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ── Checkpoint helpers ─────────────────────────────────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_sd = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable_sd[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                trainable_sd[key] = full_sd[key]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable}/{total} params "
            f"({100.0 * trainable / total:.2f}%), plus {buffers} buffer values"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        """Load partial checkpoint (trainable params + buffers only)."""
        full_keys = set(super().state_dict().keys())
        trainable_keys = {n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys = {n for n, _ in self.named_buffers() if n in full_keys}
        expected_keys = trainable_keys | buffer_keys

        missing = [k for k in expected_keys if k not in state_dict]
        unexpected = [k for k in state_dict if k not in expected_keys]
        if missing:
            self.print(f"Warning: Missing keys in checkpoint (first 5): {missing[:5]}")
        if unexpected:
            self.print(f"Warning: Unexpected keys in checkpoint (first 5): {unexpected[:5]}")
        return super().load_state_dict(state_dict, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Node 3-1-1: AIDO.Cell-10M + Additive Head")
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "data"),
    )
    p.add_argument("--micro-batch-size",       type=int,   default=8)
    p.add_argument("--global-batch-size",      type=int,   default=64)
    p.add_argument("--max-epochs",             type=int,   default=100)
    p.add_argument("--backbone-lr",            type=float, default=3e-5)
    p.add_argument("--head-lr",                type=float, default=3e-4)
    p.add_argument("--weight-decay",           type=float, default=0.01)
    p.add_argument("--head-proj-dim",          type=int,   default=256)
    p.add_argument("--dropout",                type=float, default=0.3)
    p.add_argument("--lora-r",                 type=int,   default=4)
    p.add_argument("--lora-alpha",             type=int,   default=8)
    p.add_argument("--lora-dropout",           type=float, default=0.1)
    p.add_argument("--focal-gamma",            type=float, default=2.0)
    p.add_argument("--label-smoothing",        type=float, default=0.05)
    p.add_argument("--warmup-epochs",          type=int,   default=5)
    p.add_argument("--early-stopping-patience",type=int,   default=20)
    p.add_argument("--num-workers",            type=int,   default=0)
    p.add_argument("--val-check-interval",     type=float, default=1.0)
    p.add_argument("--debug-max-step",         type=int,   default=None)
    p.add_argument("--fast-dev-run",           action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
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
        filename="node3-1-1-{epoch:03d}-{val_f1:.4f}",
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
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    # Strategy: DDP for multi-GPU, SingleDevice for single GPU
    if n_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(seconds=300),
        )
    else:
        strategy = SingleDeviceStrategy(device="cuda:0")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit,
        limit_val_batches=limit,
        limit_test_batches=limit,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
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
        head_proj_dim=args.head_proj_dim,
        dropout=args.dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
    )

    trainer.fit(model_module, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(
            model_module, datamodule=datamodule, ckpt_path="best"
        )

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        primary_val = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else float("nan")
        )
        score_path.write_text(
            f"# Node 3-1-1 Test Evaluation Results\n"
            f"# Model: AIDO.Cell-10M + LoRA (r=4, last 3 layers) + Additive Interaction Head\n"
            f"# Primary metric: f1_score (macro-averaged per-gene F1)\n"
            f"# Trainable params: ~1.85M (LoRA ~18K + head ~1.83M)\n"
            f"val_f1_best: {primary_val:.6f}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
