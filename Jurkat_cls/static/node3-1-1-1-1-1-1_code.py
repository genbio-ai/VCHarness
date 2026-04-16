#!/usr/bin/env python3
"""
Node node3-1-1-1-1-1-1 (child of node3-1-1-1-1-1):
    AIDO.Cell-10M + LoRA (r=8, last 3 layers) + concat+MLP Head (512-dim bottleneck)
==================================================================================
Improves on parent (test F1=0.4069) by addressing two bottlenecks identified
in the parent's feedback report:
  (1) MLP head bottleneck too narrow (256-dim → should be 512-dim like node1-1-1)
  (2) LoRA rank too low (r=4 → r=8) for sufficient backbone adaptation capacity

ROOT CAUSE:
  Parent achieved 0.4069 — well-generalized (test-val gap <0.001) and healthy
  training dynamics. The ceiling is NOT overfitting but representational capacity:
  - 256-dim intermediate layer is half of node1-1-1's proven 512-dim (F1=0.411)
  - LoRA r=4 provides ~18K backbone params; r=8 doubles to ~36K for better adaptation
  Both changes were Priority 1 and 2 recommendations from parent feedback.

KEY CHANGES from parent (node3-1-1-1-1-1, test F1=0.4069):
  1. HEAD MLP BOTTLENECK: 256-dim → 512-dim (PRIMARY FIX)
         Before: Linear(512→256) → GELU → Linear(256→19920)  [~5.1M params]
         After:  Linear(512→512) → GELU → Linear(512→19920)  [~10.2M params]
     - 512-dim exactly matches node1-1-1's proven architecture (test F1=0.411)
     - No information compression in the intermediate layer
     - Evidence: node1-1-1 (AIDO.Cell-100M, MLP 1280→512→19920) achieved best 0.411
     - Expected impact: +0.003-0.005 F1 (from feedback: "Priority 1: +0.003-0.005 F1")

  2. LORA RANK: r=4 → r=8 (SECONDARY FIX)
     - Doubles backbone adaptation capacity from ~18K to ~36K params
     - node2-1 used r=8 on ALL 8 layers and achieved 0.4101; here scoped to last 3
     - Scoping to last 3 layers controls overfitting risk (1,500 training samples)
     - lora_alpha updated from 8 → 16 (keeps alpha/r = 2 ratio for proper LoRA scaling)
     - Expected impact: +0.001-0.003 F1

  3. EXTENDED TRAINING: max_epochs 150 → 200, early_stop_patience 25 → 30
     - Parent best checkpoint at epoch 58, still improving at epoch 83
     - Model was making marginal improvements at epoch 83 when stopped
     - Extended epochs give the wider head more time to converge

  4. HEAD LR: 3e-4 → 4e-4 (MINOR TUNING)
     - 512-dim head has ~10.2M params (2× parent's 5.1M)
     - Slightly higher LR (4e-4) helps explore the larger parameter space faster
     - Still 13× lower than typical dense layer LRs for stable convergence
     - LR patience kept at 8 for responsive scheduling

Unchanged from parent:
  - AIDO.Cell-10M backbone (256-dim, 8 layers)
  - LoRA targets layers 5,6,7 (last 3) — Q/K/V projections
  - Dual pooling: concat([pert_hidden, global_mean]) → [B, 512]
  - Focal loss (γ=2, class weights [5.0, 1.0, 10.0], label_smoothing=0.1)
  - Backbone LR: 3e-5
  - ReduceLROnPlateau (mode=max, factor=0.5)
  - Weight decay: 0.01
  - Dropout: 0.2 (kept from parent — working well)
  - Gradient checkpointing enabled

Total trainable parameters: ~10.25M
  - LoRA adapters (r=8, Q/K/V, layers 5-7): ~36K (vs parent's ~18K)
  - LayerNorm(512): ~1K
  - Linear(512→512) + bias: ~262K
  - Linear(512→3×6640) + bias: ~9.96M
  - Total: ~10.25M  (vs parent's 5.25M)

Evidence from MCTS tree:
  - node1-1-1 (F1=0.411): MLP(1280→512→19920), 512-dim intermediate, r=16
  - node2-1   (F1=0.4101): r=8 ALL layers → achieved 0.4101
  - parent    (F1=0.4069): MLP(512→256→19920), 256-dim intermediate, r=4
  - This node: MLP(512→512→19920), 512-dim intermediate, r=8 last 3 → expected ~0.41+
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

# Moderate class weights: corrects for severe imbalance (~95% class 0/unchanged)
# [5.0, 1.0, 10.0] for {down, unchanged, up} — proven effective in node2-1 (0.4101)
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Multi-class focal loss with class weights and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
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
        """
        gene_pos_map: Dict[str, int] = {}
        unique_base_ids = list(set(pid.split(".")[0] for pid in all_pert_ids))
        print(f"[DataModule] Building gene position map for {len(unique_base_ids)} unique genes...")

        # Primary: use gene_id_to_index (maps ENSG → position)
        if hasattr(tokenizer, "gene_id_to_index"):
            gid2idx = tokenizer.gene_id_to_index
            for base_pid in unique_base_ids:
                if base_pid in gid2idx:
                    gene_pos_map[base_pid] = gid2idx[base_pid]
            if len(gene_pos_map) > 0:
                print(f"[DataModule] ENSG→pos via gene_id_to_index: "
                      f"{len(gene_pos_map)}/{len(unique_base_ids)} found")
                return gene_pos_map

        # Fallback: gene_to_index lookup unavailable — return empty map
        if hasattr(tokenizer, "gene_to_index"):
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
    AIDO.Cell-10M + LoRA (r=8, last 3 layers) + concat+MLP Head.

    KEY CHANGES vs parent:
      1. Wider MLP bottleneck: 256 → 512 (matches node1-1-1's proven architecture)
         LayerNorm(512) → Dropout(0.2)
         → Linear(512→512) → GELU
         → Dropout(0.2) → Linear(512→3*6640)

      2. LoRA rank: r=4 → r=8 (doubles backbone adaptation capacity)
         lora_alpha: 8 → 16 (maintains alpha/r = 2 ratio)

    Trainable params:
      LoRA adapters (r=8, Q/K/V, layers 5,6,7): ~36K (vs parent's ~18K)
      LayerNorm(512):                             ~1K
      Linear(512→512) + bias:                    ~262K
      Linear(512→3*6640) + bias:                 ~9.96M
      Total:                                     ~10.25M  (vs parent's ~5.25M)
    """

    def __init__(
        self,
        dropout: float = 0.2,
        mlp_hidden: int = 512,
        lora_r: int = 8,
        lora_alpha: int = 16,
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
        # r=8 doubles backbone adaptation capacity vs parent's r=4
        # lora_alpha=16 maintains the standard alpha/r = 2 ratio
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

        # ── 2-layer WIDER MLP head (512-dim, PRIMARY FIX) ──────────────────
        # Input: concat([pert_hidden, global_mean]) = 2 × HIDDEN_DIM = 512
        in_dim = HIDDEN_DIM * 2  # 512
        out_dim = N_CLASSES * N_GENES_OUT  # 3 × 6640 = 19920

        self.head_norm = nn.LayerNorm(in_dim)
        self.head_dropout1 = nn.Dropout(dropout)
        self.head_proj1 = nn.Linear(in_dim, mlp_hidden, bias=True)    # 512→512
        self.head_act = nn.GELU()
        self.head_dropout2 = nn.Dropout(dropout)
        self.head_proj2 = nn.Linear(mlp_hidden, out_dim, bias=True)   # 512→19920

        # Initialize with truncated normal (smaller std for proj2 given large output)
        nn.init.trunc_normal_(self.head_proj1.weight, std=0.02)
        nn.init.zeros_(self.head_proj1.bias)
        nn.init.trunc_normal_(self.head_proj2.weight, std=0.02)
        nn.init.zeros_(self.head_proj2.bias)

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
        safe_pos = pert_pos.clamp(min=0)  # [B], -1 → 0 temporarily
        pos_idx = safe_pos.view(B, 1, 1).expand(B, 1, HIDDEN_DIM)  # [B, 1, 256]
        pert_hidden = hidden.gather(1, pos_idx).squeeze(1)  # [B, 256]

        # For genes not in vocabulary: fall back to global pool
        unknown_mask = (pert_pos < 0)
        if unknown_mask.any():
            pert_hidden = pert_hidden.clone()
            pert_hidden[unknown_mask] = global_pool[unknown_mask]

        # ── 3. concat + WIDER MLP head (512-dim bottleneck, PRIMARY FIX) ───
        dual_pool = torch.cat([pert_hidden, global_pool], dim=1)  # [B, 512]
        x = self.head_norm(dual_pool)                              # [B, 512]
        x = self.head_dropout1(x)                                  # [B, 512]
        x = self.head_proj1(x)                                     # [B, 512] (512-dim: was 256)
        x = self.head_act(x)                                       # [B, 512]
        x = self.head_dropout2(x)                                  # [B, 512]
        logits = self.head_proj2(x)                                # [B, 19920]

        return logits.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


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
        dropout: float = 0.2,
        mlp_hidden: int = 512,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        backbone_lr: float = 3e-5,
        head_lr: float = 4e-4,
        weight_decay: float = 0.01,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        max_epochs: int = 200,
        lr_reduce_patience: int = 8,
        lr_reduce_factor: float = 0.5,
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
                dropout=self.hparams.dropout,
                mlp_hidden=self.hparams.mlp_hidden,
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

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # Check if distributed is actually initialized (torchrun may set WORLD_SIZE
        # env var without initializing distributed when nproc-per-node=1)
        dist_init = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = int(os.environ.get("WORLD_SIZE", 1)) if dist_init else 1

        if dist_init and world_size > 1:
            # Multi-GPU: pad to same size, gather across ranks, deduplicate
            all_sizes = [0] * world_size
            torch.distributed.all_gather_object(all_sizes, li.size(0))
            max_n = max(all_sizes) if all_sizes else li.size(0)

            if li.size(0) < max_n:
                pad_li = torch.full((max_n - li.size(0),), -1, dtype=li.dtype, device=li.device)
                li = torch.cat([li, pad_li], dim=0)
                pad_ll = torch.full((max_n - ll.size(0), ll.size(1)), 0, dtype=ll.dtype, device=ll.device)
                ll = torch.cat([ll, pad_ll], dim=0)
                pad_lp = torch.zeros((max_n - lp.size(0), lp.size(1), lp.size(2)), dtype=lp.dtype, device=lp.device)
                lp = torch.cat([lp, pad_lp], dim=0)

            ap = self.all_gather(lp)  # [world, max_n, 3, 6640]
            al = self.all_gather(ll)  # [world, max_n, 6640]
            ai = self.all_gather(li)  # [world, max_n]

            preds_list: List[np.ndarray] = []
            labels_list: List[np.ndarray] = []
            seen: set = set()

            for w in range(world_size):
                for i in range(max_n):
                    global_idx = int(ai[w, i].item())
                    if global_idx < 0 or global_idx in seen:
                        continue
                    seen.add(global_idx)
                    preds_list.append(ap[w, i].cpu().numpy())
                    labels_list.append(al[w, i].cpu().numpy())

            order = np.argsort(list(seen))
            preds_arr = np.stack([preds_list[j] for j in order], axis=0)
            labels_arr = np.stack([labels_list[j] for j in order], axis=0)
        else:
            # Single-GPU (no distributed): use tensors directly
            # Convert to numpy and sort by global index
            preds_arr = lp.cpu().numpy()
            labels_arr = ll.cpu().numpy()
            indices_arr = li.cpu().numpy()
            order = np.argsort(indices_arr)
            preds_arr = preds_arr[order]
            labels_arr = labels_arr[order]

        f1_val = compute_deg_f1(preds_arr, labels_arr.astype(np.int64))
        # In DDP mode, all ranks have the same complete deduplicated data after gather,
        # so all compute the same F1. sync_dist=True handles multi-GPU aggregation.
        self.log("val_f1", f1_val, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        lp = torch.cat(self._test_preds, 0)  # [N, 3, 6640]
        li = torch.cat(self._test_indices, 0)  # [N]
        self._test_preds.clear()
        self._test_indices.clear()

        dist_init = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = int(os.environ.get("WORLD_SIZE", 1)) if dist_init else 1

        if dist_init and world_size > 1:
            # Multi-GPU: pad, gather, deduplicate
            all_sizes = [0] * world_size
            torch.distributed.all_gather_object(all_sizes, li.size(0))
            max_n = max(all_sizes) if all_sizes else li.size(0)

            if li.size(0) < max_n:
                pad_li = torch.full((max_n - li.size(0),), -1, dtype=li.dtype, device=li.device)
                li = torch.cat([li, pad_li], dim=0)
                pad_lp = torch.zeros((max_n - lp.size(0), lp.size(1), lp.size(2)), dtype=lp.dtype, device=lp.device)
                lp = torch.cat([lp, pad_lp], dim=0)

            ap = self.all_gather(lp)  # [world, max_n, 3, 6640]
            ai = self.all_gather(li)  # [world, max_n]

            preds_list: List[np.ndarray] = []
            idxs_list: List[int] = []
            seen: set = set()

            for w in range(world_size):
                for i in range(max_n):
                    global_idx = int(ai[w, i].item())
                    if global_idx < 0 or global_idx in seen:
                        continue
                    seen.add(global_idx)
                    preds_list.append(ap[w, i].cpu().numpy())
                    idxs_list.append(global_idx)

            order = np.argsort(idxs_list)
            preds_arr = np.stack([preds_list[i] for i in order], axis=0)
            idxs_arr = np.array(idxs_list, dtype=np.int64)[order]
        else:
            # Single-GPU: use tensors directly
            preds_arr = lp.cpu().numpy()
            indices_arr = li.cpu().numpy()
            order = np.argsort(indices_arr)
            preds_arr = preds_arr[order]
            idxs_arr = indices_arr[order]

        if self.trainer.is_global_zero:
            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "idx": self._test_pert_ids[i],
                    "input": self._test_symbols[i],
                    "prediction": json.dumps(preds_arr[r].tolist()),
                }
                for r, i in enumerate(idxs_arr)
            ]
            pred_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {pred_path} ({len(rows)} samples)")

    def configure_optimizers(self):
        # Separate learning rates: LoRA backbone (lower) vs. MLP head (higher)
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters() if p.requires_grad
        ]
        head_params = (
            list(self.model.head_norm.parameters()) +
            list(self.model.head_proj1.parameters()) +
            list(self.model.head_proj2.parameters())
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
                {"params": head_params,     "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
            eps=1e-8,
        )

        # ReduceLROnPlateau: monitors val_f1, reduces LR when plateau detected
        # patience=8: responsive to plateaus
        # factor=0.5: halves LR on each plateau detection
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.lr_reduce_factor,
            patience=self.hparams.lr_reduce_patience,
            min_lr=1e-8,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1,
            },
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
    p = argparse.ArgumentParser(
        description="Node node3-1-1-1-1-1-1: AIDO.Cell-10M + LoRA (r=8, last 3 layers) + concat+MLP Head (512-dim)"
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "data"),
    )
    p.add_argument("--micro_batch_size",        type=int,   default=8)
    p.add_argument("--global_batch_size",       type=int,   default=64)
    p.add_argument("--max_epochs",              type=int,   default=200)       # CHANGED: 150 → 200
    p.add_argument("--backbone_lr",             type=float, default=3e-5)
    p.add_argument("--head_lr",                 type=float, default=4e-4)      # CHANGED: 3e-4 → 4e-4
    p.add_argument("--weight_decay",            type=float, default=0.01)
    p.add_argument("--dropout",                 type=float, default=0.2)
    p.add_argument("--mlp_hidden",              type=int,   default=512)       # CHANGED: 256 → 512
    p.add_argument("--lora_r",                  type=int,   default=8)         # CHANGED: 4 → 8
    p.add_argument("--lora_alpha",              type=int,   default=16)        # CHANGED: 8 → 16
    p.add_argument("--lora_dropout",            type=float, default=0.1)
    p.add_argument("--focal_gamma",             type=float, default=2.0)
    p.add_argument("--label_smoothing",         type=float, default=0.1)
    p.add_argument("--lr_reduce_patience",      type=int,   default=8)
    p.add_argument("--lr_reduce_factor",        type=float, default=0.5)
    p.add_argument("--early_stopping_patience", type=int,   default=30)       # CHANGED: 25 → 30
    p.add_argument("--num_workers",             type=int,   default=0)
    p.add_argument("--val_check_interval",      type=float, default=1.0)
    p.add_argument("--debug_max_step",          type=int,   default=None)
    p.add_argument("--fast_dev_run",            action="store_true")
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
        filename="node3-1-1-1-1-1-1-aido10m-mlp512-r8-{epoch:03d}-{val_f1:.4f}",
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
        dropout=args.dropout,
        mlp_hidden=args.mlp_hidden,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        lr_reduce_patience=args.lr_reduce_patience,
        lr_reduce_factor=args.lr_reduce_factor,
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
            f"# Node node3-1-1-1-1-1-1: AIDO.Cell-10M + LoRA (r=8, last 3 layers) + concat+MLP Head (512-dim)\n"
            f"# Model: AIDO.Cell-10M, LoRA r=8 layers 5-7, 2-layer MLP head (512→512→19920)\n"
            f"# Primary metric: f1_score (macro-averaged per-gene F1)\n"
            f"# Key improvements: Wider MLP bottleneck (256→512), doubled LoRA rank (r=4→r=8), "
            f"extended training (150→200 epochs), increased head LR (3e-4→4e-4)\n"
            f"# Expected: ~0.41+ F1 (parent was 0.4069 with 256-dim bottleneck and r=4)\n"
            f"\n"
            f"Best val_f1 (from checkpoint): {primary_val:.6f}\n"
            f"\n"
            f"Test results:\n"
            f"{json.dumps(test_results, indent=2)}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
