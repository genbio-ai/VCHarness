#!/usr/bin/env python3
"""
Node 2-1-1-1: scFoundation + Dense Input + Full FFN LoRA (linear1+linear2+out_proj) + Wider Head
=================================================================================================
Key differences from parent node (node2-1-1, scFoundation sparse + out_proj/linear2 LoRA, F1=0.3809):

1. **Dense 19,264-token input** (SENTINEL=1.0 for all genes, 0.0 for knocked-out gene):
   - Parent used sparse 1,649-token context at CONTEXT_EXPR=100.0 → regression from 0.41 to 0.38
   - The sparse context with nearly identical token values provides less positional diversity
   - Dense input restores positional diversity: 19,263 non-zero tokens vs 1,649
   - This mirrors the proven approach from AIDO.Cell nodes (node2-1) that achieved F1=0.41

2. **Expanded LoRA targets: linear1 + linear2 + out_proj** (was only out_proj + linear2):
   - Critical fix: parent froze linear1 (FFN expand 768→3072), the most parameter-rich layer
   - Adding linear1 gives 1.77M trainable params vs ~0.5M in parent (+3.5× capacity)
   - The FFN expand layer is the primary feature transformation pathway in each transformer layer
   - Note: scFoundation uses PyTorch nn.MultiheadAttention with fused in_proj_weight (QKV fused),
     which PEFT cannot directly target. linear1 provides the best substitute for QKV adaptation.

3. **Higher LoRA rank**: r=16, alpha=32 (was r=8, alpha=16):
   - More adaptation capacity given the harder task of learning from dense 19264-token sequences
   - The feedback showed that r=8 with frozen linear1 was insufficient to break the 0.41 ceiling

4. **Wider prediction head**: 512 hidden dim (was 256):
   - Better matches the 768-dim backbone output when concatenated with 256-dim symbol embedding
   - (768+256)→512→3×6640 provides 524K params in the head vs ~250K previously

5. **Wider symbol embedding**: 256-dim (was 128-dim):
   - Orthogonal gene identity signal; increased capacity may help distinguish KO effects
   - Higher LR multiplier (10× instead of 5×) for faster warm-up of randomly-initialized embedding

6. **Reduced focal gamma**: 1.5 (was 2.0):
   - The parent's erratic val_f1 oscillations (0.349→0.381 in 24 epochs) were partly caused by
     aggressive focal weighting; gamma=1.5 reduces this instability while still emphasizing
     minority classes

7. **Longer warmup**: 10 epochs (was 5):
   - More stable gradient accumulation before peak LR, allowing the larger LoRA to settle

8. **Higher backbone LR**: 3e-4 (was 2e-4):
   - Dense input should provide better gradient signal; slight LR increase helps convergence

Architecture summary:
  - scFoundation backbone (bfloat16) with LoRA (r=16, linear1+linear2+out_proj, dropout=0.2)
  - Gradient checkpointing enabled
  - Dense input: SENTINEL=1.0 for all 19,264 genes, 0.0 for knocked-out gene
  - Symbol embedding (1668×256)
  - Head: LayerNorm(1024) → Linear(1024,512) → GELU → Dropout(0.4) → LayerNorm(512) → Linear(512,19920)
  - Focal loss (γ=1.5, class_weights=[5,1,10])
  - AdamW + OneCycleLR (backbone_lr=3e-4, head_lr=3e-3 [10× multiplier])
  - micro_batch=8, global_batch=64, max_epochs=80, early_stopping patience=15
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
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
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SCF_MODEL_DIR = "/home/Models/scFoundation"
N_GENES_SCF = 19_264      # scFoundation vocabulary size
N_GENES_OUT = 6_640       # output genes for DEG prediction
N_CLASSES = 3             # {down-regulated, unchanged, up-regulated} → {0, 1, 2}
SENTINEL = 1.0            # expression value for all non-knocked-out genes (dense input)
SCF_HIDDEN = 768          # scFoundation hidden dimension
SYMBOL_EMB_DIM = 256      # dimension of gene symbol embedding (wider than parent's 128)

# Class weights: inverse-frequency, capped to avoid extreme loss inflation.
# Train distribution: class -1 (down)=3.4%, class 0 (unchanged)=95.5%, class 1 (up)=1.1%
# Remapped: class 0=down, class 1=unchanged, class 2=up
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class weights.

    Focal loss down-weights easy examples (large pt), focusing gradient on
    hard examples. gamma=1.5 (reduced from parent's 2.0) to reduce training
    oscillations caused by aggressive minority-class emphasis.
    """

    def __init__(self, gamma: float = 1.5, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        # CE loss without reduction for per-sample weighting
        ce = F.cross_entropy(logits, targets, weight=w, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Produces scFoundation-compatible dense expression profiles for perturbation experiments.

    Design rationale:
    - Dense input: All 19,264 genes set to SENTINEL=1.0, knocked-out gene set to 0.0.
    - This produces nnz ≈ 19,263 non-zero tokens per sample.
    - scFoundation internally selects only x>0 tokens (gather_data), producing output
      of shape [B, 19263+2, 768] = [B, 19265, 768].
    - Dense input provides maximum positional diversity — the knocked-out gene position
      is uniquely identified among 19,263 context positions.
    - This approach mirrors the proven strategy from AIDO.Cell nodes (node2-1, F1=0.41)
      which used the same all-1.0-background approach.

    Key difference from parent node (node2-1-1):
    - Parent used sparse context (1,649 panel genes at 100.0) → 1,649 tokens → regression
    - This node uses dense context (all 19,264 genes at 1.0) → 19,263 tokens → better positional signal

    Attributes:
        expr_inputs: [N, 19264] float32 tensor. Pre-built; reused across epochs.
        symbol_indices: [N] int64 tensor of symbol vocab indices.
        labels: [N, 6640] int64 (labels remapped +1: {-1,0,1}→{0,1,2}), or None for test.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        scf_gene_to_pos: Dict[str, int],
        symbol_vocab: Dict[str, int],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.scf_gene_to_pos = scf_gene_to_pos
        self.is_test = is_test

        # Pre-build dense expression inputs: [N, 19264] float32
        # All genes at SENTINEL=1.0, knocked-out gene at 0.0
        self.expr_inputs = self._build_expr_tensors()

        # Symbol indices: look up symbol in vocab, fall back to 0 (UNK)
        self.symbol_indices = torch.tensor(
            [symbol_vocab.get(sym, 0) for sym in self.symbols],
            dtype=torch.long,
        )

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Remap {-1, 0, 1} → {0, 1, 2} per metric contract
            self.labels = np.array(raw_labels, dtype=np.int8) + 1
        else:
            self.labels = None

    def _build_expr_tensors(self) -> torch.Tensor:
        N = len(self.pert_ids)
        # Dense: start with all SENTINEL=1.0
        expr = torch.full((N, N_GENES_SCF), SENTINEL, dtype=torch.float32)

        # Zero-out the knocked-out gene for each sample
        for i, pert_id in enumerate(self.pert_ids):
            base = pert_id.split(".")[0]
            pos = self.scf_gene_to_pos.get(base)
            if pos is not None:
                expr[i, pos] = 0.0
            # If not in scFoundation vocab: all genes remain at SENTINEL
            # (the perturbation signal is lost for OOV genes, but context is still informative)

        return expr

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "input_ids": self.expr_inputs[idx],      # [19264] float32
            "symbol_idx": self.symbol_indices[idx],   # scalar int64
            "pert_id": self.pert_ids[idx],            # str
            "symbol": self.symbols[idx],              # str
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)  # [6640]
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),   # [B, 19264]
        "symbol_idx": torch.stack([b["symbol_idx"] for b in batch]), # [B]
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])   # [B, 6640]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    """Loads TSV splits and builds scFoundation-compatible datasets.

    Key setup steps:
    1. Load gene_ids from scFoundation → build scf_gene_to_pos mapping.
    2. Build symbol_vocab from ALL symbols across all splits (vocabulary must be
       consistent between train and test for embedding lookup).
    3. Construct PerturbationDataset for each split.
    """

    def __init__(
        self,
        data_dir: str,
        micro_batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

        # Populated in setup()
        self.scf_gene_to_pos: Dict[str, int] = {}
        self.symbol_vocab: Dict[str, int] = {}
        self.n_symbols: int = 0

        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.scf_gene_to_pos:
            self._build_vocab()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self.scf_gene_to_pos, self.symbol_vocab, is_test=False,
            )
            self.val_ds = PerturbationDataset(
                val_df, self.scf_gene_to_pos, self.symbol_vocab, is_test=False,
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self.scf_gene_to_pos, self.symbol_vocab, is_test=True,
            )
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    def _build_vocab(self) -> None:
        """Build scFoundation gene→position mapping and symbol vocab."""
        # 1. Load scFoundation gene IDs and build ENSG→position mapping
        gene_ids_path = Path(SCF_MODEL_DIR) / "gene_ids.json"
        gene_ids: List[str] = json.load(open(gene_ids_path))
        assert len(gene_ids) == N_GENES_SCF, f"Expected {N_GENES_SCF} gene IDs, got {len(gene_ids)}"
        self.scf_gene_to_pos = {gid: i for i, gid in enumerate(gene_ids)}

        # 2. Collect all symbols from all three splits
        all_symbols: List[str] = []
        for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
            path = self.data_dir / split_file
            if path.exists():
                df = pd.read_csv(path, sep="\t")
                all_symbols.extend(df["symbol"].tolist())

        # 3. Build symbol vocabulary: <UNK>=0, then all symbols sorted alphabetically
        unique_symbols = sorted(set(all_symbols))
        self.symbol_vocab = {"<UNK>": 0}
        self.symbol_vocab.update({sym: i + 1 for i, sym in enumerate(unique_symbols)})
        self.n_symbols = len(self.symbol_vocab)
        print(f"[DataModule] symbol_vocab size: {self.n_symbols}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self) -> DataLoader:
        # Return a DataLoader with explicit SequentialSampler.
        # Lightning's DDP wraps dataloaders with DistributedSampler (which interleaves data
        # across ranks), but for test prediction we need ALL samples on ALL ranks so that
        # all_gather produces the full 167 predictions. By using a SequentialSampler here,
        # we prevent Lightning from applying its DDP sampler wrapper.
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=(self.num_workers > 0),
            sampler=torch.utils.data.SequentialSampler(self.test_ds),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class ScFoundationDEGModel(nn.Module):
    """scFoundation backbone with LoRA (linear1+linear2+out_proj) + symbol embedding + DEG head.

    Key architectural changes from parent node (node2-1-1):
    1. Dense 19264-token input → dynamic output [B, 19265, 768] (nnz=19263 + 2 summary)
    2. LoRA on linear1+linear2+out_proj (was only out_proj+linear2) → 3.5× more params
    3. Wider symbol embedding: 256-dim (was 128-dim)
    4. Wider head: (768+256)→512→3×6640 (was (768+128)→256→3×6640)

    Forward pass:
      1. scFoundation processes input_ids [B, 19264]; all tokens non-zero → output [B, 19265, 768]
      2. Mean-pool over full sequence → cell_emb [B, 768]
      3. Look up symbol_idx in symbol_emb → sym_emb [B, 256]
      4. Concat → combined [B, 1024] → head → logits [B, 3, 6640]

    Note on scFoundation's dynamic output:
      With dense input (all-1.0 except one 0.0 gene), nnz=19263, so output = [B, 19265, 768].
      Mean pooling over all 19265 positions is robust and standard for cell embeddings.
    """

    def __init__(
        self,
        n_symbols: int,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.2,
        head_dropout: float = 0.4,
        head_hidden: int = 512,
        symbol_emb_dim: int = SYMBOL_EMB_DIM,
    ):
        super().__init__()

        # ── scFoundation backbone ──
        backbone = AutoModel.from_pretrained(
            SCF_MODEL_DIR,
            trust_remote_code=True,
            _use_flash_attention_2=True,
        )
        backbone.config.use_cache = False

        # ── LoRA on linear1 (FFN expand 768→3072) + linear2 (FFN contract 3072→768)
        #    + out_proj (attention output projection 768→768) ──
        # This targets ALL feed-forward transformations in each layer, providing
        # comprehensive adaptation of how information flows through the transformer.
        #
        # Note: scFoundation uses PyTorch's nn.MultiheadAttention with fused in_proj_weight
        # (QKV concatenated into a single 2304×768 matrix). PEFT cannot directly target this
        # fused weight. linear1 (768→3072 FFN expand) is the highest-capacity alternative:
        # it transforms the residual stream into the FFN's high-dimensional space and is
        # the primary feature learning pathway within each transformer layer.
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["linear1", "linear2", "out_proj"],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        # Move to bfloat16 and CUDA AFTER peft wrapping to ensure LoRA params
        # (which are created in float32 on CPU) are also placed on the correct device/dtype.
        self.backbone = self.backbone.to(torch.bfloat16)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # ── Gene symbol embedding: orthogonal identity signal ──
        # 256-dim (wider than parent's 128) for better gene-level feature discrimination
        self.symbol_emb = nn.Embedding(n_symbols, symbol_emb_dim)
        nn.init.trunc_normal_(self.symbol_emb.weight, std=0.02)

        # ── Prediction head: (768+256) → 512 → 3×6640 ──
        # Wider hidden dim (512 vs 256) to match the 768-dim backbone + 256-dim symbol input
        head_in = SCF_HIDDEN + symbol_emb_dim  # 1024
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        # Conservative initialisation: small weights prevent early training instability
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        input_ids: torch.Tensor,    # [B, 19264] float32 (dense, all 1.0 except one 0.0)
        symbol_idx: torch.Tensor,   # [B] int64 (symbol vocab indices)
    ) -> torch.Tensor:              # [B, N_CLASSES, N_GENES_OUT]
        # scFoundation forward: internally selects non-zero tokens
        # Dense input → output [B, 19265, 768] (19263 non-zero + 2 summary tokens)
        out = self.backbone(input_ids=input_ids)
        lhs = out.last_hidden_state  # [B, T, 768]  T ≈ 19265 for dense input

        # Mean pool over all output positions (robust to dense/sparse input)
        # Cast to float32 for numerical stability before aggregation
        cell_emb = lhs.float().mean(dim=1)   # [B, 768]

        # Symbol embedding lookup — always float32 (randomly initialized)
        sym_emb = self.symbol_emb(symbol_idx).float()  # [B, 256]

        # Concatenate backbone embedding and symbol embedding
        combined = torch.cat([cell_emb, sym_emb], dim=-1)  # [B, 1024]

        logits = self.head(combined)          # [B, 3 * 6640]
        B = input_ids.shape[0]
        return logits.view(B, N_CLASSES, N_GENES_OUT)   # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro F1, averaged over all 6,640 genes.

    Per-gene macro F1: for each gene, compute F1 for each class that actually
    appears in y_true, then average over present classes. Then average the
    per-gene F1 values across all genes.

    Args:
        y_pred: [n_samples, 3, n_genes] float — class logits or probabilities
        y_true_remapped: [n_samples, n_genes] int — labels in {0, 1, 2}

    Returns:
        Scalar float — macro-averaged per-gene F1 score
    """
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp_class = y_pred[:, :, g]           # [n_samples, 3]
        yhat = yp_class.argmax(axis=1)
        present = np.array([(yt == c).any() for c in range(N_CLASSES)])
        pf1 = sk_f1_score(yt, yhat, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    """Lightning wrapper for scFoundation DEG predictor.

    Handles:
    - DDP training with all_gather for consistent val/test metrics
    - OneCycleLR with separate learning rates for backbone LoRA vs. head
    - Test prediction writing (rank 0 only)
    - state_dict / load_state_dict overrides to save only trainable parameters
    """

    def __init__(
        self,
        n_symbols: int,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.2,
        head_dropout: float = 0.4,
        head_hidden: int = 512,
        symbol_emb_dim: int = SYMBOL_EMB_DIM,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        gamma_focal: float = 1.5,
        max_epochs: int = 80,
        steps_per_epoch: int = 23,   # will be overridden in main()
        warmup_epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[ScFoundationDEGModel] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators for validation and test (cleared each epoch)
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = ScFoundationDEGModel(
                n_symbols=self.hparams.n_symbols,
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                head_dropout=self.hparams.head_dropout,
                head_hidden=self.hparams.head_hidden,
                symbol_emb_dim=self.hparams.symbol_emb_dim,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
            )
            # Cast all trainable (LoRA + head + symbol_emb) parameters to float32
            # for stable AdamW optimization. Backbone non-trainable params remain bfloat16.
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        # Retrieve test metadata from datamodule when entering test stage
        if stage == "test" and hasattr(self, "trainer") and self.trainer is not None:
            dm = self.trainer.datamodule
            if dm is not None:
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(batch["input_ids"], batch["symbol_idx"])

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Reshape [B, C, G] logits → [B*G, C] for cross-entropy."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_flat = labels.reshape(-1)
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "train_loss", loss,
            on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "val_loss", loss,
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, 6640]
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        lp = torch.cat(self._val_preds, 0)   # [N_local, 3, 6640]
        ll = torch.cat(self._val_labels, 0)  # [N_local, 6640]
        li = torch.cat(self._val_indices, 0) # [N_local]

        # Gather predictions from all DDP ranks
        ap = self.all_gather(lp)  # [world_size, N_local, 3, 6640] or [N_local, 3, 6640]
        al = self.all_gather(ll)
        ai = self.all_gather(li)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # Flatten gathered tensors (handles both single-GPU and multi-GPU cases)
        preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()  # [N_total, 3, 6640]
        labels = al.cpu().view(-1, N_GENES_OUT).numpy()             # [N_total, 6640]
        idxs = ai.cpu().view(-1).numpy()                            # [N_total]

        # Deduplicate by sample index (DDP can produce duplicate samples)
        _, uniq = np.unique(idxs, return_index=True)
        preds_dedup = preds[uniq]
        labels_dedup = labels[uniq]

        # Compute F1 on rank 0, then broadcast to all ranks
        if self.global_rank == 0:
            f1_scalar = compute_deg_f1(preds_dedup, labels_dedup)
            f1_tensor = torch.tensor(f1_scalar, dtype=torch.float32, device=self.device)
        else:
            f1_tensor = torch.zeros(1, dtype=torch.float32, device=self.device)

        # Broadcast from rank 0 to all ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(f1_tensor, src=0)

        self.log("val_f1", f1_tensor.item(), prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, 6640]
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        lp = torch.cat(self._test_preds, 0)   # [N_local, 3, 6640]
        li = torch.cat(self._test_indices, 0)  # [N_local]

        self._test_preds.clear()
        self._test_indices.clear()

        # In DDP mode, Lightning uses DistributedSampler for test dataloaders, causing
        # each rank to see different subsets (interleaved). We gather all predictions
        # and deduplicate by original dataset index to produce complete results.
        #
        # For safety with potentially mismatched per-rank sample counts (due to
        # DistributedSampler + limit_test_batches), we first gather the local sizes
        # to determine max_n, then pad and all_gather predictions.
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_n = lp.shape[0]

        # Step 1: gather local sizes from all ranks
        size_tensor = torch.tensor([local_n], dtype=torch.long, device=self.device)
        all_sizes = self.all_gather(size_tensor)  # [world_size]
        max_n = int(all_sizes.max())

        # Step 2: pad local tensors to max_n for all_gather compatibility
        lp_padded = torch.zeros(max_n, N_CLASSES, N_GENES_OUT, dtype=torch.float32)
        li_padded = torch.zeros(max_n, dtype=torch.long)
        lp_padded[:local_n] = lp
        li_padded[:local_n] = li

        # Step 3: all_gather with padded tensors (same size on all ranks)
        gathered_preds = self.all_gather(lp_padded)   # [world_size, max_n, 3, 6640]
        gathered_idxs = self.all_gather(li_padded)   # [world_size, max_n]

        # Only rank 0 processes the gathered data
        if self.global_rank == 0:
            gathered_preds = gathered_preds.cpu().numpy()
            gathered_idxs = gathered_idxs.cpu().numpy()

            # Flatten: [world_size * max_n, 3, 6640]
            all_preds = gathered_preds.reshape(-1, N_CLASSES, N_GENES_OUT)
            all_idxs = gathered_idxs.reshape(-1)

            # Deduplicate by original dataset index (handles DistributedSampler duplicates)
            _, uniq = np.unique(all_idxs, return_index=True)
            preds_dedup = all_preds[uniq]
            idxs_dedup = all_idxs[uniq]

            # Sort by original dataset index for consistent ordering
            order = np.argsort(idxs_dedup)
            preds_final = preds_dedup[order]
            idxs_final = idxs_dedup[order]

            expected_n = len(self._test_pert_ids)
            if len(preds_final) != expected_n:
                self.print(
                    f"[WARN] Got {len(preds_final)} predictions, expected {expected_n}"
                )

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for local_i, orig_i in enumerate(idxs_final):
                rows.append({
                    "idx": self._test_pert_ids[int(orig_i)],
                    "input": self._test_symbols[int(orig_i)],
                    "prediction": json.dumps(preds_final[local_i].tolist()),
                })
            out_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path} ({len(rows)} rows)")

    def configure_optimizers(self):
        """Two-group AdamW + OneCycleLR.

        Backbone LoRA parameters get the base learning rate (args.lr).
        Symbol embedding and head get 10× higher LR to compensate for random init.
        OneCycleLR provides warmup + cosine decay with longer warmup (10 epochs) for
        more stable gradient accumulation with the larger LoRA (r=16).
        """
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters()
            if p.requires_grad
        ]
        # Head and symbol embedding both have random init → need higher LR
        head_and_emb_params = (
            list(self.model.head.parameters()) +
            list(self.model.symbol_emb.parameters())
        )

        max_lr_backbone = self.hparams.lr
        max_lr_head = self.hparams.lr * 10  # 10× differential LR (was 5× in parent)

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": max_lr_backbone / 25,
                 "max_lr": max_lr_backbone},
                {"params": head_and_emb_params, "lr": max_lr_head / 25,
                 "max_lr": max_lr_head},
            ],
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_steps = self.hparams.steps_per_epoch * self.hparams.max_epochs
        warmup_pct = self.hparams.warmup_epochs / self.hparams.max_epochs

        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=[max_lr_backbone, max_lr_head],
            total_steps=total_steps,
            pct_start=warmup_pct,
            anneal_strategy="cos",
            div_factor=25,        # initial_lr = max_lr / 25
            final_div_factor=1e4, # final_lr = initial_lr / 1e4
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters to keep checkpoint size small.

        scFoundation has ~100M frozen params; we only need to save LoRA weights
        (~1.77M), symbol_emb (~430K), and head (~10M) — roughly 12M total.
        """
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        out: Dict[str, Any] = {}
        # Save all trainable parameter tensors
        for name, p in self.named_parameters():
            if p.requires_grad:
                k = prefix + name
                if k in full:
                    out[k] = full[k]
        # Always save buffers (e.g., FocalLoss class weights)
        for name, buf in self.named_buffers():
            k = prefix + name
            if k in full:
                out[k] = full[k]
        return out

    def load_state_dict(self, state_dict, strict=True):
        """Non-strict loading: allows missing frozen backbone weights."""
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 2-1-1-1: scFoundation + Dense Input + Full FFN LoRA + Wider Head"
    )
    p.add_argument("--data-dir", type=str, default=None,
                   help="Path to data directory (default: 3 levels up from script)")
    p.add_argument("--micro-batch-size", type=int, default=8,
                   help="Per-GPU batch size (default: 8)")
    p.add_argument("--global-batch-size", type=int, default=64,
                   help="Global effective batch size (micro_batch * n_gpus * accumulate_grad, default: 64)")
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Backbone LoRA max LR for OneCycleLR (default: 3e-4)")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.2)
    p.add_argument("--head-dropout", type=float, default=0.4)
    p.add_argument("--head-hidden", type=int, default=512)
    p.add_argument("--gamma-focal", type=float, default=1.5)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None,
                   help="Limit training steps for debugging")
    p.add_argument("--fast-dev-run", action="store_true",
                   help="Run 1 batch per train/val/test for sanity check")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(0)
    args = parse_args()

    # Resolve data_dir: script lives at mcts/node2-1-1-1/main.py
    # → parent.parent.parent = mcts/ → parent = project_root
    if args.data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    else:
        data_dir = Path(args.data_dir)
    args.data_dir = str(data_dir)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of GPUs and compute gradient accumulation
    n_gpus = int(os.environ.get("WORLD_SIZE", 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit = args.debug_max_step if args.debug_max_step is not None else 1.0

    # Estimate steps_per_epoch for OneCycleLR total_steps calculation
    # 1500 training samples, drop_last=True; each GPU sees micro_batch_size samples/step
    n_train_samples = 1500
    steps_per_gpu_per_epoch = max(1, n_train_samples // (args.micro_batch_size * n_gpus))
    steps_per_epoch = max(1, steps_per_gpu_per_epoch // accumulate_grad)
    if args.debug_max_step is not None:
        steps_per_epoch = args.debug_max_step

    print(
        f"[Config] n_gpus={n_gpus}, micro_batch={args.micro_batch_size}, "
        f"accumulate_grad={accumulate_grad}, effective_batch={args.micro_batch_size * n_gpus * accumulate_grad}, "
        f"steps_per_epoch={steps_per_epoch}, max_epochs={args.max_epochs}"
    )

    # ── Callbacks ──
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node2-1-1-1-{epoch:03d}-{val_f1:.4f}",
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
    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # ── Loggers ──
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # ── Trainer ──
    if n_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=180))
    else:
        strategy = "auto"
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
            1.0 if (args.debug_max_step is not None or args.fast_dev_run)
            else args.val_check_interval
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=False,    # FlashAttention is non-deterministic
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    # ── DataModule ──
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    # Setup early to get n_symbols for model initialization
    datamodule.setup()

    # ── LightningModule ──
    model_module = DEGLightningModule(
        n_symbols=datamodule.n_symbols,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        head_dropout=args.head_dropout,
        head_hidden=args.head_hidden,
        symbol_emb_dim=SYMBOL_EMB_DIM,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        max_epochs=args.max_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )

    # ── Training ──
    trainer.fit(model_module, datamodule=datamodule)

    # ── Testing: use best checkpoint in full run, current weights in debug/dev run ──
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # ── Save test score on rank 0 ──
    is_rank_zero = (
        not torch.distributed.is_available() or
        not torch.distributed.is_initialized() or
        torch.distributed.get_rank() == 0
    )
    if is_rank_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"test_results: {test_results}\n"
            f"val_f1_best: {checkpoint_cb.best_model_score}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
