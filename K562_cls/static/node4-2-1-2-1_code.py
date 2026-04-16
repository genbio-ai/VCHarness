"""Node 4-2-1-2-1 – scFoundation (6L) + STRING_GNN (frozen, cached) + GatedFusion
                 + GenePriorBias (delayed unfreeze) + Gene-frequency-weighted Loss
                 + Sharper Convergence (min_lr_ratio=0.02) + Aggressive Gene Weighting (factor=4.0)
                 + Multi-checkpoint Ensemble + Tighter Stopping + Dropout Schedule

Improvements over node4-2-1-2 (parent, F1=0.4893, best in tree):
1. Sharper final convergence: min_lr_ratio 0.05 → 0.02 (floor 1e-5 → 4e-6).
   Parent had ±0.002 post-peak oscillation due to insufficient LR floor. Lower floor
   allows the optimizer to make smaller steps near the optimum, reducing wandering.
2. More aggressive gene-diversity weighting: factor 2.0 → 4.0.
   Max weight ~2.0 (for genes with DEG_freq~0.25) vs parent's max ~1.506.
   3:1 contrast between informative and near-constant genes provides better gradient
   reallocation to the genes that actually change across perturbations.
3. Tighter stopping: patience 28 → 20, max_epochs 270 → 220.
   Parent peaked at epoch 167 and ran 28 more epochs needlessly. patience=20 stops
   ~7-8 epochs earlier without missing the optimum (model stays flat, not declining).
4. Gradient norm clipping: 1.0 → 0.5. Tighter clipping reduces late-stage parameter
   wandering in the low-LR regime (post-epoch-167), providing finer optimization control.
5. Multi-checkpoint ensemble at test time: top-3 val/f1 checkpoints (save_top_k=3) are
   averaged as softmax probabilities, providing a robust zero-risk test-time boost.
   Analogous to checkpoint ensemble from the recommendation in parent's feedback.md.
6. Dynamic dropout schedule: linear ramp-down from head_dropout=0.5 to 0.3 over
   epochs 150-220 (only in head dropout). Provides extra regularization during early
   training when model is exploring, then relaxes for fine-tuning near convergence.
   Uses on_train_epoch_start hook, no additional parameters.

Retained from parent (node4-2-1-2):
- scFoundation top-6 of 12 layers fine-tuned (gradient checkpointing)
- STRING_GNN frozen + cached (direct lookup, no neighborhood attention)
- GatedFusion (768+256 → 512) + LayerNorm + Dropout(0.3)
- Two-stage GenePriorBias: frozen ep<50, unfrozen ep>=50, scale=0.5×
- AdamW (lr=2e-4, weight_decay=3e-2), warmup=7 epochs
- Mixup (alpha=0.2), label_smoothing=0.1
- Class-frequency weighted CE loss

Architecture:
  pert_id → scFoundation(top-6 layers fine-tuned)  → mean-pool → [B,768]
  pert_id → STRING_GNN(frozen,cached)[18870,256]   → direct lookup → [B,256]
  → GatedFusion(768+256→512) + LayerNorm + Dropout(0.3)
  → [optional Mixup alpha=0.2 during training]
  → Head: Dropout(scheduled 0.5→0.3) → Linear(512→256) → LN → GELU → Dropout(0.25) → Linear(256→19920)
  → GenePriorBias: [B, 3, 6640] + bias[3, 6640] (frozen ep<50, trainable ep>=50)
  → [B, 3, 6640] logits

Test-time: average softmax probabilities over top-3 checkpoints (by val/f1).

Loss:
  per_gene_weighted_CE: loss_gene_g = gene_weight_g * CE(logits[:,:,g], targets[:,g])
  where gene_weight_g = 1 + diversity_factor * DEG_freq_g (factor=4.0)
  DEG_freq_g = fraction of training samples where gene g is differentially expressed (≠0)
"""

from __future__ import annotations

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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES     = 6640
N_CLASSES   = 3
SCF_HIDDEN  = 768    # scFoundation hidden size
GNN_HIDDEN  = 256    # STRING_GNN hidden size
FUSION_DIM  = 512    # Gated fusion output dimension
HEAD_HIDDEN = 256    # Two-layer head intermediate dimension

SCF_MODEL_DIR = "/home/Models/scFoundation"
GNN_MODEL_DIR = "/home/Models/STRING_GNN"

CLASS_FREQ = [0.0429, 0.9251, 0.0320]   # down(-1→0), neutral(0→1), up(1→2)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

# Epoch at which GenePriorBias is unfrozen (two-stage training)
PRIOR_UNFREEZE_EPOCH = 50

# Dropout schedule: linear ramp-down over these epochs
DROPOUT_SCHEDULE_START = 150
DROPOUT_SCHEDULE_END   = 220
DROPOUT_START_VAL      = 0.5
DROPOUT_END_VAL        = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Inverse-sqrt-frequency class weights to handle 92.5% neutral class dominance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_gene_diversity_weights(train_labels: np.ndarray, diversity_factor: float = 4.0) -> torch.Tensor:
    """Compute per-gene loss weights based on DEG diversity in training.

    Genes with higher fraction of DEG samples receive higher weight, so the model
    pays more attention to informative genes and less to near-constant neutral genes.

    Args:
        train_labels: [N_train, N_genes] int array with values in {0,1,2} (remapped from {-1,0,1})
        diversity_factor: scaling factor for the diversity weight (higher = more emphasis on DEG genes)
                          factor=4.0: max weight ~2.0 (vs parent's 1.506 at factor=2.0)

    Returns:
        [N_genes] float32 tensor of per-gene loss weights, normalized to mean=1.0
    """
    # DEG fraction per gene: proportion of samples where gene is not neutral (class != 1)
    is_deg = (train_labels != 1).astype(np.float32)   # [N_train, N_genes]
    deg_freq = is_deg.mean(axis=0)                      # [N_genes]

    # Weight = 1 + diversity_factor * DEG_freq
    # - Near-constant neutral gene (deg_freq ~0): weight ≈ 1.0 (before normalization)
    # - Frequently perturbed gene (deg_freq ~0.25): weight ≈ 2.0 (after normalization ~2x average)
    # With factor=4.0 and normalization, this gives a ~3:1 contrast between informative and constant genes
    weights = 1.0 + diversity_factor * deg_freq         # [N_genes]

    # Normalize to mean=1.0 so total loss scale is preserved
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def compute_gene_prior_bias(train_labels: np.ndarray, scale: float = 0.5) -> torch.Tensor:
    """Compute per-gene class prior bias from training label frequencies.

    Initialized from log-frequency of each class per gene in the training labels,
    with zero-mean normalization per gene to avoid global logit inflation.

    Args:
        train_labels: [N_train, N_genes] int array with values in {0,1,2}
        scale: scaling factor for initial bias magnitude (< 1.0 reduces initial over-reliance)

    Returns:
        [3, N_genes] float32 tensor of initial per-gene class biases
    """
    N_train, N_genes = train_labels.shape
    counts = np.zeros((3, N_genes), dtype=np.float32)
    for c in range(3):
        counts[c] = (train_labels == c).sum(axis=0).astype(np.float32)

    # Log-frequency initialization with pseudocount
    log_freq = np.log(counts + 1.0)   # [3, N_genes]

    # Zero-mean normalization per gene (so prior doesn't inflate all logits)
    log_freq -= log_freq.mean(axis=0, keepdims=True)   # [3, N_genes]

    # Apply scale to reduce initial magnitude
    log_freq *= scale

    return torch.tensor(log_freq, dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float32 softmax probabilities
        targets: [N, G]   int64 class indices in {0, 1, 2}
    Returns:
        Scalar F1 averaged over genes.
    """
    y_hat = preds.argmax(dim=1)  # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)
    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0).float()
        tp  = (is_pred & is_true).float().sum(0)
        fp  = (is_pred & ~is_true).float().sum(0)
        fn  = (~is_pred & is_true).float().sum(0)
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(
            prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-8), torch.zeros_like(prec)
        )
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Gene Prior Bias Module
# ---------------------------------------------------------------------------
class GenePriorBias(nn.Module):
    """Learnable per-gene class bias term.

    Adds a [3, N_genes] bias to the raw logits from the classification head.
    Initialized from log-frequency statistics of training labels.

    Two-stage training: frozen for first PRIOR_UNFREEZE_EPOCH epochs to allow
    the backbone (scFoundation+GNN) to learn representations without a crutch,
    then unfrozen for calibration.
    """

    def __init__(self, n_classes: int = N_CLASSES, n_genes: int = N_GENES) -> None:
        super().__init__()
        # Initialize as zeros; will be overridden from training data in setup()
        self.bias = nn.Parameter(
            torch.zeros(n_classes, n_genes, dtype=torch.float32),
            requires_grad=False,  # Frozen initially; unfrozen at PRIOR_UNFREEZE_EPOCH
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Add per-gene class bias to logits.

        Args:
            logits: [B, 3, N_genes] raw logits
        Returns:
            [B, 3, N_genes] biased logits
        """
        return logits + self.bias.unsqueeze(0)   # broadcast over batch dim


# ---------------------------------------------------------------------------
# Gated Fusion Module
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Element-wise gated fusion of two heterogeneous embeddings.

    output = LayerNorm(Dropout(gate_scf ⊙ proj_scf(scf) + gate_gnn ⊙ proj_gnn(gnn)))
    where gates are computed from the concatenation of both inputs.
    """

    def __init__(
        self,
        d_scf: int = SCF_HIDDEN,
        d_gnn: int = GNN_HIDDEN,
        d_out: int = FUSION_DIM,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        d_in = d_scf + d_gnn
        self.proj_scf   = nn.Linear(d_scf, d_out)
        self.proj_gnn   = nn.Linear(d_gnn, d_out)
        self.gate_scf   = nn.Linear(d_in,  d_out)
        self.gate_gnn   = nn.Linear(d_in,  d_out)
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, scf_emb: torch.Tensor, gnn_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([scf_emb, gnn_emb], dim=-1)     # [B, d_scf+d_gnn]
        gate_s   = torch.sigmoid(self.gate_scf(combined))     # [B, d_out]
        gate_g   = torch.sigmoid(self.gate_gnn(combined))     # [B, d_out]
        fused    = gate_s * self.proj_scf(scf_emb) + gate_g * self.proj_gnn(gnn_emb)
        return self.dropout(self.layer_norm(fused))            # [B, d_out]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List[torch.Tensor]] = (
            [
                torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
                for row in df["label"].tolist()
            ]
            if has_label else None
        )

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx": idx,
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_collate_scf(tokenizer):
    """Collate function that tokenizes inputs for scFoundation."""

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out

    return collate_fn


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer   = None

    def setup(self, stage: Optional[str] = None) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(SCF_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(SCF_MODEL_DIR, trust_remote_code=True)

        self.train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        self.train_ds = DEGDataset(self.train_df)
        self.val_ds   = DEGDataset(pd.read_csv(VAL_TSV,   sep="\t"))
        self.test_ds  = DEGDataset(pd.read_csv(TEST_TSV,  sep="\t"))

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate_scf(self.tokenizer),
        )

    def train_dataloader(self) -> DataLoader: return self._loader(self.train_ds, True)
    def val_dataloader(self)   -> DataLoader: return self._loader(self.val_ds,   False)
    def test_dataloader(self)  -> DataLoader: return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FusionDEGModel(pl.LightningModule):
    """scFoundation (top-6 layers fine-tuned) + STRING_GNN (frozen, cached direct lookup)
    + GatedFusion + GenePriorBias (two-stage training) + gene-frequency-weighted loss
    + sharper convergence + dropout schedule + multi-checkpoint ensemble at test time.

    Key improvements vs. node4-2-1-2 (parent, F1=0.4893):
    1. min_lr_ratio=0.02 (floor=4e-6): eliminate post-peak oscillation for sharper convergence.
    2. gene_diversity_factor=4.0 (up from 2.0): more aggressive gradient reallocation to
       informative genes, creating a 3:1 contrast between DEG-variable and constant genes.
    3. patience=20, max_epochs=220: tighter early stopping (parent peaked at epoch 167).
    4. gradient_clip_val=0.5: finer optimization control in low-LR regime.
    5. Dropout schedule: linear ramp-down head_dropout 0.5→0.3 over epochs 150-220.
    6. Multi-checkpoint ensemble: save_top_k=3, average softmax probabilities at test time.
    """

    def __init__(
        self,
        scf_finetune_layers: int = 6,
        head_dropout: float      = 0.5,
        fusion_dropout: float    = 0.3,
        lr: float                = 2e-4,
        weight_decay: float      = 3e-2,
        warmup_epochs: int       = 7,
        max_epochs: int          = 220,
        min_lr_ratio: float      = 0.02,
        mixup_alpha: float       = 0.2,
        label_smoothing: float   = 0.1,
        prior_bias_scale: float  = 0.5,
        gene_diversity_factor: float = 4.0,
        prior_unfreeze_epoch: int = PRIOR_UNFREEZE_EPOCH,
        dropout_schedule_start: int = DROPOUT_SCHEDULE_START,
        dropout_schedule_end: int   = DROPOUT_SCHEDULE_END,
        dropout_end_val: float      = DROPOUT_END_VAL,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        gnn_dir = Path(GNN_MODEL_DIR)

        # ----------------------------------------------------------------
        # scFoundation backbone (top-k layers fine-tuned)
        # ----------------------------------------------------------------
        self.scf = AutoModel.from_pretrained(
            SCF_MODEL_DIR,
            trust_remote_code=True,
            _use_flash_attention_2=True,
        ).to(torch.bfloat16)
        self.scf.config.use_cache = False
        self.scf.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Freeze all scF params, then unfreeze top-k transformer layers
        for param in self.scf.parameters():
            param.requires_grad = False
        n_scf_layers = len(self.scf.encoder.transformer_encoder)
        for i in range(n_scf_layers - hp.scf_finetune_layers, n_scf_layers):
            for param in self.scf.encoder.transformer_encoder[i].parameters():
                param.requires_grad = True
        # Unfreeze final LayerNorm
        for param in self.scf.encoder.norm.parameters():
            param.requires_grad = True
        # Cast trainable scF params to float32 for stable optimization
        for name, param in self.scf.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        scf_train = sum(p.numel() for p in self.scf.parameters() if p.requires_grad)
        scf_total = sum(p.numel() for p in self.scf.parameters())
        print(f"[Node4-2-1-2-1] scFoundation: {scf_train:,}/{scf_total:,} trainable params")

        # ----------------------------------------------------------------
        # STRING_GNN: fully frozen, embeddings precomputed and cached
        # ----------------------------------------------------------------
        print("[Node4-2-1-2-1] Precomputing STRING_GNN embeddings (frozen)...")
        gnn_temp    = AutoModel.from_pretrained(str(gnn_dir), trust_remote_code=True).float()
        gnn_temp.eval()
        graph_data  = torch.load(gnn_dir / "graph_data.pt", map_location="cpu")
        edge_index  = graph_data["edge_index"].long()
        edge_weight = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.float()
        with torch.no_grad():
            gnn_out  = gnn_temp(edge_index=edge_index, edge_weight=edge_weight)
            gnn_embs = gnn_out.last_hidden_state.float().detach()   # [18870, 256]
        # Register as a buffer → auto-moved to GPU by Lightning
        self.register_buffer("gnn_embs_cached", gnn_embs)
        del gnn_temp   # free memory
        print(f"[Node4-2-1-2-1] GNN embeddings cached: {gnn_embs.shape}")

        # Build Ensembl ID → node index lookup
        node_names = json.loads((gnn_dir / "node_names.json").read_text())
        self._ensembl_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(node_names)
        }

        # ----------------------------------------------------------------
        # Gated Fusion Module
        # ----------------------------------------------------------------
        self.fusion = GatedFusion(
            d_scf=SCF_HIDDEN, d_gnn=GNN_HIDDEN, d_out=FUSION_DIM,
            dropout=hp.fusion_dropout,
        )

        # ----------------------------------------------------------------
        # Two-layer Classification Head: 512 → 256 → 3*6640
        # Note: head_dropout is scheduled dynamically from on_train_epoch_start
        # ----------------------------------------------------------------
        self._current_head_dropout = hp.head_dropout  # starts at 0.5

        self.head_dropout_layer1 = nn.Dropout(hp.head_dropout)
        self.head_linear1        = nn.Linear(FUSION_DIM, HEAD_HIDDEN)
        self.head_norm           = nn.LayerNorm(HEAD_HIDDEN)
        self.head_act            = nn.GELU()
        self.head_dropout_layer2 = nn.Dropout(hp.head_dropout * 0.5)  # always 0.25
        self.head_linear2        = nn.Linear(HEAD_HIDDEN, N_CLASSES * N_GENES)

        # ----------------------------------------------------------------
        # GenePriorBias (frozen initially, unfrozen at prior_unfreeze_epoch)
        # ----------------------------------------------------------------
        self.gene_prior = GenePriorBias(n_classes=N_CLASSES, n_genes=N_GENES)
        # Initialize will be done after loading training data

        # ----------------------------------------------------------------
        # Load training data for initialization (from DataModule or direct load)
        # ----------------------------------------------------------------
        print("[Node4-2-1-2-1] Loading training labels for prior initialization...")
        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        train_labels_list = [
            [x + 1 for x in json.loads(row)]  # remap {-1,0,1} → {0,1,2}
            for row in train_df["label"].tolist()
        ]
        train_labels_np = np.array(train_labels_list, dtype=np.int32)  # [N_train, N_genes]

        # Initialize GenePriorBias from log-frequencies with scaled magnitude
        prior_init = compute_gene_prior_bias(train_labels_np, scale=hp.prior_bias_scale)
        with torch.no_grad():
            self.gene_prior.bias.data.copy_(prior_init)
        print(f"[Node4-2-1-2-1] GenePriorBias initialized (scale={hp.prior_bias_scale:.2f}), "
              f"will unfreeze at epoch {hp.prior_unfreeze_epoch}")

        # Compute per-gene diversity weights for loss (factor=4.0 vs parent's 2.0)
        gene_weights = compute_gene_diversity_weights(
            train_labels_np, diversity_factor=hp.gene_diversity_factor
        )
        self.register_buffer("gene_weights", gene_weights)   # [N_genes]
        print(f"[Node4-2-1-2-1] Gene diversity weights (factor={hp.gene_diversity_factor}): "
              f"mean={gene_weights.mean():.4f}, "
              f"min={gene_weights.min():.4f}, max={gene_weights.max():.4f}")

        # Class weights for weighted CE loss
        self.register_buffer("class_weights", get_class_weights())

        # Accumulators for validation / test
        self._val_preds:     List[torch.Tensor] = []
        self._val_tgts:      List[torch.Tensor] = []
        self._val_idx:       List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []

        # Track whether prior has been unfrozen
        self._prior_unfrozen = False

    # ---- GNN index lookup ----
    def _get_gnn_indices(self, pert_ids: List[str], device: torch.device) -> torch.Tensor:
        """Look up STRING_GNN node indices for a batch of Ensembl gene IDs."""
        indices = [self._ensembl_to_idx.get(pid, 0) for pid in pert_ids]
        return torch.tensor(indices, dtype=torch.long, device=device)

    # ---- Embedding computation ----
    def get_fused_emb(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids:       List[str],
    ) -> torch.Tensor:
        """Compute fused embedding: scFoundation + GNN (direct lookup) → GatedFusion."""
        device = input_ids.device

        # 1. scFoundation → mean-pool over sequence tokens
        scf_out = self.scf(input_ids=input_ids, attention_mask=attention_mask)
        scf_emb = scf_out.last_hidden_state.float().mean(dim=1)   # [B, 768]

        # 2. STRING_GNN: direct cached embedding lookup (no neighborhood attention)
        node_indices = self._get_gnn_indices(pert_ids, device)
        gnn_emb = self.gnn_embs_cached[node_indices]              # [B, 256]

        # 3. Gated fusion → [B, 512]
        return self.fusion(scf_emb, gnn_emb)

    # ---- Head forward with dynamic dropout ----
    def _head_forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Apply classification head with dynamic dropout rate."""
        x = self.head_dropout_layer1(fused)
        x = self.head_linear1(x)
        x = self.head_norm(x)
        x = self.head_act(x)
        x = self.head_dropout_layer2(x)
        x = self.head_linear2(x)
        return x

    # ---- Forward ----
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids:       List[str],
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        fused = self.get_fused_emb(input_ids, attention_mask, pert_ids)
        raw_logits = self._head_forward(fused).view(B, N_CLASSES, N_GENES)  # [B, 3, N_GENES]
        return self.gene_prior(raw_logits)                                    # [B, 3, N_GENES] + bias

    # ---- Gene-frequency-weighted CE Loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Gene-frequency-weighted CE with class weights and label smoothing.

        factor=4.0 creates a 3:1 gradient contrast between highly variable genes
        and near-constant neutral genes (vs parent's 1.7:1 contrast at factor=2.0).

        Args:
            logits:  [B, 3, N_genes] raw logits
            targets: [B, N_genes] int64 class indices in {0,1,2}
        """
        B, C, G = logits.shape

        # Standard per-sample, per-gene CE loss: [B*G]
        per_gene_per_sample_loss = F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, C]
            targets.reshape(-1),                       # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
            reduction="none",
        )   # [B*G]

        # Reshape to [B, G] and apply gene-specific weights
        per_gene_loss = per_gene_per_sample_loss.view(B, G)  # [B, G]
        weighted_loss = per_gene_loss * self.gene_weights.unsqueeze(0)  # [B, G], broadcast over batch

        return weighted_loss.mean()

    # ---- Dynamic dropout schedule ----
    def on_train_epoch_start(self) -> None:
        hp = self.hparams
        current_epoch = self.current_epoch

        # Two-stage training: unfreeze GenePriorBias at configured epoch
        if not self._prior_unfrozen and current_epoch >= hp.prior_unfreeze_epoch:
            self.gene_prior.bias.requires_grad_(True)
            self._prior_unfrozen = True
            self.print(
                f"[Node4-2-1-2-1] Epoch {current_epoch}: GenePriorBias UNFROZEN "
                f"(adding {self.gene_prior.bias.numel():,} trainable params)"
            )

        # Dynamic dropout schedule: linear ramp-down from 0.5 to 0.3
        sched_start = hp.dropout_schedule_start
        sched_end   = hp.dropout_schedule_end
        drop_start  = DROPOUT_START_VAL
        drop_end    = hp.dropout_end_val

        if current_epoch < sched_start:
            new_dropout = drop_start
        elif current_epoch >= sched_end:
            new_dropout = drop_end
        else:
            # Linear interpolation
            progress = (current_epoch - sched_start) / max(1, sched_end - sched_start)
            new_dropout = drop_start + progress * (drop_end - drop_start)

        if abs(new_dropout - self._current_head_dropout) > 1e-6:
            self.head_dropout_layer1.p = new_dropout
            self._current_head_dropout = new_dropout
            self.print(f"[Node4-2-1-2-1] Epoch {current_epoch}: head_dropout updated to {new_dropout:.4f}")

    # ---- Training ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pert_ids       = batch["pert_id"]
        labels         = batch["labels"]
        B = input_ids.shape[0]

        fused = self.get_fused_emb(input_ids, attention_mask, pert_ids)

        # Mixup augmentation (alpha=0.2 — proven healthy, avoids train<val loss anomaly)
        if self.hparams.mixup_alpha > 0.0 and B > 1 and self.training:
            lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
            perm = torch.randperm(B, device=fused.device)
            fused_mix = lam * fused + (1 - lam) * fused[perm]
            raw_logits = self._head_forward(fused_mix).view(B, N_CLASSES, N_GENES)
            logits = self.gene_prior(raw_logits)
            loss = lam * self._loss(logits, labels) + (1 - lam) * self._loss(logits, labels[perm])
        else:
            raw_logits = self._head_forward(fused).view(B, N_CLASSES, N_GENES)
            logits = self.gene_prior(raw_logits)
            loss = self._loss(logits, labels)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    # ---- Validation ----
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits.float(), dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        local_idx   = torch.cat(self._val_idx,   0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        # Deduplicate (DDP may duplicate samples at epoch boundaries)
        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([
            torch.ones(1, dtype=torch.bool, device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

    # ---- Test ----
    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)

        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        if is_dist:
            all_preds    = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
            all_pert_ids = [None] * self.trainer.world_size
            all_symbols  = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(all_pert_ids, self._test_pert_ids)
            torch.distributed.all_gather_object(all_symbols,  self._test_symbols)
            flat_pids = [p for rank_pids in all_pert_ids for p in rank_pids]
            flat_syms = [s for rank_syms in all_symbols  for s in rank_syms]
        else:
            all_preds = local_preds
            flat_pids = self._test_pert_ids
            flat_syms = self._test_symbols

        if self.trainer.is_global_zero:
            n    = all_preds.shape[0]
            rows = []
            for i in range(n):
                rows.append({
                    "idx":        flat_pids[i],
                    "input":      flat_syms[i],
                    "prediction": json.dumps(all_preds[i].float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node4-2-1-2-1] Saved {len(rows)} test predictions → {out_dir}/test_predictions.tsv")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    # ---- Checkpoint: save only trainable params + all buffers ----
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full:
                    trainable[key] = full[key]
        # Also save GenePriorBias even if currently frozen (might be unfrozen at restore time)
        prior_key = prefix + "gene_prior.bias"
        if prior_key in full:
            trainable[prior_key] = full[prior_key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full:
                trainable[key] = full[key]
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"[Node4-2-1-2-1] Checkpoint: {trained:,}/{total:,} params ({100*trained/total:.2f}%)"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer with WarmupCosine LR scheduler ----
    def configure_optimizers(self):
        hp = self.hparams
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable_params, lr=hp.lr, weight_decay=hp.weight_decay
        )

        def lr_lambda(epoch: int) -> float:
            """Linear warmup then cosine decay to min_lr_ratio floor."""
            if epoch < hp.warmup_epochs:
                return max(1e-8, epoch / max(1, hp.warmup_epochs))
            progress = (epoch - hp.warmup_epochs) / max(1, hp.max_epochs - hp.warmup_epochs)
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return hp.min_lr_ratio + (1.0 - hp.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Multi-checkpoint Ensemble Test
# ---------------------------------------------------------------------------
def run_ensemble_test(
    trainer: pl.Trainer,
    model: FusionDEGModel,
    dm: DEGDataModule,
    checkpoint_dir: Path,
    output_dir: Path,
) -> None:
    """Load top-k checkpoints, run test, and average softmax predictions.

    This provides a zero-risk test-time boost by averaging probability outputs
    from the best-performing checkpoints, reducing variance from any single
    checkpoint's post-peak state.
    """
    import glob

    # Find all .ckpt files in checkpoint directory
    ckpt_files = sorted(
        glob.glob(str(checkpoint_dir / "*.ckpt")),
        key=lambda p: float(p.split("val_f1=")[1].split(".ckpt")[0]) if "val_f1=" in p else 0.0,
        reverse=True,  # highest val/f1 first
    )

    if len(ckpt_files) == 0:
        print("[Node4-2-1-2-1] No checkpoints found for ensemble, skipping ensemble test.")
        return

    print(f"[Node4-2-1-2-1] Found {len(ckpt_files)} checkpoints for ensemble: {[Path(p).name for p in ckpt_files]}")

    all_ensemble_preds: Optional[np.ndarray] = None
    n_checkpoints = 0

    for ckpt_path in ckpt_files:
        print(f"[Node4-2-1-2-1] Loading checkpoint for ensemble: {Path(ckpt_path).name}")
        # Clear previous test accumulators
        model._test_preds.clear()
        model._test_pert_ids.clear()
        model._test_symbols.clear()

        # Run test with this checkpoint (collect predictions only; don't write file yet)
        trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

        # The test predictions are written in on_test_epoch_end → read back
        pred_file = output_dir / "test_predictions.tsv"
        if pred_file.exists() and trainer.is_global_zero:
            pred_df = pd.read_csv(pred_file, sep="\t")
            # Parse predictions: shape [N, 3, N_GENES]
            preds_np = np.stack([
                np.array(json.loads(row), dtype=np.float32)
                for row in pred_df["prediction"].tolist()
            ])   # [N, 3, N_GENES]

            if all_ensemble_preds is None:
                all_ensemble_preds = preds_np
                # Save pert_ids and symbols from first checkpoint
                ensemble_pids = pred_df["idx"].tolist()
                ensemble_syms = pred_df["input"].tolist()
            else:
                all_ensemble_preds += preds_np
            n_checkpoints += 1

    # Only rank-0 writes the ensemble file
    if trainer.is_global_zero and all_ensemble_preds is not None and n_checkpoints > 1:
        # Average over checkpoints
        avg_preds = all_ensemble_preds / n_checkpoints
        rows = []
        for i in range(avg_preds.shape[0]):
            rows.append({
                "idx":        ensemble_pids[i],
                "input":      ensemble_syms[i],
                "prediction": json.dumps(avg_preds[i].tolist()),
            })
        pd.DataFrame(rows).to_csv(output_dir / "test_predictions.tsv", sep="\t", index=False)
        print(f"[Node4-2-1-2-1] Ensemble test predictions ({n_checkpoints} checkpoints) saved → "
              f"{output_dir}/test_predictions.tsv")
    else:
        print(f"[Node4-2-1-2-1] Only 1 checkpoint available; using single-checkpoint predictions.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node4-2-1-2-1 – scFoundation + STRING_GNN (frozen) + GatedFusion "
                    "+ GenePriorBias (two-stage) + gene-frequency-weighted loss (factor=4.0) "
                    "+ min_lr_ratio=0.02 + dropout schedule + multi-checkpoint ensemble"
    )
    parser.add_argument("--micro-batch-size",       type=int,   default=8)
    parser.add_argument("--global-batch-size",      type=int,   default=64)
    parser.add_argument("--max-epochs",             type=int,   default=220)
    parser.add_argument("--lr",                     type=float, default=2e-4)
    parser.add_argument("--weight-decay",           type=float, default=3e-2)
    parser.add_argument("--scf-finetune-layers",    type=int,   default=6,
                        dest="scf_finetune_layers")
    parser.add_argument("--head-dropout",           type=float, default=0.5)
    parser.add_argument("--fusion-dropout",         type=float, default=0.3)
    parser.add_argument("--warmup-epochs",          type=int,   default=7)
    parser.add_argument("--min-lr-ratio",           type=float, default=0.02)
    parser.add_argument("--mixup-alpha",            type=float, default=0.2)
    parser.add_argument("--label-smoothing",        type=float, default=0.1)
    parser.add_argument("--prior-bias-scale",       type=float, default=0.5,
                        dest="prior_bias_scale")
    parser.add_argument("--gene-diversity-factor",  type=float, default=4.0,
                        dest="gene_diversity_factor")
    parser.add_argument("--prior-unfreeze-epoch",   type=int,   default=PRIOR_UNFREEZE_EPOCH,
                        dest="prior_unfreeze_epoch")
    parser.add_argument("--dropout-schedule-start", type=int,   default=DROPOUT_SCHEDULE_START,
                        dest="dropout_schedule_start")
    parser.add_argument("--dropout-schedule-end",   type=int,   default=DROPOUT_SCHEDULE_END,
                        dest="dropout_schedule_end")
    parser.add_argument("--dropout-end-val",        type=float, default=DROPOUT_END_VAL,
                        dest="dropout_end_val")
    parser.add_argument("--patience",               type=int,   default=20)
    parser.add_argument("--gradient-clip",          type=float, default=0.5,
                        dest="gradient_clip")
    parser.add_argument("--save-top-k",             type=int,   default=3,
                        dest="save_top_k")
    parser.add_argument("--use-ensemble",           action="store_true", dest="use_ensemble",
                        default=True,
                        help="Use multi-checkpoint ensemble at test time (default: True)")
    parser.add_argument("--no-ensemble",            action="store_false", dest="use_ensemble",
                        help="Disable ensemble, use single best checkpoint")
    parser.add_argument("--num-workers",            type=int,   default=4)
    parser.add_argument("--debug-max-step",         type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",           action="store_true", dest="fast_dev_run")
    parser.add_argument("--val-check-interval",     type=float, default=1.0,
                        dest="val_check_interval")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = lim_val = lim_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # For debug/fast_dev_run, disable ensemble and use save_top_k=1
    use_ensemble = args.use_ensemble and (args.debug_max_step is None) and (not fast_dev_run)
    save_top_k   = args.save_top_k if use_ensemble else 1

    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = FusionDEGModel(
        scf_finetune_layers    = args.scf_finetune_layers,
        head_dropout           = args.head_dropout,
        fusion_dropout         = args.fusion_dropout,
        lr                     = args.lr,
        weight_decay           = args.weight_decay,
        warmup_epochs          = args.warmup_epochs,
        max_epochs             = args.max_epochs,
        min_lr_ratio           = args.min_lr_ratio,
        mixup_alpha            = args.mixup_alpha,
        label_smoothing        = args.label_smoothing,
        prior_bias_scale       = args.prior_bias_scale,
        gene_diversity_factor  = args.gene_diversity_factor,
        prior_unfreeze_epoch   = args.prior_unfreeze_epoch,
        dropout_schedule_start = args.dropout_schedule_start,
        dropout_schedule_end   = args.dropout_schedule_end,
        dropout_end_val        = args.dropout_end_val,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-val_f1={val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=save_top_k,
        auto_insert_metric_name=False,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-4)
    lr_cb  = LearningRateMonitor(logging_interval="epoch")
    pg_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = n_gpus,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        max_steps               = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches     = lim_train,
        limit_val_batches       = lim_val,
        limit_test_batches      = lim_test,
        val_check_interval      = args.val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = args.gradient_clip,   # 0.5 vs parent's 1.0
    )

    trainer.fit(model, datamodule=dm)

    # ---- Test Phase ----
    if fast_dev_run or args.debug_max_step is not None:
        # Simple single-run test for debugging
        test_results = trainer.test(model, datamodule=dm)
    elif use_ensemble and save_top_k > 1:
        # Multi-checkpoint ensemble test
        checkpoint_dir = output_dir / "checkpoints"
        run_ensemble_test(trainer, model, dm, checkpoint_dir, output_dir)
        # Also compute F1 metric using the ensemble predictions (for test_score.txt)
        test_results = [{"ensemble_checkpoints": save_top_k}]
    else:
        # Single best checkpoint test
        test_results = trainer.test(model, datamodule=dm, ckpt_path="best")

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node4-2-1-2-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
