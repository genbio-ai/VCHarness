#!/usr/bin/env python3
"""
Node node3-1-1-1-1-1-2-1: Cross-Attention Fusion with Summary Token + Cosine Warmup
======================================================================================
Builds upon parent (node3-1-1-1-1-1-2, test F1=0.5049, TREE-BEST) with four synergistic
improvements targeting convergence quality and representation capacity.

PARENT ARCHITECTURE (test F1=0.5049):
  - AIDO.Cell-10M + LoRA (r=4, alpha=8, all 8 layers, Q/K/V)
  - Frozen STRING GNN (PPI embeddings, 256-dim)
  - Symbol CNN (3-branch char-level, 64→256-dim)
  - 4-token cross-attention (3-layer TransformerEncoder, nhead=8, dim_ff=384)
  - Mean pooling over 4 tokens
  - ReduceLROnPlateau (patience=10, factor=0.5)
  - Manifold mixup (alpha=0.3), Focal loss (gamma=1.5, cw=[6,1,12], ls=0.05)
  - Top-3 checkpoint averaging

KEY IMPROVEMENTS:
  1. [PRIMARY] CosineAnnealing with 5-epoch linear warmup (replaces ReduceLROnPlateau)
     → Training dynamics analysis: breakthrough at epoch 37 was triggered by the first LR
       reduction (reactive). A planned warmup→cosine schedule enables systematic LR
       annealing that should find an even better basin without waiting for plateau detection.
     → warmup: 5 epochs linear from 1e-6 → backbone_lr/head_lr
     → cosine: decay from peak to eta_min=5e-8 over remaining epochs

  2. [SECONDARY] LoRA rank r=8, alpha=16 (doubled from r=4, alpha=8)
     → Provides ~96K trainable backbone params (vs ~48K) — more adaptation capacity
       for the complex multi-source cross-modal fusion task.
     → Risk-mitigated by weight_decay=0.10 + mixup + dropout (proven regularization suite).

  3. [TERTIARY] 5-token cross-attention with learnable [SUMMARY] token, 4 fusion layers
     → Adds a learnable [SUMMARY] token (inspired by BERT's [CLS]) to the 4-token sequence.
       The Transformer can "write" a task-specific summary by attending to all data tokens.
     → Use [SUMMARY] token output as pooled representation (instead of mean-pooling 4 tokens).
     → 4 fusion layers (vs 3): deeper cross-modal interactions.
     → Manifold mixup applied to data tokens only; [SUMMARY] token remains unchanged.

  4. [QUATERNARY] Top-5 checkpoint averaging + max_epochs=120, patience=25
     → Averages 5 best checkpoints (vs 3) for a more stable, lower-variance final model.
     → Extended training allows the cosine schedule to complete a full annealing cycle.
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
AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT = 6_640
N_GENES_MODEL = 19_264
N_CLASSES = 3
HIDDEN_DIM = 256       # AIDO.Cell-10M hidden size
FUSION_DIM = 256       # Cross-attention token dimension
CNN_DIM = 64           # Symbol CNN output dimension
STRING_DIM = 256       # STRING GNN output dimension

# Class weights: proven configuration from parent (test F1=0.5049)
CLASS_WEIGHTS = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Multi-class focal loss with class weights and label smoothing."""

    def __init__(
        self,
        gamma: float = 1.5,
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
# Symbol CNN
# ─────────────────────────────────────────────────────────────────────────────
class SymbolCNN(nn.Module):
    """
    3-branch character-level CNN for gene symbol encoding.
    Input: symbol string (up to 16 chars), padded/truncated
    Output: [B, CNN_DIM] = [B, 64]
    """
    VOCAB = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-. "
    PAD_IDX = 0
    MAX_LEN = 16

    def __init__(self, out_dim: int = 64):
        super().__init__()
        vocab_size = len(self.VOCAB) + 1  # +1 for pad
        emb_dim = 32
        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=self.PAD_IDX)

        # 3 parallel convolution branches with different kernel sizes
        self.conv1 = nn.Conv1d(emb_dim, out_dim, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(emb_dim, out_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(emb_dim, out_dim, kernel_size=4, padding=2)
        self.proj = nn.Linear(out_dim * 3, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self._build_char_map()

    def _build_char_map(self):
        self._char2idx: Dict[str, int] = {}
        for i, c in enumerate(self.VOCAB):
            self._char2idx[c] = i + 1  # 0 = pad

    def encode_symbols(self, symbols: List[str]) -> torch.Tensor:
        """Convert list of symbol strings to [B, MAX_LEN] integer tensor."""
        ids = []
        for sym in symbols:
            sym_upper = sym.upper()[:self.MAX_LEN]
            row = [self._char2idx.get(c, self.PAD_IDX) for c in sym_upper]
            row += [self.PAD_IDX] * (self.MAX_LEN - len(row))
            ids.append(row)
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: [B, MAX_LEN]
        x = self.char_emb(char_ids)             # [B, MAX_LEN, emb_dim]
        x = x.transpose(1, 2)                   # [B, emb_dim, MAX_LEN]

        x1 = F.relu(self.conv1(x))              # [B, out_dim, MAX_LEN+1]
        x2 = F.relu(self.conv2(x))              # [B, out_dim, MAX_LEN]
        x3 = F.relu(self.conv3(x))              # [B, out_dim, MAX_LEN+2]

        # Global max pool each branch
        x1 = x1.max(dim=-1).values              # [B, out_dim]
        x2 = x2.max(dim=-1).values              # [B, out_dim]
        x3 = x3.max(dim=-1).values              # [B, out_dim]

        x = torch.cat([x1, x2, x3], dim=-1)     # [B, out_dim * 3]
        x = self.proj(x)                         # [B, out_dim]
        return self.norm(x)                      # [B, out_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Perturbation DEG prediction dataset.
    Synthetic expression encoding: all 19264 genes at 1.0, perturbed gene at 0.0.
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

        # Pre-compute expression vectors
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
                expr = base_expr.clone()
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
        "expr": torch.stack([b["expr"] for b in batch]),                   # [B, 19264]
        "pert_pos": torch.tensor([b["pert_pos"] for b in batch], dtype=torch.long),
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])         # [B, 6640]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, micro_batch_size: int = 4, num_workers: int = 0):
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
        """Build mapping from ENSG gene ID to its position in the 19264-gene vocabulary."""
        gene_pos_map: Dict[str, int] = {}
        unique_base_ids = list(set(pid.split(".")[0] for pid in all_pert_ids))
        print(f"[DataModule] Building gene position map for {len(unique_base_ids)} unique genes...")

        if hasattr(tokenizer, "gene_id_to_index"):
            gid2idx = tokenizer.gene_id_to_index
            for base_pid in unique_base_ids:
                if base_pid in gid2idx:
                    gene_pos_map[base_pid] = gid2idx[base_pid]
            if len(gene_pos_map) > 0:
                print(f"[DataModule] ENSG→pos via gene_id_to_index: "
                      f"{len(gene_pos_map)}/{len(unique_base_ids)} found")
                return gene_pos_map

        print(f"[DataModule] No gene position mapping available; all pert_pos will be -1")
        return gene_pos_map

    def setup(self, stage: Optional[str] = None) -> None:
        # Initialize AIDO.Cell tokenizer: rank-0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        # Build gene position map once
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
# Model: Cross-Attention Fusion with Summary Token
# ─────────────────────────────────────────────────────────────────────────────
class SummaryTokenFusionDEGModel(nn.Module):
    """
    Improved Cross-Attention Fusion with [SUMMARY] token.

    Combines:
      1. AIDO.Cell-10M (LoRA r=8, all 8 layers) → global_emb + pert_emb [B, 256]
      2. Frozen STRING GNN PPI embeddings (precomputed) → ppi_feat [B, 256]
      3. Character-level Symbol CNN (3-branch, 64-dim → 256-dim) → sym_proj [B, 256]

    Fuses via 5-token sequence:
      [SUMMARY](learnable) + global_emb + pert_emb + sym_feat + ppi_feat
      → 4-layer TransformerEncoder (nhead=8, dim_ff=384, pre-norm)
      → Use [SUMMARY] token output (position 0) as pooled representation
      → MLP head: 256 → 256 → 19920
    """

    def __init__(
        self,
        ppi_embeddings: torch.Tensor,      # [N_nodes, 256] precomputed STRING GNN embeddings
        ppi_node_names: List[str],          # ENSG IDs for each node
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        num_fusion_layers: int = 4,
        num_fusion_heads: int = 8,
        fusion_dim_ff: int = 384,
        fusion_dropout: float = 0.2,
        head_dropout: float = 0.5,
        cnn_dim: int = 64,
    ):
        super().__init__()

        # ── PPI embedding table (frozen, precomputed from STRING GNN) ─────────
        self.register_buffer("_ppi_emb", ppi_embeddings.float())  # [N, 256]
        # Build ENSG → node index lookup
        self._ppi_name2idx: Dict[str, int] = {
            name: i for i, name in enumerate(ppi_node_names)
        }

        # ── AIDO.Cell-10M backbone with LoRA (r=8, all 8 layers) ─────────────
        backbone = AutoModel.from_pretrained(
            AIDO_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16
        )
        backbone.config.use_cache = False

        # Monkey-patch enable_input_require_grads for PEFT compatibility
        def noop_enable_input_require_grads(self_):
            pass
        backbone.enable_input_require_grads = noop_enable_input_require_grads.__get__(
            backbone, type(backbone)
        )

        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # LoRA r=8 on ALL 8 layers (doubled from parent's r=4)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=list(range(8)),  # all 8 layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Cast LoRA (trainable) params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Symbol CNN ────────────────────────────────────────────────────────
        self.sym_cnn = SymbolCNN(out_dim=cnn_dim)  # [B, 64]
        self.sym_proj = nn.Linear(cnn_dim, FUSION_DIM)  # [B, 64 → 256]
        self.sym_norm = nn.LayerNorm(FUSION_DIM)

        # ── PPI projection (STRING GNN 256-dim → fusion 256-dim) ─────────────
        self.ppi_norm = nn.LayerNorm(STRING_DIM)

        # ── Learnable [SUMMARY] token ─────────────────────────────────────────
        # Inspired by BERT's [CLS] token: a learnable "query" that attends to
        # all 4 data tokens and aggregates a task-specific summary representation.
        self.summary_token = nn.Parameter(
            torch.randn(1, 1, FUSION_DIM) * 0.02
        )

        # ── 5-token Cross-Attention Fusion (4-layer TransformerEncoder) ───────
        # 5 tokens: [SUMMARY], global_emb, pert_emb, sym_feat, ppi_feat
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FUSION_DIM,
            nhead=num_fusion_heads,
            dim_feedforward=fusion_dim_ff,
            dropout=fusion_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm (more stable)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_fusion_layers,
        )

        # ── 2-layer MLP head ─────────────────────────────────────────────────
        out_dim = N_CLASSES * N_GENES_OUT  # 3 × 6640 = 19920
        self.head_fc1 = nn.Linear(FUSION_DIM, FUSION_DIM)
        self.head_act = nn.GELU()
        self.head_drop = nn.Dropout(head_dropout)
        self.head_fc2 = nn.Linear(FUSION_DIM, out_dim)

        # Initialize prediction head
        nn.init.trunc_normal_(self.head_fc1.weight, std=0.02)
        nn.init.zeros_(self.head_fc1.bias)
        nn.init.trunc_normal_(self.head_fc2.weight, std=0.02)
        nn.init.zeros_(self.head_fc2.bias)

    def _lookup_ppi_embeddings(
        self, pert_ids: List[str]
    ) -> torch.Tensor:
        """Look up PPI embeddings for a batch of pert_ids."""
        device = self._ppi_emb.device
        idxs = []
        for pid in pert_ids:
            base_pid = pid.split(".")[0]
            idx = self._ppi_name2idx.get(base_pid, -1)
            idxs.append(idx)

        result = torch.zeros(len(pert_ids), STRING_DIM, device=device)
        valid_mask = [i >= 0 for i in idxs]
        valid_idxs_list = [max(i, 0) for i in idxs]
        valid_idxs = torch.tensor(valid_idxs_list, dtype=torch.long, device=device)

        valid_bool = torch.tensor(valid_mask, dtype=torch.bool, device=device)
        if valid_bool.any():
            result[valid_bool] = self._ppi_emb[valid_idxs[valid_bool]]

        return result  # [B, 256], zeros for genes not in STRING GNN

    def forward(
        self,
        expr: torch.Tensor,       # [B, 19264] float32
        pert_pos: torch.Tensor,   # [B] long
        char_ids: torch.Tensor,   # [B, 16] long (symbol characters)
        pert_ids: List[str],      # list of ENSG IDs
    ) -> torch.Tensor:
        """
        Returns logits: [B, 3, 6640]
        """
        B = expr.shape[0]
        device = expr.device

        # ── 1. AIDO.Cell-10M forward pass ────────────────────────────────────
        outputs = self.backbone(
            input_ids=expr,
            attention_mask=torch.ones(B, N_GENES_MODEL, dtype=torch.long, device=device),
        )
        hidden = outputs.last_hidden_state.float()  # [B, 19266, 256]

        # Global mean pool (gene positions only, exclude 2 summary tokens)
        global_emb = hidden[:, :N_GENES_MODEL, :].mean(dim=1)  # [B, 256]

        # Per-sample perturbed-gene positional embedding
        safe_pos = pert_pos.clamp(min=0)
        pos_idx = safe_pos.view(B, 1, 1).expand(B, 1, HIDDEN_DIM)
        pert_emb = hidden.gather(1, pos_idx).squeeze(1)  # [B, 256]

        # For genes not in vocabulary: fall back to global pool
        unknown_mask = (pert_pos < 0)
        if unknown_mask.any():
            pert_emb = pert_emb.clone()
            pert_emb[unknown_mask] = global_emb[unknown_mask]

        # ── 2. STRING GNN PPI embeddings (frozen, precomputed) ───────────────
        ppi_feat = self._lookup_ppi_embeddings(pert_ids).to(device)  # [B, 256]
        ppi_feat = self.ppi_norm(ppi_feat)                            # [B, 256]

        # ── 3. Symbol CNN ────────────────────────────────────────────────────
        char_ids = char_ids.to(device)
        sym_feat = self.sym_cnn(char_ids)                  # [B, 64]
        sym_feat = self.sym_norm(self.sym_proj(sym_feat))  # [B, 256]

        # ── 4. 5-token cross-attention fusion ────────────────────────────────
        # Stack 4 data tokens: global, perturbed, symbol, PPI
        data_tokens = torch.stack(
            [global_emb, pert_emb, sym_feat, ppi_feat], dim=1
        )  # [B, 4, 256]

        # Prepend learnable [SUMMARY] token
        summary = self.summary_token.expand(B, 1, FUSION_DIM)  # [B, 1, 256]
        tokens = torch.cat([summary, data_tokens], dim=1)       # [B, 5, 256]

        fused = self.transformer_encoder(tokens)   # [B, 5, 256]
        pooled = fused[:, 0, :]                    # [SUMMARY] token output [B, 256]

        # ── 5. MLP prediction head ───────────────────────────────────────────
        x = self.head_fc1(pooled)                  # [B, 256]
        x = self.head_act(x)
        x = self.head_drop(x)
        logits = self.head_fc2(x)                  # [B, 19920]

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
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        num_fusion_layers: int = 4,
        num_fusion_heads: int = 8,
        fusion_dim_ff: int = 384,
        fusion_dropout: float = 0.2,
        head_dropout: float = 0.5,
        cnn_dim: int = 64,
        backbone_lr: float = 2e-4,
        head_lr: float = 6e-4,
        weight_decay: float = 0.10,
        focal_gamma: float = 1.5,
        label_smoothing: float = 0.05,
        max_epochs: int = 120,
        warmup_epochs: int = 5,
        eta_min: float = 5e-8,
        mixup_alpha: float = 0.3,
        top_k_avg: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[SummaryTokenFusionDEGModel] = None
        self.loss_fn: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        # Symbol CNN helper for encoding symbols
        self._sym_cnn_helper: Optional[SymbolCNN] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            # Load STRING GNN embeddings (precomputed, frozen)
            print("[Model] Loading STRING GNN for precomputed embeddings...")
            string_model_dir = Path(STRING_GNN_DIR)
            string_gnn = AutoModel.from_pretrained(
                str(string_model_dir), trust_remote_code=True
            )
            string_gnn = string_gnn.cuda()
            string_gnn.eval()

            graph = torch.load(str(string_model_dir / "graph_data.pt"))
            node_names = json.loads((string_model_dir / "node_names.json").read_text())

            edge_index = graph["edge_index"].cuda()
            edge_weight = graph.get("edge_weight", None)
            if edge_weight is not None:
                edge_weight = edge_weight.cuda()

            with torch.no_grad():
                string_outputs = string_gnn(
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
            ppi_embeddings = string_outputs.last_hidden_state.cpu()  # [18870, 256]
            print(f"[Model] STRING GNN embeddings: {ppi_embeddings.shape}")

            # Free STRING GNN memory
            del string_gnn, string_outputs, edge_index
            if edge_weight is not None:
                del edge_weight
            torch.cuda.empty_cache()

            self.model = SummaryTokenFusionDEGModel(
                ppi_embeddings=ppi_embeddings,
                ppi_node_names=node_names,
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                num_fusion_layers=self.hparams.num_fusion_layers,
                num_fusion_heads=self.hparams.num_fusion_heads,
                fusion_dim_ff=self.hparams.fusion_dim_ff,
                fusion_dropout=self.hparams.fusion_dropout,
                head_dropout=self.hparams.head_dropout,
                cnn_dim=self.hparams.cnn_dim,
            )
            self._sym_cnn_helper = self.model.sym_cnn

            self.loss_fn = FocalLoss(
                gamma=self.hparams.focal_gamma,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        # Populate test metadata
        if stage in ("test", None):
            dm = getattr(self, "trainer", None)
            if dm is not None:
                dm = getattr(self.trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "test_pert_ids") and dm.test_pert_ids:
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols

    def _encode_symbols(self, symbols: List[str]) -> torch.Tensor:
        """Encode a list of symbol strings to character ID tensors."""
        return self._sym_cnn_helper.encode_symbols(symbols)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        char_ids = self._encode_symbols(batch["symbols"])  # [B, MAX_LEN]
        return self.model(
            batch["expr"],
            batch["pert_pos"],
            char_ids,
            batch["pert_ids"],
        )

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lam: Optional[float] = None,
        labels_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # logits: [B, 3, 6640], labels: [B, 6640]
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                               # [B*6640]

        if lam is not None and labels_b is not None:
            labels_b_flat = labels_b.reshape(-1)
            loss = lam * self.loss_fn(logits_flat, labels_flat) + \
                   (1 - lam) * self.loss_fn(logits_flat, labels_b_flat)
        else:
            loss = self.loss_fn(logits_flat, labels_flat)
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        char_ids = self._encode_symbols(batch["symbols"])

        # Get backbone features
        B = batch["expr"].shape[0]
        device = batch["expr"].device

        outputs = self.model.backbone(
            input_ids=batch["expr"],
            attention_mask=torch.ones(B, N_GENES_MODEL, dtype=torch.long, device=device),
        )
        hidden = outputs.last_hidden_state.float()  # [B, 19266, 256]

        global_emb = hidden[:, :N_GENES_MODEL, :].mean(dim=1)
        safe_pos = batch["pert_pos"].clamp(min=0)
        pos_idx = safe_pos.view(B, 1, 1).expand(B, 1, HIDDEN_DIM)
        pert_emb = hidden.gather(1, pos_idx).squeeze(1)
        unknown_mask = (batch["pert_pos"] < 0)
        if unknown_mask.any():
            pert_emb = pert_emb.clone()
            pert_emb[unknown_mask] = global_emb[unknown_mask]

        ppi_feat = self.model._lookup_ppi_embeddings(batch["pert_ids"]).to(device)
        ppi_feat = self.model.ppi_norm(ppi_feat)

        char_ids = char_ids.to(device)
        sym_feat = self.model.sym_cnn(char_ids)
        sym_feat = self.model.sym_norm(self.model.sym_proj(sym_feat))

        # Stack 4 data tokens: global, perturbed, symbol, PPI
        data_tokens = torch.stack(
            [global_emb, pert_emb, sym_feat, ppi_feat], dim=1
        )  # [B, 4, 256]

        # Apply manifold mixup to data tokens only (not [SUMMARY] token)
        labels = batch["label"]
        lam = None
        labels_b = None
        if self.hparams.mixup_alpha > 0 and self.training:
            lam_val = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
            perm = torch.randperm(B, device=device)
            data_tokens = lam_val * data_tokens + (1 - lam_val) * data_tokens[perm]
            lam = lam_val
            labels_b = labels[perm]

        # Prepend learnable [SUMMARY] token (not mixed)
        summary = self.model.summary_token.expand(B, 1, FUSION_DIM)
        tokens = torch.cat([summary, data_tokens], dim=1)  # [B, 5, 256]

        # Cross-attention fusion
        fused = self.model.transformer_encoder(tokens)  # [B, 5, 256]
        pooled = fused[:, 0, :]                         # [SUMMARY] token output [B, 256]

        # Prediction head
        x = self.model.head_fc1(pooled)
        x = self.model.head_act(x)
        x = self.model.head_drop(x)
        logits = self.model.head_fc2(x).view(B, N_CLASSES, N_GENES_OUT)

        loss = self._compute_loss(logits, labels, lam=lam, labels_b=labels_b)
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

        # Determine world size: use distributed if available
        use_dist = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if use_dist:
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1

        if world_size > 1:
            # Multi-GPU: gather from all ranks and de-duplicate
            ap = self.all_gather(lp)
            al = self.all_gather(ll)
            ai = self.all_gather(li.long())

            preds_list: List[np.ndarray] = []
            labels_list: List[np.ndarray] = []
            seen: set = set()

            for w in range(world_size):
                for i in range(ap.shape[1]):
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
            # Single-GPU: no deduplication needed
            order = np.argsort(li.cpu().numpy())
            preds_arr = lp.cpu().numpy()[order]
            labels_arr = ll.cpu().numpy()[order]

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        f1_val = compute_deg_f1(preds_arr, labels_arr.astype(np.int64))
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

        # Determine world size: use distributed if available
        use_dist = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if use_dist:
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1

        if world_size > 1:
            ap = self.all_gather(lp)
            ai = self.all_gather(li.long())

            preds_list: List[np.ndarray] = []
            idxs_list: List[int] = []
            seen: set = set()

            for w in range(world_size):
                for i in range(ap.shape[1]):
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
            # Single-GPU: no deduplication needed
            order = np.argsort(li.cpu().numpy())
            preds_arr = lp.cpu().numpy()[order]
            idxs_arr = li.cpu().numpy()[order]

        self._test_preds.clear()
        self._test_indices.clear()

        if self.trainer.is_global_zero:
            # Ensure test pert_ids and symbols are available from datamodule
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "test_pert_ids") and dm.test_pert_ids:
                test_pert_ids = dm.test_pert_ids
                test_symbols = dm.test_symbols
            else:
                # Fallback: use stored metadata (may be empty in fast_dev_run)
                test_pert_ids = self._test_pert_ids
                test_symbols = self._test_symbols

            # Verify alignment: idxs_arr values should be valid indices into test_pert_ids
            if len(test_pert_ids) == 0:
                self.print(
                    "[WARNING] test_pert_ids is empty! Cannot write predictions. "
                    "Ensure test dataset is properly loaded."
                )
                return

            # Map global indices to pert_id/symbol strings
            rows = []
            for r, global_idx in enumerate(idxs_arr):
                idx_int = int(global_idx)
                if 0 <= idx_int < len(test_pert_ids):
                    rows.append({
                        "idx": test_pert_ids[idx_int],
                        "input": test_symbols[idx_int],
                        "prediction": json.dumps(preds_arr[r].tolist()),
                    })
                else:
                    self.print(f"[WARNING] idx {idx_int} out of range for test set (len={len(test_pert_ids)})")
                    rows.append({
                        "idx": f"idx_{idx_int}",
                        "input": "unknown",
                        "prediction": json.dumps(preds_arr[r].tolist()),
                    })

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            pred_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {pred_path} ({len(rows)} samples)")

    def configure_optimizers(self):
        # Separate learning rates: LoRA backbone (lower) vs head (higher)
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters() if p.requires_grad
        ]
        head_params = (
            list(self.model.sym_cnn.parameters()) +
            list(self.model.sym_proj.parameters()) +
            list(self.model.sym_norm.parameters()) +
            list(self.model.ppi_norm.parameters()) +
            [self.model.summary_token] +
            list(self.model.transformer_encoder.parameters()) +
            list(self.model.head_fc1.parameters()) +
            list(self.model.head_fc2.parameters())
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
                {"params": head_params,     "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
            eps=1e-8,
        )

        # Cosine annealing with linear warmup
        # Phase 1 (warmup_epochs): linear ramp from near-zero to peak LR
        # Phase 2 (remaining epochs): cosine decay from peak to eta_min
        warmup_epochs = self.hparams.warmup_epochs
        total_epochs = self.hparams.max_epochs
        cosine_epochs = max(1, total_epochs - warmup_epochs)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6 / max(self.hparams.backbone_lr, 1e-10),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=self.hparams.eta_min,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                # No "monitor" key needed for non-plateau schedulers
            },
        }

    # ── Checkpoint helpers ─────────────────────────────────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters (exclude large precomputed PPI buffer)."""
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_sd = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable_sd[key] = full_sd[key]
        # Note: model.summary_token is a Parameter (requires_grad=True) — saved automatically
        # Skip PPI buffer (_ppi_emb) — large (~19MB), precomputed at setup time
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving checkpoint: {trainable}/{total} params "
            f"({100.0 * trainable / total:.2f}%)"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        """Load partial checkpoint (trainable params only)."""
        trainable_keys = {n for n, p in self.named_parameters() if p.requires_grad}

        missing = [k for k in trainable_keys if k not in state_dict]
        unexpected = [k for k in state_dict if k not in trainable_keys]
        if missing:
            self.print(f"Warning: Missing keys (first 5): {missing[:5]}")
        if unexpected:
            self.print(f"Warning: Unexpected keys (first 5): {unexpected[:5]}")
        return super().load_state_dict(state_dict, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# Top-K Checkpoint Averaging
# ─────────────────────────────────────────────────────────────────────────────
def average_top_k_checkpoints(
    ckpt_dir: Path,
    model_module: DEGLightningModule,
    top_k: int = 5,
) -> None:
    """
    Load top-k checkpoints by val_f1 score and average their weights.
    Updates model_module.model in-place.
    """
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    # Filter out 'last.ckpt'
    ckpt_files = [f for f in ckpt_files if f.stem != "last"]

    if len(ckpt_files) == 0:
        print("[CheckpointAvg] No checkpoint files found, skipping averaging")
        return

    # Extract val_f1 from filename
    def extract_val_f1(path: Path) -> float:
        stem = path.stem
        if "val_f1=" in stem:
            try:
                return float(stem.split("val_f1=")[-1])
            except Exception:
                pass
        return 0.0

    ckpt_files.sort(key=extract_val_f1, reverse=True)
    top_ckpts = ckpt_files[:top_k]

    if len(top_ckpts) < 2:
        print(f"[CheckpointAvg] Only {len(top_ckpts)} checkpoints found, loading best only")
        state = torch.load(str(top_ckpts[0]), map_location="cpu", weights_only=False)
        sd = state.get("state_dict", state)
        model_module.load_state_dict(sd, strict=False)
        return

    print(f"[CheckpointAvg] Averaging top-{len(top_ckpts)} checkpoints:")
    for c in top_ckpts:
        print(f"  {c.name} (val_f1={extract_val_f1(c):.4f})")

    # Load and average state dicts
    state_dicts = []
    for ckpt_path in top_ckpts:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = state.get("state_dict", state)
        state_dicts.append(sd)

    # Average matching keys
    avg_sd = {}
    for key in state_dicts[0]:
        tensors = [sd[key].float() for sd in state_dicts if key in sd]
        if tensors:
            avg_sd[key] = torch.stack(tensors).mean(0)

    model_module.load_state_dict(avg_sd, strict=False)
    print(f"[CheckpointAvg] Successfully averaged {len(state_dicts)} checkpoints")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node node3-1-1-1-1-1-2-1: Summary-Token Cross-Attention Fusion (AIDO.Cell + STRING GNN + Symbol CNN)"
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "data"),
    )
    p.add_argument("--micro_batch_size",        type=int,   default=4)
    p.add_argument("--global_batch_size",        type=int,   default=32)
    p.add_argument("--max_epochs",               type=int,   default=120)
    p.add_argument("--backbone_lr",              type=float, default=2e-4)
    p.add_argument("--head_lr",                  type=float, default=6e-4)
    p.add_argument("--weight_decay",             type=float, default=0.10)
    p.add_argument("--lora_r",                   type=int,   default=8)
    p.add_argument("--lora_alpha",               type=int,   default=16)
    p.add_argument("--lora_dropout",             type=float, default=0.1)
    p.add_argument("--num_fusion_layers",        type=int,   default=4)
    p.add_argument("--num_fusion_heads",         type=int,   default=8)
    p.add_argument("--fusion_dim_ff",            type=int,   default=384)
    p.add_argument("--fusion_dropout",           type=float, default=0.2)
    p.add_argument("--head_dropout",             type=float, default=0.5)
    p.add_argument("--cnn_dim",                  type=int,   default=64)
    p.add_argument("--focal_gamma",              type=float, default=1.5)
    p.add_argument("--label_smoothing",          type=float, default=0.05)
    p.add_argument("--mixup_alpha",              type=float, default=0.3)
    p.add_argument("--warmup_epochs",            type=int,   default=5)
    p.add_argument("--eta_min",                  type=float, default=5e-8)
    p.add_argument("--early_stopping_patience",  type=int,   default=25)
    p.add_argument("--top_k_avg",                type=int,   default=5)
    p.add_argument("--num_workers",              type=int,   default=0)
    p.add_argument("--val_check_interval",       type=float, default=1.0)
    p.add_argument("--debug_max_step",           type=int,   default=None)
    p.add_argument("--fast_dev_run",             action="store_true")
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
        filename="node3-1-1-1-1-1-2-1-xattn-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=args.top_k_avg,
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
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_fusion_layers=args.num_fusion_layers,
        num_fusion_heads=args.num_fusion_heads,
        fusion_dim_ff=args.fusion_dim_ff,
        fusion_dropout=args.fusion_dropout,
        head_dropout=args.head_dropout,
        cnn_dim=args.cnn_dim,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.eta_min,
        mixup_alpha=args.mixup_alpha,
        top_k_avg=args.top_k_avg,
    )

    trainer.fit(model_module, datamodule=datamodule)

    # Test: use best checkpoint or top-k average
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    elif n_gpus == 1 and args.top_k_avg > 1:
        # Single-GPU mode: apply top-k checkpoint averaging
        ckpt_dir = output_dir / "checkpoints"
        if ckpt_dir.exists():
            average_top_k_checkpoints(ckpt_dir, model_module, top_k=args.top_k_avg)
            print(f"[Main] Applied top-{args.top_k_avg} checkpoint averaging for test")
            test_results = trainer.test(model_module, datamodule=datamodule)
        else:
            test_results = trainer.test(
                model_module, datamodule=datamodule, ckpt_path="best"
            )
    else:
        # Multi-GPU mode: load best checkpoint
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
            f"# Node node3-1-1-1-1-1-2-1: Summary-Token Cross-Attention Fusion\n"
            f"# Architecture: AIDO.Cell-10M (LoRA r=8, all 8 layers) + STRING GNN + Symbol CNN\n"
            f"# Fusion: 4-layer TransformerEncoder + learnable [SUMMARY] token (5-token sequence)\n"
            f"# LR Schedule: 5-epoch warmup + CosineAnnealingLR (replaces ReduceLROnPlateau)\n"
            f"# Primary metric: f1_score (macro-averaged per-gene F1)\n"
            f"# Parent: node3-1-1-1-1-1-2 (test F1=0.5049, tree-best)\n"
            f"\n"
            f"Best val_f1 (from checkpoint): {primary_val:.6f}\n"
            f"\n"
        )
        print(f"Score saved → {score_path}")
        print(f"Best val_f1: {primary_val:.6f}")


if __name__ == "__main__":
    main()
