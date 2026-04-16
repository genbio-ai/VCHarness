#!/usr/bin/env python3
"""
Node 1-2-2-3: LoRA AIDO + Static STRING + CharCNN 4-Source Fusion
   with ReduceLROnPlateau(patience=40) — Matching node3-2 Training Dynamics
==========================================================================
This node applies the exact training dynamics of node3-2 (reference best, F1=0.462)
to the node1-2-2 branch. The architecture is IDENTICAL to the parent (node1-2-2).
The only changes are:

  1. ReduceLROnPlateau patience: 8 → 40
     - With patience=40 and typical val_f1 oscillation ±0.005,
       the scheduler effectively NEVER fires — matching node3-2's behavior.
     - Both siblings (node1-2-2-1, node1-2-2-2) tried cosine LR variants
       that still reduced the LR aggressively, causing regression.
     - Parent's patience=8 fired at E47 and E56, causing double regression.
     - The fix: patience so high that RLROP never fires during training.

  2. Early stopping patience: 25 → 20
     - Parent ran 25 extra epochs post-peak (E38→E63), overfitting throughout.
     - Sibling 1's patience=12 was too tight (E33 best, stopped at E45).
     - patience=20 is the balanced midpoint: stops at E58 if best at E38.

  3. max_epochs: 120 → 80
     - Best epochs in this lineage are E14-E38; patience=20 stops at E34-E58.
     - 80 epochs provides sufficient headroom.

Everything else is UNCHANGED from parent node1-2-2:
  - Architecture: LoRA r=4 all 8 Q/K/V + static STRING + CharCNN + 384-head
  - weight_decay=0.03 (NOT 0.05 like siblings — sibling 1 showed over-regularization)
  - head_dropout=0.4 (NOT 0.45 like siblings — proven optimal)
  - class_weights=[5.0, 1.0, 10.0] (NOT relaxed like sibling 2 — causes regression)
  - backbone_lr=2e-4, head_lr=6e-4 (proven differential LRs)
  - seed=7 (confirmed better in node2-3 lineage)
  - No gradient clipping (sibling 1 showed clip=1.0 provides no benefit)
  - save_top_k=1 (sibling 2 showed averaging declining checkpoints hurts)

Evidence for patience=40 approach:
  - node3-2 (F1=0.462): patience=40, never fires → best in entire tree
  - node2-3-1-1-1-1 (F1=0.4573): patience=20, rarely fires → second-best
  - node1-3-1-1-1-1 (F1=0.4490): patience=30, never fires → consistent
  - node1-2-2 (F1=0.4300): patience=8, fires TWICE → double degradation
  - Pattern: higher patience → flat LR → better performance
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
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
N_GENES = 6_640
N_CLASSES = 3

AIDO_MODEL_PATH = "/home/Models/AIDO.Cell-10M"
STRING_GNN_PATH = "/home/Models/STRING_GNN"

AIDO_DIM = 256          # AIDO.Cell-10M hidden dim
STRING_DIM = 256        # STRING_GNN output dim
CNN_DIM = 64            # Character-level CNN output dim
DUAL_POOL_DIM = AIDO_DIM * 2    # 512 (gene_pos + mean_pool)
FUSION_DIM = DUAL_POOL_DIM + STRING_DIM + CNN_DIM  # 832

# Class weights: proven optimal in node3-2 and node2-3 lineage
# [5.0, 1.0, 10.0] for [down-regulated, unchanged, up-regulated]
# node1-2-2-2 (sibling 2) showed that relaxing to [4,1,8] causes regression
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)

# Character vocabulary for Ensembl gene IDs (e.g. "ENSG00000001084")
ENSEMBL_CHARS = "ENSG0123456789"  # 14 unique characters
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(ENSEMBL_CHARS)}  # 1-indexed, 0=pad
CHAR_VOCAB_SIZE = len(ENSEMBL_CHARS) + 1
ENSEMBL_MAX_LEN = 15  # length of "ENSG00000001084" + buffer


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weighting and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
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
            logits,
            targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none").detach())
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper (mirrors calc_metric.py exactly)
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
# Character-level CNN for Ensembl IDs
# ─────────────────────────────────────────────────────────────────────────────
class GeneSymbolCNN(nn.Module):
    """
    3-branch character-level CNN for Ensembl gene IDs.
    Each branch: Conv1d(embed_dim, out_channels, kernel_size) → GELU → AdaptiveMaxPool1d(1)
    Concatenated branches → linear projection to cnn_out_dim.
    """

    def __init__(
        self,
        vocab_size: int = CHAR_VOCAB_SIZE,
        embed_dim: int = 32,
        out_channels: int = 32,
        cnn_out_dim: int = CNN_DIM,
        kernel_sizes: tuple = (3, 5, 7),
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, out_channels, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(out_channels * len(kernel_sizes), cnn_out_dim),
            nn.GELU(),
        )

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        char_ids: [B, L] int64
        Returns:  [B, cnn_out_dim]
        """
        x = self.embedding(char_ids)   # [B, L, embed_dim]
        x = x.transpose(1, 2)          # [B, embed_dim, L]
        branch_outputs = []
        for conv in self.convs:
            h = F.gelu(conv(x))        # [B, out_channels, L]
            h = self.pool(h).squeeze(-1)  # [B, out_channels]
            branch_outputs.append(h)
        x = torch.cat(branch_outputs, dim=-1)  # [B, out_channels * n_branches]
        return self.proj(x)  # [B, cnn_out_dim]


def encode_ensembl_id(pert_id: str, max_len: int = ENSEMBL_MAX_LEN) -> List[int]:
    """Encode Ensembl gene ID as character indices (1-indexed, 0=pad)."""
    chars = [CHAR_TO_IDX.get(c, 0) for c in pert_id[:max_len]]
    chars = chars + [0] * (max_len - len(chars))
    return chars[:max_len]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Dataset that pre-tokenizes AIDO.Cell inputs and stores:
    - STRING node embeddings (pre-computed static, 256-dim per sample)
    - char-CNN indices for Ensembl IDs
    """

    def __init__(
        self,
        df: pd.DataFrame,
        input_ids: torch.Tensor,        # [N, 19264] float32 (AIDO.Cell input)
        pert_positions: torch.Tensor,   # [N] int64 (-1 if gene not in AIDO vocab)
        string_embs: torch.Tensor,      # [N, 256] float32 (pre-computed STRING embs)
        char_ids: torch.Tensor,         # [N, ENSEMBL_MAX_LEN] int64
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.input_ids = input_ids
        self.pert_positions = pert_positions
        self.string_embs = string_embs
        self.char_ids = char_ids
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            arr = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1} → {0,1,2}
            self.labels = torch.from_numpy(arr).long()      # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "input_ids": self.input_ids[idx],            # [19264] float32
            "pert_pos": self.pert_positions[idx],         # int64 (-1 if unknown)
            "string_emb": self.string_embs[idx],          # [256] float32
            "char_ids": self.char_ids[idx],               # [ENSEMBL_MAX_LEN] int64
        }
        if not self.is_test:
            item["label"] = self.labels[idx]   # [6640] int64
        return item


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 8,
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
        """Rank-safe tokenizer initialization."""
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
    ) -> tuple:
        """Tokenize pert_ids for AIDO.Cell and get positional indices."""
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        chunk_size = 128
        all_input_ids: List[torch.Tensor] = []
        for i in range(0, len(expr_dicts), chunk_size):
            chunk = expr_dicts[i : i + chunk_size]
            toks = tokenizer(chunk, return_tensors="pt")
            all_input_ids.append(toks["input_ids"])
        input_ids = torch.cat(all_input_ids, dim=0)  # [N, 19264] float32
        non_missing = input_ids > -0.5
        has_gene = non_missing.any(dim=1)
        pert_positions = non_missing.long().argmax(dim=1)
        pert_positions[~has_gene] = -1
        coverage = 100.0 * has_gene.float().mean().item()
        print(f"  [{split_name}] AIDO vocab coverage: "
              f"{has_gene.sum().item()}/{len(pert_ids)} ({coverage:.1f}%)")
        return input_ids, pert_positions

    def _precompute_string_embs(
        self,
        pert_ids: List[str],
        split_name: str = "split",
    ) -> torch.Tensor:
        """
        Pre-compute STRING_GNN embeddings ONCE at setup.
        Returns [N, 256] float32 tensor. Uses null_emb for unknown genes.
        """
        model_dir = Path(STRING_GNN_PATH)
        node_names = json.loads((model_dir / "node_names.json").read_text())
        node_name_to_idx = {name: i for i, name in enumerate(node_names)}
        null_emb = torch.zeros(STRING_DIM, dtype=torch.float32)

        gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
        gnn_model.eval()
        for param in gnn_model.parameters():
            param.requires_grad = False

        graph = torch.load(str(model_dir / "graph_data.pt"), map_location="cpu")
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gnn_model = gnn_model.to(device)
        edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        with torch.no_grad():
            out = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            all_embs = out.last_hidden_state.float().cpu()  # [18870, 256]

        del gnn_model, edge_index, edge_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Look up per-sample embeddings
        n_found = 0
        result = []
        for pid in pert_ids:
            idx = node_name_to_idx.get(pid, -1)
            if idx >= 0:
                result.append(all_embs[idx])
                n_found += 1
            else:
                result.append(null_emb.clone())
        coverage = 100.0 * n_found / len(pert_ids)
        print(f"  [{split_name}] STRING vocab coverage: "
              f"{n_found}/{len(pert_ids)} ({coverage:.1f}%)")
        return torch.stack(result, dim=0)  # [N, 256]

    def _encode_char_ids(self, pert_ids: List[str]) -> torch.Tensor:
        """Encode Ensembl IDs to char-level indices. Returns [N, ENSEMBL_MAX_LEN]."""
        char_id_list = [encode_ensembl_id(pid) for pid in pert_ids]
        return torch.tensor(char_id_list, dtype=torch.long)

    def setup(self, stage: Optional[str] = None) -> None:
        tokenizer = self._init_tokenizer()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")

            print("Preparing train set...")
            tr_ids, tr_pos = self._tokenize_and_get_positions(
                tokenizer, train_df["pert_id"].tolist(), "train")
            tr_str_embs = self._precompute_string_embs(
                train_df["pert_id"].tolist(), "train")
            tr_char_ids = self._encode_char_ids(train_df["pert_id"].tolist())

            print("Preparing val set...")
            va_ids, va_pos = self._tokenize_and_get_positions(
                tokenizer, val_df["pert_id"].tolist(), "val")
            va_str_embs = self._precompute_string_embs(
                val_df["pert_id"].tolist(), "val")
            va_char_ids = self._encode_char_ids(val_df["pert_id"].tolist())

            self.train_ds = PerturbationDataset(
                train_df, tr_ids, tr_pos, tr_str_embs, tr_char_ids, is_test=False)
            self.val_ds = PerturbationDataset(
                val_df, va_ids, va_pos, va_str_embs, va_char_ids, is_test=False)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            print("Preparing test set...")
            te_ids, te_pos = self._tokenize_and_get_positions(
                tokenizer, test_df["pert_id"].tolist(), "test")
            te_str_embs = self._precompute_string_embs(
                test_df["pert_id"].tolist(), "test")
            te_char_ids = self._encode_char_ids(test_df["pert_id"].tolist())

            self.test_ds = PerturbationDataset(
                test_df, te_ids, te_pos, te_str_embs, te_char_ids, is_test=True)
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
# Model: 4-Source Fusion (LoRA AIDO + Static STRING + Char-CNN)
# ─────────────────────────────────────────────────────────────────────────────
class FourSourceFusionEncoder(nn.Module):
    """
    Architecture:
      ┌─ AIDO.Cell-10M + LoRA r=4 (all 8 Q/K/V layers) ─────────────────────┐
      │  dual pooling: [gene_pos_emb (256)] + [mean_pool (256)] = 512-dim    │
      └──────────────────────────────────────────────────────────────────────┘
                  ↓ 512-dim
      ┌─ Pre-computed STRING GNN PPI (frozen, static) ─────────────────────────┐
      │  Single forward at setup → [N, 256] stored in dataset                  │
      └──────────────────────────────────────────────────────────────────────┘
                  ↓ 256-dim (from dataset)
      ┌─ Character-level CNN on Ensembl ID ─────────────────────────────────────┐
      │  3-branch Conv1d(k=3,5,7) → AdaptiveMaxPool1d → cat → Linear(96→64)    │
      └──────────────────────────────────────────────────────────────────────┘
                  ↓ 64-dim

      fusion = cat([AIDO_dual_512, STRING_256, CharCNN_64]) = 832-dim
                  ↓
      LayerNorm(832)
      → Linear(832, head_width=384) → GELU → Dropout(head_dropout=0.4)
      → LayerNorm(384)
      → Linear(384, 3×6640)
      → [B, 3, 6640] logits
    """

    def __init__(self, head_width: int = 384, head_dropout: float = 0.4):
        super().__init__()
        self.head_width = head_width
        self.head_dropout = head_dropout

        # AIDO backbone (initialized in setup)
        self.aido_backbone: Optional[nn.Module] = None

        # Character-level CNN
        self.gene_symbol_cnn: Optional[GeneSymbolCNN] = None

        # Head (initialized in setup after knowing fusion dim)
        self.head: Optional[nn.Sequential] = None

    def initialize_backbones(self, lora_r: int = 4) -> None:
        """Load AIDO.Cell-10M, apply LoRA, and initialize CharCNN + head."""
        from transformers import AutoConfig
        # Load config first to enable Flash Attention 2 (reduces attention memory ~10x)
        config = AutoConfig.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        config._use_flash_attention_2 = True
        # Load AIDO.Cell-10M with Flash Attention enabled
        backbone = AutoModel.from_pretrained(
            AIDO_MODEL_PATH, trust_remote_code=True, config=config)
        # CRITICAL: cast to bf16 BEFORE LoRA wrapping and before any forward pass.
        # This ensures LayerNorm weights are bf16, enabling Flash Attention to be used
        # (bf16 weights → bf16 activations → use_flash_attention=True).
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        # Apply LoRA on all 8 Q/K/V layers
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,  # all 8 layers
        )
        backbone = get_peft_model(backbone, lora_cfg)
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Do NOT cast LoRA params to float32. The backbone is in bf16 for Flash Attention.
        # AdamW optimizer handles bf16 fine in mixed precision mode.

        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in backbone.parameters())
        print(f"AIDO.Cell-10M+LoRA: {trainable:,}/{total:,} trainable params")
        self.aido_backbone = backbone

        # Character-level CNN
        self.gene_symbol_cnn = GeneSymbolCNN(
            vocab_size=CHAR_VOCAB_SIZE,
            embed_dim=32,
            out_channels=32,
            cnn_out_dim=CNN_DIM,
            kernel_sizes=(3, 5, 7),
        )

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, self.head_width),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.LayerNorm(self.head_width),
            nn.Linear(self.head_width, N_CLASSES * N_GENES),
        )

        # Cast head and CNN params to float32 for stable optimization
        for k, v in self.head.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()
        for k, v in self.gene_symbol_cnn.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32
        pert_positions: torch.Tensor,  # [B] int64
        string_embs: torch.Tensor,     # [B, 256] float32 (pre-computed)
        char_ids: torch.Tensor,        # [B, ENSEMBL_MAX_LEN] int64
    ) -> torch.Tensor:
        """Returns [B, 3, 6640] logits."""
        B = input_ids.shape[0]

        # ── AIDO.Cell-10M forward ──────────────────────────────────────────
        aido_outputs = self.aido_backbone(
            input_ids=input_ids,
            output_hidden_states=False,
        )
        last_hidden = aido_outputs.last_hidden_state.float()  # [B, 19266, 256]
        gene_hidden = last_hidden[:, :19264, :]  # [B, 19264, 256]

        # Gene positional embedding: hidden state at the perturbed gene's position
        # For genes not in AIDO vocab (pert_pos=-1), use mean of all gene positions
        has_pos = pert_positions >= 0  # [B]
        gene_pos_emb = gene_hidden.mean(dim=1)  # fallback: mean-pool [B, 256]

        if has_pos.any():
            valid_pos = pert_positions.clamp(min=0)  # safe indexing
            pos_emb = gene_hidden[torch.arange(B, device=gene_hidden.device), valid_pos]
            gene_pos_emb[has_pos] = pos_emb[has_pos]

        # Mean pool over all 19264 gene positions (dual-pooling: gene_pos + mean)
        mean_pool = gene_hidden.mean(dim=1)  # [B, 256]

        # Dual-pooled AIDO features: [gene_pos_emb, mean_pool]
        aido_dual = torch.cat([gene_pos_emb, mean_pool], dim=-1)  # [B, 512]

        # ── Pre-computed STRING embeddings (already on device via batch) ──
        string_features = string_embs.float()  # [B, 256]

        # ── Character-level CNN ──────────────────────────────────────────
        cnn_features = self.gene_symbol_cnn(char_ids)  # [B, 64]

        # ── Fusion & head ─────────────────────────────────────────────────
        fusion = torch.cat([aido_dual, string_features, cnn_features], dim=-1)  # [B, 832]
        out = self.head(fusion)  # [B, 3×6640]
        return out.reshape(B, N_CLASSES, N_GENES)  # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 4,
        head_width: int = 384,
        head_dropout: float = 0.4,
        lr_backbone: float = 2e-4,
        lr_head: float = 6e-4,
        weight_decay: float = 0.03,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        lr_patience: int = 40,   # KEY CHANGE: 8 → 40 (effectively never fires, like node3-2)
        lr_factor: float = 0.5,
        output_dir: Path = Path("run"),
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lora_r = lora_r
        self.head_width = head_width
        self.head_dropout = head_dropout
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head
        self.weight_decay = weight_decay
        self.gamma_focal = gamma_focal
        self.label_smoothing = label_smoothing
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.output_dir = output_dir

        self.model: Optional[FourSourceFusionEncoder] = None
        self.loss_fn: Optional[FocalLoss] = None

        # Validation collection
        self._val_preds: List[np.ndarray] = []
        self._val_labels: List[np.ndarray] = []

        # Test collection
        self._test_preds: List[np.ndarray] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._test_sample_indices: List[int] = []

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize model components (rank-safe)."""
        if self.model is None:
            self.model = FourSourceFusionEncoder(
                head_width=self.head_width,
                head_dropout=self.head_dropout,
            )
            self.model.initialize_backbones(lora_r=self.lora_r)

        if self.loss_fn is None:
            self.loss_fn = FocalLoss(
                gamma=self.gamma_focal,
                weight=CLASS_WEIGHTS.clone(),
                label_smoothing=self.label_smoothing,
            )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, 3, G]  float32
        labels: [B, G]     int64 {0,1,2}
        """
        B, C, G = logits.shape
        # Reshape to [B*G, C] for loss computation
        logits_flat = logits.permute(0, 2, 1).reshape(B * G, C).float()
        labels_flat = labels.reshape(B * G)
        return self.loss_fn(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self.model(
            input_ids=batch["input_ids"],
            pert_positions=batch["pert_pos"],
            string_embs=batch["string_emb"],
            char_ids=batch["char_ids"],
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        with torch.no_grad():
            logits = self.model(
                input_ids=batch["input_ids"],
                pert_positions=batch["pert_pos"],
                string_embs=batch["string_emb"],
                char_ids=batch["char_ids"],
            )
            loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        probs = logits.float().softmax(dim=1)  # [B, 3, G]
        self._val_preds.append(probs.cpu().numpy())
        self._val_labels.append(batch["label"].cpu().numpy())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        local_preds = np.concatenate(self._val_preds, axis=0)   # [N_local, 3, G]
        local_labels = np.concatenate(self._val_labels, axis=0)  # [N_local, G]

        # Gather across DDP ranks via all_gather
        if self.trainer.world_size > 1:
            preds_tensor = torch.from_numpy(local_preds).to(self.device)    # [N_local, 3, G]
            labels_tensor = torch.from_numpy(local_labels).to(self.device)  # [N_local, G]
            gathered_preds = self.all_gather(preds_tensor)    # [world_size, N_local, 3, G]
            gathered_labels = self.all_gather(labels_tensor)  # [world_size, N_local, G]
            # Flatten world_size and N_local dimensions
            all_preds = gathered_preds.reshape(-1, N_CLASSES, N_GENES).cpu().numpy()
            all_labels = gathered_labels.reshape(-1, N_GENES).cpu().numpy()
        else:
            all_preds = local_preds
            all_labels = local_labels

        # Log on all ranks with sync_dist so EarlyStopping can monitor val_f1 on all ranks.
        # val_f1 is the same value (computed from gathered data) on every rank.
        val_f1 = compute_deg_f1(all_preds, all_labels)
        self.log("val_f1", val_f1, on_step=False, on_epoch=True,
                  prog_bar=True, sync_dist=True)

        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        with torch.no_grad():
            logits = self.model(
                input_ids=batch["input_ids"],
                pert_positions=batch["pert_pos"],
                string_embs=batch["string_emb"],
                char_ids=batch["char_ids"],
            )
        probs = logits.float().softmax(dim=1)  # [B, 3, G]
        self._test_preds.append(probs.cpu().numpy())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        self._test_sample_indices.extend(batch["idx"].cpu().tolist())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = np.concatenate(self._test_preds, axis=0)
        local_indices = self._test_sample_indices[:]
        local_pert_ids = self._test_pert_ids[:]
        local_symbols = self._test_symbols[:]

        # Gather across DDP ranks
        if self.trainer.world_size > 1:
            gathered_preds = self.all_gather(
                torch.from_numpy(local_preds).to(self.device)
            )
            gathered_indices = self.all_gather(
                torch.tensor(local_indices, dtype=torch.long, device=self.device)
            )
            all_preds_tensor = gathered_preds.reshape(-1, N_CLASSES, N_GENES)
            all_indices = gathered_indices.reshape(-1).cpu().tolist()
            all_preds = all_preds_tensor.cpu().numpy()
        else:
            all_preds = local_preds
            all_indices = local_indices

        if self.trainer.is_global_zero:
            # Deduplicate by sample index
            seen = set()
            unique_rows = []
            for i, idx in enumerate(all_indices):
                if idx not in seen:
                    seen.add(idx)
                    unique_rows.append(i)

            all_preds = all_preds[unique_rows]
            unique_indices = [all_indices[i] for i in unique_rows]

            # Gather metadata (all ranks same for test)
            if self.trainer.world_size > 1:
                # Reconstruct pert_ids/symbols from datamodule
                test_pert_ids = self.trainer.datamodule.test_pert_ids
                test_symbols = self.trainer.datamodule.test_symbols
                rows_pert_ids = [test_pert_ids[i] for i in unique_indices]
                rows_symbols = [test_symbols[i] for i in unique_indices]
            else:
                rows_pert_ids = [local_pert_ids[unique_rows[j]] for j in range(len(unique_rows))]
                rows_symbols = [local_symbols[unique_rows[j]] for j in range(len(unique_rows))]

            # Sort by original index for reproducibility
            order = np.argsort(unique_indices)
            all_preds = all_preds[order]
            rows_pert_ids = [rows_pert_ids[i] for i in order]
            rows_symbols = [rows_symbols[i] for i in order]

            # Save predictions
            output_path = self.output_dir / "test_predictions.tsv"
            with open(output_path, "w") as f:
                f.write("idx\tinput\tprediction\n")
                for pid, sym, pred in zip(rows_pert_ids, rows_symbols, all_preds):
                    pred_list = pred.tolist()  # [3, 6640]
                    pred_str = json.dumps(pred_list)
                    f.write(f"{pid}\t{sym}\t{pred_str}\n")
            self.print(f"Test predictions saved to {output_path} "
                       f"({len(rows_pert_ids)} rows)")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_sample_indices.clear()

    def configure_optimizers(self):
        """
        Differential learning rates: backbone (LoRA) vs head.

        KEY CHANGE vs parent: lr_patience=40 (instead of 8).
        With patience=40 and typical val_f1 oscillation ±0.005,
        the ReduceLROnPlateau scheduler effectively NEVER fires —
        matching node3-2's proven behavior (best F1=0.462 in entire tree).

        With early_stopping_patience=20, training stops ~20 epochs after peak.
        The RLROP would need patience+20 epochs of no improvement to fire,
        which won't happen because early stopping catches the peak first.
        This guarantees flat LR throughout training — the optimal regime
        for this task based on empirical evidence across all nodes.
        """
        # Separate LoRA params and head params
        lora_params = []
        head_params = []
        cnn_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "aido_backbone" in name:
                lora_params.append(param)
            elif "gene_symbol_cnn" in name:
                cnn_params.append(param)
            elif "head" in name:
                head_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": self.lr_backbone, "weight_decay": self.weight_decay},
                {"params": cnn_params + head_params, "lr": self.lr_head, "weight_decay": self.weight_decay},
            ],
            lr=self.lr_head,
        )

        # patience=40: scheduler will effectively NEVER fire during normal training
        # (early stopping with patience=20 will stop training first)
        # This matches node3-2's behavior exactly: flat LR throughout training.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=self.lr_patience,  # 40 — effectively never fires
            factor=self.lr_factor,
            min_lr=1e-6,  # slightly higher floor vs parent's 1e-7 (emergency backup)
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

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters."""
        full_state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_state_dict:
                    trainable_state_dict[key] = full_state_dict[key]
        for name, buffer in self.named_buffers():
            key = prefix + name
            if key in full_state_dict:
                trainable_state_dict[key] = full_state_dict[key]
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%), "
            f"plus {total_buffers} buffer values"
        )
        return trainable_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable parameters from a partial checkpoint."""
        full_state_keys = set(super().state_dict().keys())
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        buffer_keys = {
            name for name, _ in self.named_buffers() if name in full_state_keys
        }
        expected_keys = trainable_keys | buffer_keys

        missing_keys = [k for k in expected_keys if k not in state_dict]
        unexpected_keys = [k for k in state_dict if k not in expected_keys]

        if missing_keys:
            self.print(f"Warning: Missing checkpoint keys: {missing_keys[:5]}...")
        if unexpected_keys:
            self.print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

        loaded_trainable = len([k for k in state_dict if k in trainable_keys])
        loaded_buffers = len([k for k in state_dict if k in buffer_keys])
        self.print(
            f"Loading checkpoint: {loaded_trainable} trainable parameters and "
            f"{loaded_buffers} buffers"
        )
        return super().load_state_dict(state_dict, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Node1-2-2-3: LoRA AIDO + Static STRING + CharCNN | "
                    "ReduceLROnPlateau(patience=40) — matches node3-2 training dynamics"
    )
    # Batch size
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=64)

    # Training duration
    # max_epochs=80: best epoch in this lineage is E14-E38; with ES patience=20,
    # training stops E34-E58 at most. 80 epochs provides sufficient headroom.
    parser.add_argument("--max-epochs", type=int, default=80)
    # early-stopping-patience=20: balanced between parent's 25 (too loose) and
    # sibling's 12 (too tight). Allows sufficient recovery from local dips.
    parser.add_argument("--early-stopping-patience", type=int, default=20)

    # Model (UNCHANGED from parent)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--head-width", type=int, default=384)
    parser.add_argument("--head-dropout", type=float, default=0.4)

    # Optimizer (UNCHANGED from parent — proven optimal)
    parser.add_argument("--lr-backbone", type=float, default=2e-4)
    parser.add_argument("--lr-head", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)

    # Scheduler — KEY CHANGE: patience=40 (never fires, like node3-2)
    # vs parent's patience=8 (fired 2x causing regression)
    # vs sibling 1's CosineAnnealingLR (always reduces LR, causing regression)
    parser.add_argument("--lr-patience", type=int, default=40)
    parser.add_argument("--lr-factor", type=float, default=0.5)

    # Loss (UNCHANGED from parent — proven optimal)
    parser.add_argument("--gamma-focal", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    # Other
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--debug-max-step", type=int, default=None)
    parser.add_argument("--fast-dev-run", action="store_true")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Seed for reproducibility — seed=7 is confirmed better in node2-3 lineage
    pl.seed_everything(7)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    fast_dev_run = args.fast_dev_run
    debug_max_step = args.debug_max_step
    is_debug = fast_dev_run or debug_max_step is not None

    if is_debug:
        limit_train = 1 if fast_dev_run else debug_max_step
        limit_val = 1.0
        limit_test = 1.0
        max_steps = debug_max_step if debug_max_step is not None else -1
    else:
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        max_steps = -1

    accumulate_grad_batches = max(
        1, args.global_batch_size // (args.micro_batch_size * n_gpus)
    )

    datamodule = DEGDataModule(
        data_dir="data",
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    model = DEGLightningModule(
        lora_r=args.lora_r,
        head_width=args.head_width,
        head_dropout=args.head_dropout,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        output_dir=output_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stopping_patience,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # Only add ModelCheckpoint in production (not debug mode)
    callbacks = [early_stop_callback, lr_monitor, progress_bar]
    if not is_debug:
        callbacks.insert(0, checkpoint_callback)

    csv_logger = CSVLogger(
        save_dir=str(output_dir / "logs"), name="csv_logs"
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    if n_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=True,  # needed: frozen STRING GNN in setup, not forward
            timeout=timedelta(seconds=120),
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
        accumulate_grad_batches=accumulate_grad_batches,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=args.val_check_interval if (
            debug_max_step is None and not fast_dev_run
        ) else 1.0,
        num_sanity_val_steps=2,
        callbacks=callbacks,
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=10,
        deterministic=False,  # disabled: AdaptiveMaxPool1d backward has no deterministic CUDA impl
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    # Test on best checkpoint
    if fast_dev_run or debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Save test score
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        with open(score_path, "w") as f:
            f.write(f"Test results: {test_results}\n")
        print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
