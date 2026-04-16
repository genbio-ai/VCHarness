#!/usr/bin/env python3
"""
Node 1-2: 4-Source Fusion DEG Predictor
========================================
Architecture: AIDO.Cell-10M (LoRA r=4, all 8 QKV layers) + frozen STRING_GNN PPI
embeddings + character-level CNN on gene symbol → 832-dim fusion → 384→19920 MLP head.

Key design decisions (distinct from parent and sibling):
  Parent (node1-2): Fully frozen AIDO.Cell-10M + STRING_GNN + low-rank head (rank=64)
    → ceiling at F1=0.4101 due to frozen-backbone capacity bottleneck
  Sibling (node1-1): QKV fine-tuning on LAST 2 LAYERS ONLY + learnable STRING projection
    + reduced rank=32 head → regressed to F1=0.3933 due to backbone destabilization

  This node breaks the frozen-backbone ceiling by:
  1. LoRA r=4 on ALL 8 QKV layers (full adaptation, not just last-2)
     → learns perturbation-specific attention across the whole backbone
  2. Pre-computed frozen STRING_GNN PPI embeddings (no learnable projection)
     → avoids introducing trainable params for STRING (proven better than STRING proj)
  3. Character-level CNN on gene symbol (3-branch Conv1d, 64-dim)
     → orthogonal gene-family signal from naming conventions (NDUF, KDM prefixes etc.)
  4. 832→384→19920 single-stage MLP head (not low-rank bottleneck)
     → matching the proven tree-best architecture dimension
  5. Differential LR: backbone 2e-4, head 6e-4 (stable backbone + fast head convergence)
  6. Class weights [7.0, 1.0, 15.0] for severe imbalance (3.41% down, 95.48% unchanged,
     1.10% up) → proven in tree-best nodes (node2-2-3-1-2, F1=0.4638)
  7. ReduceLROnPlateau on val_f1 with patience=8 (proven to fire correctly)
  8. Top-3 checkpoint averaging for test inference (variance reduction)
  9. weight_decay=0.03, focal_gamma=2.0, label_smoothing=0.05
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

AIDO_MODEL_PATH = "/home/Models/AIDO.Cell-10M"   # 10M variant, 256-dim hidden
STRING_GNN_PATH = "/home/Models/STRING_GNN"

AIDO_DIM = 256          # AIDO.Cell-10M hidden dim
STRING_DIM = 256        # STRING_GNN output dim
SYMBOL_CNN_DIM = 64     # Character-level CNN output dim
DUAL_POOL_DIM = AIDO_DIM * 2     # 512 (gene_pos + mean_pool)
FUSION_DIM = DUAL_POOL_DIM + STRING_DIM + SYMBOL_CNN_DIM  # 832

# Class weights: severe imbalance handling
# Train: ~3.41% down(-1), ~95.48% unchanged(0), ~1.10% up(+1)
# Remapped: class0=down, class1=unchanged, class2=up → [7, 1, 15] proven in tree-best
CLASS_WEIGHTS = torch.tensor([7.0, 1.0, 15.0], dtype=torch.float32)

# Charset for character-level CNN (gene symbols are alphanumeric + dash)
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._"
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARSET)}  # 1-indexed (0=padding)
VOCAB_SIZE = len(CHARSET) + 1  # +1 for padding
SYMBOL_MAX_LEN = 10  # pad/truncate to this length


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
# STRING_GNN Embedding Extraction (pre-computed frozen embeddings)
# ─────────────────────────────────────────────────────────────────────────────
def extract_string_gnn_embeddings(model_path: str = STRING_GNN_PATH) -> tuple:
    """
    Run STRING_GNN once to extract frozen node embeddings.
    Returns:
        emb_matrix: [18870, 256] float32 tensor on CPU
        node_name_to_idx: dict mapping Ensembl ID → row index in emb_matrix
    """
    model_dir = Path(model_path)
    node_names = json.loads((model_dir / "node_names.json").read_text())
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

    gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    gnn_model.eval()

    graph = torch.load(str(model_dir / "graph_data.pt"), map_location="cpu")
    edge_index = graph["edge_index"]
    edge_weight = graph["edge_weight"]

    with torch.no_grad():
        outputs = gnn_model(edge_index=edge_index, edge_weight=edge_weight)

    emb_matrix = outputs.last_hidden_state.detach().cpu().float()  # [18870, 256]
    del gnn_model
    torch.cuda.empty_cache()

    return emb_matrix, node_name_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# Character-level Symbol Encoder
# ─────────────────────────────────────────────────────────────────────────────
def encode_symbol(symbol: str, max_len: int = SYMBOL_MAX_LEN) -> List[int]:
    """Encode gene symbol as character indices (1-indexed, 0=padding)."""
    chars = [CHAR_TO_IDX.get(c, 0) for c in symbol[:max_len]]
    # Pad to max_len
    chars = chars + [0] * (max_len - len(chars))
    return chars


class SymbolCNN(nn.Module):
    """
    3-branch character-level CNN over gene symbols (multi-scale).
    Input: [B, max_len] int64 character indices
    Output: [B, out_dim] float32
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 32,
                 out_dim: int = 64, max_len: int = SYMBOL_MAX_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 3 kernel sizes: 2, 3, 4 → captures bigrams, trigrams, tetragrams
        branch_dim = out_dim // 3
        self.conv2 = nn.Conv1d(embed_dim, branch_dim, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(embed_dim, branch_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, out_dim - 2 * branch_dim, kernel_size=4, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, max_len] int64 → [B, out_dim] float32"""
        e = self.embed(x).transpose(1, 2)  # [B, embed_dim, max_len]
        b2 = F.gelu(self.conv2(e))          # [B, branch_dim, max_len+1]
        b3 = F.gelu(self.conv3(e))          # [B, branch_dim, max_len]
        b4 = F.gelu(self.conv4(e))          # [B, out_dim-2*branch_dim, max_len+2]
        # Global max-pool each branch
        p2 = self.pool(b2).squeeze(-1)      # [B, branch_dim]
        p3 = self.pool(b3).squeeze(-1)      # [B, branch_dim]
        p4 = self.pool(b4).squeeze(-1)      # [B, out_dim-2*branch_dim]
        return torch.cat([p2, p3, p4], dim=-1)  # [B, out_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Dataset with pre-computed STRING_GNN embeddings + AIDO.Cell tokenized inputs."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_embs: torch.Tensor,      # [N, 256] float32 (STRING GNN embeddings per sample)
        string_valid: torch.Tensor,     # [N] bool (True if gene found in STRING)
        input_ids: torch.Tensor,        # [N, 19264] float32 (AIDO.Cell input)
        pert_positions: torch.Tensor,   # [N] int64 (-1 if gene not in AIDO vocab)
        symbol_ids: torch.Tensor,       # [N, SYMBOL_MAX_LEN] int64 (char-encoded symbol)
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.string_embs = string_embs
        self.string_valid = string_valid
        self.input_ids = input_ids
        self.pert_positions = pert_positions
        self.symbol_ids = symbol_ids
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
            "string_emb": self.string_embs[idx],        # [256] float32
            "string_valid": self.string_valid[idx],      # bool
            "input_ids": self.input_ids[idx],            # [19264] float32
            "pert_pos": self.pert_positions[idx],         # int64 (-1 if unknown)
            "symbol_ids": self.symbol_ids[idx],          # [SYMBOL_MAX_LEN] int64
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

        # Pre-computed STRING_GNN embeddings (set in setup)
        self._string_emb_matrix: Optional[torch.Tensor] = None
        self._string_node_to_idx: Optional[Dict[str, int]] = None
        self._string_null_emb: Optional[torch.Tensor] = None

    def _init_string_gnn(self) -> None:
        """Extract STRING_GNN embeddings once (rank-0 first, then all ranks load from CPU)."""
        if self._string_emb_matrix is not None:
            return

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            print("Extracting STRING_GNN embeddings (once)...")
            emb, node_to_idx = extract_string_gnn_embeddings()
            self._string_emb_matrix = emb
            self._string_node_to_idx = node_to_idx
            self._string_null_emb = emb.mean(dim=0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if local_rank != 0:
            # Other ranks also load (all from same local model path)
            emb, node_to_idx = extract_string_gnn_embeddings()
            self._string_emb_matrix = emb
            self._string_node_to_idx = node_to_idx
            self._string_null_emb = emb.mean(dim=0)

    def _init_tokenizer(self) -> AutoTokenizer:
        """Rank-safe tokenizer initialization (rank 0 first if distributed)."""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)

    def _get_string_embs(self, pert_ids: List[str]) -> tuple:
        """Look up STRING_GNN embeddings for a list of pert_ids."""
        null = self._string_null_emb
        embs = []
        valid_flags = []
        for pid in pert_ids:
            if pid in self._string_node_to_idx:
                idx = self._string_node_to_idx[pid]
                embs.append(self._string_emb_matrix[idx])
                valid_flags.append(True)
            else:
                embs.append(null)
                valid_flags.append(False)
        return torch.stack(embs, dim=0), torch.tensor(valid_flags, dtype=torch.bool)

    def _tokenize_and_get_positions(
        self,
        tokenizer: AutoTokenizer,
        pert_ids: List[str],
        split_name: str = "split",
    ) -> tuple:
        """Tokenize all pert_ids and find each gene's position in the 19264-gene vocabulary."""
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        chunk_size = 128
        all_input_ids: List[torch.Tensor] = []
        for i in range(0, len(expr_dicts), chunk_size):
            chunk = expr_dicts[i:i + chunk_size]
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

    def _encode_symbols(self, symbols: List[str]) -> torch.Tensor:
        """Encode all symbols as character index tensors."""
        ids = [encode_symbol(s) for s in symbols]
        return torch.tensor(ids, dtype=torch.long)  # [N, SYMBOL_MAX_LEN]

    def setup(self, stage: Optional[str] = None) -> None:
        # Initialize STRING_GNN embeddings
        self._init_string_gnn()
        tokenizer = self._init_tokenizer()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")

            print("Preparing train set...")
            tr_str_embs, tr_str_valid = self._get_string_embs(train_df["pert_id"].tolist())
            tr_ids, tr_pos = self._tokenize_and_get_positions(
                tokenizer, train_df["pert_id"].tolist(), "train")
            tr_sym = self._encode_symbols(train_df["symbol"].tolist())

            print("Preparing val set...")
            va_str_embs, va_str_valid = self._get_string_embs(val_df["pert_id"].tolist())
            va_ids, va_pos = self._tokenize_and_get_positions(
                tokenizer, val_df["pert_id"].tolist(), "val")
            va_sym = self._encode_symbols(val_df["symbol"].tolist())

            self.train_ds = PerturbationDataset(
                train_df, tr_str_embs, tr_str_valid, tr_ids, tr_pos, tr_sym, is_test=False)
            self.val_ds = PerturbationDataset(
                val_df, va_str_embs, va_str_valid, va_ids, va_pos, va_sym, is_test=False)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            print("Preparing test set...")
            te_str_embs, te_str_valid = self._get_string_embs(test_df["pert_id"].tolist())
            te_ids, te_pos = self._tokenize_and_get_positions(
                tokenizer, test_df["pert_id"].tolist(), "test")
            te_sym = self._encode_symbols(test_df["symbol"].tolist())

            self.test_ds = PerturbationDataset(
                test_df, te_str_embs, te_str_valid, te_ids, te_pos, te_sym, is_test=True)
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
# Model: LoRA AIDO.Cell-10M + STRING_GNN + Symbol CNN + MLP Head
# ─────────────────────────────────────────────────────────────────────────────
class LoRAFusionModel(nn.Module):
    """
    4-source DEG prediction model:
      1. AIDO.Cell-10M with LoRA r=4 on all 8 QKV layers → dual-pool [B, 512]
      2. Pre-computed frozen STRING_GNN PPI embeddings → [B, 256]
      3. Character-level CNN on gene symbol → [B, 64]
      Fusion: cat → [B, 832]
      Head: Linear(832, 384) → GELU → Dropout → Linear(384, 3*6640)
    """

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        head_hidden: int = 384,
        head_dropout: float = 0.4,
    ):
        super().__init__()
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.head_hidden = head_hidden
        self.head_dropout = head_dropout

        # Backbone (initialized in initialize_backbone)
        self.aido_backbone: Optional[nn.Module] = None

        # Symbol CNN (always trainable)
        self.symbol_cnn = SymbolCNN(vocab_size=VOCAB_SIZE, embed_dim=32,
                                    out_dim=SYMBOL_CNN_DIM, max_len=SYMBOL_MAX_LEN)

        # Learnable fallback embedding for genes not in STRING_GNN vocabulary
        self.string_null_emb = nn.Parameter(torch.zeros(STRING_DIM))

        # MLP head (always trainable)
        self.head: Optional[nn.Sequential] = None

    def initialize_backbone(self) -> None:
        """Load AIDO.Cell-10M with LoRA r=4 on all 8 transformer layers."""
        base_model = AutoModel.from_pretrained(
            AIDO_MODEL_PATH, trust_remote_code=True)
        base_model = base_model.to(torch.bfloat16)
        base_model.config.use_cache = False

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,  # all 8 layers
            bias="none",
        )
        self.aido_backbone = get_peft_model(base_model, lora_cfg)
        self.aido_backbone.config.use_cache = False
        # Enable gradient checkpointing to save activation memory
        self.aido_backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        trainable = sum(p.numel() for p in self.aido_backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.aido_backbone.parameters())
        print(f"AIDO.Cell-10M+LoRA: {trainable:,}/{total:,} trainable params "
              f"({100.0 * trainable / total:.2f}%)")

    def initialize_head(self) -> None:
        """Create MLP output head."""
        self.head = nn.Sequential(
            nn.Linear(FUSION_DIM, self.head_hidden),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.head_hidden, N_CLASSES * N_GENES),
        )
        # Truncated-normal init for stable early training
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_aido_dual_pool(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32
        pert_positions: torch.Tensor,  # [B] int64 (-1 for unknown)
    ) -> torch.Tensor:
        """
        Run AIDO.Cell-10M (LoRA adapted) and return dual-pooled representation.
        Returns: [B, 512] float32 (gene_pos_emb ++ mean_pool)
        """
        backbone_device = next(self.aido_backbone.parameters()).device
        input_ids_dev = input_ids.to(backbone_device)
        attn_mask = torch.ones(input_ids_dev.shape[0], input_ids_dev.shape[1],
                               dtype=torch.long, device=backbone_device)
        # Forward through LoRA-adapted backbone (gradients flow through LoRA params)
        out = self.aido_backbone(
            input_ids=input_ids_dev,
            attention_mask=attn_mask,
        )
        hidden = out.last_hidden_state  # [B, 19266, 256] bfloat16

        # Global mean-pool over gene positions (exclude the 2 summary tokens)
        mean_pool = hidden[:, :19264, :].mean(dim=1).float()  # [B, 256]

        # Per-gene positional extraction
        B = hidden.size(0)
        hidden_device = hidden.device
        pert_positions_dev = pert_positions.to(hidden_device)
        valid = pert_positions_dev >= 0
        safe_pos = pert_positions_dev.clamp(min=0)

        gene_emb_raw = hidden[
            torch.arange(B, device=hidden_device), safe_pos, :
        ].float()  # [B, 256]

        # Differentiable fallback to mean_pool for unknown genes
        valid_f = valid.float().unsqueeze(-1)
        gene_emb = gene_emb_raw * valid_f + mean_pool * (1.0 - valid_f)  # [B, 256]

        return torch.cat([gene_emb, mean_pool], dim=-1)  # [B, 512]

    def forward(
        self,
        string_emb: torch.Tensor,      # [B, 256] float32 (pre-computed, STRING_GNN)
        string_valid: torch.Tensor,    # [B] bool (True if gene in STRING_GNN)
        input_ids: torch.Tensor,       # [B, 19264] float32 (AIDO.Cell input)
        pert_positions: torch.Tensor,  # [B] int64 (-1 for unknown)
        symbol_ids: torch.Tensor,      # [B, SYMBOL_MAX_LEN] int64
    ) -> torch.Tensor:
        """Returns: [B, 3, N_GENES] logits."""
        B = string_emb.size(0)

        # ── STRING_GNN feature ───────────────────────────────────────────────
        valid_f = string_valid.float().unsqueeze(-1)  # [B, 1]
        null_exp = self.string_null_emb.float().unsqueeze(0).expand(B, -1)  # [B, 256]
        str_feat = string_emb.float() * valid_f + null_exp * (1.0 - valid_f)  # [B, 256]

        # ── AIDO.Cell-10M dual-pool feature (LoRA adapted) ──────────────────
        aido_feat = self._get_aido_dual_pool(input_ids, pert_positions)  # [B, 512]
        aido_feat = aido_feat.to(str_feat.device)

        # ── Symbol CNN feature ───────────────────────────────────────────────
        sym_feat = self.symbol_cnn(symbol_ids.to(str_feat.device))  # [B, 64]

        # ── Fusion ───────────────────────────────────────────────────────────
        combined = torch.cat([aido_feat, str_feat, sym_feat], dim=-1)  # [B, 832]

        # ── MLP head ─────────────────────────────────────────────────────────
        logits = self.head(combined)                   # [B, 3*6640]
        return logits.view(B, N_CLASSES, N_GENES)      # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        head_hidden: int = 384,
        head_dropout: float = 0.4,
        lr_backbone: float = 2e-4,
        lr_head: float = 6e-4,
        weight_decay: float = 3e-2,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 100,
        plateau_patience: int = 8,
        plateau_factor: float = 0.5,
        top_k_ckpts: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialized in setup()
        self.model: Optional[LoRAFusionModel] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators (cleared each epoch)
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._test_labels: Optional[torch.Tensor] = None
        self._test_f1_computed: Optional[float] = None

        # For top-k checkpoint averaging
        self._top_k_ckpts = top_k_ckpts

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = LoRAFusionModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                head_hidden=self.hparams.head_hidden,
                head_dropout=self.hparams.head_dropout,
            )
            self.model.initialize_backbone()
            self.model.initialize_head()

            # Cast trainable parameters to float32 for stable optimization
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()

            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

            # Print trainable parameter count
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            self.print(f"Total trainable params: {trainable:,} / {total:,} "
                       f"({100.0 * trainable / max(total, 1):.2f}%)")
            self.print(f"Trainable params per training sample: "
                       f"{trainable / 1500:.0f}")

        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols
            # Load test ground-truth labels for metric computation
            test_df = pd.read_csv(
                Path(self.trainer.datamodule.data_dir) / "test.tsv", sep="\t")
            raw_test_labels = [json.loads(x) for x in test_df["label"].tolist()]
            test_labels_arr = np.array(raw_test_labels, dtype=np.int8) + 1
            self._test_labels = torch.from_numpy(test_labels_arr).long()

    def forward(
        self,
        string_emb: torch.Tensor,
        string_valid: torch.Tensor,
        input_ids: torch.Tensor,
        pert_positions: torch.Tensor,
        symbol_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(string_emb, string_valid, input_ids, pert_positions, symbol_ids)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G], labels: [B, G] ({0,1,2}) → scalar loss."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        labels_flat = labels.reshape(-1)                       # [B*G]
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["string_emb"], batch["string_valid"],
            batch["input_ids"], batch["pert_pos"], batch["symbol_ids"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(
            batch["string_emb"], batch["string_valid"],
            batch["input_ids"], batch["pert_pos"], batch["symbol_ids"])
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
        local_preds = torch.cat(self._val_preds, dim=0)
        local_labels = torch.cat(self._val_labels, dim=0)
        local_idx = torch.cat(self._val_indices, dim=0)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        world_size = self.trainer.world_size if self.trainer.world_size else 1
        if world_size > 1:
            all_preds = self.all_gather(local_preds)
            all_labels = self.all_gather(local_labels)
            all_idx = self.all_gather(local_idx)

            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            labels_flat = all_labels.view(-1, N_GENES).cpu().numpy()
            idx_flat = all_idx.view(-1).cpu().numpy()

            unique_pos = np.unique(idx_flat, return_index=True)[1]
            preds_flat = preds_flat[unique_pos]
            labels_flat = labels_flat[unique_pos]
            order = np.argsort(idx_flat[unique_pos])
            preds_flat = preds_flat[order]
            labels_flat = labels_flat[order]
            f1 = compute_deg_f1(preds_flat, labels_flat)
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        else:
            f1 = compute_deg_f1(local_preds.numpy(), local_labels.numpy())
            self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(
            batch["string_emb"], batch["string_valid"],
            batch["input_ids"], batch["pert_pos"], batch["symbol_ids"])
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

            unique_pos = np.unique(idxs, return_index=True)[1]
            preds = preds[unique_pos]
            sorted_idxs = idxs[unique_pos]

            order = np.argsort(sorted_idxs)
            preds = preds[order]
            final_idxs = sorted_idxs[order]

            # Compute test F1 if ground-truth labels are available
            test_f1 = None
            if hasattr(self, "_test_labels") and self._test_labels is not None:
                gt_labels = self._test_labels.numpy()
                if len(preds) == len(gt_labels):
                    test_f1 = compute_deg_f1(preds, gt_labels)
                    self.log("test_f1", test_f1, prog_bar=True, sync_dist=False)
                    self.print(f"Test F1: {test_f1:.4f}")

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
            self._test_f1_computed = test_f1

    def configure_optimizers(self):
        # Differential learning rates: backbone LoRA params vs head params
        backbone_params = [p for n, p in self.model.named_parameters()
                           if p.requires_grad and "aido_backbone" in n]
        head_and_other_params = [p for n, p in self.model.named_parameters()
                                  if p.requires_grad and "aido_backbone" not in n]

        param_groups = [
            {"params": backbone_params,       "lr": self.hparams.lr_backbone},
            {"params": head_and_other_params,  "lr": self.hparams.lr_head},
        ]
        opt = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",  # monitoring val_f1 (higher is better)
            factor=self.hparams.plateau_factor,
            patience=self.hparams.plateau_patience,
            verbose=True,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1,
            },
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
# Top-K Checkpoint Averaging for test inference
# ─────────────────────────────────────────────────────────────────────────────
def average_top_k_checkpoints(
    model_module: DEGLightningModule,
    checkpoint_cb: ModelCheckpoint,
    k: int = 3,
) -> DEGLightningModule:
    """
    Average the top-k checkpoints by loading each and computing a parameter mean.
    Returns the model with averaged weights loaded.
    """
    # Collect top-k best checkpoint paths
    best_k_paths = checkpoint_cb.best_k_models
    if not best_k_paths or len(best_k_paths) == 0:
        print("No top-k checkpoints found, using current model weights.")
        return model_module

    # Sort by score (descending for val_f1 mode=max)
    sorted_paths = sorted(best_k_paths.items(), key=lambda x: x[1], reverse=True)
    top_k = sorted_paths[:k]
    if len(top_k) == 0:
        print("No valid top-k checkpoints, skipping averaging.")
        return model_module

    print(f"Averaging {len(top_k)} checkpoints:")
    for path, score in top_k:
        print(f"  {path}: val_f1={score:.4f}")

    # Load and average weights
    # Load first checkpoint to initialize the average
    loaded_states = []
    for path, _ in top_k:
        try:
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            # Cast all values to float32 on CPU for averaging
            state_f32 = {k: v.float().cpu() for k, v in state.items()}
            loaded_states.append(state_f32)
        except Exception as e:
            print(f"  Warning: failed to load {path}: {e}")

    if len(loaded_states) == 0:
        print("All checkpoint loads failed, using current model weights.")
        return model_module

    # Compute averaged state dict
    avg_state = {}
    for key in loaded_states[0]:
        try:
            tensors = [s[key] for s in loaded_states if key in s]
            if len(tensors) > 0:
                avg_state[key] = torch.stack(tensors, dim=0).mean(dim=0)
        except Exception as e:
            print(f"  Warning: failed to average key {key}: {e}")
            avg_state[key] = loaded_states[0][key]

    model_module.load_state_dict(avg_state)
    print(f"Successfully averaged {len(loaded_states)} checkpoints.")
    return model_module


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="4-Source LoRA Fusion DEG predictor (AIDO.Cell-10M + STRING + Symbol CNN)"
    )
    p.add_argument("--data-dir",               type=str,   default="data")
    p.add_argument("--micro-batch-size",        type=int,   default=8)
    p.add_argument("--global-batch-size",       type=int,   default=64)
    p.add_argument("--max-epochs",              type=int,   default=100)
    p.add_argument("--lr-backbone",             type=float, default=2e-4)
    p.add_argument("--lr-head",                 type=float, default=6e-4)
    p.add_argument("--weight-decay",            type=float, default=3e-2)
    p.add_argument("--lora-r",                  type=int,   default=4)
    p.add_argument("--lora-alpha",              type=int,   default=8)
    p.add_argument("--lora-dropout",            type=float, default=0.05)
    p.add_argument("--head-hidden",             type=int,   default=384)
    p.add_argument("--head-dropout",            type=float, default=0.4)
    p.add_argument("--gamma-focal",             type=float, default=2.0)
    p.add_argument("--label-smoothing",         type=float, default=0.05)
    p.add_argument("--plateau-patience",        type=int,   default=8)
    p.add_argument("--plateau-factor",          type=float, default=0.5)
    p.add_argument("--early-stopping-patience", type=int,   default=25)
    p.add_argument("--top-k-ckpts",             type=int,   default=3)
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
            find_unused_parameters=True,  # LoRA may have unused params
            timeout=timedelta(seconds=300),
        )

    # ── Callbacks ────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=args.top_k_ckpts,
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
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        top_k_ckpts=args.top_k_ckpts,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Test with checkpoint averaging ───────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        # Debug mode: test with current weights
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        # Production mode: average top-k checkpoints, then test
        if args.top_k_ckpts > 1:
            model_module = average_top_k_checkpoints(
                model_module, checkpoint_cb, k=args.top_k_ckpts)
            # Test with averaged weights (no ckpt_path since we manually loaded)
            test_results = trainer.test(model_module, datamodule=datamodule)
        else:
            test_results = trainer.test(
                model_module, datamodule=datamodule, ckpt_path="best")

    # ── Save test score ───────────────────────────────────────────────────────
    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        test_f1 = getattr(model_module, "_test_f1_computed", None)
        best_score = checkpoint_cb.best_model_score
        if test_f1 is not None:
            val_f1_str = f"{best_score:.6f}" if best_score is not None else "N/A"
            score_path.write_text(
                f"test_f1: {test_f1:.6f}\n"
                f"val_f1_best: {val_f1_str}\n"
            )
            print(f"Test score saved → {score_path}")
        elif test_results and test_results[0]:
            score_path.write_text(f"test_results: {test_results}\n")
            print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
