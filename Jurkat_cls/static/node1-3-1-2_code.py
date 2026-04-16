#!/usr/bin/env python3
"""
Node 1-3-1-2: LoRA AIDO.Cell-10M + Cross-Attention Fusion + STRING PPI + Char CNN
===================================================================================
Strategy:
  - AIDO.Cell-10M with LoRA fine-tuning (r=4, all 8 QKV layers) — breaks frozen-backbone ceiling
  - Pre-computed STRING PPI node embeddings (256-dim) for each perturbed gene
  - Character-level multi-scale CNN on gene symbol (64-dim)
  - 3-layer TransformerEncoder cross-attention fusion over 4 feature tokens → mean-pool
  - Single-stage MLP head (256→3×6640)
  - ReduceLROnPlateau on val_loss (reactive to overfitting onset)
  - Manifold mixup (alpha=0.3) for regularization
  - Aggressive class weights [6.0, 1.0, 12.0] to handle severe imbalance
  - ModelCheckpoint + EarlyStopping both monitor val_f1 (mode=max)

Key improvements over parent (node1-3-1, test F1=0.4420):
  1. LoRA backbone (r=4, all 8 layers) → breaks frozen-backbone ceiling at 0.44 F1
  2. Cross-attention transformer fusion → models inter-source interactions (PPI modulates
     cell-state, gene names inform positional attention)
  3. ReduceLROnPlateau on val_loss → reactive to overfitting, fires at optimal moment
  4. Manifold mixup (alpha=0.3) → proven regularization for 1,500-sample regime
  5. Weight decay=0.10 → proven sweet spot for cross-attention architecture
  6. Stronger class weights [6,1,12] → better minority class sensitivity

Architecture:
  AIDO.Cell-10M (LoRA r=4, all 8 QKV layers):
    Input: synthetic expression (perturbed gene=1.0) → dual-pool [gene_pos(256) + mean(256)] = [512]
  STRING GNN (pre-computed frozen):
    Lookup: Ensembl ID → [256]
  Char CNN:
    symbol → [64]

  4-token cross-attention fusion:
    tokens: [global_emb[256], pert_emb[256], sym_proj[256], ppi_feat[256]]
    3-layer TransformerEncoder (nhead=8, dim_ff=256, dropout=0.1)
    mean-pool → [256]

  Head: LayerNorm(256) → Linear(256→256) → GELU → Dropout(0.4) → Linear(256→3×6640)
  Output: reshape [B, 3, 6640]

Total trainable: ~8.1M (LoRA ~18K + fusion ~2M + head ~5.1M + char CNN ~10K)
Expected test F1: 0.46-0.50 (targeting cross-attention breakthrough seen in node3-1-3-1-1-1-1=0.4768)
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
N_GENES = 6_640
N_CLASSES = 3
AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

# Class weights: aggressive to handle severe imbalance (~95.5% class 0, ~3.4% class -1, ~1.1% class +1)
# Proven effective by node3-1-3-1-1-1-1 (0.4768) and node2-2-3-1-1-1-1 (0.4655)
CLASS_WEIGHTS = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)

CHAR_VOCAB = {c: i + 1 for i, c in enumerate(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.abcdefghijklmnopqrstuvwxyz"
)}
CHAR_VOCAB["<pad>"] = 0
MAX_SYM_LEN = 16

AIDO_HIDDEN_DIM = 256    # AIDO.Cell-10M hidden dimension per token
STRING_HIDDEN_DIM = 256  # STRING GNN output dimension
CHAR_CNN_DIM = 64        # SymbolCNN output dimension
FUSION_TOKEN_DIM = 256   # Dimension for each of the 4 fusion tokens


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weighting and label smoothing."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: [N, C], targets: [N] int64"""
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=w,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def encode_symbol(symbol: str, max_len: int = MAX_SYM_LEN) -> torch.Tensor:
    """Encode gene symbol as padded integer sequence for CNN."""
    sym = symbol.upper()[:max_len]
    ids = [CHAR_VOCAB.get(c, 0) for c in sym]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """
    Compute macro-averaged per-gene F1, mirroring calc_metric.py.
    y_pred: [N, 3, G] float (probabilities/logits)
    y_true_remapped: [N, G] int (0/1/2 after +1 remap)
    """
    y_hat = y_pred.argmax(axis=1)  # [N, G]
    f1_vals: List[float] = []
    for g in range(y_true_remapped.shape[1]):
        yt = y_true_remapped[:, g]
        yp = y_hat[:, g]
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


def apply_manifold_mixup(
    features: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.3,
) -> tuple:
    """
    Apply mixup augmentation in feature space (manifold mixup).
    Returns mixed features, and mixed one-hot labels.
    features: [B, D], labels: [B, G] int
    """
    if alpha > 0.0 and features.requires_grad:
        # Only apply during training
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = features.size(0)
    index = torch.randperm(batch_size, device=features.device)

    mixed_features = lam * features + (1 - lam) * features[index]
    labels_a, labels_b = labels, labels[index]

    return mixed_features, labels_a, labels_b, lam


# ──────────────────────────────────────────────────────────────────────────────
# Feature Pre-computation (STRING only, AIDO with LoRA computed online)
# ──────────────────────────────────────────────────────────────────────────────
def precompute_string_features(data_dir: Path, cache_dir: Path) -> Dict[str, np.ndarray]:
    """
    Pre-compute frozen STRING PPI features for all data splits.
    Only STRING features are cached (AIDO features computed online with LoRA).
    DDP-safe: caller must ensure only rank 0 calls this before a distributed barrier.
    """
    cache_path = cache_dir / "string_cache.npz"
    if cache_path.exists():
        print(f"[PreCompute] Loading cached STRING features from {cache_path}")
        return dict(np.load(str(cache_path), allow_pickle=True))

    print("[PreCompute] Computing STRING features (this runs once)...")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ── Load STRING GNN (frozen, float32) ─────────────────────────────────────
    string_model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
    string_model = string_model.eval().to(device)

    graph = torch.load(str(STRING_GNN_DIR / "graph_data.pt"), map_location=device)
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    string_name_to_idx = {n: i for i, n in enumerate(node_names)}

    edge_index = graph["edge_index"]
    edge_weight = graph.get("edge_weight")
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    # Run STRING GNN once to get all node embeddings [18870, 256]
    with torch.no_grad():
        string_outputs = string_model(edge_index=edge_index, edge_weight=edge_weight)
    all_string_embs = string_outputs.last_hidden_state.float().cpu().numpy()  # [18870, 256]
    print(f"[PreCompute] STRING GNN: {all_string_embs.shape[0]} node embeddings extracted")

    # ── Process each data split ────────────────────────────────────────────────
    results: Dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        tsv_path = data_dir / f"{split}.tsv"
        if not tsv_path.exists():
            continue
        df = pd.read_csv(tsv_path, sep="\t")
        pert_ids: List[str] = df["pert_id"].tolist()
        N = len(pert_ids)
        print(f"[PreCompute] Processing '{split}' split: {N} samples")

        # ── STRING PPI features [N, 256] ───────────────────────────────────────
        # Zero vector for genes not in STRING (will be replaced by learned embedding in model)
        string_feats = np.zeros((N, STRING_HIDDEN_DIM), dtype=np.float32)
        string_found = 0
        for j, pid in enumerate(pert_ids):
            eid = pid.split(".")[0]
            idx = string_name_to_idx.get(eid)
            if idx is not None:
                string_feats[j] = all_string_embs[idx]
                string_found += 1
        print(f"    STRING coverage: {string_found}/{N} ({100*string_found/N:.1f}%) genes found")
        results[f"{split}_string"] = string_feats.astype(np.float32)

    # ── Save cache ────────────────────────────────────────────────────────────
    np.savez(str(cache_path), **results)
    print(f"[PreCompute] STRING feature cache saved → {cache_path}")

    # Free GPU memory before training begins
    del string_model
    torch.cuda.empty_cache()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# AIDO Tokenizer & Input Preparation
# ──────────────────────────────────────────────────────────────────────────────
def load_aido_tokenizer():
    """Load AIDO tokenizer with proper DDP rank-0-first pattern."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    return AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataset(Dataset):
    """DEG dataset with pre-computed STRING features and online AIDO tokenization."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_feats: np.ndarray,  # [N, 256] float32
        tokenizer,
        gene_id_to_pos: Dict[str, int],
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.string_feats = torch.from_numpy(string_feats.astype(np.float32))  # [N, 256]
        self.tokenizer = tokenizer
        self.gene_id_to_pos = gene_id_to_pos

        # Always load labels if available (needed for test metric computation)
        if "label" in df.columns:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Remap: {-1→0, 0→1, 1→2}
            self.labels = torch.tensor(
                np.array(raw_labels, dtype=np.int8) + 1, dtype=torch.long
            )  # [N, 6640]
        else:
            self.labels = None

        # Pre-tokenize all samples for efficiency
        print(f"    Tokenizing {len(self.pert_ids)} samples...")
        self._tokenized = self._pretokenize_all()

    def _pretokenize_all(self):
        """Pre-tokenize all samples to avoid tokenizer overhead during training."""
        input_ids_list = []
        attention_mask_list = []
        gene_positions = []

        batch_size = 64
        for start in range(0, len(self.pert_ids), batch_size):
            batch_pids = self.pert_ids[start:start + batch_size]
            batch_data = [
                {"gene_ids": [pid.split(".")[0]], "expression": [1.0]}
                for pid in batch_pids
            ]
            batch_inputs = self.tokenizer(batch_data, return_tensors="pt")
            input_ids_list.append(batch_inputs["input_ids"])
            attention_mask_list.append(batch_inputs["attention_mask"])

            for pid in batch_pids:
                eid = pid.split(".")[0]
                pos = self.gene_id_to_pos.get(eid)
                if pos is not None:
                    gene_positions.append(max(0, min(int(pos), 19263)))
                else:
                    gene_positions.append(-1)  # -1 → use mean pool as fallback

        return {
            "input_ids": torch.cat(input_ids_list, dim=0),          # [N, 19264+]
            "attention_mask": torch.cat(attention_mask_list, dim=0), # [N, 19264+]
            "gene_positions": torch.tensor(gene_positions, dtype=torch.long),  # [N]
        }

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "input_ids": self._tokenized["input_ids"][idx],          # [seq_len]
            "attention_mask": self._tokenized["attention_mask"][idx], # [seq_len]
            "gene_pos": self._tokenized["gene_positions"][idx],       # scalar
            "string_feats": self.string_feats[idx],                   # [256]
            "sym_ids": encode_symbol(self.symbols[idx]),              # [MAX_SYM_LEN]
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]  # [6640]
        return item


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate preserving pert_id as list of strings."""
    result: Dict[str, Any] = {}
    for key in batch[0]:
        if key == "pert_id":
            result[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], int):
            result[key] = torch.tensor([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result


class DEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        micro_batch_size: int = 8,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

        self.train_ds: Optional[DEGDataset] = None
        self.val_ds: Optional[DEGDataset] = None
        self.test_ds: Optional[DEGDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []
        self._string_cache: Optional[Dict[str, np.ndarray]] = None
        self._tokenizer = None
        self._gene_id_to_pos: Dict[str, int] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        cache_path = self.cache_dir / "string_cache.npz"

        # Rank 0 computes STRING features if cache doesn't exist; other ranks wait at barrier
        if local_rank == 0 and not cache_path.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            precompute_string_features(self.data_dir, self.cache_dir)

        # DDP barrier: ensure rank 0 finishes before other ranks try to load
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # All ranks load from cache (guaranteed to exist after barrier)
        if self._string_cache is None:
            self._string_cache = dict(np.load(str(cache_path), allow_pickle=True))

        # Load tokenizer (DDP-safe)
        if self._tokenizer is None:
            self._tokenizer = load_aido_tokenizer()
            if hasattr(self._tokenizer, "gene_id_to_index"):
                self._gene_id_to_pos = {k: int(v) for k, v in self._tokenizer.gene_id_to_index.items()}
            elif hasattr(self._tokenizer, "gene_to_index"):
                self._gene_id_to_pos = {k: int(v) for k, v in self._tokenizer.gene_to_index.items()}

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = DEGDataset(
                train_df,
                self._string_cache["train_string"],
                self._tokenizer,
                self._gene_id_to_pos,
            )
            self.val_ds = DEGDataset(
                val_df,
                self._string_cache["val_string"],
                self._tokenizer,
                self._gene_id_to_pos,
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = DEGDataset(
                test_df,
                self._string_cache["test_string"],
                self._tokenizer,
                self._gene_id_to_pos,
            )
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Model Components
# ──────────────────────────────────────────────────────────────────────────────
class SymbolCNN(nn.Module):
    """Character-level multi-scale CNN for gene symbol encoding."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, out_dim, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(out_dim * 2)
        self.out_dim = out_dim * 2  # 64

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, L] → [B, out_dim*2]"""
        x = self.embedding(ids).transpose(1, 2)       # [B, embed_dim, L]
        f3 = F.gelu(self.conv3(x)).max(dim=-1).values  # [B, out_dim]
        f5 = F.gelu(self.conv5(x)).max(dim=-1).values  # [B, out_dim]
        return self.norm(torch.cat([f3, f5], dim=-1))  # [B, 64]


class LoRALinear(nn.Module):
    """
    LoRA adapter for a linear layer.

    Exposes .weight and .bias as property proxies so that the AIDO
    attention layer (which accesses self.self.query.weight.dtype) can
    read the underlying linear's dtype without modification.
    """

    def __init__(self, linear: nn.Linear, r: int = 4, lora_alpha: float = 1.0,
                 lora_dropout: float = 0.1):
        super().__init__()
        self.linear = linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(lora_dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False

    # ── Property proxies for AIDO attention layer compatibility ─────────────────
    @property
    def weight(self) -> nn.Parameter:
        """Proxy to underlying linear weight (needed by AIDO attention code)."""
        return self.linear.weight

    @property
    def bias(self) -> Optional[nn.Parameter]:
        """Proxy to underlying linear bias (needed by AIDO attention code)."""
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def apply_lora_to_model(model, r: int = 4, lora_alpha: float = 4.0,
                         lora_dropout: float = 0.1, target_modules=("query", "key", "value")):
    """
    Apply LoRA adapters to all target QKV projection layers in AIDO model.
    Freezes all non-LoRA parameters.
    """
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    lora_count = 0
    # Find and replace QKV linear layers with LoRA variants
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Get parent module and attribute name
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr_name = parts[-1]
                lora_module = LoRALinear(module, r=r, lora_alpha=lora_alpha,
                                          lora_dropout=lora_dropout)
                setattr(parent, attr_name, lora_module)
                lora_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Applied {lora_count} LoRA adapters. Trainable: {trainable:,} / {total:,} params")
    return model


class CrossAttentionFusion(nn.Module):
    """
    3-layer TransformerEncoder cross-attention fusion for 4 feature tokens.
    Tokens: [global_emb, pert_emb, sym_proj, ppi_feat], each [B, 256]
    Output: mean-pool across 4 tokens → [B, 256]

    Architecture proven by node3-1-3-1-1-1-1 (test F1=0.4768) with dim_ff=256.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 256,
        n_layers: int = 3,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

        # Learned positional embeddings for the 4 tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, 4, d_model]
        Returns: [B, d_model]
        """
        tokens = tokens + self.pos_embed  # Add positional embeddings
        out = self.transformer(tokens)    # [B, 4, d_model]
        out = self.out_norm(out)
        return out.mean(dim=1)            # [B, d_model] — mean-pool across tokens


class DEGModel(nn.Module):
    """
    DEG predictor with LoRA AIDO + cross-attention fusion of 4 feature streams.

    Input:
        input_ids:    [B, seq_len] — AIDO tokenized input
        attention_mask: [B, seq_len]
        gene_pos:     [B] — perturbed gene positions in AIDO sequence
        string_feats: [B, 256] — frozen STRING PPI node embedding
        sym_ids:      [B, 16] — character-encoded gene symbol

    Output:
        logits: [B, 3, 6640]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.4,
        lora_r: int = 4,
        lora_alpha: float = 4.0,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        # Character-level CNN for gene symbol
        self.symbol_cnn = SymbolCNN(vocab_size=len(CHAR_VOCAB), embed_dim=32, out_dim=32)
        # Project char CNN output to fusion token dim
        self.sym_proj = nn.Linear(CHAR_CNN_DIM, FUSION_TOKEN_DIM)

        # Learned embedding for genes absent from STRING (replaces zero vector)
        self.string_missing_emb = nn.Parameter(torch.zeros(STRING_HIDDEN_DIM))
        # Project STRING to fusion token dim
        self.ppi_proj = nn.Linear(STRING_HIDDEN_DIM, FUSION_TOKEN_DIM)

        # Project AIDO dual-pool features to fusion token dimensions
        self.global_proj = nn.Linear(AIDO_HIDDEN_DIM, FUSION_TOKEN_DIM)  # from mean-pool [256]
        self.pert_proj = nn.Linear(AIDO_HIDDEN_DIM, FUSION_TOKEN_DIM)    # from gene-pos [256]

        # 3-layer cross-attention fusion
        self.fusion = CrossAttentionFusion(
            d_model=FUSION_TOKEN_DIM,
            nhead=8,
            dim_ff=256,
            n_layers=3,
            attn_dropout=0.1,
            dropout=0.1,
        )

        # Prediction head: 256 → 256 → 3×6640
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_TOKEN_DIM),
            nn.Linear(FUSION_TOKEN_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N_CLASSES * N_GENES),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, LoRALinear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.string_missing_emb, mean=0.0, std=0.01)

    def extract_aido_features(
        self,
        backbone,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_pos: torch.Tensor,
    ) -> tuple:
        """
        Extract dual-pool features from AIDO backbone.
        Returns: (global_emb [B, 256], pert_emb [B, 256])
        """
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state.float()  # [B, seq_len, 256]
        gene_hidden = last_hidden[:, :19264, :]          # [B, 19264, 256]

        # Global mean-pool
        global_emb = gene_hidden.mean(dim=1)  # [B, 256]

        # Gene-specific positional embedding
        pert_embs = []
        for j in range(input_ids.shape[0]):
            pos = gene_pos[j].item()
            if pos >= 0:
                pert_embs.append(gene_hidden[j, pos, :])
            else:
                pert_embs.append(global_emb[j])
        pert_emb = torch.stack(pert_embs, dim=0)  # [B, 256]

        return global_emb, pert_emb

    def forward(
        self,
        backbone,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_pos: torch.Tensor,
        string_feats: torch.Tensor,
        sym_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Extract AIDO dual-pool features
        global_emb, pert_emb = self.extract_aido_features(
            backbone, input_ids, attention_mask, gene_pos
        )

        # Handle STRING missing genes (zero vectors → learned embedding)
        string_is_zero = (string_feats.abs().sum(dim=-1, keepdim=True) == 0)
        missing_emb = self.string_missing_emb.unsqueeze(0).expand(string_feats.shape[0], -1)
        string_feats = torch.where(string_is_zero, missing_emb.to(string_feats.dtype), string_feats)

        # Symbol CNN encoding
        sym_feats = self.symbol_cnn(sym_ids)  # [B, 64]

        # Project each stream to fusion token dimension
        t_global = self.global_proj(global_emb.float())    # [B, 256]
        t_pert = self.pert_proj(pert_emb.float())          # [B, 256]
        t_sym = self.sym_proj(sym_feats.float())           # [B, 256]
        t_ppi = self.ppi_proj(string_feats.float())        # [B, 256]

        # Stack into 4-token sequence for cross-attention fusion
        tokens = torch.stack([t_global, t_pert, t_sym, t_ppi], dim=1)  # [B, 4, 256]

        # Cross-attention fusion → [B, 256]
        fused = self.fusion(tokens)

        # Prediction
        logits = self.head(fused)  # [B, 3*6640]
        return logits.view(-1, N_CLASSES, N_GENES)  # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.4,
        lora_r: int = 4,
        lora_alpha: float = 4.0,
        lora_dropout: float = 0.1,
        backbone_lr: float = 2e-4,
        head_lr: float = 6e-4,
        weight_decay: float = 0.10,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        plateau_patience: int = 5,
        plateau_factor: float = 0.5,
        mixup_alpha: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = None
        self.model: Optional[DEGModel] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators for val/test steps
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.backbone is None:
            # Load AIDO.Cell-10M backbone in bf16 to activate FlashAttention
            # FlashAttention requires bf16/fp16 dtype (not float32)
            self.backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
            self.backbone = self.backbone.bfloat16()  # bf16 → enables FlashAttention

            # Apply LoRA to all 8 layers (QKV projections)
            self.backbone = apply_lora_to_model(
                self.backbone,
                r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                target_modules=("query", "key", "value"),
            )

            # Disable KV cache (saves memory during training)
            self.backbone.config.use_cache = False
            # Enable gradient checkpointing to reduce activation memory
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

            # Cast trainable LoRA params to bf16 to match backbone dtype
            # (enables FlashAttention while keeping LoRA params trainable)
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    param.data = param.data.bfloat16()
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        if self.model is None:
            self.model = DEGModel(
                hidden_dim=self.hparams.hidden_dim,
                dropout=self.hparams.dropout,
                lora_r=self.hparams.lora_r,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )
            # Cast all trainable model params to float32
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        # Grab test metadata from DataModule — runs on ALL ranks so both ranks
        # have the pert_ids/symbols for on_test_epoch_end indexing (fixes NCCL
        # hang when checkpoint is loaded and setup("test") is not re-run).
        if (self.trainer is not None and hasattr(self.trainer, "datamodule")
                and self.trainer.datamodule is not None
                and hasattr(self.trainer.datamodule, "test_pert_ids")):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def _forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            backbone=self.backbone,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            gene_pos=batch["gene_pos"],
            string_feats=batch["string_feats"],
            sym_ids=batch["sym_ids"],
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G], labels: [B, G] int (0/1/2)"""
        B, C, G = logits.shape
        return self.criterion(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, 3]
            labels.reshape(-1),                        # [B*G]
        )

    def _compute_mixup_loss(
        self, logits: torch.Tensor,
        labels_a: torch.Tensor, labels_b: torch.Tensor, lam: float
    ) -> torch.Tensor:
        """Compute mixup-weighted focal loss."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)
        loss_a = self.criterion(logits_flat, labels_a.reshape(-1))
        loss_b = self.criterion(logits_flat, labels_b.reshape(-1))
        return lam * loss_a + (1 - lam) * loss_b

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Extract AIDO dual-pool features for mixup
        global_emb, pert_emb = self.model.extract_aido_features(
            self.backbone, batch["input_ids"], batch["attention_mask"], batch["gene_pos"]
        )

        # Apply manifold mixup at the AIDO feature level
        alpha = self.hparams.mixup_alpha
        if alpha > 0.0:
            lam = float(np.random.beta(alpha, alpha))
            batch_size = global_emb.size(0)
            index = torch.randperm(batch_size, device=global_emb.device)
            mixed_global = lam * global_emb + (1 - lam) * global_emb[index]
            mixed_pert = lam * pert_emb + (1 - lam) * pert_emb[index]
            labels_a = batch["label"]
            labels_b = batch["label"][index]
        else:
            mixed_global, mixed_pert = global_emb, pert_emb
            labels_a = batch["label"]
            labels_b = batch["label"]
            lam = 1.0

        # Finish forward pass with mixed AIDO features + other features unchanged
        # Handle STRING missing genes
        string_feats = batch["string_feats"].float()
        string_is_zero = (string_feats.abs().sum(dim=-1, keepdim=True) == 0)
        missing_emb = self.model.string_missing_emb.unsqueeze(0).expand(string_feats.shape[0], -1)
        string_feats = torch.where(string_is_zero, missing_emb.to(string_feats.dtype), string_feats)

        # Symbol CNN encoding
        sym_feats = self.model.symbol_cnn(batch["sym_ids"])

        # Project to fusion token dimension
        t_global = self.model.global_proj(mixed_global.float())
        t_pert = self.model.pert_proj(mixed_pert.float())
        t_sym = self.model.sym_proj(sym_feats.float())
        t_ppi = self.model.ppi_proj(string_feats.float())

        # Cross-attention fusion
        tokens = torch.stack([t_global, t_pert, t_sym, t_ppi], dim=1)
        fused = self.model.fusion(tokens)

        # Prediction head
        logits = self.model.head(fused)
        logits = logits.view(-1, N_CLASSES, N_GENES)

        # Compute mixup loss
        if abs(lam - 1.0) < 1e-6:
            loss = self._compute_loss(logits, labels_a)
        else:
            loss = self._compute_mixup_loss(logits, labels_a, labels_b, lam)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward(batch)
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
        local_preds = torch.cat(self._val_preds, dim=0).numpy()    # [N_local, 3, G]
        local_labels = torch.cat(self._val_labels, dim=0).numpy()  # [N_local, G]
        local_idx = torch.cat(self._val_indices, dim=0)

        # Deduplicate by index (avoid DDP duplications at val)
        unique_pos = np.unique(local_idx.numpy(), return_index=True)[1]
        local_preds = local_preds[unique_pos]
        local_labels = local_labels[unique_pos]

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        local_f1 = compute_deg_f1(local_preds, local_labels)

        # Average F1 across DDP ranks via scalar all-reduce
        world_size = self.trainer.world_size if self.trainer.world_size else 1
        if world_size > 1:
            import torch.distributed as dist
            f1_t = torch.tensor(local_f1, dtype=torch.float32, device="cuda")
            dist.all_reduce(f1_t, op=dist.ReduceOp.SUM)
            f1 = (f1_t / world_size).item()
        else:
            f1 = local_f1

        self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward(batch)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self) -> None:
        # ── Avoid NCCL all_gather hangs (seen in DDP test with AIDO.Cell models).
        # Each rank writes its own shard to disk; rank 0 reads all shards and merges.
        # ──────────────────────────────────────────────────────────────────────────────
        if not self._test_preds:
            return

        output_dir = Path(__file__).parent / "run"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Serialize local predictions/indices to disk (one file per rank)
        rank = self.global_rank if hasattr(self, "global_rank") else 0
        shard_path = output_dir / f"_test_shard_rank{rank}.npz"

        local_preds = torch.cat(self._test_preds, dim=0).numpy()
        local_idx = torch.cat(self._test_indices, dim=0).numpy()
        local_labels = (torch.cat(self._test_labels, dim=0).numpy()
                        if self._test_labels else None)
        local_pert_ids = [self._test_pert_ids[i] for i in local_idx]
        local_symbols = [self._test_symbols[i] for i in local_idx]

        np.savez(shard_path,
                 preds=local_preds,
                 idx=local_idx,
                 labels=local_labels,
                 pert_ids=local_pert_ids,
                 symbols=local_symbols)
        self.print(f"[Rank {rank}] Wrote shard → {shard_path} ({len(local_idx)} samples)")

        self._test_preds.clear()
        self._test_indices.clear()
        self._test_labels.clear()

        # Barrier: ensure all ranks finish writing before rank 0 reads
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Rank 0 aggregates all shards and writes the final prediction file
        if self.trainer.is_global_zero:
            world_size = self.trainer.world_size
            all_preds, all_idx, all_labels = [], [], []
            all_pert_ids, all_symbols = [], []

            for r in range(world_size):
                shard = np.load(output_dir / f"_test_shard_rank{r}.npz",
                                allow_pickle=True)
                all_preds.append(shard["preds"])
                all_idx.append(shard["idx"])
                if shard["labels"] is not None:
                    all_labels.append(shard["labels"])
                all_pert_ids.extend(list(shard["pert_ids"]))
                all_symbols.extend(list(shard["symbols"]))

            # Concatenate across ranks and deduplicate by original index
            preds = np.concatenate(all_preds, axis=0)
            idxs = np.concatenate(all_idx, axis=0)

            unique_pos = np.unique(idxs, return_index=True)[1]
            preds = preds[unique_pos]
            sorted_idxs = idxs[unique_pos]
            order = np.argsort(sorted_idxs)
            preds = preds[order]
            final_idxs = sorted_idxs[order]
            final_pert_ids = [all_pert_ids[i] for i in unique_pos[order]]
            final_symbols = [all_symbols[i] for i in unique_pos[order]]

            # Compute test F1 if labels are available
            if all_labels:
                labels = np.concatenate(all_labels, axis=0)[unique_pos][order]
                test_f1 = compute_deg_f1(preds, labels)
                self.log("test_f1", test_f1, prog_bar=True)
                self.print(f"Test F1: {test_f1:.4f}")

            # Write final test_predictions.tsv
            out_path = output_dir / "test_predictions.tsv"
            rows = []
            for i in range(len(preds)):
                rows.append({
                    "idx": final_pert_ids[i],
                    "input": final_symbols[i],
                    "prediction": json.dumps(preds[i].tolist()),
                })
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path} ({len(rows)} rows)")

            # Cleanup shard files
            for r in range(world_size):
                (output_dir / f"_test_shard_rank{r}.npz").unlink(missing_ok=True)

    def configure_optimizers(self):
        # Separate param groups: LoRA backbone params, head params, other params
        lora_params = [
            p for name, p in self.backbone.named_parameters() if p.requires_grad
        ]
        head_params = list(self.model.head.parameters())
        fusion_params = list(self.model.fusion.parameters())
        proj_params = (
            list(self.model.global_proj.parameters()) +
            list(self.model.pert_proj.parameters()) +
            list(self.model.sym_proj.parameters()) +
            list(self.model.ppi_proj.parameters()) +
            list(self.model.symbol_cnn.parameters()) +
            [self.model.string_missing_emb]
        )

        all_model_params = set(id(p) for p in self.model.parameters() if p.requires_grad)
        head_param_ids = set(id(p) for p in head_params)
        fusion_param_ids = set(id(p) for p in fusion_params)
        proj_param_ids = set(id(p) for p in proj_params)
        other_model_params = [
            p for p in self.model.parameters()
            if p.requires_grad
            and id(p) not in head_param_ids
            and id(p) not in fusion_param_ids
            and id(p) not in proj_param_ids
        ]

        param_groups = [
            {"params": lora_params, "lr": self.hparams.backbone_lr,
             "weight_decay": self.hparams.weight_decay},
            {"params": head_params + fusion_params + proj_params,
             "lr": self.hparams.head_lr,
             "weight_decay": self.hparams.weight_decay},
        ]
        if other_model_params:
            param_groups.append({
                "params": other_model_params,
                "lr": self.hparams.head_lr,
                "weight_decay": self.hparams.weight_decay
            })

        opt = torch.optim.AdamW(param_groups)

        # ReduceLROnPlateau monitoring val_loss — reactive to overfitting onset
        # Proven by node2-3-1-1 (0.4555), node3-1-3-1-1-1-1 (0.4768)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            patience=self.hparams.plateau_patience,
            factor=self.hparams.plateau_factor,
            verbose=True,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable params + buffers ─────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
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
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )
        return trainable_state_dict

    def load_state_dict(self, state_dict, strict=True):
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
            self.print(f"Warning: Unexpected checkpoint keys: {unexpected_keys[:5]}...")
        loaded_trainable = len([k for k in state_dict if k in trainable_keys])
        loaded_buffers = len([k for k in state_dict if k in buffer_keys])
        self.print(
            f"Loading checkpoint: {loaded_trainable} trainable params "
            f"and {loaded_buffers} buffers"
        )
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 1-3-1-2: LoRA AIDO.Cell-10M + Cross-Attention Fusion DEG predictor"
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--backbone-lr", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=float, default=4.0)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--plateau-patience", type=int, default=5,
                   help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--plateau-factor", type=float, default=0.5,
                   help="ReduceLROnPlateau factor")
    p.add_argument("--mixup-alpha", type=float, default=0.3,
                   help="Manifold mixup alpha (0 = disabled)")
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug_max_step", type=int, default=None,
                   help="Limit batches for quick debugging")
    p.add_argument("--fast_dev_run", action="store_true",
                   help="Run 1 batch train/val/test for pipeline validation")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Trainer configuration ──────────────────────────────────────────────────
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

    if n_gpus == 1:
        strategy = SingleDeviceStrategy(device="cuda:0")
    else:
        # find_unused_parameters=True: frozen backbone params not used in fwd pass
        strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-3-1-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.early_stopping_patience, verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

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

    # ── DataModule & Model ─────────────────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        cache_dir=str(cache_dir),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        mixup_alpha=args.mixup_alpha,
    )

    # ── Training ───────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Testing ────────────────────────────────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # Save test score summary
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        with open(score_path, "w") as f:
            f.write(f"Test results: {json.dumps(test_results, indent=2)}\n")
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
