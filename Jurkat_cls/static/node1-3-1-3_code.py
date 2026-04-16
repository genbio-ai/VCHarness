#!/usr/bin/env python3
"""
Node 1-3-1-3: LoRA AIDO.Cell-10M + CosineAnnealingLR + 3-layer Cross-Attention Fusion
========================================================================================
Strategy:
  - LoRA AIDO.Cell-10M (r=4, all 8 layers, PEFT): breaks the frozen-backbone ceiling (~0.44)
  - STRING PPI embeddings pre-computed once (frozen cache), AIDO computed online
  - 3-layer TransformerEncoder cross-attention over 4 feature tokens (d_model=256, nhead=8)
  - CosineAnnealingLR(T_max=100, eta_min=1e-6) — THE critical fix from sibling node1-3-1-2 failure
    (ReduceLROnPlateau fired 6 times in 56 epochs, exhausting LR budget; CosineAnnealing prevents this)
  - class_weights=[4.0, 1.0, 8.0]: moderate (between parent [3,1,5] and sibling [6,1,12])
  - Manifold mixup (alpha=0.3) applied consistently to ALL 4 feature tokens
  - Single best checkpoint (no averaging: sibling 1 confirmed averaging is harmful)
  - weight_decay=0.10: proven sweet spot for cross-attention + 1500 samples

Key differences from siblings:
  vs sibling 1 (node1-3-1-1, F1=0.4187): Uses LoRA backbone (not frozen), aggressive weights,
                                           no checkpoint averaging
  vs sibling 2 (node1-3-1-2, F1=0.4182): CosineAnnealingLR (not ReduceLROnPlateau),
                                           moderate class_weights [4,1,8] vs [6,1,12],
                                           consistent mixup on all tokens

Architecture:
  AIDO.Cell-10M (LoRA r=4, all 8 layers):
    Input: synthetic expression (perturbed gene=1.0, others=-1.0)
    Output: last_hidden_state [B, 19266, 256]
    Dual pooling: global_emb [B, 256] + pert_emb [B, 256]
    → Linear(256→256) projection each
  STRING GNN (pre-computed frozen):
    Lookup: Ensembl ID → [256] → Linear(256→256)
  Symbol CNN (trainable):
    char IDs → multi-scale Conv1d → [64] → Linear(64→256)
  4-token cross-attention fusion:
    stack([t_global, t_pert, t_sym, t_ppi]) [B, 4, 256]
    + learned positional embeddings [4, 256]
    3-layer TransformerEncoder (nhead=8, dim_ff=256, norm_first=True)
    mean-pool → [B, 256]
  MLP head: LayerNorm(256) → Linear(256→256) → GELU → Dropout(0.4) → Linear(256→3×6640)

Expected test F1: 0.45-0.49 (targeting tree-best LoRA+cross-attn range)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import subprocess
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
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from peft import LoraConfig, get_peft_model, TaskType
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

# class_weights=[4.0, 1.0, 8.0]: moderate between parent's [3,1,5] and sibling's [6,1,12]
# Reduces val-test gap while maintaining minority class sensitivity
CLASS_WEIGHTS = torch.tensor([4.0, 1.0, 8.0], dtype=torch.float32)

CHAR_VOCAB = {c: i + 1 for i, c in enumerate(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.abcdefghijklmnopqrstuvwxyz"
)}
CHAR_VOCAB["<pad>"] = 0
MAX_SYM_LEN = 16

AIDO_HIDDEN_DIM = 256    # AIDO.Cell-10M hidden dimension
STRING_HIDDEN_DIM = 256  # STRING GNN output dimension
FUSION_D_MODEL = 256     # Cross-attention token dimension


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


# ──────────────────────────────────────────────────────────────────────────────
# STRING Feature Pre-computation (frozen, run once per node)
# ──────────────────────────────────────────────────────────────────────────────
def precompute_string_features(data_dir: Path, cache_dir: Path) -> Dict[str, np.ndarray]:
    """
    Pre-compute frozen STRING PPI features for all data splits.
    File-cached: if cache exists, loads from disk; otherwise computes and saves.
    DDP-safe: caller must ensure only rank 0 calls this before a distributed barrier.
    """
    cache_path = cache_dir / "string_cache.npz"
    if cache_path.exists():
        print(f"[PreCompute] Loading STRING cache from {cache_path}")
        return dict(np.load(str(cache_path), allow_pickle=True))

    print("[PreCompute] Computing STRING PPI features (this runs once)...")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load STRING GNN (frozen, float32)
    string_model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
    string_model = string_model.eval().to(device)
    graph = torch.load(str(STRING_GNN_DIR / "graph_data.pt"), map_location=device)
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    string_name_to_idx = {n: i for i, n in enumerate(node_names)}

    edge_index = graph["edge_index"]
    edge_weight = graph.get("edge_weight")
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        string_outputs = string_model(edge_index=edge_index, edge_weight=edge_weight)
    all_string_embs = string_outputs.last_hidden_state.float().cpu().numpy()  # [18870, 256]
    print(f"[PreCompute] STRING GNN: {all_string_embs.shape[0]} node embeddings extracted")

    del string_model
    torch.cuda.empty_cache()

    results: Dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        tsv_path = data_dir / f"{split}.tsv"
        if not tsv_path.exists():
            continue
        df = pd.read_csv(tsv_path, sep="\t")
        pert_ids: List[str] = df["pert_id"].tolist()
        N = len(pert_ids)

        string_feats = np.zeros((N, STRING_HIDDEN_DIM), dtype=np.float32)
        n_found = 0
        for j, pid in enumerate(pert_ids):
            eid = pid.split(".")[0]
            idx = string_name_to_idx.get(eid)
            if idx is not None:
                string_feats[j] = all_string_embs[idx]
                n_found += 1
        print(f"    STRING {split}: {n_found}/{N} ({100*n_found/N:.1f}%) genes found")
        results[f"{split}_string"] = string_feats

    np.savez(str(cache_path), **results)
    print(f"[PreCompute] STRING cache saved → {cache_path}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# AIDO Input Tokenization (pre-tokenized offline for speed)
# ──────────────────────────────────────────────────────────────────────────────
def tokenize_pert_ids(
    pert_ids: List[str],
    tokenizer: Any,
    gene_id_to_pos: Dict[str, int],
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-tokenize perturbation IDs for AIDO model.
    Each sample sets the perturbed gene expression=1.0, all others=-1.0 (missing).

    Returns:
        input_ids: np.ndarray [N, 19264] float16 (expression values, cast to f16 to save memory)
        pert_positions: np.ndarray [N] int32 (gene position in AIDO output, -1 if not found)
    """
    N = len(pert_ids)
    all_input_ids = []

    for start in range(0, N, batch_size):
        batch_chunk = pert_ids[start:start + batch_size]
        batch_data = [
            {"gene_ids": [pid.split(".")[0]], "expression": [1.0]}
            for pid in batch_chunk
        ]
        batch_inputs = tokenizer(batch_data, return_tensors="pt")
        # input_ids: [B, 19264] float32 — store as float16 to save 50% memory
        all_input_ids.append(batch_inputs["input_ids"].numpy().astype(np.float16))
        if (start // batch_size) % 10 == 0:
            print(f"    Tokenizing: {min(start + batch_size, N)}/{N} samples done")

    all_input_ids = np.concatenate(all_input_ids, axis=0)  # [N, 19264] float16

    # Compute per-sample AIDO gene position for the perturbed gene
    pert_positions = np.full(N, -1, dtype=np.int32)
    for i, pid in enumerate(pert_ids):
        eid = pid.split(".")[0]
        pos = gene_id_to_pos.get(eid)
        if pos is not None:
            pert_positions[i] = int(min(int(pos), 19263))

    n_found = int((pert_positions >= 0).sum())
    print(f"    Gene position lookup: {n_found}/{N} ({100*n_found/N:.1f}%) genes found in AIDO vocab")
    return all_input_ids, pert_positions


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataset(Dataset):
    """DEG dataset with pre-tokenized AIDO inputs and pre-computed STRING features."""

    def __init__(
        self,
        df: pd.DataFrame,
        input_ids: np.ndarray,       # [N, 19264] float16
        pert_positions: np.ndarray,  # [N] int32
        string_feats: np.ndarray,    # [N, 256] float32
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.input_ids = input_ids           # numpy float16, convert to float32 in __getitem__
        self.pert_positions = pert_positions # numpy int32
        self.string_feats = torch.from_numpy(string_feats.astype(np.float32))  # [N, 256]
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Remap: {-1→0, 0→1, 1→2}
            self.labels = torch.tensor(
                np.array(raw_labels, dtype=np.int8) + 1, dtype=torch.long
            )  # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            # Cast float16 → float32 for model input
            "input_ids": torch.from_numpy(self.input_ids[idx].astype(np.float32)),  # [19264]
            "pert_pos": int(self.pert_positions[idx]),
            "string_feats": self.string_feats[idx],   # [256]
            "sym_ids": encode_symbol(self.symbols[idx]),  # [MAX_SYM_LEN]
        }
        if not self.is_test:
            item["label"] = self.labels[idx]  # [6640]
        return item


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate: preserves pert_id strings and pert_pos integers."""
    result: Dict[str, Any] = {}
    for key in batch[0]:
        if key == "pert_id":
            result[key] = [item[key] for item in batch]
        elif key == "pert_pos":
            result[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
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
        num_workers: int = 4,
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

        # ── Step 1: Tokenizer (rank 0 first to handle any init, then all ranks) ──
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
            # Build Ensembl gene ID → AIDO position index map
            if hasattr(self._tokenizer, "gene_id_to_index"):
                self._gene_id_to_pos = {k: int(v) for k, v in self._tokenizer.gene_id_to_index.items()}
            elif hasattr(self._tokenizer, "gene_to_index"):
                self._gene_id_to_pos = {k: int(v) for k, v in self._tokenizer.gene_to_index.items()}

        # ── Step 2: STRING feature pre-computation (rank 0 only) ─────────────
        string_cache_path = self.cache_dir / "string_cache.npz"
        if local_rank == 0 and not string_cache_path.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            precompute_string_features(self.data_dir, self.cache_dir)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self._string_cache is None:
            self._string_cache = dict(np.load(str(string_cache_path), allow_pickle=True))

        # ── Step 3: Create datasets ───────────────────────────────────────────
        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")

            print("[DataModule] Pre-tokenizing training samples...")
            train_input_ids, train_pos = tokenize_pert_ids(
                train_df["pert_id"].tolist(), self._tokenizer, self._gene_id_to_pos
            )
            print("[DataModule] Pre-tokenizing validation samples...")
            val_input_ids, val_pos = tokenize_pert_ids(
                val_df["pert_id"].tolist(), self._tokenizer, self._gene_id_to_pos
            )

            self.train_ds = DEGDataset(
                train_df, train_input_ids, train_pos, self._string_cache["train_string"]
            )
            self.val_ds = DEGDataset(
                val_df, val_input_ids, val_pos, self._string_cache["val_string"]
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            print("[DataModule] Pre-tokenizing test samples...")
            test_input_ids, test_pos = tokenize_pert_ids(
                test_df["pert_id"].tolist(), self._tokenizer, self._gene_id_to_pos
            )
            self.test_ds = DEGDataset(
                test_df, test_input_ids, test_pos, self._string_cache["test_string"],
                is_test=True
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
        x = self.embedding(ids).transpose(1, 2)        # [B, embed_dim, L]
        f3 = F.gelu(self.conv3(x)).max(dim=-1).values  # [B, out_dim]
        f5 = F.gelu(self.conv5(x)).max(dim=-1).values  # [B, out_dim]
        return self.norm(torch.cat([f3, f5], dim=-1))  # [B, 64]


class CrossAttentionFusion(nn.Module):
    """
    3-layer TransformerEncoder over 4 feature tokens with learned positional embeddings.
    Input: [B, 4, d_model] → Output: [B, d_model]
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        # 4 learned positional embeddings (one per feature token)
        self.pos_emb = nn.Embedding(4, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, 4, d_model] → [B, d_model]"""
        pos_ids = torch.arange(4, device=tokens.device)
        tokens = tokens + self.pos_emb(pos_ids)  # add positional embeddings
        fused = self.transformer(tokens)          # [B, 4, d_model]
        return fused.mean(dim=1)                  # mean pool → [B, d_model]


class DEGHeadModel(nn.Module):
    """
    All trainable non-backbone components:
    - AIDO dual-pool → Linear(256→256) for each of global/pert streams
    - STRING PPI → learned missing embedding + Linear(256→256)
    - Symbol CNN → Linear(64→256)
    - 4-token cross-attention fusion
    - 2-stage MLP head: 256→256→3×6640
    """

    def __init__(self, dropout: float = 0.4):
        super().__init__()
        # Per-stream projections to FUSION_D_MODEL=256
        self.global_proj = nn.Sequential(
            nn.LayerNorm(AIDO_HIDDEN_DIM),
            nn.Linear(AIDO_HIDDEN_DIM, FUSION_D_MODEL),
        )
        self.pert_proj = nn.Sequential(
            nn.LayerNorm(AIDO_HIDDEN_DIM),
            nn.Linear(AIDO_HIDDEN_DIM, FUSION_D_MODEL),
        )

        # STRING PPI
        self.string_missing_emb = nn.Parameter(torch.zeros(STRING_HIDDEN_DIM))
        self.ppi_proj = nn.Sequential(
            nn.LayerNorm(STRING_HIDDEN_DIM),
            nn.Linear(STRING_HIDDEN_DIM, FUSION_D_MODEL),
        )

        # Symbol CNN + projection
        self.symbol_cnn = SymbolCNN(vocab_size=len(CHAR_VOCAB), embed_dim=32, out_dim=32)
        self.sym_proj = nn.Linear(self.symbol_cnn.out_dim, FUSION_D_MODEL)  # 64→256

        # Cross-attention fusion (3 layers, d_model=256, nhead=8, dim_ff=256)
        self.fusion = CrossAttentionFusion(
            d_model=FUSION_D_MODEL, nhead=8, num_layers=3,
            dim_feedforward=256, dropout=0.1
        )

        # 2-stage MLP head: 256 → 256 → 3×6640
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_D_MODEL),
            nn.Linear(FUSION_D_MODEL, FUSION_D_MODEL),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(FUSION_D_MODEL, N_CLASSES * N_GENES),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.string_missing_emb, mean=0.0, std=0.01)

    def _prepare_string_feats(self, string_feats: torch.Tensor) -> torch.Tensor:
        """Replace zero vectors (missing genes) with learned missing embedding."""
        is_zero = (string_feats.abs().sum(dim=-1, keepdim=True) == 0)  # [B, 1] bool
        missing = self.string_missing_emb.unsqueeze(0).expand(string_feats.shape[0], -1)
        return torch.where(is_zero, missing.to(string_feats.dtype), string_feats)

    def build_tokens(
        self,
        global_emb: torch.Tensor,   # [B, 256] float32
        pert_emb: torch.Tensor,      # [B, 256] float32
        string_feats: torch.Tensor,  # [B, 256] float32
        sym_ids: torch.Tensor,       # [B, MAX_SYM_LEN] long
    ) -> torch.Tensor:               # [B, 4, 256]
        """Build 4 feature tokens for cross-attention fusion."""
        t_global = self.global_proj(global_emb)                             # [B, 256]
        t_pert = self.pert_proj(pert_emb)                                    # [B, 256]
        ppi_feats = self._prepare_string_feats(string_feats)
        t_ppi = self.ppi_proj(ppi_feats.float())                            # [B, 256]
        sym_feats = self.symbol_cnn(sym_ids)                                 # [B, 64]
        t_sym = self.sym_proj(sym_feats)                                     # [B, 256]
        # Stack: [global, pert, sym, ppi] → [B, 4, 256]
        return torch.stack([t_global, t_pert, t_sym, t_ppi], dim=1)

    def predict_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, 4, 256] → logits [B, 3, 6640]"""
        fused = self.fusion(tokens)             # [B, 256]
        logits = self.head(fused)               # [B, 3*6640]
        return logits.view(-1, N_CLASSES, N_GENES)


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        dropout: float = 0.4,
        backbone_lr: float = 2e-4,
        head_lr: float = 6e-4,
        weight_decay: float = 0.10,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        mixup_alpha: float = 0.3,
        t_max: int = 100,
        eta_min: float = 1e-6,
        lora_r: int = 4,
        lora_alpha_val: float = 4.0,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.aido_model = None
        self.head_model: Optional[DEGHeadModel] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulation buffers for validation and test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.aido_model is None:
            # ── Load AIDO.Cell-10M with LoRA ──────────────────────────────────
            aido_base = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
            aido_base = aido_base.to(torch.bfloat16)

            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha_val,
                lora_dropout=self.hparams.lora_dropout,
                target_modules=["query", "key", "value"],
                # target_modules path: bert.encoder.layer.*.attention.self.{query,key,value}
                layers_to_transform=None,  # all 8 transformer layers
                bias="none",
            )
            self.aido_model = get_peft_model(aido_base, lora_cfg)
            self.aido_model.config.use_cache = False
            # Gradient checkpointing to reduce activation memory
            self.aido_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            n_lora_params = sum(p.numel() for p in self.aido_model.parameters() if p.requires_grad)
            self.print(f"AIDO LoRA: {n_lora_params:,} trainable parameters")

            # ── Create head model ─────────────────────────────────────────────
            self.head_model = DEGHeadModel(dropout=self.hparams.dropout)

            # ── Cast all trainable parameters to float32 for stable optimization ──
            for p in self.aido_model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
            for p in self.head_model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()

        if self.criterion is None:
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        # Retrieve test metadata from DataModule
        if stage == "test":
            dm = getattr(self.trainer, "datamodule", None) if self.trainer else None
            if dm is not None:
                self._test_pert_ids = getattr(dm, "test_pert_ids", [])
                self._test_symbols = getattr(dm, "test_symbols", [])

    def _get_aido_features(
        self,
        input_ids: torch.Tensor,   # [B, 19264] float32
        pert_pos: torch.Tensor,    # [B] long (gene position in AIDO output, -1=missing)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run AIDO backbone (LoRA) and extract:
          global_emb: [B, 256] mean pool over all gene positions
          pert_emb:   [B, 256] at the perturbed gene's position
        """
        B, L = input_ids.shape
        # Create attention_mask (always all-ones for AIDO, but pass for API compatibility)
        attention_mask = torch.ones(B, L, dtype=torch.long, device=input_ids.device)
        outputs = self.aido_model(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: [B, 19266, 256]; cast bf16→f32 for stable downstream ops
        hidden = outputs.last_hidden_state.float()
        gene_hidden = hidden[:, :19264, :]  # [B, 19264, 256] — first 19264 = gene positions

        # Global mean pool over all gene positions
        global_emb = gene_hidden.mean(dim=1)  # [B, 256]

        # Perturbed gene positional embedding (vectorized)
        valid_pos = pert_pos.clamp(min=0).to(device=gene_hidden.device)  # [B] long (clamp -1→0)
        batch_idx = torch.arange(B, device=gene_hidden.device)
        pert_emb = gene_hidden[batch_idx, valid_pos]  # [B, 256]

        # For genes not in AIDO vocabulary (pert_pos == -1), fall back to global mean
        missing_mask = (pert_pos < 0).to(device=gene_hidden.device).unsqueeze(-1).expand_as(pert_emb)
        pert_emb = torch.where(missing_mask, global_emb, pert_emb)

        return global_emb, pert_emb

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G], labels: [B, G] int (0/1/2)"""
        B, C, G = logits.shape
        return self.criterion(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, 3]
            labels.reshape(-1),                        # [B*G]
        )

    def _extract_all_features(self, batch: Dict[str, Any]):
        """Extract AIDO + STRING + symbol features from batch."""
        input_ids = batch["input_ids"]      # [B, 19264] float32
        pert_pos = batch["pert_pos"]        # [B] long
        string_feats = batch["string_feats"]  # [B, 256]
        sym_ids = batch["sym_ids"]            # [B, MAX_SYM_LEN] long
        global_emb, pert_emb = self._get_aido_features(input_ids, pert_pos)
        return global_emb, pert_emb, string_feats, sym_ids

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        global_emb, pert_emb, string_feats, sym_ids = self._extract_all_features(batch)
        labels = batch["label"]  # [B, 6640]
        B = global_emb.shape[0]

        # Build 4 feature tokens [B, 4, 256]
        tokens = self.head_model.build_tokens(global_emb, pert_emb, string_feats, sym_ids)

        # Manifold mixup: apply consistently to ALL 4 tokens to avoid token mismatch
        # (sibling node1-3-1-2 only mixed AIDO tokens but not STRING token → inconsistency)
        if self.hparams.mixup_alpha > 0:
            lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
            perm = torch.randperm(B, device=tokens.device)
            tokens_mixed = lam * tokens + (1 - lam) * tokens[perm]
            labels_a, labels_b = labels, labels[perm]

            logits = self.head_model.predict_from_tokens(tokens_mixed)  # [B, 3, 6640]
            loss = lam * self._compute_loss(logits, labels_a) + \
                   (1 - lam) * self._compute_loss(logits, labels_b)
        else:
            logits = self.head_model.predict_from_tokens(tokens)
            loss = self._compute_loss(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        global_emb, pert_emb, string_feats, sym_ids = self._extract_all_features(batch)
        labels = batch["label"]

        tokens = self.head_model.build_tokens(global_emb, pert_emb, string_feats, sym_ids)
        logits = self.head_model.predict_from_tokens(tokens)
        loss = self._compute_loss(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
        self._val_preds.append(probs)
        self._val_labels.append(labels.cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, dim=0).numpy()    # [N_local, 3, G]
        local_labels = torch.cat(self._val_labels, dim=0).numpy()  # [N_local, G]
        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        local_f1 = compute_deg_f1(local_preds, local_labels)

        # Synchronize F1 across DDP ranks via scalar all-reduce
        world_size = self.trainer.world_size if self.trainer else 1
        if world_size > 1:
            import torch.distributed as dist
            f1_t = torch.tensor(local_f1, dtype=torch.float32, device="cuda")
            dist.all_reduce(f1_t, op=dist.ReduceOp.SUM)
            f1 = (f1_t / world_size).item()
        else:
            f1 = local_f1

        self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        global_emb, pert_emb, string_feats, sym_ids = self._extract_all_features(batch)
        tokens = self.head_model.build_tokens(global_emb, pert_emb, string_feats, sym_ids)
        logits = self.head_model.predict_from_tokens(tokens)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx = torch.cat(self._test_indices, dim=0)  # [N_local]

        # DDP: gather predictions from all ranks
        all_preds = self.all_gather(local_preds)  # [world_size, N_local, 3, G]
        all_idx = self.all_gather(local_idx)       # [world_size, N_local]

        self._test_preds.clear()
        self._test_indices.clear()

        if self.trainer.is_global_zero:
            preds = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            idxs = all_idx.view(-1).cpu().numpy()

            # Deduplicate (DDP may repeat samples at dataset boundaries)
            unique_pos = np.unique(idxs, return_index=True)[1]
            preds = preds[unique_pos]
            sorted_idxs = idxs[unique_pos]

            # Sort by original index
            order = np.argsort(sorted_idxs)
            preds = preds[order]
            final_idxs = sorted_idxs[order]

            # Write test_predictions.tsv
            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"
            rows = []
            for rank_i, orig_i in enumerate(final_idxs):
                rows.append({
                    "idx": self._test_pert_ids[int(orig_i)],
                    "input": self._test_symbols[int(orig_i)],
                    "prediction": json.dumps(preds[rank_i].tolist()),
                })
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path} ({len(rows)} rows)")

    def configure_optimizers(self):
        # Separate LoRA backbone and head parameter groups with different LRs
        backbone_params = [p for p in self.aido_model.parameters() if p.requires_grad]
        head_params = list(self.head_model.parameters())

        param_groups = [
            {"params": backbone_params, "lr": self.hparams.backbone_lr,
             "weight_decay": self.hparams.weight_decay},
            {"params": head_params, "lr": self.hparams.head_lr,
             "weight_decay": self.hparams.weight_decay},
        ]
        opt = torch.optim.AdamW(param_groups, eps=1e-6)

        # CosineAnnealingLR: THE critical fix from sibling node1-3-1-2's ReduceLROnPlateau failure
        # ReduceLROnPlateau fired 6 times in 56 epochs (patience=5), exhausting LR budget
        # CosineAnnealingLR provides smooth, predictable decay without over-reacting to noise
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.t_max, eta_min=self.hparams.eta_min,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1},
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
        trainable_keys = {name for name, param in self.named_parameters() if param.requires_grad}
        buffer_keys = {name for name, _ in self.named_buffers() if name in full_state_keys}
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
        description="Node 1-3-1-3: LoRA AIDO.Cell-10M + CosineAnnealingLR + CrossAttn DEG predictor"
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--micro-batch-size", type=int, default=8,
                   help="Per-GPU batch size for AIDO online forward pass")
    p.add_argument("--global-batch-size", type=int, default=64,
                   help="Effective global batch size (multiple of micro_batch_size * 8)")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--backbone-lr", type=float, default=2e-4,
                   help="Learning rate for LoRA backbone parameters")
    p.add_argument("--head-lr", type=float, default=6e-4,
                   help="Learning rate for head model parameters (3x backbone LR)")
    p.add_argument("--weight-decay", type=float, default=0.10,
                   help="AdamW weight decay (proven sweet spot for cross-attn + 1500 samples)")
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--mixup-alpha", type=float, default=0.3,
                   help="Manifold mixup alpha (0 = disable)")
    p.add_argument("--t-max", type=int, default=100,
                   help="CosineAnnealingLR T_max (epochs)")
    p.add_argument("--eta-min", type=float, default=1e-6,
                   help="CosineAnnealingLR minimum LR")
    p.add_argument("--lora-r", type=int, default=4,
                   help="LoRA rank (r=4: minimal capacity, proven by tree-best nodes)")
    p.add_argument("--lora-alpha", type=float, default=4.0,
                   help="LoRA alpha (scaling = alpha/r = 1.0)")
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--early-stopping-patience", type=int, default=25,
                   help="EarlyStopping patience on val_f1 (generous: CosineAnnealing is monotonic)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step", type=int, default=None,
                   help="Limit batches for quick debugging")
    p.add_argument("--fast-dev-run", action="store_true",
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

    # ── Trainer configuration ─────────────────────────────────────────────────
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
        # find_unused_parameters=True: non-LoRA AIDO backbone params are frozen and
        # do not appear in the backward graph → required for DDP correctness
        strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-3-1-3-{epoch:03d}-{val_f1:.4f}",
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

    # ── DataModule & Model ────────────────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        cache_dir=str(cache_dir),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        dropout=args.dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        t_max=args.t_max,
        eta_min=args.eta_min,
        lora_r=args.lora_r,
        lora_alpha_val=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Testing ───────────────────────────────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # ── Save test score summary ───────────────────────────────────────────────
    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        # Run calc_metric.py to get the official F1 score
        pred_path = output_dir / "test_predictions.tsv"
        test_tsv_path = Path(args.data_dir) / "test.tsv"
        calc_script = Path(args.data_dir) / "calc_metric.py"
        try:
            result = subprocess.run(
                ["python", str(calc_script), str(pred_path), str(test_tsv_path)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                metric_data = json.loads(result.stdout.strip())
                f1_value = metric_data.get("value", "N/A")
                metric_details = metric_data.get("details", {})
                score_content = (
                    f"f1_score: {f1_value}\n"
                    f"recall: {metric_details.get('recall', 'N/A')}\n"
                    f"precision: {metric_details.get('precision', 'N/A')}\n"
                    f"roc_auc: {metric_details.get('roc_auc', 'N/A')}\n"
                    f"average_precision: {metric_details.get('average_precision', 'N/A')}\n"
                    f"raw_json: {result.stdout.strip()}\n"
                )
            else:
                score_content = (
                    f"calc_metric.py failed (returncode={result.returncode})\n"
                    f"stderr: {result.stderr[:500]}\n"
                )
        except Exception as e:
            score_content = f"Error running calc_metric.py: {e}\n"
        with open(score_path, "w") as f:
            f.write(score_content)
        print(f"Test summary saved → {score_path}")
        print(f"Score content:\n{score_content}")


if __name__ == "__main__":
    main()
