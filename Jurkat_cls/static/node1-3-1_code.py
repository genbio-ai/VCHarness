#!/usr/bin/env python3
"""
Node 1-2: Frozen AIDO.Cell-10M + Pre-computed STRING PPI + Char CNN + 3-stage MLP Head
========================================================================================
Strategy:
  - Frozen AIDO.Cell-10M backbone (NO LoRA) — eliminates overfitting from excess trainable params
  - Pre-computed dual-pool AIDO features (gene-pos + mean, 512-dim) cached to disk once
  - Pre-computed STRING PPI node embeddings (256-dim) for each perturbed gene
  - Character-level multi-scale CNN on gene symbol (128-dim)
  - 3-stage MLP head: 896→384→256→19920 with strong regularization (dropout=0.5)
  - CosineAnnealingLR for cleaner convergence
  - ModelCheckpoint + EarlyStopping both monitor val_f1 (mode=max)

Key improvements over parent (node1-3, test F1=0.4344):
  1. NO LoRA → removes the primary source of overfitting (parent's val-test gap was 0.080)
  2. Pre-computed features → 50x faster training, enables 200 epochs with large batches
  3. STRING PPI topology embeddings → orthogonal biological signal about gene connectivity
  4. 3-stage head (896→384→256→19920) mirrors tree-best node3-2's proven architecture
  5. CosineAnnealingLR (T_max=150) provides clean monotonic LR decay vs ReduceLROnPlateau
     which saturated after epoch 40 in the parent node
  6. Larger effective batch size (global_batch_size=256 vs 32) for better gradient estimates

Architecture:
  Pre-computed (frozen, run once):
    AIDO.Cell-10M → dual-pool: [gene_pos_emb(256), mean_pool(256)] = [512]
    STRING GNN → lookup table [18870, 256] → perturbed gene emb [256]
  Online (trainable):
    Char CNN: symbol → [128]
  Fusion: cat([512, 256, 128]) = [896]
  Head: LN(896) → 384 → GELU → Drop(0.5) → LN(384) → 256 → GELU → Drop(0.5) → 19920 → [3, 6640]

Total trainable: ~5.5M (vs parent's ~7.3M with LoRA)
Expected test F1: 0.45-0.47 (targeting node3-2's 0.462 as baseline)
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

# Class weights: moderate to balance minority class sensitivity vs. loss stability
# down ~3.56%, unchanged ~94.82%, up ~1.63% → capped moderate weights
CLASS_WEIGHTS = torch.tensor([3.0, 1.0, 5.0], dtype=torch.float32)

CHAR_VOCAB = {c: i + 1 for i, c in enumerate(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.abcdefghijklmnopqrstuvwxyz"
)}
CHAR_VOCAB["<pad>"] = 0
MAX_SYM_LEN = 16

AIDO_HIDDEN_DIM = 256    # AIDO.Cell-10M hidden dimension
STRING_HIDDEN_DIM = 256  # STRING GNN output dimension
CHAR_CNN_DIM = 128       # SymbolCNN output dimension
FUSION_DIM = AIDO_HIDDEN_DIM * 2 + STRING_HIDDEN_DIM + CHAR_CNN_DIM  # 512+256+128 = 896


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
# Feature Pre-computation (AIDO + STRING, frozen, run once)
# ──────────────────────────────────────────────────────────────────────────────
def precompute_features(data_dir: Path, cache_dir: Path,
                        precompute_batch_size: int = 8) -> Dict[str, np.ndarray]:
    """
    Pre-compute frozen AIDO dual-pool + STRING PPI features for all data splits.
    Uses file caching: if cache exists, loads from disk; otherwise computes and saves.
    DDP-safe: caller must ensure only rank 0 calls this before a distributed barrier.
    """
    cache_path = cache_dir / "feature_cache.npz"
    if cache_path.exists():
        print(f"[PreCompute] Loading cached features from {cache_path}")
        return dict(np.load(str(cache_path), allow_pickle=True))

    print("[PreCompute] Computing AIDO + STRING features (this runs once)...")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ── Load AIDO-10M tokenizer ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

    # Build Ensembl → AIDO position map for dual-pool gene-specific extraction
    gene_id_to_pos: Dict[str, int] = {}
    if hasattr(tokenizer, "gene_id_to_index"):
        gene_id_to_pos = {k: int(v) for k, v in tokenizer.gene_id_to_index.items()}
    elif hasattr(tokenizer, "gene_to_index"):
        gene_id_to_pos = {k: int(v) for k, v in tokenizer.gene_to_index.items()}

    # ── Load AIDO.Cell-10M backbone (frozen, bfloat16 for FlashAttention) ─────
    aido_model = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
    aido_model = aido_model.to(torch.bfloat16).eval().to(device)

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

        # ── AIDO dual-pool features [N, 512] ──────────────────────────────────
        aido_feats_list: List[np.ndarray] = []
        for start in range(0, N, precompute_batch_size):
            batch_ids = pert_ids[start:start + precompute_batch_size]
            # Tokenize: set perturbed gene = 1.0, all others filled with -1.0 (missing)
            batch_data = [
                {"gene_ids": [pid.split(".")[0]], "expression": [1.0]}
                for pid in batch_ids
            ]
            batch_inputs = tokenizer(batch_data, return_tensors="pt")
            input_ids = batch_inputs["input_ids"].to(device)         # [B, 19264] float32
            attention_mask = batch_inputs["attention_mask"].to(device)  # [B, 19264] int64

            with torch.no_grad():
                outputs = aido_model(input_ids=input_ids, attention_mask=attention_mask)
            # last_hidden_state: [B, 19266, 256]; first 19264 = gene positions, last 2 = summary
            last_hidden = outputs.last_hidden_state.float()  # cast bf16→f32
            gene_hidden = last_hidden[:, :19264, :]  # [B, 19264, 256]

            # Global mean-pool over all gene positions
            mean_emb = gene_hidden.mean(dim=1)  # [B, 256]

            # Gene-specific positional embedding at perturbed gene's position
            gene_pos_embs: List[torch.Tensor] = []
            for j, pid in enumerate(batch_ids):
                eid = pid.split(".")[0]
                pos = gene_id_to_pos.get(eid)
                if pos is not None:
                    pos_clamped = max(0, min(int(pos), 19263))
                    gene_pos_embs.append(gene_hidden[j, pos_clamped, :])
                else:
                    # Gene not in AIDO vocabulary: use mean-pool as fallback
                    gene_pos_embs.append(mean_emb[j])

            gene_pos_emb = torch.stack(gene_pos_embs)  # [B, 256]
            dual_pool = torch.cat([gene_pos_emb, mean_emb], dim=-1)  # [B, 512]
            aido_feats_list.append(dual_pool.cpu().float().numpy())

            if (start // precompute_batch_size) % 20 == 0:
                print(f"    AIDO: {start + len(batch_ids)}/{N} samples done")

        aido_feats = np.concatenate(aido_feats_list, axis=0)  # [N, 512]
        assert aido_feats.shape == (N, 512), f"Expected ({N}, 512), got {aido_feats.shape}"

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

        results[f"{split}_aido"] = aido_feats.astype(np.float32)
        results[f"{split}_string"] = string_feats.astype(np.float32)

    # ── Save cache ────────────────────────────────────────────────────────────
    np.savez(str(cache_path), **results)
    print(f"[PreCompute] Feature cache saved → {cache_path}")

    # Free GPU memory before training begins
    del aido_model, string_model
    torch.cuda.empty_cache()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Dataset (lightweight, uses pre-computed features)
# ──────────────────────────────────────────────────────────────────────────────
class PrecomputedDEGDataset(Dataset):
    """DEG dataset using pre-computed AIDO + STRING features."""

    def __init__(
        self,
        df: pd.DataFrame,
        aido_feats: np.ndarray,    # [N, 512] float32
        string_feats: np.ndarray,  # [N, 256] float32
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.aido_feats = torch.from_numpy(aido_feats.astype(np.float32))      # [N, 512]
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
            "aido_feats": self.aido_feats[idx],      # [512]
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
        micro_batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

        self.train_ds: Optional[PrecomputedDEGDataset] = None
        self.val_ds: Optional[PrecomputedDEGDataset] = None
        self.test_ds: Optional[PrecomputedDEGDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []
        self._cache: Optional[Dict[str, np.ndarray]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        cache_path = self.cache_dir / "feature_cache.npz"

        # Rank 0 computes features if cache doesn't exist; other ranks wait at barrier
        if local_rank == 0 and not cache_path.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            precompute_features(self.data_dir, self.cache_dir)

        # DDP barrier: ensure rank 0 finishes before other ranks try to load
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # All ranks load from cache (guaranteed to exist after barrier)
        if self._cache is None:
            self._cache = dict(np.load(str(cache_path), allow_pickle=True))

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PrecomputedDEGDataset(
                train_df, self._cache["train_aido"], self._cache["train_string"]
            )
            self.val_ds = PrecomputedDEGDataset(
                val_df, self._cache["val_aido"], self._cache["val_string"]
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PrecomputedDEGDataset(
                test_df, self._cache["test_aido"], self._cache["test_string"], is_test=True
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

    def __init__(self, vocab_size: int, embed_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, out_dim, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(out_dim * 2)
        self.out_dim = out_dim * 2  # 128

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, L] → [B, out_dim*2]"""
        x = self.embedding(ids).transpose(1, 2)       # [B, embed_dim, L]
        f3 = F.gelu(self.conv3(x)).max(dim=-1).values  # [B, out_dim]
        f5 = F.gelu(self.conv5(x)).max(dim=-1).values  # [B, out_dim]
        return self.norm(torch.cat([f3, f5], dim=-1))  # [B, 128]


class DEGModel(nn.Module):
    """
    Lightweight DEG predictor using pre-computed AIDO + STRING features + char CNN.

    Input:
        aido_feats:   [B, 512]  — frozen dual-pool from AIDO.Cell-10M
        string_feats: [B, 256]  — frozen STRING PPI node embedding
        sym_ids:      [B, 16]   — character-encoded gene symbol

    Output:
        logits:       [B, 3, 6640]
    """

    def __init__(self, hidden_dim: int = 384, dropout: float = 0.5):
        super().__init__()

        # Character-level CNN for gene symbol
        self.symbol_cnn = SymbolCNN(vocab_size=len(CHAR_VOCAB), embed_dim=32, out_dim=64)

        # Learned embedding for genes absent from STRING (replaces zero vector)
        self.string_missing_emb = nn.Parameter(torch.zeros(STRING_HIDDEN_DIM))

        # Per-stream normalization before fusion
        self.aido_norm = nn.LayerNorm(AIDO_HIDDEN_DIM * 2)    # 512
        self.string_norm = nn.LayerNorm(STRING_HIDDEN_DIM)     # 256

        # 3-stage MLP head: 896 → hidden_dim → 256 → N_CLASSES*N_GENES
        # Mirrors tree-best node3-2's proven 3-stage structure
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, N_CLASSES * N_GENES),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize missing embedding close to zero (neutral prior)
        nn.init.normal_(self.string_missing_emb, mean=0.0, std=0.01)

    def forward(
        self,
        aido_feats: torch.Tensor,    # [B, 512]
        string_feats: torch.Tensor,  # [B, 256]
        sym_ids: torch.Tensor,       # [B, MAX_SYM_LEN]
    ) -> torch.Tensor:
        # Detect genes missing from STRING (stored as zero vectors in pre-computation)
        string_is_zero = (string_feats.abs().sum(dim=-1, keepdim=True) == 0)  # [B, 1] bool

        # Replace zero vectors with learned "missing gene" embedding
        missing_emb = self.string_missing_emb.unsqueeze(0).expand(string_feats.shape[0], -1)
        string_feats = torch.where(string_is_zero, missing_emb.to(string_feats.dtype), string_feats)

        # Normalize each feature stream for stable fusion
        aido_feats = self.aido_norm(aido_feats.float())
        string_feats = self.string_norm(string_feats.float())

        # Symbol CNN
        sym_feats = self.symbol_cnn(sym_ids)  # [B, 128]

        # Feature fusion and prediction
        fused = torch.cat([aido_feats, string_feats, sym_feats], dim=-1)  # [B, 896]
        logits = self.head(fused)                                          # [B, 3*6640]
        return logits.view(-1, N_CLASSES, N_GENES)                         # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 384,
        dropout: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 5e-2,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.10,
        t_max: int = 150,
        eta_min: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[DEGModel] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators for val/test steps
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = DEGModel(
                hidden_dim=self.hparams.hidden_dim,
                dropout=self.hparams.dropout,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )
            # All parameters trainable — cast to float32 for stable optimization
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        # Grab test metadata from DataModule
        if stage == "test":
            if self.trainer is not None and hasattr(self.trainer, "datamodule"):
                dm = self.trainer.datamodule
                if dm is not None and hasattr(dm, "test_pert_ids"):
                    self._test_pert_ids = dm.test_pert_ids
                    self._test_symbols = dm.test_symbols

    def _forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            aido_feats=batch["aido_feats"],
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

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self._forward(batch)
        loss = self._compute_loss(logits, batch["label"])
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

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)
        local_idx = torch.cat(self._test_indices, dim=0)

        # DDP: gather predictions from all ranks
        all_preds = self.all_gather(local_preds)  # [world_size, N_local, 3, G]
        all_idx = self.all_gather(local_idx)       # [world_size, N_local]

        self._test_preds.clear()
        self._test_indices.clear()

        if self.trainer.is_global_zero:
            preds = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            idxs = all_idx.view(-1).cpu().numpy()

            # Deduplicate (DDP may duplicate samples at dataset boundaries)
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
        # Separate param groups: head parameters get higher LR, others standard LR
        head_params = list(self.model.head.parameters())
        other_params = [
            p for p in self.model.parameters()
            if p.requires_grad and not any(p is hp for hp in head_params)
        ]

        param_groups = [
            {"params": head_params, "lr": self.hparams.lr, "weight_decay": self.hparams.weight_decay},
            {"params": other_params, "lr": self.hparams.lr * 0.5, "weight_decay": self.hparams.weight_decay},
        ]
        opt = torch.optim.AdamW(param_groups)

        # CosineAnnealingLR: clean monotonic decay from lr to eta_min over T_max epochs
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
        description="Node 1-2: Frozen AIDO.Cell-10M + STRING PPI + Char CNN DEG predictor"
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.10)
    p.add_argument("--t-max", type=int, default=150,
                   help="CosineAnnealingLR T_max (epochs)")
    p.add_argument("--eta-min", type=float, default=1e-6,
                   help="CosineAnnealingLR minimum learning rate")
    p.add_argument("--early-stopping-patience", type=int, default=30)
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
        # find_unused_parameters=False since all model params are used (no frozen backbone)
        strategy = DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120))

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-{epoch:03d}-{val_f1:.4f}",
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
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        t_max=args.t_max,
        eta_min=args.eta_min,
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
