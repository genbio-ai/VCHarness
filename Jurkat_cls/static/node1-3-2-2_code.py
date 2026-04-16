#!/usr/bin/env python3
"""
Node 1-3-2-2: LoRA AIDO.Cell-10M (r=4, ALL 8 layers) + STRING PPI + Char CNN
==============================================================================
Strategy:
  - LoRA fine-tuning on AIDO.Cell-10M (r=4, all 8 Q/K/V layers) — live backbone
  - Pre-computed STRING GNN PPI embeddings (frozen, run once) — STRING pre-computed
  - 64-dim Character-level CNN on gene symbol (matching tree-best node2-2-3 architecture)
  - Single-stage MLP head: 832→320→19920 (reduced from 384→320 to balance capacity/overfitting)
  - Class weights [7.0, 1.0, 15.0] (matching tree-best node2-2-3-1-1-1 configuration)
  - CosineAnnealingLR (T_max=40, eta_min=1e-6) — eliminates ReduceLROnPlateau instability
  - Dropout 0.55 in head (between parent's 0.5 and sibling's 0.6)
  - Uniform weight_decay 4e-2 (slight increase from 3e-2)
  - Top-5 checkpoint averaging at test time (more robust than top-3)

Key differences from parent (node1-3-2, test F1=0.4344):
  - CosineAnnealingLR replaces ReduceLROnPlateau (fixes LR scheduling instability)
  - Head dim 384→320 (reduced capacity to combat 3.4M-param head overfitting on 1,500 samples)
  - Dropout 0.5→0.55 (moderate increase in regularization)
  - weight_decay 3e-2→4e-2 (stronger L2 regularization)
  - Class weights [5,1,10]→[7,1,15] (tree-best proven configuration)
  - head LR 9e-4→3e-4 (equal to backbone LR, prevents head memorization)
  - Top-5 checkpoint averaging (vs top-3)

Key differences from sibling (node1-3-2-1, test F1=0.4479):
  - CosineAnnealingLR (T_max=40) vs ReduceLROnPlateau on val_loss
  - Head dim 320 vs 384 (sibling retained 384, we reduce for better regularization)
  - Dropout 0.55 vs 0.5 (slightly more aggressive)
  - weight_decay 4e-2 vs 3e-2 (slightly stronger)
  - head LR 3e-4 vs 4.5e-4 (lower, equal to backbone)
  - Top-5 checkpoint averaging vs top-3

Key differences from node1-3-2-1-1 (nephew, test F1=0.4468):
  - Head dim 320 vs 256 (restore some capacity, 256 was slightly too small)
  - Dropout 0.55 vs 0.6 (less aggressive, 0.6 was over-regularizing)
  - Uniform weight_decay 4e-2 vs separate head wd 5e-2 (simpler, less aggressive)
  - T_max=40 vs T_max=30 (longer cosine cycle, delayed minimum)
  - Top-5 checkpoint averaging vs top-3

Architecture:
  Pre-computed (frozen, STRING GNN runs once):
    STRING GNN → lookup [18870, 256] → perturbed gene emb [256]
  Online (trainable, ~3.5M params):
    AIDO.Cell-10M (LoRA r=4, all 8 layers) → dual-pool:
      [gene_pos_emb(256), mean_pool(256)] = [512]
    Char CNN: symbol → [64]
  Fusion: cat([512, 256, 64]) = [832]
  Head: LN(832) → 320 → GELU → Drop(0.55) → LN(320) → 19920 → [3, 6640]

Note: CosineAnnealingLR avoids the ReduceLROnPlateau problem where val_loss was
monotonically increasing and the scheduler fired prematurely/spuriously.
The cosine schedule provides smooth, predictable LR decay that does not depend on
any specific validation metric.
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
from peft import LoraConfig, get_peft_model, TaskType

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
N_GENES = 6_640          # output genes
N_CLASSES = 3            # {0=down, 1=unchanged, 2=up}

# Model paths
AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

# Feature dimensions
AIDO_HIDDEN_DIM = 256    # AIDO.Cell-10M hidden dimension
STRING_HIDDEN_DIM = 256  # STRING GNN output dimension
SYM_CNN_DIM = 64         # Symbol CNN output (32 * 2 from 2 conv kernels)
# Fusion: dual-pool AIDO (512) + STRING (256) + sym CNN (64) = 832
FUSION_DIM = AIDO_HIDDEN_DIM * 2 + STRING_HIDDEN_DIM + SYM_CNN_DIM  # 832

# Class weights: [7.0, 1.0, 15.0] — matching tree-best node2-2-3-1-1-1
# Proven progression: [5,1,10]→0.4592, [6,1,12]→0.4625, [7,1,15]→0.4655
CLASS_WEIGHTS = torch.tensor([7.0, 1.0, 15.0], dtype=torch.float32)

# Character vocabulary for gene symbol CNN
CHAR_VOCAB = {c: i + 1 for i, c in enumerate(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.abcdefghijklmnopqrstuvwxyz"
)}
CHAR_VOCAB["<pad>"] = 0
MAX_SYM_LEN = 16  # longest gene symbol we expect


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
    y_pred:          [N, 3, G] float (probabilities/logits)
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
# STRING Feature Pre-computation (runs once, cached to disk)
# Only STRING is pre-computed; AIDO runs live with LoRA during training.
# ──────────────────────────────────────────────────────────────────────────────
def precompute_string_features(data_dir: Path, cache_dir: Path) -> Dict[str, np.ndarray]:
    """
    Pre-compute frozen STRING GNN PPI features for all data splits.
    Uses file caching: loads from disk if cache exists; otherwise computes and saves.
    DDP-safe: caller must ensure only rank 0 calls this before a distributed barrier.

    Only STRING features are pre-computed here.
    AIDO features run live with LoRA during training (not pre-computed).
    """
    cache_path = cache_dir / "string_cache.npz"
    if cache_path.exists():
        print(f"[StringPreCompute] Loading cached STRING features from {cache_path}")
        return dict(np.load(str(cache_path), allow_pickle=True))

    print("[StringPreCompute] Computing STRING GNN features (this runs once, ~1 min)...")
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

    # Run STRING GNN once to get all node embeddings [18870, 256]
    with torch.no_grad():
        string_outputs = string_model(edge_index=edge_index, edge_weight=edge_weight)
    all_string_embs = string_outputs.last_hidden_state.float().cpu().numpy()  # [18870, 256]
    print(f"[StringPreCompute] STRING GNN: {all_string_embs.shape[0]} node embeddings extracted")

    # Process each data split
    results: Dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        tsv_path = data_dir / f"{split}.tsv"
        if not tsv_path.exists():
            continue
        df = pd.read_csv(tsv_path, sep="\t")
        pert_ids: List[str] = df["pert_id"].tolist()
        N = len(pert_ids)

        # STRING PPI features [N, 256]
        # Zero vector for genes not in STRING (replaced by learned embedding in model)
        string_feats = np.zeros((N, STRING_HIDDEN_DIM), dtype=np.float32)
        string_found = 0
        for j, pid in enumerate(pert_ids):
            eid = pid.split(".")[0]
            idx = string_name_to_idx.get(eid)
            if idx is not None:
                string_feats[j] = all_string_embs[idx]
                string_found += 1
        print(
            f"[StringPreCompute] '{split}': {string_found}/{N} "
            f"({100 * string_found / N:.1f}%) STRING genes found"
        )
        results[f"{split}_string"] = string_feats.astype(np.float32)

    # Save cache
    np.savez(str(cache_path), **results)
    print(f"[StringPreCompute] Cache saved → {cache_path}")

    # Free GPU memory
    del string_model
    torch.cuda.empty_cache()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Dataset: live AIDO tokenization + pre-computed STRING features
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    DEG dataset combining live AIDO.Cell-10M tokenization (for LoRA training)
    with pre-computed STRING PPI features (frozen, from cache).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        string_feats: np.ndarray,   # [N, 256] float32
        tokenizer,
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.string_feats = torch.from_numpy(string_feats.astype(np.float32))  # [N, 256]
        self.tokenizer = tokenizer
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
        pert_id = self.pert_ids[idx]
        symbol = self.symbols[idx]

        # AIDO tokenization: set perturbed gene to 1.0, all others filled with -1.0 (missing)
        # Single-sample tokenization: returns [19264] tensors (no batch dim, or squeezed)
        token_inputs = self.tokenizer(
            {"gene_ids": [pert_id], "expression": [1.0]},
            return_tensors="pt",
        )
        input_ids = token_inputs["input_ids"]
        attention_mask = token_inputs["attention_mask"]
        # Ensure 1D shape [19264] (squeeze if batch dim accidentally added)
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)

        item = {
            "idx": idx,
            "pert_id": pert_id,
            "input_ids": input_ids,                     # [19264] float32
            "attention_mask": attention_mask,            # [19264] int64
            "string_feats": self.string_feats[idx],     # [256] float32
            "sym_ids": encode_symbol(symbol),            # [MAX_SYM_LEN] int64
        }
        if not self.is_test:
            item["label"] = self.labels[idx]            # [6640] int64
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
        micro_batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

        self.tokenizer = None
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []
        self._string_cache: Optional[Dict[str, np.ndarray]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Step 1: Load tokenizer (rank 0 downloads first, then all ranks load)
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        # Step 2: Pre-compute STRING features (rank 0 computes if not cached, then barrier)
        cache_path = self.cache_dir / "string_cache.npz"
        if local_rank == 0 and not cache_path.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            precompute_string_features(self.data_dir, self.cache_dir)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # All ranks load STRING cache
        if self._string_cache is None:
            self._string_cache = dict(np.load(str(cache_path), allow_pickle=True))

        # Step 3: Build datasets
        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self._string_cache["train_string"], self.tokenizer
            )
            self.val_ds = PerturbationDataset(
                val_df, self._string_cache["val_string"], self.tokenizer
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self._string_cache["test_string"], self.tokenizer, is_test=True
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
    """Character-level multi-scale CNN for gene symbol encoding.
    Output: 64-dim (matching tree-best node2-2-3 architecture, NOT 128-dim).
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, out_dim, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(out_dim * 2)
        self.out_dim = out_dim * 2  # 64

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, L] → [B, 64]"""
        x = self.embedding(ids).transpose(1, 2)         # [B, embed_dim, L]
        f3 = F.gelu(self.conv3(x)).max(dim=-1).values   # [B, out_dim]
        f5 = F.gelu(self.conv5(x)).max(dim=-1).values   # [B, out_dim]
        return self.norm(torch.cat([f3, f5], dim=-1))   # [B, 64]


class DEGModel(nn.Module):
    """
    DEG predictor with live LoRA-tuned AIDO.Cell-10M backbone.

    Feature pipeline:
      AIDO (live, LoRA r=4 all-8-layers):
        last_hidden_state [B, 19266, 256] → dual-pool:
          gene_pos_emb [B, 256] + mean_emb [B, 256] → [B, 512]
      STRING (pre-computed, from batch):
        string_feats [B, 256] → replace zeros with learned missing_emb
      Char CNN (online):
        sym_ids [B, 16] → [B, 64]
      Fusion:
        cat([512, 256, 64]) = [B, 832]
      Head (reduced capacity for better generalization):
        LN(832) → Linear(832, 320) → GELU → Dropout(0.55) → LN(320) → Linear(320, 19920)
        → reshape [B, 3, 6640]

    Key change vs parent: hidden_dim=320 (vs 384). This reduces head params from
    ~3.4M to ~2.8M, improving regularization on the 1,500-sample training set.
    Combined with dropout=0.55 (vs 0.5), provides moderate additional regularization
    without over-constraining the model (unlike node1-3-2-1-1's head_dim=256+dropout=0.6).
    """

    def __init__(self, hidden_dim: int = 320, dropout: float = 0.55):
        super().__init__()

        # Symbol CNN: 64-dim output (not 128!) matching node2-2-3 architecture
        self.symbol_cnn = SymbolCNN(vocab_size=len(CHAR_VOCAB), embed_dim=32, out_dim=32)

        # Learned embedding for genes absent from STRING (zero-vector fallback → learned vector)
        self.string_missing_emb = nn.Parameter(torch.zeros(STRING_HIDDEN_DIM))

        # Per-stream normalization before fusion
        self.aido_norm = nn.LayerNorm(AIDO_HIDDEN_DIM * 2)   # LN(512)
        self.string_norm = nn.LayerNorm(STRING_HIDDEN_DIM)    # LN(256)

        # Single-stage MLP head: 832 → 320 → 19920
        # Reduced from 384 to 320 to lower head capacity (2.8M vs 3.4M params)
        # This better matches the 1,500-sample training set capacity.
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),                    # LN(832)
            nn.Linear(FUSION_DIM, hidden_dim),            # 832 → 320
            nn.GELU(),
            nn.Dropout(dropout),                          # 0.55 (vs 0.5 in parent)
            nn.LayerNorm(hidden_dim),                     # LN(320)
            nn.Linear(hidden_dim, N_CLASSES * N_GENES),   # 320 → 19920
        )

        self._init_weights()

        # Set externally after build_backbone() is called
        self.backbone = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize missing embedding close to zero (neutral prior for unknown genes)
        nn.init.normal_(self.string_missing_emb, mean=0.0, std=0.01)

    def build_backbone(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
    ):
        """
        Build AIDO.Cell-10M with LoRA on ALL 8 layers (Q/K/V).
        r=4 (vs parent's r=8) reduces overfitting on 1,500 samples.
        All 8 layers (vs parent's last 6) provides richer adaptation.
        """
        backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        # CRITICAL: Cast to bfloat16 BEFORE applying LoRA.
        # AIDO.Cell uses FlashAttention when dtype ∈ {fp16, bf16}.
        backbone = backbone.to(torch.bfloat16)

        # LoRA on ALL 8 layers (layers_to_transform=None means all layers)
        # r=4 provides less capacity than parent's r=8, reducing overfitting
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,  # ALL 8 layers (node2-2-3 proven configuration)
        )
        backbone = get_peft_model(backbone, lora_cfg)

        # Disable KV cache and enable gradient checkpointing for memory efficiency
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA adapter params to float32 for stable gradient updates.
        # Base backbone weights stay bfloat16 (ensures FlashAttention activation).
        for name, param in backbone.named_parameters():
            if param.requires_grad and "lora_" in name:
                param.data = param.data.float()

        self.backbone = backbone

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32
        attention_mask: torch.Tensor,  # [B, 19264] int64
        string_feats: torch.Tensor,    # [B, 256] float32
        sym_ids: torch.Tensor,         # [B, MAX_SYM_LEN] int64
        pert_positions: Optional[torch.Tensor] = None,  # [B] int64
    ) -> torch.Tensor:
        """Returns logits [B, 3, N_GENES]"""
        # ── AIDO backbone (live, with LoRA) ────────────────────────────────────
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last_hidden_state: [B, 19266, 256] (G+2 = 19264 gene positions + 2 summary tokens)
        last_hidden = outputs.last_hidden_state
        gene_hidden = last_hidden[:, :19264, :]  # [B, 19264, 256] — gene positions only

        # Global mean-pool over all gene positions
        mean_emb = gene_hidden.mean(dim=1)  # [B, 256]

        # Perturbed gene positional embedding (gene-specific signal from dual-pool)
        if pert_positions is not None:
            B = input_ids.shape[0]
            pos_idx = pert_positions.clamp(0, 19263).unsqueeze(1).unsqueeze(2).expand(B, 1, 256)
            gene_pos_emb = gene_hidden.gather(1, pos_idx).squeeze(1)  # [B, 256]
        else:
            gene_pos_emb = mean_emb  # fallback for unknown genes

        # Dual-pool: [gene_pos_emb, mean_emb] → [B, 512], cast bf16→f32
        aido_feats = torch.cat([gene_pos_emb, mean_emb], dim=-1).float()

        # ── STRING features (from pre-computed batch data) ─────────────────────
        # Detect genes absent from STRING (stored as zero vectors in pre-computation)
        string_is_zero = (string_feats.abs().sum(dim=-1, keepdim=True) == 0)  # [B, 1]
        # Replace zero vectors with learned "missing gene" embedding
        missing_emb = self.string_missing_emb.unsqueeze(0).expand(string_feats.shape[0], -1)
        string_feats = torch.where(
            string_is_zero,
            missing_emb.to(string_feats.dtype),
            string_feats,
        )

        # ── Per-stream normalization before fusion ─────────────────────────────
        aido_feats = self.aido_norm(aido_feats.float())
        string_feats = self.string_norm(string_feats.float())

        # ── Symbol CNN features ────────────────────────────────────────────────
        sym_feats = self.symbol_cnn(sym_ids)  # [B, 64]

        # ── Fusion: [B, 832] ───────────────────────────────────────────────────
        fused = torch.cat([aido_feats, string_feats, sym_feats], dim=-1)

        # ── Head: [B, 3, 6640] ─────────────────────────────────────────────────
        logits = self.head(fused)                       # [B, 3*6640]
        return logits.view(-1, N_CLASSES, N_GENES)      # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint Averaging Utility
# ──────────────────────────────────────────────────────────────────────────────
def get_top_k_checkpoints(checkpoint_cb: ModelCheckpoint, k: int = 5) -> List[str]:
    """Extract top-k checkpoint paths sorted by val_f1 (descending)."""
    if not hasattr(checkpoint_cb, 'best_k_models') or not checkpoint_cb.best_k_models:
        best = checkpoint_cb.best_model_path
        return [best] if best and Path(best).exists() else []

    # Sort by score descending (val_f1 maximization)
    sorted_items = sorted(
        checkpoint_cb.best_k_models.items(),
        key=lambda x: float(x[1]),
        reverse=(checkpoint_cb.mode == "max"),
    )
    valid_paths = [p for p, _ in sorted_items if Path(p).exists()]
    return valid_paths[:k]


def average_checkpoint_weights(
    ckpt_paths: List[str],
    model_module: "DEGLightningModule",
    output_path: Path,
    local_rank: int = 0,
) -> str:
    """
    Load top-k checkpoints, average their trainable parameter weights,
    and save the averaged checkpoint to output_path.
    DDP-safe: All ranks compute the average (same result), but only local_rank=0 saves.
    Returns the path to the averaged checkpoint.
    """
    if len(ckpt_paths) == 1:
        if local_rank == 0:
            print(f"[CheckpointAvg] Only 1 checkpoint found, using best directly.")
        return ckpt_paths[0]

    if local_rank == 0:
        print(f"[CheckpointAvg] Averaging {len(ckpt_paths)} checkpoints:")
        for p in ckpt_paths:
            print(f"  - {p}")

    # Load all state dicts on all ranks (safe since state dicts are small)
    state_dicts = []
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location="cpu")
        state_dicts.append(ckpt["state_dict"])

    # Average all parameters element-wise
    avg_state: Dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        tensors = [sd[key].float() for sd in state_dicts if key in sd]
        if tensors:
            avg_state[key] = torch.stack(tensors).mean(0)

    # Only rank 0 saves the averaged checkpoint
    if local_rank == 0:
        # Build averaged checkpoint (reuse structure from best checkpoint)
        base_ckpt = torch.load(ckpt_paths[0], map_location="cpu")
        base_ckpt["state_dict"] = avg_state
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(base_ckpt, str(output_path))
        print(f"[CheckpointAvg] Averaged checkpoint saved → {output_path}")

    return str(output_path)


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 320,
        dropout: float = 0.55,
        lr_backbone: float = 3e-4,
        lr_other: float = 4.5e-4,
        lr_head: float = 3e-4,
        weight_decay: float = 4e-2,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        cosine_t_max: int = 40,
        cosine_eta_min: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[DEGModel] = None
        self.criterion: Optional[FocalLoss] = None
        self._tokenizer = None

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._pert_pos_cache: Dict[str, Optional[int]] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = DEGModel(
                hidden_dim=self.hparams.hidden_dim,
                dropout=self.hparams.dropout,
            )
            self.model.build_backbone(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )
            # Cast non-backbone trainable params (head + sym CNN + missing emb) to float32
            # NOTE: LoRA backbone params remain in bfloat16 to preserve FlashAttention
            for name, param in self.model.named_parameters():
                if param.requires_grad and "backbone" not in name:
                    param.data = param.data.float()

        # Get tokenizer reference from DataModule (needed for positional lookup)
        if self.trainer is not None and hasattr(self.trainer, "datamodule"):
            dm = self.trainer.datamodule
            if dm is not None and hasattr(dm, "tokenizer") and dm.tokenizer is not None:
                self._tokenizer = dm.tokenizer

        # Get test metadata from DataModule
        if stage == "test":
            if self.trainer is not None and hasattr(self.trainer, "datamodule"):
                dm = self.trainer.datamodule
                if dm is not None and hasattr(dm, "test_pert_ids"):
                    self._test_pert_ids = dm.test_pert_ids
                    self._test_symbols = dm.test_symbols

    def _get_pert_gene_pos(self, pert_id: str) -> Optional[int]:
        """Get gene position in AIDO vocabulary for dual-pool extraction."""
        if self._tokenizer is None:
            return None
        eid = pert_id.split(".")[0]
        if hasattr(self._tokenizer, "gene_id_to_index"):
            pos = self._tokenizer.gene_id_to_index.get(eid)
            if pos is not None:
                return int(pos)
        if hasattr(self._tokenizer, "gene_to_index"):
            pos = self._tokenizer.gene_to_index.get(eid)
            if pos is not None:
                return int(pos)
        return None

    def _get_pert_positions(self, pert_ids: List[str]) -> Optional[torch.Tensor]:
        """Get gene positions for a batch of pert_ids (cached)."""
        if self._tokenizer is None:
            return None
        positions = []
        has_any = False
        for pid in pert_ids:
            if pid not in self._pert_pos_cache:
                self._pert_pos_cache[pid] = self._get_pert_gene_pos(pid)
            pos = self._pert_pos_cache[pid]
            if pos is not None:
                positions.append(pos)
                has_any = True
            else:
                positions.append(0)  # fallback to position 0
        if not has_any:
            return None
        return torch.tensor(positions, dtype=torch.long)

    def _forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        pert_ids = batch.get("pert_id")
        pert_ids_list = list(pert_ids) if isinstance(pert_ids, (list, tuple)) else None

        pert_positions = None
        if pert_ids_list is not None:
            pos_cpu = self._get_pert_positions(pert_ids_list)
            if pos_cpu is not None:
                pert_positions = pos_cpu.to(batch["input_ids"].device)

        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            string_feats=batch["string_feats"],
            sym_ids=batch["sym_ids"],
            pert_positions=pert_positions,
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

        # Scalar all-reduce across DDP ranks (efficient, no large tensor all_gather)
        world_size = self.trainer.world_size if self.trainer.world_size else 1
        if world_size > 1:
            import torch.distributed as dist
            f1_t = torch.tensor(local_f1, dtype=torch.float32, device="cuda")
            dist.all_reduce(f1_t, op=dist.ReduceOp.SUM)
            f1 = (f1_t / world_size).item()
        else:
            f1 = local_f1

        # Note: In LoRA+STRING regime, val_f1 depends on class weight configuration.
        # With CosineAnnealingLR (vs ReduceLROnPlateau), val_f1 should better track test F1.
        # node1-3-2-1-1 (CosineAnnealingLR) achieved val_f1=0.4478 ≈ test F1=0.4468 (near-zero gap)
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

            # Deduplicate (DDP may replicate samples at dataset boundaries)
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
        # 3-tier differential LR:
        #   backbone LoRA params:    lr = lr_backbone (3e-4)
        #   other trainable params:  lr = lr_other    (4.5e-4, 1.5× backbone)
        #   head params:             lr = lr_head     (3e-4, = backbone LR)
        # NOTE: head LR reduced to 3e-4 (same as backbone) to prevent head overfitting.
        # Previous head LR of 9e-4 (3× backbone) caused catastrophic overfitting.
        # 4.5e-4 (sibling node1-3-2-1) was an improvement but still too high.
        # Equal LR (3e-4) ensures head adaptation rate matches backbone adaptation.
        backbone_lora_params = [
            p for n, p in self.model.backbone.named_parameters()
            if p.requires_grad and "lora_" in n
        ]
        head_params = list(self.model.head.parameters())
        other_params = [
            p for p in self.model.parameters()
            if p.requires_grad
            and not any(p is bp for bp in backbone_lora_params)
            and not any(p is hp for hp in head_params)
        ]

        param_groups = [
            {
                "params": backbone_lora_params,
                "lr": self.hparams.lr_backbone,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": other_params,
                "lr": self.hparams.lr_other,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": head_params,
                "lr": self.hparams.lr_head,
                "weight_decay": self.hparams.weight_decay,
            },
        ]
        opt = torch.optim.AdamW(param_groups)

        # CosineAnnealingLR (T_max=40, eta_min=1e-6) — KEY CHANGE from ReduceLROnPlateau
        # Rationale: In the parent and sibling, val_loss was monotonically increasing from
        # epoch 1 onward. ReduceLROnPlateau monitoring val_loss was therefore firing because
        # "loss is always going up", not because it genuinely plateaued. This means the
        # scheduler was making poor decisions. CosineAnnealingLR provides smooth, predictable
        # LR decay that does not depend on any specific validation metric.
        # Evidence: node1-3-2-1-1 (CosineAnnealingLR T_max=30) achieved near-zero val-test gap
        # (+0.001), confirming CosineAnnealingLR is the right choice.
        # T_max=40 (vs 30 in node1-3-2-1-1): Longer cycle gives more epochs at moderate LR
        # before reaching the low-LR plateau, allowing more training before convergence.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams.cosine_t_max,
            eta_min=self.hparams.cosine_eta_min,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable params + buffers (memory efficient) ──
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
        description="Node 1-3-2-2: LoRA AIDO.Cell-10M (r=4, all-8-layers) + STRING PPI + Char CNN"
    )
    p.add_argument("--data-dir", type=str,
                   default=str(Path(__file__).parent.parent.parent / "data"))
    p.add_argument("--micro-batch-size", type=int, default=4,
                   help="Per-GPU batch size (AIDO-10M LoRA memory: ~3.21 GiB/sample)")
    p.add_argument("--global-batch-size", type=int, default=32,
                   help="Effective batch size; accumulate_grad = global/(micro*n_gpus)")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--lr-backbone", type=float, default=3e-4,
                   help="LR for LoRA adapter params (backbone)")
    p.add_argument("--lr-other", type=float, default=4.5e-4,
                   help="LR for sym CNN + string missing emb (1.5× backbone)")
    p.add_argument("--lr-head", type=float, default=3e-4,
                   help="LR for MLP head params (= backbone LR, prevents head memorization)")
    p.add_argument("--weight-decay", type=float, default=4e-2,
                   help="AdamW weight decay (increased from 3e-2 for stronger regularization)")
    p.add_argument("--hidden-dim", type=int, default=320,
                   help="MLP head hidden dimension (832→hidden_dim→19920, reduced from 384)")
    p.add_argument("--dropout", type=float, default=0.55,
                   help="Head dropout (between parent's 0.5 and node1-3-2-1-1's 0.6)")
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lora-r", type=int, default=4,
                   help="LoRA rank (r=4 proven in node2-2-3 lineage; less overfitting than r=8)")
    p.add_argument("--lora-alpha", type=int, default=8,
                   help="LoRA alpha (standard 2× rank)")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--cosine-t-max", type=int, default=40,
                   help="CosineAnnealingLR T_max (longer than node1-3-2-1-1's T_max=30)")
    p.add_argument("--cosine-eta-min", type=float, default=1e-6,
                   help="CosineAnnealingLR minimum LR")
    p.add_argument("--early-stopping-patience", type=int, default=20,
                   help="EarlyStopping patience on val_f1 (generous for cosine schedule)")
    p.add_argument("--top-k-avg", type=int, default=5,
                   help="Number of top checkpoints to average at test time (increased from 3)")
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
    cache_dir = output_dir / "string_cache"
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
        # find_unused_parameters=True: frozen backbone parameters (non-LoRA) not in graph
        strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    # Save top-k checkpoints by val_f1 for checkpoint averaging
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-3-2-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=args.top_k_avg,  # save top-5 for checkpoint averaging
        save_last=True,
    )
    # EarlyStopping on val_f1 (generous patience for cosine schedule)
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stopping_patience,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    num_sanity = 0 if args.fast_dev_run else 2

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
        num_sanity_val_steps=num_sanity,
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
        lr_backbone=args.lr_backbone,
        lr_other=args.lr_other,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        cosine_t_max=args.cosine_t_max,
        cosine_eta_min=args.cosine_eta_min,
    )

    # ── Training ───────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Testing (with checkpoint averaging) ───────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        # Quick debug: just run test with current model state
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        # Top-5 checkpoint averaging: load top-k checkpoints, average weights, then test
        # node1-3-2-1-1 confirmed CosineAnnealingLR gives near-zero val-test gap,
        # so top-5 checkpoints should represent the genuine generalization peak.
        top_k_paths = get_top_k_checkpoints(checkpoint_cb, k=args.top_k_avg)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        avg_ckpt_path = output_dir / "checkpoints" / "averaged_top5.ckpt"
        avg_ckpt = average_checkpoint_weights(
            top_k_paths, model_module, avg_ckpt_path, local_rank=local_rank
        )

        # All ranks wait for rank 0 to finish saving the averaged checkpoint
        if n_gpus > 1:
            import torch.distributed as dist
            dist.barrier()

        if avg_ckpt and Path(avg_ckpt).exists():
            test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path=avg_ckpt)
        else:
            # Fallback: use best checkpoint
            best_ckpt = checkpoint_cb.best_model_path if checkpoint_cb.best_model_path else "best"
            test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path=best_ckpt)

    # ── Save test score summary ────────────────────────────────────────────────
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        with open(score_path, "w") as f:
            f.write(f"Test results: {json.dumps(test_results, indent=2)}\n")
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
