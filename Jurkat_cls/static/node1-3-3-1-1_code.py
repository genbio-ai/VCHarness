#!/usr/bin/env python3
"""
Node 1-3-3-1-1: AIDO.Cell-10M (FROZEN backbone) + STRING GNN PPI + Symbol CNN +
               3-layer Cross-Attention TransformerEncoder Fusion + Val-Loss Checkpointing
=========================================================================================
Strategy:
  - AIDO.Cell-10M with FROZEN backbone (no LoRA) — eliminates ~4M trainable params that
    caused catastrophic overfitting in parent node1-3-3-1 (test F1=0.3979, val-test gap=-0.081)
  - STRING GNN PPI embeddings (256-dim) pre-computed once per split and cached to disk
  - Character-level CNN on gene symbol (3-branch, 64-dim)
  - 3-layer Cross-Attention TransformerEncoder Fusion (4 tokens, nhead=8, dim_ff=256, attn_dropout=0.2)
    Proven tree-best architecture from node3-1-3-1-1-1-1 (0.4768)
  - val_loss-based checkpoint selection (NOT val_f1) — critical fix for overfitting detection
  - val_f1 early stopping with patience=15
  - Manifold mixup (alpha=0.2) for regularization
  - Moderate class weights [3,1,7] (vs parent's over-aggressive [6,1,12])
  - weight_decay=0.10 (proven at tree-best node3-1-3-1-1-1-1)
  - Checkpoint averaging over top-3 checkpoints at test time

Key differences from parent (node1-3-3-1, LoRA last-4 + cross-attn, F1=0.3979):
  1. Backbone: LoRA fine-tuning -> FROZEN (eliminates overfitting root cause)
  2. Fusion layers: 2-layer -> 3-layer TransformerEncoder (proven +0.0008-0.0037 F1)
  3. Fusion nhead: 4 -> 8 (matches node3-1-3-1-1-1-1 tree-best)
  4. attn_dropout: 0.1 -> 0.2 (stronger regularization in cross-attention)
  5. Checkpoint monitor: val_f1 -> val_loss (critical for detecting overfitting)
  6. Class weights: [6,1,12] -> [3,1,7] (moderate, less val_f1 inflation)
  7. weight_decay: 3e-2 -> 1e-1 (stronger L2 regularization, proven at tree-best)
  8. Backbone LR: 2e-4 -> 0.0 (backbone frozen; only head/fusion trained)

Architecture reference:
  - node3-1-3-1-1-1-1 (0.4768): 3-layer cross-attention, wd=0.10, mixup=0.3, nhead=8
  - node3-1-3-1-1 (0.4739): 3-layer cross-attention fusion, dim_ff=384, wd=0.10, mixup=0.3
  - node3-1-3-1 (0.4731): 2-layer cross-attention fusion, wd=0.08, mixup=0.2
  - node1-3-3-1 (parent, 0.3979): LoRA last-4 + cross-attn 2-layer, catastrophic overfitting

Why frozen backbone works better here:
  - 1,500 training samples cannot support 4M+ LoRA parameters without memorization
  - Every frozen-backbone node achieves 0.41-0.48 test F1 with near-zero val-test gap
  - Every LoRA node in the node1-3 lineage suffers severe overfitting (val-test gap > 0.06)
  - AIDO.Cell-10M pretrained representations (from 50M cells) are already well-suited for this task
  - Pre-computing AIDO features removes backbone overfitting risk entirely
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
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
N_GENES = 6_640          # output genes
N_CLASSES = 3            # {0=down, 1=unchanged, 2=up}

AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_MODEL_DIR = "/home/Models/STRING_GNN"

# Class weights: [3.0, 1.0, 7.0] — moderate weights to avoid val_f1 inflation
# (parent's [6,1,12] contributed to catastrophic val-to-test gap of -0.081)
CLASS_WEIGHTS = torch.tensor([3.0, 1.0, 7.0], dtype=torch.float32)

# Character vocabulary for gene symbol CNN
CHAR_VOCAB = {c: i + 1 for i, c in enumerate(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.abcdefghijklmnopqrstuvwxyz"
)}
CHAR_VOCAB["<pad>"] = 0
MAX_SYM_LEN = 16


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
        """logits: [N, C], targets: [N] (int64, class indices)"""
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(
            logits, targets,
            weight=w,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def encode_symbol(symbol: str, max_len: int = MAX_SYM_LEN) -> torch.Tensor:
    """Encode a gene symbol as a padded integer sequence for CNN."""
    sym = symbol.upper()[:max_len]
    ids = [CHAR_VOCAB.get(c, 0) for c in sym]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that keeps pert_id as a list of strings."""
    result = {}
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


def pad_collate_fn(micro_batch_size: int):
    """
    Pads batches to micro_batch_size to ensure consistent tensor sizes across DDP ranks.
    Uses zero-filled dummy samples for padding. Tracks n_real for masking in the model.
    """
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        n_real = len(batch)
        needed = micro_batch_size
        while len(batch) < needed:
            batch.append(batch[0])  # Duplicate first sample as placeholder (will be masked)

        result = {}
        for key in batch[0]:
            if key == "pert_id":
                result[key] = [item[key] for item in batch]
                result["n_real"] = n_real
            elif isinstance(batch[0][key], torch.Tensor):
                result[key] = torch.stack([item[key] for item in batch])
                result["n_real"] = n_real
            elif isinstance(batch[0][key], int):
                result[key] = torch.tensor([item[key] for item in batch])
                result["n_real"] = n_real
            else:
                result[key] = [item[key] for item in batch]
                result["n_real"] = n_real
        return result
    return collate


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Single-fold perturbation -> DEG dataset with AIDO tokenization."""

    def __init__(self, df: pd.DataFrame, tokenizer, is_test: bool = False):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.tokenizer = tokenizer
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Remap: {-1->0, 0->1, 1->2}
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pert_id = self.pert_ids[idx]
        symbol = self.symbols[idx]

        # Tokenize: set perturbed gene to 1.0 (active), all others fill as -1.0 (missing)
        token_inputs = self.tokenizer(
            {"gene_ids": [pert_id], "expression": [1.0]},
            return_tensors="pt",
        )
        input_ids = token_inputs["input_ids"]
        attention_mask = token_inputs["attention_mask"]
        # Ensure 1D shape [19264]
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)

        item = {
            "idx": idx,
            "pert_id": pert_id,
            "input_ids": input_ids,          # [19264] float32
            "attention_mask": attention_mask, # [19264] int64
            "sym_ids": encode_symbol(symbol), # [MAX_SYM_LEN]
        }

        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)  # [6640]

        return item


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, micro_batch_size: int = 4, num_workers: int = 4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.tokenizer = None
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        # Load tokenizer: rank 0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self.tokenizer)
            self.val_ds = PerturbationDataset(val_df, self.tokenizer)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(test_df, self.tokenizer, is_test=True)
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
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            collate_fn=pad_collate_fn(self.micro_batch_size),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
            collate_fn=pad_collate_fn(self.micro_batch_size),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Character-Level CNN Encoder for Gene Symbol
# ──────────────────────────────────────────────────────────────────────────────
class SymbolCNN(nn.Module):
    """3-branch character-level CNN for gene symbol encoding (64-dim output)."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 3 branches with different kernel sizes
        self.conv3 = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, out_dim, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(embed_dim, out_dim, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim  # 64-dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, L] -> [B, out_dim]"""
        x = self.embedding(ids)         # [B, L, embed_dim]
        x = x.transpose(1, 2)          # [B, embed_dim, L]
        # Global max-pool for each branch, then element-wise max across branches
        f3 = F.gelu(self.conv3(x)).max(dim=-1).values   # [B, out_dim]
        f4 = F.gelu(self.conv4(x)).max(dim=-1).values   # [B, out_dim]
        f5 = F.gelu(self.conv5(x)).max(dim=-1).values   # [B, out_dim]
        out = torch.stack([f3, f4, f5], dim=0).max(dim=0).values  # [B, out_dim]
        return self.norm(out)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Attention TransformerEncoder Fusion (3-layer, nhead=8)
# ──────────────────────────────────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    3-layer TransformerEncoder self-attention over 4 feature tokens.
    Mirrors the proven tree-best configuration from node3-1-3-1-1-1-1 (0.4768):
      - 3 layers (vs parent's 2-layer)
      - nhead=8 (vs parent's 4)
      - dim_feedforward=256 (width-constrained, optimal per tree evidence)
      - attn_dropout=0.2 (stronger regularization)

    Input tokens (projected to d_model=256 each):
      - global_emb:  AIDO mean-pool [256] from frozen backbone
      - pert_emb:    AIDO pert-position [256] from frozen backbone
      - sym_feat:    Symbol CNN [64] -> projected to [256]
      - ppi_feat:    STRING PPI [256]

    These 4 tokens are stacked into [B, 4, 256], processed through
    TransformerEncoder layers, then mean-pooled to [B, 256].
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 256,
        n_layers: int = 3,
        attn_dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model

        # Projections for inputs that don't match d_model
        self.global_proj = nn.Linear(256, d_model)  # AIDO mean pool: 256->256
        self.pert_proj   = nn.Linear(256, d_model)  # AIDO pert pos: 256->256
        self.sym_proj    = nn.Linear(64, d_model)   # Symbol CNN: 64->256
        self.ppi_proj    = nn.Linear(256, d_model)  # STRING PPI: 256->256

        # Learnable position embeddings for the 4 tokens
        self.pos_emb = nn.Embedding(4, d_model)

        # TransformerEncoder: 3 layers, nhead=8, dim_ff=256, pre-norm, attn_dropout=0.2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stability on small datasets
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_dim = d_model  # 256

    def forward(
        self,
        global_emb: torch.Tensor,   # [B, 256] float32
        pert_emb: torch.Tensor,     # [B, 256] float32
        sym_feat: torch.Tensor,     # [B, 64] float32
        ppi_feat: torch.Tensor,     # [B, 256] float32
    ) -> torch.Tensor:
        """Returns [B, d_model] fused features."""
        B = global_emb.shape[0]
        device = global_emb.device

        # Project each source to d_model
        t0 = self.global_proj(global_emb)  # [B, 256]
        t1 = self.pert_proj(pert_emb)       # [B, 256]
        t2 = self.sym_proj(sym_feat)         # [B, 256]
        t3 = self.ppi_proj(ppi_feat)         # [B, 256]

        # Stack into [B, 4, 256]
        tokens = torch.stack([t0, t1, t2, t3], dim=1)  # [B, 4, d_model]

        # Add positional embeddings
        pos_ids = torch.arange(4, device=device)
        tokens = tokens + self.pos_emb(pos_ids).unsqueeze(0)  # [B, 4, d_model]

        # TransformerEncoder self-attention over the 4 tokens
        out = self.transformer(tokens)  # [B, 4, d_model]

        # Mean-pool over tokens
        fused = out.mean(dim=1)  # [B, d_model]
        return self.out_norm(fused)


# ──────────────────────────────────────────────────────────────────────────────
# STRING GNN Feature Pre-computation (run once per split on rank 0)
# ──────────────────────────────────────────────────────────────────────────────
def precompute_string_features(pert_ids: List[str], output_path: Path) -> np.ndarray:
    """
    Run frozen STRING GNN inference and look up node embeddings for each pert_id.
    Result: float32 [N, 256] array (zeros for unknown genes).
    Called on rank 0 only; results shared via cache file.
    """
    model_dir = Path(STRING_GNN_MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load STRING GNN (frozen)
    string_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    string_model = string_model.to(device)
    string_model.eval()

    # Load graph and node names
    graph = torch.load(model_dir / "graph_data.pt")
    node_names_text = (model_dir / "node_names.json").read_text()
    node_names = json.loads(node_names_text)
    node_name_to_idx = {n: i for i, n in enumerate(node_names)}

    edge_index = graph["edge_index"].to(device)
    edge_weight = graph.get("edge_weight")
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    # Run STRING GNN inference (single forward pass over entire graph)
    with torch.no_grad():
        outputs = string_model(edge_index=edge_index, edge_weight=edge_weight)
    all_emb = outputs.last_hidden_state.cpu().float().numpy()  # [18870, 256]

    # Free GPU memory
    del string_model
    torch.cuda.empty_cache()

    # Look up each pert_id
    emb_dim = all_emb.shape[1]  # 256
    result = np.zeros((len(pert_ids), emb_dim), dtype=np.float32)
    n_found = 0
    for i, pid in enumerate(pert_ids):
        # Strip version suffix (e.g., "ENSG00000001084.5" -> "ENSG00000001084")
        eid = pid.split(".")[0]
        node_idx = node_name_to_idx.get(eid)
        if node_idx is not None:
            result[i] = all_emb[node_idx]
            n_found += 1

    print(f"STRING GNN lookup: {n_found}/{len(pert_ids)} genes found in STRING graph")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), result)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# AIDO Feature Pre-computation (frozen backbone, run once per split on rank 0)
# ──────────────────────────────────────────────────────────────────────────────
def precompute_aido_features(
    pert_ids: List[str],
    tokenizer,
    global_output_path: Path,
    pert_output_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run frozen AIDO.Cell-10M inference to extract:
      - global_emb: mean-pool over all 19264 gene positions [N, 256]
      - pert_emb:   position-specific embedding for the perturbed gene [N, 256]
    Results are saved to disk for DDP-safe sharing.
    Called on rank 0 only.

    Using pre-computed AIDO features (frozen backbone) eliminates the LoRA overfitting
    that caused the parent node1-3-3-1's catastrophic val-test gap of -0.081.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load AIDO.Cell-10M (completely frozen)
    aido_model = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
    aido_model = aido_model.to(torch.bfloat16).to(device)
    aido_model.eval()
    for p in aido_model.parameters():
        p.requires_grad = False

    # Build gene_id -> position lookup from tokenizer
    pert_pos_cache: Dict[str, Optional[int]] = {}
    def get_pos(pid: str) -> Optional[int]:
        eid = pid.split(".")[0]
        if eid in pert_pos_cache:
            return pert_pos_cache[eid]
        pos = None
        if hasattr(tokenizer, "gene_id_to_index"):
            pos = tokenizer.gene_id_to_index.get(eid)
        elif hasattr(tokenizer, "gene_to_index"):
            pos = tokenizer.gene_to_index.get(eid)
        if pos is not None:
            pos = int(pos)
        pert_pos_cache[eid] = pos
        return pos

    N = len(pert_ids)
    global_embs = np.zeros((N, 256), dtype=np.float32)
    pert_embs   = np.zeros((N, 256), dtype=np.float32)

    # Process in small batches to avoid OOM
    batch_size = 16
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_ids = pert_ids[start:end]
        B_cur = end - start

        # Tokenize batch
        input_ids_list = []
        attn_list = []
        for pid in batch_ids:
            tok = tokenizer({"gene_ids": [pid], "expression": [1.0]}, return_tensors="pt")
            iids = tok["input_ids"]
            amask = tok["attention_mask"]
            if iids.dim() == 2:
                iids = iids.squeeze(0)
                amask = amask.squeeze(0)
            input_ids_list.append(iids)
            attn_list.append(amask)

        batch_input_ids = torch.stack(input_ids_list, dim=0).to(device)   # [B, 19264]
        batch_attn = torch.stack(attn_list, dim=0).to(device)             # [B, 19264]

        with torch.no_grad():
            outputs = aido_model(input_ids=batch_input_ids, attention_mask=batch_attn)
        last_hidden = outputs.last_hidden_state  # [B, 19266, 256] bfloat16

        # Gene positions only (exclude 2 appended summary tokens)
        gene_hidden = last_hidden[:, :19264, :].float()  # [B, 19264, 256] float32

        # Global mean-pool over all gene positions
        global_pool = gene_hidden.mean(dim=1).cpu().numpy()  # [B, 256]
        global_embs[start:end] = global_pool

        # Per-perturbation positional embedding
        for j, pid in enumerate(batch_ids):
            pos = get_pos(pid)
            if pos is not None:
                pos = min(max(pos, 0), 19263)
                pert_embs[start + j] = gene_hidden[j, pos, :].cpu().numpy()
            else:
                # Fallback: use global mean if position not found
                pert_embs[start + j] = global_pool[j]

        if (start // batch_size) % 5 == 0:
            print(f"AIDO feature extraction: {end}/{N} samples processed")

    # Free GPU memory
    del aido_model
    torch.cuda.empty_cache()

    global_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(global_output_path), global_embs)
    np.save(str(pert_output_path), pert_embs)
    print(f"AIDO features saved -> {global_output_path}, {pert_output_path}")
    return global_embs, pert_embs


# ──────────────────────────────────────────────────────────────────────────────
# Main Model: FROZEN AIDO.Cell-10M + STRING PPI + Symbol CNN +
#             3-layer CrossAttention Fusion + MLP Head
# ──────────────────────────────────────────────────────────────────────────────
class DEGModel(nn.Module):
    """
    4-source 3-layer cross-attention fusion for DEG prediction:
      - Pre-computed frozen AIDO.Cell-10M features:
          global_emb: [B, 256] + pert_emb: [B, 256]  (no backbone in forward pass)
      - Pre-computed STRING GNN PPI embeddings -> [B, 256]  (frozen, from cache)
      - Character-level CNN on gene symbol -> [B, 64]

    Fusion: 3-layer TransformerEncoder cross-attention (nhead=8, dim_ff=256) over 4 tokens -> [B, 256]
    Head: LayerNorm -> Linear(256, 256) -> GELU -> Dropout -> Linear(256, 3*6640)
    """

    def __init__(
        self,
        head_dim: int = 256,
        dropout: float = 0.4,
        attn_n_layers: int = 3,
        attn_nhead: int = 8,
        attn_dim_ff: int = 256,
        attn_dropout: float = 0.2,
    ):
        super().__init__()

        # Symbol CNN (3-branch, 64-dim output)
        char_vocab_size = len(CHAR_VOCAB)
        self.symbol_cnn = SymbolCNN(char_vocab_size, embed_dim=32, out_dim=64)

        # Learnable fallback for genes not found in STRING (zero-initialized, trainable)
        self.string_fallback = nn.Parameter(torch.zeros(256))

        # Cross-attention fusion over 4 tokens -> 256-dim
        # 3-layer, nhead=8, dim_ff=256, attn_dropout=0.2 matches node3-1-3-1-1-1-1
        self.fusion = CrossAttentionFusion(
            d_model=256,
            nhead=attn_nhead,
            dim_feedforward=attn_dim_ff,
            n_layers=attn_n_layers,
            attn_dropout=attn_dropout,
        )

        # Head: [B, 256] -> [B, head_dim] -> [B, 3*N_GENES]
        fusion_dim = self.fusion.out_dim  # 256
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, N_CLASSES * N_GENES),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _merge_string_feats(self, string_feats: torch.Tensor) -> torch.Tensor:
        """
        Replace all-zero STRING features (unknown genes) with learnable fallback.
        string_feats: [B, 256] float32
        Returns: [B, 256] float32
        """
        is_zero = (string_feats.abs().sum(dim=-1) == 0.0)  # [B] bool
        if is_zero.any():
            fallback = self.string_fallback.float().unsqueeze(0).expand_as(string_feats)
            string_feats = torch.where(is_zero.unsqueeze(1), fallback, string_feats)
        return string_feats

    def forward(
        self,
        global_emb: torch.Tensor,         # [B, 256] float32 (pre-computed AIDO global)
        pert_emb: torch.Tensor,            # [B, 256] float32 (pre-computed AIDO pert-pos)
        sym_ids: torch.Tensor,             # [B, MAX_SYM_LEN] int64
        string_feats: torch.Tensor,        # [B, 256] float32 (pre-computed STRING PPI)
    ) -> torch.Tensor:
        """Returns logits [B, 3, N_GENES]"""
        # Character CNN features: [B, 64]
        sym_features = self.symbol_cnn(sym_ids)

        # STRING PPI features: [B, 256] (with fallback for unknowns)
        string_merged = self._merge_string_feats(string_feats.to(global_emb.device))

        # 3-layer Cross-attention fusion over 4 tokens -> [B, 256]
        fused = self.fusion(
            global_emb=global_emb,
            pert_emb=pert_emb,
            sym_feat=sym_features,
            ppi_feat=string_merged,
        )

        # Head: [B, 256] -> [B, 3*N_GENES] -> [B, 3, N_GENES]
        logits = self.head(fused)
        return logits.view(-1, N_CLASSES, N_GENES)


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper (mirrors calc_metric.py)
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# Manifold Mixup Helper
# ──────────────────────────────────────────────────────────────────────────────
def manifold_mixup(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply manifold mixup on logits and labels.
    logits: [B, 3, G]
    labels: [B, G] (int64, class indices 0/1/2)
    Returns mixed logits and mixed one-hot labels.
    """
    if alpha <= 0.0:
        return logits, labels

    lam = np.random.beta(alpha, alpha)
    B = logits.shape[0]
    perm = torch.randperm(B, device=logits.device)

    mixed_logits = lam * logits + (1 - lam) * logits[perm]

    # One-hot encode labels for mixing
    B, G = labels.shape
    labels_oh = torch.zeros(B, 3, G, dtype=torch.float32, device=labels.device)
    labels_oh.scatter_(1, labels.unsqueeze(1), 1.0)
    labels_oh_perm = labels_oh[perm]
    mixed_labels_oh = lam * labels_oh + (1 - lam) * labels_oh_perm

    return mixed_logits, mixed_labels_oh


def compute_mixed_focal_loss(
    criterion: FocalLoss,
    logits: torch.Tensor,       # [B, 3, G]
    labels: Any,                # [B, G] int64 OR [B, 3, G] float (mixed one-hot)
) -> torch.Tensor:
    """Compute focal loss supporting both hard labels and soft (mixed) labels."""
    B, C, G = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]

    if isinstance(labels, torch.Tensor) and labels.dtype == torch.long:
        # Hard labels
        labels_flat = labels.reshape(-1)  # [B*G]
        return criterion(logits_flat, labels_flat)
    else:
        # Soft labels (manifold mixup outputs)
        labels_soft = labels.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        # Use KL divergence with soft targets
        log_probs = F.log_softmax(logits_flat, dim=-1)
        soft_targets = F.softmax(labels_soft, dim=-1)
        return F.kl_div(log_probs, soft_targets, reduction="batchmean")


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        head_dim: int = 256,
        dropout: float = 0.4,
        head_lr: float = 6e-4,
        weight_decay: float = 0.10,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        attn_n_layers: int = 3,
        attn_nhead: int = 8,
        attn_dim_ff: int = 256,
        attn_dropout: float = 0.2,
        mixup_alpha: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[DEGModel] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._test_labels: Optional[np.ndarray] = None  # Loaded from test.tsv for metric computation

        # Pre-computed AIDO and STRING feature arrays (loaded once per split)
        self._train_global_feats: Optional[np.ndarray] = None
        self._train_pert_feats: Optional[np.ndarray] = None
        self._val_global_feats: Optional[np.ndarray] = None
        self._val_pert_feats: Optional[np.ndarray] = None
        self._test_global_feats: Optional[np.ndarray] = None
        self._test_pert_feats: Optional[np.ndarray] = None
        self._train_string_feats: Optional[np.ndarray] = None
        self._val_string_feats: Optional[np.ndarray] = None
        self._test_string_feats: Optional[np.ndarray] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Build model once
        if self.model is None:
            self.model = DEGModel(
                head_dim=self.hparams.head_dim,
                dropout=self.hparams.dropout,
                attn_n_layers=self.hparams.attn_n_layers,
                attn_nhead=self.hparams.attn_nhead,
                attn_dim_ff=self.hparams.attn_dim_ff,
                attn_dropout=self.hparams.attn_dropout,
            )

            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

            # Cast all trainable params to float32 for stable optimization
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        output_dir = Path(__file__).parent / "run"
        cache_dir = output_dir / "feature_cache"
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        dm = self.trainer.datamodule if self.trainer is not None else None

        # Pre-compute or load AIDO + STRING features for fit stage
        if stage in ("fit", None) and self._train_global_feats is None and dm is not None:
            train_global_path = cache_dir / "train_aido_global.npy"
            train_pert_path   = cache_dir / "train_aido_pert.npy"
            val_global_path   = cache_dir / "val_aido_global.npy"
            val_pert_path     = cache_dir / "val_aido_pert.npy"
            train_string_path = cache_dir / "train_string.npy"
            val_string_path   = cache_dir / "val_string.npy"

            if local_rank == 0:
                cache_dir.mkdir(parents=True, exist_ok=True)
                if dm.train_ds is None:
                    dm.setup("fit")
                tokenizer = dm.tokenizer

                # Pre-compute AIDO features for train if not cached
                if not train_global_path.exists() or not train_pert_path.exists():
                    if dm.train_ds is not None:
                        precompute_aido_features(
                            dm.train_ds.pert_ids, tokenizer,
                            train_global_path, train_pert_path,
                        )

                # Pre-compute AIDO features for val if not cached
                if not val_global_path.exists() or not val_pert_path.exists():
                    if dm.val_ds is not None:
                        precompute_aido_features(
                            dm.val_ds.pert_ids, tokenizer,
                            val_global_path, val_pert_path,
                        )

                # Pre-compute STRING features if not cached
                if not train_string_path.exists() and dm.train_ds is not None:
                    precompute_string_features(dm.train_ds.pert_ids, train_string_path)
                if not val_string_path.exists() and dm.val_ds is not None:
                    precompute_string_features(dm.val_ds.pert_ids, val_string_path)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

            if train_global_path.exists():
                self._train_global_feats = np.load(str(train_global_path))
            if train_pert_path.exists():
                self._train_pert_feats = np.load(str(train_pert_path))
            if val_global_path.exists():
                self._val_global_feats = np.load(str(val_global_path))
            if val_pert_path.exists():
                self._val_pert_feats = np.load(str(val_pert_path))
            if train_string_path.exists():
                self._train_string_feats = np.load(str(train_string_path))
            if val_string_path.exists():
                self._val_string_feats = np.load(str(val_string_path))

        # Pre-compute or load features for test stage
        if stage in ("test", None) and self._test_global_feats is None and dm is not None:
            test_global_path = cache_dir / "test_aido_global.npy"
            test_pert_path   = cache_dir / "test_aido_pert.npy"
            test_string_path = cache_dir / "test_string.npy"

            if local_rank == 0:
                cache_dir.mkdir(parents=True, exist_ok=True)
                tokenizer = dm.tokenizer
                if tokenizer is None:
                    dm.setup("test")
                    tokenizer = dm.tokenizer

                if not test_global_path.exists() or not test_pert_path.exists():
                    if dm.test_pert_ids:
                        precompute_aido_features(
                            dm.test_pert_ids, tokenizer,
                            test_global_path, test_pert_path,
                        )
                if not test_string_path.exists() and dm.test_pert_ids:
                    precompute_string_features(dm.test_pert_ids, test_string_path)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

            if test_global_path.exists():
                self._test_global_feats = np.load(str(test_global_path))
            if test_pert_path.exists():
                self._test_pert_feats = np.load(str(test_pert_path))
            if test_string_path.exists():
                self._test_string_feats = np.load(str(test_string_path))

        # Store test metadata from DataModule
        if stage == "test" and dm is not None and hasattr(dm, "test_pert_ids"):
            self._test_pert_ids = dm.test_pert_ids
            self._test_symbols = dm.test_symbols
            # Load test labels for metric computation
            test_df_path = Path(dm.data_dir) / "test.tsv"
            if test_df_path.exists() and self._test_labels is None:
                test_df = pd.read_csv(test_df_path, sep="\t")
                raw_test_labels = [json.loads(x) for x in test_df["label"].tolist()]
                # Remap: {-1->0, 0->1, 1->2}
                self._test_labels = np.array(raw_test_labels, dtype=np.int8) + 1  # [N_test, 6640]

    def _get_feats_for_batch(
        self,
        batch_indices: torch.Tensor,
        split: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (global_emb, pert_emb, string_feats) for a batch."""
        idx = batch_indices.cpu().numpy()

        def _load(cache, fallback_dim):
            if cache is None:
                B = len(idx)
                return torch.zeros(B, fallback_dim, dtype=torch.float32)
            return torch.from_numpy(cache[idx]).float()

        if split == "train":
            g = _load(self._train_global_feats, 256)
            p = _load(self._train_pert_feats, 256)
            s = _load(self._train_string_feats, 256)
        elif split == "val":
            g = _load(self._val_global_feats, 256)
            p = _load(self._val_pert_feats, 256)
            s = _load(self._val_string_feats, 256)
        else:  # test
            g = _load(self._test_global_feats, 256)
            p = _load(self._test_pert_feats, 256)
            s = _load(self._test_string_feats, 256)

        return g, p, s

    def _forward_batch(self, batch: Dict[str, Any], split: str) -> torch.Tensor:
        """Run forward pass. Returns logits [B, 3, N_GENES]."""
        device = batch["sym_ids"].device
        global_emb, pert_emb, string_feats = self._get_feats_for_batch(batch["idx"], split)
        global_emb = global_emb.to(device)
        pert_emb = pert_emb.to(device)
        string_feats = string_feats.to(device)

        return self.model(
            global_emb=global_emb,
            pert_emb=pert_emb,
            sym_ids=batch["sym_ids"],
            string_feats=string_feats,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self._forward_batch(batch, "train")
        labels = batch["label"]

        # Apply manifold mixup during training
        if self.hparams.mixup_alpha > 0.0 and self.training:
            mixed_logits, mixed_labels = manifold_mixup(logits, labels, self.hparams.mixup_alpha)
            loss = compute_mixed_focal_loss(self.criterion, mixed_logits, mixed_labels)
        else:
            B, C, G = logits.shape
            logits_flat = logits.permute(0, 2, 1).reshape(-1, C)
            labels_flat = labels.reshape(-1)
            loss = self.criterion(logits_flat, labels_flat)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward_batch(batch, "val")
        labels = batch["label"]

        B, C, G = logits.shape
        n_real = batch.get("n_real", B)
        if n_real < B:
            # Mask padded samples: zero out logits for padded positions
            mask = torch.zeros(B, dtype=torch.bool, device=logits.device)
            mask[n_real:] = True
            logits_masked = logits.clone()
            logits_masked[mask] = 0.0
            logits_flat = logits_masked.permute(0, 2, 1).reshape(-1, C)
        else:
            logits_flat = logits.permute(0, 2, 1).reshape(-1, C)
        labels_flat = labels.reshape(-1)
        loss = self.criterion(logits_flat, labels_flat)

        # Log val_loss for checkpoint monitoring (critical fix over parent)
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._val_preds.append(probs)
        self._val_labels.append(labels.cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, dim=0)
        local_labels = torch.cat(self._val_labels, dim=0)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        local_f1 = compute_deg_f1(local_preds.numpy(), local_labels.numpy())

        # Reduce F1 across all ranks (scalar all-reduce)
        world_size = self.trainer.world_size if self.trainer is not None else 1
        if world_size > 1:
            import torch.distributed as dist
            local_f1_t = torch.tensor(local_f1, dtype=torch.float32, device="cuda")
            dist.all_reduce(local_f1_t, op=dist.ReduceOp.SUM)
            f1 = (local_f1_t / world_size).item()
        else:
            f1 = local_f1

        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward_batch(batch, "test")
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        n_real = batch.get("n_real", probs.shape[0])

        # Handle batch indices: could be list (regular collate) or Tensor (pad collate)
        raw_idx = batch["idx"]
        if n_real < probs.shape[0]:
            probs = probs[:n_real]
            if isinstance(raw_idx, torch.Tensor):
                raw_idx = raw_idx[:n_real]
            elif isinstance(raw_idx, list):
                raw_idx = raw_idx[:n_real]

        # Convert to tensor for consistent all_gather
        if isinstance(raw_idx, torch.Tensor):
            idx_tensor = raw_idx.cpu()
        else:
            idx_tensor = torch.tensor(raw_idx, dtype=torch.long)
        self._test_preds.append(probs)
        self._test_indices.append(idx_tensor)

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

            # De-duplicate
            seen: Dict[int, int] = {}
            unique_preds_list: List[np.ndarray] = []
            unique_idxs_list: List[int] = []
            for i, idx in enumerate(idxs):
                if idx not in seen:
                    seen[idx] = len(seen)
                    unique_preds_list.append(preds[i])
                    unique_idxs_list.append(int(idx))

            unique_preds = np.stack(unique_preds_list, axis=0)
            unique_idxs = np.array(unique_idxs_list)

            # Reorder by original index
            order = np.argsort(unique_idxs)
            ordered_preds = unique_preds[order]
            ordered_idxs = unique_idxs[order]

            # Write test_predictions.tsv
            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"

            rows = []
            for j, orig_i in enumerate(ordered_idxs):
                rows.append({
                    "idx": self._test_pert_ids[int(orig_i)],
                    "input": self._test_symbols[int(orig_i)],
                    "prediction": json.dumps(ordered_preds[j].tolist()),
                })
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved -> {out_path}")

            # Compute test F1 only when ALL test samples are processed
            # (skip during fast_dev_run where only 1 batch is tested)
            if self._test_labels is not None and ordered_preds.shape[0] == self._test_labels.shape[0]:
                test_f1 = compute_deg_f1(ordered_preds, self._test_labels)
                self.log("test_f1", test_f1, prog_bar=True, sync_dist=False)
                self.print(f"Test F1: {test_f1:.4f}")

    def configure_optimizers(self):
        # Only head, fusion, symbol_cnn, and string_fallback are trainable
        # (backbone is completely frozen — no parameters to optimize)
        symbol_params = []
        fusion_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "symbol_cnn" in name:
                symbol_params.append(param)
            elif "fusion" in name or "string_fallback" in name:
                fusion_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": symbol_params, "lr": self.hparams.head_lr * 0.5,
             "weight_decay": self.hparams.weight_decay},
            {"params": fusion_params, "lr": self.hparams.head_lr,
             "weight_decay": self.hparams.weight_decay},
            {"params": head_params, "lr": self.hparams.head_lr,
             "weight_decay": self.hparams.weight_decay},
        ]
        # Filter out empty groups
        param_groups = [pg for pg in param_groups if len(pg["params"]) > 0]
        opt = torch.optim.AdamW(param_groups)

        # ReduceLROnPlateau monitoring val_loss (not val_f1)
        # val_loss is more reliable for detecting overfitting than val_f1
        # (val_f1 can be artificially inflated by aggressive class weights)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10, min_lr=1e-6,
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

    # ── Checkpoint: save only trainable params ────────────────────────────────
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
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint Averaging
# ──────────────────────────────────────────────────────────────────────────────
def average_checkpoints(ckpt_paths: List[str], model_module: DEGLightningModule) -> None:
    """
    Average top-k checkpoint state dicts in-place on the model.
    Proven +0.003 F1 in node2-2-3-1-1-1 (0.4625 -> 0.4655).
    Note: load tensors to float32 to avoid dtype averaging issues.
    """
    if not ckpt_paths:
        return

    print(f"Averaging {len(ckpt_paths)} checkpoints for test prediction...")
    avg_state: Optional[Dict[str, torch.Tensor]] = None
    n_loaded = 0

    for ckpt_path in ckpt_paths:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)

            if avg_state is None:
                # Convert to float32 explicitly to avoid bfloat16 averaging issues
                avg_state = {k: v.float().clone() for k, v in sd.items()}
            else:
                for k in avg_state:
                    if k in sd:
                        avg_state[k] += sd[k].float()
            n_loaded += 1
            print(f"  Loaded: {ckpt_path}")
        except Exception as e:
            print(f"  Warning: failed to load {ckpt_path}: {e}")

    if avg_state is None or n_loaded == 0:
        print("  No checkpoints could be loaded for averaging.")
        return

    # Average
    for k in avg_state:
        avg_state[k] /= n_loaded

    model_module.load_state_dict(avg_state, strict=False)
    print(f"  Checkpoint averaging complete ({n_loaded} checkpoints).")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 1-3-3-1-1: AIDO.Cell-10M (FROZEN) + STRING PPI + 3-layer Cross-Attention Fusion"
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--micro-batch-size", type=int, default=16,
                   help="Micro batch size (larger possible since no backbone gradient)")
    p.add_argument("--global-batch-size", type=int, default=128,
                   help="Global batch size (must be multiple of micro_batch_size * 8 = 128)")
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--head-lr", type=float, default=6e-4,
                   help="LR for fusion + MLP head params")
    p.add_argument("--weight-decay", type=float, default=0.10,
                   help="Weight decay (proven at tree-best node3-1-3-1-1-1-1)")
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--attn-n-layers", type=int, default=3,
                   help="Number of TransformerEncoder layers in fusion (3=tree-best)")
    p.add_argument("--attn-nhead", type=int, default=8,
                   help="Number of attention heads (8=tree-best node3-1-3-1-1-1-1)")
    p.add_argument("--attn-dim-ff", type=int, default=256,
                   help="Feedforward dim in TransformerEncoder (256=width-constrained optimal)")
    p.add_argument("--attn-dropout", type=float, default=0.2,
                   help="Attention dropout (0.2=proven at tree-best)")
    p.add_argument("--mixup-alpha", type=float, default=0.2,
                   help="Manifold mixup alpha (0=disabled)")
    p.add_argument("--early-stopping-patience", type=int, default=15,
                   help="Early stopping patience on val_loss")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--no-checkpoint-avg", action="store_true",
                   help="Disable checkpoint averaging at test time")
    return p.parse_args()


def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Trainer settings ──────────────────────────────────────────────────────
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit_train: Any = 1.0
    limit_val: Any = 1.0
    limit_test: Any = 1.0
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
        strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    # ModelCheckpoint monitors val_loss (min) — critical fix for overfitting detection
    # val_loss is harder to "game" than val_f1 with aggressive class weights
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-3-3-1-1-{epoch:03d}-{val_loss:.4f}-{val_f1:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,   # save top-3 for checkpoint averaging
        save_last=True,
    )
    # EarlyStopping on val_loss (tight patience=15 epochs)
    early_stop_cb = EarlyStopping(
        monitor="val_loss", mode="min",
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
    # Override fast_dev_run's num_sanity_val_steps=0 to ensure DDP rank sync
    # even in fast_dev_run mode (prevents NCCL hangs from unsynchronized ranks)
    trainer.num_sanity_val_steps = 2

    # ── Data & model ──────────────────────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        head_dim=args.head_dim,
        dropout=args.dropout,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        attn_n_layers=args.attn_n_layers,
        attn_nhead=args.attn_nhead,
        attn_dim_ff=args.attn_dim_ff,
        attn_dropout=args.attn_dropout,
        mixup_alpha=args.mixup_alpha,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Test (with optional checkpoint averaging) ──────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        # Debug mode: skip checkpoint averaging
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        ckpt_dir = output_dir / "checkpoints"
        use_avg = (not args.no_checkpoint_avg) and ckpt_dir.exists()

        if use_avg:
            # Gather top-k checkpoint files sorted by val_loss (min is best)
            ckpt_files = []
            for f in ckpt_dir.glob("node1-3-3-1-1-*.ckpt"):
                if "last" in f.name:
                    continue
                # Parse val_loss from filename
                try:
                    parts = f.stem.split("-val_loss=")
                    if len(parts) >= 2:
                        val_loss_str = parts[1].split("-")[0]
                        val_loss_val = float(val_loss_str)
                        ckpt_files.append((val_loss_val, f))
                except (ValueError, IndexError):
                    pass
            ckpt_files.sort(key=lambda x: x[0])  # ascending: lower val_loss is better
            top_paths = [str(p) for _, p in ckpt_files[:3]]

            if len(top_paths) >= 2:
                # Load best single checkpoint first, then average top-k
                # The trainer's DDP model is already initialized with the last checkpoint
                # from training. We load "best" checkpoint into the model_module directly
                # (before DDP wraps it), then apply averaging.
                best_ckpt = str(ckpt_dir / "node1-3-3-1-1-epoch=001-val_loss=0.3516-val_f1=0.4643.ckpt")
                if Path(best_ckpt).exists():
                    best_state = torch.load(best_ckpt, map_location="cpu")
                    best_sd = best_state.get("state_dict", best_state)
                    # Convert to float32 for stable averaging
                    best_sd = {k: v.float() for k, v in best_sd.items()}
                    model_module.load_state_dict(best_sd, strict=False)
                # Apply checkpoint averaging on top of best checkpoint
                average_checkpoints(top_paths, model_module)
                # Run test ONCE with averaged weights (ckpt_path=None uses current model state)
                test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path=None)
            else:
                test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")
        else:
            test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # Write test score summary
    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"Test results: {test_results}\n")
    print(f"Done. Test score summary -> {score_path}")


if __name__ == "__main__":
    main()
