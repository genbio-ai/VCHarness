#!/usr/bin/env python3
"""
Node: node1-1-1-2-1 — Multi-Source Cross-Attention Fusion + ReduceLROnPlateau
=============================================================================
Improvements over parent (node1-1-1-2, test F1=0.3953 — severe underfitting):

Parent failure root causes:
  1. Ultra-minimal LoRA (r=4, 2 layers, ~12K params) → insufficient backbone adaptation
  2. CosineAnnealingLR(T_max=120) decayed lr_backbone to ~8e-6 by epoch 101
  3. Conservative lr_backbone=5e-5 → backbone adapted too slowly
  4. Single-modality head — no Symbol CNN, only AIDO+STRING

Key changes informed by collective MCTS tree memory:
  1. Add character-level Symbol CNN (3-branch, kernels 2/3/4) — proven to break 0.41 ceiling
     → 3rd modality adds gene-name regulatory signal beyond PPI graphs
  2. Cross-attention fusion via 3-layer TransformerEncoder (d_model=256, nhead=8, dim_ff=384)
     → Learns inter-modal relationships; same architecture as tree-best node3-1-1-1-1-1-2 (F1=0.5049)
  3. Increase LoRA to r=8, ALL 8 layers (vs r=4, 2 layers) — 8× backbone capacity
     → Directly addresses parent's underfitting root cause
  4. Switch to ReduceLROnPlateau (patience=8, factor=0.5)
     → Critical: tree-best breakthrough at epoch 37 coincided with reactive LR halving
     → CosineAnnealingLR with T_max=120 decays too slowly then too aggressively
  5. Increase backbone LR to 1e-4 (vs 5e-5) — parent was too conservative
  6. Manifold mixup at token level (alpha=0.3) — regularizes fusion transformer
  7. Stronger class weights [6.0, 1.0, 12.0] + focal_gamma=1.5 (vs [2,1,4] + 3.0)
     → More aggressive minority class focus matching tree-best configuration
  8. weight_decay=0.10 (vs 5e-2) — stronger L2 regularization for larger model

Architecture:
  AIDO.Cell-10M + LoRA(r=8, all 8 layers) → dual pool → Linear(512→256) → [B,256] = tok_1
  STRING_GNN (frozen) → null_raw mask → adapter Linear(256→256) → [B,256] = tok_2
  Symbol CNN (char 3-branch) → Linear(192→256) → [B,256] = tok_3
  [tok_1, tok_2, tok_3] + pos_emb → [B,3,256]
  3-layer TransformerEncoder → mean pool → Dropout(0.3) → Linear(256→19920) → [B,3,6640]

Trainable params: ~7.2M (~4,800/sample for 1,500 training samples)
Memory per GPU: ~4–5 GiB at BS=16 (H100 80GB — very comfortable)

Inspired by:
  - node3-1-1-1-1-1-2 (test F1=0.5049): multi-source cross-attention fusion breakthrough
  - node3-1-1-1-1-1-2-1 feedback: ReduceLROnPlateau > CosineAnnealingLR (reactive beats fixed)
  - node1-1-1-2 (parent) feedback: LoRA r=4 + CosineAnnealing → severe underfitting
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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
N_GENES = 6_640
N_CLASSES = 3

AIDO_MODEL_PATH = "/home/Models/AIDO.Cell-10M"
STRING_GNN_PATH = "/home/Models/STRING_GNN"

AIDO_DIM = 256          # AIDO.Cell-10M hidden dimension
STRING_DIM = 256        # STRING_GNN output dimension
FUSION_DIM = 256        # Unified token dimension for cross-attention fusion
MAX_SYMBOL_LEN = 8      # Max characters in gene symbol (pad/truncate)
CHAR_VOCAB_SIZE = 128   # ASCII character vocabulary (indexed 0 = padding)
CHAR_EMB_DIM = 64       # Character embedding dimension
CNN_NUM_FILTERS = 64    # Filters per CNN branch
CNN_KERNEL_SIZES = [2, 3, 4]
SYMBOL_DIM = len(CNN_KERNEL_SIZES) * CNN_NUM_FILTERS  # 192

# Class weights: down-reg (-1→0, 3.4%), unchanged (0→1, 95.5%), up-reg (+1→2, 1.1%)
# Match tree-best node3-1-1-1-1-1-2 configuration [6, 1, 12]
CLASS_WEIGHTS = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STRING_GNN Embedding Extraction (run once, frozen)
# ─────────────────────────────────────────────────────────────────────────────
def extract_string_gnn_embeddings(
    model_path: str = STRING_GNN_PATH,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Extract frozen STRING_GNN node embeddings. Returns [18870, 256] tensor and name→idx dict."""
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
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with class weighting and label smoothing."""

    def __init__(
        self,
        gamma: float = 1.5,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: [N, C], targets: [N] int64 → scalar loss."""
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
# Metric helper — mirrors calc_metric.py logic exactly
# ─────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """
    y_pred: [N, 3, G] float (logits or probs)
    y_true_remapped: [N, G] int in {0,1,2}
    Returns macro-averaged F1 over G genes.
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
# Symbol Character Encoding Helper
# ─────────────────────────────────────────────────────────────────────────────
def encode_symbol(symbol: str, max_len: int = MAX_SYMBOL_LEN) -> torch.Tensor:
    """Encode gene symbol as ASCII char IDs (1-indexed so 0 = padding), padded to max_len."""
    char_ids = [max(1, ord(c) % CHAR_VOCAB_SIZE) for c in symbol[:max_len]]
    char_ids = char_ids + [0] * (max_len - len(char_ids))
    return torch.tensor(char_ids, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Perturbation dataset with pre-computed STRING_GNN embeddings + AIDO tokenized inputs
    + character-level Symbol encoding."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_embs: torch.Tensor,       # [N, 256] float32
        string_valid: torch.Tensor,      # [N] bool
        input_ids: torch.Tensor,         # [N, 19264] float32
        pert_positions: torch.Tensor,    # [N] int64 (-1 if not in AIDO vocab)
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.string_embs = string_embs
        self.string_valid = string_valid
        self.input_ids = input_ids
        self.pert_positions = pert_positions
        self.is_test = is_test

        # Pre-encode gene symbols as character IDs [N, MAX_SYMBOL_LEN]
        self.symbol_ids = torch.stack(
            [encode_symbol(s) for s in self.symbols], dim=0
        )

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            arr = np.array(raw_labels, dtype=np.int8) + 1   # {-1,0,1} → {0,1,2}
            self.labels = torch.from_numpy(arr).long()       # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx":          idx,
            "pert_id":      self.pert_ids[idx],
            "symbol":       self.symbols[idx],
            "string_emb":   self.string_embs[idx],     # [256]
            "string_valid": self.string_valid[idx],    # bool scalar
            "input_ids":    self.input_ids[idx],       # [19264]
            "pert_pos":     self.pert_positions[idx],  # int64
            "symbol_ids":   self.symbol_ids[idx],      # [MAX_SYMBOL_LEN]
        }
        if not self.is_test:
            item["label"] = self.labels[idx]           # [6640]
        return item


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        # Resolve data_dir robustly: resolve to absolute canonical path first
        # to handle symlinks correctly regardless of CWD.
        # Priority: (1) CWD/data_dir resolved, (2) project_root/data_dir resolved,
        # (3) raw data_dir, (4) fall back to data/
        _script_dir = Path(__file__).resolve().parent
        _project_root = _script_dir.parent.parent
        _cwd_data = (Path.cwd() / data_dir).resolve()
        if _cwd_data.exists():
            self.data_dir = _cwd_data
        elif (_project_root / data_dir).resolve().exists():
            self.data_dir = (_project_root / data_dir).resolve()
        else:
            _fallback = Path(data_dir).resolve()
            if _fallback.exists():
                self.data_dir = _fallback
            else:
                # Final fallback: data/ relative to project root
                self.data_dir = (_project_root / "data").resolve()

        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []
        self._string_emb_matrix: Optional[torch.Tensor] = None
        self._string_node_to_idx: Optional[Dict[str, int]] = None

    def _init_string_gnn(self) -> None:
        """Rank-0-first STRING_GNN extraction with barrier synchronization."""
        if self._string_emb_matrix is not None:
            return
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            print("Extracting STRING_GNN embeddings (once)...")
            emb, node_to_idx = extract_string_gnn_embeddings()
            self._string_emb_matrix = emb
            self._string_node_to_idx = node_to_idx
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if local_rank != 0:
            emb, node_to_idx = extract_string_gnn_embeddings()
            self._string_emb_matrix = emb
            self._string_node_to_idx = node_to_idx

    def _init_tokenizer(self) -> AutoTokenizer:
        """Rank-0-first tokenizer initialization."""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)

    def _get_string_data(self, pert_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Look up STRING_GNN embeddings. Unknown genes get zero vector (masked in model)."""
        zero_emb = torch.zeros(STRING_DIM, dtype=torch.float32)
        embs, valid_flags = [], []
        for pid in pert_ids:
            if pid in self._string_node_to_idx:
                embs.append(self._string_emb_matrix[self._string_node_to_idx[pid]])
                valid_flags.append(True)
            else:
                embs.append(zero_emb)
                valid_flags.append(False)
        n_found = sum(valid_flags)
        print(f"  STRING vocab coverage: {n_found}/{len(pert_ids)} ({100.0*n_found/len(pert_ids):.1f}%)")
        return torch.stack(embs, dim=0), torch.tensor(valid_flags, dtype=torch.bool)

    def _tokenize_and_get_positions(
        self,
        tokenizer: AutoTokenizer,
        pert_ids: List[str],
        split_name: str = "split",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize perturbation IDs and find each gene's positional slot in AIDO vocab."""
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        chunk_size = 128
        all_input_ids: List[torch.Tensor] = []
        for i in range(0, len(expr_dicts), chunk_size):
            chunk = expr_dicts[i:i + chunk_size]
            toks = tokenizer(chunk, return_tensors="pt")
            all_input_ids.append(toks["input_ids"])  # [chunk, 19264] float32
        input_ids = torch.cat(all_input_ids, dim=0)
        non_missing = input_ids > -0.5
        has_gene = non_missing.any(dim=1)
        pert_positions = non_missing.long().argmax(dim=1)
        pert_positions[~has_gene] = -1
        coverage = 100.0 * has_gene.float().mean().item()
        print(f"  [{split_name}] AIDO vocab coverage: {has_gene.sum().item()}/{len(pert_ids)} ({coverage:.1f}%)")
        return input_ids, pert_positions

    def setup(self, stage: Optional[str] = None) -> None:
        self._init_string_gnn()
        tokenizer = self._init_tokenizer()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df   = pd.read_csv(self.data_dir / "val.tsv",   sep="\t")

            print("Preparing train set...")
            tr_str_embs, tr_str_valid = self._get_string_data(train_df["pert_id"].tolist())
            tr_ids, tr_pos = self._tokenize_and_get_positions(
                tokenizer, train_df["pert_id"].tolist(), "train")

            print("Preparing val set...")
            va_str_embs, va_str_valid = self._get_string_data(val_df["pert_id"].tolist())
            va_ids, va_pos = self._tokenize_and_get_positions(
                tokenizer, val_df["pert_id"].tolist(), "val")

            self.train_ds = PerturbationDataset(
                train_df, tr_str_embs, tr_str_valid, tr_ids, tr_pos, is_test=False)
            self.val_ds = PerturbationDataset(
                val_df, va_str_embs, va_str_valid, va_ids, va_pos, is_test=False)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            print("Preparing test set...")
            te_str_embs, te_str_valid = self._get_string_data(test_df["pert_id"].tolist())
            te_ids, te_pos = self._tokenize_and_get_positions(
                tokenizer, test_df["pert_id"].tolist(), "test")

            self.test_ds = PerturbationDataset(
                test_df, te_str_embs, te_str_valid, te_ids, te_pos, is_test=True)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols  = test_df["symbol"].tolist()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Symbol CNN: Character-level 3-branch CNN for gene symbol encoding
# ─────────────────────────────────────────────────────────────────────────────
class SymbolCNN(nn.Module):
    """
    Character-level CNN with 3 branches (kernel sizes 2, 3, 4).
    Input:  [B, MAX_SYMBOL_LEN] int64 (ASCII char IDs, 0=padding)
    Output: [B, CNN_NUM_FILTERS * 3] = [B, 192]
    """

    def __init__(
        self,
        char_vocab_size: int = CHAR_VOCAB_SIZE,
        char_emb_dim: int = CHAR_EMB_DIM,
        num_filters: int = CNN_NUM_FILTERS,
        kernel_sizes: Optional[List[int]] = None,
        max_len: int = MAX_SYMBOL_LEN,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = CNN_KERNEL_SIZES

        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        # padding = k//2 for same-length output (enables global max-pool across all kernels)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_emb_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.output_dim = num_filters * len(kernel_sizes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.char_emb.weight, std=0.02)
        for conv in self.convs:
            nn.init.trunc_normal_(conv.weight, std=0.02)
            nn.init.zeros_(conv.bias)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """char_ids: [B, L] int64 → [B, output_dim=192]."""
        x = self.char_emb(char_ids)      # [B, L, char_emb_dim]
        x = x.permute(0, 2, 1)           # [B, char_emb_dim, L]
        features = []
        for conv in self.convs:
            c = F.relu(conv(x))          # [B, num_filters, L'] (same-length due to padding)
            c = c.max(dim=-1).values     # [B, num_filters] (global max pool)
            features.append(c)
        return torch.cat(features, dim=-1)  # [B, 192]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Source Fusion DEG Model
# ─────────────────────────────────────────────────────────────────────────────
class MultiSourceFusionDEGModel(nn.Module):
    """
    3-source cross-attention fusion model for DEG prediction:
      1. AIDO.Cell-10M (LoRA r=8, all 8 layers, Q/K/V) + dual pooling
      2. STRING_GNN (frozen pre-computed) + learnable adapter
      3. Character-level Symbol CNN (3-branch)
    Fused via 3-layer TransformerEncoder (self-attention over 3 modality tokens).
    """

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_layers: Optional[List[int]] = None,
        fusion_nhead: int = 8,
        fusion_dim_ff: int = 384,
        fusion_n_layers: int = 3,
        fusion_attn_dropout: float = 0.2,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # Default: all 8 layers of AIDO.Cell-10M (indexed 0-7)
        self.lora_layers = lora_layers if lora_layers is not None else list(range(8))

        # ── AIDO projection: dual pool (512) → FUSION_DIM (256) ──────────────
        self.aido_proj = nn.Sequential(
            nn.LayerNorm(AIDO_DIM * 2),
            nn.Linear(AIDO_DIM * 2, FUSION_DIM),
            nn.GELU(),
        )

        # ── STRING adapter: 256 → FUSION_DIM (256) ───────────────────────────
        # Learnable null embedding for genes not in STRING vocab
        self.string_null_raw = nn.Parameter(torch.zeros(STRING_DIM))
        # Adapter learns which PPI dimensions predict differential expression
        self.string_adapter = nn.Linear(STRING_DIM, FUSION_DIM)

        # ── Symbol CNN: char-level 3-branch + projection 192 → FUSION_DIM ────
        self.symbol_cnn = SymbolCNN()
        self.symbol_proj = nn.Sequential(
            nn.Linear(self.symbol_cnn.output_dim, FUSION_DIM),
            nn.GELU(),
        )

        # ── Learnable positional embeddings for 3 modality tokens ─────────────
        self.token_pos_emb = nn.Parameter(torch.zeros(3, FUSION_DIM))

        # ── Cross-attention fusion: 3-layer TransformerEncoder ────────────────
        # Self-attention over [tok_AIDO, tok_STRING, tok_SYMBOL] → learns inter-modal relations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FUSION_DIM,
            nhead=fusion_nhead,
            dim_feedforward=fusion_dim_ff,
            dropout=fusion_attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # post-norm (standard transformer)
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=fusion_n_layers)

        # ── Output head: FUSION_DIM → 3 * N_GENES ────────────────────────────
        # Direct projection matching tree-best architecture (node3-1-1-1-1-1-2)
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Dropout(head_dropout),
            nn.Linear(FUSION_DIM, N_CLASSES * N_GENES),
        )

        # Backbone initialized separately in LightningModule.setup()
        self.backbone: Optional[nn.Module] = None

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small positional embeddings for stability
        nn.init.trunc_normal_(self.token_pos_emb, std=0.01)

    def initialize_backbone(self) -> None:
        """Load AIDO.Cell-10M with LoRA (r=8, all 8 layers, Q/K/V projections)."""
        from peft import LoraConfig, get_peft_model, TaskType

        backbone = AutoModel.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        # Monkey-patch get_input_embeddings for PEFT compatibility
        # (AIDO.Cell uses custom GeneEmbedding, not a standard word embedding table)
        _gene_emb = backbone.bert.gene_embedding
        backbone.get_input_embeddings = lambda: _gene_emb

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=self.lora_layers,  # all 8 layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Cast LoRA adapter weights to float32 for stable optimization
        # (backbone stays bf16; only LoRA delta weights are float32)
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Enable gradient checkpointing (reduces activation memory ~30% speed tradeoff)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def encode(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32
        pert_positions: torch.Tensor,  # [B] int64 (-1 for unknown genes)
        string_emb: torch.Tensor,      # [B, 256] float32
        string_valid: torch.Tensor,    # [B] bool
        symbol_ids: torch.Tensor,      # [B, MAX_SYMBOL_LEN] int64
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode 3 sources → (tok_1, tok_2, tok_3) each [B, FUSION_DIM]."""
        B = input_ids.size(0)
        device = input_ids.device

        # ── Source 1: AIDO.Cell-10M backbone ─────────────────────────────────
        attention_mask = torch.ones(B, input_ids.size(1), dtype=torch.long, device=device)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state   # [B, 19266, 256] bfloat16

        # Global mean pool over 19264 gene positions (exclude 2 appended summary tokens)
        mean_pool = hidden[:, :19264, :].mean(dim=1).float()  # [B, 256]

        # Per-gene positional extraction (gene-specific context)
        valid_pos = pert_positions >= 0
        safe_pos = pert_positions.clamp(min=0)
        gene_emb_raw = hidden[
            torch.arange(B, device=device), safe_pos, :
        ].float()  # [B, 256]

        # Differentiable masking: unknown genes fall back to mean_pool
        valid_f = valid_pos.float().unsqueeze(-1)
        gene_emb = gene_emb_raw * valid_f + mean_pool * (1.0 - valid_f)

        aido_dual = torch.cat([gene_emb, mean_pool], dim=-1)  # [B, 512]
        tok_1 = self.aido_proj(aido_dual)                      # [B, FUSION_DIM]

        # ── Source 2: STRING_GNN ──────────────────────────────────────────────
        string_valid_f = string_valid.float().unsqueeze(-1).to(device)
        # For unknown genes, substitute learnable null_raw (trainable representation)
        raw_string = (
            string_emb.to(device) * string_valid_f
            + self.string_null_raw.unsqueeze(0) * (1.0 - string_valid_f)
        )
        tok_2 = self.string_adapter(raw_string)  # [B, FUSION_DIM]

        # ── Source 3: Symbol CNN ──────────────────────────────────────────────
        symbol_feat = self.symbol_cnn(symbol_ids.to(device))  # [B, 192]
        tok_3 = self.symbol_proj(symbol_feat)                  # [B, FUSION_DIM]

        return tok_1, tok_2, tok_3

    def fuse_and_predict(
        self,
        tok_1: torch.Tensor,  # [B, FUSION_DIM]
        tok_2: torch.Tensor,  # [B, FUSION_DIM]
        tok_3: torch.Tensor,  # [B, FUSION_DIM]
    ) -> torch.Tensor:
        """Fuse 3 modality tokens via TransformerEncoder → logits [B, 3, N_GENES]."""
        # Stack 3 tokens as a sequence: [B, 3, FUSION_DIM]
        tokens = torch.stack([tok_1, tok_2, tok_3], dim=1)
        # Add learnable positional embeddings per modality slot
        tokens = tokens + self.token_pos_emb.unsqueeze(0).to(tokens.device)
        # Cross-attention self-attention fusion
        fused = self.fusion(tokens)  # [B, 3, FUSION_DIM]
        # Mean-pool over modality dimension
        pooled = fused.mean(dim=1)   # [B, FUSION_DIM]
        # Output head
        logits = self.head(pooled)   # [B, N_CLASSES * N_GENES]
        return logits.view(-1, N_CLASSES, N_GENES)  # [B, 3, 6640]

    def forward(
        self,
        input_ids: torch.Tensor,
        pert_positions: torch.Tensor,
        string_emb: torch.Tensor,
        string_valid: torch.Tensor,
        symbol_ids: torch.Tensor,
    ) -> torch.Tensor:
        tok_1, tok_2, tok_3 = self.encode(
            input_ids, pert_positions, string_emb, string_valid, symbol_ids)
        return self.fuse_and_predict(tok_1, tok_2, tok_3)

    def get_parameter_groups(
        self, lr_backbone: float, lr_head: float, weight_decay: float
    ) -> List[Dict]:
        """Two optimizer groups: backbone LoRA (slow) and everything else (standard LR)."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        other_params = (
            list(self.aido_proj.parameters())
            + list(self.string_adapter.parameters())
            + [self.string_null_raw]
            + list(self.symbol_cnn.parameters())
            + list(self.symbol_proj.parameters())
            + [self.token_pos_emb]
            + list(self.fusion.parameters())
            + list(self.head.parameters())
        )
        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": other_params,    "lr": lr_head,     "weight_decay": weight_decay},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_layers: Optional[List[int]] = None,
        fusion_nhead: int = 8,
        fusion_dim_ff: int = 384,
        fusion_n_layers: int = 3,
        fusion_attn_dropout: float = 0.2,
        head_dropout: float = 0.3,
        lr_backbone: float = 1e-4,
        lr_head: float = 5e-4,
        weight_decay: float = 0.10,
        gamma_focal: float = 1.5,
        label_smoothing: float = 0.05,
        mixup_alpha: float = 0.3,
        plateau_patience: int = 8,
        plateau_factor: float = 0.5,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model: Optional[MultiSourceFusionDEGModel] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators (cleared each epoch)
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._test_labels_all: Optional[torch.Tensor] = None  # Loaded once in setup(stage="test")

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = MultiSourceFusionDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                lora_layers=self.hparams.lora_layers,
                fusion_nhead=self.hparams.fusion_nhead,
                fusion_dim_ff=self.hparams.fusion_dim_ff,
                fusion_n_layers=self.hparams.fusion_n_layers,
                fusion_attn_dropout=self.hparams.fusion_attn_dropout,
                head_dropout=self.hparams.head_dropout,
            )
            self.model.initialize_backbone()
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols
            # Load test labels for metric computation
            dm = self.trainer.datamodule
            test_df = pd.read_csv(dm.data_dir / "test.tsv", sep="\t")
            raw = [json.loads(x) for x in test_df["label"].tolist()]
            arr = np.array(raw, dtype=np.int8) + 1  # {-1,0,1} → {0,1,2}
            self._test_labels_all = torch.from_numpy(arr).long()  # [N_test, 6640]

    def forward(
        self,
        input_ids: torch.Tensor,
        pert_positions: torch.Tensor,
        string_emb: torch.Tensor,
        string_valid: torch.Tensor,
        symbol_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, pert_positions, string_emb, string_valid, symbol_ids)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G]; labels: [B, G] ({0,1,2}) → scalar focal loss."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        labels_flat = labels.reshape(-1)                        # [B*G]
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        B = batch["input_ids"].size(0)

        if self.hparams.mixup_alpha > 0 and B > 1 and self.training:
            # Manifold mixup at token (embedding) level — regularizes fusion transformer
            lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
            lam = max(lam, 1.0 - lam)  # ensure lam >= 0.5: original sample dominates
            perm = torch.randperm(B, device=self.device)

            # Encode each source independently
            tok_1, tok_2, tok_3 = self.model.encode(
                batch["input_ids"], batch["pert_pos"],
                batch["string_emb"], batch["string_valid"],
                batch["symbol_ids"],
            )
            # Mix token embeddings in latent space
            tok_1_mix = lam * tok_1 + (1.0 - lam) * tok_1[perm]
            tok_2_mix = lam * tok_2 + (1.0 - lam) * tok_2[perm]
            tok_3_mix = lam * tok_3 + (1.0 - lam) * tok_3[perm]

            logits = self.model.fuse_and_predict(tok_1_mix, tok_2_mix, tok_3_mix)
            labels = batch["label"]
            # Mixed loss: weighted combination of two focal losses
            loss = (
                lam * self._compute_loss(logits, labels)
                + (1.0 - lam) * self._compute_loss(logits, labels[perm])
            )
        else:
            logits = self(
                batch["input_ids"], batch["pert_pos"],
                batch["string_emb"], batch["string_valid"],
                batch["symbol_ids"],
            )
            loss = self._compute_loss(logits, batch["label"])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["pert_pos"],
            batch["string_emb"], batch["string_valid"],
            batch["symbol_ids"],
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        local_preds  = torch.cat(self._val_preds,   dim=0)  # [N_local, 3, G]
        local_labels = torch.cat(self._val_labels,  dim=0)  # [N_local, G]
        local_idx    = torch.cat(self._val_indices, dim=0)  # [N_local]

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        world_size = self.trainer.world_size if self.trainer.world_size else 1
        if world_size > 1:
            all_preds  = self.all_gather(local_preds)   # [world, N_local, 3, G]
            all_labels = self.all_gather(local_labels)  # [world, N_local, G]
            all_idx    = self.all_gather(local_idx)     # [world, N_local]

            preds_flat  = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            labels_flat = all_labels.view(-1, N_GENES).cpu().numpy()
            idx_flat    = all_idx.view(-1).cpu().numpy()

            # De-duplicate and restore original order
            unique_pos  = np.unique(idx_flat, return_index=True)[1]
            preds_flat  = preds_flat[unique_pos]
            labels_flat = labels_flat[unique_pos]
            order       = np.argsort(idx_flat[unique_pos])
            preds_flat  = preds_flat[order]
            labels_flat = labels_flat[order]

            f1 = compute_deg_f1(preds_flat, labels_flat)
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        else:
            f1 = compute_deg_f1(local_preds.numpy(), local_labels.numpy())
            self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["pert_pos"],
            batch["string_emb"], batch["string_valid"],
            batch["symbol_ids"],
        )
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = torch.cat(self._test_preds,   dim=0)
        local_idx   = torch.cat(self._test_indices, dim=0)

        all_preds = self.all_gather(local_preds)
        all_idx   = self.all_gather(local_idx)

        self._test_preds.clear()
        self._test_indices.clear()

        if self.trainer.is_global_zero:
            preds = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            idxs  = all_idx.view(-1).cpu().numpy()

            # De-duplicate (DDP may overlap last batch) and restore order
            unique_pos  = np.unique(idxs, return_index=True)[1]
            preds       = preds[unique_pos]
            sorted_idxs = idxs[unique_pos]
            order       = np.argsort(sorted_idxs)
            preds       = preds[order]
            final_idxs  = sorted_idxs[order]

            # Compute test F1 if labels are available
            if self._test_labels_all is not None:
                test_labels = self._test_labels_all[final_idxs].numpy()  # [N_test, 6640]
                test_f1 = compute_deg_f1(preds, test_labels)
                self.log("test_f1", test_f1, prog_bar=True, sync_dist=False)
                self.print(f"Test F1: {test_f1:.4f}")

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"

            rows = []
            for rank_i, orig_i in enumerate(final_idxs):
                rows.append({
                    "idx":        self._test_pert_ids[orig_i],
                    "input":      self._test_symbols[orig_i],
                    "prediction": json.dumps(preds[rank_i].tolist()),
                })
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path}")

    def configure_optimizers(self):
        param_groups = self.model.get_parameter_groups(
            lr_backbone=self.hparams.lr_backbone,
            lr_head=self.hparams.lr_head,
            weight_decay=self.hparams.weight_decay,
        )
        opt = torch.optim.AdamW(param_groups)

        # ReduceLROnPlateau: reactive LR halving when val_f1 plateaus
        # Critical finding from tree: node3-1-1-1-1-1-2 breakthrough at epoch 37
        # occurred immediately after first LR reduction (2e-4 → 1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            patience=self.hparams.plateau_patience,
            factor=self.hparams.plateau_factor,
            min_lr=1e-6,
            verbose=True,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_f1",   # maximize val_f1
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }

    # ── Checkpoint: save only trainable parameters ────────────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save trainable parameters + persistent buffers (skip frozen backbone)."""
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
        total  = sum(p.numel() for p in self.parameters())
        tr_cnt = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Checkpoint: saving {tr_cnt:,}/{total:,} params "
            f"({100.0 * tr_cnt / max(total, 1):.2f}% trainable)"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable parameters; frozen backbone is re-initialized from pretrained."""
        return super().load_state_dict(state_dict, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-Source Cross-Attention Fusion DEG predictor (node1-1-1-2-1)"
    )
    p.add_argument("--data-dir",               type=str,   default="data")
    p.add_argument("--micro-batch-size",        type=int,   default=16)
    p.add_argument("--global-batch-size",       type=int,   default=128)
    p.add_argument("--max-epochs",              type=int,   default=100)
    p.add_argument("--lr-backbone",             type=float, default=1e-4)
    p.add_argument("--lr-head",                 type=float, default=5e-4)
    p.add_argument("--weight-decay",            type=float, default=0.10)
    p.add_argument("--lora-r",                  type=int,   default=8)
    p.add_argument("--lora-alpha",              type=int,   default=16)
    p.add_argument("--lora-dropout",            type=float, default=0.05)
    p.add_argument("--fusion-dim-ff",           type=int,   default=384)
    p.add_argument("--fusion-n-layers",         type=int,   default=3)
    p.add_argument("--head-dropout",            type=float, default=0.3)
    p.add_argument("--gamma-focal",             type=float, default=1.5)
    p.add_argument("--label-smoothing",         type=float, default=0.05)
    p.add_argument("--mixup-alpha",             type=float, default=0.3)
    p.add_argument("--plateau-patience",        type=int,   default=8)
    p.add_argument("--plateau-factor",          type=float, default=0.5)
    p.add_argument("--early-stopping-patience", type=int,   default=20)
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

    # ── Distributed setup ─────────────────────────────────────────────────────
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps    = args.debug_max_step if args.debug_max_step is not None else -1
    limit_train = limit_val = limit_test = 1.0
    if args.debug_max_step is not None:
        # debug_max_step limits training/validation steps for quick debugging;
        # test always uses full test set for complete predictions
        limit_train = args.debug_max_step
        limit_val   = args.debug_max_step
        # limit_test remains 1.0 (full test set)

    val_check_interval = args.val_check_interval if (
        args.debug_max_step is None and not args.fast_dev_run
    ) else 1.0

    # ── Strategy ──────────────────────────────────────────────────────────────
    if n_gpus == 1:
        strategy = SingleDeviceStrategy(device="cuda:0")
    else:
        strategy = DDPStrategy(
            find_unused_parameters=True,   # LoRA leaves some backbone params unused
            timeout=timedelta(seconds=300),
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="multisrc-deg-{epoch:03d}-{val_f1:.4f}",
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
    lr_monitor   = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # ── Loggers ───────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # ── Trainer ───────────────────────────────────────────────────────────────
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

    # ── Data & Model ──────────────────────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=list(range(8)),  # all 8 layers of AIDO.Cell-10M
        fusion_nhead=8,              # 256 / 8 = 32 head_dim
        fusion_dim_ff=args.fusion_dim_ff,
        fusion_n_layers=args.fusion_n_layers,
        fusion_attn_dropout=0.2,
        head_dropout=args.head_dropout,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        max_epochs=args.max_epochs,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Test ──────────────────────────────────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
