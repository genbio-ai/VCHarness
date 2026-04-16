#!/usr/bin/env python3
"""
Node 1-2-3: 4-Token Cross-Attention Fusion DEG Predictor
=========================================================
Key design changes from parent node (node1-2 / node2-1: AIDO.Cell-10M + LoRA all-8-layers,
test F1=0.4101):

  1. Cross-attention transformer fusion: 4 feature tokens (AIDO.Cell-10M global mean-pool,
     AIDO.Cell-10M perturbed-gene positional embedding, symbol CNN, frozen STRING GNN) are
     fused via a 3-layer pre-norm TransformerEncoder (d_model=256, nhead=8, dim_ff=256,
     attn_dropout=0.2). This is the highest-performing fusion strategy in the tree:
     node3-1-3-1-1-1-1 achieved test F1=0.4768 with this approach.

  2. Manifold mixup (alpha=0.3): applied in the token latent space during training to
     provide strong data augmentation for the 1,500-sample dataset.

  3. Stronger regularization:
     - weight_decay=0.10 (vs parent's 0.01): 10× stronger L2 regularization
     - class_weights=[6.0, 1.0, 12.0] (vs parent's [5.0, 1.0, 10.0]): stronger minority boost
     - label_smoothing=0.05 in focal loss
     - lora_dropout=0.2, LoRA r=4 (vs parent's r=8, lora_dropout=0.25)

  4. Character-level gene symbol CNN (3-branch Conv1d, 64-dim → 256-dim token): orthogonal
     gene-identity signal. Proven to add +0.04 F1 in the tree.

  5. Frozen STRING GNN PPI embeddings (256-dim token): gene regulatory network topology.
     Proven to add +0.017 F1 in the tree (node3-2 vs symbol-CNN-only).

  6. CosineAnnealingLR (T_max=100, eta_min=1e-7): smooth LR decay matching the behavior
     of the tree-best node (which used ReduceLROnPlateau that never triggered, effectively
     constant LR for 100 epochs; cosine decay is a gentle, principled alternative).

Key differences from sibling nodes:
  - node1-2-1 (scFoundation + sparse context): failed with -7.1% regression (0.3809).
    This node uses proven AIDO.Cell-10M backbone.
  - node1-2-2 (AIDO.Cell + symbol CNN + STRING GNN + concat MLP, test F1=0.4453):
    This node replaces simple concatenation + MLP with 3-layer cross-attention transformer
    fusion, adds manifold mixup, and uses stronger regularization (wd=0.10 vs 0.01).

Memories that influenced this design:
  - node3-1-3-1-1-1-1 (test F1=0.4768): 3-layer cross-attention fusion + mixup=0.3 + wd=0.10
  - node2-1-2 / node1-2-2 (test F1=0.4453): symbol CNN + STRING GNN confirmed +0.035 F1
  - node2-2 / node3-2 (test F1=0.4622): 4-source AIDO+symbol+STRING best concat approach
  - node3-1-3-1-1-1-1-1 (test F1=0.4698): 4-layer fusion REGRESSED; confirmed 3-layer optimal
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
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
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ── Constants ──────────────────────────────────────────────────────────────────
AIDO_CELL_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_MODEL_DIR = "/home/Models/STRING_GNN"
N_GENES_AIDO = 19_264   # AIDO.Cell vocabulary size
N_GENES_OUT = 6_640     # output genes
N_CLASSES = 3
SENTINEL_EXPR = 1.0     # baseline expression (non-perturbed genes)
KNOCKOUT_EXPR = 0.0     # expression for knocked-out gene
AIDO_HIDDEN = 256       # AIDO.Cell-10M hidden dimension
D_MODEL = 256           # fusion transformer d_model (= AIDO_HIDDEN)
STRING_GNN_DIM = 256    # STRING GNN output dimension
SYMBOL_MAX_LEN = 30     # max gene symbol length (padded/truncated)

# Character vocabulary for gene symbol encoding
CHAR_VOCAB = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. "
char_to_idx = {c: i + 1 for i, c in enumerate(CHAR_VOCAB)}  # 0 = padding
CHAR_VOCAB_SIZE = len(CHAR_VOCAB) + 1  # +1 for padding idx=0

# Class weights: [down-regulated, unchanged, up-regulated]
# Stronger minority class boost from tree-best node3-1-3-1-1-1-1 (test F1=0.4768)
# Train distribution: down~3.4%, unchanged~95.5%, up~1.1%
CLASS_WEIGHTS = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)


# ── Focal Loss with label smoothing ───────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weights and label smoothing.

    Combines:
    - Label smoothing (via F.cross_entropy built-in) for regularization
    - Focal weighting (via unsmoothed pt) for minority class emphasis
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        # Smoothed cross-entropy for regularization
        ce = F.cross_entropy(
            logits, targets, weight=w, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        # Focal weight based on unsmoothed probability (preserves focal sharpness)
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ── Character encoding helpers ─────────────────────────────────────────────────
def encode_symbol(symbol: str, max_len: int = SYMBOL_MAX_LEN) -> List[int]:
    """Encode gene symbol string to character indices (0-padded to max_len)."""
    encoded = [char_to_idx.get(c.upper(), 0) for c in symbol[:max_len]]
    if len(encoded) < max_len:
        encoded.extend([0] * (max_len - len(encoded)))
    return encoded


# ── Symbol CNN ─────────────────────────────────────────────────────────────────
class SymbolCNN(nn.Module):
    """3-branch character-level CNN → D_MODEL-dim token.

    Architecture:
        char_indices [B, max_len] → embed [B, embed_dim, max_len]
        3 × Conv1d(embed_dim, num_filters, k={2,3,4}) + MaxPool → [B, 3*num_filters]
        → Linear(3*num_filters, out_dim) + GELU → [B, out_dim=D_MODEL]
    """

    def __init__(
        self,
        vocab_size: int = CHAR_VOCAB_SIZE,
        embed_dim: int = 32,
        num_filters: int = 64,
        kernel_sizes: Tuple[int, ...] = (2, 3, 4),
        out_dim: int = D_MODEL,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.proj = nn.Linear(num_filters * len(kernel_sizes), out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, SYMBOL_MAX_LEN] int64 → [B, D_MODEL] float32"""
        e = self.embed(x).transpose(1, 2)   # [B, embed_dim, L]
        pooled = [
            F.gelu(conv(e)).max(dim=-1).values  # [B, num_filters]
            for conv in self.convs
        ]
        cat = torch.cat(pooled, dim=-1)          # [B, 3*num_filters]
        return self.dropout(F.gelu(self.proj(cat)))  # [B, D_MODEL]


# ── Dataset ────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Loads perturbation data and pre-builds:
    - expr: synthetic expression profile [19264] (all 1.0 except KO gene=0.0)
    - gene_pos: AIDO.Cell position of knocked-out gene (-1 if not in vocab)
    - sym_chars: character indices of gene symbol [SYMBOL_MAX_LEN]
    - gnn_idx: STRING GNN node index (-1 if not in STRING graph)
    - label: DEG labels remapped {-1,0,1} → {0,1,2} (training only)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],
        symbol_to_chars: Dict[str, List[int]],
        pert_to_gnn_idx: Dict[str, int],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.symbol_to_chars = symbol_to_chars
        self.pert_to_gnn_idx = pert_to_gnn_idx
        self.is_test = is_test

        # Pre-build expression tensors: [N, 19264] float32
        self.expr_inputs = self._build_expr_tensors()

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Remap {-1,0,1} → {0,1,2} per metric contract
            self.labels = np.array(raw_labels, dtype=np.int8) + 1
        else:
            self.labels = None

    def _build_expr_tensors(self) -> torch.Tensor:
        N = len(self.pert_ids)
        expr = torch.full((N, N_GENES_AIDO), SENTINEL_EXPR, dtype=torch.float32)
        for i, pert_id in enumerate(self.pert_ids):
            base = pert_id.split(".")[0]
            pos = self.gene_to_pos.get(base)
            if pos is not None:
                expr[i, pos] = KNOCKOUT_EXPR
        return expr

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.pert_ids[idx].split(".")[0]
        gene_pos = self.gene_to_pos.get(base, -1)
        sym = self.symbols[idx]
        sym_chars = self.symbol_to_chars.get(sym, [0] * SYMBOL_MAX_LEN)
        gnn_idx = self.pert_to_gnn_idx.get(base, -1)

        item: Dict[str, Any] = {
            "idx": idx,
            "expr": self.expr_inputs[idx],                           # [19264] float32
            "gene_pos": gene_pos,                                    # int
            "sym_chars": torch.tensor(sym_chars, dtype=torch.long),  # [30] int64
            "gnn_idx": gnn_idx,                                      # int
            "pert_id": self.pert_ids[idx],
            "symbol": sym,
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)  # [6640]
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "expr": torch.stack([b["expr"] for b in batch]),                    # [B, 19264]
        "gene_pos": torch.tensor([b["gene_pos"] for b in batch], dtype=torch.long),  # [B]
        "sym_chars": torch.stack([b["sym_chars"] for b in batch]),          # [B, 30]
        "gnn_idx": torch.tensor([b["gnn_idx"] for b in batch], dtype=torch.long),    # [B]
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])         # [B, 6640]
    return result


# ── DataModule ─────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        micro_batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.gene_to_pos: Dict[str, int] = {}
        self.symbol_to_chars: Dict[str, List[int]] = {}
        self.pert_to_gnn_idx: Dict[str, int] = {}
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        # ── Tokenizer: rank 0 downloads/caches first, then all ranks load ──
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # ── ENSG → AIDO.Cell position mapping ──
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)

        # ── Symbol → character indices mapping ──
        if not self.symbol_to_chars:
            all_symbols: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_symbols.extend(df["symbol"].tolist())
            for sym in set(all_symbols):
                self.symbol_to_chars[sym] = encode_symbol(sym)

        # ── ENSG → STRING GNN node index mapping ──
        if not self.pert_to_gnn_idx:
            self.pert_to_gnn_idx = self._build_gnn_idx_mapping()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self.gene_to_pos, self.symbol_to_chars, self.pert_to_gnn_idx,
            )
            self.val_ds = PerturbationDataset(
                val_df, self.gene_to_pos, self.symbol_to_chars, self.pert_to_gnn_idx,
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self.gene_to_pos, self.symbol_to_chars, self.pert_to_gnn_idx,
                is_test=True,
            )
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    @staticmethod
    def _build_gene_to_pos(tokenizer, gene_ids: List[str]) -> Dict[str, int]:
        """Map each ENSG gene_id to its position index in AIDO.Cell vocab."""
        mapping: Dict[str, int] = {}
        PROBE_VAL = 50.0
        for gene_id in gene_ids:
            try:
                inputs = tokenizer(
                    {"gene_ids": [gene_id], "expression": [PROBE_VAL]},
                    return_tensors="pt",
                )
                ids = inputs["input_ids"]
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                pos = (ids[0] == PROBE_VAL).nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    mapping[gene_id] = int(pos[0].item())
            except Exception:
                pass
        return mapping

    @staticmethod
    def _build_gnn_idx_mapping() -> Dict[str, int]:
        """Build ENSG → STRING GNN node index from node_names.json."""
        node_names_path = Path(STRING_GNN_MODEL_DIR) / "node_names.json"
        if not node_names_path.exists():
            return {}
        node_names = json.loads(node_names_path.read_text())
        mapping: Dict[str, int] = {}
        for i, name in enumerate(node_names):
            base = name.split(".")[0]  # strip version suffix e.g. "ENSG00000000003.1"
            mapping[base] = i
        return mapping

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


# ── STRING GNN frozen inference ────────────────────────────────────────────────
def compute_frozen_gnn_embeddings() -> Optional[torch.Tensor]:
    """Run frozen STRING GNN inference on the full graph. Returns [N_nodes, 256]."""
    model_dir = Path(STRING_GNN_MODEL_DIR)
    if not (model_dir / "graph_data.pt").exists():
        print(f"WARNING: STRING GNN graph_data.pt not found at {model_dir}. Skipping STRING GNN feature.")
        return None
    try:
        gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
        gnn_model.eval()
        graph = torch.load(str(model_dir / "graph_data.pt"), map_location="cpu")
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)
        with torch.no_grad():
            outputs = gnn_model(edge_index=edge_index, edge_weight=edge_weight)
        emb = outputs.last_hidden_state  # [N_nodes, 256]
        print(f"STRING GNN embeddings computed: shape {emb.shape}")
        return emb.cpu().float()
    except Exception as e:
        print(f"WARNING: STRING GNN inference failed: {e}. Skipping STRING GNN feature.")
        return None


# ── Cross-Attention Fusion Model ───────────────────────────────────────────────
class CrossAttentionFusionDEGModel(nn.Module):
    """
    4-Token Cross-Attention Fusion DEG Predictor.

    Feature tokens (each D_MODEL=256-dim):
      1. AIDO.Cell-10M global mean-pool → [B, 256]
      2. AIDO.Cell-10M perturbed-gene positional embedding → [B, 256]
      3. Symbol CNN (char-level) → [B, 256]
      4. Frozen STRING GNN PPI embedding → [B, 256]

    Fusion:
      Stack tokens → [B, 4, 256]
      → 3-layer pre-norm TransformerEncoder (nhead=8, dim_ff=256, attn_dropout=0.2)
      → mean pool over tokens → [B, 256]

    Head:
      LayerNorm(256) → Dropout(0.4) → Linear(256, 3×6640)

    Optional manifold mixup (training only):
      Mix tokens in latent space with λ ~ Beta(alpha, alpha)
    """

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.2,
        n_fusion_layers: int = 3,
        n_heads: int = 8,
        dim_ff: int = 256,
        attn_dropout: float = 0.2,
        head_dropout: float = 0.4,
        string_gnn_emb: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA on all 8 Q/K/V layers ──
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,  # all 8 layers (vs parent's all 8, r=8)
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # ── Symbol CNN: char-level encoding → D_MODEL-dim token ──
        self.symbol_cnn = SymbolCNN(
            vocab_size=CHAR_VOCAB_SIZE,
            embed_dim=32,
            num_filters=64,
            kernel_sizes=(2, 3, 4),
            out_dim=D_MODEL,
        )

        # ── STRING GNN: frozen embeddings buffer ──
        # Stored as a buffer (non-trainable, moved to device with model)
        if string_gnn_emb is not None:
            self.register_buffer("gnn_emb", string_gnn_emb.float())
        else:
            # Fallback: zero tensor if STRING GNN unavailable
            self.register_buffer("gnn_emb", torch.zeros(18870, STRING_GNN_DIM))

        # ── 3-layer pre-norm TransformerEncoder for cross-attention fusion ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-LayerNorm for better training stability
        )
        self.fusion = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_fusion_layers,
            norm=nn.LayerNorm(D_MODEL),
        )

        # ── Prediction head ──
        # Input: D_MODEL (from mean-pooled transformer output)
        # Output: 3 × 6640 = 19920
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Dropout(head_dropout),
            nn.Linear(D_MODEL, N_CLASSES * N_GENES_OUT),
        )
        # Conservative initialization for stable early training
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        expr: torch.Tensor,       # [B, 19264] float32
        gene_pos: torch.Tensor,   # [B] int64 (-1 if not in vocab)
        sym_chars: torch.Tensor,  # [B, 30] int64
        gnn_idx: torch.Tensor,    # [B] int64 (-1 if not in STRING)
        mixup_lambda: Optional[float] = None,
        mixup_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:            # [B, 3, 6640]
        """
        Args:
            mixup_lambda: If not None, apply manifold mixup in token space with this λ.
            mixup_idx: Permutation index for mixup (same size as batch dimension).
        """
        # ── Backbone: AIDO.Cell-10M ──
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, 19266, 256] (+2 summary tokens)

        # Token 1: Global mean-pool over 19264 gene positions
        gene_lhs = lhs[:, :N_GENES_AIDO, :]  # [B, 19264, 256]
        token1 = gene_lhs.mean(dim=1).float()  # [B, 256]

        # Token 2: Perturbed-gene positional embedding
        B = expr.shape[0]
        token2 = torch.zeros(B, AIDO_HIDDEN, device=lhs.device, dtype=torch.float32)
        valid_mask = gene_pos >= 0
        if valid_mask.any():
            valid_pos = gene_pos[valid_mask]
            token2[valid_mask] = lhs[valid_mask, valid_pos, :].float()
        # Fallback for genes not in AIDO.Cell vocab: use global mean-pool
        token2[~valid_mask] = token1[~valid_mask]

        # Token 3: Symbol CNN
        token3 = self.symbol_cnn(sym_chars)  # [B, 256] (Lightning moves to device)

        # Token 4: Frozen STRING GNN embedding
        token4 = torch.zeros(B, STRING_GNN_DIM, device=lhs.device, dtype=torch.float32)
        valid_gnn = gnn_idx >= 0
        if valid_gnn.any():
            token4[valid_gnn] = self.gnn_emb[gnn_idx[valid_gnn]].float()

        # ── Stack 4 tokens: [B, 4, 256] ──
        tokens = torch.stack([token1, token2, token3, token4], dim=1)

        # ── Manifold mixup (training only) ──
        if mixup_lambda is not None and mixup_idx is not None:
            tokens = mixup_lambda * tokens + (1 - mixup_lambda) * tokens[mixup_idx]

        # ── Cross-attention fusion ──
        fused = self.fusion(tokens)    # [B, 4, 256]
        pooled = fused.mean(dim=1)     # [B, 256] (mean-pool over 4 tokens)

        # ── Prediction head ──
        logits = self.head(pooled)                     # [B, 3*6640]
        return logits.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ── Metric helper ──────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro F1, averaged over all genes.

    Args:
        y_pred: [n_samples, 3, n_genes] (3-class logits or probabilities)
        y_true_remapped: [n_samples, n_genes] (labels in {0,1,2})
    """
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp_class = y_pred[:, :, g]             # [n_samples, 3]
        yhat = yp_class.argmax(axis=1)
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yhat, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ── LightningModule ────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.2,
        n_fusion_layers: int = 3,
        n_heads: int = 8,
        dim_ff: int = 256,
        attn_dropout: float = 0.2,
        head_dropout: float = 0.4,
        lr: float = 2e-4,          # backbone max LR; other params get 3× this
        weight_decay: float = 0.10,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        mixup_alpha: float = 0.3,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[CrossAttentionFusionDEGModel] = None
        self.criterion: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._gnn_emb: Optional[torch.Tensor] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            # ── Compute frozen STRING GNN embeddings (rank 0, broadcast) ──
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

            if world_size == 1:
                gnn_emb = compute_frozen_gnn_embeddings()
            else:
                if local_rank == 0:
                    gnn_emb = compute_frozen_gnn_embeddings()
                else:
                    gnn_emb = None

                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    has_gnn = torch.tensor(
                        [1 if gnn_emb is not None else 0],
                        dtype=torch.long,
                        device=torch.cuda.current_device(),
                    )
                    torch.distributed.broadcast(has_gnn, src=0)
                    if has_gnn.item() == 1:
                        if local_rank == 0:
                            emb_shape = torch.tensor(
                                list(gnn_emb.shape), dtype=torch.long,
                                device=torch.cuda.current_device(),
                            )
                        else:
                            emb_shape = torch.zeros(2, dtype=torch.long,
                                                    device=torch.cuda.current_device())
                        torch.distributed.broadcast(emb_shape, src=0)
                        if local_rank != 0:
                            gnn_emb = torch.zeros(
                                emb_shape[0].item(), emb_shape[1].item(),
                                dtype=torch.float32,
                            )
                        gnn_emb_dev = gnn_emb.to(torch.cuda.current_device())
                        torch.distributed.broadcast(gnn_emb_dev, src=0)
                        gnn_emb = gnn_emb_dev.cpu()
                    else:
                        gnn_emb = None

            self._gnn_emb = gnn_emb

            # ── Build model ──
            self.model = CrossAttentionFusionDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                n_fusion_layers=self.hparams.n_fusion_layers,
                n_heads=self.hparams.n_heads,
                dim_ff=self.hparams.dim_ff,
                attn_dropout=self.hparams.attn_dropout,
                head_dropout=self.hparams.head_dropout,
                string_gnn_emb=self._gnn_emb,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )
            # Cast trainable parameters to float32 for stable AdamW optimization
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        if stage == "test" and hasattr(self.trainer, "datamodule") \
                and self.trainer.datamodule is not None:
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            batch["expr"], batch["gene_pos"],
            batch["sym_chars"], batch["gnn_idx"],
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()  # [B*G, 3]
        labels_flat = labels.reshape(-1)                               # [B*G]
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        labels = batch["label"]
        B = labels.shape[0]

        if self.hparams.mixup_alpha > 0:
            # Manifold mixup: mix in token latent space
            lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
            idx_perm = torch.randperm(B, device=self.device)
            logits = self.model(
                batch["expr"], batch["gene_pos"],
                batch["sym_chars"], batch["gnn_idx"],
                mixup_lambda=lam, mixup_idx=idx_perm,
            )
            # Mix-loss: λ × loss(labels_a) + (1-λ) × loss(labels_b)
            loss = (lam * self._compute_loss(logits, labels) +
                    (1 - lam) * self._compute_loss(logits, labels[idx_perm]))
        else:
            logits = self(batch)
            loss = self._compute_loss(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        lp = torch.cat(self._val_preds, 0)
        ll = torch.cat(self._val_labels, 0)
        li = torch.cat(self._val_indices, 0)
        ap = self.all_gather(lp)
        al = self.all_gather(ll)
        ai = self.all_gather(li)
        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # Deduplicate by sample index (handles DDP duplication from drop_last=False)
        preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
        labels = al.cpu().view(-1, N_GENES_OUT).numpy()
        idxs = ai.cpu().view(-1).numpy()
        _, uniq = np.unique(idxs, return_index=True)
        f1 = compute_deg_f1(preds[uniq], labels[uniq])

        # All-reduce so all DDP ranks report the same val_f1
        f1_tensor = torch.tensor(f1, dtype=torch.float32, device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(f1_tensor, op=torch.distributed.ReduceOp.SUM)
            f1_tensor = f1_tensor / self.trainer.world_size

        self.log("val_f1", f1_tensor.item(), prog_bar=True, sync_dist=True)

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
        ap = self.all_gather(lp)
        ai = self.all_gather(li)
        self._test_preds.clear()
        self._test_indices.clear()

        if self.global_rank == 0:
            preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
            idxs = ai.cpu().view(-1).numpy()
            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]
            idxs = idxs[uniq]
            order = np.argsort(idxs)
            preds = preds[order]
            idxs = idxs[order]

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for rank_i, orig_i in enumerate(idxs):
                rows.append({
                    "idx": self._test_pert_ids[orig_i],
                    "input": self._test_symbols[orig_i],
                    "prediction": json.dumps(preds[rank_i].tolist()),
                })
            out_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path}")

    def configure_optimizers(self):
        # Differential LR: backbone (LoRA) gets lower LR; symbol CNN, fusion, head get higher
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters() if p.requires_grad
        ]
        other_params = (
            list(self.model.symbol_cnn.parameters()) +
            list(self.model.fusion.parameters()) +
            list(self.model.head.parameters())
        )

        backbone_lr = self.hparams.lr       # 2e-4
        other_lr = self.hparams.lr * 3      # 6e-4 (3× for randomly-initialized modules)

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": other_params, "lr": other_lr},
            ],
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # CosineAnnealingLR: smooth decay from initial LR to eta_min over max_epochs
        # Matches the effective behavior of the tree-best node (ReduceLROnPlateau that
        # never triggered = approximately constant LR for 100 epochs)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams.max_epochs,
            eta_min=1e-7,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
        full = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        out = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                k = prefix + name
                if k in full:
                    out[k] = full[k]
        for name, buf in self.named_buffers():
            k = prefix + name
            if k in full:
                out[k] = full[k]
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        return out

    def load_state_dict(self, state_dict, strict=True):
        """Load partial checkpoint (trainable params + buffers only)."""
        return super().load_state_dict(state_dict, strict=False)


# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 1-2-3: 4-Token Cross-Attention Fusion DEG Predictor"
    )
    p.add_argument("--data-dir", type=str, default=None,
                   help="Path to data directory (default: auto-detect from script location)")
    p.add_argument("--micro-batch-size", type=int, default=8,
                   help="Per-GPU micro batch size")
    p.add_argument("--global-batch-size", type=int, default=64,
                   help="Effective global batch size (must be multiple of micro_batch × 8)")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Backbone max LR; other params get 3× this LR")
    p.add_argument("--weight-decay", type=float, default=0.10,
                   help="Strong L2 regularization (vs parent's 0.01)")
    p.add_argument("--lora-r", type=int, default=4,
                   help="LoRA rank (vs parent's 8; lower = more regularized)")
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.2)
    p.add_argument("--n-fusion-layers", type=int, default=3,
                   help="Number of transformer fusion layers (3 is optimal per tree-best)")
    p.add_argument("--n-heads", type=int, default=8,
                   help="Number of attention heads in fusion transformer")
    p.add_argument("--dim-ff", type=int, default=256,
                   help="Feed-forward dim in fusion transformer (=d_model for width constraint)")
    p.add_argument("--attn-dropout", type=float, default=0.2,
                   help="Dropout in fusion transformer")
    p.add_argument("--head-dropout", type=float, default=0.4,
                   help="Dropout before prediction linear layer")
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05,
                   help="Label smoothing in focal loss for additional regularization")
    p.add_argument("--mixup-alpha", type=float, default=0.3,
                   help="Beta distribution alpha for manifold mixup (0=disabled)")
    p.add_argument("--early-stopping-patience", type=int, default=15,
                   help="EarlyStopping patience on val_f1")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step", type=int, default=None,
                   dest="debug_max_step",
                   help="Limit training/val/test steps for debugging")
    p.add_argument("--debug_max_step", type=int, default=None,
                   dest="debug_max_step",
                   help="Limit training/val/test steps for debugging (underscore variant)")
    p.add_argument("--fast-dev-run", action="store_true",
                   help="Run 1 batch per phase for quick validation")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    pl.seed_everything(0)
    args = parse_args()

    # ── Resolve data directory ──
    if args.data_dir is None:
        # mcts/node1-2-3/../../.. → project root → data/
        args.data_dir = str(Path(__file__).parent.parent.parent / "data")

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Debug / fast_dev_run settings ──
    fast_dev_run = args.fast_dev_run
    debug_max_step = args.debug_max_step

    if debug_max_step is not None:
        limit_train_batches = debug_max_step
        limit_val_batches = debug_max_step
        limit_test_batches = debug_max_step
        max_steps = debug_max_step
    else:
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        max_steps = -1

    # ── GPU count and gradient accumulation ──
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(1, n_gpus)
    accumulate_grad_batches = args.global_batch_size // (args.micro_batch_size * n_gpus)
    accumulate_grad_batches = max(1, accumulate_grad_batches)

    # ── Lightning modules ──
    modelmodule = DEGLightningModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        n_fusion_layers=args.n_fusion_layers,
        n_heads=args.n_heads,
        dim_ff=args.dim_ff,
        attn_dropout=args.attn_dropout,
        head_dropout=args.head_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        max_epochs=args.max_epochs,
    )
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=4,
    )

    # ── Callbacks ──
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
        min_delta=0.0,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # ── Loggers ──
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    # ── Trainer ──
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(seconds=120),
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad_batches,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=(
            args.val_check_interval
            if (debug_max_step is None and not fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, progress_bar],
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        gradient_clip_val=1.0,
        fast_dev_run=fast_dev_run,
    )

    # ── Train ──
    trainer.fit(modelmodule, datamodule=datamodule)

    # ── Test ──
    if fast_dev_run or debug_max_step is not None:
        test_results = trainer.test(modelmodule, datamodule=datamodule)
    else:
        test_results = trainer.test(modelmodule, datamodule=datamodule, ckpt_path="best")

    # ── Save test results summary ──
    if trainer.global_rank == 0:
        score_path = Path(__file__).parent / "test_score.txt"
        with open(score_path, "w") as f:
            f.write(f"test_results: {test_results}\n")
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
