#!/usr/bin/env python3
"""
Node 1-2: AIDO.Cell-10M + Symbol CNN + Frozen STRING GNN – Multi-Source DEG Predictor
========================================================================================
Key design differences from parent node (node2-1 / AIDO.Cell-10M + LoRA all 8 layers):
  1. Character-level gene symbol CNN (3-branch Conv1d, 64-dim) — the single biggest
     improvement in the MCTS tree (node2-2: +0.041 F1 by adding this feature).
  2. LoRA applied to LAST 4 LAYERS only (not all 8) — consistent with the node2-2
     lineage which consistently outperforms all-layer LoRA.
  3. Frozen STRING GNN PPI embeddings (256-dim → projected to 128-dim) — node3-2 showed
     +0.017 F1 gain by adding PPI topology signal; frozen avoids overfitting on 1500 samples.
  4. ReduceLROnPlateau instead of OneCycleLR — eliminates the ±0.01 val_f1 oscillation
     caused by cosine-annealing-driven LR changes independent of actual progress.
  5. Extended training (max_epochs=150, patience=35) — ReduceLROnPlateau needs multiple
     LR reductions to converge; more budget allows this multi-step process to complete.
  6. Moderate class weights [3.0, 1.0, 7.0] — slightly reduced from parent's [5.0, 1.0, 10.0],
     matching the successful node2-2 lineage configuration.

Key differences from sibling node (node1-1 / scFoundation + sparse context):
  - Stays with AIDO.Cell-10M backbone (scFoundation caused -7.1% regression in sibling).
  - Uses dense 19,264-token input (sibling's sparse 1,649 tokens hurt positional diversity).
  - Adds symbol CNN + STRING GNN as complementary features (not backbone replacement).
  - LoRA on QKV (proven) not out_proj/linear2 (ineffective in sibling).
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
from lightning.pytorch.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
AIDO_CELL_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_MODEL_DIR = "/home/Models/STRING_GNN"
N_GENES_AIDO = 19_264    # AIDO.Cell vocabulary size
N_GENES_OUT = 6_640      # output genes (task target)
N_CLASSES = 3
SENTINEL_EXPR = 1.0      # baseline expression (non-perturbed genes)
KNOCKOUT_EXPR = 0.0      # expression for knocked-out gene
AIDO_HIDDEN = 256        # AIDO.Cell-10M hidden dimension
AIDO_N_LAYERS = 8        # AIDO.Cell-10M transformer layers

# Moderate class weights — reduced from parent's [5.0, 1.0, 10.0]
# Matching the node2-2 lineage configuration that achieved best F1.
CLASS_WEIGHTS = torch.tensor([3.0, 1.0, 7.0], dtype=torch.float32)

# Character-level encoding for gene symbols
SYMBOL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-." + "_"  # + padding
CHAR_VOCAB_SIZE = len(SYMBOL_CHARS) + 1  # +1 for unknown
SYMBOL_MAX_LEN = 30  # max symbol length (padded/truncated to this)

# STRING GNN constants
STRING_GNN_DIM = 256   # STRING GNN hidden dimension
STRING_PROJ_DIM = 128  # projected dimension for head input


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weights."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=w, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Character-level symbol encoder
# ──────────────────────────────────────────────────────────────────────────────
def encode_symbol(symbol: str, max_len: int = SYMBOL_MAX_LEN) -> List[int]:
    """Encode a gene symbol string to character indices."""
    char_to_idx = {c: i + 1 for i, c in enumerate(SYMBOL_CHARS)}  # 0 = unknown
    encoded = [char_to_idx.get(c.upper(), 0) for c in symbol[:max_len]]
    # Pad to max_len
    encoded = encoded + [0] * (max_len - len(encoded))
    return encoded


class SymbolCNN(nn.Module):
    """3-branch character-level CNN for gene symbol encoding.

    Input: [B, max_len] character indices
    Output: [B, out_dim] symbol embedding
    """

    def __init__(
        self,
        vocab_size: int = CHAR_VOCAB_SIZE,
        embed_dim: int = 32,
        num_filters: int = 64,
        kernel_sizes: Tuple[int, ...] = (2, 3, 4),
        max_len: int = SYMBOL_MAX_LEN,
        out_dim: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.1)
        fusion_dim = num_filters * len(kernel_sizes)
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, max_len] int
        emb = self.embedding(x)         # [B, max_len, embed_dim]
        emb = self.dropout(emb)
        emb = emb.permute(0, 2, 1)      # [B, embed_dim, max_len]
        branch_outputs = []
        for conv in self.convs:
            c = conv(emb)                # [B, num_filters, L']
            c = F.gelu(c)
            c, _ = c.max(dim=-1)        # [B, num_filters]  global max-pool
            branch_outputs.append(c)
        fused = torch.cat(branch_outputs, dim=-1)  # [B, num_filters * n_branches]
        return self.proj(fused)         # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Returns AIDO.Cell expression tensors, perturbed-gene position, symbol char indices,
    STRING GNN index, and labels.
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

        # Pre-build expression input tensors: [N, 19264] float32
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
        pert_id = self.pert_ids[idx]
        symbol = self.symbols[idx]
        base = pert_id.split(".")[0]
        gene_pos = self.gene_to_pos.get(base, -1)
        sym_chars = self.symbol_to_chars.get(symbol, [0] * SYMBOL_MAX_LEN)
        gnn_idx = self.pert_to_gnn_idx.get(base, -1)
        item = {
            "idx": idx,
            "expr": self.expr_inputs[idx],               # [19264] float32
            "gene_pos": gene_pos,                         # int (-1 if not in vocab)
            "sym_chars": torch.tensor(sym_chars, dtype=torch.long),  # [max_len]
            "gnn_idx": gnn_idx,                           # int (-1 if not in STRING)
            "pert_id": pert_id,
            "symbol": symbol,
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "expr": torch.stack([b["expr"] for b in batch]),           # [B, 19264]
        "gene_pos": torch.tensor([b["gene_pos"] for b in batch], dtype=torch.long),
        "sym_chars": torch.stack([b["sym_chars"] for b in batch]), # [B, max_len]
        "gnn_idx": torch.tensor([b["gnn_idx"] for b in batch], dtype=torch.long),
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
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
        # ── Tokenizer: rank 0 downloads first ──
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # ── Build ENSG→position mapping ──
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)

        # ── Build symbol → char indices mapping ──
        if not self.symbol_to_chars:
            all_symbols: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_symbols.extend(df["symbol"].tolist())
            for sym in set(all_symbols):
                self.symbol_to_chars[sym] = encode_symbol(sym)

        # ── Build ENSG → STRING GNN node index mapping ──
        if not self.pert_to_gnn_idx:
            self.pert_to_gnn_idx = self._build_gnn_idx_mapping()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self.gene_to_pos, self.symbol_to_chars,
                self.pert_to_gnn_idx,
            )
            self.val_ds = PerturbationDataset(
                val_df, self.gene_to_pos, self.symbol_to_chars,
                self.pert_to_gnn_idx,
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self.gene_to_pos, self.symbol_to_chars,
                self.pert_to_gnn_idx, is_test=True,
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
        """Build ENSG→STRING GNN node index from node_names.json."""
        node_names_path = Path(STRING_GNN_MODEL_DIR) / "node_names.json"
        if not node_names_path.exists():
            return {}
        node_names = json.loads(node_names_path.read_text())
        # node_names[i] is the ENSG id for GNN node index i
        mapping: Dict[str, int] = {}
        for i, name in enumerate(node_names):
            # Strip version suffixes (e.g., "ENSG00000000003.1" → "ENSG00000000003")
            base = name.split(".")[0]
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


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class MultiSourceDEGModel(nn.Module):
    """AIDO.Cell-10M (LoRA last 4 layers) + Symbol CNN + Frozen STRING GNN.

    Three feature sources fused into a unified prediction head.
    """

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.2,
        head_dropout: float = 0.35,
        head_hidden: int = 320,
        string_gnn_emb: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            string_gnn_emb: Pre-computed STRING GNN embeddings [N_gnn_nodes, 256].
                            Passed in at setup time after frozen GNN inference.
        """
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA on last 4 layers ──
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA on QKV of layers 4-7 (last 4 of 8)
        # Layers 0-7; last 4 = layers 4, 5, 6, 7
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=[4, 5, 6, 7],  # last 4 layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # ── Symbol CNN ──
        self.symbol_cnn = SymbolCNN(
            vocab_size=CHAR_VOCAB_SIZE,
            embed_dim=32,
            num_filters=64,
            kernel_sizes=(2, 3, 4),
            max_len=SYMBOL_MAX_LEN,
            out_dim=64,
        )

        # ── STRING GNN projection (frozen GNN embeddings → 128-dim) ──
        # The pre-computed embeddings are stored as a buffer; the projection is learned
        if string_gnn_emb is not None:
            self.register_buffer("gnn_emb", string_gnn_emb.float())
        else:
            self.register_buffer("gnn_emb", None)

        self.gnn_proj = nn.Sequential(
            nn.Linear(STRING_GNN_DIM, STRING_PROJ_DIM),
            nn.GELU(),
            nn.Linear(STRING_PROJ_DIM, STRING_PROJ_DIM),
        )

        # ── Prediction head ──
        # Input: AIDO_HIDDEN * 2 (512) + 64 (symbol) + STRING_PROJ_DIM (128) = 704
        head_in = AIDO_HIDDEN * 2 + 64 + STRING_PROJ_DIM  # 704
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        # Initialise output layer conservatively
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        expr: torch.Tensor,       # [B, 19264] float32
        gene_pos: torch.Tensor,   # [B]        int64  (-1 if not in vocab)
        sym_chars: torch.Tensor,  # [B, max_len] int64
        gnn_idx: torch.Tensor,    # [B]          int64  (-1 if not in STRING)
    ) -> torch.Tensor:
        # ── AIDO.Cell backbone ──
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, 19266, 256]

        # (a) Global mean-pool over all 19264 gene positions
        gene_emb = lhs[:, :N_GENES_AIDO, :]       # [B, 19264, 256]
        global_emb = gene_emb.mean(dim=1)          # [B, 256]

        # (b) Perturbed-gene positional embedding
        B = expr.shape[0]
        pert_emb = torch.zeros(B, AIDO_HIDDEN, device=lhs.device, dtype=lhs.dtype)
        valid_mask = gene_pos >= 0
        if valid_mask.any():
            valid_pos = gene_pos[valid_mask]
            pert_emb[valid_mask] = lhs[valid_mask, valid_pos, :]
        pert_emb[~valid_mask] = global_emb[~valid_mask]

        # Concat backbone features → [B, 512]
        backbone_emb = torch.cat([global_emb, pert_emb], dim=-1).float()

        # ── Symbol CNN ──
        sym_emb = self.symbol_cnn(sym_chars)  # [B, 64] float32

        # ── STRING GNN feature ──
        if self.gnn_emb is not None:
            # Build STRING GNN embeddings for each sample in batch
            gnn_feats = torch.zeros(B, STRING_GNN_DIM, device=expr.device, dtype=torch.float32)
            valid_gnn_mask = gnn_idx >= 0
            if valid_gnn_mask.any():
                valid_gnn_idx = gnn_idx[valid_gnn_mask]
                gnn_feats[valid_gnn_mask] = self.gnn_emb[valid_gnn_idx].float()
            gnn_proj_emb = self.gnn_proj(gnn_feats)  # [B, 128]
        else:
            gnn_proj_emb = torch.zeros(B, STRING_PROJ_DIM, device=expr.device, dtype=torch.float32)

        # ── Fusion → head ──
        combined = torch.cat([backbone_emb, sym_emb, gnn_proj_emb], dim=-1)  # [B, 704]
        logits = self.head(combined)                                            # [B, 3*6640]
        return logits.view(B, N_CLASSES, N_GENES_OUT)                          # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# STRING GNN frozen inference (to compute node embeddings once at setup)
# ──────────────────────────────────────────────────────────────────────────────
def compute_frozen_gnn_embeddings() -> Optional[torch.Tensor]:
    """Run frozen STRING GNN inference on the full graph once.

    Returns:
        Tensor of shape [N_nodes, 256] if STRING GNN files are available,
        else None.
    """
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
            outputs = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
        emb = outputs.last_hidden_state  # [N_nodes, 256]
        print(f"STRING GNN embeddings computed: shape {emb.shape}")
        return emb.cpu()
    except Exception as e:
        print(f"WARNING: STRING GNN inference failed: {e}. Skipping STRING GNN feature.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro F1, averaged over all genes.

    y_pred: [n_samples, 3, n_genes]   (3-class logits or probabilities)
    y_true_remapped: [n_samples, n_genes]  (labels in {0,1,2})
    """
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp_class = y_pred[:, :, g]
        yhat = yp_class.argmax(axis=1)
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yhat, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.2,
        head_dropout: float = 0.35,
        head_hidden: int = 320,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        gamma_focal: float = 2.0,
        max_epochs: int = 150,
        rlop_patience: int = 5,
        rlop_factor: float = 0.7,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[MultiSourceDEGModel] = None
        self.criterion: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        # GNN embeddings computed once at setup
        self._gnn_emb: Optional[torch.Tensor] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            # Compute STRING GNN embeddings once (frozen inference)
            # Only rank 0 computes, then broadcasts
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

            if world_size == 1:
                # Single-GPU: no broadcast needed
                gnn_emb = compute_frozen_gnn_embeddings()
            else:
                # Multi-GPU DDP: rank 0 computes, broadcast to others
                if local_rank == 0:
                    gnn_emb = compute_frozen_gnn_embeddings()
                else:
                    gnn_emb = None

                # Broadcast gnn_emb across ranks using NCCL (GPU tensors required)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    # Broadcast whether gnn_emb is available (tensor must be on GPU for NCCL)
                    has_gnn = torch.tensor(
                        [1 if gnn_emb is not None else 0],
                        dtype=torch.long,
                        device=torch.cuda.current_device(),
                    )
                    torch.distributed.broadcast(has_gnn, src=0)
                    if has_gnn.item() == 1:
                        if local_rank == 0:
                            emb_shape = torch.tensor(
                                list(gnn_emb.shape),
                                dtype=torch.long,
                                device=torch.cuda.current_device(),
                            )
                        else:
                            emb_shape = torch.zeros(
                                2, dtype=torch.long, device=torch.cuda.current_device()
                            )
                        torch.distributed.broadcast(emb_shape, src=0)
                        if local_rank != 0:
                            gnn_emb = torch.zeros(
                                emb_shape[0].item(),
                                emb_shape[1].item(),
                                dtype=torch.float32,
                            )
                        # Broadcast gnn_emb (GPU tensor for NCCL)
                        gnn_emb_device = gnn_emb.to(torch.cuda.current_device())
                        torch.distributed.broadcast(gnn_emb_device, src=0)
                        gnn_emb = gnn_emb_device.cpu()
                    else:
                        gnn_emb = None

            self._gnn_emb = gnn_emb

            self.model = MultiSourceDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                head_dropout=self.hparams.head_dropout,
                head_hidden=self.hparams.head_hidden,
                string_gnn_emb=self._gnn_emb,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
            )
            # Cast trainable params to float32 for stable AdamW optimization
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        if stage == "test" and hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            batch["expr"], batch["gene_pos"],
            batch["sym_chars"], batch["gnn_idx"],
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_flat = labels.reshape(-1)
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
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

        # Compute val_f1 on full gathered dataset, deduplicated by sample index
        preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
        labels = al.cpu().view(-1, N_GENES_OUT).numpy()
        idxs = ai.cpu().view(-1).numpy()
        _, uniq = np.unique(idxs, return_index=True)
        f1 = compute_deg_f1(preds[uniq], labels[uniq])

        # All-reduce so all DDP ranks share the same val_f1
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
        # Separate parameter groups with differential LR:
        # LoRA backbone params: lower LR (pretrained)
        # Symbol CNN + GNN projection + head: higher LR (randomly initialized)
        backbone_params = [p for n, p in self.model.backbone.named_parameters()
                           if p.requires_grad]
        other_params = (
            list(self.model.symbol_cnn.parameters()) +
            list(self.model.gnn_proj.parameters()) +
            list(self.model.head.parameters())
        )

        max_lr_backbone = self.hparams.lr
        max_lr_other = self.hparams.lr * 5  # 5× higher for randomly-init modules

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": max_lr_backbone},
                {"params": other_params, "lr": max_lr_other},
            ],
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # ReduceLROnPlateau: reduce LR when val_f1 stops improving
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            patience=self.hparams.rlop_patience,
            factor=self.hparams.rlop_factor,
            min_lr=1e-6,
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

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and buffers."""
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        out = {}
        trainable_keys = set()
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_keys.add(prefix + name)
        for name, buf in self.named_buffers():
            trainable_keys.add(prefix + name)
        for k in full:
            if k in trainable_keys:
                out[k] = full[k]
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.parameters())
        self.print(
            f"Saving checkpoint: {trainable_count}/{total_count} trainable params "
            f"({100 * trainable_count / max(total_count, 1):.2f}%)"
        )
        return out

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 1-2: AIDO.Cell-10M + Symbol CNN + STRING GNN DEG predictor"
    )
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.2)
    p.add_argument("--head-dropout", type=float, default=0.35)
    p.add_argument("--head-hidden", type=int, default=320)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--rlop-patience", type=int, default=5)
    p.add_argument("--rlop-factor", type=float, default=0.7)
    p.add_argument("--early-stopping-patience", type=int, default=35)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def main():
    pl.seed_everything(0)
    args = parse_args()

    # Resolve data_dir relative to project root (3 levels up from mcts/node1-2/)
    if args.data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    else:
        data_dir = Path(args.data_dir)
    args.data_dir = str(data_dir)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit = args.debug_max_step if args.debug_max_step is not None else 1.0

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
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
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=180)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit,
        limit_val_batches=limit,
        limit_test_batches=limit,
        val_check_interval=(
            1.0 if (args.debug_max_step is not None or args.fast_dev_run)
            else args.val_check_interval
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=False,   # FlashAttention is non-deterministic
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
        head_dropout=args.head_dropout,
        head_hidden=args.head_hidden,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        max_epochs=args.max_epochs,
        rlop_patience=args.rlop_patience,
        rlop_factor=args.rlop_factor,
    )

    trainer.fit(model_module, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # Save test score
    is_rank_zero = (
        not torch.distributed.is_available() or
        not torch.distributed.is_initialized() or
        torch.distributed.get_rank() == 0
    )
    if is_rank_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"test_results: {test_results}\n"
            f"val_f1_best: {checkpoint_cb.best_model_score}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
