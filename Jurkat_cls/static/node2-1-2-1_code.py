#!/usr/bin/env python3
"""
Node 2-1-2-1: AIDO.Cell-10M + Symbol CNN + Frozen STRING GNN – Optimized for Tree-Best Alignment
==================================================================================================
Key design differences from parent node (node2-1-2 / test F1=0.4453):
  1. LoRA configuration: r=4 on ALL 8 layers (vs. r=8 last-4-layers in parent).
     node3-2's best score (0.4622) used r=4 all-8-layers; node3-3 with r=8 last-4 achieved
     only 0.4513. Distributing LoRA capacity across all 8 layers with lower rank per layer
     provides better generalization with the same total parameter count.

  2. STRING GNN projection expanded to 256-dim (single-layer, no bottleneck).
     Parent projected 256→128 with a 2-layer MLP. Reverting to the full 256-dim feature
     as used by node3-2 (832-dim fusion = 512+64+256) doubles the PPI signal capacity
     and matches the proven architecture that achieved 0.4622. Single linear projection
     preserves the information without intermediate compression loss.

  3. Head fusion dimension 832 (vs. 704 in parent) and head_hidden=384 (vs. 320).
     Directly adopting node3-2's head architecture: 832→384→19920. The larger head
     provides sufficient capacity to combine 3 feature sources without the compression
     bottleneck of the parent's 704→320 configuration.

  4. Class weights [5.0, 1.0, 10.0] (vs. parent's [3.0, 1.0, 7.0]).
     Restoring the stronger minority-class emphasis from node3-2's proven configuration.
     Higher minority-class weight directly helps the F1 metric which heavily depends on
     predicting rare down-regulated (-1) and up-regulated (+1) gene classes.

  5. Label smoothing 0.05 added.
     node3-2 and node3-3 both used label_smoothing=0.05, providing regularization on
     the 6,640-gene output space that compensates for the aggressive class weights.

  6. Head dropout increased to 0.4 (vs. 0.35 in parent).
     Matching node3-2's dropout configuration; provides better regularization for the
     wider 384-dim head with more trainable parameters.

  7. Weight decay 0.03 (vs. 0.01 in parent).
     node3-2 used weight_decay=0.03 which proved more effective at controlling overfitting
     while maintaining generalization; the current parent's 0.01 showed 16× train-val gap.

  8. ReduceLROnPlateau: patience=8, factor=0.5, monitor val_f1 (max).
     - patience=8 matches node3-2's proven setting that correctly fired reductions;
       patience=5 in parent fired too frequently (5 reductions) without F1 recovery.
     - factor=0.5 (vs. 0.7 in parent) gives larger LR step for more decisive optimization
       phase transitions, matching node3-2-1's finding that factor=0.5 triggers properly.
     - Keeping val_f1 (max) as the monitored metric — critical lesson from node3-2-2
       where val_loss monitoring caused premature LR reductions that regressed to 0.442.

  9. Initial backbone LR lowered to 2e-4 (vs. 3e-4 in parent), head/sym/GNN LR=6e-4 (3×).
     Parent's feedback noted the 3e-4 backbone LR held 38 epochs before first reduction
     while val_loss climbed steadily — early overfitting. Lower initial LR with same
     patience forces earlier LR reductions, reducing the overfitting window.

Key differences from sibling nodes:
  - Adopts node3-2's proven architectural configuration (r=4 all-8 + 832-dim + 384 head)
    which is the highest-performing configuration in the tree for this input paradigm.
  - Retains the stable ReduceLROnPlateau on val_f1 (NOT val_loss) to avoid node3-2-2's
    premature LR reduction failure.
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

# Restored to stronger minority-class weights matching node3-2's proven config (0.4622)
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)

# Character-level encoding for gene symbols
SYMBOL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-." + "_"  # + padding
CHAR_VOCAB_SIZE = len(SYMBOL_CHARS) + 1  # +1 for unknown
SYMBOL_MAX_LEN = 30  # max symbol length (padded/truncated to this)

# STRING GNN constants — using full 256-dim (no bottleneck), matching node3-2
STRING_GNN_DIM = 256    # STRING GNN hidden dimension
STRING_PROJ_DIM = 256   # Projected to full 256-dim (changed from parent's 128)


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss with label smoothing
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weights and label smoothing."""

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
        ce = F.cross_entropy(
            logits, targets, weight=w,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # Compute pt WITHOUT label smoothing for focal weighting (standard practice)
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


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
        self.dropout = nn.Dropout(0.2)  # Slightly higher (0.1→0.2) for better regularization
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
    """AIDO.Cell-10M (LoRA r=4 all-8-layers) + Symbol CNN + Frozen STRING GNN.

    Key differences from parent:
    - LoRA r=4 applied to ALL 8 transformer layers (vs. r=8 last-4 in parent)
    - STRING GNN projected to 256-dim (single linear, no bottleneck, vs. parent's 2-layer 128-dim)
    - Fusion dimension: 512 + 64 + 256 = 832-dim (vs. parent's 704-dim)
    - Head: 832→384→19920 (vs. parent's 704→320→19920)
    """

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        head_dropout: float = 0.4,
        head_hidden: int = 384,
        string_gnn_emb: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA on ALL 8 layers ──
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA r=4 on QKV of ALL 8 layers (layers 0-7)
        # This matches node3-2's proven configuration (0.4622 tree-best)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            # layers_to_transform=None means apply to ALL layers
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

        # ── STRING GNN projection (frozen GNN embeddings → 256-dim, single linear) ──
        # The pre-computed embeddings are stored as a buffer; the projection is learned
        # Single-layer projection preserves full information (no bottleneck compression)
        if string_gnn_emb is not None:
            self.register_buffer("gnn_emb", string_gnn_emb.float())
        else:
            self.register_buffer("gnn_emb", None)

        # Single-layer projection (256→256) — minimal capacity for PPI features
        # Following node3-2's approach (1-layer projection is proven sufficient)
        self.gnn_proj = nn.Sequential(
            nn.Linear(STRING_GNN_DIM, STRING_PROJ_DIM),
            nn.GELU(),
        )

        # ── Prediction head ──
        # Input: AIDO_HIDDEN * 2 (512) + 64 (symbol) + STRING_PROJ_DIM (256) = 832
        head_in = AIDO_HIDDEN * 2 + 64 + STRING_PROJ_DIM  # 832
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
            gnn_proj_emb = self.gnn_proj(gnn_feats)  # [B, 256]
        else:
            gnn_proj_emb = torch.zeros(B, STRING_PROJ_DIM, device=expr.device, dtype=torch.float32)

        # ── Fusion → head ──
        combined = torch.cat([backbone_emb, sym_emb, gnn_proj_emb], dim=-1)  # [B, 832]
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
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        head_dropout: float = 0.4,
        head_hidden: int = 384,
        lr: float = 2e-4,
        weight_decay: float = 0.03,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 150,
        rlop_patience: int = 8,
        rlop_factor: float = 0.5,
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
                label_smoothing=self.hparams.label_smoothing,
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
        max_lr_other = self.hparams.lr * 3  # 3× higher for randomly-init modules (reduced from parent's 5×)

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": max_lr_backbone},
                {"params": other_params, "lr": max_lr_other},
            ],
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # ReduceLROnPlateau monitoring val_f1 (max):
        # CRITICAL: Do NOT monitor val_loss — node3-2-2 showed that val_loss monitoring
        # caused premature LR reductions (fires when val_f1 still improving) and regressed from
        # 0.462 to 0.442. val_f1 is the only reliable scheduling signal for this task with focal loss.
        # Patience=8 (matches node3-2) allows full exploitation before each reduction.
        # Factor=0.5 provides decisive LR steps for phase transitions.
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
        description="Node 2-1-2-1: AIDO.Cell-10M + Symbol CNN + STRING GNN DEG predictor (node3-2 aligned)"
    )
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--head-dropout", type=float, default=0.4)
    p.add_argument("--head-hidden", type=int, default=384)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--rlop-patience", type=int, default=8)
    p.add_argument("--rlop-factor", type=float, default=0.5)
    p.add_argument("--early-stopping-patience", type=int, default=25)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def main():
    pl.seed_everything(0)
    args = parse_args()

    # Resolve data_dir relative to project root (3 levels up from mcts/node2-1-2-1/)
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
        filename="node2-1-2-1-{epoch:03d}-{val_f1:.4f}",
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
        label_smoothing=args.label_smoothing,
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
