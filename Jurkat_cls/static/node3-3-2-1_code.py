#!/usr/bin/env python3
"""
Node 3-3-2-1: AIDO.Cell-10M + LoRA r=4 all-8 + Symbol CNN + Frozen STRING GNN PPI
             + 3-layer TransformerEncoder Cross-Attention Fusion + Manifold Mixup
===================================================================================
Improved version of parent node3-3-2 (test F1=0.4555).

The critical paradigm shift in this node:
  Concat+MLP HEAD (parent, ceiling ~0.46) → Cross-Attention Fusion (tree-best: 0.5049)

Key improvements from parent (node3-3-2, test F1=0.4555):

1. ARCHITECTURE PARADIGM SHIFT: Cross-Attention Fusion (3-layer TransformerEncoder)
   - Parent used concat([global_emb, pert_emb, sym_feat, ppi_feat]) → 832-dim MLP head
   - This node uses 4-token TransformerEncoder self-attention over 4 feature tokens
   - Proven in tree-best node3-1-1-1-1-1-2 (F1=0.5049): +0.031 over the concat+MLP ceiling
   - Cross-attention allows inter-source feature interactions (PPI ↔ expression ↔ symbol)
   - Architecture: 4 tokens [global, pert, sym, ppi] → 3-layer TransformerEncoder →
                   mean-pool → 256-dim → MLP → [B, 3, 6640]

2. REMOVE SWA (parent feedback: SWA actively destabilized performance)
   - SWA started after best epoch, replaced ReduceLROnPlateau, val_loss exploded +30%
   - Without SWA: ReduceLROnPlateau can fire at the right moment for breakthrough

3. MONITOR val_f1 (not val_loss) for ReduceLROnPlateau
   - Tree-best achieved breakthrough at epoch 37 when LR reduced 2e-4→1e-4 on val_f1
   - Parent/siblings monitored val_loss which oscillates heavily under focal loss
   - val_f1 monitoring is more stable and triggers reduction at the optimal time

4. MANIFOLD MIXUP (alpha=0.3)
   - Proven regularization in all cross-attention top-performers (node3-1-1-1-1-1-2 etc.)
   - Interpolates in the 256-dim fusion feature space (not input space)
   - Reduces overfitting while maintaining gradient flow through fusion transformer

5. STRONGER REGULARIZATION: weight_decay=0.10
   - Tree-best cross-attention nodes consistently use wd=0.10
   - Parent's wd=0.03 is tuned for concat+MLP, insufficient for cross-attention
   - wd=0.10 prevents the fusion transformer from memorizing training patterns

6. EXTENDED TRAINING: max_epochs=150, patience=30
   - Cross-attention nodes need longer training than concat+MLP (typically 60-100 epochs)
   - node2-2-1-1-2-1 (F1=0.4843) trained 126 epochs, peak at epoch 71
   - patience=30 allows the model to fully explore the low-LR regime after reduction

7. TOP-3 CHECKPOINT AVERAGING (after training)
   - node2-2-1-1-2-1 (F1=0.4843) used top-5 averaging; tree-best used top-3
   - Reduces variance from single-checkpoint selection
   - Implemented as post-training weight averaging over top-3 val_f1 checkpoints

Training Configuration:
  - Global batch: 64 (micro_batch=8, 8 GPUs, accumulate=1)
  - LR: backbone=2e-4, head=6e-4, symbol=4e-4, ppi=5e-4 (differential LR)
  - Weight decay: 0.10
  - LR scheduler: ReduceLROnPlateau on val_f1 (patience=12, factor=0.5)
  - Early stopping: patience=30 on val_f1
  - Max epochs: 150
  - Manifold mixup: alpha=0.3 in 256-dim fusion space
  - Class weights: [6.0, 1.0, 12.0] (proven in node2-2-3-1-1)
  - Focal loss gamma: 1.5 (matching tree-best cross-attention nodes)
  - Label smoothing: 0.05
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault('NCCL_IB_DISABLE', '1')
os.environ.setdefault('NCCL_NET_GDR_LEVEL', '0')

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
N_GENES_OUT = 6_640      # output genes
N_CLASSES = 3
SENTINEL_EXPR = 1.0      # baseline expression for non-perturbed genes
KNOCKOUT_EXPR = 0.0      # expression for knocked-out gene
AIDO_HIDDEN = 256        # AIDO.Cell-10M hidden dimension
AIDO_N_LAYERS = 8        # AIDO.Cell-10M transformer layers
STRING_GNN_DIM = 256     # STRING GNN embedding dimension
FUSION_DIM = 256         # cross-attention fusion hidden dimension

# Stronger class weights proven in node2-2-3-1-1 (0.4625 F1) and tree-best lineages
# Train: class 0 (down) ~3.4%, class 1 (unchanged) ~95.5%, class 2 (up) ~1.1%
CLASS_WEIGHTS = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)

# Character vocabulary for gene symbol encoding
SYMBOL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
SYMBOL_PAD_IDX = len(SYMBOL_CHARS)          # 39 → padding index
SYMBOL_VOCAB_SIZE = len(SYMBOL_CHARS) + 1   # 40
SYMBOL_MAX_LEN = 12                          # max gene symbol length


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss with label smoothing
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weights and label smoothing."""

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
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=w, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Gene Symbol Encoder (Character-level CNN)
# ──────────────────────────────────────────────────────────────────────────────
def symbol_to_indices(symbol: str) -> List[int]:
    """Convert a gene symbol string to a list of character indices."""
    char_to_idx = {c: i for i, c in enumerate(SYMBOL_CHARS)}
    indices = []
    for ch in symbol.upper()[:SYMBOL_MAX_LEN]:
        idx = char_to_idx.get(ch, SYMBOL_PAD_IDX)
        indices.append(idx)
    while len(indices) < SYMBOL_MAX_LEN:
        indices.append(SYMBOL_PAD_IDX)
    return indices


class SymbolEncoder(nn.Module):
    """
    Character-level CNN encoder for gene symbol strings.
    Three parallel Conv1d filters at kernel sizes 2, 3, 4 → max-pool → 256-dim (=FUSION_DIM).
    Output is projected to FUSION_DIM to be used as one of the 4 cross-attention tokens.
    """

    def __init__(self, out_dim: int = FUSION_DIM, embed_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(SYMBOL_VOCAB_SIZE, embed_dim, padding_idx=SYMBOL_PAD_IDX)
        self.conv2 = nn.Conv1d(embed_dim, 64, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, 64, kernel_size=4, padding=2)
        self.proj = nn.Sequential(
            nn.Linear(192, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for conv in [self.conv2, self.conv3, self.conv4]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, symbol_ids: torch.Tensor) -> torch.Tensor:
        """symbol_ids: [B, SYMBOL_MAX_LEN] int64 → [B, out_dim]"""
        x = self.embed(symbol_ids)          # [B, L, embed_dim]
        x = x.transpose(1, 2)              # [B, embed_dim, L]
        f2 = F.relu(self.conv2(x))
        f3 = F.relu(self.conv3(x))
        f4 = F.relu(self.conv4(x))
        f2 = F.adaptive_max_pool1d(f2, 1).squeeze(-1)  # [B, 64]
        f3 = F.adaptive_max_pool1d(f3, 1).squeeze(-1)  # [B, 64]
        f4 = F.adaptive_max_pool1d(f4, 1).squeeze(-1)  # [B, 64]
        feat = torch.cat([f2, f3, f4], dim=-1)          # [B, 192]
        return self.proj(feat)                           # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# STRING GNN Static Embedding Extractor
# ──────────────────────────────────────────────────────────────────────────────
def build_frozen_string_embeddings(
    string_gnn_dir: str,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Run STRING GNN once (inference only) to extract frozen [18870, 256] node embeddings.
    Returns (frozen_embeddings, ensg_to_gnn_idx_mapping).
    """
    import json as _json
    from transformers import AutoModel as _AutoModel

    model_dir = Path(string_gnn_dir)
    node_names = _json.loads((model_dir / "node_names.json").read_text())
    ensg_to_gnn_idx = {n: i for i, n in enumerate(node_names) if n.startswith("ENSG")}

    graph = torch.load(model_dir / "graph_data.pt")
    edge_index = graph["edge_index"]
    edge_weight = graph.get("edge_weight", None)

    # Load STRING GNN and run inference on CPU
    gnn_model = _AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    gnn_model.eval()

    # Use CPU for this one-time computation to avoid GPU memory issues
    with torch.no_grad():
        outputs = gnn_model(
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
    # embeddings: [18870, 256] float32
    embeddings = outputs.last_hidden_state.cpu().float()

    del gnn_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return embeddings, ensg_to_gnn_idx


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Returns pre-built AIDO.Cell expression profiles, gene positions, symbol indices,
    STRING GNN embedding indices, and labels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],    # ENSG_base → AIDO.Cell position [0, 19264)
        ensg_to_gnn: Dict[str, int],    # ENSG_base → STRING GNN node index
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.ensg_to_gnn = ensg_to_gnn
        self.is_test = is_test

        # Pre-build AIDO.Cell expression profile tensors: [N, 19264] float32
        # Baseline: 1.0 everywhere, knocked-out gene: 0.0
        self.expr_inputs = self._build_expr_tensors()

        # Pre-build symbol character index tensors: [N, SYMBOL_MAX_LEN] int64
        self.symbol_ids = self._build_symbol_tensors()

        # Pre-build STRING GNN index tensors: [N] int64 (-1 if not in STRING)
        self.gnn_indices = self._build_gnn_indices()

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1} → {0,1,2}
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

    def _build_symbol_tensors(self) -> torch.Tensor:
        N = len(self.symbols)
        sym_ids = torch.zeros((N, SYMBOL_MAX_LEN), dtype=torch.long)
        for i, symbol in enumerate(self.symbols):
            sym_ids[i] = torch.tensor(symbol_to_indices(symbol), dtype=torch.long)
        return sym_ids

    def _build_gnn_indices(self) -> torch.Tensor:
        """Build [N] int64 tensor of STRING GNN node indices (-1 if not found)."""
        N = len(self.pert_ids)
        gnn_idx = torch.full((N,), -1, dtype=torch.long)
        for i, pert_id in enumerate(self.pert_ids):
            base = pert_id.split(".")[0]
            idx = self.ensg_to_gnn.get(base)
            if idx is not None:
                gnn_idx[i] = idx
        return gnn_idx

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.pert_ids[idx].split(".")[0]
        gene_pos = self.gene_to_pos.get(base, -1)
        item = {
            "idx": idx,
            "expr": self.expr_inputs[idx],        # [19264] float32
            "gene_pos": gene_pos,                  # int (-1 if not in AIDO vocab)
            "symbol_ids": self.symbol_ids[idx],    # [SYMBOL_MAX_LEN] int64
            "gnn_idx": self.gnn_indices[idx].item(),  # int (-1 if not in STRING)
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "expr": torch.stack([b["expr"] for b in batch]),              # [B, 19264]
        "gene_pos": torch.tensor([b["gene_pos"] for b in batch], dtype=torch.long),
        "symbol_ids": torch.stack([b["symbol_ids"] for b in batch]),  # [B, SYMBOL_MAX_LEN]
        "gnn_idx": torch.tensor([b["gnn_idx"] for b in batch], dtype=torch.long),  # [B]
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
        self.ensg_to_gnn: Dict[str, int] = {}
        self.frozen_gnn_embs: Optional[torch.Tensor] = None  # [18870, 256]
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        # Rank-0 downloads tokenizer first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # Build ENSG→position mapping for AIDO.Cell
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)
            print(f"[DEGDataModule] AIDO.Cell gene vocab coverage: "
                  f"{len(self.gene_to_pos)}/{len(unique_ids)} genes")

        # Build frozen STRING GNN embeddings once
        if self.frozen_gnn_embs is None:
            print("[DEGDataModule] Building frozen STRING GNN embeddings (one-time)...")
            self.frozen_gnn_embs, self.ensg_to_gnn = build_frozen_string_embeddings(
                STRING_GNN_MODEL_DIR, device="cpu"
            )
            print(f"[DEGDataModule] STRING GNN embeddings: {self.frozen_gnn_embs.shape}, "
                  f"coverage: {len(self.ensg_to_gnn)} ENSG IDs in STRING")

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self.gene_to_pos, self.ensg_to_gnn)
            self.val_ds = PerturbationDataset(val_df, self.gene_to_pos, self.ensg_to_gnn)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self.gene_to_pos, self.ensg_to_gnn, is_test=True
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
# Cross-Attention Fusion Model
# ──────────────────────────────────────────────────────────────────────────────
class AIDOCellCrossAttnDEGModel(nn.Module):
    """
    AIDO.Cell-10M + LoRA (r=4, ALL 8 layers) + Symbol CNN + Frozen STRING GNN
    → 4-token Cross-Attention Fusion → 3-class per-gene DEG prediction.

    Architecture:
      4 feature tokens (each FUSION_DIM=256):
        (a) global_emb = AIDO mean-pool over 19264 gene positions → [B, 256]
        (b) pert_emb   = AIDO positional emb at perturbed gene    → [B, 256]
        (c) sym_feat   = Symbol CNN → project to 256-dim          → [B, 256]
        (d) ppi_feat   = STRING GNN lookup → 1-layer proj          → [B, 256]

      Tokens stacked: [B, 4, 256] → 3-layer TransformerEncoder → mean-pool → [B, 256]
      MLP head: LayerNorm(256) → Linear(256→256) → GELU → Dropout → Linear(256→19920)
      Output: [B, 19920] → reshape → [B, 3, 6640]

    Key design decision: d_model=256 (=AIDO_HIDDEN) avoids projection mismatches.
    Each source is projected to 256-dim before stacking as tokens.
    The fusion transformer uses dim_ff=384 (per tree-best recommendation).
    """

    HIDDEN_DIM = 256          # AIDO.Cell-10M hidden size
    PPI_DIM = 256             # STRING GNN embedding dim
    FUSION_D = FUSION_DIM     # = 256, dimension for all fusion tokens

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        head_hidden: int = 256,
        head_dropout: float = 0.4,
        attn_n_layers: int = 3,
        attn_n_heads: int = 8,
        attn_ff_dim: int = 384,
        attn_dropout: float = 0.1,
        frozen_gnn_embs: Optional[torch.Tensor] = None,  # [18870, 256] float32
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA on ALL 8 layers ──────────────────
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA on Q/K/V of ALL 8 transformer layers (proven optimal from node3-2)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            # No layers_to_transform → applies to ALL layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Cast trainable LoRA params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Gene symbol character-level CNN encoder → 256-dim ─────────────────
        # Output dim matches FUSION_DIM=256 for direct use as fusion token
        self.symbol_encoder = SymbolEncoder(out_dim=self.FUSION_D, embed_dim=32)

        # ── Frozen STRING GNN embedding table ─────────────────────────────────
        if frozen_gnn_embs is not None:
            self.register_buffer("gnn_emb_table", frozen_gnn_embs)  # [18870, 256]
        else:
            self.register_buffer("gnn_emb_table",
                                 torch.zeros(18870, self.PPI_DIM, dtype=torch.float32))

        # Learnable fallback embedding for genes not in STRING GNN (~6.5% of genes)
        self.gnn_fallback = nn.Parameter(torch.zeros(self.PPI_DIM))
        nn.init.trunc_normal_(self.gnn_fallback, std=0.02)

        # 1-layer projection to align STRING GNN embedding with AIDO.Cell space
        # Input: 256-dim PPI, Output: 256-dim (=FUSION_DIM)
        self.ppi_proj = nn.Sequential(
            nn.Linear(self.PPI_DIM, self.FUSION_D),
            nn.GELU(),
            nn.LayerNorm(self.FUSION_D),
        )
        nn.init.xavier_uniform_(self.ppi_proj[0].weight)
        nn.init.zeros_(self.ppi_proj[0].bias)

        # ── 3-layer TransformerEncoder cross-attention fusion ──────────────────
        # 4 tokens of dim FUSION_D=256
        # Proven architecture from tree-best (node3-1-1-1-1-1-2, F1=0.5049):
        #   nhead=8, dim_feedforward=384, dropout=0.1, 3 layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.FUSION_D,
            nhead=attn_n_heads,
            dim_feedforward=attn_ff_dim,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=attn_n_layers,
            enable_nested_tensor=False,
        )

        # ── Prediction head ────────────────────────────────────────────────────
        # FUSION_DIM → head_hidden → N_CLASSES * N_GENES_OUT
        # head_hidden=256 matches the fusion dimension
        self.head = nn.Sequential(
            nn.LayerNorm(self.FUSION_D),
            nn.Linear(self.FUSION_D, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        expr: torch.Tensor,          # [B, 19264] float32
        gene_pos: torch.Tensor,      # [B]         int64 (-1 if not in AIDO vocab)
        symbol_ids: torch.Tensor,    # [B, SYMBOL_MAX_LEN] int64
        gnn_idx: torch.Tensor,       # [B]         int64 (-1 if not in STRING)
    ) -> torch.Tensor:
        B = expr.shape[0]

        # ── AIDO.Cell backbone forward ─────────────────────────────────────────
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, 19266, 256]

        # (a) Global mean-pool over all gene positions (exclude 2 summary tokens)
        gene_emb = lhs[:, :N_GENES_AIDO, :]          # [B, 19264, 256]
        global_emb = gene_emb.mean(dim=1)             # [B, 256]

        # (b) Perturbed-gene positional embedding
        pert_emb = torch.zeros(B, self.HIDDEN_DIM, device=lhs.device, dtype=lhs.dtype)
        valid_aido = gene_pos >= 0
        if valid_aido.any():
            valid_pos = gene_pos[valid_aido]  # [k]
            pert_emb[valid_aido] = lhs[valid_aido, valid_pos, :]
        pert_emb[~valid_aido] = global_emb[~valid_aido]

        # Convert backbone features to float32 for fusion computation
        global_emb_f = global_emb.float()    # [B, 256]
        pert_emb_f = pert_emb.float()        # [B, 256]

        # (c) Gene symbol character CNN → [B, 256]
        sym_feat = self.symbol_encoder(symbol_ids)    # [B, 256] float32

        # (d) Frozen STRING GNN PPI embedding lookup
        ppi_emb = self.gnn_fallback.unsqueeze(0).expand(B, -1).clone()  # [B, 256]
        valid_gnn = gnn_idx >= 0
        if valid_gnn.any():
            valid_gnn_idx = gnn_idx[valid_gnn]  # [k]
            with torch.no_grad():
                ppi_raw = self.gnn_emb_table[valid_gnn_idx]  # [k, 256]
            ppi_emb[valid_gnn] = ppi_raw

        # Project PPI features (trainable linear to adapt frozen embeddings)
        ppi_feat = self.ppi_proj(ppi_emb.to(global_emb_f.device))  # [B, 256]

        # ── Stack 4 tokens and run through TransformerEncoder ─────────────────
        # tokens: [B, 4, 256]
        tokens = torch.stack(
            [global_emb_f, pert_emb_f, sym_feat, ppi_feat], dim=1
        )  # [B, 4, FUSION_DIM]

        # Cross-attention fusion (all 4 tokens attend to each other)
        fused = self.fusion_transformer(tokens)  # [B, 4, 256]

        # Mean-pool over 4 tokens → [B, 256]
        fused_mean = fused.mean(dim=1)  # [B, 256]

        # ── Prediction head ────────────────────────────────────────────────────
        logits = self.head(fused_mean)                   # [B, 3 * 6640]
        return logits.view(B, N_CLASSES, N_GENES_OUT)    # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro F1, averaged over all genes.
    y_pred: [n_samples, 3, n_genes]  (3-class probability distributions)
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
        head_hidden: int = 256,
        head_dropout: float = 0.4,
        attn_n_layers: int = 3,
        attn_n_heads: int = 8,
        attn_ff_dim: int = 384,
        attn_dropout: float = 0.1,
        lr: float = 2e-4,
        head_lr_multiplier: float = 3.0,
        symbol_encoder_lr_multiplier: float = 2.0,
        ppi_proj_lr_multiplier: float = 2.5,
        weight_decay: float = 0.10,
        gamma_focal: float = 1.5,
        label_smoothing: float = 0.05,
        # LR scheduler — monitor val_f1 for reliable breakthrough (proven in tree-best)
        plateau_patience: int = 12,
        plateau_factor: float = 0.5,
        plateau_min_lr: float = 1e-7,
        # Manifold mixup
        mixup_alpha: float = 0.3,
        # Top-K checkpoint averaging
        top_k_avg: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCellCrossAttnDEGModel] = None
        self.criterion: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._mixup_alpha = mixup_alpha

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            # Get frozen STRING GNN embeddings from the datamodule
            frozen_gnn_embs = None
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
                dm = self.trainer.datamodule
                if hasattr(dm, "frozen_gnn_embs") and dm.frozen_gnn_embs is not None:
                    frozen_gnn_embs = dm.frozen_gnn_embs

            self.model = AIDOCellCrossAttnDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                head_hidden=self.hparams.head_hidden,
                head_dropout=self.hparams.head_dropout,
                attn_n_layers=self.hparams.attn_n_layers,
                attn_n_heads=self.hparams.attn_n_heads,
                attn_ff_dim=self.hparams.attn_ff_dim,
                attn_dropout=self.hparams.attn_dropout,
                frozen_gnn_embs=frozen_gnn_embs,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        if stage == "test" and hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            batch["expr"],
            batch["gene_pos"],
            batch["symbol_ids"],
            batch["gnn_idx"],
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_flat = labels.reshape(-1)
        return self.criterion(logits_flat, labels_flat)

    def _manifold_mixup(
        self,
        tokens: torch.Tensor,    # [B, 4, 256] fusion token sequence
        labels: torch.Tensor,    # [B, N_GENES_OUT] long
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Manifold Mixup in the 4-token feature space (after source projection, before fusion).
        Mix two random samples with weight lambda drawn from Beta(alpha, alpha).
        Returns: (mixed_tokens, labels_a, labels_b, lam)
        """
        if self.hparams.mixup_alpha > 0 and self.training:
            lam = float(
                torch.distributions.Beta(
                    self.hparams.mixup_alpha, self.hparams.mixup_alpha
                ).sample().item()
            )
        else:
            return tokens, labels, labels, 1.0

        B = tokens.shape[0]
        perm = torch.randperm(B, device=tokens.device)
        mixed_tokens = lam * tokens + (1 - lam) * tokens[perm]
        labels_b = labels[perm]
        return mixed_tokens, labels, labels_b, lam

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Build features up to the token stacking stage
        expr = batch["expr"]
        gene_pos = batch["gene_pos"]
        symbol_ids = batch["symbol_ids"]
        gnn_idx = batch["gnn_idx"]
        labels = batch["label"]

        B = expr.shape[0]

        # ── AIDO.Cell backbone forward ─────────────────────────────────────────
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.model.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, 19266, 256]

        gene_emb = lhs[:, :N_GENES_AIDO, :]
        global_emb = gene_emb.mean(dim=1)

        pert_emb = torch.zeros(B, AIDO_HIDDEN, device=lhs.device, dtype=lhs.dtype)
        valid_aido = gene_pos >= 0
        if valid_aido.any():
            valid_pos = gene_pos[valid_aido]
            pert_emb[valid_aido] = lhs[valid_aido, valid_pos, :]
        pert_emb[~valid_aido] = global_emb[~valid_aido]

        global_emb_f = global_emb.float()
        pert_emb_f = pert_emb.float()

        sym_feat = self.model.symbol_encoder(symbol_ids)  # [B, 256]

        ppi_emb = self.model.gnn_fallback.unsqueeze(0).expand(B, -1).clone()
        valid_gnn = gnn_idx >= 0
        if valid_gnn.any():
            valid_gnn_idx = gnn_idx[valid_gnn]
            with torch.no_grad():
                ppi_raw = self.model.gnn_emb_table[valid_gnn_idx]
            ppi_emb[valid_gnn] = ppi_raw

        ppi_feat = self.model.ppi_proj(ppi_emb.to(global_emb_f.device))

        # Stack 4 tokens → [B, 4, 256]
        tokens = torch.stack([global_emb_f, pert_emb_f, sym_feat, ppi_feat], dim=1)

        # Apply manifold mixup in the token space
        mixed_tokens, labels_a, labels_b, lam = self._manifold_mixup(tokens, labels)

        # Fuse via TransformerEncoder
        fused = self.model.fusion_transformer(mixed_tokens)  # [B, 4, 256]
        fused_mean = fused.mean(dim=1)  # [B, 256]
        logits = self.model.head(fused_mean)  # [B, 3*6640]
        logits = logits.view(B, N_CLASSES, N_GENES_OUT)

        # Mixed loss: lam * loss(a) + (1-lam) * loss(b)
        loss_a = self._compute_loss(logits, labels_a)
        if lam < 1.0 and not torch.equal(labels_a, labels_b):
            loss_b = self._compute_loss(logits, labels_b)
            loss = lam * loss_a + (1 - lam) * loss_b
        else:
            loss = loss_a

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

        preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
        labels = al.cpu().view(-1, N_GENES_OUT).numpy()
        idxs = ai.cpu().view(-1).numpy()
        _, uniq = np.unique(idxs, return_index=True)
        f1 = compute_deg_f1(preds[uniq], labels[uniq])

        # All-reduce so all ranks log the same val_f1
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

        dm = None
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
        all_pert_ids = getattr(dm, "test_pert_ids", None) or self._test_pert_ids
        all_symbols = getattr(dm, "test_symbols", None) or self._test_symbols

        is_global_zero = (
            getattr(self.trainer, "is_global_zero", None) or
            (getattr(self.trainer, "global_rank", 0) == 0)
        )
        if is_global_zero:
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
                pert_id = (all_pert_ids[int(orig_i)]
                           if all_pert_ids is not None and int(orig_i) < len(all_pert_ids)
                           else str(orig_i))
                symbol = (all_symbols[int(orig_i)]
                          if all_symbols is not None and int(orig_i) < len(all_symbols)
                          else "")
                rows.append({
                    "idx": pert_id,
                    "input": symbol,
                    "prediction": json.dumps(preds[rank_i].tolist()),
                })
            pd.DataFrame(rows).to_csv(output_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"Test predictions saved ({len(rows)} samples) → {output_dir / 'test_predictions.tsv'}")

    def configure_optimizers(self):
        hp = self.hparams
        # Parameter groups with different learning rates
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        ppi_proj_params = (
            list(self.model.ppi_proj.parameters()) +
            [self.model.gnn_fallback]
        )
        symbol_params = list(self.model.symbol_encoder.parameters())
        # Fusion transformer + head share "head" LR
        fusion_head_params = (
            list(self.model.fusion_transformer.parameters()) +
            list(self.model.head.parameters())
        )

        backbone_lr = hp.lr                                          # 2e-4
        ppi_lr = hp.lr * hp.ppi_proj_lr_multiplier                 # 5e-4
        symbol_lr = hp.lr * hp.symbol_encoder_lr_multiplier        # 4e-4
        head_lr = hp.lr * hp.head_lr_multiplier                    # 6e-4

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": ppi_proj_params, "lr": ppi_lr},
                {"params": symbol_params, "lr": symbol_lr},
                {"params": fusion_head_params, "lr": head_lr},
            ],
            weight_decay=hp.weight_decay,
        )

        # Monitor val_f1 for ReduceLROnPlateau (proven to trigger at the right moment for tree-best)
        # patience=12 allows sufficient exploration before each reduction
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",        # maximize val_f1
            factor=hp.plateau_factor,
            patience=hp.plateau_patience,
            min_lr=hp.plateau_min_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_f1",   # val_f1 monitoring (proven for breakthrough in tree-best)
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
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
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        pct = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0
        msg = (
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({pct:.2f}%), plus {total_buffers} buffer values"
        )
        try:
            self.print(msg)
        except RuntimeError:
            print(msg)
        return out

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable parameters and buffers from a partial checkpoint."""
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

        def _safe_print(msg: str) -> None:
            try:
                self.print(msg)
            except RuntimeError:
                print(msg)

        if missing_keys:
            _safe_print(f"Warning: Missing checkpoint keys: {missing_keys[:5]}...")
        if unexpected_keys:
            _safe_print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

        loaded_trainable = len([k for k in state_dict if k in trainable_keys])
        loaded_buffers = len([k for k in state_dict if k in buffer_keys])
        _safe_print(
            f"Loading checkpoint: {loaded_trainable} trainable parameters and "
            f"{loaded_buffers} buffers"
        )
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Top-K Checkpoint Averaging
# ──────────────────────────────────────────────────────────────────────────────
def apply_top_k_checkpoint_averaging(
    model_module: DEGLightningModule,
    checkpoint_cb: ModelCheckpoint,
    datamodule: DEGDataModule,
    trainer: pl.Trainer,
    top_k: int = 3,
    output_dir: Path = None,
) -> None:
    """
    Post-training top-K checkpoint averaging.
    Loads the top-K best checkpoints (by val_f1), averages their trainable
    parameters, and runs test inference with the averaged model.

    This is equivalent to what node2-2-1-1-2-1 (F1=0.4843) used with top-5 averaging
    and tree-best node3-1-1-1-1-1-2 (F1=0.5049) used with top-3 averaging.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "run"

    # Get top-K checkpoint paths by val_f1 score
    best_k_paths = checkpoint_cb.best_k_models
    if not best_k_paths:
        print("[TopKAvg] No checkpoint paths found, skipping checkpoint averaging.")
        return

    # Sort by score (descending for val_f1 maximization)
    sorted_ckpts = sorted(
        best_k_paths.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    if len(sorted_ckpts) < 2:
        print(f"[TopKAvg] Only {len(sorted_ckpts)} checkpoint(s) found; skipping averaging.")
        return

    print(f"[TopKAvg] Averaging top-{len(sorted_ckpts)} checkpoints:")
    for path, score in sorted_ckpts:
        print(f"  {path}: val_f1={score:.6f}")

    # Load all checkpoint state dicts
    ckpt_state_dicts = []
    for path, score in sorted_ckpts:
        try:
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            ckpt_state_dicts.append(state)
        except Exception as e:
            print(f"[TopKAvg] Warning: Could not load {path}: {e}")

    if len(ckpt_state_dicts) < 2:
        print("[TopKAvg] Not enough valid checkpoints to average; skipping.")
        return

    # Average the state dicts
    avg_state = {}
    n = len(ckpt_state_dicts)
    for key in ckpt_state_dicts[0].keys():
        tensors = []
        for sd in ckpt_state_dicts:
            if key in sd:
                tensors.append(sd[key].float())
        if len(tensors) == n:
            avg_state[key] = torch.stack(tensors).mean(dim=0)
        elif tensors:
            avg_state[key] = tensors[0]

    # ── Build a proper Lightning-format checkpoint with required metadata ──────────
    # Load the first (best) checkpoint as a template to preserve all Lightning metadata
    # (pytorch-lightning_version, hyper_parameters, callbacks, etc.)
    best_path = sorted_ckpts[0][0]
    try:
        template_ckpt = torch.load(best_path, map_location="cpu")
    except Exception as e:
        print(f"[TopKAvg] Could not load template checkpoint {best_path}: {e}")
        # Fallback: save minimal checkpoint (may fail later in trainer.test)
        avg_ckpt_path = output_dir / "checkpoints" / "top_k_averaged.ckpt"
        avg_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": avg_state}, str(avg_ckpt_path))
        print(f"[TopKAvg] Averaged checkpoint saved (minimal format) → {avg_ckpt_path}")
        return

    # Build the averaged checkpoint by merging averaged state dict into template
    avg_ckpt = dict(template_ckpt)
    avg_ckpt["state_dict"] = avg_state
    # Remove epoch/step tracking so trainer doesn't confuse this with a normal epoch
    avg_ckpt.pop("epoch", None)
    avg_ckpt.pop("global_step", None)

    # Save averaged checkpoint with full Lightning metadata
    avg_ckpt_path = output_dir / "checkpoints" / "top_k_averaged.ckpt"
    avg_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avg_ckpt, str(avg_ckpt_path))
    print(f"[TopKAvg] Averaged checkpoint saved (Lightning format) → {avg_ckpt_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 3-3-2-1: AIDO.Cell-10M + LoRA r=4 all-8 + Cross-Attention Fusion + Manifold Mixup"
    )
    # Data path: data/ relative to the project root (3 levels up from mcts/node3-3-2-1/)
    p.add_argument("--data-dir", type=str, default=str(Path(__file__).parent.parent.parent / "data"))
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    # Extended max_epochs to allow cross-attention to find its optimum (typically E30-80)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--head-lr-multiplier", type=float, default=3.0)
    p.add_argument("--symbol-encoder-lr-multiplier", type=float, default=2.0)
    p.add_argument("--ppi-proj-lr-multiplier", type=float, default=2.5)
    # weight_decay=0.10 proven for cross-attention architecture (vs 0.03 for concat+MLP)
    p.add_argument("--weight-decay", type=float, default=0.10)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    # Cross-attention fusion hyperparameters
    p.add_argument("--attn-n-layers", type=int, default=3)
    p.add_argument("--attn-n-heads", type=int, default=8)
    p.add_argument("--attn-ff-dim", type=int, default=384)
    p.add_argument("--attn-dropout", type=float, default=0.1)
    # Head hyperparameters (smaller head as fusion transformer handles representations)
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.4)
    # Focal loss: gamma=1.5 matches tree-best cross-attention nodes
    p.add_argument("--gamma-focal", type=float, default=1.5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    # Monitor val_f1 for ReduceLROnPlateau (not val_loss)
    p.add_argument("--plateau-patience", type=int, default=12)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--plateau-min-lr", type=float, default=1e-7)
    # patience=30 for early stopping: cross-attention needs longer to find breakthrough
    p.add_argument("--early-stopping-patience", type=int, default=30)
    # Manifold mixup in 256-dim fusion token space
    p.add_argument("--mixup-alpha", type=float, default=0.3)
    # Top-K checkpoint averaging
    p.add_argument("--top-k-avg", type=int, default=3)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None,
                   help="Limit total number of training/val/test steps for quick debugging. "
                        "When set, limits both training and validation batches.")
    p.add_argument("--fast_dev_run", action="store_true")
    return p.parse_args()


def main():
    pl.seed_everything(seed=0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = args.fast_dev_run
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit_train_batches = args.debug_max_step if args.debug_max_step is not None else 1.0
    limit_val_batches = args.debug_max_step if args.debug_max_step is not None else 1.0
    limit_test_batches = 1.0  # Always run full test set

    # Checkpoint monitors val_f1 (save top-K for averaging)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node3-3-2-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=args.top_k_avg, save_last=True,
    )
    # Early stopping patience=30: allows breakthrough after LR reduction
    early_stop_cb = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.early_stopping_patience, verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    callbacks_list = [checkpoint_cb, early_stop_cb, lr_monitor, progress_bar]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=180)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=1.0 if (args.debug_max_step is not None or args.fast_dev_run)
                           else args.val_check_interval,
        num_sanity_val_steps=2,
        callbacks=callbacks_list,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic="warn",  # warn on non-deterministic ops (adaptive_max_pool has no deterministic impl)
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
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        attn_n_layers=args.attn_n_layers,
        attn_n_heads=args.attn_n_heads,
        attn_ff_dim=args.attn_ff_dim,
        attn_dropout=args.attn_dropout,
        lr=args.lr,
        head_lr_multiplier=args.head_lr_multiplier,
        symbol_encoder_lr_multiplier=args.symbol_encoder_lr_multiplier,
        ppi_proj_lr_multiplier=args.ppi_proj_lr_multiplier,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        plateau_min_lr=args.plateau_min_lr,
        mixup_alpha=args.mixup_alpha,
        top_k_avg=args.top_k_avg,
    )

    trainer.fit(model_module, datamodule=datamodule)

    # Post-training: apply top-K checkpoint averaging (only in full run mode)
    # In DDP mode, each GPU has a copy; averaging is done on rank 0
    use_avg = (
        not args.fast_dev_run
        and args.debug_max_step is None
        and args.top_k_avg > 1
        and trainer.is_global_zero
    )

    if use_avg:
        # Collect top-K checkpoints and average weights
        print(f"\n[TopKAvg] Applying top-{args.top_k_avg} checkpoint averaging...")
        apply_top_k_checkpoint_averaging(
            model_module=model_module,
            checkpoint_cb=checkpoint_cb,
            datamodule=datamodule,
            trainer=trainer,
            top_k=args.top_k_avg,
            output_dir=output_dir,
        )
        avg_ckpt = str(output_dir / "checkpoints" / "top_k_averaged.ckpt")
        print(f"[TopKAvg] Averaged checkpoint saved: {avg_ckpt}")
        # Update best_model_path so trainer.test(ckpt_path="best") uses the averaged checkpoint
        # We must update the checkpoint_cb's internal state, then reload the averaged
        # checkpoint into the model on ALL ranks before calling trainer.test().
        # Use the main trainer to load (DDP is already initialized, avoids NCCL conflict).
        checkpoint_cb.best_model_path = avg_ckpt
        test_ckpt_path = avg_ckpt
    elif args.fast_dev_run or args.debug_max_step is not None:
        test_ckpt_path = None
    else:
        test_ckpt_path = "best"

    test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path=test_ckpt_path)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        primary_val = float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score is not None else float("nan")
        score_path.write_text(f"{primary_val:.6f}\n")
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
