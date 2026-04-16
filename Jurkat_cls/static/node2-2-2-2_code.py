#!/usr/bin/env python3
"""
Node 2-2-2-2: AIDO.Cell-10M + LoRA (QKV + FFN last 4 layers) + Symbol CNN
           + STRING GNN PPI + 4-token TransformerEncoder Cross-Attention Fusion
           + Manifold Mixup
==============================================================================
This node combines the parent node2-2-2's FFN LoRA extension (QKV+FFN on last 4
of 8 layers) with the highest-performing architectural pattern in the entire MCTS
tree: 4-token TransformerEncoder cross-attention fusion with manifold mixup.

Key differences from sibling node2-2-2-1 (concat+MLP + val_loss scheduler):
  1. Cross-attention fusion (3-layer TransformerEncoder, nhead=8, dim_ff=384)
     vs sibling's simple concat+MLP — cross-attention models inter-source
     feature interactions that concat+MLP cannot capture.
  2. Manifold mixup (alpha=0.3) in the fused token space — proven crucial for
     cross-attention DEG prediction in node3-1-3-1-1 (0.4739) and
     node3-1-1-1-1-1-2 (0.5049).
  3. weight_decay=0.10 — proven optimal for cross-attention models (all top
     cross-attention nodes use wd=0.10).
  4. ReduceLROnPlateau monitoring val_f1 (vs sibling's val_loss) — val_f1 is
     the direct optimization target and avoids the focal loss confounding issue.
  5. backbone_lr=3e-4 — corrected from parent's too-low 2e-4, as recommended
     by node2-2-2 feedback for FFN LoRA.

Evidence base:
  - node3-1-1-1-1-1-2: 4-token XAttn + manifold mixup → test F1=0.5049 (tree-best)
  - node3-1-3-1-1-1-1: 4-token XAttn + wd=0.10 + mixup=0.3 → test F1=0.4768
  - node3-1-3-1-1: 3-layer XAttn + wd=0.10 + mixup=0.3 → test F1=0.4739
  - node3-1-3-1: 2-layer XAttn + mixup=0.2 → test F1=0.4731 (beats all concat+MLP)
  - All concat+MLP nodes plateau at ~0.462–0.465 regardless of architecture changes
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
N_GENES_OUT = 6_640      # output genes
N_CLASSES = 3
SENTINEL_EXPR = 1.0      # baseline expression (non-perturbed genes)
KNOCKOUT_EXPR = 0.0      # expression for knocked-out gene
AIDO_HIDDEN = 256        # AIDO.Cell-10M hidden dimension
AIDO_N_LAYERS = 8        # AIDO.Cell-10M transformer layers
PPI_HIDDEN = 256         # STRING GNN output dimension

# Class weights for severely imbalanced dataset
# Train distribution: class 0 (down) ~3.4%, class 1 (unchanged) ~95.5%, class 2 (up) ~1.1%
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

    Architecture:
      1. Character embedding: [B, L] → [B, L, embed_dim]
      2. 3 parallel Conv1d layers with adaptive max-pool
      3. Projection: 96 → out_dim with LayerNorm
    """

    def __init__(self, out_dim: int = 64, embed_dim: int = 32, cnn_dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Embedding(SYMBOL_VOCAB_SIZE, embed_dim, padding_idx=SYMBOL_PAD_IDX)
        self.conv2 = nn.Conv1d(embed_dim, 32, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, 32, kernel_size=4, padding=2)
        self.dropout = nn.Dropout(cnn_dropout)
        self.proj = nn.Sequential(
            nn.Linear(96, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for conv in [self.conv2, self.conv3, self.conv4]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, symbol_ids: torch.Tensor) -> torch.Tensor:
        """symbol_ids: [B, SYMBOL_MAX_LEN] → [B, out_dim]"""
        x = self.embed(symbol_ids)          # [B, L, embed_dim]
        x = x.transpose(1, 2)              # [B, embed_dim, L]
        f2 = F.relu(self.conv2(x))
        f3 = F.relu(self.conv3(x))
        f4 = F.relu(self.conv4(x))
        f2 = F.adaptive_max_pool1d(f2, 1).squeeze(-1)  # [B, 32]
        f3 = F.adaptive_max_pool1d(f3, 1).squeeze(-1)  # [B, 32]
        f4 = F.adaptive_max_pool1d(f4, 1).squeeze(-1)  # [B, 32]
        feat = torch.cat([f2, f3, f4], dim=-1)          # [B, 96]
        feat = self.dropout(feat)
        return self.proj(feat)                           # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Dataset for perturbation DEG prediction."""

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.is_test = is_test

        # Pre-build ENSG base ids for STRING lookup
        self.ensg_bases: List[str] = [p.split(".")[0] for p in self.pert_ids]

        # Pre-build expression tensors
        self.expr_inputs = self._build_expr_tensors()

        # Pre-build symbol character tensors
        self.symbol_ids = self._build_symbol_tensors()

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1} → {0,1,2}
        else:
            self.labels = None

    def _build_expr_tensors(self) -> torch.Tensor:
        """Pre-compute [N, 19264] float32 expression inputs."""
        N = len(self.pert_ids)
        expr = torch.full((N, N_GENES_AIDO), SENTINEL_EXPR, dtype=torch.float32)
        for i, ensg_base in enumerate(self.ensg_bases):
            pos = self.gene_to_pos.get(ensg_base)
            if pos is not None:
                expr[i, pos] = KNOCKOUT_EXPR
        return expr

    def _build_symbol_tensors(self) -> torch.Tensor:
        """Pre-compute [N, SYMBOL_MAX_LEN] int64 character index tensors."""
        N = len(self.symbols)
        sym_ids = torch.zeros((N, SYMBOL_MAX_LEN), dtype=torch.long)
        for i, symbol in enumerate(self.symbols):
            indices = symbol_to_indices(symbol)
            sym_ids[i] = torch.tensor(indices, dtype=torch.long)
        return sym_ids

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ensg_base = self.ensg_bases[idx]
        gene_pos = self.gene_to_pos.get(ensg_base, -1)
        item = {
            "idx": idx,
            "expr": self.expr_inputs[idx],
            "gene_pos": gene_pos,
            "symbol_ids": self.symbol_ids[idx],
            "ensg_base": ensg_base,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "expr": torch.stack([b["expr"] for b in batch]),
        "gene_pos": torch.tensor([b["gene_pos"] for b in batch], dtype=torch.long),
        "symbol_ids": torch.stack([b["symbol_ids"] for b in batch]),
        "ensg_bases": [b["ensg_base"] for b in batch],
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
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []
        self.test_labels: Optional[np.ndarray] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Rank-0 downloads tokenizer first, then barrier
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # Build ENSG→position mapping
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self.gene_to_pos)
            self.val_ds = PerturbationDataset(val_df, self.gene_to_pos)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(test_df, self.gene_to_pos, is_test=True)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()
            raw_labels = [json.loads(x) for x in test_df["label"].tolist()]
            self.test_labels = np.array(raw_labels, dtype=np.int8) + 1

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
class AIDOCellXAttnDEGModel(nn.Module):
    """
    4-source cross-attention DEG predictor:
      (a) AIDO.Cell-10M + LoRA QKV+FFN (last 4 of 8 layers)
          → global_emb [B,256] + pert_emb [B,256]
      (b) Gene symbol CNN → sym_feat [B,64]
      (c) Frozen STRING GNN PPI → ppi_feat [B,256]

    Fusion:
      Project each source to 256-dim → 4 tokens [B,4,256]
      3-layer TransformerEncoder self-attention (nhead=8, dim_ff=384)
      Mean-pool over tokens → [B,256]
      Head: LayerNorm → Linear(256→256) → GELU → Dropout → Linear(256→19920)

    Manifold Mixup in the fused token space during training.
    """

    HIDDEN_DIM = 256        # AIDO.Cell-10M hidden size
    SYMBOL_DIM = 64         # gene symbol embedding dimension
    FUSION_DIM = 256        # cross-attention token dimension

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_layers_start: int = 4,   # last 4 of 8 layers (indices 4-7)
        head_hidden: int = 256,
        head_dropout: float = 0.4,
        cnn_dropout: float = 0.2,
        attn_n_layers: int = 3,
        attn_nhead: int = 8,
        attn_dim_ff: int = 384,
        attn_dropout: float = 0.1,
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA on QKV + FFN (last 4 layers) ──────
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.3,
            target_modules=["query", "key", "value", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=list(range(lora_layers_start, AIDO_N_LAYERS)),
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Cast trainable LoRA params to float32 for stability
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Gene symbol CNN encoder ─────────────────────────────────────────────
        self.symbol_encoder = SymbolEncoder(
            out_dim=self.SYMBOL_DIM, embed_dim=32, cnn_dropout=cnn_dropout
        )

        # ── STRING GNN PPI frozen embedding ─────────────────────────────────────
        # Will be loaded in setup() — stored as a buffer
        self.register_buffer("ppi_table", torch.zeros(1, PPI_HIDDEN))
        self.ppi_fallback = nn.Parameter(torch.zeros(PPI_HIDDEN))
        self.ppi_ensg_to_idx: Dict[str, int] = {}

        # PPI projection: 256 → 256
        self.ppi_proj = nn.Sequential(
            nn.Linear(PPI_HIDDEN, self.FUSION_DIM),
            nn.GELU(),
            nn.LayerNorm(self.FUSION_DIM),
        )

        # ── Source projections to FUSION_DIM ──────────────────────────────────
        # global_emb: 256 → 256 (identity but with norm + optional transform)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.Linear(self.HIDDEN_DIM, self.FUSION_DIM),
        )
        # pert_emb: 256 → 256
        self.pert_proj = nn.Sequential(
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.Linear(self.HIDDEN_DIM, self.FUSION_DIM),
        )
        # sym_feat: 64 → 256
        self.sym_proj = nn.Sequential(
            nn.LayerNorm(self.SYMBOL_DIM),
            nn.Linear(self.SYMBOL_DIM, self.FUSION_DIM),
        )

        # ── 4-token TransformerEncoder cross-attention ─────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.FUSION_DIM,
            nhead=attn_nhead,
            dim_feedforward=attn_dim_ff,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=attn_n_layers)

        # ── Prediction head ────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(self.FUSION_DIM),
            nn.Linear(self.FUSION_DIM, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        # Conservative initialization
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def build_ppi_table(self):
        """
        Load STRING GNN model, run one forward pass, store the resulting
        [18870, 256] embedding table as a frozen buffer. Called once in setup().
        """
        import json as _json
        from pathlib import Path as _Path
        from transformers import AutoModel as _AutoModel

        model_dir = _Path(STRING_GNN_MODEL_DIR)
        device = next(self.backbone.parameters()).device

        gnn = _AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device)
        gnn.eval()

        graph = torch.load(model_dir / "graph_data.pt", map_location=device)
        node_names = _json.loads((model_dir / "node_names.json").read_text())

        edge_index = graph["edge_index"].to(device)
        edge_weight = graph["edge_weight"]
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        with torch.no_grad():
            outputs = gnn(edge_index=edge_index, edge_weight=edge_weight)
            emb = outputs.last_hidden_state  # [18870, 256]

        # Build ENSG → index mapping (strip version suffixes)
        ensg_to_idx = {}
        for i, name in enumerate(node_names):
            base = name.split(".")[0]
            ensg_to_idx[base] = i

        # Store as buffer (non-trainable)
        self.ppi_table = emb.float().detach()
        self.ppi_ensg_to_idx = ensg_to_idx

        # Clean up GNN from GPU to free memory
        del gnn, graph
        torch.cuda.empty_cache()

    def lookup_ppi_embeddings(self, ensg_bases: List[str]) -> torch.Tensor:
        """
        Look up PPI embeddings for a batch of ENSG base IDs.
        Falls back to learnable ppi_fallback for genes not in STRING.
        Returns [B, PPI_HIDDEN] float32.
        """
        device = self.ppi_table.device
        B = len(ensg_bases)
        result = torch.zeros(B, PPI_HIDDEN, dtype=torch.float32, device=device)
        for i, ensg in enumerate(ensg_bases):
            idx = self.ppi_ensg_to_idx.get(ensg)
            if idx is not None:
                result[i] = self.ppi_table[idx]
            else:
                result[i] = self.ppi_fallback.float()
        return result

    def forward(
        self,
        expr: torch.Tensor,             # [B, 19264]
        gene_pos: torch.Tensor,         # [B]
        symbol_ids: torch.Tensor,       # [B, SYMBOL_MAX_LEN]
        ensg_bases: List[str],
        mixup_alpha: float = 0.0,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns (logits [B, 3, 6640], mixed_labels or None).
        If mixup_alpha > 0 and labels is not None, performs manifold mixup
        in the fused token space.
        """
        # ── AIDO.Cell backbone ─────────────────────────────────────────────────
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, 19266, 256]

        gene_emb = lhs[:, :N_GENES_AIDO, :]      # [B, 19264, 256]
        global_emb = gene_emb.mean(dim=1)         # [B, 256]

        B = expr.shape[0]
        pert_emb = torch.zeros(B, self.HIDDEN_DIM, device=lhs.device, dtype=lhs.dtype)
        valid_mask = gene_pos >= 0
        if valid_mask.any():
            valid_pos = gene_pos[valid_mask]
            pert_emb[valid_mask] = lhs[valid_mask, valid_pos, :]
        pert_emb[~valid_mask] = global_emb[~valid_mask]

        # Convert to float32
        global_emb = global_emb.float()
        pert_emb = pert_emb.float()

        # ── Symbol encoder ─────────────────────────────────────────────────────
        sym_feat = self.symbol_encoder(symbol_ids)   # [B, 64]

        # ── PPI embeddings ─────────────────────────────────────────────────────
        ppi_raw = self.lookup_ppi_embeddings(ensg_bases)   # [B, 256]
        ppi_feat = self.ppi_proj(ppi_raw)                  # [B, 256]

        # ── Project each source to fusion dim ──────────────────────────────────
        t_global = self.global_proj(global_emb)    # [B, 256]
        t_pert = self.pert_proj(pert_emb)          # [B, 256]
        t_sym = self.sym_proj(sym_feat)            # [B, 256]
        t_ppi = ppi_feat                           # [B, 256]

        # Stack into 4 tokens: [B, 4, 256]
        tokens = torch.stack([t_global, t_pert, t_sym, t_ppi], dim=1)

        # ── Manifold Mixup in token space ──────────────────────────────────────
        mixed_labels = None
        if mixup_alpha > 0.0 and labels is not None and self.training:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx_perm = torch.randperm(B, device=tokens.device)
            tokens = lam * tokens + (1.0 - lam) * tokens[idx_perm]
            # One-hot mix the labels
            labels_a = labels  # [B, N_GENES_OUT] in {0,1,2}
            labels_b = labels[idx_perm]
            mixed_labels = (lam, labels_a, 1.0 - lam, labels_b)

        # ── TransformerEncoder cross-attention ─────────────────────────────────
        fused = self.transformer(tokens)          # [B, 4, 256]
        pooled = fused.mean(dim=1)                # [B, 256]

        # ── Head ───────────────────────────────────────────────────────────────
        logits = self.head(pooled)                # [B, 3*6640]
        return logits.view(B, N_CLASSES, N_GENES_OUT), mixed_labels


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro F1, averaged over all genes.
    y_pred: [n_samples, 3, n_genes]
    y_true_remapped: [n_samples, n_genes] in {0,1,2}
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
        lora_layers_start: int = 4,
        head_hidden: int = 256,
        head_dropout: float = 0.4,
        cnn_dropout: float = 0.2,
        attn_n_layers: int = 3,
        attn_nhead: int = 8,
        attn_dim_ff: int = 384,
        attn_dropout: float = 0.1,
        backbone_lr: float = 3e-4,
        symbol_lr_multiplier: float = 2.0,
        head_lr_multiplier: float = 3.0,
        weight_decay: float = 0.10,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        mixup_alpha: float = 0.3,
        max_epochs: int = 100,
        plateau_patience: int = 8,
        plateau_factor: float = 0.5,
        plateau_min_lr: float = 1e-7,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCellXAttnDEGModel] = None
        self.criterion: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._test_labels_np: Optional[np.ndarray] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = AIDOCellXAttnDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_layers_start=self.hparams.lora_layers_start,
                head_hidden=self.hparams.head_hidden,
                head_dropout=self.hparams.head_dropout,
                cnn_dropout=self.hparams.cnn_dropout,
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
            # Build STRING GNN PPI table (once per setup)
            self.model.build_ppi_table()

        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols
            self._test_labels_np = self.trainer.datamodule.test_labels

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Any]:
        return self.model(
            batch["expr"], batch["gene_pos"], batch["symbol_ids"],
            batch["ensg_bases"],
            mixup_alpha=0.0,
        )

    def _compute_loss_simple(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Standard focal loss without mixup."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_flat = labels.reshape(-1)
        return self.criterion(logits_flat, labels_flat)

    def _compute_loss_mixed(
        self,
        logits: torch.Tensor,
        mixed_labels: Tuple,
    ) -> torch.Tensor:
        """Focal loss for manifold mixup output: lam*L(a) + (1-lam)*L(b)."""
        lam, labels_a, one_minus_lam, labels_b = mixed_labels
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        loss_a = self.criterion(logits_flat, labels_a.reshape(-1))
        loss_b = self.criterion(logits_flat, labels_b.reshape(-1))
        return lam * loss_a + one_minus_lam * loss_b

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits, mixed_labels = self.model(
            batch["expr"], batch["gene_pos"], batch["symbol_ids"],
            batch["ensg_bases"],
            mixup_alpha=self.hparams.mixup_alpha,
            labels=batch["label"],
        )
        if mixed_labels is not None:
            loss = self._compute_loss_mixed(logits, mixed_labels)
        else:
            loss = self._compute_loss_simple(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        # No mixup during validation
        logits, _ = self.model(
            batch["expr"], batch["gene_pos"], batch["symbol_ids"],
            batch["ensg_bases"],
            mixup_alpha=0.0,
        )
        loss = self._compute_loss_simple(logits, batch["label"])
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
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits, _ = self.model(
            batch["expr"], batch["gene_pos"], batch["symbol_ids"],
            batch["ensg_bases"],
            mixup_alpha=0.0,
        )
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

        if self.trainer.is_global_zero:
            preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
            idxs = ai.cpu().view(-1).numpy()
            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]
            idxs = idxs[uniq]
            order = np.argsort(idxs)
            preds = preds[order]
            idxs = idxs[order]

            if self._test_labels_np is not None and len(preds) == len(self._test_labels_np):
                f1 = compute_deg_f1(preds, self._test_labels_np)
                self.log("test_f1", f1, prog_bar=True, sync_dist=False)
            elif self._test_labels_np is not None and len(preds) < len(self._test_labels_np):
                self.print(
                    f"Warning: Only {len(preds)}/{len(self._test_labels_np)} test samples "
                    f"predicted. Skipping test_f1."
                )

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for rank_i, orig_i in enumerate(idxs):
                rows.append({
                    "idx": self._test_pert_ids[orig_i],
                    "input": self._test_symbols[orig_i],
                    "prediction": json.dumps(preds[rank_i].tolist()),
                })
            pd.DataFrame(rows).to_csv(output_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"Test predictions saved → {output_dir / 'test_predictions.tsv'}")

    def configure_optimizers(self):
        hp = self.hparams
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        # Group symbol encoder, PPI projection, source projections, transformer, ppi_fallback
        # as "middle" components at 2x backbone LR
        symbol_params = list(self.model.symbol_encoder.parameters())
        ppi_proj_params = list(self.model.ppi_proj.parameters())
        source_proj_params = (
            list(self.model.global_proj.parameters()) +
            list(self.model.pert_proj.parameters()) +
            list(self.model.sym_proj.parameters())
        )
        # ppi_fallback is a Parameter
        ppi_fallback_params = [self.model.ppi_fallback]
        middle_params = (
            symbol_params + ppi_proj_params + source_proj_params + ppi_fallback_params
        )

        # Transformer + head at 3x backbone LR
        transformer_params = list(self.model.transformer.parameters())
        head_params = list(self.model.head.parameters())
        head_group_params = transformer_params + head_params

        backbone_lr = hp.backbone_lr
        middle_lr = hp.backbone_lr * hp.symbol_lr_multiplier
        head_lr = hp.backbone_lr * hp.head_lr_multiplier

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": middle_params, "lr": middle_lr},
                {"params": head_group_params, "lr": head_lr},
            ],
            weight_decay=hp.weight_decay,
        )

        # ReduceLROnPlateau monitoring val_f1 (maximize)
        # This differentiates from sibling node2-2-2-1 which monitors val_loss (minimize)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            patience=hp.plateau_patience,
            factor=hp.plateau_factor,
            min_lr=hp.plateau_min_lr,
            verbose=True,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_f1",
                "reduce_on_plateau": True,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save trainable parameters and persistent buffers (including ppi_table)."""
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
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )
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

        if missing_keys:
            self.print(f"Warning: Missing checkpoint keys: {missing_keys[:5]}...")
        if unexpected_keys:
            self.print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

        loaded_trainable = len([k for k in state_dict if k in trainable_keys])
        loaded_buffers = len([k for k in state_dict if k in buffer_keys])
        self.print(
            f"Loading checkpoint: {loaded_trainable} trainable parameters and "
            f"{loaded_buffers} buffers"
        )
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 2-2-2-2: AIDO.Cell-10M + LoRA QKV+FFN + Symbol CNN + STRING GNN + Cross-Attention Fusion + Manifold Mixup"
    )
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--global_batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--backbone_lr", type=float, default=3e-4)
    p.add_argument("--symbol_lr_multiplier", type=float, default=2.0)
    p.add_argument("--head_lr_multiplier", type=float, default=3.0)
    p.add_argument("--weight_decay", type=float, default=0.10)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_layers_start", type=int, default=4)
    p.add_argument("--head_hidden", type=int, default=256)
    p.add_argument("--head_dropout", type=float, default=0.4)
    p.add_argument("--cnn_dropout", type=float, default=0.2)
    p.add_argument("--attn_n_layers", type=int, default=3)
    p.add_argument("--attn_nhead", type=int, default=8)
    p.add_argument("--attn_dim_ff", type=int, default=384)
    p.add_argument("--attn_dropout", type=float, default=0.1)
    p.add_argument("--gamma_focal", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--mixup_alpha", type=float, default=0.3)
    p.add_argument("--plateau_patience", type=int, default=8)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--plateau_min_lr", type=float, default=1e-7)
    p.add_argument("--early_stopping_patience", type=int, default=20)
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    return p.parse_args()


def main():
    pl.seed_everything(seed=0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit_train_batches = args.debug_max_step if args.debug_max_step is not None else 1.0
    limit_val_batches = args.debug_max_step if args.debug_max_step is not None else 1.0
    limit_test_batches = args.debug_max_step if args.debug_max_step is not None else 1.0

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node2-2-2-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=3, save_last=True,
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
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=1.0 if (args.debug_max_step is not None or args.fast_dev_run) else args.val_check_interval,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        # Note: deterministic=True removed — adaptive_max_pool1d (CNN) has no deterministic CUDA implementation
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
        lora_layers_start=args.lora_layers_start,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        cnn_dropout=args.cnn_dropout,
        attn_n_layers=args.attn_n_layers,
        attn_nhead=args.attn_nhead,
        attn_dim_ff=args.attn_dim_ff,
        attn_dropout=args.attn_dropout,
        backbone_lr=args.backbone_lr,
        symbol_lr_multiplier=args.symbol_lr_multiplier,
        head_lr_multiplier=args.head_lr_multiplier,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        max_epochs=args.max_epochs,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        plateau_min_lr=args.plateau_min_lr,
    )

    trainer.fit(model_module, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    score_path = Path(__file__).parent / "test_score.txt"
    if test_results:
        score_path.write_text(f"test_results: {json.dumps(test_results)}\n")


if __name__ == "__main__":
    main()
