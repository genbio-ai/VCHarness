#!/usr/bin/env python3
"""
Node 3-1-3: AIDO.Cell-10M + LoRA (r=4, all 8 layers) + Symbol CNN + STRING GNN PPI
             4-Source Fusion with Enhanced Regularization
================================================================================
Building on sibling node3-1-2 (test F1=0.4577) which confirmed the 4-source fusion
architecture is sound but suffered from overfitting (train_loss dropped 51% while
val_loss rose 22%). This node applies targeted regularization improvements and
tightened early stopping to capture the performance peak before overfitting.

Key changes from sibling node3-1-2:
  1. Dropout: 0.4 → 0.5 (sibling feedback explicitly recommended 0.5)
  2. Weight decay: 0.03 → 0.06 (sibling feedback: "0.05–0.10 range")
  3. Head hidden dim: 384 → 320 (capacity reduction to reduce overfitting)
  4. EarlyStopping min_delta: None → 0.002 (prevent trivial oscillations extending training)
  5. EarlyStopping patience: 18 → 15 (capture peak before overfitting sets in)
  6. Class weights: [5.0, 1.0, 10.0] → [6.0, 1.0, 12.0]
     (node2-2-3-1-1 achieved tree-best 0.4625 with these weights)
  7. Max epochs: 150 → 100 (sufficient; convergence typically by epoch 40)

All other architectural components preserved from sibling:
  - AIDO.Cell-10M backbone with LoRA r=4 on all 8 layers (Q/K/V)
  - Dual pooling: global mean-pool + perturbed gene positional embedding → [B, 512]
  - Character-level Symbol CNN (3-branch Conv1d, 64-dim)
  - Frozen STRING GNN PPI embeddings (256-dim, 1-layer projection with LayerNorm)
  - concat+MLP head: 832 → 320 → 19920 (single-stage, proven architecture)
  - Focal Loss (γ=2.0), ReduceLROnPlateau on val_f1 (patience=8, factor=0.5)
  - AdamW differential LR: backbone=2e-4, head=6e-4
  - Gradient clipping (norm=1.0)

Architecture overview:
  Source 1: AIDO.Cell-10M + LoRA → dual-pool → [B, 512]
  Source 2: Symbol CNN (3-branch char-level) → [B, 64]
  Source 3: STRING GNN PPI (frozen static, projected) → [B, 256]
  Fusion: concat → [B, 832]
  Head: LayerNorm(832) → Linear(832→320) → GELU → Dropout(0.5)
        → LayerNorm(320) → Linear(320→3×6640)
  Output: [B, 3, 6640]
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
from lightning.pytorch.callbacks import (
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
AIDO_CELL_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_MODEL_DIR = Path("/home/Models/STRING_GNN")

N_GENES_AIDO = 19_264    # AIDO.Cell vocabulary size
N_GENES_OUT = 6_640      # output genes (DEG prediction targets)
N_CLASSES = 3            # {0:down-regulated, 1:unchanged, 2:up-regulated}
AIDO_N_LAYERS = 8        # AIDO.Cell-10M transformer layers

SENTINEL_EXPR = 1.0      # expression for all non-perturbed genes
KNOCKOUT_EXPR = 0.0      # expression for the knocked-out gene

# Class weights: [6.0, 1.0, 12.0] — node2-2-3-1-1 achieved tree-best 0.4625 with these
# down ~3.56%, unchanged ~94.82%, up ~1.63%
CLASS_WEIGHTS = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)

# Symbol character encoding
CHAR_VOCAB = sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._/")) + ["<UNK>", "<PAD>"]
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}
PAD_IDX = CHAR_TO_IDX["<PAD>"]
UNK_IDX = CHAR_TO_IDX["<UNK>"]
SYMBOL_MAX_LEN = 16


def symbol_to_indices(symbol: str) -> List[int]:
    """Convert gene symbol string to padded character index list."""
    chars = list(symbol.upper())[:SYMBOL_MAX_LEN]
    idxs = [CHAR_TO_IDX.get(c, UNK_IDX) for c in chars]
    idxs += [PAD_IDX] * (SYMBOL_MAX_LEN - len(idxs))
    return idxs


def build_frozen_string_embeddings(model_dir: Path, device: str = "cpu"):
    """
    Pre-compute frozen STRING GNN embeddings once on CPU.
    Returns:
      embs: [18870, 256] float32 tensor
      ensg_to_gnn: dict mapping ENSG id → row index in embs
    """
    gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    gnn_model.eval()
    gnn_model = gnn_model.to(device)

    graph_data = torch.load(str(model_dir / "graph_data.pt"), map_location=device)
    edge_index = graph_data["edge_index"].long().to(device)
    edge_w = graph_data.get("edge_weight")
    if edge_w is not None:
        edge_w = edge_w.float().to(device)
    else:
        edge_w = torch.ones(edge_index.shape[1], device=device)

    with torch.no_grad():
        out = gnn_model(
            edge_index=edge_index,
            edge_weight=edge_w,
        ).last_hidden_state.float().cpu()  # [18870, 256]

    node_names = json.loads((model_dir / "node_names.json").read_text())
    ensg_to_gnn = {name: i for i, name in enumerate(node_names)}

    del gnn_model
    if device != "cpu":
        torch.cuda.empty_cache()

    return out, ensg_to_gnn


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Multi-class focal loss to down-weight easy examples (dominant class 1=unchanged)."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [N, C], targets: [N]
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(
            logits, targets,
            weight=w,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Pre-builds AIDO.Cell expression profiles, symbol indices,
    STRING GNN indices, and labels for efficient batch loading.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_to_pos: Dict[str, int],   # ENSG_base → AIDO.Cell position [0, 19264)
        ensg_to_gnn: Dict[str, int],   # ENSG_base → STRING GNN node index
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_to_pos = gene_to_pos
        self.ensg_to_gnn = ensg_to_gnn
        self.is_test = is_test

        # Pre-build expression profile tensors: [N, N_GENES_AIDO] float32
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
            "expr": self.expr_inputs[idx],              # [N_GENES_AIDO] float32
            "gene_pos": gene_pos,                        # int (-1 if not in AIDO vocab)
            "symbol_ids": self.symbol_ids[idx],          # [SYMBOL_MAX_LEN] int64
            "gnn_idx": self.gnn_indices[idx].item(),     # int (-1 if not in STRING)
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "expr": torch.stack([b["expr"] for b in batch]),              # [B, N_GENES_AIDO]
        "gene_pos": torch.tensor([b["gene_pos"] for b in batch], dtype=torch.long),  # [B]
        "symbol_ids": torch.stack([b["symbol_ids"] for b in batch]),  # [B, SYMBOL_MAX_LEN]
        "gnn_idx": torch.tensor([b["gnn_idx"] for b in batch], dtype=torch.long),  # [B]
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])  # [B, N_GENES_OUT]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        micro_batch_size: int = 8,
        num_workers: int = 0,
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
        # Rank-safe tokenizer loading: rank 0 downloads first
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_MODEL_DIR, trust_remote_code=True)

        # Build ENSG → AIDO.Cell position mapping (probe each gene)
        if not self.gene_to_pos:
            all_ids: List[str] = []
            for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
                path = self.data_dir / split_file
                if path.exists():
                    df = pd.read_csv(path, sep="\t")
                    all_ids.extend(df["pert_id"].tolist())
            unique_ids = list({pid.split(".")[0] for pid in all_ids})
            self.gene_to_pos = self._build_gene_to_pos(tokenizer, unique_ids)
            print(
                f"[DEGDataModule] AIDO.Cell gene vocab coverage: "
                f"{len(self.gene_to_pos)}/{len(unique_ids)} genes"
            )

        # Build frozen STRING GNN embeddings once (CPU)
        if self.frozen_gnn_embs is None:
            print("[DEGDataModule] Pre-computing STRING GNN PPI embeddings (CPU)...")
            self.frozen_gnn_embs, self.ensg_to_gnn = build_frozen_string_embeddings(
                STRING_GNN_MODEL_DIR, device="cpu"
            )
            print(
                f"[DEGDataModule] GNN embs shape: {self.frozen_gnn_embs.shape}, "
                f"vocab size: {len(self.ensg_to_gnn)}"
            )

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self.gene_to_pos, self.ensg_to_gnn
            )
            self.val_ds = PerturbationDataset(
                val_df, self.gene_to_pos, self.ensg_to_gnn
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self.gene_to_pos, self.ensg_to_gnn, is_test=True
            )
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    @staticmethod
    def _build_gene_to_pos(tokenizer, gene_ids: List[str]) -> Dict[str, int]:
        """Map each ENSG gene_id to its position index in AIDO.Cell vocab via probing."""
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
            collate_fn=collate_fn, persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=collate_fn, persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=collate_fn, persistent_workers=self.num_workers > 0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Symbol Encoder (3-branch character-level CNN)
# ──────────────────────────────────────────────────────────────────────────────
class SymbolEncoder(nn.Module):
    """
    3-branch character-level CNN for gene symbol strings.
    Kernels [3, 5, 7] → max-pool → concat → Linear(192→out_dim).
    """

    def __init__(self, out_dim: int = 64, embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.char_emb = nn.Embedding(len(CHAR_VOCAB), embed_dim, padding_idx=PAD_IDX)
        nn.init.normal_(self.char_emb.weight, std=0.02)
        self.char_emb.weight.data[PAD_IDX].zero_()

        branch_dim = out_dim
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, branch_dim, kernel_size=k, padding=k // 2),
                nn.GELU(),
            )
            for k in [3, 5, 7]
        ])

        self.fusion = nn.Sequential(
            nn.Linear(3 * branch_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, symbol_ids: torch.Tensor) -> torch.Tensor:
        # symbol_ids: [B, SYMBOL_MAX_LEN] int64
        x = self.char_emb(symbol_ids)        # [B, L, embed_dim]
        x = x.transpose(1, 2)                # [B, embed_dim, L]

        branch_outs = []
        for branch in self.branches:
            out = branch(x)                  # [B, branch_dim, L]
            out = out.max(dim=-1).values     # [B, branch_dim] global max-pool
            branch_outs.append(out)

        concat = torch.cat(branch_outs, dim=-1)  # [B, 3*branch_dim]
        return self.fusion(concat)               # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# DEG Model (4-source fusion with enhanced regularization)
# ──────────────────────────────────────────────────────────────────────────────
class AIDOCellPPIDEGModel(nn.Module):
    """
    4-source feature fusion DEG predictor with enhanced regularization:
      (a) AIDO.Cell-10M global mean-pool → [B, 256]
      (b) AIDO.Cell-10M perturbed-gene positional embedding → [B, 256]
      (c) Character-level Symbol CNN → [B, 64]
      (d) Frozen STRING GNN PPI embedding → [B, 256]
      Fusion: concat([a,b,c,d]) → [B, 832]
      Head: LayerNorm(832) → Linear(832→320) → GELU → Dropout(0.5)
            → LayerNorm(320) → Linear(320→19920)
      Output: [B, 3, 6640]

    Key improvements over sibling node3-1-2:
      - head_hidden: 384 → 320 (capacity reduction)
      - head_dropout: 0.4 → 0.5 (stronger regularization)
    """

    HIDDEN_DIM = 256          # AIDO.Cell-10M hidden size
    SYMBOL_DIM = 64           # symbol CNN output dim
    PPI_DIM = 256             # STRING GNN embedding dim
    FUSED_DIM = 256 * 2 + 64 + 256  # 832

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        head_hidden: int = 320,
        head_dropout: float = 0.5,
        frozen_gnn_embs: Optional[torch.Tensor] = None,  # [18870, 256]
    ):
        super().__init__()

        # ── AIDO.Cell-10M with LoRA on ALL 8 layers ──
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
            layers_to_transform=list(range(AIDO_N_LAYERS)),  # all 8 layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Cast trainable LoRA params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Symbol CNN ──
        self.symbol_encoder = SymbolEncoder(out_dim=self.SYMBOL_DIM, embed_dim=32)

        # ── Frozen STRING GNN embedding table ──
        if frozen_gnn_embs is not None:
            self.register_buffer("gnn_emb_table", frozen_gnn_embs.float())
        else:
            self.register_buffer(
                "gnn_emb_table",
                torch.zeros(18870, self.PPI_DIM, dtype=torch.float32)
            )

        # Learnable fallback for genes not in STRING GNN
        self.gnn_fallback = nn.Parameter(torch.zeros(self.PPI_DIM))
        nn.init.trunc_normal_(self.gnn_fallback, std=0.02)

        # Trainable PPI projection (1-layer + LayerNorm, matching sibling)
        self.ppi_proj = nn.Sequential(
            nn.Linear(self.PPI_DIM, self.PPI_DIM),
            nn.GELU(),
            nn.LayerNorm(self.PPI_DIM),
        )
        nn.init.xavier_uniform_(self.ppi_proj[0].weight)
        nn.init.zeros_(self.ppi_proj[0].bias)

        # ── Prediction head (enhanced regularization) ──
        # head_hidden=320 (vs 384 in sibling), head_dropout=0.5 (vs 0.4)
        # Added LayerNorm after hidden layer for additional regularization
        self.head = nn.Sequential(
            nn.LayerNorm(self.FUSED_DIM),
            nn.Linear(self.FUSED_DIM, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        expr: torch.Tensor,          # [B, N_GENES_AIDO] float32
        gene_pos: torch.Tensor,      # [B] int64 (-1 if not in AIDO vocab)
        symbol_ids: torch.Tensor,    # [B, SYMBOL_MAX_LEN] int64
        gnn_idx: torch.Tensor,       # [B] int64 (-1 if not in STRING)
    ) -> torch.Tensor:
        B = expr.shape[0]
        device = expr.device

        # ── AIDO.Cell backbone ──
        attn_mask = torch.ones_like(expr, dtype=torch.long)
        out = self.backbone(input_ids=expr, attention_mask=attn_mask)
        lhs = out.last_hidden_state  # [B, N_GENES_AIDO+2, 256]

        # (a) Global mean-pool over gene positions (exclude 2 summary tokens)
        gene_emb = lhs[:, :N_GENES_AIDO, :]       # [B, 19264, 256]
        global_emb = gene_emb.mean(dim=1)           # [B, 256]

        # (b) Perturbed-gene positional embedding (fallback to global_emb if not found)
        pert_emb = torch.zeros(B, self.HIDDEN_DIM, device=device, dtype=lhs.dtype)
        valid_aido = gene_pos >= 0
        if valid_aido.any():
            valid_pos = gene_pos[valid_aido]
            pert_emb[valid_aido] = lhs[valid_aido, valid_pos, :]
        pert_emb[~valid_aido] = global_emb[~valid_aido]

        # Convert to float32 for head computation
        backbone_feat = torch.cat([global_emb, pert_emb], dim=-1).float()  # [B, 512]

        # (c) Symbol CNN
        sym_feat = self.symbol_encoder(symbol_ids)  # [B, 64] float32

        # (d) Frozen STRING GNN PPI lookup + trainable projection
        ppi_emb = self.gnn_fallback.unsqueeze(0).expand(B, -1).clone()  # [B, 256]
        valid_gnn = gnn_idx >= 0
        if valid_gnn.any():
            with torch.no_grad():
                ppi_raw = self.gnn_emb_table[gnn_idx[valid_gnn]]  # [k, 256]
            ppi_emb[valid_gnn] = ppi_raw
        ppi_feat = self.ppi_proj(ppi_emb.to(device))  # [B, 256]

        # 4-source fusion and prediction
        combined = torch.cat([backbone_feat, sym_feat, ppi_feat], dim=-1)  # [B, 832]
        logits = self.head(combined)              # [B, 3 * N_GENES_OUT]
        return logits.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro-averaged F1 (matches calc_metric.py logic)."""
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
        head_hidden: int = 320,
        head_dropout: float = 0.5,
        backbone_lr: float = 2e-4,
        head_lr: float = 6e-4,
        weight_decay: float = 0.06,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 100,
        plateau_patience: int = 8,
        plateau_factor: float = 0.5,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCellPPIDEGModel] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            # Get frozen GNN embeddings from datamodule (already computed in setup)
            frozen_gnn_embs = None
            if hasattr(self, "trainer") and self.trainer is not None:
                dm = getattr(self.trainer, "datamodule", None)
                if dm is not None and dm.frozen_gnn_embs is not None:
                    frozen_gnn_embs = dm.frozen_gnn_embs

            self.model = AIDOCellPPIDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                head_hidden=self.hparams.head_hidden,
                head_dropout=self.hparams.head_dropout,
                frozen_gnn_embs=frozen_gnn_embs,
            )

            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print(
                f"[setup] Model: {trainable:,}/{total:,} params trainable "
                f"({100 * trainable / total:.2f}%)"
            )

            self.loss_fn = FocalLoss(
                gamma=self.hparams.focal_gamma,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        if stage in ("test", None):
            self._populate_test_metadata()

    def _populate_test_metadata(self) -> None:
        if hasattr(self, "trainer") and self.trainer is not None:
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "test_pert_ids") and dm.test_pert_ids:
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            expr=batch["expr"],
            gene_pos=batch["gene_pos"],
            symbol_ids=batch["symbol_ids"],
            gnn_idx=batch["gnn_idx"],
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, 3, 6640], labels: [B, 6640]
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                               # [B*6640]
        return self.loss_fn(logits_flat, labels_flat)

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
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, 6640]
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        lp = (torch.cat(self._val_preds, 0)
              if self._val_preds else torch.zeros(0, N_CLASSES, N_GENES_OUT))
        ll = (torch.cat(self._val_labels, 0)
              if self._val_labels else torch.zeros(0, N_GENES_OUT, dtype=torch.long))
        li = (torch.cat(self._val_indices, 0)
              if self._val_indices else torch.zeros(0, dtype=torch.long))

        ap = self.all_gather(lp)
        al = self.all_gather(ll)
        ai = self.all_gather(li)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
        labels = al.view(-1, N_GENES_OUT).cpu().numpy()
        idxs = ai.view(-1).cpu().numpy()
        _, uniq = np.unique(idxs, return_index=True)
        f1_val = compute_deg_f1(preds[uniq], labels[uniq])
        self.log("val_f1", f1_val, prog_bar=True, sync_dist=True)

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

        if self.trainer.is_global_zero:
            preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
            idxs = ai.view(-1).cpu().numpy()
            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]
            idxs = idxs[uniq]
            order = np.argsort(idxs)
            preds = preds[order]
            idxs = idxs[order]

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for r, i in enumerate(idxs):
                rows.append({
                    "idx": self._test_pert_ids[i],
                    "input": self._test_symbols[i],
                    "prediction": json.dumps(preds[r].tolist()),
                })
            pred_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {pred_path}")

    def configure_optimizers(self):
        # Differential learning rates:
        #   backbone LoRA params: backbone_lr=2e-4
        #   head + symbol encoder + ppi_proj: head_lr=6e-4
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters()
            if p.requires_grad
        ]
        other_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not n.startswith("backbone.")
        ]

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
                {"params": other_params, "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        # ReduceLROnPlateau on val_f1 (mode=max)
        # patience=8: node3-2 used this — scheduler never fired (correct behavior)
        # val_f1 monitoring: focal loss makes val_loss inversely correlated with val_f1
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=self.hparams.plateau_factor,
            patience=self.hparams.plateau_patience,
            verbose=True,
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
        """Save only trainable parameters and persistent buffers."""
        full_state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_state_dict:
                    trainable_state_dict[key] = full_state_dict[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_state_dict:
                trainable_state_dict[key] = full_state_dict[key]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable:,}/{total:,} params "
            f"({100 * trainable / total:.2f}%), plus {buffers:,} buffer values"
        )
        return trainable_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable parameters from partial checkpoint."""
        full_state_keys = set(super().state_dict().keys())
        trainable_keys = {
            name for name, p in self.named_parameters() if p.requires_grad
        }
        buffer_keys = {
            name for name, _ in self.named_buffers() if name in full_state_keys
        }
        expected_keys = trainable_keys | buffer_keys
        missing = [k for k in expected_keys if k not in state_dict]
        unexpected = [k for k in state_dict if k not in expected_keys]
        if missing:
            self.print(f"Warning: Missing checkpoint keys: {missing[:5]}...")
        if unexpected:
            self.print(f"Warning: Unexpected keys: {unexpected[:5]}...")
        loaded_t = len([k for k in state_dict if k in trainable_keys])
        loaded_b = len([k for k in state_dict if k in buffer_keys])
        self.print(
            f"Loading checkpoint: {loaded_t} trainable params + {loaded_b} buffers"
        )
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 3-1-3: 4-Source Fusion DEG predictor (enhanced regularization)"
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "data"),
    )
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--backbone-lr", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.06)
    p.add_argument("--head-hidden", type=int, default=320)
    p.add_argument("--head-dropout", type=float, default=0.5)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--plateau-patience", type=int, default=8)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.002)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit = args.debug_max_step if args.debug_max_step is not None else 1.0

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node3-1-3-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(seconds=300),
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit,
        limit_val_batches=limit,
        limit_test_batches=limit,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
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
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    trainer.fit(model_module, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(
            model_module, datamodule=datamodule, ckpt_path="best"
        )

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        primary_val = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else float("nan")
        )
        score_path.write_text(str(primary_val))
        print(f"Test score saved → {score_path} (val_f1={primary_val:.6f})")


if __name__ == "__main__":
    main()
