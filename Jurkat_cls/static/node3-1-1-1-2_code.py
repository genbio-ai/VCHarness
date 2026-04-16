#!/usr/bin/env python3
"""
Node 1-2: AIDO.Cell-10M + LoRA (r=4, ALL 8 layers) + STRING GNN PPI + Symbol CNN + 4-source MLP Head
======================================================================================================
Implements the proven 4-source feature fusion architecture from node3-2 (tree-best, F1=0.462):
  1. AIDO.Cell-10M backbone — dual pooling: [pert_hidden + global_mean] → 512-dim
  2. Frozen STRING GNN PPI embeddings — 256-dim PPI topology per gene
  3. Character-level Symbol CNN — 64-dim gene symbol features (3-branch conv1d)
  4. 4-source MLP head: concat([512+256+64]) = 832-dim → 384 → 19920 → [B, 3, 6640]

Key design principles:
  1. AIDO.Cell-10M backbone (256-dim, 8 layers) with LoRA r=4 on ALL 8 layers
  2. Frozen STRING GNN PPI embeddings via node_names.json lookup
  3. 3-branch char-level CNN over 5-char gene symbol strings → 64-dim
  4. 4-source fusion MLP head (832→384→19920) matching node3-2's proven config
  5. ReduceLROnPlateau on val_f1 (patience=8) — avoiding the val_loss pitfall
  6. Focal loss (gamma=2.0), class weights [5.0, 1.0, 10.0], label_smoothing=0.05

This node breaks the ~0.40 AIDO.Cell-only ceiling by adding STRING PPI and Symbol CNN.
Every node in the MCTS tree achieving ≥0.45 F1 uses STRING GNN PPI embeddings as a 4th channel.

Differentiation from parent (node3-1-2, test F1=0.3965):
  - Adds STRING GNN PPI (new 256-dim channel)
  - Adds Symbol CNN (new 64-dim channel)
  - LoRA on all 8 layers (vs last 3 only)
  - 4-source fusion MLP head (vs single Linear projection)

Differentiation from sibling (node1-1):
  - Adds STRING GNN PPI (sibling keeps AIDO.Cell only)
  - All-8-layer LoRA (sibling keeps last-3-layers)
  - Wider 832→384→19920 MLP (sibling uses 512→128→19920 bottleneck)

Total trainable parameters: ~8.1M (LoRA ~47K + Symbol CNN ~50K + MLP head ~8.0M)
Expected target: Test F1 ≥ 0.45 (approaching tree-best node3-2 at 0.462)
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
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_MODEL_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT = 6_640
N_GENES_MODEL = 19_264
N_CLASSES = 3
HIDDEN_DIM = 256        # AIDO.Cell-10M hidden size
PPI_DIM = 256           # STRING GNN output dimension
SYM_DIM = 64            # Symbol CNN output dimension
FUSED_DIM = HIDDEN_DIM * 2 + PPI_DIM + SYM_DIM  # 512 + 256 + 64 = 832
HEAD_DIM = 384          # MLP head intermediate dimension (matches node3-2)

# Class weights: moderate correction for imbalance (proven in node3-2, node2-1 at 0.41)
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)

# Character vocabulary for symbol CNN: a-z (26 chars) + padding
SYMBOL_CHARS = "abcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX: Dict[str, int] = {c: i + 1 for i, c in enumerate(SYMBOL_CHARS)}
CHAR_VOCAB_SIZE = len(SYMBOL_CHARS) + 1  # +1 for padding (index 0)
SYMBOL_LEN = 5  # Fixed length of symbol strings


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Multi-class focal loss with class weights and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [N, C], targets: [N]
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Symbol CNN
# ─────────────────────────────────────────────────────────────────────────────
class SymbolCNN(nn.Module):
    """
    Character-level 3-branch Conv1d CNN over 5-char gene symbol strings.

    Each branch uses a different kernel size (2, 3, 4) with 64 filters.
    Outputs are max-pooled and concatenated → 192-dim → Linear(192→64) → GELU.
    Total: 64-dim symbol features per gene.

    Matches the proven architecture of node3-2 (test F1=0.4622).
    """

    def __init__(self, char_embed_dim: int = 32, num_filters: int = 64, out_dim: int = 64):
        super().__init__()
        self.char_embed = nn.Embedding(CHAR_VOCAB_SIZE, char_embed_dim, padding_idx=0)

        # 3 parallel branches with kernel sizes 2, 3, 4
        self.conv2 = nn.Conv1d(char_embed_dim, num_filters, kernel_size=2, padding=0)
        self.conv3 = nn.Conv1d(char_embed_dim, num_filters, kernel_size=3, padding=0)
        self.conv4 = nn.Conv1d(char_embed_dim, num_filters, kernel_size=4, padding=0)

        # Projection from 3 * num_filters → out_dim
        self.proj = nn.Sequential(
            nn.Linear(3 * num_filters, out_dim),
            nn.GELU(),
        )

    def forward(self, symbol_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            symbol_ids: [B, SYMBOL_LEN] long tensor of character indices
        Returns:
            [B, out_dim] float features
        """
        # [B, L, char_embed_dim] → [B, char_embed_dim, L] for Conv1d
        x = self.char_embed(symbol_ids).transpose(1, 2)

        # Each branch: conv → GELU → max pool over sequence dim
        f2 = F.gelu(self.conv2(x)).max(dim=2).values  # [B, num_filters]
        f3 = F.gelu(self.conv3(x)).max(dim=2).values
        f4 = F.gelu(self.conv4(x)).max(dim=2).values

        # Concatenate and project
        fused = torch.cat([f2, f3, f4], dim=1)  # [B, 3*num_filters]
        return self.proj(fused)                   # [B, out_dim]


def encode_symbol(symbol: str) -> List[int]:
    """Convert a gene symbol string to character indices (padded to SYMBOL_LEN)."""
    symbol = symbol.lower()[:SYMBOL_LEN]
    indices = [CHAR_TO_IDX.get(c, 0) for c in symbol]
    # Pad with zeros to SYMBOL_LEN
    while len(indices) < SYMBOL_LEN:
        indices.append(0)
    return indices


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Perturbation dataset with synthetic expression + symbol encoding.

    Input encoding:
      - expr: all 19264 genes at 1.0 (baseline), perturbed gene at 0.0
      - symbol_ids: character indices for the 5-char gene symbol
      - pert_pos: gene position in AIDO.Cell vocabulary (-1 if unknown)
      - ppi_idx: gene index in STRING node_names (-1 if not in STRING)

    Labels: {-1,0,1} remapped to {0,1,2}.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_pos_map: Dict[str, int],
        ppi_idx_map: Dict[str, int],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Keep labels as {-1, 0, 1}; remap to {0,1,2} only in loss computation
            self.labels = np.array(raw_labels, dtype=np.int8)
        else:
            self.labels = None

        # Pre-compute expression vectors
        base_expr = torch.ones(N_GENES_MODEL, dtype=torch.float32)
        self._exprs: List[torch.Tensor] = []
        self._pert_positions: List[int] = []
        self._ppi_indices: List[int] = []
        self._symbol_ids: List[List[int]] = []

        covered_aido = 0
        covered_ppi = 0

        for pid, sym in zip(self.pert_ids, self.symbols):
            base_pid = pid.split(".")[0]

            # AIDO.Cell position
            pos = gene_pos_map.get(base_pid, -1)
            self._pert_positions.append(pos)
            if pos >= 0:
                expr = base_expr.clone()
                expr[pos] = 0.0
                covered_aido += 1
            else:
                expr = base_expr.clone()
            self._exprs.append(expr)

            # STRING GNN PPI index
            ppi_idx = ppi_idx_map.get(base_pid, -1)
            self._ppi_indices.append(ppi_idx)
            if ppi_idx >= 0:
                covered_ppi += 1

            # Symbol character encoding
            self._symbol_ids.append(encode_symbol(sym))

        if not is_test:
            n = len(self.pert_ids)
            print(f"[Dataset] {n} samples: "
                  f"AIDO vocab coverage {covered_aido}/{n} ({100.0*covered_aido/n:.1f}%), "
                  f"STRING PPI coverage {covered_ppi}/{n} ({100.0*covered_ppi/n:.1f}%)")

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "expr": self._exprs[idx],                          # [19264] float32
            "pert_pos": self._pert_positions[idx],             # int
            "ppi_idx": self._ppi_indices[idx],                 # int
            "symbol_ids": torch.tensor(self._symbol_ids[idx], dtype=torch.long),  # [5]
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)  # [6640] in {-1,0,1}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
        "expr": torch.stack([b["expr"] for b in batch]),                    # [B, 19264]
        "pert_pos": torch.tensor([b["pert_pos"] for b in batch], dtype=torch.long),
        "ppi_idx": torch.tensor([b["ppi_idx"] for b in batch], dtype=torch.long),
        "symbol_ids": torch.stack([b["symbol_ids"] for b in batch]),        # [B, 5]
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])          # [B, 6640]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, micro_batch_size: int = 4, num_workers: int = 0):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []
        self._gene_pos_map: Optional[Dict[str, int]] = None
        self._ppi_idx_map: Optional[Dict[str, int]] = None

    def _build_gene_pos_map(
        self, tokenizer, all_pert_ids: List[str]
    ) -> Dict[str, int]:
        """Build mapping from ENSG gene ID to position in AIDO.Cell 19264-gene vocabulary."""
        gene_pos_map: Dict[str, int] = {}
        unique_base_ids = list(set(pid.split(".")[0] for pid in all_pert_ids))

        if hasattr(tokenizer, "gene_id_to_index"):
            gid2idx = tokenizer.gene_id_to_index
            for base_pid in unique_base_ids:
                if base_pid in gid2idx:
                    gene_pos_map[base_pid] = gid2idx[base_pid]
            print(f"[DataModule] AIDO.Cell gene_id_to_index: "
                  f"{len(gene_pos_map)}/{len(unique_base_ids)} genes found")
        else:
            print(f"[DataModule] No gene_id_to_index attribute found")
        return gene_pos_map

    def _build_ppi_idx_map(self, all_pert_ids: List[str]) -> Dict[str, int]:
        """Build mapping from ENSG gene ID to STRING GNN node index."""
        node_names_path = Path(STRING_MODEL_DIR) / "node_names.json"
        node_names: List[str] = json.loads(node_names_path.read_text())
        name_to_idx = {name: i for i, name in enumerate(node_names)}

        ppi_idx_map: Dict[str, int] = {}
        unique_base_ids = list(set(pid.split(".")[0] for pid in all_pert_ids))
        for base_pid in unique_base_ids:
            if base_pid in name_to_idx:
                ppi_idx_map[base_pid] = name_to_idx[base_pid]

        print(f"[DataModule] STRING GNN coverage: "
              f"{len(ppi_idx_map)}/{len(unique_base_ids)} genes in PPI graph")
        return ppi_idx_map

    def setup(self, stage: Optional[str] = None) -> None:
        # Initialize tokenizer: rank-0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        # Collect all pert_ids to build maps once
        if self._gene_pos_map is None or self._ppi_idx_map is None:
            all_ids: List[str] = []
            for fname in ["train.tsv", "val.tsv", "test.tsv"]:
                fpath = self.data_dir / fname
                if fpath.exists():
                    df_tmp = pd.read_csv(fpath, sep="\t")
                    if "pert_id" in df_tmp.columns:
                        all_ids.extend(df_tmp["pert_id"].tolist())
            self._gene_pos_map = self._build_gene_pos_map(tokenizer, all_ids)
            self._ppi_idx_map = self._build_ppi_idx_map(all_ids)

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self._gene_pos_map, self._ppi_idx_map)
            self.val_ds = PerturbationDataset(val_df, self._gene_pos_map, self._ppi_idx_map)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self._gene_pos_map, self._ppi_idx_map, is_test=True
            )
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    def _loader(self, ds: PerturbationDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class AIDOCell4SourceDEGModel(nn.Module):
    """
    4-source DEG prediction model:
      1. AIDO.Cell-10M + LoRA (r=4, all 8 layers) → dual pool → 512-dim
      2. Frozen STRING GNN PPI embeddings → 256-dim
      3. Symbol CNN → 64-dim
      4. 4-source MLP head: 832 → 384 → 19920 → [B, 3, 6640]

    This architecture exactly matches node3-2 (test F1=0.462, tree best).
    """

    def __init__(
        self,
        dropout: float = 0.4,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        # ── Branch 1: AIDO.Cell-10M backbone with LoRA (all 8 layers) ─────
        backbone = AutoModel.from_pretrained(
            AIDO_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16
        )
        backbone.config.use_cache = False

        # Monkey-patch enable_input_require_grads to avoid AIDO.Cell NotImplementedError
        def noop_enable_input_require_grads(self):
            pass
        backbone.enable_input_require_grads = noop_enable_input_require_grads.__get__(
            backbone, type(backbone)
        )

        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # LoRA on ALL 8 layers (0-7) — matching node3-2's proven configuration
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=list(range(8)),  # All 8 layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Cast LoRA (trainable) params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Branch 2: Frozen STRING GNN PPI embeddings ─────────────────────
        # Load frozen STRING GNN once and extract static embeddings matrix
        # We precompute all 18870 PPI node embeddings and store them as a non-trainable buffer
        ppi_model = AutoModel.from_pretrained(
            Path(STRING_MODEL_DIR), trust_remote_code=True
        )
        ppi_model.eval()

        graph = torch.load(Path(STRING_MODEL_DIR) / "graph_data.pt", weights_only=False)
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)

        with torch.no_grad():
            outputs = ppi_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            ppi_emb_matrix = outputs.last_hidden_state  # [18870, 256]

        # Register as non-trainable buffer
        self.register_buffer("ppi_emb_matrix", ppi_emb_matrix.float())  # [18870, 256]
        del ppi_model, graph

        # Learnable fallback embedding for genes not in STRING graph
        self.ppi_unk_embed = nn.Parameter(torch.zeros(PPI_DIM))
        nn.init.normal_(self.ppi_unk_embed, std=0.02)

        # ── Branch 3: Symbol CNN ────────────────────────────────────────────
        self.symbol_cnn = SymbolCNN(char_embed_dim=32, num_filters=64, out_dim=SYM_DIM)

        # ── 4-source Fusion MLP Head ─────────────────────────────────────────
        # Input: concat([aido_pool(512), ppi_emb(256), sym_emb(64)]) = 832-dim
        in_dim = FUSED_DIM       # 832
        mid_dim = HEAD_DIM       # 384
        out_dim = N_CLASSES * N_GENES_OUT  # 3 × 6640 = 19920

        self.head_norm = nn.LayerNorm(in_dim)
        self.head_dropout1 = nn.Dropout(dropout)
        self.head_fc1 = nn.Linear(in_dim, mid_dim)
        self.head_dropout2 = nn.Dropout(dropout)
        self.head_fc2 = nn.Linear(mid_dim, out_dim)

        # Initialize head weights
        nn.init.trunc_normal_(self.head_fc1.weight, std=0.02)
        nn.init.zeros_(self.head_fc1.bias)
        nn.init.trunc_normal_(self.head_fc2.weight, std=0.02)
        nn.init.zeros_(self.head_fc2.bias)

    def forward(
        self,
        expr: torch.Tensor,
        pert_pos: torch.Tensor,
        ppi_idx: torch.Tensor,
        symbol_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            expr:       [B, 19264] float32 — synthetic expression (1.0 baseline, 0.0 at pert)
            pert_pos:   [B] long — position of perturbed gene in AIDO.Cell vocab (-1 if unknown)
            ppi_idx:    [B] long — index of perturbed gene in STRING GNN (-1 if unknown)
            symbol_ids: [B, 5] long — character indices of gene symbol
        Returns:
            logits: [B, 3, 6640]
        """
        B = expr.shape[0]
        device = expr.device

        # ── Branch 1: AIDO.Cell backbone → dual pool ──────────────────────
        outputs = self.backbone(
            input_ids=expr,
            attention_mask=torch.ones(B, N_GENES_MODEL, dtype=torch.long, device=device),
        )
        hidden = outputs.last_hidden_state.float()  # [B, 19266, 256]

        # Global mean pool (exclude 2 summary tokens)
        global_pool = hidden[:, :N_GENES_MODEL, :].mean(dim=1)  # [B, 256]

        # Perturbed-gene positional embedding
        safe_pos = pert_pos.clamp(min=0)
        pos_idx = safe_pos.view(B, 1, 1).expand(B, 1, HIDDEN_DIM)
        pert_hidden = hidden.gather(1, pos_idx).squeeze(1)  # [B, 256]

        # Fall back to global pool for unknown genes
        unknown_mask = (pert_pos < 0)
        if unknown_mask.any():
            pert_hidden = pert_hidden.clone()
            pert_hidden[unknown_mask] = global_pool[unknown_mask]

        aido_pool = torch.cat([pert_hidden, global_pool], dim=1)  # [B, 512]

        # ── Branch 2: STRING GNN PPI lookup ──────────────────────────────
        # ppi_emb_matrix is registered as a buffer [18870, 256]
        ppi_emb_buf = self.ppi_emb_matrix.to(device)
        safe_ppi = ppi_idx.clamp(min=0)  # [B]
        ppi_emb = ppi_emb_buf[safe_ppi].clone()  # [B, 256] — clone to ensure writeable

        # Use learnable unknown embedding for genes not in STRING graph
        unknown_ppi = (ppi_idx < 0)
        if unknown_ppi.any():
            unk_count = unknown_ppi.sum().item()
            unk_indices = torch.where(unknown_ppi)[0]
            ppi_emb[unk_indices] = self.ppi_unk_embed.to(device).unsqueeze(0).expand(unk_count, -1)

        # ── Branch 3: Symbol CNN ──────────────────────────────────────────
        sym_emb = self.symbol_cnn(symbol_ids.to(device))  # [B, 64]

        # ── 4-source Fusion MLP Head ──────────────────────────────────────
        fused = torch.cat([aido_pool, ppi_emb, sym_emb], dim=1)  # [B, 832]
        fused = self.head_norm(fused)
        fused = self.head_dropout1(fused)
        fused = F.gelu(self.head_fc1(fused))  # [B, 384]
        fused = self.head_dropout2(fused)
        logits = self.head_fc2(fused)          # [B, 19920]

        return logits.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """
    Per-gene macro-averaged F1 score, matching calc_metric.py logic exactly.

    y_pred: [n_samples, 3, n_genes] — class probabilities
    y_true_remapped: [n_samples, n_genes] — labels in {0,1,2} (y_true + 1)
    """
    n_genes = y_true_remapped.shape[1]
    y_hat = y_pred.argmax(axis=1)  # [n_samples, n_genes]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp = y_hat[:, g]
        present = np.array([(yt == c).any() for c in range(N_CLASSES)])
        pf1 = sk_f1_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        dropout: float = 0.4,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        backbone_lr: float = 2e-4,
        head_lr: float = 6e-4,
        weight_decay: float = 0.03,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 80,
        lr_reduce_patience: int = 8,
        lr_reduce_factor: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCell4SourceDEGModel] = None
        self.loss_fn: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = AIDOCell4SourceDEGModel(
                dropout=self.hparams.dropout,
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
            )
            self.loss_fn = FocalLoss(
                gamma=self.hparams.focal_gamma,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        # Populate test metadata for prediction saving
        if stage in ("test", None):
            dm = getattr(self, "trainer", None)
            if dm is not None:
                dm = getattr(self.trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "test_pert_ids") and dm.test_pert_ids:
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            batch["expr"],
            batch["pert_pos"],
            batch["ppi_idx"],
            batch["symbol_ids"],
        )  # [B, 3, 6640]

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, 3, 6640], labels: [B, 6640] in {-1,0,1}
        # CrossEntropyLoss requires {0,1,2}, so remap here
        labels = labels + 1  # {-1,0,1} → {0,1,2}
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
        if not self._val_preds:
            return

        lp = torch.cat(self._val_preds, 0)   # [N, 3, 6640]
        ll = torch.cat(self._val_labels, 0)  # [N, 6640]
        li = torch.cat(self._val_indices, 0) # [N]

        # Gather from all ranks and de-duplicate
        ap = self.all_gather(lp)  # [world, N, 3, 6640]
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

            # Safely get test metadata — prefer trainer.datamodule reference first,
            # fall back to stored instance attributes (populated in setup("test")).
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "test_pert_ids") and dm.test_pert_ids:
                pert_ids = dm.test_pert_ids
                symbols = dm.test_symbols
            else:
                pert_ids = self._test_pert_ids
                symbols = self._test_symbols

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for r, i in enumerate(idxs):
                idx_int = int(i)
                if pert_ids and idx_int < len(pert_ids):
                    row_pert_id = pert_ids[idx_int]
                    row_symbol = symbols[idx_int] if symbols and idx_int < len(symbols) else ""
                else:
                    row_pert_id = str(idx_int)
                    row_symbol = ""
                rows.append({
                    "idx": row_pert_id,
                    "input": row_symbol,
                    "prediction": json.dumps(preds[r].tolist()),
                })
            pred_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {pred_path}")

    def configure_optimizers(self):
        # Separate parameter groups with different LRs
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters() if p.requires_grad
        ]
        ppi_unk_params = [self.model.ppi_unk_embed]
        symbol_cnn_params = list(self.model.symbol_cnn.parameters())
        head_params = (
            list(self.model.head_norm.parameters()) +
            list(self.model.head_fc1.parameters()) +
            list(self.model.head_fc2.parameters())
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params,  "lr": self.hparams.backbone_lr},
                {"params": ppi_unk_params,   "lr": self.hparams.head_lr},
                {"params": symbol_cnn_params, "lr": self.hparams.head_lr},
                {"params": head_params,       "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
            eps=1e-8,
        )

        # ReduceLROnPlateau on val_f1 (mode=max) — proven effective in node3-2
        # IMPORTANT: Do NOT use val_loss monitoring — it fires too early with focal loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.lr_reduce_factor,
            patience=self.hparams.lr_reduce_patience,
            min_lr=1e-8,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ── Checkpoint helpers ─────────────────────────────────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_sd = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable_sd[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                trainable_sd[key] = full_sd[key]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable}/{total} params "
            f"({100.0 * trainable / total:.2f}%), plus {buffers} buffer values"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        """Load partial checkpoint (trainable params + buffers only)."""
        full_keys = set(super().state_dict().keys())
        trainable_keys = {n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys = {n for n, _ in self.named_buffers() if n in full_keys}
        expected_keys = trainable_keys | buffer_keys

        missing = [k for k in expected_keys if k not in state_dict]
        unexpected = [k for k in state_dict if k not in expected_keys]
        if missing:
            self.print(f"Warning: Missing keys in checkpoint (first 5): {missing[:5]}")
        if unexpected:
            self.print(f"Warning: Unexpected keys in checkpoint (first 5): {unexpected[:5]}")
        return super().load_state_dict(state_dict, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 1-2: AIDO.Cell-10M + LoRA (all 8 layers) + STRING GNN + Symbol CNN"
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "data"),
    )
    p.add_argument("--micro_batch_size",        type=int,   default=4)
    p.add_argument("--global_batch_size",       type=int,   default=32)
    p.add_argument("--max_epochs",              type=int,   default=80)
    p.add_argument("--backbone_lr",             type=float, default=2e-4)
    p.add_argument("--head_lr",                 type=float, default=6e-4)
    p.add_argument("--weight_decay",            type=float, default=0.03)
    p.add_argument("--dropout",                 type=float, default=0.4)
    p.add_argument("--lora_r",                  type=int,   default=4)
    p.add_argument("--lora_alpha",              type=int,   default=8)
    p.add_argument("--lora_dropout",            type=float, default=0.1)
    p.add_argument("--focal_gamma",             type=float, default=2.0)
    p.add_argument("--label_smoothing",         type=float, default=0.05)
    p.add_argument("--lr_reduce_patience",      type=int,   default=8)
    p.add_argument("--lr_reduce_factor",        type=float, default=0.5)
    p.add_argument("--early_stopping_patience", type=int,   default=18)
    p.add_argument("--num_workers",             type=int,   default=0)
    p.add_argument("--val_check_interval",      type=float, default=1.0)
    p.add_argument("--debug_max_step",          type=int,   default=None)
    p.add_argument("--fast_dev_run",            action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
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
        filename="node1-2-{epoch:03d}-{val_f1:.4f}",
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
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    # Strategy: DDP for multi-GPU, SingleDevice for single GPU
    if n_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(seconds=300),
        )
    else:
        strategy = SingleDeviceStrategy(device="cuda:0")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit,
        limit_val_batches=limit,
        # Always process ALL test batches to save complete predictions
        limit_test_batches=1.0,
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
        dropout=args.dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        lr_reduce_patience=args.lr_reduce_patience,
        lr_reduce_factor=args.lr_reduce_factor,
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
        score_path.write_text(
            f"# Node 1-2 Test Evaluation Results\n"
            f"# Model: AIDO.Cell-10M + LoRA r=4 all-8-layers + STRING GNN PPI + Symbol CNN\n"
            f"# Head: 832→384→19920 MLP (4-source fusion)\n"
            f"# Primary metric: f1_score (macro-averaged per-gene F1)\n"
            f"# Trainable params: ~8.1M (LoRA ~47K + SymCNN ~50K + MLP ~8.0M)\n"
            f"val_f1_best: {primary_val:.6f}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
