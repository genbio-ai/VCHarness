#!/usr/bin/env python3
"""
Node 3-1-1-2-1: AIDO.Cell-10M + LoRA (r=4, ALL 8 layers) + Symbol CNN + Frozen STRING GNN PPI + 4-source MLP Head
=================================================================================================================
Improves upon node3-1-1-2 (parent, best_val_f1=0.4428, test_F1~0.3965) by adding the single
most critical missing component identified in its feedback report:

  1. Frozen STRING GNN PPI embeddings (256-dim): The primary bottleneck in the parent is the
     absence of STRING PPI biological signal. EVERY node achieving >=0.45 F1 in the MCTS tree
     used STRING PPI topology embeddings as a 4th information channel:
       - node3-2 (tree-best, 0.462): AIDO.Cell LoRA + Symbol CNN + STRING PPI + MLP(384)
       - node2-3-1-1 (0.4555): AIDO.Cell LoRA + Symbol CNN + STRING PPI + MLP(384)
       - node3-3 (0.4513): AIDO.Cell LoRA + Symbol CNN + STRING PPI + MLP(384)
     The parent's 3-source fusion (AIDO.Cell + Symbol CNN + MLP) is capped at ~0.44 F1 —
     identical to node2-2's 3-source ceiling (0.4453). STRING PPI breaks this ceiling.

  2. Head widened to 384-dim (vs parent's 256-dim): Matches node3-2's proven head hidden size.
     With 4 input sources (960-dim fusion), a wider bottleneck (384-dim) provides sufficient
     capacity to learn non-linear perturbation-gene interactions.

  3. Head dropout increased to 0.4 (vs parent's 0.3): node3-2 uses 0.4. Combined with the
     wider STRING PPI signal, stronger regularization is needed.

  4. Weight decay increased to 0.03 (vs parent's 0.01): node3-2 uses 0.03. With 4 input sources,
     stronger L2 regularization reduces overfitting on the 1,500 training samples.

  5. Backbone/head LR tuned to 2e-4/6e-4 (vs parent's 3e-4/9e-4): Matching the best-performing
     node2-3-1-1 (0.4555) and node3-2 (0.462) configurations.

  6. Extended early stopping patience to 30 (vs parent's 25): node2-3-1-1-1-1 showed patience=20
     was too short, stopping before the val_f1 could converge. Longer patience lets val_f1
     explore more of the optimization landscape.

  7. PPI projection: 1-layer linear (256→256) — proven in node3-2/node3-3/node2-2-2-1.

Architecture:
  Input: pert_id (ENSG gene ID) + symbol (gene symbol string)
    ↓
  AIDO.Cell-10M backbone (LoRA r=4, Q/K/V on all 8 layers)
    → dual-pooling (pert_hidden + global_mean) → [B, 512]
  Symbol CNN (3-branch, 64-dim/branch) → [B, 192]
  Frozen STRING GNN PPI (pre-computed, projected 256→256) → [B, 256]
    ↓
  Fusion: concat([512, 192, 256]) → [B, 960]
    ↓
  MLP head: LayerNorm(960) → Linear(960→384) → GELU → Dropout(0.4) → Linear(384→3*6640)
    ↓
  logits: [B, 3, 6640]

Total trainable parameters: ~13.4M
  - LoRA adapters (r=4, Q/K/V, all 8 layers): ~36K
  - Symbol CNN: ~98K
  - PPI linear projection: ~66K
  - head LayerNorm(960): ~2K
  - head Linear(960→384): ~369K
  - head Linear(384→3*6640): ~7.7M
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
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT = 6_640
N_GENES_MODEL = 19_264
N_CLASSES = 3
HIDDEN_DIM = 256  # AIDO.Cell-10M hidden size
PPI_DIM = 256     # STRING GNN output dim

# Class weights: proven effective in node2-2, node3-2, node2-3-1-1
# [5.0, 1.0, 10.0] for [down-reg (-1→0), unchanged (0→1), up-reg (+1→2)]
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)

# Character vocabulary for Symbol CNN (ASCII printable chars)
CHAR_VOCAB = {c: i + 1 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")}
CHAR_VOCAB_SIZE = len(CHAR_VOCAB) + 1  # +1 for padding
SYMBOL_MAX_LEN = 16  # pad/truncate gene symbols to this length


# ─────────────────────────────────────────────────────────────────────────────
# Symbol CNN utilities
# ─────────────────────────────────────────────────────────────────────────────
def encode_symbol(symbol: str, max_len: int = SYMBOL_MAX_LEN) -> torch.Tensor:
    """Encode gene symbol string to integer token sequence."""
    s = symbol.upper()[:max_len]
    tokens = [CHAR_VOCAB.get(c, 0) for c in s]
    while len(tokens) < max_len:
        tokens.append(0)
    return torch.tensor(tokens, dtype=torch.long)


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
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Pre-builds synthetic expression vectors for each perturbation sample.

    Input encoding: all 19264 genes at 1.0 (baseline), perturbed gene at 0.0.
    Also encodes gene symbols as character token sequences for Symbol CNN.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_pos_map: Dict[str, int],
        ppi_node_idx_map: Dict[str, int],  # ENSG_id -> STRING GNN node index
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Map {-1,0,1} → {0,1,2} to match calc_metric.py's y_true + 1 convention
            self.labels = np.array(raw_labels, dtype=np.int8) + 1
        else:
            self.labels = None

        # Pre-compute expression vectors for efficiency
        base_expr = torch.ones(N_GENES_MODEL, dtype=torch.float32)

        self._exprs: List[torch.Tensor] = []
        self._pert_positions: List[int] = []
        self._symbol_tokens: List[torch.Tensor] = []
        self._ppi_node_indices: List[int] = []  # index into STRING GNN embedding matrix
        covered_aido = 0
        covered_ppi = 0

        for pid, sym in zip(self.pert_ids, self.symbols):
            base_pid = pid.split(".")[0]

            # AIDO.Cell position
            pos = gene_pos_map.get(base_pid, -1)
            self._pert_positions.append(pos)
            if pos >= 0:
                expr = base_expr.clone()
                expr[pos] = 0.0  # knockout signal
                covered_aido += 1
            else:
                expr = base_expr.clone()
            self._exprs.append(expr)
            self._symbol_tokens.append(encode_symbol(sym))

            # STRING GNN PPI node index
            ppi_idx = ppi_node_idx_map.get(base_pid, -1)
            self._ppi_node_indices.append(ppi_idx)
            if ppi_idx >= 0:
                covered_ppi += 1

        if not is_test:
            n = len(self.pert_ids)
            print(f"[Dataset] {n} samples, "
                  f"AIDO.Cell coverage: {covered_aido}/{n} ({100.0*covered_aido/n:.1f}%), "
                  f"STRING PPI coverage: {covered_ppi}/{n} ({100.0*covered_ppi/n:.1f}%)")

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "expr": self._exprs[idx],                         # [19264] float32
            "pert_pos": self._pert_positions[idx],            # int
            "symbol_tokens": self._symbol_tokens[idx],        # [SYMBOL_MAX_LEN] long
            "ppi_idx": self._ppi_node_indices[idx],           # int
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
        "expr": torch.stack([b["expr"] for b in batch]),                       # [B, 19264]
        "pert_pos": torch.tensor([b["pert_pos"] for b in batch], dtype=torch.long),
        "symbol_tokens": torch.stack([b["symbol_tokens"] for b in batch]),     # [B, SYMBOL_MAX_LEN]
        "ppi_idx": torch.tensor([b["ppi_idx"] for b in batch], dtype=torch.long),  # [B]
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])             # [B, 6640]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, micro_batch_size: int = 8, num_workers: int = 0):
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
        self._ppi_node_idx_map: Optional[Dict[str, int]] = None

    def _build_gene_pos_map(
        self, tokenizer, all_pert_ids: List[str]
    ) -> Dict[str, int]:
        """Build mapping from ENSG gene ID to its position in the 19264-gene vocabulary."""
        gene_pos_map: Dict[str, int] = {}
        unique_base_ids = list(set(pid.split(".")[0] for pid in all_pert_ids))
        print(f"[DataModule] Building gene position map for {len(unique_base_ids)} unique genes...")

        if hasattr(tokenizer, "gene_id_to_index"):
            gid2idx = tokenizer.gene_id_to_index
            for base_pid in unique_base_ids:
                if base_pid in gid2idx:
                    gene_pos_map[base_pid] = gid2idx[base_pid]
            if len(gene_pos_map) > 0:
                print(f"[DataModule] ENSG→pos via gene_id_to_index: "
                      f"{len(gene_pos_map)}/{len(unique_base_ids)} found")
                return gene_pos_map

        print(f"[DataModule] No gene position mapping available; all pert_pos will be -1")
        return gene_pos_map

    def _build_ppi_node_idx_map(self, all_pert_ids: List[str]) -> Dict[str, int]:
        """Build mapping from ENSG gene ID to its index in STRING GNN node_names.json."""
        ppi_map: Dict[str, int] = {}
        node_names_path = Path(STRING_GNN_DIR) / "node_names.json"
        if not node_names_path.exists():
            print(f"[DataModule] STRING GNN node_names.json not found at {node_names_path}")
            return ppi_map

        node_names = json.loads(node_names_path.read_text())
        # node_names is a list of ENSG IDs aligned with STRING GNN node indices
        node_name_to_idx = {name: i for i, name in enumerate(node_names)}

        unique_base_ids = list(set(pid.split(".")[0] for pid in all_pert_ids))
        found = 0
        for base_pid in unique_base_ids:
            if base_pid in node_name_to_idx:
                ppi_map[base_pid] = node_name_to_idx[base_pid]
                found += 1

        print(f"[DataModule] STRING PPI node mapping: {found}/{len(unique_base_ids)} genes found")
        return ppi_map

    def setup(self, stage: Optional[str] = None) -> None:
        # Tokenizer: rank-0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

        # Collect all pert_ids for building maps
        if self._gene_pos_map is None or self._ppi_node_idx_map is None:
            all_ids: List[str] = []
            for fname in ["train.tsv", "val.tsv", "test.tsv"]:
                fpath = self.data_dir / fname
                if fpath.exists():
                    df_tmp = pd.read_csv(fpath, sep="\t")
                    if "pert_id" in df_tmp.columns:
                        all_ids.extend(df_tmp["pert_id"].tolist())

            if self._gene_pos_map is None:
                self._gene_pos_map = self._build_gene_pos_map(tokenizer, all_ids)
            if self._ppi_node_idx_map is None:
                self._ppi_node_idx_map = self._build_ppi_node_idx_map(all_ids)

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self._gene_pos_map, self._ppi_node_idx_map
            )
            self.val_ds = PerturbationDataset(
                val_df, self._gene_pos_map, self._ppi_node_idx_map
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self._gene_pos_map, self._ppi_node_idx_map, is_test=True
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
# Symbol CNN
# ─────────────────────────────────────────────────────────────────────────────
class SymbolCNN(nn.Module):
    """
    3-branch character-level CNN for gene symbol encoding.

    Proven effective in node2-2 (test F1=0.4453), node3-2 (0.462), etc.
    Captures gene family naming conventions (prefixes, suffixes) as orthogonal
    signal to the AIDO.Cell positional embedding.

    Input:  [B, SYMBOL_MAX_LEN] long (char token indices)
    Output: [B, out_dim]
    """

    def __init__(self, out_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(CHAR_VOCAB_SIZE, 32, padding_idx=0)

        # 3 parallel convolutions with different kernel sizes
        self.conv3 = nn.Conv1d(32, out_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(32, out_dim, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(32, out_dim, kernel_size=5, padding=2)

        self.dropout = nn.Dropout(dropout)
        self._out_dim = out_dim * 3  # concatenated output

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L]
        x = self.embedding(tokens)   # [B, L, 32]
        x = x.permute(0, 2, 1)       # [B, 32, L]

        # Parallel convolutions + max-pool
        h3 = F.gelu(self.conv3(x)).max(dim=2).values   # [B, out_dim]
        h4 = F.gelu(self.conv4(x)).max(dim=2).values
        h5 = F.gelu(self.conv5(x)).max(dim=2).values

        out = torch.cat([h3, h4, h5], dim=1)            # [B, 3*out_dim]
        return self.dropout(out)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class AIDOCellDEGModel(nn.Module):
    """
    4-source feature fusion DEG predictor:
      - AIDO.Cell-10M backbone with LoRA r=4 on ALL 8 layers (Q/K/V)
      - Character-level Symbol CNN (3-branch, 64-dim per branch → 192-dim total)
      - Frozen STRING GNN PPI embeddings (pre-computed, projected 256→256) → [B, 256]
      - Dual pooling: [pert_hidden || global_mean] → [B, 512]
      - Fusion: concat([512-dim backbone, 192-dim symbol, 256-dim PPI]) → [B, 960]
      - 2-stage MLP head: LayerNorm(960) → Linear(960→384) → GELU → Dropout(0.4) → Linear(384→3*6640)

    Total trainable parameters: ~13.4M
      - LoRA adapters (r=4, Q/K/V, all 8 layers): ~36K
      - Symbol CNN: ~98K
      - PPI linear projection (256→256): ~66K
      - head Linear(960→384): ~369K
      - head Linear(384→3*6640): ~7.7M
      - head LayerNorm(960) + biases: ~2K
    """

    def __init__(
        self,
        symbol_cnn_dim: int = 64,
        symbol_cnn_dropout: float = 0.2,
        head_hidden_dim: int = 384,
        head_dropout: float = 0.4,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        ppi_emb_matrix: Optional[torch.Tensor] = None,  # [N_STRING_NODES, 256]
    ):
        super().__init__()

        # ── Backbone: AIDO.Cell-10M with LoRA on ALL 8 layers ──────────────
        backbone = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16)
        backbone.config.use_cache = False

        # Monkey-patch enable_input_require_grads to avoid NotImplementedError
        def noop_enable_input_require_grads(self):
            pass
        backbone.enable_input_require_grads = noop_enable_input_require_grads.__get__(
            backbone, type(backbone))

        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # LoRA on ALL 8 transformer layers (0-7) — r=4, Q/K/V only
        # Proven configuration in node3-2 (0.462), node2-3-1-1 (0.4555)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=list(range(8)),  # all 8 layers of AIDO.Cell-10M
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Cast LoRA (trainable) params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Symbol CNN ─────────────────────────────────────────────────────
        self.symbol_cnn = SymbolCNN(out_dim=symbol_cnn_dim, dropout=symbol_cnn_dropout)
        symbol_out_dim = self.symbol_cnn.output_dim  # 3 * symbol_cnn_dim = 192

        # ── Frozen STRING GNN PPI embeddings ────────────────────────────────
        # Pre-computed frozen embedding matrix [N_STRING_NODES, 256]
        # Stored as a buffer (not trainable), indexed by per-sample ppi_idx
        if ppi_emb_matrix is not None:
            self.register_buffer("ppi_emb_matrix", ppi_emb_matrix.float())
        else:
            # Placeholder — will be replaced after loading
            self.register_buffer("ppi_emb_matrix", torch.zeros(1, PPI_DIM))

        # 1-layer linear projection for PPI features: 256→256
        # This allows the model to adapt the frozen embeddings for the DEG task
        self.ppi_proj = nn.Linear(PPI_DIM, PPI_DIM, bias=True)
        nn.init.eye_(self.ppi_proj.weight)
        nn.init.zeros_(self.ppi_proj.bias)

        # ── Fusion + MLP head ──────────────────────────────────────────────
        # Dual-pool backbone: pert_hidden + global_mean → 512-dim
        # Symbol CNN: → 192-dim
        # PPI embeddings (after projection): → 256-dim
        # Total fusion: 512 + 192 + 256 = 960-dim input to head
        fusion_dim = HIDDEN_DIM * 2 + symbol_out_dim + PPI_DIM  # 512 + 192 + 256 = 960

        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, head_hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, N_CLASSES * N_GENES_OUT, bias=True),
        )

        # Initialize head weights
        nn.init.trunc_normal_(self.head[1].weight, std=0.02)
        nn.init.zeros_(self.head[1].bias)
        nn.init.trunc_normal_(self.head[4].weight, std=0.01)
        nn.init.zeros_(self.head[4].bias)

        # Cast ppi_proj and head to float32
        self.ppi_proj = self.ppi_proj.float()
        self.head = self.head.float()
        self.symbol_cnn = self.symbol_cnn.float()

    def forward(
        self, expr: torch.Tensor, pert_pos: torch.Tensor,
        symbol_tokens: torch.Tensor, ppi_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            expr:          [B, 19264] float32 — synthetic expression (1.0 baseline, 0.0 at pert)
            pert_pos:      [B] long — position of perturbed gene in AIDO.Cell vocab (-1 if unknown)
            symbol_tokens: [B, SYMBOL_MAX_LEN] long — encoded gene symbol chars
            ppi_idx:       [B] long — index into STRING GNN PPI embedding matrix (-1 if unknown)
        Returns:
            logits: [B, 3, 6640]
        """
        B = expr.shape[0]
        device = expr.device

        # ── 1. AIDO.Cell forward pass ──────────────────────────────────────
        outputs = self.backbone(
            input_ids=expr,
            attention_mask=torch.ones(B, N_GENES_MODEL, dtype=torch.long, device=device),
        )
        # Cast to float32 for stable head computation
        hidden = outputs.last_hidden_state.float()  # [B, 19266, 256]

        # ── 2. Dual pooling ────────────────────────────────────────────────
        # Global mean pool over gene positions (exclude 2 summary tokens)
        global_pool = hidden[:, :N_GENES_MODEL, :].mean(dim=1)  # [B, 256]

        # Per-sample perturbed-gene positional embedding
        safe_pos = pert_pos.clamp(min=0)  # [B]
        pos_idx = safe_pos.view(B, 1, 1).expand(B, 1, HIDDEN_DIM)  # [B, 1, 256]
        pert_hidden = hidden.gather(1, pos_idx).squeeze(1)  # [B, 256]

        # Fall back to global pool for genes not in vocabulary
        unknown_mask = (pert_pos < 0)
        if unknown_mask.any():
            pert_hidden = pert_hidden.clone()
            pert_hidden[unknown_mask] = global_pool[unknown_mask]

        # Concatenate dual pool: [B, 512]
        backbone_feat = torch.cat([pert_hidden, global_pool], dim=1)  # [B, 512]

        # ── 3. Symbol CNN ──────────────────────────────────────────────────
        sym_feat = self.symbol_cnn(symbol_tokens)  # [B, 192]

        # ── 4. STRING GNN PPI embeddings ───────────────────────────────────
        # Look up pre-computed frozen PPI embeddings by index
        safe_ppi_idx = ppi_idx.clamp(min=0)  # [B], clamp -1 → 0
        ppi_raw = self.ppi_emb_matrix[safe_ppi_idx]  # [B, 256]

        # Project PPI embeddings
        ppi_feat = self.ppi_proj(ppi_raw)  # [B, 256]

        # Fall back to zeros for genes not in PPI (no PPI signal)
        ppi_unknown_mask = (ppi_idx < 0)
        if ppi_unknown_mask.any():
            ppi_feat = ppi_feat.clone()
            ppi_feat[ppi_unknown_mask] = 0.0

        # ── 5. Feature fusion ──────────────────────────────────────────────
        fused = torch.cat([backbone_feat, sym_feat, ppi_feat], dim=1)  # [B, 960]

        # ── 6. 2-layer MLP head ────────────────────────────────────────────
        logits_flat = self.head(fused)                        # [B, 3*6640]
        logits = logits_flat.view(B, N_GENES_OUT, N_CLASSES)  # [B, 6640, 3]

        return logits.permute(0, 2, 1)  # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """
    Per-gene macro-averaged F1 score, matching calc_metric.py logic exactly.

    y_pred: [n_samples, 3, n_genes] — class probabilities
    y_true_remapped: [n_samples, n_genes] — labels in {0,1,2} (i.e., y_true+1)
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
        symbol_cnn_dim: int = 64,
        symbol_cnn_dropout: float = 0.2,
        head_hidden_dim: int = 384,
        head_dropout: float = 0.4,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        backbone_lr: float = 2e-4,
        head_lr: float = 6e-4,
        weight_decay: float = 0.03,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        plateau_patience: int = 8,
        plateau_factor: float = 0.5,
        min_lr: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCellDEGModel] = None
        self.loss_fn: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def _load_ppi_embeddings(self) -> torch.Tensor:
        """
        Load and compute frozen STRING GNN PPI embeddings.
        Returns [N_STRING_NODES, 256] float32 tensor.
        Computed once on rank-0 to avoid repeated inference.
        """
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        string_gnn_dir = Path(STRING_GNN_DIR)
        graph_path = string_gnn_dir / "graph_data.pt"

        # Load STRING GNN model
        gnn_model = AutoModel.from_pretrained(str(string_gnn_dir), trust_remote_code=True)
        gnn_model = gnn_model.to(device).eval()

        # Load graph data
        graph = torch.load(graph_path, map_location=device)
        edge_index = graph["edge_index"].to(device)
        edge_weight = graph.get("edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        # Run inference to get frozen PPI embeddings
        with torch.no_grad():
            outputs = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            ppi_emb = outputs.last_hidden_state.cpu().float()  # [18870, 256]

        # Free GNN model memory
        del gnn_model, graph, edge_index
        if edge_weight is not None:
            del edge_weight
        torch.cuda.empty_cache()

        print(f"[PPI] STRING GNN embeddings loaded: shape={ppi_emb.shape}")
        return ppi_emb

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            # Load PPI embeddings on rank-0, then broadcast to all ranks
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank == 0:
                ppi_emb = self._load_ppi_embeddings()
            else:
                ppi_emb = torch.zeros(18870, PPI_DIM, dtype=torch.float32)

            # Broadcast PPI embeddings across all DDP ranks
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                ppi_emb = ppi_emb.cuda()
                torch.distributed.broadcast(ppi_emb, src=0)
                ppi_emb = ppi_emb.cpu()

            self.model = AIDOCellDEGModel(
                symbol_cnn_dim=self.hparams.symbol_cnn_dim,
                symbol_cnn_dropout=self.hparams.symbol_cnn_dropout,
                head_hidden_dim=self.hparams.head_hidden_dim,
                head_dropout=self.hparams.head_dropout,
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                ppi_emb_matrix=ppi_emb,
            )
            self.loss_fn = FocalLoss(
                gamma=self.hparams.focal_gamma,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        # Populate test metadata from the datamodule.
        # This runs during both "fit" and "test" stages to ensure metadata is
        # available regardless of whether trainer.fit() or trainer.test() is called first.
        # The test dataloader and test_ds are created during DataModule.setup(None)
        # (called before setup("fit") in Lightning), so test_ds is always available.
        dm = getattr(self, "trainer", None)
        if dm is not None:
            dm = getattr(self.trainer, "datamodule", None)
        if dm is not None:
            # Primary source: test_ds has pert_ids/symbols as instance attributes
            test_ds = getattr(dm, "test_ds", None)
            if test_ds is not None and hasattr(test_ds, "pert_ids"):
                self._test_pert_ids = test_ds.pert_ids
                self._test_symbols = test_ds.symbols
            # Fallback: dm's own test attributes (populated during DataModule.setup("test"))
            elif hasattr(dm, "test_pert_ids") and dm.test_pert_ids:
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            batch["expr"],
            batch["pert_pos"],
            batch["symbol_tokens"],
            batch["ppi_idx"],
        )  # [B, 3, 6640]

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

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "idx": self._test_pert_ids[i],
                    "input": self._test_symbols[i],
                    "prediction": json.dumps(preds[r].tolist()),
                }
                for r, i in enumerate(idxs)
            ]
            pred_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {pred_path}")

    def configure_optimizers(self):
        # Separate learning rates:
        #   backbone LoRA params (lowest LR — backbone adaptation)
        #   symbol CNN params (medium LR)
        #   PPI projection + head params (higher LR — task-specific layers)
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters() if p.requires_grad
        ]
        symbol_params = list(self.model.symbol_cnn.parameters())
        head_params = (
            list(self.model.ppi_proj.parameters()) +
            list(self.model.head.parameters())
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
                {"params": symbol_params,   "lr": self.hparams.head_lr},
                {"params": head_params,     "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
            eps=1e-8,
        )

        # ReduceLROnPlateau monitoring val_f1 (mode=max)
        # CRITICAL: Do NOT monitor val_loss — node3-2-2 confirmed this causes
        # premature LR reductions because focal loss makes val_loss inversely
        # correlated with val_f1 during the learning phase.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=self.hparams.plateau_patience,
            factor=self.hparams.plateau_factor,
            min_lr=self.hparams.min_lr,
            verbose=True,
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
        description="Node 3-1-1-2-1: AIDO.Cell-10M + LoRA all-8-layers + Symbol CNN + STRING PPI + 4-source MLP head"
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "data"),
    )
    p.add_argument("--micro-batch-size",        type=int,   default=8)
    p.add_argument("--global-batch-size",       type=int,   default=64)
    p.add_argument("--max-epochs",              type=int,   default=100)
    p.add_argument("--backbone-lr",             type=float, default=2e-4)
    p.add_argument("--head-lr",                 type=float, default=6e-4)
    p.add_argument("--weight-decay",            type=float, default=0.03)
    p.add_argument("--symbol-cnn-dim",          type=int,   default=64)
    p.add_argument("--symbol-cnn-dropout",      type=float, default=0.2)
    p.add_argument("--head-hidden-dim",         type=int,   default=384)
    p.add_argument("--head-dropout",            type=float, default=0.4)
    p.add_argument("--lora-r",                  type=int,   default=4)
    p.add_argument("--lora-alpha",              type=int,   default=8)
    p.add_argument("--lora-dropout",            type=float, default=0.1)
    p.add_argument("--focal-gamma",             type=float, default=2.0)
    p.add_argument("--label-smoothing",         type=float, default=0.05)
    p.add_argument("--plateau-patience",        type=int,   default=8)
    p.add_argument("--plateau-factor",          type=float, default=0.5)
    p.add_argument("--min-lr",                  type=float, default=1e-8)
    p.add_argument("--early-stopping-patience", type=int,   default=30)
    p.add_argument("--num-workers",             type=int,   default=0)
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

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit_train = args.debug_max_step if args.debug_max_step is not None else 1.0
    limit_val = 1.0
    limit_test = 1.0

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node3-1-1-2-1-{epoch:03d}-{val_f1:.4f}",
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
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
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
        symbol_cnn_dim=args.symbol_cnn_dim,
        symbol_cnn_dropout=args.symbol_cnn_dropout,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        min_lr=args.min_lr,
    )

    trainer.fit(model_module, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(
            model_module, datamodule=datamodule, ckpt_path="best"
        )

    # Only write test_score.txt for production runs (not fast_dev_run / debug_max_step).
    # For debug runs, the score is meaningless and should not overwrite a real result.
    if trainer.is_global_zero and not args.fast_dev_run and args.debug_max_step is None:
        score_path = Path(__file__).parent / "test_score.txt"
        best_val_f1 = checkpoint_cb.best_model_score
        if best_val_f1 is not None:
            score = float(best_val_f1)
        else:
            score = 0.0
        score_path.write_text(f"{score}\n")
        print(f"Best val_f1 = {score:.6f} → {score_path}")


if __name__ == "__main__":
    main()
