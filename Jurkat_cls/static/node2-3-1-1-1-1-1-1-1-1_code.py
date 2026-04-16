#!/usr/bin/env python3
"""
Node 2-3-1-1-1-1-1-1-1-1: AIDO.Cell-10M + LoRA (r=4, ALL 8 layers) + Symbol CNN
                + Frozen STRING GNN PPI Embeddings
                + FIXED Top-3 Checkpoint Averaging + Reverted weight_decay=0.03
                + Increased plateau_patience=7 to avoid premature LR reduction
===========================================================================
Parent: node2-3-1-1-1-1-1-1-1 (test F1=0.4586, val_f1=0.4631 at E14)

Root cause of parent regression (from feedback.md):
1. CHECKPOINT AVERAGING FAILED (CRITICAL):
   - Error: "No backend type associated with device type cpu"
   - Cause: torch.stack().mean() on CPU-loaded bfloat16 checkpoint tensors
   - Result: Primary improvement (+0.003 F1 variance reduction) was completely lost
   - Fix: Load checkpoints to CUDA, explicitly cast to float32 before stacking

2. WEIGHT DECAY 0.05 WAS COUNTERPRODUCTIVE:
   - 0.05 caused training instability vs grandparent's 0.03
   - val_loss worsened (0.531→0.576 vs grandparent's 0.531→0.55)
   - Feedback: "weight_decay=0.05 caused marginal training instability"
   - Fix: Revert to 0.03 (grandparent's proven setting)

3. PREMATURE LR REDUCTION:
   - Parent feedback: LR fired at E10→E11 (2e-4→1e-4) due to brief val_loss dip
   - This may have disrupted the optimization trajectory prematurely
   - Fix: Increase plateau_patience from 5 to 7 to tolerate brief val_loss fluctuations

This node applies THREE targeted fixes:

1. FIX CHECKPOINT AVERAGING (CRITICAL):
   - Load checkpoints to device="cuda" (or the correct GPU device)
   - Cast all tensors to float32 before stacking/averaging
   - This fixes the "No backend type associated with device type cpu" error
   - Expected: +0.002-0.005 F1 (variance reduction, same as tree-best node2-2-3-1-1-1)

2. REVERT WEIGHT DECAY: 0.05 → 0.03:
   - Feedback confirms 0.05 was too high and caused instability
   - Grandparent at 0.03 had identical peak val_f1=0.4631 with better val_loss trajectory
   - Expected: +0.001-0.002 F1 from improved generalization dynamics

3. INCREASE PLATEAU PATIENCE: 5 → 7:
   - Parent fired LR at E10 (brief dip) then again at E15→E16
   - Increase plateau_patience to 7 prevents premature LR reduction on brief fluctuations
   - Expected: 0 to +0.001 F1 from smoother LR schedule and delayed unnecessary LR reductions

Architecture (unchanged from grandparent node2-3-1-1-1-1-1-1, test F1=0.4631):
  - AIDO.Cell-10M: r=4 LoRA on all 8 transformer layers (Q/K/V)
  - Symbol CNN: 3-branch char-level Conv1d → 64-dim
  - STRING GNN: frozen inference → static [18870, 256] table → 1-layer projection → 256-dim
  - Head: LayerNorm(832) → Linear(832→384) → GELU → Dropout(0.40) → LayerNorm(384) → Linear(384→19920)
  - Combined input: [global_emb(256) + pert_emb(256) + sym(64) + ppi(256)] = 832-dim
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
N_GENES_OUT = 6_640      # output genes (DEG prediction target panel)
N_CLASSES = 3
SENTINEL_EXPR = 1.0      # baseline expression for non-perturbed genes
KNOCKOUT_EXPR = 0.0      # expression for knocked-out gene
AIDO_HIDDEN = 256        # AIDO.Cell-10M hidden dimension
AIDO_N_LAYERS = 8        # AIDO.Cell-10M total transformer layers
STRING_GNN_DIM = 256     # STRING GNN embedding dimension

# Class weights for focal loss: [down-reg, unchanged, up-reg]
# Train: class 0 (down) ~3.4%, class 1 (unchanged) ~95.5%, class 2 (up) ~1.1%
# [7.0, 1.0, 15.0]: tree-best node2-2-3-1-1-1 (0.4655) used this configuration
CLASS_WEIGHTS = torch.tensor([7.0, 1.0, 15.0], dtype=torch.float32)

# Character vocabulary for gene symbol CNN encoder
SYMBOL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
SYMBOL_PAD_IDX = len(SYMBOL_CHARS)           # 39 → padding index
SYMBOL_VOCAB_SIZE = len(SYMBOL_CHARS) + 1    # 40
SYMBOL_MAX_LEN = 12                           # max gene symbol length to encode


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss with Label Smoothing
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weights and label smoothing.

    Focal weight computed from hard-target CE (no smoothing) to preserve the
    hard-example emphasis property. CE term uses smoothed targets to prevent
    overconfident predictions on the dominant class (95% unchanged).
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
        """
        logits:  [N, C]  float32 — raw class scores
        targets: [N]     int64   — hard class indices
        """
        w = self.weight.to(logits.device) if self.weight is not None else None

        # (1) Focal weight from hard-target CE (no smoothing)
        with torch.no_grad():
            ce_hard = F.cross_entropy(logits.float(), targets, reduction="none")
            pt = torch.exp(-ce_hard)
            focal_weight = (1.0 - pt) ** self.gamma

        # (2) Smoothed CE with class weighting for gradient signal
        ce_smooth = F.cross_entropy(
            logits.float(),
            targets,
            weight=w,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        return (focal_weight * ce_smooth).mean()


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
    """Character-level CNN encoder for gene symbol strings.

    Three parallel Conv1d filters at kernel sizes 2, 3, 4 → max-pool → 64-dim.
    Captures character n-gram patterns: gene family prefixes, numeric suffixes, etc.
    Proven effective in node2-2 and node3-2 lineages.
    """

    def __init__(self, out_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(SYMBOL_VOCAB_SIZE, embed_dim, padding_idx=SYMBOL_PAD_IDX)
        self.conv2 = nn.Conv1d(embed_dim, 32, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, 32, kernel_size=4, padding=2)
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
        """symbol_ids: [B, SYMBOL_MAX_LEN] int64 → [B, out_dim]"""
        x = self.embed(symbol_ids)          # [B, L, embed_dim]
        x = x.transpose(1, 2)              # [B, embed_dim, L]
        f2 = F.relu(self.conv2(x))
        f3 = F.relu(self.conv3(x))
        f4 = F.relu(self.conv4(x))
        f2 = F.adaptive_max_pool1d(f2, 1).squeeze(-1)  # [B, 32]
        f3 = F.adaptive_max_pool1d(f3, 1).squeeze(-1)  # [B, 32]
        f4 = F.adaptive_max_pool1d(f4, 1).squeeze(-1)  # [B, 32]
        feat = torch.cat([f2, f3, f4], dim=-1)          # [B, 96]
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
    model_dir = Path(string_gnn_dir)
    node_names = json.loads((model_dir / "node_names.json").read_text())
    ensg_to_gnn_idx = {n: i for i, n in enumerate(node_names) if n.startswith("ENSG")}

    graph = torch.load(model_dir / "graph_data.pt", map_location="cpu")
    edge_index = graph["edge_index"]
    edge_weight = graph.get("edge_weight", None)

    # Load STRING GNN and run inference on CPU to avoid GPU memory issues
    gnn_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    gnn_model.eval()

    with torch.no_grad():
        outputs = gnn_model(
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
    # embeddings: [18870, 256] float32
    embeddings = outputs.last_hidden_state.cpu().float()

    del gnn_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings, ensg_to_gnn_idx


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """DEG prediction dataset.

    Pre-builds:
      - AIDO.Cell expression profile tensors [N, 19264]: all genes=1.0 except KO gene=0.0
      - Symbol character index tensors [N, SYMBOL_MAX_LEN]
      - STRING GNN index tensors [N]: node index in STRING graph (-1 if not found)
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
            # Remap {-1, 0, 1} → {0, 1, 2}
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
            "expr": self.expr_inputs[idx],            # [19264] float32
            "gene_pos": gene_pos,                      # int (-1 if not in AIDO vocab)
            "symbol_ids": self.symbol_ids[idx],        # [SYMBOL_MAX_LEN] int64
            "gnn_idx": self.gnn_indices[idx].item(),   # int (-1 if not in STRING)
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

        # Build frozen STRING GNN embeddings once per process
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
        """Map each ENSG gene_id to its position index in AIDO.Cell vocabulary."""
        mapping: Dict[str, int] = {}
        PROBE_VAL = 50.0  # distinctive float to locate gene position in tokenizer output
        for gene_id in gene_ids:
            try:
                inputs = tokenizer(
                    {"gene_ids": [gene_id], "expression": [PROBE_VAL]},
                    return_tensors="pt",
                )
                ids = inputs["input_ids"]
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)   # [1, 19264]
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
# Model
# ──────────────────────────────────────────────────────────────────────────────
class AIDOCellPPIDEGModel(nn.Module):
    """AIDO.Cell-10M + LoRA (r=4, ALL 8 layers) + Symbol CNN + Frozen STRING GNN PPI.

    Feature fusion (4 sources → 832-dim):
      (a) Global mean-pool of AIDO.Cell last_hidden_state    → [B, 256]
          Captures the average perturbation context (distributed cell state)
      (b) Perturbed-gene positional embedding from AIDO.Cell → [B, 256]
          Per-sample differentiation via the knock-out gene's local context
      (c) Gene symbol character-level CNN                    → [B, 64]
          Gene family / naming convention features
      (d) Frozen STRING GNN PPI topology embedding           → [B, 256]
          Protein interaction network position (biological signal)

    Combined: [B, 832] → MLP head (384-dim) → [B, 3, 6640]
    """

    HIDDEN_DIM = 256          # AIDO.Cell-10M hidden size
    SYMBOL_DIM = 64           # symbol CNN output dim
    PPI_DIM = 256             # STRING GNN embedding dim
    HEAD_INPUT_DIM = 256 * 2 + 64 + 256  # global + pert + sym + ppi = 832

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        head_hidden: int = 384,
        head_dropout: float = 0.40,
        frozen_gnn_embs: Optional[torch.Tensor] = None,  # [18870, 256] float32
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA on ALL 8 layers ──────────────────
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA on Q/K/V of ALL 8 transformer layers (layers 0-7)
        # Proven in node3-2 (0.462) to outperform last-4-only configuration
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
        for _name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Gene symbol character-level CNN encoder ────────────────────────────
        self.symbol_encoder = SymbolEncoder(out_dim=self.SYMBOL_DIM, embed_dim=32)

        # ── Frozen STRING GNN embedding table ─────────────────────────────────
        # Register as buffer: moves to correct device automatically,
        # but does NOT participate in gradient computation
        if frozen_gnn_embs is not None:
            self.register_buffer("gnn_emb_table", frozen_gnn_embs)  # [18870, 256]
        else:
            # Fallback: zero embeddings (should not happen in normal usage)
            self.register_buffer("gnn_emb_table",
                                 torch.zeros(18870, self.PPI_DIM, dtype=torch.float32))

        # Learnable fallback embedding for genes not in STRING GNN (~6.5% of genes)
        self.gnn_fallback = nn.Parameter(torch.zeros(self.PPI_DIM))
        nn.init.trunc_normal_(self.gnn_fallback, std=0.02)

        # 1-layer PPI projection: align frozen STRING embeddings with AIDO.Cell space
        self.ppi_proj = nn.Sequential(
            nn.Linear(self.PPI_DIM, self.PPI_DIM),
            nn.GELU(),
            nn.LayerNorm(self.PPI_DIM),
        )
        nn.init.xavier_uniform_(self.ppi_proj[0].weight)
        nn.init.zeros_(self.ppi_proj[0].bias)

        # ── Prediction head ────────────────────────────────────────────────────
        head_in = self.HEAD_INPUT_DIM  # 832
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )
        # Conservative initialization to stabilize early training
        nn.init.trunc_normal_(self.head[1].weight, std=0.02)
        nn.init.zeros_(self.head[1].bias)
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
        lhs = out.last_hidden_state  # [B, 19266, 256] (includes 2 summary tokens)

        # (a) Global mean-pool over all gene positions (exclude 2 summary tokens)
        gene_emb = lhs[:, :N_GENES_AIDO, :]          # [B, 19264, 256]
        global_emb = gene_emb.mean(dim=1)             # [B, 256]

        # (b) Perturbed-gene positional embedding
        pert_emb = torch.zeros(B, self.HIDDEN_DIM, device=lhs.device, dtype=lhs.dtype)
        valid_aido = gene_pos >= 0
        if valid_aido.any():
            valid_pos = gene_pos[valid_aido]  # [k]
            pert_emb[valid_aido] = lhs[valid_aido, valid_pos, :]
        # Fallback for genes not in AIDO.Cell vocab
        pert_emb[~valid_aido] = global_emb[~valid_aido]

        # Convert backbone features to float32 for head computation
        backbone_feat = torch.cat([global_emb, pert_emb], dim=-1).float()  # [B, 512]

        # (c) Gene symbol character CNN
        sym_feat = self.symbol_encoder(symbol_ids)    # [B, 64] float32

        # (d) Frozen STRING GNN PPI embedding lookup
        # gnn_idx: -1 for genes not in STRING GNN (~6.5% coverage gap)
        ppi_emb = self.gnn_fallback.unsqueeze(0).expand(B, -1).clone()  # [B, 256]
        valid_gnn = gnn_idx >= 0
        if valid_gnn.any():
            valid_gnn_idx = gnn_idx[valid_gnn]  # [k]
            # Lookup from frozen embedding table (no gradient)
            with torch.no_grad():
                ppi_raw = self.gnn_emb_table[valid_gnn_idx]  # [k, 256]
            ppi_emb[valid_gnn] = ppi_raw

        # Project PPI features (trainable linear to adapt frozen embeddings to task)
        ppi_feat = self.ppi_proj(ppi_emb.to(backbone_feat.device))  # [B, 256]

        # Concatenate all 4 feature sources: [B, 832]
        combined = torch.cat([backbone_feat, sym_feat, ppi_feat], dim=-1)

        logits = self.head(combined)                   # [B, 3 * 6640]
        return logits.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro F1, averaged over all N_GENES_OUT genes.

    Matches the evaluation logic in data/calc_metric.py:
      - For each gene, compute F1 over only the classes actually present.
      - Average per-gene F1 scores over all genes.

    y_pred:          [n_samples, 3, n_genes]  (3-class probability distributions)
    y_true_remapped: [n_samples, n_genes]     (labels in {0, 1, 2})
    """
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp_class = y_pred[:, :, g]         # [n_samples, 3]
        yhat = yp_class.argmax(axis=1)     # [n_samples]
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
        head_hidden: int = 384,
        head_dropout: float = 0.40,
        backbone_lr: float = 2e-4,
        head_lr_multiplier: float = 3.0,
        symbol_lr_multiplier: float = 2.0,
        ppi_lr_multiplier: float = 2.5,
        weight_decay: float = 0.03,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        plateau_patience: int = 7,
        plateau_factor: float = 0.5,
        plateau_min_lr: float = 1e-7,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCellPPIDEGModel] = None
        self.criterion: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            # Get frozen STRING GNN embeddings from the datamodule
            frozen_gnn_embs = None
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
                dm = self.trainer.datamodule
                if hasattr(dm, "frozen_gnn_embs") and dm.frozen_gnn_embs is not None:
                    frozen_gnn_embs = dm.frozen_gnn_embs

            self.model = AIDOCellPPIDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                head_hidden=self.hparams.head_hidden,
                head_dropout=self.hparams.head_dropout,
                frozen_gnn_embs=frozen_gnn_embs,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        if stage == "test":
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None:
                if not hasattr(dm, "test_pert_ids") or not dm.test_pert_ids:
                    dm.setup("test")
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols

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

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # NO expression noise augmentation — proven to hurt performance.
        # Clean inputs (all genes=1.0, KO=0.0) provide more stable gradient signals.
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

        # Gather across all DDP ranks
        ap = self.all_gather(lp)
        al = self.all_gather(ll)
        ai = self.all_gather(li)
        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # Compute F1 only on global rank 0 (F1 is NOT additive)
        is_global_zero = self.trainer.global_rank == 0
        if is_global_zero:
            preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
            labels = al.cpu().view(-1, N_GENES_OUT).numpy()
            idxs = ai.cpu().view(-1).numpy()
            _, uniq = np.unique(idxs, return_index=True)
            f1 = compute_deg_f1(preds[uniq], labels[uniq])
            f1_tensor = torch.tensor(f1, dtype=torch.float32, device=self.device)
        else:
            f1_tensor = torch.zeros(1, dtype=torch.float32, device=self.device)

        # Broadcast scalar F1 from rank 0 so all ranks have identical val_f1
        # (required for EarlyStopping and checkpoint monitoring)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(f1_tensor, src=0)
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

        # Gather across all DDP ranks
        ap = self.all_gather(lp)
        ai = self.all_gather(li)
        self._test_preds.clear()
        self._test_indices.clear()

        # Retrieve test metadata from datamodule
        dm = None
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
        all_pert_ids = getattr(dm, "test_pert_ids", None) or self._test_pert_ids
        all_symbols = getattr(dm, "test_symbols", None) or self._test_symbols

        if self.trainer.is_global_zero:
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

            out_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved ({len(rows)} samples) → {out_path}")

    def configure_optimizers(self):
        hp = self.hparams
        # Four parameter groups with different learning rates:
        #   1. backbone LoRA (lowest lr) — fine-tuning pre-trained weights
        #   2. ppi_proj (moderate lr) — adapting frozen PPI features to task
        #   3. symbol_encoder (moderate lr) — learning gene family patterns
        #   4. head (highest lr) — learning the prediction mapping
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        ppi_proj_params = (
            list(self.model.ppi_proj.parameters()) +
            [self.model.gnn_fallback]
        )
        symbol_params = list(self.model.symbol_encoder.parameters())
        head_params = list(self.model.head.parameters())

        backbone_lr = hp.backbone_lr                                    # 2e-4
        ppi_lr = hp.backbone_lr * hp.ppi_lr_multiplier                 # 5e-4
        symbol_lr = hp.backbone_lr * hp.symbol_lr_multiplier           # 4e-4
        head_lr = hp.backbone_lr * hp.head_lr_multiplier               # 6e-4

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": ppi_proj_params, "lr": ppi_lr},
                {"params": symbol_params, "lr": symbol_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=hp.weight_decay,
        )

        # ReduceLROnPlateau on val_loss with increased patience=7
        # Increased from parent's 5 to 7 to tolerate brief val_loss fluctuations
        # that caused premature LR reduction at E10 in the parent run.
        # The parent's LR fired at E10→E11 (brief dip) and E15→E16 (stagnation),
        # which may have destabilized post-peak optimization.
        plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",              # val_loss should decrease
            factor=hp.plateau_factor,   # 0.5
            patience=hp.plateau_patience,   # 7 (increased from parent's 5)
            min_lr=hp.plateau_min_lr,
        )

        return [opt], [
            {
                "scheduler": plateau_sched,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
                "name": "plateau_lr",
            },
        ]

    # ── Checkpoint helpers ───────────────────────────────────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        out = {}
        trainable_keys = {name for name, p in self.named_parameters() if p.requires_grad}
        buffer_keys = {name for name, _ in self.named_buffers()}
        expected_keys = trainable_keys | buffer_keys

        for k, v in full.items():
            rel_key = k[len(prefix):] if k.startswith(prefix) else k
            if rel_key in expected_keys:
                out[k] = v

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%), "
            f"plus {total_buffers} buffer values"
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
# Top-K Checkpoint Averaging (FIXED VERSION)
# ──────────────────────────────────────────────────────────────────────────────
def average_top_k_checkpoints(
    model_module: DEGLightningModule,
    checkpoint_cb: ModelCheckpoint,
    top_k: int = 3,
) -> bool:
    """
    Load the top-k saved checkpoints and average their trainable parameters
    element-wise in the model_module.

    CRITICAL FIX vs parent node:
    The parent node failed with "No backend type associated with device type cpu"
    when calling torch.stack().mean() on CPU-loaded bfloat16 tensors.

    Fix:
    1. Load checkpoints to the GPU device (not CPU)
    2. Cast ALL tensors to float32 before stacking/averaging
    3. Use element-wise summation (/ count) instead of torch.stack() as a fallback
       to avoid potential memory issues with large stacked tensors

    Returns True on success, False if averaging failed (caller should use best single ckpt).
    """
    # Get the best k checkpoint paths sorted by score (descending)
    best_k = checkpoint_cb.best_k_models  # Dict[path, score]
    if not best_k:
        print("[CheckpointAveraging] No checkpoints found, skipping averaging.")
        return False

    # Sort by score descending, take top-k
    sorted_ckpts = sorted(best_k.items(), key=lambda x: float(x[1]), reverse=True)
    selected = sorted_ckpts[:min(top_k, len(sorted_ckpts))]
    ckpt_paths = [p for p, _ in selected]
    ckpt_scores = [float(s) for _, s in selected]

    print(f"[CheckpointAveraging] Averaging {len(ckpt_paths)} checkpoints:")
    for path, score in zip(ckpt_paths, ckpt_scores):
        print(f"  {Path(path).name} (val_f1={score:.4f})")

    if len(ckpt_paths) == 1:
        # Only one checkpoint available — just load it
        print("[CheckpointAveraging] Only 1 checkpoint available, loading single best.")
        # FIXED: use cuda device for loading
        cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(ckpt_paths[0], map_location=cuda_device)
        sd = state.get("state_dict", state)
        # Cast to float32 for safe loading
        sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
        model_module.load_state_dict(sd, strict=False)
        return True

    # Determine target device for averaging — use GPU to avoid CPU bfloat16 issues
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Load all checkpoints to GPU and extract state dicts
        # FIXED: map_location=cuda_device instead of "cpu"
        state_dicts = []
        for ckpt_path in ckpt_paths:
            state = torch.load(ckpt_path, map_location=cuda_device)
            sd = state.get("state_dict", state)
            # FIXED: Cast ALL tensors to float32 before stacking
            # This resolves "No backend type associated with device type cpu"
            # which occurs with bfloat16 tensors on CPU
            sd_f32 = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
            state_dicts.append(sd_f32)

        # Average parameters element-wise across checkpoints
        # Only average keys present in all checkpoints
        all_keys = set(state_dicts[0].keys())
        for sd in state_dicts[1:]:
            all_keys &= set(sd.keys())

        # Use element-wise summation to avoid torch.stack memory overhead
        averaged_sd = {}
        n_ckpts = len(state_dicts)
        for key in all_keys:
            # Sum across all checkpoints, then divide
            avg_tensor = state_dicts[0][key].clone()
            for sd in state_dicts[1:]:
                avg_tensor = avg_tensor + sd[key]
            averaged_sd[key] = avg_tensor / n_ckpts

        # Load averaged state dict into model
        print(f"[CheckpointAveraging] Averaged {len(averaged_sd)} parameter tensors "
              f"using device={cuda_device}.")
        model_module.load_state_dict(averaged_sd, strict=False)
        print("[CheckpointAveraging] Averaged model loaded successfully.")

        # Free GPU memory used by state dicts
        del state_dicts, averaged_sd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"[CheckpointAveraging] Averaging failed ({e}). "
              f"Will fall back to best single checkpoint.")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 2-3-1-1-1-1-1-1-1-1: AIDO.Cell-10M + LoRA all-8-layers + Symbol CNN + "
                    "Frozen STRING GNN PPI + FIXED Top-3 Checkpoint Averaging + "
                    "weight_decay=0.03 (reverted) + plateau_patience=7"
    )
    # Data
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--num-workers", type=int, default=4)
    # Batch / training
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--early-stopping-patience", type=int, default=25,
                   help="Early stopping patience on val_f1 (max mode). "
                        "25 epochs: parent found this is optimal (best at E14, stops ~E39).")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    # LoRA
    p.add_argument("--lora-r", type=int, default=4,
                   help="LoRA rank for all 8 AIDO.Cell transformer layers")
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    # Head (proven settings from grandparent / parent)
    p.add_argument("--head-hidden", type=int, default=384,
                   help="Head hidden dimension (384 proven optimal)")
    p.add_argument("--head-dropout", type=float, default=0.40,
                   help="Head dropout (0.40 proven effective)")
    # Optimizer
    p.add_argument("--backbone-lr", type=float, default=2e-4)
    p.add_argument("--head-lr-multiplier", type=float, default=3.0)
    p.add_argument("--symbol-lr-multiplier", type=float, default=2.0)
    p.add_argument("--ppi-lr-multiplier", type=float, default=2.5)
    p.add_argument("--weight-decay", type=float, default=0.03,
                   help="Weight decay: reverted to 0.03 from parent's 0.05. "
                        "Feedback confirmed 0.05 caused training instability. "
                        "Grandparent at 0.03 had identical peak val_f1=0.4631 "
                        "with better val_loss trajectory.")
    # Loss
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    # LR Scheduler
    p.add_argument("--plateau-patience", type=int, default=7,
                   help="ReduceLROnPlateau patience on val_loss (min mode). "
                        "Increased from parent's 5 to 7 to avoid premature LR reduction. "
                        "Parent fired at E10→E11 on a brief val_loss dip, disrupting "
                        "late-stage optimization. Patience=7 tolerates short fluctuations.")
    p.add_argument("--plateau-factor", type=float, default=0.5,
                   help="ReduceLROnPlateau factor (0.5 proven to enable multi-stage optimization)")
    p.add_argument("--plateau-min-lr", type=float, default=1e-7)
    # Checkpoint averaging
    p.add_argument("--top-k-avg", type=int, default=3,
                   help="Number of top checkpoints to average for test prediction. "
                        "Default: 3 (proven in node2-2-3-1-1-1 for tree-best 0.4655). "
                        "Set to 1 to disable averaging (single best checkpoint).")
    # Debug
    p.add_argument("--debug-max-step", "--debug_max_step", dest="debug_max_step",
                   type=int, default=None)
    p.add_argument("--fast-dev-run", "--fast_dev_run", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # seed=7: proven best in node2-3-1 lineage
    # seed=7 produced F1=0.4573 (grandparent) and F1=0.4631 (great-grandparent) — lineage best
    pl.seed_everything(7)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False

    # Debug mode: limit training batches for quick iteration
    if args.debug_max_step is not None:
        max_steps = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val = 1.0   # Use full validation set for meaningful debug check
        limit_test = 1.0
        val_check_interval = 1.0
        max_epochs_eff = 2
    else:
        max_steps = -1
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        val_check_interval = args.val_check_interval
        max_epochs_eff = args.max_epochs

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node2-3-1-1-1-1-1-1-1-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=3, save_last=True,  # save top-3 for checkpoint averaging
    )
    # EarlyStopping on val_f1 (primary metric) — patience=25
    # Parent confirmed best at E14; patience=25 → stop ~E39, saving wasted epochs.
    early_stop_cb = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.early_stopping_patience,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    if args.debug_max_step is not None:
        callbacks = [checkpoint_cb, lr_monitor, progress_bar]
    else:
        callbacks = [checkpoint_cb, early_stop_cb, lr_monitor, progress_bar]

    # ── Loggers ───────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=180)),
        precision="bf16-mixed",
        max_epochs=max_epochs_eff,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=False,   # FlashAttention is non-deterministic
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,  # Standard clipping for AIDO.Cell-10M
    )

    # ── Datamodule & LightningModule ──────────────────────────────────────────
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
        backbone_lr=args.backbone_lr,
        head_lr_multiplier=args.head_lr_multiplier,
        symbol_lr_multiplier=args.symbol_lr_multiplier,
        ppi_lr_multiplier=args.ppi_lr_multiplier,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        plateau_min_lr=args.plateau_min_lr,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── FIXED Top-K Checkpoint Averaging ─────────────────────────────────────
    # After training, average the top-3 checkpoints by val_f1 to reduce variance.
    #
    # CRITICAL FIX: The parent node (node2-3-1-1-1-1-1-1-1) had this code but it
    # FAILED with "No backend type associated with device type cpu" because:
    #   - Checkpoints were loaded with map_location="cpu"
    #   - bfloat16 tensors do NOT have a CPU computation backend in PyTorch
    #   - torch.stack().mean() requires a computation backend, which bf16 lacks on CPU
    #
    # FIX: Load checkpoints to the CUDA device, cast all tensors to float32 FIRST.
    # This correctly enables all tensor operations (stacking, averaging) to succeed.
    #
    # DDP strategy:
    #   1. Each rank independently loads the same checkpoint files → identical result
    #   2. The averaged model state is applied in-memory on each rank
    #   3. No ckpt_path is passed to test() so Lightning uses the in-memory averaged model
    #   4. A torch.distributed.barrier() synchronizes ranks after averaging
    #
    # Skip averaging in debug/fast_dev_run modes
    if not args.fast_dev_run and args.debug_max_step is None and args.top_k_avg > 1:
        # Load best single checkpoint first to initialize model to a known good state
        best_ckpt = checkpoint_cb.best_model_path
        if best_ckpt:
            print(f"[CheckpointAveraging] Initializing from best checkpoint: {Path(best_ckpt).name}")
            cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
            state = torch.load(best_ckpt, map_location=cuda_device)
            sd = state.get("state_dict", state)
            # Cast to float32 for safe loading
            sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
            model_module.load_state_dict(sd, strict=False)
            print("[CheckpointAveraging] Best checkpoint loaded as initialization.")

        # Attempt top-k averaging with FIXED implementation
        avg_success = average_top_k_checkpoints(
            model_module=model_module,
            checkpoint_cb=checkpoint_cb,
            top_k=args.top_k_avg,
        )

        if avg_success:
            # Synchronize all DDP ranks (ensures all ranks have identical averaged weights)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # Move each parameter to GPU first (bf16 tensors on CPU fail with broadcast)
                cuda_device_idx = int(os.environ.get("LOCAL_RANK", "0"))
                for param in model_module.parameters():
                    if param.device.type == "cpu":
                        param.data = param.data.to(f"cuda:{cuda_device_idx}")
                torch.distributed.barrier()
                print("[CheckpointAveraging] Synchronized averaged weights across all DDP ranks.")

            # Run test with in-memory averaged model (no ckpt_path)
            print("[CheckpointAveraging] Running test with averaged model (in-memory).")
            test_results = trainer.test(model_module, datamodule=datamodule)
        else:
            # Fallback: use standard best single checkpoint
            print("[CheckpointAveraging] Falling back to best single checkpoint.")
            test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")
    else:
        # Debug mode or averaging disabled: use standard single-best checkpoint
        if args.fast_dev_run or args.debug_max_step is not None:
            test_results = trainer.test(model_module, datamodule=datamodule)
        else:
            # args.top_k_avg == 1: load best single checkpoint
            test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
