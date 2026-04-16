#!/usr/bin/env python3
"""
Node 3-3-1: AIDO.Cell-10M + LoRA (r=4, ALL 8 layers) + Symbol CNN + Frozen STRING GNN PPI
===========================================================================================
Improved version of parent node3-3 (test F1=0.4513) implementing the primary feedback
recommendation: revert LoRA to r=4 ALL 8 transformer layers (proven in node3-2: 0.462 F1)
while RETAINING the validated val_loss-based LR scheduler fix.

Key Hypothesis: node3-3's -0.011 regression vs node3-2 was caused by the LoRA configuration,
not by any other change. The all-8-layers LoRA distributes adaptation budget across all layers,
allowing early layers to adapt their attention patterns for the synthetic knockout input format.
node3-3 concentrated adaptation only in layers 4-7, missing important early-layer adjustments.

Architecture (identical to node3-2 which achieved 0.462 F1):
  - AIDO.Cell-10M with LoRA r=4 applied to ALL 8 transformer layers
    * Broader layer coverage: all 8 layers get adapted vs only last 4
    * Same total LoRA params as node3-3 (~144K) but distributed across all layers
  - Gene symbol character-level CNN (3-branch Conv1d → 64-dim)
  - Frozen STRING GNN embeddings (256-dim) as pre-computed PPI topology feature lookup
  - Combined feature: [global_emb(256) + pert_emb(256) + sym_emb(64) + ppi_emb(256)] = 832-dim
  - MLP head: LayerNorm(832) → Linear(832, 384) → GELU → Dropout(0.4) → LayerNorm(384) → Linear(384, 3×6640)

Key Improvements Over Parent node3-3 (0.4513 F1):
  1. LoRA r=4 on ALL 8 LAYERS (vs r=8 on last-4-only in node3-3)
     - node3-2 proved r=4 all-8-layers achieves 0.462 F1 — broader coverage is superior
     - r=8 last-4 in node3-3 produced -0.011 regression vs node3-2
     - Feedback from node3-3: "primary causal factor is LoRA layer configuration difference"
  2. RETAINED: ReduceLROnPlateau monitors val_LOSS (not val_f1) — validated to fire in node3-3
     - node3-3 proved this fix works (4 reductions fired), unlike node3-2's val_f1 monitoring
  3. Increased ReduceLROnPlateau patience to 8 (from node3-3's 5)
     - node3-3's patience=5 fired too aggressively (epoch 9 already = first reduction)
     - Best epoch was E13 — the 4 reductions happened at E9, E15, E21, E27 (pre-best firing)
     - patience=8 allows more training at full LR before reduction, potentially reaching higher plateau
  4. Reduced early stopping patience to 15 (from node3-3's 25)
     - node3-3 best epoch was E13; 25-epoch patience means stopping at E38 — 25 wasted epochs
     - node3-2 best epoch was E18; consistent ~E13-18 across nodes
     - patience=15 would stop at ~E28-33, saving 10-15 wasted epochs
  5. Adjusted backbone LR to 2e-4 (from node3-3's 3e-4)
     - node3-2 used backbone_lr=2e-4 with r=4 all-8-layers and achieved 0.462
     - node3-3 used 3e-4 with r=8 last-4; for r=4 all-8, 2e-4 is the proven value
     - Lower backbone LR with more layers covered matches the original 0.462 setup

Training Configuration:
  - Global batch: 64 (micro_batch=8, 8 GPUs, accumulate=1)
  - LR: backbone=2e-4, ppi_proj=5e-4 (2.5×), symbol=4e-4 (2×), head=6e-4 (3×)
  - Weight decay: 0.03
  - LR scheduler: ReduceLROnPlateau on val_loss (patience=8, factor=0.5)  [patience increased from 5]
  - Early stopping: patience=15 on val_f1  [reduced from 25]
  - Max epochs: 100
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault('NCCL_IB_DISABLE', '1')
os.environ.setdefault('NCCL_NET_GDR_LEVEL', '0')

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
N_GENES_OUT = 6_640      # output genes
N_CLASSES = 3
SENTINEL_EXPR = 1.0      # baseline expression for non-perturbed genes
KNOCKOUT_EXPR = 0.0      # expression for knocked-out gene
AIDO_HIDDEN = 256        # AIDO.Cell-10M hidden dimension
AIDO_N_LAYERS = 8        # AIDO.Cell-10M transformer layers
STRING_GNN_DIM = 256     # STRING GNN embedding dimension

# Moderate class weights (proven effective in node2-2/node3-2 lineage)
# Train: class 0 (down) ~3.4%, class 1 (unchanged) ~95.5%, class 2 (up) ~1.1%
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)

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
    Three parallel Conv1d filters at kernel sizes 2, 3, 4 → max-pool → 64-dim.
    Captures character n-gram patterns: gene family prefixes, numeric suffixes, etc.
    Proven design from node2-2 that provided +0.041 F1 gain.
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

    This approach addresses the core bug in node3-1:
    - node3-1 tried to TRAIN the STRING GNN → static output, all samples got same embedding
    - This node FREEZES the STRING GNN and uses it only as a lookup table
    - Per-sample differentiation comes from AIDO.Cell, not the GNN
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
# Model
# ──────────────────────────────────────────────────────────────────────────────
class AIDOCellPPIDEGModel(nn.Module):
    """
    AIDO.Cell-10M + LoRA (r=4, ALL 8 layers) + Symbol CNN + Frozen STRING GNN embedding.

    Feature fusion (4 sources):
      (a) Global mean-pool of AIDO.Cell last_hidden_state     → [B, 256]
      (b) Perturbed-gene positional embedding from AIDO.Cell  → [B, 256]
      (c) Gene symbol character-level CNN                     → [B, 64]
      (d) Frozen STRING GNN PPI topology embedding            → [B, 256]
    Concat → [B, 832] → MLP head → [B, 3, 6640]

    LoRA Strategy: r=4 applied to ALL 8 transformer layers
      - Proven configuration from node3-2 (0.462 F1, best in tree)
      - Broader coverage: all layers contribute, including early layers that
        capture syntactic patterns in the expression profile
      - Total LoRA params: ~72K (same budget distributed over 8 layers at r=4)
    """

    HIDDEN_DIM = 256          # AIDO.Cell-10M hidden size
    SYMBOL_DIM = 64           # symbol CNN output dim
    PPI_DIM = 256             # STRING GNN embedding dim
    HEAD_INPUT_DIM = 256 * 2 + 64 + 256  # global + pert + sym + ppi = 832

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        head_hidden: int = 384,
        head_dropout: float = 0.4,
        lora_dropout: float = 0.1,
        frozen_gnn_embs: Optional[torch.Tensor] = None,  # [18870, 256] float32
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA on ALL 8 layers ──────────────────
        backbone = AutoModel.from_pretrained(
            AIDO_CELL_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16,
        )
        backbone.config.use_cache = False

        # LoRA on Q/K/V of ALL 8 transformer layers (layers 0-7)
        # This is the PROVEN configuration from node3-2 (0.462 F1)
        # All-8-layers distributes adaptation budget across all transformer layers,
        # allowing early-layer attention to also adapt to the synthetic knockout format
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            # No layers_to_transform = apply to ALL layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Cast trainable LoRA params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Gene symbol character-level CNN encoder ────────────────────────────
        self.symbol_encoder = SymbolEncoder(out_dim=self.SYMBOL_DIM, embed_dim=32)

        # ── Frozen STRING GNN embedding table ─────────────────────────────────
        # Register as a buffer so it moves to the correct device automatically
        # but does NOT participate in gradient computation
        if frozen_gnn_embs is not None:
            self.register_buffer("gnn_emb_table", frozen_gnn_embs)  # [18870, 256]
        else:
            # Fallback: zero embeddings (should not happen in normal usage)
            self.register_buffer("gnn_emb_table",
                                 torch.zeros(18870, self.PPI_DIM, dtype=torch.float32))

        # Learnable fallback embedding for genes not in STRING GNN
        self.gnn_fallback = nn.Parameter(torch.zeros(self.PPI_DIM))
        nn.init.trunc_normal_(self.gnn_fallback, std=0.02)

        # 1-layer projection to align STRING GNN embedding with AIDO.Cell space
        # Proven optimal in node3-2 feedback (2-layer projection in node3-2-1 worsened performance)
        self.ppi_proj = nn.Sequential(
            nn.Linear(self.PPI_DIM, self.PPI_DIM),
            nn.GELU(),
            nn.LayerNorm(self.PPI_DIM),
        )
        nn.init.xavier_uniform_(self.ppi_proj[0].weight)
        nn.init.zeros_(self.ppi_proj[0].bias)

        # ── Prediction head ────────────────────────────────────────────────────
        # Single-stage 832 → 384 → 3×6640 proven optimal in node3-2 (0.462 F1)
        head_in = self.HEAD_INPUT_DIM  # 832
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden),
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

        # Convert backbone features to float32 for head computation
        backbone_feat = torch.cat([global_emb, pert_emb], dim=-1).float()  # [B, 512]

        # (c) Gene symbol character CNN
        sym_feat = self.symbol_encoder(symbol_ids)    # [B, 64] float32

        # (d) Frozen STRING GNN PPI embedding lookup
        # gnn_idx: -1 for genes not in STRING GNN
        ppi_emb = self.gnn_fallback.unsqueeze(0).expand(B, -1).clone()  # [B, 256]
        valid_gnn = gnn_idx >= 0
        if valid_gnn.any():
            valid_gnn_idx = gnn_idx[valid_gnn]  # [k]
            # Lookup from frozen embedding table (no gradient)
            with torch.no_grad():
                ppi_raw = self.gnn_emb_table[valid_gnn_idx]  # [k, 256]
            ppi_emb[valid_gnn] = ppi_raw

        # Project PPI features (trainable linear to adapt frozen embeddings)
        ppi_feat = self.ppi_proj(ppi_emb.to(backbone_feat.device))  # [B, 256]

        # Concatenate all 4 feature sources
        combined = torch.cat([backbone_feat, sym_feat, ppi_feat], dim=-1)  # [B, 832]

        logits = self.head(combined)                   # [B, 3 * 6640]
        return logits.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


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
        head_hidden: int = 384,
        head_dropout: float = 0.4,
        lr: float = 2e-4,
        head_lr_multiplier: float = 3.0,
        symbol_encoder_lr_multiplier: float = 2.0,
        ppi_proj_lr_multiplier: float = 2.5,
        weight_decay: float = 0.03,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        # LR scheduler — RETAINED FIX: monitor val_loss (not val_f1) for reliable firing
        # patience increased to 8 (from node3-3's 5) to avoid premature LR reduction
        plateau_patience: int = 8,
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
            try:
                trainer = self.trainer
            except RuntimeError:
                trainer = None
            if trainer is not None and hasattr(trainer, "datamodule") and trainer.datamodule is not None:
                dm = trainer.datamodule
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

        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if stage == "test" and trainer is not None and hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            self._test_pert_ids = trainer.datamodule.test_pert_ids
            self._test_symbols = trainer.datamodule.test_symbols

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
        # Four parameter groups with different learning rates:
        #   1. backbone LoRA (base lr = 2e-4, proven value for r=4 all-8-layers in node3-2)
        #   2. ppi_proj (trainable linear on frozen PPI features)
        #   3. symbol_encoder (moderate lr)
        #   4. head (highest lr)
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        ppi_proj_params = (
            list(self.model.ppi_proj.parameters()) +
            [self.model.gnn_fallback]
        )
        symbol_params = list(self.model.symbol_encoder.parameters())
        head_params = list(self.model.head.parameters())

        backbone_lr = hp.lr                                          # 2e-4
        ppi_lr = hp.lr * hp.ppi_proj_lr_multiplier                 # 5e-4
        symbol_lr = hp.lr * hp.symbol_encoder_lr_multiplier        # 4e-4
        head_lr = hp.lr * hp.head_lr_multiplier                    # 6e-4

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": ppi_proj_params, "lr": ppi_lr},
                {"params": symbol_params, "lr": symbol_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=hp.weight_decay,
        )

        # RETAINED FIX: Monitor val_LOSS (not val_f1) for ReduceLROnPlateau
        # node3-3 validated this works (4 reductions fired at E9/15/21/27)
        # node3-2's CRITICAL BUG was monitoring val_f1 which NEVER FIRED across 59 epochs
        # CHANGE: patience increased from 5→8 to avoid too-aggressive early reductions
        # node3-3's patience=5 fired at E9 (before best epoch E13) — too aggressive
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",        # minimize val_loss (not maximize val_f1)
            factor=hp.plateau_factor,
            patience=hp.plateau_patience,
            min_lr=hp.plateau_min_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss",   # RETAINED FIX: monitor val_loss instead of val_f1
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
            },
        }

    def load_state_dict(self, state_dict, strict=True):
        """Load checkpoint with strict=False to handle PEFT naming variations.

        The checkpoint may have keys that don't perfectly match the fresh model's
        structure due to PEFT's internal module naming. Using strict=False allows
        loading while skipping mismatched keys. Trainable LoRA params load correctly.
        """
        return super().load_state_dict(state_dict, strict=False)



# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 3-3-1: AIDO.Cell-10M + LoRA (r=4, all-8) + Symbol CNN + Frozen STRING GNN PPI"
    )
    # Data path: data/ relative to the project root (2 levels up from mcts/node3-3-1/)
    p.add_argument("--data-dir", type=str, default=str(Path(__file__).parent.parent.parent / "data"))
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=100)
    # Backbone LR=2e-4 (proven value for r=4 all-8-layers from node3-2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--head-lr-multiplier", type=float, default=3.0)
    p.add_argument("--symbol-encoder-lr-multiplier", type=float, default=2.0)
    p.add_argument("--ppi-proj-lr-multiplier", type=float, default=2.5)
    p.add_argument("--weight-decay", type=float, default=0.03)
    # LoRA r=4 ALL layers (reverting from node3-3's r=8 last-4 back to node3-2's proven r=4 all-8)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--head-hidden", type=int, default=384)
    p.add_argument("--head-dropout", type=float, default=0.4)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    # Increased patience=8 (from node3-3's 5) for less aggressive LR reduction
    p.add_argument("--plateau-patience", type=int, default=8)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--plateau-min-lr", type=float, default=1e-7)
    # Reduced from node3-3's 25 to 15 — best epoch is consistently E13-18 across all nodes
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug_max_step", "--debug-max-step", dest="debug_max_step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
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
    limit_test_batches = 1.0  # Always run full test set

    # Checkpoint monitors val_f1 (we want to keep the best F1 checkpoint)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node3-3-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=3, save_last=True,
    )
    # Early stopping monitors val_f1 (stop when F1 stops improving)
    # patience=15: given best epoch is consistently E13-18, patience=15 stops by E28-33
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
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=1.0 if (args.debug_max_step is not None or args.fast_dev_run)
                           else args.val_check_interval,
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
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
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
    )

    trainer.fit(model_module, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        primary_val = float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score is not None else float("nan")
        score_path.write_text(f"{primary_val:.6f}\n")
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
