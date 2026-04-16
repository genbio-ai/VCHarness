#!/usr/bin/env python3
"""
Node 2-3-1-1-1: AIDO.Cell-10M + LoRA (r=4, ALL 8 layers) + Symbol CNN
               + Frozen STRING GNN PPI Embeddings + Checkpoint Averaging
               + LR Warmup + Increased Head Capacity (512) + Factor=0.3
=======================================================================
Key improvements over parent node2-3-1-1 (test F1=0.4555):

1. TOP-3 CHECKPOINT AVERAGING (MOST RECOMMENDED by parent feedback)
   - Parent feedback explicitly recommends averaging epochs 14/12/9 for +0.003 F1
   - Load top-3 checkpoints by val_f1, average trainable params element-wise
   - Use averaged model for test inference; reduces sensitivity to best-epoch selection
   - Parent's near-zero val-test gap (0.0005) means safe to average without overfitting risk

2. INCREASED HEAD CAPACITY: head_hidden 384→512, dropout 0.4→0.35
   - Parent feedback: "Possibly +0.002-0.005 if regularization simultaneously adjusted"
   - 832→512 intermediate provides richer feature combinations before 512→19920 expansion
   - Dropout reduced to 0.35 to prevent under-training from the larger intermediate layer
   - Compensated by tighter early stopping (patience 15→12) for larger-capacity head

3. LR WARMUP: 2-epoch linear warmup before ReduceLROnPlateau
   - Parent's epoch 0 val_f1=0.383 shows large initial gradient updates
   - Linear warmup from 10% to 100% of target LR over 2 epochs
   - Stabilizes early training, especially for LoRA parameters in the backbone

4. MORE AGGRESSIVE LR REDUCTION: plateau_factor 0.5→0.3
   - Parent feedback recommends: "factor of 0.3 might produce cleaner convergence"
   - Parent's 4 reductions at factor=0.5: 2e-4 → 1.25e-5 (16× decay)
   - With factor=0.3: 2e-4 → 1.6e-6 after 4 reductions (125× decay)
   - More decisive drops may allow model to settle into better convergence basins

5. DIFFERENT RANDOM SEED (42 vs parent's 0)
   - Parent analysis: "0.007 gap to node3-2 is stochastic variance, not systematic flaw"
   - node3-2's 0.462 likely from favorable seed initialization
   - Seed 42 explores different loss landscape trajectory

6. TIGHTER EARLY STOPPING (patience 15→12)
   - Larger head (512-dim) may overfit earlier than parent's 384-dim head
   - 12 epochs is sufficient for 2-3 LR reductions to fire post-peak
   - Reduces wasted computation if overfitting onset is earlier

Architecture (unchanged from parent):
  - AIDO.Cell-10M: r=4 LoRA on all 8 transformer layers (Q/K/V), global mean-pool + pert_emb
  - Symbol CNN: 3-branch char-level Conv1d → 64-dim
  - STRING GNN: frozen inference → static [18870, 256] table → 1-layer projection → 256-dim
  - Head: LayerNorm(832) → Linear(832→512) → GELU → Dropout(0.35) → LayerNorm(512) → Linear(512→19920)
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
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)

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

    Key insight from node3-2 (test F1=0.462, tree best):
    - Freezing the STRING GNN and using it as a lookup table avoids the sample-invariant
      bug in node3-1 where GNN outputs were identical for all samples
    - Per-sample differentiation comes from AIDO.Cell's expression profile
    - STRING GNN provides static PPI topology features (network position, interactions)
    - This adds genuine biological signal that breaks the ~0.447 synthetic-input ceiling
    """
    import json as _json
    from transformers import AutoModel as _AutoModel

    model_dir = Path(string_gnn_dir)
    node_names = _json.loads((model_dir / "node_names.json").read_text())
    ensg_to_gnn_idx = {n: i for i, n in enumerate(node_names) if n.startswith("ENSG")}

    graph = torch.load(model_dir / "graph_data.pt", map_location="cpu")
    edge_index = graph["edge_index"]
    edge_weight = graph.get("edge_weight", None)

    # Load STRING GNN and run inference on CPU to avoid GPU memory issues
    gnn_model = _AutoModel.from_pretrained(model_dir, trust_remote_code=True)
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

    Combined: [B, 832] → MLP head (512-dim, improved from parent's 384-dim) → [B, 3, 6640]

    LoRA Strategy: r=4 applied to ALL 8 transformer layers
      - Proven in node3-2 (F1=0.462, tree best) to outperform r=8 last-4-only
      - Broader coverage allows global attention adaptation for this task
      - Lower rank (r=4) distributes regularization uniformly across all layers
      - Total LoRA params: ~72K (suitable for 1,500 training samples)
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
        head_hidden: int = 512,
        head_dropout: float = 0.35,
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
        # Proven in node3-2 (1-layer) and node3-2-1-1 to be optimal
        # (2-layer projection in node3-2-1 added noise without benefit)
        self.ppi_proj = nn.Sequential(
            nn.Linear(self.PPI_DIM, self.PPI_DIM),
            nn.GELU(),
            nn.LayerNorm(self.PPI_DIM),
        )
        nn.init.xavier_uniform_(self.ppi_proj[0].weight)
        nn.init.zeros_(self.ppi_proj[0].bias)

        # ── Prediction head ────────────────────────────────────────────────────
        # Single-stage head: 832 → 512 → 3×6640
        # Increased from parent's 384-dim per feedback recommendation (+0.002-0.005 expected)
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
        head_hidden: int = 512,
        head_dropout: float = 0.35,
        backbone_lr: float = 2e-4,
        head_lr_multiplier: float = 3.0,
        symbol_lr_multiplier: float = 2.0,
        ppi_lr_multiplier: float = 2.5,
        weight_decay: float = 0.03,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        plateau_patience: int = 5,
        plateau_factor: float = 0.3,
        plateau_min_lr: float = 1e-7,
        warmup_epochs: int = 2,
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
        is_global_zero = getattr(self.trainer, "is_global_zero", True)
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

        # LR schedule: linear warmup for warmup_epochs, then ReduceLROnPlateau on val_loss.
        #
        # Implementation: Use a LambdaLR warmup scheduler as a SequentialLR combined
        # with ReduceLROnPlateau. However, SequentialLR doesn't support ReduceLROnPlateau
        # directly. We use a workaround: a custom scheduler that wraps both.
        #
        # Simple approach: only use ReduceLROnPlateau, but scale initial LR by warmup
        # factor in the first warmup_epochs via on_train_epoch_start hook.
        # We use LambdaLR for warmup only, and report val_loss as usual.
        # After warmup, we switch to ReduceLROnPlateau semantics by using
        # ChainedScheduler pattern.
        #
        # Cleanest approach for Lightning: use warmup LambdaLR for first N epochs,
        # then switch to ReduceLROnPlateau via conditional logic in configure_optimizers.
        # Lightning v2.5+ supports multiple schedulers with different intervals.

        warmup_epochs = hp.warmup_epochs

        # Warmup scheduler: linearly scale LR from 10% to 100% over warmup_epochs
        def warmup_fn(epoch: int) -> float:
            if epoch < warmup_epochs:
                return 0.1 + 0.9 * (epoch / max(warmup_epochs, 1))
            return 1.0

        warmup_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=warmup_fn)

        # CRITICAL FIX (from parent): monitor val_loss (not val_f1)
        # node3-2's scheduler NEVER fired because val_f1 oscillates ±0.005 within patience
        # val_loss shows monotonically increasing trend post-best-epoch → reliable signal
        # factor=0.3 is more decisive than parent's 0.5 (per feedback recommendation)
        plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",              # val_loss should decrease
            factor=hp.plateau_factor,   # 0.3 (more aggressive than parent's 0.5)
            patience=hp.plateau_patience,
            min_lr=hp.plateau_min_lr,
        )

        # Return both schedulers: warmup runs every epoch for warmup_epochs,
        # then plateau runs every epoch monitoring val_loss.
        # Lightning will call both schedulers; the warmup scheduler will be a no-op
        # after warmup_epochs (returns multiplier=1.0), while plateau scheduler
        # handles val_loss-based reduction.
        # Note: warmup_sched step is called every epoch (interval='epoch')
        # and is controlled by the lambda function to be a no-op after warmup.
        return [opt], [
            {
                "scheduler": warmup_sched,
                "interval": "epoch",
                "frequency": 1,
                "name": "warmup_lr",
            },
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
# Checkpoint Averaging Utility
# ──────────────────────────────────────────────────────────────────────────────
def average_top_k_checkpoints(
    model_module: DEGLightningModule,
    checkpoint_cb: ModelCheckpoint,
    top_k: int = 3,
    device: str = "cpu",
) -> bool:
    """
    Average the top-k checkpoints by val_f1 and load the averaged state into model_module.

    This is the primary improvement over the parent node:
    - Parent feedback explicitly recommends averaging epochs ~14/12/9 for +0.003 F1
    - Smooths parameter space to reduce sensitivity to best-epoch selection
    - Safe due to near-zero val-test gap in parent (strong generalization)

    Returns True if averaging was performed, False if fewer than 2 checkpoints available.
    """
    best_k_models = checkpoint_cb.best_k_models
    if not best_k_models or len(best_k_models) < 2:
        print("[CheckpointAveraging] Fewer than 2 checkpoints available, skipping averaging.")
        return False

    # Sort by score (val_f1, higher is better) — top_k models
    sorted_paths = sorted(best_k_models.items(), key=lambda x: float(x[1]), reverse=True)
    top_paths = sorted_paths[:min(top_k, len(sorted_paths))]
    print(f"[CheckpointAveraging] Averaging top-{len(top_paths)} checkpoints:")
    for path, score in top_paths:
        print(f"  {path} (val_f1={float(score):.4f})")

    # Load and average state dicts
    avg_state: Dict[str, torch.Tensor] = {}
    n = len(top_paths)
    for ckpt_path, _ in top_paths:
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device)
            state = ckpt.get("state_dict", ckpt)
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if k not in avg_state:
                        avg_state[k] = v.float().clone() / n
                    else:
                        avg_state[k] = avg_state[k] + v.float() / n
        except Exception as e:
            print(f"[CheckpointAveraging] Warning: Failed to load {ckpt_path}: {e}")
            return False

    if not avg_state:
        print("[CheckpointAveraging] No valid state dicts loaded, skipping averaging.")
        return False

    # Load averaged state into the model
    try:
        model_module.load_state_dict(avg_state, strict=False)
        print(f"[CheckpointAveraging] Successfully loaded averaged state from top-{n} checkpoints.")
        return True
    except Exception as e:
        print(f"[CheckpointAveraging] Warning: Failed to load averaged state: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 2-3-1-1-1: AIDO.Cell-10M + LoRA all-8-layers + Symbol CNN + "
                    "Frozen STRING GNN PPI + Top-3 Checkpoint Averaging + LR Warmup"
    )
    # Data
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--num-workers", type=int, default=4)
    # Batch / training
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--early-stopping-patience", type=int, default=12,
                   help="Early stopping patience on val_f1 (max mode). "
                        "Reduced from 15 to 12 to account for larger head capacity (512-dim).")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    # LoRA
    p.add_argument("--lora-r", type=int, default=4,
                   help="LoRA rank for all 8 AIDO.Cell transformer layers")
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    # Head (increased from parent's 384/0.4)
    p.add_argument("--head-hidden", type=int, default=512,
                   help="Head hidden dimension (increased from parent's 384; feedback: +0.002-0.005)")
    p.add_argument("--head-dropout", type=float, default=0.35,
                   help="Head dropout (reduced from parent's 0.4 to compensate capacity increase)")
    # Optimizer
    p.add_argument("--backbone-lr", type=float, default=2e-4)
    p.add_argument("--head-lr-multiplier", type=float, default=3.0)
    p.add_argument("--symbol-lr-multiplier", type=float, default=2.0)
    p.add_argument("--ppi-lr-multiplier", type=float, default=2.5)
    p.add_argument("--weight-decay", type=float, default=0.03)
    # Loss
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    # LR Scheduler (more aggressive factor per feedback recommendation)
    p.add_argument("--plateau-patience", type=int, default=5,
                   help="ReduceLROnPlateau patience on val_loss (min mode)")
    p.add_argument("--plateau-factor", type=float, default=0.3,
                   help="ReduceLROnPlateau factor (0.3 more aggressive than parent's 0.5)")
    p.add_argument("--plateau-min-lr", type=float, default=1e-7)
    p.add_argument("--warmup-epochs", type=int, default=2,
                   help="Number of linear LR warmup epochs before ReduceLROnPlateau")
    # Checkpoint averaging
    p.add_argument("--checkpoint-avg-top-k", type=int, default=3,
                   help="Number of top checkpoints to average for test inference (0 to disable)")
    # Debug
    p.add_argument("--debug-max-step", "--debug_max_step", dest="debug_max_step",
                   type=int, default=None)
    p.add_argument("--fast-dev-run", "--fast_dev_run", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(42)  # Changed from parent's seed=0 for stochastic diversity
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False

    # Debug mode: limit batches for quick iteration
    if args.debug_max_step is not None:
        max_steps = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
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
        filename="node2-3-1-1-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=3, save_last=True,
    )
    # EarlyStopping on val_f1 (primary metric) — distinct from LR scheduler
    # The LR scheduler monitors val_loss (reliable signal for LR reduction)
    # EarlyStopping monitors val_f1 (the primary metric we want to maximize)
    # Reduced patience 15→12 to account for larger head capacity (512-dim)
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
        warmup_epochs=args.warmup_epochs,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Top-3 Checkpoint Averaging (PRIMARY IMPROVEMENT) ─────────────────────
    # Load and average top-k checkpoints before test inference.
    # Parent feedback explicitly recommends this for +0.003 F1.
    # All ranks must call this to ensure DDP state is synchronized.
    # Only rank 0 actually loads the checkpoint; other ranks get the averaged
    # state via broadcast below.
    avg_performed = False
    if not args.fast_dev_run and args.debug_max_step is None and args.checkpoint_avg_top_k > 0:
        if trainer.is_global_zero:
            avg_performed = average_top_k_checkpoints(
                model_module=model_module,
                checkpoint_cb=checkpoint_cb,
                top_k=args.checkpoint_avg_top_k,
                device="cpu",
            )
        # Broadcast avg_performed flag from rank 0 to all ranks so all enter the same branch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            avg_performed_list = [avg_performed]
            torch.distributed.broadcast_object_list(avg_performed_list, src=0)
            avg_performed = avg_performed_list[0]

    # ── Testing ───────────────────────────────────────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    elif avg_performed:
        # Use averaged model directly (already loaded into model_module on rank 0)
        # Broadcast the averaged state_dict from rank 0 to all other ranks before test
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            state_dict_list = [model_module.state_dict() if trainer.is_global_zero else None]
            torch.distributed.broadcast_object_list(state_dict_list, src=0)
            if not trainer.is_global_zero:
                model_module.load_state_dict(state_dict_list[0], strict=False)
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        # Fall back to best single checkpoint — all ranks load from shared checkpoint path
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # ── Save test score ────────────────────────────────────────────────────────
    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_value = None
        if test_results and isinstance(test_results, list) and len(test_results) > 0:
            first_result = test_results[0]
            for key in ["test_f1", "test_loss", "f1", "f1_score"]:
                if key in first_result:
                    score_value = float(first_result[key])
                    break
            if score_value is None:
                for v in first_result.values():
                    if isinstance(v, (int, float)):
                        score_value = float(v)
                        break

        val_f1_best = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else None
        )

        score_path.write_text(
            f"node: node2-3-1-1-1\n"
            f"test_score: {score_value}\n"
            f"best_val_f1: {val_f1_best}\n"
            f"checkpoint_averaging: top-{args.checkpoint_avg_top_k} "
            f"({'performed' if avg_performed else 'skipped/unavailable'})\n"
            f"seed: 42\n"
            f"head_hidden: {args.head_hidden}\n"
            f"head_dropout: {args.head_dropout}\n"
            f"plateau_factor: {args.plateau_factor}\n"
            f"warmup_epochs: {args.warmup_epochs}\n"
            f"early_stopping_patience: {args.early_stopping_patience}\n"
            f"lora_r: {args.lora_r}\n"
        )
        print(f"Test score saved: {score_path}")


if __name__ == "__main__":
    main()
