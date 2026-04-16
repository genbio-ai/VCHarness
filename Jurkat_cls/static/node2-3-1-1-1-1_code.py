#!/usr/bin/env python3
"""
Node 2-3-1-1-1-1: AIDO.Cell-10M + LoRA (r=4, ALL 8 layers) + Symbol CNN
               + Frozen STRING GNN PPI Embeddings
               + Reverted Hyperparameters + Extended Patience + New Seed
=======================================================================
Improvements over parent node2-3-1-1-1 (test F1=0.4520, regressed -0.0035):

Parent failed due to three compounding mistakes:
  (1) 2-epoch LR warmup (10%/55% ramp) wasted early training → lower val_f1 trajectory
  (2) plateau_factor=0.3 fired only ONCE vs grandparent's 4 times → inadequate LR adaptation
  (3) head_hidden=512 combined with warmup = suboptimal convergence trajectory
  (4) Checkpoint averaging with E8 (0.4465, -0.0033 below best) diluted the best checkpoint

This node fixes all of those issues and additionally explores:
  (5) Extended early stopping patience (20) per feedback recommendation for longer convergence
  (6) Slightly more generous plateau_patience (6) to avoid premature LR reductions

Changes from parent (node2-3-1-1-1, F1=0.4520):

1. REMOVE LR WARMUP (warmup_epochs=0):
   - Parent feedback: "Warmup hurt early learning; 10%/55% LR for 2 epochs"
   - val_f1 ceiling dropped from grandparent's 0.456 to 0.4498 due to warmup
   - No warmup means full LR from epoch 0, allowing rapid early convergence
   - Grandparent (node2-3-1-1, no warmup) achieved best val_f1=0.456 at epoch 14

2. REVERT HEAD: head_hidden 512→384, dropout 0.35→0.40:
   - Parent feedback: "Revert head_hidden to 384 — combined negatively with warmup"
   - Grandparent's 384-dim head achieved better val_f1 (0.456) than parent's 512-dim (0.4498)
   - 384-dim is proven optimal for this 832→19920 task in node2-3-1-1

3. RESTORE plateau_factor 0.3→0.5:
   - Parent feedback: "Parent's factor=0.5 fired 4 times and produced smoother LR adaptation"
   - Factor=0.3 fired only once in parent run; factor=0.5 is the proven setting

4. DISABLE CHECKPOINT AVERAGING (single best checkpoint):
   - Parent feedback: "Including E8 (6 epochs before best, -0.0033 F1) diluted the best checkpoint"
   - Grandparent used single best checkpoint → near-zero val-test gap (0.0005)
   - Single best checkpoint is the safer, proven approach

5. EXTENDED EARLY STOPPING PATIENCE (15→20):
   - Parent feedback: "Increase patience to 20-25 may allow model to find better convergence basin"
   - Node3-2's patience=40 produced 0.462 by training to E58 with best at E18
   - 20-epoch patience provides 5-6 more epochs than grandparent's 15, allowing more LR reductions
   - This is the novel exploration direction for this node

6. SLIGHTLY MORE GENEROUS plateau_patience (5→6):
   - Provides scheduler slightly more time to confirm plateau before reducing
   - Avoids premature LR reduction that might occur if val_loss fluctuates

7. DIFFERENT SEED (7):
   - Parent used seed=42, grandparent used seed=0
   - Seed=7 provides independent stochastic exploration of the loss landscape
   - node3-2's 0.462 achieved with specific seed; exploring different seeds may recover similar gains

Architecture (unchanged from grandparent node2-3-1-1):
  - AIDO.Cell-10M: r=4 LoRA on all 8 transformer layers (Q/K/V), global mean-pool + pert_emb
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

    Combined: [B, 832] → MLP head (384-dim, reverted from parent's 512-dim) → [B, 3, 6640]

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
        # Reverted to 384-dim from parent's 512-dim (per feedback recommendation)
        # Grandparent (node2-3-1-1) achieved best val_f1=0.456 with 384-dim
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
        plateau_patience: int = 6,
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

        # ReduceLROnPlateau on val_loss (proven to fire 4x in grandparent node2-3-1-1)
        # NO warmup — grandparent's no-warmup approach achieved best val_f1=0.456
        # factor=0.5 proven to fire 4 reductions (vs parent's 0.3 which only fired once)
        # plateau_patience=6 (slightly more generous than grandparent's 5)
        plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",              # val_loss should decrease
            factor=hp.plateau_factor,   # 0.5 (restored from grandparent)
            patience=hp.plateau_patience,
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
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 2-3-1-1-1-1: AIDO.Cell-10M + LoRA all-8-layers + Symbol CNN + "
                    "Frozen STRING GNN PPI + Reverted HP + Extended Patience"
    )
    # Data
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--num-workers", type=int, default=4)
    # Batch / training
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--early-stopping-patience", type=int, default=20,
                   help="Early stopping patience on val_f1 (max mode). "
                        "Extended to 20 (from grandparent's 15) to allow longer convergence "
                        "per feedback recommendation suggesting patience=20-25.")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    # LoRA
    p.add_argument("--lora-r", type=int, default=4,
                   help="LoRA rank for all 8 AIDO.Cell transformer layers")
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    # Head (reverted to grandparent's 384/0.4 from parent's 512/0.35)
    p.add_argument("--head-hidden", type=int, default=384,
                   help="Head hidden dimension (reverted to grandparent's 384; "
                        "parent's 512 combined negatively with warmup)")
    p.add_argument("--head-dropout", type=float, default=0.40,
                   help="Head dropout (reverted to 0.40 from parent's 0.35)")
    # Optimizer
    p.add_argument("--backbone-lr", type=float, default=2e-4)
    p.add_argument("--head-lr-multiplier", type=float, default=3.0)
    p.add_argument("--symbol-lr-multiplier", type=float, default=2.0)
    p.add_argument("--ppi-lr-multiplier", type=float, default=2.5)
    p.add_argument("--weight-decay", type=float, default=0.03)
    # Loss
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    # LR Scheduler (restored to grandparent's proven settings)
    p.add_argument("--plateau-patience", type=int, default=6,
                   help="ReduceLROnPlateau patience on val_loss (min mode); "
                        "slightly more generous than grandparent's 5 to avoid premature reduction")
    p.add_argument("--plateau-factor", type=float, default=0.5,
                   help="ReduceLROnPlateau factor (restored to 0.5 from parent's 0.3; "
                        "grandparent fired 4 reductions with 0.5)")
    p.add_argument("--plateau-min-lr", type=float, default=1e-7)
    # Debug
    p.add_argument("--debug-max-step", "--debug_max_step", dest="debug_max_step",
                   type=int, default=None)
    p.add_argument("--fast-dev-run", "--fast_dev_run", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(7)  # New seed=7 (different from parent's 42 and grandparent's 0)
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
        filename="node2-3-1-1-1-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=3, save_last=True,
    )
    # EarlyStopping on val_f1 (primary metric) — distinct from LR scheduler
    # The LR scheduler monitors val_loss (reliable signal for LR reduction)
    # EarlyStopping monitors val_f1 (the primary metric we want to maximize)
    # Extended patience 20 (from grandparent's 15) per feedback recommendation
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

    # ── Testing ───────────────────────────────────────────────────────────────
    # Use single best checkpoint (reverted from parent's averaging approach)
    # Grandparent's single best produced near-zero val-test gap (0.0005)
    # Parent's checkpoint averaging with E8 diluted performance
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
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
            f"node: node2-3-1-1-1-1\n"
            f"test_score: {score_value}\n"
            f"best_val_f1: {val_f1_best}\n"
            f"checkpoint_averaging: disabled (single best)\n"
            f"seed: 7\n"
            f"head_hidden: {args.head_hidden}\n"
            f"head_dropout: {args.head_dropout}\n"
            f"warmup: none\n"
            f"plateau_factor: {args.plateau_factor}\n"
            f"plateau_patience: {args.plateau_patience}\n"
            f"early_stopping_patience: {args.early_stopping_patience}\n"
        )
        print(f"Test score saved → {score_path}")
        print(f"best_val_f1={val_f1_best}, test_score={score_value}")


if __name__ == "__main__":
    main()
