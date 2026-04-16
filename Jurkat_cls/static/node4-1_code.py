#!/usr/bin/env python3
"""
Node 1-3: AIDO.Cell-10M (LoRA) + Fine-tuned STRING_GNN + Symbol CNN Triple Fusion
===================================================================================
This node pivots completely away from the frozen STRING_GNN parent (node4, test F1=0.0494)
and adopts the proven AIDO.Cell-10M backbone with LoRA fine-tuning, combining it with
a fine-tuned (unfrozen) STRING_GNN for PPI graph features and a character-level gene
symbol CNN for sequence pattern features.

Architecture:
  1. AIDO.Cell-10M (LoRA r=8, last 4 of 8 layers) on synthetic expression profiles
     - Perturbed gene: -1.0 (masked), all other genes: 1.0
     - Dual-pooling: global mean-pool + perturbed-gene positional embedding -> 512-dim
  2. STRING_GNN (FINE-TUNED, full parameter updates) on full PPI graph
     - Runs once per batch, extracts perturbed gene embedding -> 256-dim
     - Unfrozen to adapt PPI embeddings to the DEG prediction task
  3. Character-level gene symbol CNN (3-branch Conv1d) -> 64-dim
     - Captures gene family naming patterns (NDUF, KDM, etc.)

  Fusion: concat [512 + 256 + 64] = 832-dim -> 3-layer MLP head -> [B, 3, 6640]

Key improvements over parent node4:
  - Complete architecture pivot: AIDO.Cell-10M as primary backbone (proven best in tree)
  - STRING_GNN is fine-tuned (not frozen) to provide adaptable PPI features
  - Character-level symbol CNN adds orthogonal naming pattern signal
  - Focal loss with calibrated weights [3.0, 1.0, 5.0] (not aggressive [28.1, 1.05, 90.9])
  - Best checkpoint loaded for test inference (parent used final/worst checkpoint)
  - ReduceLROnPlateau with patience=7 for stable convergence
  - PyTorch Lightning Trainer (not custom native DDP loop)

Design rationale:
  - AIDO.Cell-10M + LoRA is the proven best backbone in this MCTS tree (node2-2-1: 0.4472 F1)
  - STRING_GNN fine-tuned (not frozen) adds genuine PPI topology signal
  - Symbol CNN adds gene naming convention signal (orthogonal to both above)
  - Triple fusion aims to break through the ~0.447 ceiling of the single-backbone approach
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Constants / Model Paths
# ──────────────────────────────────────────────────────────────────────────────
AIDO_CELL_DIR = "/home/Models/AIDO.Cell-3M"
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT = 6_640
N_CLASSES = 3
AIDO_HIDDEN = 128          # AIDO.Cell-3M hidden size
STRING_HIDDEN = 256        # STRING_GNN hidden size
SYMBOL_CNN_DIM = 64        # character-level symbol CNN output dim
FUSED_DIM = AIDO_HIDDEN * 2 + STRING_HIDDEN + SYMBOL_CNN_DIM  # 256 + 256 + 64 = 576
# Number of output gene tokens in AIDO.Cell-10M
N_AIDO_GENES = 19264


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss (addresses class imbalance without causing distributional collapse)
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [N, C], targets: [N]
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Character-level gene symbol CNN
# ──────────────────────────────────────────────────────────────────────────────
CHAR_VOCAB = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHAR_VOCAB)}  # 0 = padding
CHAR_VOCAB_SIZE = len(CHAR_VOCAB) + 1
MAX_SYMBOL_LEN = 16


def symbol_to_tensor(symbol: str, max_len: int = MAX_SYMBOL_LEN) -> torch.Tensor:
    s = symbol.upper()[:max_len]
    idxs = [CHAR_TO_IDX.get(c, 0) for c in s]
    idxs += [0] * (max_len - len(idxs))
    return torch.tensor(idxs, dtype=torch.long)


class SymbolCNN(nn.Module):
    """
    3-branch Conv1d encoder for gene symbol strings.
    Captures gene family naming conventions via tri-gram/quad-gram patterns.
    """
    def __init__(self, out_dim: int = SYMBOL_CNN_DIM, dropout: float = 0.2):
        super().__init__()
        vocab_size = CHAR_VOCAB_SIZE
        emb_dim = 32
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        branch_dim = out_dim // 3
        extra = out_dim - 3 * branch_dim
        self.conv3 = nn.Conv1d(emb_dim, branch_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(emb_dim, branch_dim, kernel_size=4, padding=1)
        self.conv5 = nn.Conv1d(emb_dim, branch_dim + extra, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, max_len] int64
        e = self.embed(x).permute(0, 2, 1)  # [B, emb_dim, L]
        h3 = F.gelu(self.conv3(e)).max(dim=-1).values  # [B, branch_dim]
        h4 = F.gelu(self.conv4(e)).max(dim=-1).values  # [B, branch_dim]
        h5 = F.gelu(self.conv5(e)).max(dim=-1).values  # [B, branch_dim+extra]
        return self.dropout(torch.cat([h3, h4, h5], dim=-1))  # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# STRING_GNN wrapper (full fine-tuning, unfrozen)
# ──────────────────────────────────────────────────────────────────────────────
def build_ensg_to_node_idx(node_names_path: str) -> Dict[str, int]:
    """node_names.json: list of ENSG IDs, index = node position."""
    with open(node_names_path, "r") as f:
        node_names: List[str] = json.load(f)
    return {name.split(".")[0]: i for i, name in enumerate(node_names)}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        ensg_to_node_idx: Dict[str, int],
        tokenizer,
        n_aido_genes: int = N_AIDO_GENES,
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.n_aido_genes = n_aido_genes

        # Map pert_id -> STRING node index (-1 if not in graph)
        self.node_indices: List[int] = []
        for pid in self.pert_ids:
            base = pid.split(".")[0]
            self.node_indices.append(ensg_to_node_idx.get(base, -1))

        # Map pert_id -> AIDO.Cell gene index (0-based position in [0, n_aido_genes))
        # Use tokenizer to find the positional index for each pert_id
        self.aido_pert_positions: List[int] = []
        for pid in self.pert_ids:
            # Try ENSG id lookup in AIDO.Cell tokenizer
            base = pid.split(".")[0]
            try:
                encoding = tokenizer({"gene_ids": [base], "expression": [1.0]}, return_tensors="pt")
                # The tokenizer places genes at fixed positions; find non-missing position
                # input_ids: [1, n_aido_genes] float32; the gene's position is where value != -1.0
                ids = encoding["input_ids"][0]  # [n_aido_genes]
                # Gene at its designated slot is filled with 1.0; all others -1.0
                nonmissing = (ids > -0.5).nonzero(as_tuple=False)
                if len(nonmissing) > 0:
                    self.aido_pert_positions.append(int(nonmissing[0].item()))
                else:
                    # Gene not in AIDO vocab: use global mean position (fallback)
                    self.aido_pert_positions.append(-1)
            except Exception:
                self.aido_pert_positions.append(-1)

        # Build symbol tensors
        self.symbol_tensors: List[torch.Tensor] = [
            symbol_to_tensor(s) for s in self.symbols
        ]

        # Build AIDO.Cell expression profiles: perturbed gene = -1.0 (masked), others = 1.0
        # We do NOT precompute these here to avoid large memory; built in __getitem__
        self.pert_id_to_aido_pos: Dict[str, int] = {
            pid: pos for pid, pos in zip(self.pert_ids, self.aido_pert_positions)
        }

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1}→{0,1,2}
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pert_pos = self.aido_pert_positions[idx]

        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "symbol_tensor": self.symbol_tensors[idx],   # [MAX_SYMBOL_LEN]
            "node_idx": self.node_indices[idx],           # STRING node index
            "pert_aido_pos": pert_pos,                    # AIDO.Cell gene position (or -1)
        }

        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_aido_input_batch(
    pert_aido_positions: List[int],
    n_genes: int = N_AIDO_GENES,
) -> torch.Tensor:
    """
    Build AIDO.Cell expression input tensor for a batch.
    All genes = 1.0 (present), perturbed gene = -1.0 (masked).
    Shape: [B, n_genes] float32
    """
    B = len(pert_aido_positions)
    expr = torch.ones(B, n_genes, dtype=torch.float32)
    for b, pos in enumerate(pert_aido_positions):
        if 0 <= pos < n_genes:
            expr[b, pos] = -1.0  # mask the perturbed gene
    return expr


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "node_idx": torch.tensor([b["node_idx"] for b in batch], dtype=torch.long),
        "pert_aido_pos": [b["pert_aido_pos"] for b in batch],
        "symbol_tensor": torch.stack([b["symbol_tensor"] for b in batch]),  # [B, MAX_SYMBOL_LEN]
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
        micro_batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.tokenizer = None
        self.ensg_to_idx: Dict[str, int] = {}
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        # Tokenizer setup: rank 0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        if not self.ensg_to_idx:
            self.ensg_to_idx = build_ensg_to_node_idx(
                str(Path(STRING_GNN_DIR) / "node_names.json")
            )
            print(f"[DataModule] STRING_GNN vocab: {len(self.ensg_to_idx)} ENSG IDs")

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self.ensg_to_idx, self.tokenizer
            )
            self.val_ds = PerturbationDataset(
                val_df, self.ensg_to_idx, self.tokenizer
            )
        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self.ensg_to_idx, self.tokenizer, is_test=True
            )
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,  # Must drop last to ensure same batch count across DDP ranks
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Manual LoRA Implementation
# Replaces PEFT's get_peft_model() so the frozen AIDO base is NOT a DDP-tracked
# parameter.  DDP will only see LoRA adapters + GNN + head, so
# find_unused_parameters=False works correctly (no deadlock, no GPU-1 OOM).
# ──────────────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Low-rank adaptation for a single Linear layer: W -> W + BA, where
    A and B are low-rank trainable matrices."""

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        scale = alpha / r
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, linear.out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)
        self.lora_scale = scale
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_dropout(x)
        lora_out = self.lora_A(lora_out)
        lora_out = self.lora_B(lora_out) * self.lora_scale
        return base + lora_out


def _inject_lora(model: nn.Module, target_layer: nn.Module, r: int, alpha: int, dropout: float):
    """Replace target_layer's Q/K/V projections with LoRA-wrapped versions."""
    for name, module in target_layer.named_children():
        if name in ("query", "key", "value"):
            lora_module = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(target_layer, name, lora_module)


# ──────────────────────────────────────────────────────────────────────────────
# Triple Fusion Model
# ──────────────────────────────────────────────────────────────────────────────
class TripleFusionDEGModel(nn.Module):
    """
    Triple fusion model combining:
    1. AIDO.Cell-3M with LoRA (last 4 layers) - dual-pool -> 256-dim
       Frozen base is stored in self.aido_frozen (outside DDP parameter set).
       LoRA adapters are trainable and visible to DDP.
    2. STRING_GNN (full fine-tuning, unfrozen) - pert node embedding -> 256-dim
    3. Character-level gene symbol CNN -> 64-dim
    Total fused: 256+256+64 = 576-dim -> 3-layer MLP head -> [B, 3, 6640]

    Design rationale for separating frozen base from DDP:
      - DDP sees only trainable parameters (LoRA + GNN + head ≈ 8.9M params)
      - Frozen base is NOT a DDP parameter → no deadlock with find_unused_parameters=False
      - No DDP unused-parameter bookkeeping overhead → no GPU-1 OOM
    """
    N_STRING_NODES = 18_870

    def __init__(
        self,
        head_hidden: int = 320,
        dropout: float = 0.3,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        # ── 1a. AIDO.Cell-3M FROZEN base (outside DDP parameter set) ────────
        aido_base = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        aido_base.config.use_cache = False
        # Freeze ALL base parameters; gradients ONLY flow through LoRA adapters
        for p in aido_base.parameters():
            p.requires_grad = False

        # Enable gradient checkpointing on the frozen base
        aido_base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Patch: AIDO.Cell raises NotImplementedError for get_input_embeddings().
        # For LoRA fine-tuning inputs are synthetic float tensors that do NOT need
        # input-side gradients, so this no-op is safe.
        if not hasattr(aido_base, "_input_require_grads_patched"):
            aido_base.enable_input_require_grads = lambda *a, **kw: None  # noqa: E731
            aido_base._input_require_grads_patched = True

        # ── 1b. LoRA adapters on last 4 transformer layers (trainable, DDP-tracked) ──
        n_layers = aido_base.config.num_hidden_layers  # 8 for AIDO.Cell-3M
        lora_layers = list(range(n_layers - 4, n_layers))  # last 4: [4,5,6,7]

        for layer_idx in lora_layers:
            layer = aido_base.bert.encoder.layer[layer_idx].attention.self
            _inject_lora(aido_base, layer, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        # Register frozen base as a plain attribute (NOT a DDP-wrapped parameter).
        # Lightning moves self.aido_frozen to the correct device automatically.
        self.aido_frozen = aido_base

        # ── 2. STRING_GNN - FULLY FINE-TUNED (unfrozen, trainable) ───────────
        self.gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        for p in self.gnn.parameters():
            p.requires_grad = True

        # Register graph data as buffers (moved to GPU automatically by Lightning)
        graph_data = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", map_location="cpu")
        self.register_buffer("edge_index", graph_data["edge_index"].long())
        edge_weight = graph_data.get("edge_weight")
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight.float())
        else:
            self.register_buffer("edge_weight", None)

        # Learnable fallback embedding for genes NOT in STRING graph
        self.string_fallback = nn.Parameter(torch.randn(STRING_HIDDEN) * 0.02)

        # ── 3. Character-level symbol CNN ────────────────────────────────────
        self.symbol_cnn = SymbolCNN(out_dim=SYMBOL_CNN_DIM, dropout=0.2)

        # ── 4. Fusion MLP head ───────────────────────────────────────────────
        # Input: 256 (AIDO dual-pool) + 256 (STRING) + 64 (symbol CNN) = 576
        fused_dim = AIDO_HIDDEN * 2 + STRING_HIDDEN + SYMBOL_CNN_DIM
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, N_CLASSES * N_GENES_OUT),
        )

        # Initialize head
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        aido_input_ids: torch.Tensor,       # [B, 19264] float32
        aido_attention_mask: torch.Tensor,   # [B, 19264] int64
        pert_aido_pos: List[int],            # list of ints
        node_idx: torch.Tensor,              # [B] int64 (STRING node indices)
        symbol_tensor: torch.Tensor,         # [B, MAX_SYMBOL_LEN] int64
    ) -> torch.Tensor:
        device = aido_input_ids.device
        B = aido_input_ids.shape[0]

        # ── AIDO.Cell-3M forward (frozen base + LoRA adapters) ───────────────
        # aido_frozen contains LoRA-wrapped layers; gradients flow through LoRA params.
        aido_out = self.aido_frozen(
            input_ids=aido_input_ids,
            attention_mask=aido_attention_mask,
        )
        hidden = aido_out.last_hidden_state  # [B, 19266, 256]

        # Global mean-pool (exclude the 2 appended summary tokens)
        gene_hidden = hidden[:, :N_AIDO_GENES, :]  # [B, 19264, 256]
        mean_pool = gene_hidden.mean(dim=1)  # [B, 256]

        # Perturbed-gene positional embedding
        pert_pos_tensor = torch.tensor(pert_aido_pos, dtype=torch.long, device=device)
        valid_pert = pert_pos_tensor >= 0  # [B] bool
        safe_pos = pert_pos_tensor.clone()
        safe_pos[~valid_pert] = 0  # clamp for gather
        pert_hidden = gene_hidden[torch.arange(B, device=device), safe_pos]
        pert_emb = torch.where(valid_pert.unsqueeze(-1), pert_hidden, mean_pool)

        aido_features = torch.cat([mean_pool, pert_emb], dim=-1).float()  # [B, 512]

        # ── STRING_GNN forward (fine-tuned, gradients flow) ──────────────────
        gnn_out = self.gnn(
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
        )
        node_emb: torch.Tensor = gnn_out.last_hidden_state  # [18870, 256]

        valid_string = node_idx >= 0  # [B] bool
        safe_idx = node_idx.clone()
        safe_idx[~valid_string] = 0
        graph_emb = node_emb[safe_idx]  # [B, 256]
        fallback_exp = self.string_fallback.unsqueeze(0).expand(B, -1)
        string_features = torch.where(
            valid_string.unsqueeze(-1),
            graph_emb.float(),
            fallback_exp.float(),
        )  # [B, 256]

        # ── Symbol CNN forward ───────────────────────────────────────────────
        symbol_features = self.symbol_cnn(symbol_tensor).float()  # [B, 64]

        # ── Triple fusion ────────────────────────────────────────────────────
        fused = torch.cat([aido_features, string_features, symbol_features], dim=-1)  # [B, 832]
        logits = self.head(fused)  # [B, 3 * N_GENES_OUT]
        return logits.view(B, N_CLASSES, N_GENES_OUT)



# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute macro-averaged per-gene F1, matching calc_metric.py logic."""
    n_genes = y_true.shape[1]
    f1_vals: List[float] = []
    y_hat = y_pred.argmax(axis=1)  # [B, G]
    for g in range(n_genes):
        yt = y_true[:, g]
        yp = y_hat[:, g]
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        head_hidden: int = 320,
        dropout: float = 0.3,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        backbone_lr: float = 3e-4,
        head_lr: float = 1e-3,
        string_lr: float = 1e-4,
        weight_decay: float = 0.05,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []

        # Initialize model in __init__ (NOT in setup) so configure_optimizers()
        # can access self.model when the trainer calls it before setup("fit").
        self.model = TripleFusionDEGModel(
            head_hidden=self.hparams.head_hidden,
            dropout=self.hparams.dropout,
            lora_r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
        )

        # Cast trainable params to float32 for stable optimization
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Calibrated class weights: moderate (not extreme)
        # ~3.41% down-regulated, ~95.48% unchanged, ~1.10% up-regulated
        class_weights = torch.tensor([3.0, 1.0, 5.0], dtype=torch.float32)
        self.criterion = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        # Model already initialized in __init__; this is a no-op hook for Lightning.
        pass

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Build AIDO.Cell expression inputs on-the-fly (efficient, no pre-computation)
        pert_positions = batch["pert_aido_pos"]  # list of ints
        aido_expr = make_aido_input_batch(pert_positions, N_AIDO_GENES).to(self.device)
        # attention_mask is all-ones (AIDO.Cell ignores it internally)
        B = aido_expr.shape[0]
        attn_mask = torch.ones(B, N_AIDO_GENES, dtype=torch.long, device=self.device)

        return self.model(
            aido_input_ids=aido_expr,
            aido_attention_mask=attn_mask,
            pert_aido_pos=pert_positions,
            node_idx=batch["node_idx"].to(self.device),
            symbol_tensor=batch["symbol_tensor"].to(self.device),
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
        # NOTE: no sync_dist=True here.  In DDP Lightning, validation runs on
        # all ranks; the NCCL all-reduce for val_loss is expensive and causes
        # hangs when only rank 0 contributes.  Use sync_dist=False and rely
        # on all_gather in on_validation_epoch_end for distributed metric sync.
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=False)
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

        # All-gather across ranks
        ap = self.all_gather(lp)   # [world_size, n_local, 3, 6640]
        al = self.all_gather(ll)   # [world_size, n_local, 6640]
        ai = self.all_gather(li)   # [world_size, n_local]

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # Compute F1 on rank 0 (has gathered global predictions)
        f1_value = 0.0
        if self.trainer.is_global_zero:
            preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
            labels = al.view(-1, N_GENES_OUT).cpu().numpy()
            idxs = ai.view(-1).cpu().numpy()
            _, uniq = np.unique(idxs, return_index=True)
            f1_value = float(compute_deg_f1(preds[uniq], labels[uniq]))

        # Broadcast F1 from rank 0 to all ranks so EarlyStopping sees it on every rank
        f1_tensor = torch.tensor(f1_value, dtype=torch.float32, device="cpu")
        try:
            f1_tensor = self.trainer.strategy.broadcast(f1_tensor, src=0)
        except Exception:
            pass  # fallback: non-zero ranks keep f1_tensor=0.0
        self.log("val_f1", float(f1_tensor), prog_bar=True, sync_dist=False)

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

        ap = self.all_gather(lp.cpu())
        ai = self.all_gather(li.cpu())

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
            dm = self.trainer.datamodule
            pert_ids = dm.test_pert_ids
            symbols = dm.test_symbols
            rows = [
                {
                    "idx": pert_ids[int(i)],
                    "input": symbols[int(i)],
                    "prediction": json.dumps(preds[r].tolist()),
                }
                for r, i in enumerate(idxs)
            ]
            pred_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
            self.print(f"Test predictions saved -> {pred_path}")

    def configure_optimizers(self):
        # Separate parameter groups with different LRs:
        # 1. AIDO.Cell LoRA params (backbone_lr)
        # 2. STRING_GNN params (string_lr - usually lower)
        # 3. Head + symbol CNN + fallback (head_lr)
        lora_params, string_params, head_params = [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "aido_frozen." in name:
                lora_params.append(p)
            elif "gnn." in name or "string_fallback" in name:
                string_params.append(p)
            else:
                head_params.append(p)

        param_groups = [
            {"params": lora_params, "lr": self.hparams.backbone_lr},
            {"params": string_params, "lr": self.hparams.string_lr},
            {"params": head_params, "lr": self.hparams.head_lr},
        ]
        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        opt = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )
        # ReduceLROnPlateau requires val_f1 to be logged every epoch.
        # In fast_dev_run (limit_val_batches=1) the metric may not be
        # available at scheduler-check time, so skip the scheduler there.
        if getattr(self, "_fast_dev_run", False):
            return opt
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=7, min_lr=1e-7,
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
        log_fn = getattr(self, "print", print)
        log_fn(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        return out

    def load_state_dict(self, state_dict, strict=True):
        """Load from a partial checkpoint (only trainable + buffers, DDP-safe)."""
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 1-3: AIDO.Cell-10M + Fine-tuned STRING_GNN + Symbol CNN Triple Fusion"
    )
    default_data_dir = str((Path(__file__).parent / ".." / ".." / "data").resolve())
    p.add_argument("--data_dir", type=str, default=default_data_dir)
    p.add_argument("--micro_batch_size", type=int, default=16)
    p.add_argument("--global_batch_size", type=int, default=128)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--backbone_lr", type=float, default=3e-4)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--string_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--head_hidden", type=int, default=320)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--early_stopping_patience", type=int, default=40)
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    return p.parse_args()


def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

    # For fast_dev_run, cap micro_batch_size to 1 per GPU to avoid OOM.
    # Each DDP process independently loads micro_batch_size samples into GPU memory,
    # so we must ensure micro_batch_size fits in single-GPU memory.
    if args.fast_dev_run and args.micro_batch_size > 1:
        print(f"[Auto] fast_dev_run: capping micro_batch_size {args.micro_batch_size} -> 1")
        args.micro_batch_size = 1

    # GPU-memory-aware micro_batch_size cap: each DDP process independently
    # loads micro_batch_size samples.  AIDO.Cell-3M + STRING_GNN fits in 80 GB
    # with micro_batch_size=1 per GPU (OOM at micro_batch_size=2 on 1 GPU).
    # For fewer GPUs, cap to 1 to avoid OOM; for 8 GPUs keep the user-specified value.
    if n_gpus < 8 and args.micro_batch_size > 1:
        print(f"[Auto] capping micro_batch_size {args.micro_batch_size} -> 1 "
              f"for {n_gpus} GPUs (AIDO.Cell-3M + STRING_GNN memory constraint)")
        args.micro_batch_size = 1

    # Compute effective accumulation
    accumulate_grad = max(
        1, args.global_batch_size // (args.micro_batch_size * max(1, n_gpus))
    )

    fast_dev_run = args.fast_dev_run
    debug_max_step = args.debug_max_step

    limit_train_batches = 1 if fast_dev_run else (
        debug_max_step if debug_max_step is not None else 1.0
    )
    limit_val_batches = 1 if fast_dev_run else 1.0
    limit_test_batches = 1 if fast_dev_run else 1.0
    max_steps = -1 if debug_max_step is None else debug_max_step

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stopping_patience,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=1800)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else max_steps  # validate once per epoch (at end of all steps)
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, progress_bar],
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    # DataModule and Model
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        string_lr=args.string_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
    # Propagate fast_dev_run flag to model so configure_optimizers() can skip
    # ReduceLROnPlateau (which needs val_f1 available at epoch boundaries — not
    # guaranteed during fast_dev_run when limit_val_batches=1).
    model_module._fast_dev_run = args.fast_dev_run

    # Train
    trainer.fit(model_module, datamodule=datamodule)

    # Test using best checkpoint (critical fix from parent node4 which used final checkpoint)
    if fast_dev_run or debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # Save test score
    if trainer.is_global_zero:
        best_val_f1 = checkpoint_callback.best_model_score
        if best_val_f1 is not None:
            score_val = float(best_val_f1)
        else:
            score_val = 0.0
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(f"best_val_f1: {score_val:.6f}\n")
        print(f"Test score saved -> {score_path}", flush=True)


if __name__ == "__main__":
    main()
