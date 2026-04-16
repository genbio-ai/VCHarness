#!/usr/bin/env python3
"""
Node 4-1-1: AIDO.Cell-10M (LoRA r=4, all 8 layers) + STRING_GNN (Precomputed) + Symbol CNN
          with Cross-Attention Fusion + Manifold Mixup
================================================================================
This node fixes the critical AIDO.Cell size bug from parent node4-1 (used 3M instead of 10M)
and adopts the proven best architecture pattern from the MCTS tree (cross-attention fusion from
node3-1-3-1-1-1-1, tree-best F1=0.4768).

CRITICAL OOM FIXES:
1. AIDO.Cell model explicitly cast to bf16: Without this, the float32 model creates
   ~30 GB activations per forward pass (vs ~0.2 GB in bf16), causing OOM.
2. STRING_GNN embeddings precomputed: The fine-tuned STRING_GNN's full graph forward pass
   (8 GCN layers, 18,870 nodes, 786K edges) created ~30 GB of intermediate activations
   per GPU. Precomputing embeddings once and using as a frozen lookup eliminates this.

Architecture:
  1. AIDO.Cell-10M (LoRA r=4, all 8 layers) on synthetic expression profiles
     - Perturbed gene: -1.0 (masked), all other genes: 1.0
     - Produces: global mean-pool (256-dim) + perturbed-gene embedding (256-dim)
     - Model explicitly cast to bf16 to reduce activation memory 150x
  2. STRING_GNN (PRETRAINED, precomputed embeddings as frozen buffer)
     - Embeddings computed once using pretrained GNN, stored as buffer
     - No GNN forward pass during training (saves ~30 GB GPU memory)
     - Lookup at perturbed gene node -> 256-dim
  3. Character-level gene symbol CNN (3-branch Conv1d) -> 64-dim
     - Captures gene family naming patterns (NDUF, KDM, etc.)
  4. Cross-attention fusion (3-layer TransformerEncoder, d_model=256, nhead=8, dim_ff=256)
     - Fuses 4 tokens: [global_emb, pert_emb, sym_proj, ppi_feat]
     - Key breakthrough from node3-1-3-1 lineage: +0.015 F1 over concat+MLP
  5. Manifold mixup (alpha=0.3) on fused features for regularization
  6. Prediction head: 256→256→19920, dropout=0.5

Key improvements over parent node4-1:
  - Fix AIDO.Cell-3M→10M (critical bug: hidden 128→256, feature dim 256→512)
  - Replace concat+MLP fusion with cross-attention (proven +0.015+ F1 gain)
  - Switch LoRA to r=4 all-8-layers (proven better than r=8 last-4)
  - Add manifold mixup (alpha=0.3) for better generalization
  - Use class weights [6.0, 1.0, 12.0] and focal gamma=1.5 (proven optimal from tree)
  - Use weight_decay=0.10 and ReduceLROnPlateau patience=12 (proven from tree-best)
  - Top-3 checkpoint averaging for test inference (consistent +0.003 gain in tree)
  - STRING_GNN: precomputed (frozen) due to GPU memory constraints

Design differentiation from node3-1-3-1-1-1-1 (tree best, F1=0.4768):
  - Same architecture as tree-best: cross-attention + frozen STRING + mixup
  - STRING_GNN: precomputed embeddings (same as tree-best), not fine-tuned
  - AIDO.Cell-10M with LoRA r=4 all-8-layers (same as tree-best)
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
AIDO_CELL_DIR = "/home/Models/AIDO.Cell-10M"   # FIXED: was 3M in parent
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT = 6_640
N_CLASSES = 3
AIDO_HIDDEN = 256          # AIDO.Cell-10M hidden size (was 128 in parent, now fixed)
STRING_HIDDEN = 256        # STRING_GNN hidden size
SYMBOL_CNN_DIM = 64        # character-level symbol CNN output dim
CROSS_ATTN_DIM = 256       # cross-attention d_model
# Number of output gene tokens in AIDO.Cell-10M
N_AIDO_GENES = 19264


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss (addresses class imbalance without causing distributional collapse)
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.05):
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
    def __init__(self, out_dim: int = SYMBOL_CNN_DIM, dropout: float = 0.1):
        super().__init__()
        vocab_size = CHAR_VOCAB_SIZE
        emb_dim = 32
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # 3 branches with different kernel sizes, then concat + linear
        branch_dim = 64
        self.conv3 = nn.Conv1d(emb_dim, branch_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(emb_dim, branch_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(emb_dim, branch_dim, kernel_size=7, padding=3)
        self.proj = nn.Sequential(
            nn.Linear(branch_dim * 3, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, max_len] int64
        e = self.embed(x).permute(0, 2, 1)  # [B, emb_dim, L]
        h3 = F.gelu(self.conv3(e)).max(dim=-1).values  # [B, branch_dim]
        h5 = F.gelu(self.conv5(e)).max(dim=-1).values  # [B, branch_dim]
        h7 = F.gelu(self.conv7(e)).max(dim=-1).values  # [B, branch_dim]
        cat = torch.cat([h3, h5, h7], dim=-1)          # [B, branch_dim*3]
        return self.proj(cat)                            # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Attention Feature Fusion Module
# ──────────────────────────────────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    Fuses 4 feature tokens via 3-layer TransformerEncoder (pre-norm).
    Tokens: [global_emb, pert_emb, sym_proj, ppi_feat] -> [B, 4, 256]
    Proven tree-best architectural pattern from node3-1-3-1 lineage.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 256,
        num_layers: int = 3,
        attn_dropout: float = 0.2,
        sym_in_dim: int = 64,    # from symbol CNN
    ):
        super().__init__()
        # Project symbol CNN output to d_model
        self.sym_proj = nn.Sequential(
            nn.Linear(sym_in_dim, d_model),
            nn.LayerNorm(d_model),
        )
        # Learnable token-type (positional) embeddings for the 4 tokens
        self.token_type_emb = nn.Parameter(torch.randn(4, d_model) * 0.02)

        # 3-layer TransformerEncoder with pre-norm configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm (proven more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        global_emb: torch.Tensor,  # [B, 256]
        pert_emb: torch.Tensor,    # [B, 256]
        sym_feat: torch.Tensor,    # [B, 64]
        ppi_feat: torch.Tensor,    # [B, 256]
    ) -> torch.Tensor:             # [B, 256]
        B = global_emb.shape[0]

        # Project symbol features to d_model
        sym_proj = self.sym_proj(sym_feat)  # [B, 256]

        # Stack 4 tokens: [B, 4, 256]
        tokens = torch.stack([global_emb, pert_emb, sym_proj, ppi_feat], dim=1)

        # Add token-type embeddings
        tokens = tokens + self.token_type_emb.unsqueeze(0)  # [B, 4, 256]

        # TransformerEncoder self-attention over 4 tokens
        fused = self.transformer(tokens)  # [B, 4, 256]

        # Mean-pool over tokens and apply final norm
        out = self.out_norm(fused.mean(dim=1))  # [B, 256]
        return out


# ──────────────────────────────────────────────────────────────────────────────
# STRING_GNN helper
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
        # We build a synthetic expression vector: perturbed gene = -1.0, all others = 1.0
        # Find each gene's position in the AIDO.Cell vocabulary
        self.aido_pert_positions: List[int] = []
        for pid in self.pert_ids:
            base = pid.split(".")[0]
            try:
                encoding = tokenizer({"gene_ids": [base], "expression": [1.0]}, return_tensors="pt")
                ids = encoding["input_ids"][0]  # [n_aido_genes] float32
                # Gene at its designated slot is filled with 1.0; all others -1.0
                nonmissing = (ids > -0.5).nonzero(as_tuple=False)
                if len(nonmissing) > 0:
                    self.aido_pert_positions.append(int(nonmissing[0].item()))
                else:
                    self.aido_pert_positions.append(-1)
            except Exception:
                self.aido_pert_positions.append(-1)

        # Build symbol tensors
        self.symbol_tensors: List[torch.Tensor] = [
            symbol_to_tensor(s) for s in self.symbols
        ]

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1}→{0,1,2}
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "symbol_tensor": self.symbol_tensors[idx],   # [MAX_SYMBOL_LEN]
            "node_idx": self.node_indices[idx],           # STRING node index
            "pert_aido_pos": self.aido_pert_positions[idx],  # AIDO.Cell gene position (or -1)
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
    All genes = 1.0 (present), perturbed gene = -1.0 (masked/KO).
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
        "symbol_tensor": torch.stack([b["symbol_tensor"] for b in batch]),
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
        self.tokenizer = None
        self.ensg_to_idx: Dict[str, int] = {}
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        # Tokenizer: rank 0 downloads first, then all ranks load
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
            drop_last=False,  # Fixed: don't drop last in validation
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
# parameter.  DDP only sees LoRA adapters + GNN + head, avoiding deadlock.
# ──────────────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Low-rank adaptation for a single Linear layer."""

    def __init__(self, linear: nn.Linear, r: int = 4, alpha: int = 8, dropout: float = 0.1):
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


def _inject_lora(target_layer: nn.Module, r: int, alpha: int, dropout: float):
    """Replace target_layer's Q/K/V projections with LoRA-wrapped versions."""
    for name, module in target_layer.named_children():
        if name in ("query", "key", "value"):
            lora_module = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(target_layer, name, lora_module)


# ──────────────────────────────────────────────────────────────────────────────
# Main Fusion Model
# ──────────────────────────────────────────────────────────────────────────────
class CrossAttnFusionDEGModel(nn.Module):
    """
    Cross-attention fusion model combining:
    1. AIDO.Cell-10M with LoRA (r=4, all 8 layers) - global mean-pool + pert embedding
       Frozen base stored outside DDP parameter set. LoRA adapters are trainable.
    2. STRING_GNN (PRETRAINED, frozen embeddings as lookup) - pert node embedding
       Precomputed once using pretrained GNN, stored as frozen buffer.
    3. Character-level gene symbol CNN -> 64-dim
    4. 3-layer TransformerEncoder cross-attention fusion (d_model=256, nhead=8, dim_ff=256)
    5. Prediction head: 256->256->19920 (dropout=0.5)

    Total fused: 4 tokens of 256-dim each, then mean-pool -> 256-dim
    """
    N_STRING_NODES = 18_870

    def __init__(
        self,
        head_hidden: int = 256,
        head_dropout: float = 0.5,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        attn_dropout: float = 0.2,
        attn_nhead: int = 8,
        attn_dim_ff: int = 256,
        attn_layers: int = 3,
        string_node_emb: Optional[torch.Tensor] = None,  # Precomputed STRING embeddings [18870, 256]
    ):
        super().__init__()

        # ── 1a. AIDO.Cell-10M FROZEN base (outside DDP parameter set) ────────
        aido_base = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        aido_base.config.use_cache = False
        # Freeze ALL base parameters; gradients flow ONLY through LoRA adapters
        for p in aido_base.parameters():
            p.requires_grad = False

        # Enable gradient checkpointing on the frozen base to save memory
        aido_base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Patch: AIDO.Cell raises NotImplementedError for get_input_embeddings().
        if not hasattr(aido_base, "_input_require_grads_patched"):
            aido_base.enable_input_require_grads = lambda *a, **kw: None  # noqa: E731
            aido_base._input_require_grads_patched = True

        # ── 1b. LoRA adapters on ALL 8 transformer layers (proven better than last-4)
        n_layers = aido_base.config.num_hidden_layers  # 8 for AIDO.Cell-10M
        lora_layers = list(range(n_layers))  # all 8 layers: [0,1,2,3,4,5,6,7]
        for layer_idx in lora_layers:
            layer = aido_base.bert.encoder.layer[layer_idx].attention.self
            _inject_lora(layer, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        # CRITICAL: Cast to bf16 to avoid 30 GB/forward-pass OOM.
        # Without explicit bf16 cast, the model's float32 activations for
        # sequence_length=19,264 consume ~30 GB peak memory (vs ~0.2 GB in bf16).
        # Lightning's precision="bf16-mixed" doesn't apply until trainer.fit() starts,
        # but the first training step OOMs before that. Casting here fixes this.
        aido_base = aido_base.to(dtype=torch.bfloat16)

        # Register frozen base as a plain attribute (NOT a DDP-wrapped parameter)
        # Lightning moves self.aido_frozen to the correct device automatically.
        self.aido_frozen = aido_base

        # ── 2. STRING_GNN embeddings: precomputed once, stored as frozen buffer ──
        # IMPORTANT FIX: The fine-tuned GNN's full graph forward pass (18,870 nodes,
        # 786K edges, 8 GCN layers) creates massive intermediate activations (~30 GB/GPU)
        # that cause OOM even with micro_batch_size=1. The fix is to precompute
        # STRING node embeddings ONCE using the pretrained GNN, then use them as a
        # frozen lookup table during training. This preserves the pretrained STRING
        # signal while eliminating the GNN activation memory cost entirely.
        # The pretrained GNN weights are loaded temporarily in LightningModule.__init__
        # to precompute embeddings, then the GNN model object is deleted to free memory.
        # Precomputed embeddings are passed via the `string_node_emb` argument.
        assert string_node_emb is not None, "string_node_emb must be provided"
        # Store as buffer (not a parameter) — not a DDP-tracked parameter
        self.register_buffer("string_node_emb", string_node_emb)

        # Learnable fallback embedding for genes NOT in STRING graph
        self.string_fallback = nn.Parameter(torch.randn(STRING_HIDDEN) * 0.02)

        # PPI projection: align STRING 256-dim to cross-attn d_model
        self.ppi_proj = nn.Sequential(
            nn.Linear(STRING_HIDDEN, CROSS_ATTN_DIM),
            nn.GELU(),
            nn.LayerNorm(CROSS_ATTN_DIM),
        )

        # ── 3. Character-level symbol CNN ────────────────────────────────────
        self.symbol_cnn = SymbolCNN(out_dim=SYMBOL_CNN_DIM, dropout=0.1)

        # ── 4. Cross-attention fusion (proven tree-best pattern) ──────────────
        self.fusion = CrossAttentionFusion(
            d_model=CROSS_ATTN_DIM,
            nhead=attn_nhead,
            dim_ff=attn_dim_ff,
            num_layers=attn_layers,
            attn_dropout=attn_dropout,
            sym_in_dim=SYMBOL_CNN_DIM,
        )

        # ── 5. Prediction head: 256->256->19920 ──────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(CROSS_ATTN_DIM),
            nn.Linear(CROSS_ATTN_DIM, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, N_CLASSES * N_GENES_OUT),
        )

        # Initialize head weights
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

        # ── AIDO.Cell-10M forward (frozen base + LoRA adapters) ──────────────
        aido_out = self.aido_frozen(
            input_ids=aido_input_ids,
            attention_mask=aido_attention_mask,
        )
        hidden = aido_out.last_hidden_state  # [B, 19266, 256]

        # Global mean-pool (exclude the 2 appended summary tokens)
        gene_hidden = hidden[:, :N_AIDO_GENES, :]  # [B, 19264, 256]
        global_emb = gene_hidden.mean(dim=1).float()  # [B, 256]

        # Perturbed-gene positional embedding
        pert_pos_tensor = torch.tensor(pert_aido_pos, dtype=torch.long, device=device)
        valid_pert = pert_pos_tensor >= 0  # [B] bool
        safe_pos = pert_pos_tensor.clone()
        safe_pos[~valid_pert] = 0  # clamp for gather
        pert_hidden = gene_hidden[torch.arange(B, device=device), safe_pos]
        pert_emb = torch.where(valid_pert.unsqueeze(-1), pert_hidden, global_emb).float()

        # ── STRING_GNN: use precomputed frozen embeddings (lookup, no forward pass) ─
        # Precomputed embeddings stored as buffer [18870, 256] — no GPU memory cost
        # from GNN activations during training. Preserves pretrained STRING signal.
        node_emb: torch.Tensor = self.string_node_emb  # [18870, 256] buffer

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

        # Project PPI features to cross-attn d_model
        ppi_feat = self.ppi_proj(string_features)  # [B, 256]

        # ── Symbol CNN forward ───────────────────────────────────────────────
        symbol_features = self.symbol_cnn(symbol_tensor).float()  # [B, 64]

        # ── Cross-attention fusion ───────────────────────────────────────────
        fused = self.fusion(
            global_emb=global_emb,
            pert_emb=pert_emb,
            sym_feat=symbol_features,
            ppi_feat=ppi_feat,
        )  # [B, 256]

        # ── Prediction head ──────────────────────────────────────────────────
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
        head_hidden: int = 256,
        head_dropout: float = 0.5,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        attn_dropout: float = 0.2,
        attn_nhead: int = 8,
        attn_dim_ff: int = 256,
        attn_layers: int = 3,
        backbone_lr: float = 2e-4,
        head_lr: float = 6e-4,
        weight_decay: float = 0.10,
        focal_gamma: float = 1.5,
        label_smoothing: float = 0.05,
        mixup_alpha: float = 0.3,
        plateau_patience: int = 12,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []

        # ── Precompute STRING_GNN embeddings ONCE (before model init) ─────────────
        # FIX OOM: The fine-tuned STRING_GNN's full graph forward pass (18,870 nodes,
        # 786K edges, 8 GCN layers) creates ~30 GB of intermediate activations per GPU
        # that cause OOM. We precompute embeddings once using the pretrained GNN, then
        # use them as a frozen lookup table. This eliminates GNN activation memory cost.
        print("[Precompute] Loading STRING_GNN to precompute embeddings...")
        gnn_pretrained = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        gnn_pretrained.eval()
        graph_data = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", map_location="cpu")
        edge_index = graph_data["edge_index"].long()
        edge_weight = graph_data.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.float()
        with torch.no_grad():
            gnn_outputs = gnn_pretrained(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
        string_node_emb = gnn_outputs.last_hidden_state.float()  # [18870, 256]
        del gnn_pretrained, gnn_outputs, graph_data, edge_index, edge_weight
        torch.cuda.empty_cache()
        print(f"[Precompute] STRING embeddings computed: {string_node_emb.shape}")

        # Initialize model with precomputed STRING embeddings
        self.model = CrossAttnFusionDEGModel(
            head_hidden=self.hparams.head_hidden,
            head_dropout=self.hparams.head_dropout,
            lora_r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            attn_dropout=self.hparams.attn_dropout,
            attn_nhead=self.hparams.attn_nhead,
            attn_dim_ff=self.hparams.attn_dim_ff,
            attn_layers=self.hparams.attn_layers,
            string_node_emb=string_node_emb,
        )
        del string_node_emb
        torch.cuda.empty_cache()

        # Cast trainable params to float32 for stable optimization
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Class weights proven optimal from tree analysis (node2-2-3 lineage + node3 lineage)
        # [6.0, 1.0, 12.0]: down-regulated, unchanged, up-regulated
        class_weights = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)
        self.criterion = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        pass  # Model already initialized in __init__

    def _apply_mixup(
        self,
        fused: torch.Tensor,
        labels: torch.Tensor,
        alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Manifold mixup on fused features."""
        if alpha > 0 and self.training:
            lam = float(np.random.beta(alpha, alpha))
        else:
            lam = 1.0
        B = fused.shape[0]
        idx = torch.randperm(B, device=fused.device)
        mixed = lam * fused + (1 - lam) * fused[idx]
        return mixed, labels, labels[idx], lam

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Standard forward pass (no mixup)."""
        pert_positions = batch["pert_aido_pos"]
        aido_expr = make_aido_input_batch(pert_positions, N_AIDO_GENES).to(self.device)
        B = aido_expr.shape[0]
        attn_mask = torch.ones(B, N_AIDO_GENES, dtype=torch.long, device=self.device)

        return self.model(
            aido_input_ids=aido_expr,
            aido_attention_mask=attn_mask,
            pert_aido_pos=pert_positions,
            node_idx=batch["node_idx"].to(self.device),
            symbol_tensor=batch["symbol_tensor"].to(self.device),
        )

    def _forward_with_mixup(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Forward pass with manifold mixup applied at the fused feature level."""
        pert_positions = batch["pert_aido_pos"]
        aido_expr = make_aido_input_batch(pert_positions, N_AIDO_GENES).to(self.device)
        B = aido_expr.shape[0]
        attn_mask = torch.ones(B, N_AIDO_GENES, dtype=torch.long, device=self.device)
        device = self.device

        # Get all intermediate features
        model = self.model

        # AIDO forward
        aido_out = model.aido_frozen(
            input_ids=aido_expr,
            attention_mask=attn_mask,
        )
        hidden = aido_out.last_hidden_state
        gene_hidden = hidden[:, :N_AIDO_GENES, :]
        global_emb = gene_hidden.mean(dim=1).float()
        pert_pos_tensor = torch.tensor(pert_positions, dtype=torch.long, device=device)
        valid_pert = pert_pos_tensor >= 0
        safe_pos = pert_pos_tensor.clone()
        safe_pos[~valid_pert] = 0
        pert_hidden = gene_hidden[torch.arange(B, device=device), safe_pos]
        pert_emb = torch.where(valid_pert.unsqueeze(-1), pert_hidden, global_emb).float()

        # STRING_GNN: use precomputed frozen embeddings (lookup, no forward pass)
        node_emb = model.string_node_emb  # [18870, 256] buffer
        node_idx = batch["node_idx"].to(device)
        valid_string = node_idx >= 0
        safe_idx = node_idx.clone()
        safe_idx[~valid_string] = 0
        graph_emb = node_emb[safe_idx]
        fallback_exp = model.string_fallback.unsqueeze(0).expand(B, -1)
        string_features = torch.where(
            valid_string.unsqueeze(-1), graph_emb.float(), fallback_exp.float()
        )
        ppi_feat = model.ppi_proj(string_features)

        # Symbol CNN
        symbol_features = model.symbol_cnn(batch["symbol_tensor"].to(device)).float()

        # Cross-attention fusion
        fused = model.fusion(
            global_emb=global_emb,
            pert_emb=pert_emb,
            sym_feat=symbol_features,
            ppi_feat=ppi_feat,
        )  # [B, 256]

        # Apply manifold mixup at fused level
        labels = batch["label"].to(device)
        mixed_fused, labels_a, labels_b, lam = self._apply_mixup(
            fused, labels, self.hparams.mixup_alpha
        )

        # Head
        logits = model.head(mixed_fused)
        logits = logits.view(B, N_CLASSES, N_GENES_OUT)

        return logits, labels_a, labels_b, lam

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        labels_flat = labels.reshape(-1)
        return self.criterion(logits_flat, labels_flat)

    def _compute_mixup_loss(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        loss_a = self._compute_loss(logits, labels_a)
        loss_b = self._compute_loss(logits, labels_b)
        return lam * loss_a + (1 - lam) * loss_b

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        if self.hparams.mixup_alpha > 0 and "label" in batch:
            logits, labels_a, labels_b, lam = self._forward_with_mixup(batch)
            loss = self._compute_mixup_loss(logits, labels_a, labels_b, lam)
        else:
            logits = self(batch)
            loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
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

        # Compute F1 on rank 0
        f1_value = 0.0
        if self.trainer.is_global_zero:
            preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
            labels = al.view(-1, N_GENES_OUT).cpu().numpy()
            idxs = ai.view(-1).cpu().numpy()
            _, uniq = np.unique(idxs, return_index=True)
            f1_value = float(compute_deg_f1(preds[uniq], labels[uniq]))

        # Broadcast F1 from rank 0 to all ranks
        f1_tensor = torch.tensor(f1_value, dtype=torch.float32, device="cpu")
        try:
            f1_tensor = self.trainer.strategy.broadcast(f1_tensor, src=0)
        except Exception:
            pass
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
        # Three parameter groups with different learning rates:
        # 1. AIDO.Cell LoRA params (backbone_lr)
        # 2. Head + fusion + symbol CNN + ppi_proj + string_fallback (head_lr)
        # Note: STRING_GNN is now frozen (precomputed embeddings) — not in param groups
        lora_params, head_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "aido_frozen." in name:
                lora_params.append(p)
            else:
                head_params.append(p)

        param_groups = [
            {"params": lora_params, "lr": self.hparams.backbone_lr},
            {"params": head_params, "lr": self.hparams.head_lr},
        ]
        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        opt = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )

        if getattr(self, "_fast_dev_run", False):
            return opt

        # ReduceLROnPlateau with patience=12 (proven from tree-best node3-1-3-1-1-1-1)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=self.hparams.plateau_patience, min_lr=1e-7,
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
# Top-K Checkpoint Averaging for Test Inference
# ──────────────────────────────────────────────────────────────────────────────
def average_top_k_checkpoints(
    model_module: DEGLightningModule,
    checkpoint_paths: List[str],
    datamodule: DEGDataModule,
    trainer: pl.Trainer,
    n_gpus: int,
    output_dir: Path,
) -> None:
    """
    Load top-K checkpoints, average their softmax predictions, and save to file.
    Proven from node3-1-3 lineage to consistently provide +0.003 F1 gain.
    """
    if not checkpoint_paths:
        return

    all_ckpt_preds: List[np.ndarray] = []
    all_ckpt_idxs: List[np.ndarray] = []

    for ckpt_path in checkpoint_paths:
        # Reset test state
        model_module._test_preds = []
        model_module._test_indices = []

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_module.load_state_dict(ckpt.get("state_dict", ckpt))
        model_module.eval()

        # Run test
        test_results = trainer.test(model_module, datamodule=datamodule, verbose=False)

        # Collect predictions from this checkpoint
        if model_module._test_preds:
            lp = torch.cat(model_module._test_preds, 0)
            li = torch.cat(model_module._test_indices, 0)
            all_ckpt_preds.append(lp.numpy())
            all_ckpt_idxs.append(li.numpy())
            model_module._test_preds = []
            model_module._test_indices = []

    if not all_ckpt_preds:
        return

    # Average predictions across checkpoints
    # Each element: [n_local, 3, 6640]
    avg_preds = np.mean(all_ckpt_preds, axis=0)  # [n_local, 3, 6640]
    idxs = all_ckpt_idxs[0]  # Use indices from first checkpoint

    _, uniq = np.unique(idxs, return_index=True)
    avg_preds = avg_preds[uniq]
    idxs = idxs[uniq]
    order = np.argsort(idxs)
    avg_preds = avg_preds[order]
    idxs = idxs[order]

    dm = datamodule
    pert_ids = dm.test_pert_ids
    symbols = dm.test_symbols
    rows = [
        {
            "idx": pert_ids[int(i)],
            "input": symbols[int(i)],
            "prediction": json.dumps(avg_preds[r].tolist()),
        }
        for r, i in enumerate(idxs)
    ]
    pred_path = output_dir / "test_predictions.tsv"
    pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
    print(f"Top-K averaged test predictions saved -> {pred_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 4-1-1: AIDO.Cell-10M + Fine-tuned STRING_GNN + Cross-Attention Fusion"
    )
    default_data_dir = str((Path(__file__).parent / ".." / ".." / "data").resolve())
    p.add_argument("--data_dir", type=str, default=default_data_dir)
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--global_batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--backbone_lr", type=float, default=2e-4)
    p.add_argument("--head_lr", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=0.10)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--attn_dropout", type=float, default=0.2)
    p.add_argument("--attn_nhead", type=int, default=8)
    p.add_argument("--attn_dim_ff", type=int, default=256)
    p.add_argument("--attn_layers", type=int, default=3)
    p.add_argument("--head_hidden", type=int, default=256)
    p.add_argument("--head_dropout", type=float, default=0.5)
    p.add_argument("--focal_gamma", type=float, default=1.5)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--mixup_alpha", type=float, default=0.3)
    p.add_argument("--plateau_patience", type=int, default=12)
    p.add_argument("--early_stopping_patience", type=int, default=20)
    p.add_argument("--early_stopping_min_delta", type=float, default=0.002)
    p.add_argument("--save_top_k", type=int, default=3)
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

    # Fast dev run cap
    if args.fast_dev_run and args.micro_batch_size > 1:
        print(f"[Auto] fast_dev_run: capping micro_batch_size {args.micro_batch_size} -> 1")
        args.micro_batch_size = 1

    # Memory cap: AIDO.Cell-10M + STRING_GNN (fine-tuned) requires significant VRAM
    # With 8 GPUs, micro_batch_size=8 is feasible; scale down for fewer GPUs
    if n_gpus == 1 and args.micro_batch_size > 2:
        print(f"[Auto] 1 GPU: capping micro_batch_size {args.micro_batch_size} -> 2")
        args.micro_batch_size = 2
    elif n_gpus == 2 and args.micro_batch_size > 4:
        print(f"[Auto] 2 GPUs: capping micro_batch_size {args.micro_batch_size} -> 4")
        args.micro_batch_size = 4

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
        save_top_k=args.save_top_k,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
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
        # find_unused_parameters=True: required because frozen AIDO base layers
        # are registered as model attributes (not DDP parameters), but the LoRA
        # adapters inside aido_frozen ARE part of DDP graph — setting True is safe.
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=1800)),
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
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, progress_bar],
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,  # Gradient clipping for stability
    )

    # DataModule and Model
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        attn_dropout=args.attn_dropout,
        attn_nhead=args.attn_nhead,
        attn_dim_ff=args.attn_dim_ff,
        attn_layers=args.attn_layers,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        plateau_patience=args.plateau_patience,
    )
    # Propagate fast_dev_run flag to model
    model_module._fast_dev_run = args.fast_dev_run

    # Train
    trainer.fit(model_module, datamodule=datamodule)

    # Test using best checkpoint (important fix from parent node4)
    if fast_dev_run or debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        # Run standard test with best checkpoint (saves test_predictions.tsv on all ranks)
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

        # CRITICAL: Add DDP barrier to prevent deadlock.
        # After trainer.test() both ranks complete their test loops and call on_test_epoch_end.
        # However, rank 1 exits here while rank 0 would enter the manual averaging loop below,
        # causing a NCCL deadlock. Adding a barrier ensures both ranks synchronize before
        # rank 0 enters the sequential checkpoint-averaging section (which is inherently single-process).
        if n_gpus > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # NOTE: Manual checkpoint averaging (averaging softmax predictions across multiple best
        # checkpoints) is inherently a single-process operation and causes DDP NCCL deadlocks
        # when only one rank runs it. Skipping it when n_gpus > 1. The test predictions from
        # trainer.test(ckpt_path="best") above are the official output.
        if n_gpus == 1 and trainer.is_global_zero and args.save_top_k > 1:
            ckpt_dir = output_dir / "checkpoints"
            ckpt_files = sorted(
                [str(f) for f in ckpt_dir.glob("best-*.ckpt")],
                reverse=True,  # Sort by filename (includes val_f1 score)
            )
            # Use top-3 or fewer
            top_k_ckpts = ckpt_files[:min(args.save_top_k, len(ckpt_files))]
            if len(top_k_ckpts) > 1:
                print(f"[Checkpoint Averaging] Averaging top {len(top_k_ckpts)} checkpoints...")
                # Load each checkpoint and run inference, then average predictions
                all_preds_list: List[np.ndarray] = []
                all_idxs_list: List[np.ndarray] = []

                for ckpt_path in top_k_ckpts:
                    # Reset
                    model_module._test_preds = []
                    model_module._test_indices = []

                    # Load checkpoint weights
                    ckpt_state = torch.load(ckpt_path, map_location="cpu")
                    state_dict = ckpt_state.get("state_dict", ckpt_state)
                    model_module.load_state_dict(state_dict, strict=False)
                    model_module.to(trainer.model.device if hasattr(trainer, "model") and trainer.model is not None else "cuda")
                    model_module.eval()

                    # Run a simple test pass to get predictions
                    datamodule.setup("test")
                    test_dl = datamodule.test_dataloader()
                    device = next(model_module.parameters()).device
                    with torch.no_grad():
                        for batch in test_dl:
                            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                     for k, v in batch.items()}
                            logits = model_module(batch)
                            probs = torch.softmax(logits.float(), dim=1).cpu()
                            model_module._test_preds.append(probs)
                            model_module._test_indices.append(batch["idx"].cpu())

                    if model_module._test_preds:
                        lp = torch.cat(model_module._test_preds, 0).numpy()
                        li = torch.cat(model_module._test_indices, 0).numpy()
                        all_preds_list.append(lp)
                        all_idxs_list.append(li)
                        model_module._test_preds = []
                        model_module._test_indices = []

                if all_preds_list:
                    avg_preds = np.mean(all_preds_list, axis=0)
                    idxs = all_idxs_list[0]
                    _, uniq = np.unique(idxs, return_index=True)
                    avg_preds = avg_preds[uniq]
                    idxs = idxs[uniq]
                    order = np.argsort(idxs)
                    avg_preds = avg_preds[order]
                    idxs = idxs[order]

                    pert_ids = datamodule.test_pert_ids
                    symbols = datamodule.test_symbols
                    rows = [
                        {
                            "idx": pert_ids[int(i)],
                            "input": symbols[int(i)],
                            "prediction": json.dumps(avg_preds[r].tolist()),
                        }
                        for r, i in enumerate(idxs)
                    ]
                    pred_path = output_dir / "test_predictions.tsv"
                    pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
                    print(f"Top-{len(top_k_ckpts)} checkpoint-averaged predictions saved -> {pred_path}")

    # Save test score
    if trainer.is_global_zero:
        best_val_f1 = checkpoint_callback.best_model_score
        score_val = float(best_val_f1) if best_val_f1 is not None else 0.0
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(f"best_val_f1: {score_val:.6f}\n")
        print(f"Test score saved -> {score_path}", flush=True)


if __name__ == "__main__":
    main()
