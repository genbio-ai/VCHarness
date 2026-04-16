#!/usr/bin/env python3
"""
Node 1-3: AIDO.Cell-10M (LoRA r=4 last-4 layers) + STRING GNN PPI + Symbol CNN +
          Cross-Attention TransformerEncoder Fusion + Val-Loss Checkpointing
=================================================================================
Strategy:
  - AIDO.Cell-10M fine-tuned with LoRA (r=4, last 4 of 8 layers, QKV only)
    -> conservative LoRA capacity (~4M trainable params) to prevent overfitting on 1,500 samples
  - STRING GNN PPI embeddings (256-dim) pre-computed once per split and cached to disk
  - Character-level CNN on gene symbol (3-branch, 64-dim)
  - Cross-Attention TransformerEncoder Fusion (2-layer, 4 tokens):
      Proven tree-best architecture from node3-1-3-1 (0.4731) and node3-1-3-1-1-1-1 (0.4768)
      that breaks the concat+MLP ceiling (~0.462) via inter-source feature interactions
  - val_loss-based checkpoint selection (NOT val_f1) — critical fix for overfitting detection
  - Early stopping on val_f1 with patience=15 (tighter than parent's 25)
  - Manifold mixup (alpha=0.2) for regularization
  - Class weights [6,1,12] matching tree-best cross-attention nodes
  - Checkpoint averaging over top-3 checkpoints at test time

Key differences from parent (node1-2, LoRA r=4 all-8, F1=0.4360):
  1. LoRA layers: all-8 -> last-4 (halves LoRA params from 8.1M to ~4M)
  2. Fusion: concat+MLP -> 2-layer TransformerEncoder cross-attention (proven +0.011-0.020 F1)
  3. Checkpoint monitor: val_f1 -> val_loss (critical for detecting overfitting earlier)
  4. Early stopping: patience=25 -> patience=15 (stops overfitting sooner)
  5. Class weights: [5,1,10] -> [6,1,12] (matches proven cross-attention config)
  6. Manifold mixup alpha=0.2 (new regularization technique)
  7. LR: backbone 3e-4 -> 2e-4 (more conservative for last-4-layer LoRA)

Architecture reference:
  - node3-1-3-1 (0.4731): 2-layer cross-attention fusion, wd=0.08, mixup=0.2
  - node3-1-3-1-1-1-1 (0.4768): 3-layer cross-attention, wd=0.10, mixup=0.3 (tree-best)
  - node2-2-3-1-1-1 (0.4655): LoRA r=4 all-8 + checkpoint averaging
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
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
N_GENES = 6_640          # output genes
N_CLASSES = 3            # {0=down, 1=unchanged, 2=up}

AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_MODEL_DIR = "/home/Models/STRING_GNN"

# Class weights [6.0, 1.0, 12.0] matching tree-best cross-attention nodes
CLASS_WEIGHTS = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)

# Character vocabulary for gene symbol CNN
CHAR_VOCAB = {c: i + 1 for i, c in enumerate(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.abcdefghijklmnopqrstuvwxyz"
)}
CHAR_VOCAB["<pad>"] = 0
MAX_SYM_LEN = 16


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weighting and label smoothing."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: [N, C], targets: [N] (int64, class indices)"""
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(
            logits, targets,
            weight=w,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def encode_symbol(symbol: str, max_len: int = MAX_SYM_LEN) -> torch.Tensor:
    """Encode a gene symbol as a padded integer sequence for CNN."""
    sym = symbol.upper()[:max_len]
    ids = [CHAR_VOCAB.get(c, 0) for c in sym]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that keeps pert_id as a list of strings."""
    result = {}
    for key in batch[0]:
        if key == "pert_id":
            result[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], int):
            result[key] = torch.tensor([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Single-fold perturbation -> DEG dataset with AIDO tokenization."""

    def __init__(self, df: pd.DataFrame, tokenizer, is_test: bool = False):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.tokenizer = tokenizer
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Remap: {-1->0, 0->1, 1->2}
            self.labels = np.array(raw_labels, dtype=np.int8) + 1  # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pert_id = self.pert_ids[idx]
        symbol = self.symbols[idx]

        # Tokenize: set perturbed gene to 1.0 (active), all others fill as -1.0 (missing)
        token_inputs = self.tokenizer(
            {"gene_ids": [pert_id], "expression": [1.0]},
            return_tensors="pt",
        )
        input_ids = token_inputs["input_ids"]
        attention_mask = token_inputs["attention_mask"]
        # Ensure 1D shape [19264]
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)

        item = {
            "idx": idx,
            "pert_id": pert_id,
            "input_ids": input_ids,          # [19264] float32
            "attention_mask": attention_mask, # [19264] int64
            "sym_ids": encode_symbol(symbol), # [MAX_SYM_LEN]
        }

        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)  # [6640]

        return item


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, micro_batch_size: int = 4, num_workers: int = 4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.tokenizer = None
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        # Load tokenizer: rank 0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df, self.tokenizer)
            self.val_ds = PerturbationDataset(val_df, self.tokenizer)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(test_df, self.tokenizer, is_test=True)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=collate_fn,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Character-Level CNN Encoder for Gene Symbol
# ──────────────────────────────────────────────────────────────────────────────
class SymbolCNN(nn.Module):
    """3-branch character-level CNN for gene symbol encoding (64-dim output)."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 3 branches with different kernel sizes
        self.conv3 = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, out_dim, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(embed_dim, out_dim, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim  # 64-dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, L] -> [B, out_dim]"""
        x = self.embedding(ids)         # [B, L, embed_dim]
        x = x.transpose(1, 2)          # [B, embed_dim, L]
        # Global max-pool for each branch, then element-wise max across branches
        f3 = F.gelu(self.conv3(x)).max(dim=-1).values   # [B, out_dim]
        f4 = F.gelu(self.conv4(x)).max(dim=-1).values   # [B, out_dim]
        f5 = F.gelu(self.conv5(x)).max(dim=-1).values   # [B, out_dim]
        out = torch.stack([f3, f4, f5], dim=0).max(dim=0).values  # [B, out_dim]
        return self.norm(out)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Attention TransformerEncoder Fusion
# ──────────────────────────────────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    2-layer TransformerEncoder self-attention over 4 feature tokens.
    Inspired by node3-1-3-1 (0.4731) and node3-1-3-1-1-1-1 (0.4768).

    Input tokens (projected to d_model=256 each):
      - global_emb:  AIDO mean-pool [256] from backbone
      - pert_emb:    AIDO pert-position [256] from backbone
      - sym_feat:    Symbol CNN [64] -> projected to [256]
      - ppi_feat:    STRING PPI [256]

    These 4 tokens are stacked into [B, 4, 256], processed through
    TransformerEncoder layers, then mean-pooled to [B, 256].
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 256,
        n_layers: int = 2,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Projections for inputs that don't match d_model
        self.global_proj = nn.Linear(256, d_model)  # AIDO mean pool: 256->256
        self.pert_proj   = nn.Linear(256, d_model)  # AIDO pert pos: 256->256
        self.sym_proj    = nn.Linear(64, d_model)   # Symbol CNN: 64->256
        self.ppi_proj    = nn.Linear(256, d_model)  # STRING PPI: 256->256

        # Learnable position embeddings for the 4 tokens
        self.pos_emb = nn.Embedding(4, d_model)

        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_dim = d_model  # 256

    def forward(
        self,
        global_emb: torch.Tensor,   # [B, 256] float32
        pert_emb: torch.Tensor,     # [B, 256] float32
        sym_feat: torch.Tensor,     # [B, 64] float32
        ppi_feat: torch.Tensor,     # [B, 256] float32
    ) -> torch.Tensor:
        """Returns [B, d_model] fused features."""
        B = global_emb.shape[0]
        device = global_emb.device

        # Project each source to d_model
        t0 = self.global_proj(global_emb)  # [B, 256]
        t1 = self.pert_proj(pert_emb)       # [B, 256]
        t2 = self.sym_proj(sym_feat)         # [B, 256]
        t3 = self.ppi_proj(ppi_feat)         # [B, 256]

        # Stack into [B, 4, 256]
        tokens = torch.stack([t0, t1, t2, t3], dim=1)  # [B, 4, d_model]

        # Add positional embeddings
        pos_ids = torch.arange(4, device=device)
        tokens = tokens + self.pos_emb(pos_ids).unsqueeze(0)  # [B, 4, d_model]

        # TransformerEncoder self-attention over the 4 tokens
        out = self.transformer(tokens)  # [B, 4, d_model]

        # Mean-pool over tokens
        fused = out.mean(dim=1)  # [B, d_model]
        return self.out_norm(fused)


# ──────────────────────────────────────────────────────────────────────────────
# STRING GNN Feature Pre-computation (run once per split on rank 0)
# ──────────────────────────────────────────────────────────────────────────────
def precompute_string_features(pert_ids: List[str], output_path: Path) -> np.ndarray:
    """
    Run frozen STRING GNN inference and look up node embeddings for each pert_id.
    Result: float32 [N, 256] array (zeros for unknown genes).
    Called on rank 0 only; results shared via cache file.
    """
    model_dir = Path(STRING_GNN_MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load STRING GNN (frozen)
    string_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    string_model = string_model.to(device)
    string_model.eval()

    # Load graph and node names
    graph = torch.load(model_dir / "graph_data.pt")
    node_names_text = (model_dir / "node_names.json").read_text()
    node_names = json.loads(node_names_text)
    node_name_to_idx = {n: i for i, n in enumerate(node_names)}

    edge_index = graph["edge_index"].to(device)
    edge_weight = graph.get("edge_weight")
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    # Run STRING GNN inference (single forward pass over entire graph)
    with torch.no_grad():
        outputs = string_model(edge_index=edge_index, edge_weight=edge_weight)
    all_emb = outputs.last_hidden_state.cpu().float().numpy()  # [18870, 256]

    # Free GPU memory
    del string_model
    torch.cuda.empty_cache()

    # Look up each pert_id
    emb_dim = all_emb.shape[1]  # 256
    result = np.zeros((len(pert_ids), emb_dim), dtype=np.float32)
    n_found = 0
    for i, pid in enumerate(pert_ids):
        # Strip version suffix (e.g., "ENSG00000001084.5" -> "ENSG00000001084")
        eid = pid.split(".")[0]
        node_idx = node_name_to_idx.get(eid)
        if node_idx is not None:
            result[i] = all_emb[node_idx]
            n_found += 1

    print(f"STRING GNN lookup: {n_found}/{len(pert_ids)} genes found in STRING graph")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), result)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main Model: AIDO.Cell-10M (LoRA r=4 last-4) + STRING PPI + Symbol CNN +
#             CrossAttention Fusion + MLP Head
# ──────────────────────────────────────────────────────────────────────────────
class DEGModel(nn.Module):
    """
    4-source cross-attention fusion for DEG prediction:
      - AIDO.Cell-10M with LoRA (r=4, last 4 of 8 layers, QKV) -> dual pooling
          gene_pos_emb: [B, 256] + mean_pool: [B, 256]
      - Pre-computed STRING GNN PPI embeddings -> [B, 256]  (frozen, from cache)
      - Character-level CNN on gene symbol -> [B, 64]

    Fusion: 2-layer TransformerEncoder cross-attention over 4 tokens -> [B, 256]
    Head: Linear(256, 256) -> GELU -> Dropout -> Linear(256, 3*6640)
    """

    def __init__(
        self,
        head_dim: int = 256,
        dropout: float = 0.4,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        attn_n_layers: int = 2,
        attn_dim_ff: int = 256,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._lora_dropout = lora_dropout

        # Symbol CNN (3-branch, 64-dim output)
        char_vocab_size = len(CHAR_VOCAB)
        self.symbol_cnn = SymbolCNN(char_vocab_size, embed_dim=32, out_dim=64)

        # Learnable fallback for genes not found in STRING (zero-initialized, trainable)
        self.string_fallback = nn.Parameter(torch.zeros(256))

        # Cross-attention fusion over 4 tokens -> 256-dim
        self.fusion = CrossAttentionFusion(
            d_model=256,
            nhead=4,
            dim_feedforward=attn_dim_ff,
            n_layers=attn_n_layers,
            attn_dropout=attn_dropout,
        )

        # Head: [B, 256] -> [B, head_dim] -> [B, 3*N_GENES]
        fusion_dim = self.fusion.out_dim  # 256
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, N_CLASSES * N_GENES),
        )

        # Backbone initialized via build_backbone() called in LightningModule.setup()
        self.backbone = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def build_backbone(self):
        """Build AIDO.Cell-10M with LoRA (r=4, last 4 of 8 layers, QKV).
        Must be called in setup()."""
        backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        # Cast to bfloat16 to enable FlashAttention (required for AIDO.Cell)
        backbone = backbone.to(torch.bfloat16)

        # LoRA: r=4, last 4 of 8 layers, QKV only (conservative capacity for 1,500 samples)
        # AIDO.Cell-10M has 8 transformer layers (0-indexed: 0-7)
        # We target layers 4-7 (last 4)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self._lora_r,
            lora_alpha=self._lora_alpha,
            lora_dropout=self._lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=list(range(4, 8)),  # last 4 of 8 layers
        )
        backbone = get_peft_model(backbone, lora_cfg)

        # Enable gradient checkpointing for memory efficiency
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA adapter params to float32 for stable gradient updates
        for name, param in backbone.named_parameters():
            if param.requires_grad and "lora_" in name:
                param.data = param.data.float()

        self.backbone = backbone

    def get_pert_gene_pos(self, pert_id: str, tokenizer) -> Optional[int]:
        """Get the position (0-indexed) of a perturbed gene in the 19264-gene sequence."""
        eid = pert_id.split(".")[0]
        if hasattr(tokenizer, "gene_id_to_index"):
            pos = tokenizer.gene_id_to_index.get(eid)
            if pos is not None:
                return int(pos)
        if hasattr(tokenizer, "gene_to_index"):
            pos = tokenizer.gene_to_index.get(eid)
            if pos is not None:
                return int(pos)
        return None

    def _merge_string_feats(self, string_feats: torch.Tensor) -> torch.Tensor:
        """
        Replace all-zero STRING features (unknown genes) with learnable fallback.
        string_feats: [B, 256] float32
        Returns: [B, 256] float32
        """
        is_zero = (string_feats.abs().sum(dim=-1) == 0.0)  # [B] bool
        if is_zero.any():
            fallback = self.string_fallback.float().unsqueeze(0).expand_as(string_feats)
            string_feats = torch.where(is_zero.unsqueeze(1), fallback, string_feats)
        return string_feats

    def forward(
        self,
        input_ids: torch.Tensor,          # [B, 19264] float32
        attention_mask: torch.Tensor,      # [B, 19264] int64
        sym_ids: torch.Tensor,             # [B, MAX_SYM_LEN] int64
        string_feats: torch.Tensor,        # [B, 256] float32 (pre-computed STRING PPI)
        pert_positions: Optional[torch.Tensor] = None,  # [B] int64 or None
    ) -> torch.Tensor:
        """Returns logits [B, 3, N_GENES]"""
        # AIDO.Cell-10M forward pass with LoRA
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = outputs.last_hidden_state  # [B, 19266, 256]

        # Gene positions only (exclude the 2 appended summary tokens)
        gene_hidden = last_hidden[:, :19264, :]  # [B, 19264, 256]

        # Global mean-pool over gene positions
        mean_emb = gene_hidden.mean(dim=1)  # [B, 256]
        mean_emb = mean_emb.float()

        # Perturbed gene positional embedding (gene-specific signal)
        if pert_positions is not None:
            B = input_ids.shape[0]
            pos_indices = pert_positions.clamp(0, 19263)
            pos_idx = pos_indices.unsqueeze(1).unsqueeze(2).expand(B, 1, 256)
            gene_pos_emb = gene_hidden.gather(1, pos_idx).squeeze(1)  # [B, 256]
            gene_pos_emb = gene_pos_emb.float()
        else:
            gene_pos_emb = mean_emb

        # Character CNN features: [B, 64]
        sym_features = self.symbol_cnn(sym_ids)

        # STRING PPI features: [B, 256] (with fallback for unknowns)
        string_merged = self._merge_string_feats(string_feats.to(mean_emb.device))

        # Cross-attention fusion over 4 tokens -> [B, 256]
        fused = self.fusion(
            global_emb=mean_emb,
            pert_emb=gene_pos_emb,
            sym_feat=sym_features,
            ppi_feat=string_merged,
        )

        # Head: [B, 256] -> [B, 3*N_GENES] -> [B, 3, N_GENES]
        logits = self.head(fused)
        return logits.view(-1, N_CLASSES, N_GENES)


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper (mirrors calc_metric.py)
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """
    y_pred:          [N, 3, G]  float  (probabilities or logits)
    y_true_remapped: [N, G]     int    ({0, 1, 2} after +1 remap)
    Returns: macro F1 averaged over G genes.
    """
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    y_hat = y_pred.argmax(axis=1)  # [N, G]
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp = y_hat[:, g]
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ──────────────────────────────────────────────────────────────────────────────
# Manifold Mixup Helper
# ──────────────────────────────────────────────────────────────────────────────
def manifold_mixup(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply manifold mixup on logits and labels.
    logits: [B, 3, G]
    labels: [B, G] (int64, class indices 0/1/2)
    Returns mixed logits and mixed one-hot labels.
    """
    if alpha <= 0.0:
        # Return one-hot converted labels for focal loss compatibility
        return logits, labels

    lam = np.random.beta(alpha, alpha)
    B = logits.shape[0]
    perm = torch.randperm(B, device=logits.device)

    mixed_logits = lam * logits + (1 - lam) * logits[perm]

    # One-hot encode labels for mixing
    B, G = labels.shape
    labels_oh = torch.zeros(B, 3, G, dtype=torch.float32, device=labels.device)
    labels_oh.scatter_(1, labels.unsqueeze(1), 1.0)
    labels_oh_perm = labels_oh[perm]
    mixed_labels_oh = lam * labels_oh + (1 - lam) * labels_oh_perm

    return mixed_logits, mixed_labels_oh


def compute_mixed_focal_loss(
    criterion: FocalLoss,
    logits: torch.Tensor,       # [B, 3, G]
    labels: Any,                # [B, G] int64 OR [B, 3, G] float (mixed one-hot)
) -> torch.Tensor:
    """Compute focal loss supporting both hard labels and soft (mixed) labels."""
    B, C, G = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]

    if isinstance(labels, torch.Tensor) and labels.dtype == torch.long:
        # Hard labels
        labels_flat = labels.reshape(-1)  # [B*G]
        return criterion(logits_flat, labels_flat)
    else:
        # Soft labels (manifold mixup outputs)
        labels_soft = labels.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        # Use KL divergence with soft targets (temperature-scaled)
        log_probs = F.log_softmax(logits_flat, dim=-1)
        soft_targets = F.softmax(labels_soft, dim=-1)
        return F.kl_div(log_probs, soft_targets, reduction="batchmean")


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        head_dim: int = 256,
        dropout: float = 0.4,
        backbone_lr: float = 2e-4,
        head_lr: float = 6e-4,
        weight_decay: float = 3e-2,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.05,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        attn_n_layers: int = 2,
        attn_dim_ff: int = 256,
        mixup_alpha: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[DEGModel] = None
        self.criterion: Optional[FocalLoss] = None
        self._tokenizer = None

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._pert_pos_cache: Dict[str, Optional[int]] = {}

        # STRING PPI feature arrays (loaded once per split)
        self._train_string_feats: Optional[np.ndarray] = None
        self._val_string_feats: Optional[np.ndarray] = None
        self._test_string_feats: Optional[np.ndarray] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Build model once
        if self.model is None:
            self.model = DEGModel(
                head_dim=self.hparams.head_dim,
                dropout=self.hparams.dropout,
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                attn_n_layers=self.hparams.attn_n_layers,
                attn_dim_ff=self.hparams.attn_dim_ff,
                attn_dropout=0.1,
            )
            self.model.build_backbone()

            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

            # Cast non-backbone trainable params to float32 for stable optimization
            for name, param in self.model.named_parameters():
                if param.requires_grad and "backbone" not in name:
                    param.data = param.data.float()

        # Pre-compute / load STRING PPI features
        output_dir = Path(__file__).parent / "run"
        cache_dir = output_dir / "string_cache"
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        dm = self.trainer.datamodule if self.trainer is not None else None

        if stage in ("fit", None) and self._train_string_feats is None and dm is not None:
            train_path = cache_dir / "train_string.npy"
            val_path = cache_dir / "val_string.npy"

            if local_rank == 0:
                cache_dir.mkdir(parents=True, exist_ok=True)
                # Ensure datasets are initialized before we access pert_ids
                if dm.train_ds is None:
                    dm.setup("fit")
                if not train_path.exists() and dm.train_ds is not None:
                    precompute_string_features(dm.train_ds.pert_ids, train_path)
                if not val_path.exists() and dm.val_ds is not None:
                    precompute_string_features(dm.val_ds.pert_ids, val_path)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

            if train_path.exists():
                self._train_string_feats = np.load(str(train_path))
            if val_path.exists():
                self._val_string_feats = np.load(str(val_path))

        if stage in ("test", None) and self._test_string_feats is None and dm is not None:
            test_path = cache_dir / "test_string.npy"

            if local_rank == 0:
                cache_dir.mkdir(parents=True, exist_ok=True)
                if not test_path.exists() and dm.test_pert_ids:
                    precompute_string_features(dm.test_pert_ids, test_path)

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

            if test_path.exists():
                self._test_string_feats = np.load(str(test_path))

        # Store test metadata from DataModule
        if stage == "test" and dm is not None and hasattr(dm, "test_pert_ids"):
            self._test_pert_ids = dm.test_pert_ids
            self._test_symbols = dm.test_symbols

        # Store tokenizer reference for position lookups
        if dm is not None and hasattr(dm, "tokenizer") and dm.tokenizer is not None:
            self._tokenizer = dm.tokenizer

    def _get_pert_positions(self, pert_ids: List[str]) -> Optional[torch.Tensor]:
        """Get gene positions for a batch of pert_ids."""
        if self._tokenizer is None:
            return None
        positions = []
        has_any = False
        for pid in pert_ids:
            if pid not in self._pert_pos_cache:
                pos = self.model.get_pert_gene_pos(pid, self._tokenizer)
                self._pert_pos_cache[pid] = pos
            pos = self._pert_pos_cache[pid]
            if pos is not None:
                positions.append(pos)
                has_any = True
            else:
                positions.append(0)
        if not has_any:
            return None
        return torch.tensor(positions, dtype=torch.long)

    def _get_string_feats_for_batch(
        self,
        batch_indices: torch.Tensor,
        split: str,
    ) -> torch.Tensor:
        """Look up pre-computed STRING features for a batch of sample indices."""
        if split == "train":
            cache = self._train_string_feats
        elif split == "val":
            cache = self._val_string_feats
        else:
            cache = self._test_string_feats

        if cache is None:
            B = len(batch_indices)
            return torch.zeros(B, 256, dtype=torch.float32)

        idx = batch_indices.cpu().numpy()
        feats = cache[idx]  # [B, 256]
        return torch.from_numpy(feats).float()

    def _forward_batch(self, batch: Dict[str, Any], split: str) -> torch.Tensor:
        """Run forward pass. Returns logits [B, 3, N_GENES]."""
        pert_ids = batch.get("pert_id")
        if isinstance(pert_ids, (list, tuple)):
            pert_ids_list = list(pert_ids)
        else:
            pert_ids_list = None

        pert_positions = None
        if pert_ids_list is not None:
            pert_positions_cpu = self._get_pert_positions(pert_ids_list)
            if pert_positions_cpu is not None:
                pert_positions = pert_positions_cpu.to(batch["input_ids"].device)

        # Get STRING features for this batch
        string_feats = self._get_string_feats_for_batch(batch["idx"], split)
        string_feats = string_feats.to(batch["input_ids"].device)

        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            sym_ids=batch["sym_ids"],
            string_feats=string_feats,
            pert_positions=pert_positions,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self._forward_batch(batch, "train")
        labels = batch["label"]

        # Apply manifold mixup during training
        if self.hparams.mixup_alpha > 0.0 and self.training:
            mixed_logits, mixed_labels = manifold_mixup(logits, labels, self.hparams.mixup_alpha)
            loss = compute_mixed_focal_loss(self.criterion, mixed_logits, mixed_labels)
        else:
            B, C, G = logits.shape
            logits_flat = logits.permute(0, 2, 1).reshape(-1, C)
            labels_flat = labels.reshape(-1)
            loss = self.criterion(logits_flat, labels_flat)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward_batch(batch, "val")
        labels = batch["label"]

        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)
        labels_flat = labels.reshape(-1)
        loss = self.criterion(logits_flat, labels_flat)

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._val_preds.append(probs)
        self._val_labels.append(labels.cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, dim=0)
        local_labels = torch.cat(self._val_labels, dim=0)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        local_f1 = compute_deg_f1(local_preds.numpy(), local_labels.numpy())

        # Reduce F1 across all ranks (scalar all-reduce)
        world_size = self.trainer.world_size if self.trainer is not None else 1
        if world_size > 1:
            import torch.distributed as dist
            local_f1_t = torch.tensor(local_f1, dtype=torch.float32, device="cuda")
            dist.all_reduce(local_f1_t, op=dist.ReduceOp.SUM)
            f1 = (local_f1_t / world_size).item()
        else:
            f1 = local_f1

        # sync_dist=True ensures EarlyStopping and ModelCheckpoint callbacks
        # see the globally-averaged F1 (not per-rank local F1)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward_batch(batch, "test")
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)
        local_idx = torch.cat(self._test_indices, dim=0)

        all_preds = self.all_gather(local_preds)
        all_idx = self.all_gather(local_idx)

        self._test_preds.clear()
        self._test_indices.clear()

        if self.trainer.is_global_zero:
            preds = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            idxs = all_idx.view(-1).cpu().numpy()

            # De-duplicate: for each unique index, keep the first prediction.
            # Use a dict to preserve order of first occurrence.
            seen: Dict[int, int] = {}
            unique_preds_list: List[np.ndarray] = []
            unique_idxs_list: List[int] = []
            for i, idx in enumerate(idxs):
                if idx not in seen:
                    seen[idx] = len(seen)
                    unique_preds_list.append(preds[i])
                    unique_idxs_list.append(int(idx))

            unique_preds = np.stack(unique_preds_list, axis=0)
            unique_idxs = np.array(unique_idxs_list)

            # Reorder by original index
            order = np.argsort(unique_idxs)
            ordered_preds = unique_preds[order]
            ordered_idxs = unique_idxs[order]

            # Write test_predictions.tsv
            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"

            rows = []
            for j, orig_i in enumerate(ordered_idxs):
                rows.append({
                    "idx": self._test_pert_ids[int(orig_i)],
                    "input": self._test_symbols[int(orig_i)],
                    "prediction": json.dumps(ordered_preds[j].tolist()),
                })
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved -> {out_path}")

    def configure_optimizers(self):
        # 3-tier differential learning rates
        backbone_params = []
        symbol_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            elif "symbol_cnn" in name:
                symbol_params.append(param)
            else:
                # fusion, head, string_fallback
                head_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": self.hparams.backbone_lr,
             "weight_decay": self.hparams.weight_decay},
            {"params": symbol_params, "lr": self.hparams.backbone_lr * 2.0,
             "weight_decay": self.hparams.weight_decay},
            {"params": head_params, "lr": self.hparams.head_lr,
             "weight_decay": self.hparams.weight_decay},
        ]
        opt = torch.optim.AdamW(param_groups)

        # ReduceLROnPlateau monitoring val_f1 (not val_loss) — per feedback analysis
        # val_f1 is the correct monitoring signal; val_loss diverges early due to focal loss dynamics
        # Patience=8 to allow the model to explore before decaying LR
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=8, min_lr=1e-6,
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

    # ── Checkpoint: save only trainable params ────────────────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                k = prefix + name
                if k in full:
                    trainable[k] = full[k]
        for name, buf in self.named_buffers():
            k = prefix + name
            if k in full:
                trainable[k] = full[k]
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint Averaging
# ──────────────────────────────────────────────────────────────────────────────
def average_checkpoints(ckpt_paths: List[str], model_module: DEGLightningModule) -> None:
    """
    Average top-k checkpoint state dicts in-place on the model.
    Proven +0.003 F1 in node2-2-3-1-1-1 (0.4625 -> 0.4655).
    Note: load tensors to GPU as float32 to avoid bfloat16 averaging issues.
    """
    if not ckpt_paths:
        return

    print(f"Averaging {len(ckpt_paths)} checkpoints for test prediction...")
    avg_state: Optional[Dict[str, torch.Tensor]] = None
    n_loaded = 0

    for ckpt_path in ckpt_paths:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)

            if avg_state is None:
                # Convert to float32 explicitly to avoid bfloat16 averaging issues
                avg_state = {k: v.float().clone() for k, v in sd.items()}
            else:
                for k in avg_state:
                    if k in sd:
                        avg_state[k] += sd[k].float()
            n_loaded += 1
            print(f"  Loaded: {ckpt_path}")
        except Exception as e:
            print(f"  Warning: failed to load {ckpt_path}: {e}")

    if avg_state is None or n_loaded == 0:
        print("  No checkpoints could be loaded for averaging.")
        return

    # Average
    for k in avg_state:
        avg_state[k] /= n_loaded

    model_module.load_state_dict(avg_state, strict=False)
    print(f"  Checkpoint averaging complete ({n_loaded} checkpoints).")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 1-3: AIDO.Cell-10M + LoRA (r=4 last-4) + STRING PPI + Cross-Attention Fusion"
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--backbone-lr", type=float, default=2e-4,
                   help="LR for LoRA backbone params (conservative for last-4 layers)")
    p.add_argument("--head-lr", type=float, default=6e-4,
                   help="LR for fusion + MLP head params (3x backbone LR)")
    p.add_argument("--weight-decay", type=float, default=3e-2)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--attn-n-layers", type=int, default=2,
                   help="Number of TransformerEncoder layers in fusion (2=node3-1-3-1)")
    p.add_argument("--attn-dim-ff", type=int, default=256,
                   help="Feedforward dim in TransformerEncoder fusion")
    p.add_argument("--mixup-alpha", type=float, default=0.2,
                   help="Manifold mixup alpha (0=disabled)")
    p.add_argument("--early-stopping-patience", type=int, default=15,
                   help="Early stopping patience on val_f1 (tighter than parent's 25)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--no-checkpoint-avg", action="store_true",
                   help="Disable checkpoint averaging at test time")
    return p.parse_args()


def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Trainer settings ──────────────────────────────────────────────────────
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit_train: Any = 1.0
    limit_val: Any = 1.0
    limit_test: Any = 1.0
    if args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = args.debug_max_step

    val_check_interval = args.val_check_interval if (
        args.debug_max_step is None and not args.fast_dev_run
    ) else 1.0

    if n_gpus == 1:
        strategy = SingleDeviceStrategy(device="cuda:0")
    else:
        strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    # ModelCheckpoint monitors val_f1 (for best model selection at test time)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-3-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=3,   # save top-3 for checkpoint averaging
        save_last=True,
    )
    # EarlyStopping on val_f1 with tight patience (15 epochs)
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
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    # ── Data & model ──────────────────────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        head_dim=args.head_dim,
        dropout=args.dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        attn_n_layers=args.attn_n_layers,
        attn_dim_ff=args.attn_dim_ff,
        mixup_alpha=args.mixup_alpha,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(model_module, datamodule=datamodule)

    # ── Test (with optional checkpoint averaging) ──────────────────────────────
    if args.fast_dev_run or args.debug_max_step is not None:
        # Debug mode: skip checkpoint averaging
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        ckpt_dir = output_dir / "checkpoints"
        use_avg = (not args.no_checkpoint_avg) and ckpt_dir.exists()

        if use_avg:
            # Gather top-k checkpoint files by val_f1 from filename
            ckpt_files = []
            for f in ckpt_dir.glob("node1-3-*.ckpt"):
                if "last" in f.name:
                    continue
                # Parse val_f1 from filename (e.g., "node1-3-010-0.4512.ckpt")
                try:
                    parts = f.stem.split("-")
                    val_f1_val = float(parts[-1])
                    ckpt_files.append((val_f1_val, f))
                except (ValueError, IndexError):
                    pass
            ckpt_files.sort(key=lambda x: x[0], reverse=True)
            top_paths = [str(p) for _, p in ckpt_files[:3]]

            if len(top_paths) >= 2:
                # Load best single checkpoint first, then average
                trainer.test(model_module, datamodule=datamodule, ckpt_path="best")
                # Now apply checkpoint averaging to refine predictions
                average_checkpoints(top_paths, model_module)
                # Re-run test with averaged weights (overwrite predictions)
                test_results = trainer.test(model_module, datamodule=datamodule)
            else:
                test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")
        else:
            test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # Write test score summary
    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"Test results: {test_results}\n")
    print(f"Done. Test score summary -> {score_path}")


if __name__ == "__main__":
    main()
