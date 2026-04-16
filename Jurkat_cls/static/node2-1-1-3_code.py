#!/usr/bin/env python3
"""
Node 2-1-1-3: AIDO.Cell-10M + LoRA + STRING GNN + Symbol CNN + Cross-Attention Fusion
=======================================================================================
Targeted improvements over sibling node2-1-1-2 (AIDO.Cell-10M + cross-attention, F1=0.4694).

Key changes from sibling:
1. **Reduced mixup alpha** (0.2 → 0.05): Near-disabled manifold mixup. Sibling's feedback
   showed val_loss rising while val_f1 stayed flat — a calibration drift signature of
   manifold mixup + aggressive class weights. node2-1-1-2-1 confirmed: reducing mixup
   to 0.05 + correct calibration gave +0.013 F1 improvement.

2. **Class weights [5,1,10]** instead of [6,1,12]: Slightly less aggressive minority-class
   emphasis. Sibling's [6,1,12] with mixup=0.2 drove overconfident predictions on minority
   classes in training data that did not generalize to test. Tree nodes with [5,1,10] and
   [7,1,15] both work well; at reduced mixup, less aggressive weights are more appropriate.

3. **Tighter early stopping patience=15** (from sibling's 25): Proven effective in
   node2-2-3-1-1-1 (F1=0.4655, patience=15 stopping at epoch ~36). Sibling ran 58 epochs
   with the last 26 epochs providing no improvement. Tighter stopping saves compute
   and avoids deep overfitting.

4. **Higher ReduceLROnPlateau patience=50** (from sibling's 30): Effectively never fires.
   Analysis across 50+ nodes shows: "ReduceLROnPlateau val_f1 with patience that never fires
   is the best scheduler configuration." Sibling's patience=30 didn't fire either, so this
   is a conservative safety margin.

5. **Head dropout=0.40** (from sibling's 0.45): Slightly less regularization in the head.
   With reduced mixup and class weights, the head doesn't need as aggressive dropout.

6. **LoRA lora_dropout=0.05** (from sibling's 0.1): Slightly reduced backbone LoRA dropout
   for more gradient flow through the adapted attention layers.

7. **Keep top-3 checkpoint averaging**: Proven to provide +0.003 variance reduction.

8. **Keep 4-token cross-attention fusion**: Proven effective (node3-1-3-1: 0.4731, sibling: 0.4694).

Architecture (unchanged from sibling):
  - AIDO.Cell-10M backbone (bf16) with LoRA r=4 on all 8 layers (query, key, value)
  - Dual pooling: global mean-pool [B,256] + perturbed-gene positional extraction [B,256]
  - Frozen STRING GNN PPI embeddings (256-dim, pre-computed)
  - Symbol CNN (3-branch char-level, 64-dim) → projected to [B,256]
  - 4-token TransformerEncoder cross-attention fusion (2 layers, 4 heads, d_ff=256)
  - Mean-pool fusion output → [B,256]
  - Manifold mixup alpha=0.05 during training (nearly disabled)
  - Head: LayerNorm → Dropout(0.40) → Linear(256, 3*6640)
  - Focal loss (gamma=2.0, weights=[5,1,10])
  - AdamW differential LR (backbone=2e-4, head=6e-4) + ReduceLROnPlateau(patience=50)
  - Top-3 checkpoint averaging for test predictions
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
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_AIDO = 19_264     # AIDO.Cell vocabulary size
N_GENES_OUT  = 6_640      # output genes for DEG prediction
N_CLASSES    = 3          # {down-regulated, unchanged, up-regulated} → {0, 1, 2}
AIDO_HIDDEN  = 256        # AIDO.Cell-10M hidden dimension

# Class weights: class 0 (down-regulated) ~3.4%, class 1 (unchanged) ~95.5%, class 2 (up-regulated) ~1.1%
# Reduced from sibling's [6,1,12] to [5,1,10] to reduce calibration drift with near-disabled mixup.
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)

# Cross-attention fusion parameters (unchanged from proven sibling)
FUSION_DIM = 256          # dimension of each token in fusion transformer
FUSION_N_LAYERS = 2       # number of transformer encoder layers in fusion
FUSION_NHEADS = 4         # attention heads in fusion transformer
FUSION_DIM_FF = 256       # feed-forward dim in fusion transformer
N_FUSION_TOKENS = 4       # global_emb, pert_emb, sym_proj, ppi_feat


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class weights."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=w, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Symbol CNN
# ──────────────────────────────────────────────────────────────────────────────
class SymbolCNN(nn.Module):
    """3-branch character-level CNN for gene symbol encoding.

    Captures gene family naming conventions (NDUF, KDM, DHX prefixes).
    Input: [B, seq_len] int64 (character indices)
    Output: [B, out_dim] float32
    """

    def __init__(self, vocab_size: int, emb_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # 3 branches: kernel size 2, 3, 4
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(emb_dim, out_dim, kernel_size=k, padding=k // 2),
                nn.GELU(),
            )
            for k in [2, 3, 4]
        ])
        # Project 3*out_dim → out_dim
        self.proj = nn.Linear(3 * out_dim, out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len] int64
        emb = self.emb(x).float()  # [B, seq_len, emb_dim]
        emb = emb.transpose(1, 2)  # [B, emb_dim, seq_len] for Conv1d
        branch_outs = []
        for conv in self.convs:
            h = conv(emb)          # [B, out_dim, seq_len']
            h = h.max(dim=-1)[0]   # global max-pool → [B, out_dim]
            branch_outs.append(h)
        cat = torch.cat(branch_outs, dim=-1)  # [B, 3*out_dim]
        out = self.proj(self.dropout(cat))    # [B, out_dim]
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Attention Fusion Module
# ──────────────────────────────────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """4-token TransformerEncoder for cross-source feature fusion.

    Processes N_FUSION_TOKENS tokens (global_emb, pert_emb, sym_proj, ppi_feat)
    each of dimension FUSION_DIM through a multi-layer self-attention block.
    Mean-pools output tokens to get the final fusion representation.

    This models inter-source interactions: PPI topology vs. cell state,
    gene symbol patterns vs. network neighborhood. Proven effective in:
    - node3-1-3-1: F1=0.4731 (tree-best)
    - node2-1-1-2 (sibling): F1=0.4694
    """

    def __init__(
        self,
        d_model: int = FUSION_DIM,
        nhead: int = FUSION_NHEADS,
        num_layers: int = FUSION_N_LAYERS,
        dim_feedforward: int = FUSION_DIM_FF,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # [B, seq, d_model]
            norm_first=True,   # Pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N_FUSION_TOKENS, FUSION_DIM]
        out = self.transformer(tokens)  # [B, N_FUSION_TOKENS, FUSION_DIM]
        out = self.norm(out)
        return out.mean(dim=1)          # [B, FUSION_DIM] — mean-pool over tokens


# ──────────────────────────────────────────────────────────────────────────────
# Full Model
# ──────────────────────────────────────────────────────────────────────────────
class AidoCellCrossAttnDEGModel(nn.Module):
    """AIDO.Cell-10M backbone with LoRA + STRING GNN + Symbol CNN + Cross-Attention Fusion.

    Forward pass:
      1. AIDO.Cell-10M processes [B, 19264] expression inputs.
         Output: last_hidden_state [B, 19266, 256] — fixed shape.
      2. Global mean-pool over gene positions ([:, :19264, :]) → global_emb [B, 256]
      3. Perturbed gene positional extraction → pert_emb [B, 256]
      4. Symbol CNN → sym_out [B, 64] → projected to sym_feat [B, 256]
      5. STRING GNN PPI lookup → ppi_feat [B, 256]
      6. Optional manifold mixup (alpha=0.05, near-disabled) on tokens
      7. Stack 4 tokens → CrossAttentionFusion (2L) → fusion_emb [B, 256]
      8. Head: LayerNorm → Dropout(0.40) → Linear(256, 3*6640) → [B, 3, 6640]
    """

    def __init__(
        self,
        n_symbol_chars: int,
        n_string_nodes: int,
        ppi_emb_dim: int = 256,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,   # Reduced from sibling's 0.1 for more gradient flow
        head_dropout: float = 0.40,   # Reduced from sibling's 0.45
        symbol_cnn_out: int = 64,
    ):
        super().__init__()

        # ── AIDO.Cell-10M backbone with LoRA ──
        backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=list(range(8)),  # all 8 transformer layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        # Enable gradient checkpointing for memory efficiency
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # ── Symbol CNN ──
        self.symbol_cnn = SymbolCNN(
            vocab_size=n_symbol_chars,
            emb_dim=32,
            out_dim=symbol_cnn_out,
        )
        # Project symbol CNN output to FUSION_DIM
        self.sym_proj = nn.Sequential(
            nn.Linear(symbol_cnn_out, FUSION_DIM),
            nn.GELU(),
        )

        # ── STRING PPI embeddings (frozen, pre-computed, non-persistent) ──
        # Non-persistent: not saved in checkpoints; reloaded from STRING GNN at setup.
        self.register_buffer(
            "ppi_emb",
            torch.zeros(n_string_nodes, ppi_emb_dim, dtype=torch.float32),
            persistent=False,
        )
        # Project PPI to FUSION_DIM (trained)
        self.ppi_proj = nn.Sequential(
            nn.Linear(ppi_emb_dim, FUSION_DIM),
            nn.GELU(),
        )

        # Identity projections (AIDO_HIDDEN == FUSION_DIM == 256 for 10M model)
        self.global_proj = nn.Identity() if AIDO_HIDDEN == FUSION_DIM else nn.Linear(AIDO_HIDDEN, FUSION_DIM)
        self.pert_proj = nn.Identity() if AIDO_HIDDEN == FUSION_DIM else nn.Linear(AIDO_HIDDEN, FUSION_DIM)

        # ── Cross-Attention Fusion ──
        self.fusion = CrossAttentionFusion(
            d_model=FUSION_DIM,
            nhead=FUSION_NHEADS,
            num_layers=FUSION_N_LAYERS,
            dim_feedforward=FUSION_DIM_FF,
            dropout=0.1,
        )

        # ── Prediction head: LayerNorm → Dropout(0.40) → Linear(256, 3*6640) ──
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Dropout(head_dropout),
            nn.Linear(FUSION_DIM, N_CLASSES * N_GENES_OUT),
        )
        # Conservative initialization: small weights prevent early instability
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def set_ppi_embeddings(self, emb: torch.Tensor) -> None:
        """Set pre-computed STRING PPI embeddings (called once in setup)."""
        self.ppi_emb.copy_(emb.float())

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, 19264] float32
        symbol_chars: torch.Tensor,     # [B, seq_len] int64 (character indices)
        pert_positions: torch.Tensor,   # [B] int64 (position of KO gene, -1 if unknown)
        ppi_indices: torch.Tensor,      # [B] int64 (STRING node index, -1 if unknown)
        mixup_lam: Optional[float] = None,
        mixup_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:              # [B, N_CLASSES, N_GENES_OUT]

        # ── AIDO.Cell backbone ──
        out = self.backbone(input_ids=input_ids)
        lhs = out.last_hidden_state.float()  # [B, 19266, 256]

        # Global mean-pool over gene positions (exclude 2 summary tokens)
        global_emb = lhs[:, :N_GENES_AIDO, :].mean(dim=1)  # [B, 256]

        # Perturbed gene positional extraction
        B = input_ids.shape[0]
        pert_emb = torch.zeros(B, AIDO_HIDDEN, device=lhs.device, dtype=lhs.dtype)
        valid_mask = pert_positions >= 0
        if valid_mask.any():
            valid_pos = pert_positions[valid_mask].clamp(0, N_GENES_AIDO - 1)
            pert_emb[valid_mask] = lhs[valid_mask, valid_pos, :]

        # ── Symbol CNN ──
        sym_out = self.symbol_cnn(symbol_chars).float()  # [B, 64]
        sym_feat = self.sym_proj(sym_out)                # [B, 256]

        # ── STRING PPI features ──
        ppi_feat = torch.zeros(B, self.ppi_emb.shape[1], device=lhs.device, dtype=torch.float32)
        valid_ppi = ppi_indices >= 0
        if valid_ppi.any():
            ppi_feat[valid_ppi] = self.ppi_emb[ppi_indices[valid_ppi]]
        ppi_feat = self.ppi_proj(ppi_feat)  # [B, 256]

        # ── Project to fusion dim ──
        g_tok = self.global_proj(global_emb)  # [B, 256]
        p_tok = self.pert_proj(pert_emb)       # [B, 256]

        # ── Manifold mixup (alpha=0.05, near-disabled) ──
        # With near-zero alpha, lam ~ Beta(0.05, 0.05) concentrates near 0 and 1,
        # so most batches effectively have no mixing. This provides minimal regularization
        # benefit without the calibration drift that alpha=0.2 caused in the sibling.
        if mixup_lam is not None and mixup_idx is not None:
            g_tok = mixup_lam * g_tok + (1.0 - mixup_lam) * g_tok[mixup_idx]
            p_tok = mixup_lam * p_tok + (1.0 - mixup_lam) * p_tok[mixup_idx]
            sym_feat = mixup_lam * sym_feat + (1.0 - mixup_lam) * sym_feat[mixup_idx]
            ppi_feat = mixup_lam * ppi_feat + (1.0 - mixup_lam) * ppi_feat[mixup_idx]

        # ── Stack 4 tokens ──
        tokens = torch.stack([g_tok, p_tok, sym_feat, ppi_feat], dim=1)  # [B, 4, 256]

        # ── Cross-attention fusion ──
        fusion_emb = self.fusion(tokens)  # [B, 256]

        # ── Head ──
        logits = self.head(fusion_emb)    # [B, 3 * 6640]
        return logits.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """Perturbation DEG prediction dataset for AIDO.Cell-10M.

    Input convention (proven in best-performing tree nodes):
    - All genes: 1.0 (present, uniform expression)
    - KO gene: 0.0 (knocked out)
    - AIDO.Cell internally handles expression embedding
    """

    MAX_SYMBOL_LEN = 5  # gene symbols are 5 characters

    def __init__(
        self,
        df: pd.DataFrame,
        aido_gene_to_pos: Dict[str, int],
        string_gene_to_idx: Dict[str, int],
        char_vocab: Dict[str, int],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.is_test = is_test
        self.aido_gene_to_pos = aido_gene_to_pos
        self.string_gene_to_idx = string_gene_to_idx
        self.char_vocab = char_vocab

        # Pre-build expression inputs: [N, 19264] float32
        self.expr_inputs = self._build_expr_tensors()

        # Pre-compute pert_positions [N] int64
        self.pert_positions = torch.tensor(
            [aido_gene_to_pos.get(p.split(".")[0], -1) for p in self.pert_ids],
            dtype=torch.long,
        )

        # Pre-compute ppi_indices [N] int64
        self.ppi_indices = torch.tensor(
            [string_gene_to_idx.get(p.split(".")[0], -1) for p in self.pert_ids],
            dtype=torch.long,
        )

        # Pre-build symbol character tensors [N, MAX_SYMBOL_LEN] int64
        self.symbol_chars = self._build_symbol_chars()

        # Labels: {-1,0,1} → {0,1,2}
        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            self.labels = np.array(raw_labels, dtype=np.int8) + 1
        else:
            self.labels = None

    def _build_expr_tensors(self) -> torch.Tensor:
        """Build AIDO.Cell-compatible expression tensors.
        All genes at 1.0 except KO gene at 0.0 (proven convention from best-performing nodes).
        """
        N = len(self.pert_ids)
        expr = torch.ones(N, N_GENES_AIDO, dtype=torch.float32)
        for i, pert_id in enumerate(self.pert_ids):
            base = pert_id.split(".")[0]
            pos = self.aido_gene_to_pos.get(base)
            if pos is not None:
                expr[i, pos] = 0.0
        return expr

    def _build_symbol_chars(self) -> torch.Tensor:
        """Encode gene symbols as fixed-length character index sequences."""
        N = len(self.symbols)
        chars = torch.zeros(N, self.MAX_SYMBOL_LEN, dtype=torch.long)
        for i, sym in enumerate(self.symbols):
            for j, c in enumerate(sym[:self.MAX_SYMBOL_LEN]):
                chars[i, j] = self.char_vocab.get(c, 0)
        return chars

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "input_ids": self.expr_inputs[idx],         # [19264] float32
            "symbol_chars": self.symbol_chars[idx],      # [5] int64
            "pert_position": self.pert_positions[idx],   # scalar int64
            "ppi_index": self.ppi_indices[idx],          # scalar int64
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)  # [6640]
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "symbol_chars": torch.stack([b["symbol_chars"] for b in batch]),
        "pert_positions": torch.stack([b["pert_position"] for b in batch]),
        "ppi_indices": torch.stack([b["ppi_index"] for b in batch]),
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
    """Loads TSV splits and builds AIDO.Cell-10M + STRING GNN datasets."""

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

        # Populated in setup()
        self.aido_gene_to_pos: Dict[str, int] = {}
        self.string_gene_to_idx: Dict[str, int] = {}
        self.char_vocab: Dict[str, int] = {}
        self.n_chars: int = 0
        self.n_string_nodes: int = 0
        self.ppi_embeddings: Optional[torch.Tensor] = None

        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []
        self.test_labels_all: Optional[torch.Tensor] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.aido_gene_to_pos:
            self._build_vocab()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self.aido_gene_to_pos, self.string_gene_to_idx,
                self.char_vocab, is_test=False,
            )
            self.val_ds = PerturbationDataset(
                val_df, self.aido_gene_to_pos, self.string_gene_to_idx,
                self.char_vocab, is_test=False,
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self.aido_gene_to_pos, self.string_gene_to_idx,
                self.char_vocab, is_test=True,
            )
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()
            # Store raw test labels for F1 computation (needed for top-k averaging)
            if "label" in test_df.columns:
                raw_labels = [json.loads(x) for x in test_df["label"].tolist()]
                self.test_labels_all = torch.tensor(
                    np.array(raw_labels, dtype=np.int8) + 1,
                    dtype=torch.long,
                )
            else:
                self.test_labels_all = None

    def _build_vocab(self) -> None:
        """Build AIDO.Cell gene→position, STRING gene→index, and character vocabularies."""
        # 1. Load AIDO.Cell-10M gene_ids.json
        gene_ids_path = Path(AIDO_MODEL_DIR) / "gene_ids.json"
        gene_ids: List[str] = json.load(open(gene_ids_path))
        assert len(gene_ids) == N_GENES_AIDO, f"Expected {N_GENES_AIDO} gene IDs, got {len(gene_ids)}"
        for i, gid in enumerate(gene_ids):
            base = str(gid).split(".")[0]
            self.aido_gene_to_pos[base] = i
        print(f"[DataModule] AIDO.Cell gene vocab: {len(self.aido_gene_to_pos)} entries")

        # 2. Load STRING GNN node names
        string_node_names_path = Path(STRING_GNN_DIR) / "node_names.json"
        if string_node_names_path.exists():
            node_names: List[str] = json.load(open(string_node_names_path))
            for i, gid in enumerate(node_names):
                base = str(gid).split(".")[0]
                self.string_gene_to_idx[base] = i
            self.n_string_nodes = len(node_names)
        else:
            self.n_string_nodes = 1
            print("[DataModule] WARNING: STRING node_names.json not found!")
        print(f"[DataModule] STRING GNN nodes: {self.n_string_nodes}")

        # 3. Load STRING GNN pre-computed PPI embeddings (frozen)
        self._load_ppi_embeddings()

        # 4. Build character vocabulary from all symbols
        all_symbols: List[str] = []
        for split_file in ["train.tsv", "val.tsv", "test.tsv"]:
            p = self.data_dir / split_file
            if p.exists():
                df = pd.read_csv(p, sep="\t")
                all_symbols.extend(df["symbol"].tolist())
        all_chars = sorted(set("".join(all_symbols)))
        self.char_vocab = {"<PAD>": 0}
        self.char_vocab.update({c: i + 1 for i, c in enumerate(all_chars)})
        self.n_chars = len(self.char_vocab)
        print(f"[DataModule] Character vocab size: {self.n_chars}")

    def _load_ppi_embeddings(self) -> None:
        """Load pre-computed STRING GNN embeddings (frozen, run once on rank 0)."""
        graph_data_path = Path(STRING_GNN_DIR) / "graph_data.pt"
        if not graph_data_path.exists():
            print("[DataModule] WARNING: STRING graph_data.pt not found, using zero PPI embeddings!")
            self.ppi_embeddings = torch.zeros(max(self.n_string_nodes, 1), 256, dtype=torch.float32)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        try:
            if local_rank == 0:
                string_model = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
                string_model.eval()

                graph_data = torch.load(graph_data_path, map_location="cpu", weights_only=False)
                edge_index = graph_data["edge_index"]
                edge_weight = graph_data.get("edge_weight", None)

                with torch.no_grad():
                    outputs = string_model(
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                    )
                    ppi_emb = outputs.last_hidden_state.float().cpu()
                self.ppi_embeddings = ppi_emb
                print(f"[DataModule] STRING PPI embeddings: {self.ppi_embeddings.shape}")

                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
            else:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
                # Non-rank-0 workers use zero embeddings (will be set via set_ppi_embeddings)
                self.ppi_embeddings = torch.zeros(max(self.n_string_nodes, 1), 256, dtype=torch.float32)
        except Exception as e:
            print(f"[DataModule] Failed to load STRING GNN: {e}, using zeros")
            self.ppi_embeddings = torch.zeros(max(self.n_string_nodes, 1), 256, dtype=torch.float32)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=(self.num_workers > 0),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro F1, averaged over all 6,640 genes.

    Matches calc_metric.py logic exactly:
    - For each gene g: compute per-class F1, average over present classes
    - Average the per-gene F1 values
    """
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp_class = y_pred[:, :, g]
        yhat = yp_class.argmax(axis=1)
        present = np.array([(yt == c).any() for c in range(N_CLASSES)])
        pf1 = sk_f1_score(yt, yhat, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    """Lightning wrapper for cross-attention fusion DEG predictor.

    Key improvements over sibling node2-1-1-2:
    - mixup_alpha=0.05 (near-disabled, vs sibling's 0.2)
    - class_weights=[5,1,10] (less aggressive, vs sibling's [6,1,12])
    - early stopping patience=15 (tighter, vs sibling's 25)
    - ReduceLROnPlateau patience=50 (never fires, vs sibling's 30)
    - head_dropout=0.40 (vs sibling's 0.45)
    - lora_dropout=0.05 (vs sibling's 0.1)
    """

    def __init__(
        self,
        n_symbol_chars: int,
        n_string_nodes: int,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        head_dropout: float = 0.40,
        lr: float = 2e-4,
        head_lr_multiplier: float = 3.0,
        weight_decay: float = 0.03,
        gamma_focal: float = 2.0,
        mixup_alpha: float = 0.05,
        plateau_patience: int = 50,
        plateau_factor: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AidoCellCrossAttnDEGModel] = None
        self.criterion: Optional[FocalLoss] = None
        self.mixup_alpha = mixup_alpha

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._test_labels_all: Optional[torch.Tensor] = None
        self._topk_output_path: Optional[Path] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = AidoCellCrossAttnDEGModel(
                n_symbol_chars=self.hparams.n_symbol_chars,
                n_string_nodes=self.hparams.n_string_nodes,
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                head_dropout=self.hparams.head_dropout,
            )
            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
            )
            # Cast trainable parameters to float32 for stable AdamW optimization
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        # Load PPI embeddings from datamodule
        if hasattr(self, 'trainer') and self.trainer is not None:
            dm = self.trainer.datamodule
            if dm is not None and dm.ppi_embeddings is not None:
                self.model.set_ppi_embeddings(dm.ppi_embeddings)

        # Retrieve test metadata
        if stage == "test" and hasattr(self, 'trainer') and self.trainer is not None:
            dm = self.trainer.datamodule
            if dm is not None:
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols
                self._test_labels_all = getattr(dm, 'test_labels_all', None)

    def forward(self, batch: Dict[str, Any], mixup_lam=None, mixup_idx=None) -> torch.Tensor:
        return self.model(
            input_ids=batch["input_ids"],
            symbol_chars=batch["symbol_chars"],
            pert_positions=batch["pert_positions"],
            ppi_indices=batch["ppi_indices"],
            mixup_lam=mixup_lam,
            mixup_idx=mixup_idx,
        )

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mixup_lam: Optional[float] = None,
        mixup_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reshape [B, C, G] logits → [B*G, C] for cross-entropy."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()
        if mixup_lam is not None and mixup_idx is not None:
            labels_a_flat = labels.reshape(-1)
            labels_b_flat = labels[mixup_idx].reshape(-1)
            loss_a = self.criterion(logits_flat, labels_a_flat)
            loss_b = self.criterion(logits_flat, labels_b_flat)
            return mixup_lam * loss_a + (1.0 - mixup_lam) * loss_b
        labels_flat = labels.reshape(-1)
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Near-disabled manifold mixup (alpha=0.05)
        mixup_lam, mixup_idx = None, None
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixup_lam = float(lam)
            mixup_idx = torch.randperm(batch["input_ids"].shape[0], device=self.device)

        logits = self(batch, mixup_lam=mixup_lam, mixup_idx=mixup_idx)
        loss = self._compute_loss(logits, batch["label"], mixup_lam=mixup_lam, mixup_idx=mixup_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, 6640]
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

        # Deduplicate by sample index (DDP may produce duplicates)
        _, uniq = np.unique(idxs, return_index=True)
        preds_dedup = preds[uniq]
        labels_dedup = labels[uniq]

        # Compute val_f1 on rank 0, then broadcast to all ranks via NCCL (requires GPU tensor)
        global_rank = getattr(self.trainer, "global_rank", 0)
        if global_rank == 0:
            f1_scalar = compute_deg_f1(preds_dedup, labels_dedup)
            f1_tensor = torch.tensor(f1_scalar, dtype=torch.float32, device="cuda")
        else:
            f1_tensor = torch.zeros(1, dtype=torch.float32, device="cuda")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(f1_tensor, src=0)

        self.log("val_f1", f1_tensor.item(), prog_bar=True, sync_dist=False)

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

        global_rank = getattr(self.trainer, "global_rank", 0)

        # Compute test F1 on rank 0 using stored labels, then broadcast
        test_f1 = torch.zeros(1, dtype=torch.float32, device="cuda")
        if global_rank == 0 and self._test_labels_all is not None:
            preds_np = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
            idxs = ai.cpu().view(-1).numpy()

            _, uniq = np.unique(idxs, return_index=True)
            preds_np = preds_np[uniq]
            idxs = idxs[uniq]
            order = np.argsort(idxs)
            preds_np = preds_np[order]
            idxs_ordered = idxs[order]

            n_total_test = len(self._test_labels_all)
            if len(preds_np) == n_total_test:
                labels_np = self._test_labels_all.cpu().numpy()
                test_f1[0] = compute_deg_f1(preds_np, labels_np)
                print(f"[Test] test_f1 = {test_f1[0].item():.4f} on {len(preds_np)} samples")
            else:
                print(f"[Test] Skipping F1: only {len(preds_np)}/{n_total_test} samples processed")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(test_f1, src=0)

        self.log("test_f1", test_f1.item(), prog_bar=True, sync_dist=False)

        if global_rank == 0:
            preds = ap.cpu().view(-1, N_CLASSES, N_GENES_OUT).numpy()
            idxs = ai.cpu().view(-1).numpy()

            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]
            idxs = idxs[uniq]
            order = np.argsort(idxs)
            preds = preds[order]
            idxs_ordered = idxs[order]

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)

            out_path = self._topk_output_path if self._topk_output_path is not None else output_dir / "test_predictions.tsv"

            rows = []
            for rank_i, orig_i in enumerate(idxs_ordered):
                rows.append({
                    "idx": self._test_pert_ids[orig_i],
                    "input": self._test_symbols[orig_i],
                    "prediction": json.dumps(preds[rank_i].reshape(N_CLASSES, N_GENES_OUT).tolist()),
                })

            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {out_path} (test_f1={test_f1.item():.4f})")

    def configure_optimizers(self):
        """Differential LR: backbone LoRA at base_lr, head+fusion at head_lr_multiplier * base_lr.

        ReduceLROnPlateau(patience=50) on val_f1: essentially never fires throughout training.
        The tree-best nodes consistently had schedulers that never or rarely fired.
        """
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters()
            if p.requires_grad
        ]
        # All other trainable parameters (head, symbol_cnn, ppi_proj, fusion, sym_proj, global_proj, pert_proj)
        head_params = []
        backbone_param_set = set(id(p) for p in backbone_params)
        for n, p in self.model.named_parameters():
            if p.requires_grad and id(p) not in backbone_param_set:
                head_params.append(p)

        base_lr = self.hparams.lr
        head_lr = base_lr * self.hparams.head_lr_multiplier

        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": base_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",            # maximize val_f1
            factor=self.hparams.plateau_factor,
            patience=self.hparams.plateau_patience,  # patience=50: never fires in practice
            min_lr=1e-7,
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
        """Save only trainable parameters + FocalLoss weight buffer.
        ppi_emb is non-persistent and excluded from checkpoints (reloaded at setup).
        """
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        out: Dict[str, Any] = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                k = prefix + name
                if k in full:
                    out[k] = full[k]
        # Save FocalLoss weight buffer (not ppi_emb)
        for name, buf in self.named_buffers():
            k = prefix + name
            if k in full and "weight" in name and "ppi_emb" not in name:
                out[k] = full[k]

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        return out

    def load_state_dict(self, state_dict, strict=True):
        """Non-strict loading: allows missing frozen backbone weights and ppi_emb."""
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 2-1-1-3: AIDO.Cell-10M + LoRA + STRING GNN + Symbol CNN + Cross-Attention Fusion"
    )
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--micro-batch-size", type=int, default=8,
                   help="Per-GPU batch size (default: 8)")
    p.add_argument("--global-batch-size", type=int, default=64,
                   help="Global effective batch size (default: 64)")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Backbone LoRA base learning rate (default: 2e-4)")
    p.add_argument("--head-lr-multiplier", type=float, default=3.0,
                   help="Multiplier for head/fusion LR vs backbone LR (default: 3.0)")
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--head-dropout", type=float, default=0.40)
    p.add_argument("--gamma-focal", type=float, default=2.0)
    p.add_argument("--mixup-alpha", type=float, default=0.05,
                   help="Manifold mixup alpha (0.05 = near-disabled, default: 0.05)")
    p.add_argument("--plateau-patience", type=int, default=50,
                   help="ReduceLROnPlateau patience (50 = effectively never fires, default: 50)")
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--early-stopping-patience", type=int, default=15,
                   help="Early stopping patience on val_f1 (default: 15)")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", "--debug_max_step", type=int, default=None,
                   help="Limit training/val/test to this many global steps for quick debugging")
    p.add_argument("--fast-dev-run", "--fast_dev_run", action="store_true",
                   help="Run 1 batch for train/val/test (PyTorch Lightning unit-test mode)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(0)
    args = parse_args()

    # Resolve data_dir: script lives at mcts/node2-1-1-3/main.py
    if args.data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    else:
        data_dir = Path(args.data_dir)
    args.data_dir = str(data_dir)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # Handle debug flags (hyphen and underscore variants are handled by argparse via dest)
    fast_dev_run = 1 if args.fast_dev_run else False
    _debug_step = args.debug_max_step  # argparse unifies --debug-max-step / --debug_max_step → debug_max_step

    max_steps = _debug_step if _debug_step is not None else -1
    limit = _debug_step if _debug_step is not None else 1.0

    print(
        f"[Config] n_gpus={n_gpus}, micro_batch={args.micro_batch_size}, "
        f"accumulate_grad={accumulate_grad}, "
        f"effective_batch={args.micro_batch_size * n_gpus * accumulate_grad}, "
        f"mixup_alpha={args.mixup_alpha}, "
        f"class_weights=[5,1,10], "
        f"early_stop_patience={args.early_stopping_patience}, "
        f"plateau_patience={args.plateau_patience}"
    )

    # ── Callbacks ──
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node2-1-1-3-{epoch:03d}-val_f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stopping_patience,
        min_delta=0.0,     # stop if no improvement at all (more aggressive than sibling's 0.001)
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # ── Loggers ──
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # ── Trainer ──
    if n_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=180))
    else:
        strategy = "auto"

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
        limit_test_batches=limit,
        val_check_interval=(
            1.0 if (_debug_step is not None or args.fast_dev_run)
            else args.val_check_interval
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=False,  # FlashAttention is non-deterministic
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    # ── DataModule ──
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup()

    # ── LightningModule ──
    model_module = DEGLightningModule(
        n_symbol_chars=datamodule.n_chars,
        n_string_nodes=datamodule.n_string_nodes,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        head_dropout=args.head_dropout,
        lr=args.lr,
        head_lr_multiplier=args.head_lr_multiplier,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        mixup_alpha=args.mixup_alpha,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
    )

    # ── Training ──
    trainer.fit(model_module, datamodule=datamodule)

    # ── Test: top-k checkpoint averaging (or current weights in debug mode) ──
    def _is_rank_zero() -> bool:
        """Check if this is the rank-0 process in DDP (or single-process)."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True  # Single-process mode

    is_rank_zero = _is_rank_zero()

    TOP_K_CKPT = 3  # top-3 checkpoint averaging: proven +0.003 F1 variance reduction

    if args.fast_dev_run or _debug_step is not None:
        # Debug mode: test with current weights (no checkpoint averaging)
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        # Production: top-k checkpoint averaging
        ckpt_dir = output_dir / "checkpoints"
        if ckpt_dir.exists():
            ckpt_files = sorted(
                ckpt_dir.glob("node2-1-1-3-*.ckpt"),
                key=lambda p: float(p.stem.split("val_f1=")[-1]) if "val_f1=" in p.stem else 0.0,
                reverse=True,
            )[:TOP_K_CKPT]
        else:
            ckpt_files = []

        if len(ckpt_files) >= 2:
            # Top-k averaging: run inference with each checkpoint, average predictions
            print(f"[Test] Top-{len(ckpt_files)}-checkpoint averaging:")
            topk_dir = output_dir / "topk_preds"
            topk_dir.mkdir(parents=True, exist_ok=True)

            all_preds_per_ckpt: List[np.ndarray] = []
            all_idxs_per_ckpt: List[np.ndarray] = []

            for ckpt_idx, ckpt_path in enumerate(ckpt_files):
                print(f"  Loading checkpoint: {ckpt_path.name}")
                ckpt_out_path = topk_dir / f"preds_ckpt_{ckpt_idx}.tsv"
                model_module._topk_output_path = ckpt_out_path

                trainer.test(model_module, datamodule=datamodule, ckpt_path=str(ckpt_path))

                model_module._topk_output_path = None  # reset

            # Load and average predictions on rank 0
            if is_rank_zero:
                for ckpt_idx in range(len(ckpt_files)):
                    ckpt_out_path = topk_dir / f"preds_ckpt_{ckpt_idx}.tsv"
                    df = pd.read_csv(ckpt_out_path, sep="\t")
                    preds_list = [np.array(json.loads(row)) for row in df["prediction"].values]
                    preds_arr = np.stack(preds_list, axis=0)  # [N, 3, 6640]
                    idxs_arr = np.arange(len(df))
                    all_preds_per_ckpt.append(preds_arr)
                    all_idxs_per_ckpt.append(idxs_arr)
                    print(f"  Loaded {len(df)} predictions from {ckpt_out_path.name}")

                # Average predictions across checkpoints
                stacked_preds = np.stack(all_preds_per_ckpt, axis=0)  # [K, N, 3, 6640]
                avg_preds = stacked_preds.mean(axis=0)                 # [N, 3, 6640]

                # Write averaged predictions to final output
                rows = []
                for rank_i in range(len(avg_preds)):
                    orig_i = all_idxs_per_ckpt[0][rank_i]
                    rows.append({
                        "idx": model_module._test_pert_ids[int(orig_i)],
                        "input": model_module._test_symbols[int(orig_i)],
                        "prediction": json.dumps(avg_preds[rank_i].reshape(N_CLASSES, N_GENES_OUT).tolist()),
                    })
                out_path = output_dir / "test_predictions.tsv"
                pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
                print(f"[Test] Averaged predictions saved → {out_path}")

                # Compute final F1 from averaged predictions
                if model_module._test_labels_all is not None:
                    n_total_test = len(model_module._test_labels_all)
                    if len(avg_preds) == n_total_test:
                        labels_np = model_module._test_labels_all.cpu().numpy()
                        test_f1 = compute_deg_f1(avg_preds, labels_np)
                        print(f"[Test] Averaged test_f1 = {test_f1:.4f}")
                        test_results = [{"test_f1": float(test_f1)}]
                    else:
                        print(f"[Test] Skipping F1: only {len(avg_preds)}/{n_total_test} samples processed")
                        test_results = [{"test_f1": 0.0}]
                else:
                    test_results = [{"test_f1": 0.0}]
            else:
                test_results = [{"test_f1": 0.0}]

        elif ckpt_files:
            print(f"[Test] Single checkpoint: {ckpt_files[0].name}")
            test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path=str(ckpt_files[0]))
        else:
            print("[Test] No checkpoints found, testing with current weights")
            test_results = trainer.test(model_module, datamodule=datamodule)

    # ── Save test score on rank 0 ──
    if is_rank_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        if test_results:
            test_f1_val = test_results[0].get("test_f1", test_results[0].get("test_loss", "N/A"))
            score_str = f"{test_f1_val}"
        else:
            score_str = "N/A"
        with open(score_path, "w") as f:
            f.write(f"{score_str}\n")
        print(f"Test score written → {score_path}: {score_str}")


if __name__ == "__main__":
    main()
