#!/usr/bin/env python3
"""
Node 3-1-1-1-2-1: AIDO.Cell-10M + LoRA (r=4, ALL 8 layers) + STRING GNN PPI + Symbol CNN
                   + 3-layer TransformerEncoder Cross-Attention Fusion + Manifold Mixup
=============================================================================
Fixes the catastrophic training collapse from parent (node3-1-1-1-2, test F1=0.4297)
and adopts the proven cross-attention fusion architecture from tree-best node3-1-1-1-1-1-2
(test F1=0.5049).

Key changes from parent:
  1. backbone_lr: 2e-4 → 1e-4 (PRIMARY FIX — prevents catastrophic collapse at epoch 3)
  2. Architecture: MLP head → 3-layer TransformerEncoder cross-attention (TREE-BEST PATTERN)
  3. Manifold mixup: alpha=0.3 (used in every node achieving >0.47 F1)
  4. Focal loss: gamma=2.0+[5,1,10] → gamma=1.5+[6,1,12] (matches tree-best)
  5. weight_decay: 0.03 → 0.10 (proven sweet spot for cross-attention architecture)
  6. LR scheduler: monitor val_f1 (not val_loss) with patience=12 for ReduceLROnPlateau
  7. Add 3-epoch linear warmup for backbone to smooth initial convergence
  8. max_epochs: 80 → 120 (allow breakthrough similar to tree-best at epoch 37)

Architecture:
  Input: pert_id (ENSG) + symbol (5-char string)
  Branch 1: AIDO.Cell-10M + LoRA r=4 ALL 8 layers → dual pool → [B, 512]
  Branch 2: Frozen STRING GNN PPI embeddings → [B, 256]
  Branch 3: Character-level Symbol CNN → [B, 64]

  Fusion: 3 tokens of [B, 256] → 3-layer TransformerEncoder (nhead=8, dim_ff=384)
  Head: Linear(256 → 19920) → reshape [B, 3, 6640]

Total trainable params: ~8.1M
Expected target: Test F1 ≥ 0.48 (approaching or exceeding tree-best 0.5049)
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
FUSION_D_MODEL = 256    # TransformerEncoder d_model (each token is projected to 256-dim)

# Class weights matching tree-best node3-1-1-1-1-1-2: [6, 1, 12]
CLASS_WEIGHTS = torch.tensor([6.0, 1.0, 12.0], dtype=torch.float32)

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
        gamma: float = 1.5,
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
    Outputs 64-dim symbol features. Proven in all top nodes.
    """

    def __init__(self, char_embed_dim: int = 32, num_filters: int = 64, out_dim: int = 64):
        super().__init__()
        self.char_embed = nn.Embedding(CHAR_VOCAB_SIZE, char_embed_dim, padding_idx=0)
        self.conv2 = nn.Conv1d(char_embed_dim, num_filters, kernel_size=2, padding=0)
        self.conv3 = nn.Conv1d(char_embed_dim, num_filters, kernel_size=3, padding=0)
        self.conv4 = nn.Conv1d(char_embed_dim, num_filters, kernel_size=4, padding=0)
        self.proj = nn.Sequential(
            nn.Linear(3 * num_filters, out_dim),
            nn.GELU(),
        )

    def forward(self, symbol_ids: torch.Tensor) -> torch.Tensor:
        # [B, L, char_embed_dim] → [B, char_embed_dim, L] for Conv1d
        x = self.char_embed(symbol_ids).transpose(1, 2)
        f2 = F.gelu(self.conv2(x)).max(dim=2).values
        f3 = F.gelu(self.conv3(x)).max(dim=2).values
        f4 = F.gelu(self.conv4(x)).max(dim=2).values
        fused = torch.cat([f2, f3, f4], dim=1)
        return self.proj(fused)


def encode_symbol(symbol: str) -> List[int]:
    """Convert a gene symbol string to character indices (padded to SYMBOL_LEN)."""
    symbol = symbol.lower()[:SYMBOL_LEN]
    indices = [CHAR_TO_IDX.get(c, 0) for c in symbol]
    while len(indices) < SYMBOL_LEN:
        indices.append(0)
    return indices


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Perturbation dataset with synthetic expression + symbol encoding.
    Labels: {-1,0,1} remapped to {0,1,2} in loss computation.
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
            self.labels = np.array(raw_labels, dtype=np.int8)
        else:
            self.labels = None

        base_expr = torch.ones(N_GENES_MODEL, dtype=torch.float32)
        self._exprs: List[torch.Tensor] = []
        self._pert_positions: List[int] = []
        self._ppi_indices: List[int] = []
        self._symbol_ids: List[List[int]] = []

        covered_aido = 0
        covered_ppi = 0

        for pid, sym in zip(self.pert_ids, self.symbols):
            base_pid = pid.split(".")[0]

            pos = gene_pos_map.get(base_pid, -1)
            self._pert_positions.append(pos)
            if pos >= 0:
                expr = base_expr.clone()
                expr[pos] = 0.0
                covered_aido += 1
            else:
                expr = base_expr.clone()
            self._exprs.append(expr)

            ppi_idx = ppi_idx_map.get(base_pid, -1)
            self._ppi_indices.append(ppi_idx)
            if ppi_idx >= 0:
                covered_ppi += 1

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
            "expr": self._exprs[idx],
            "pert_pos": self._pert_positions[idx],
            "ppi_idx": self._ppi_indices[idx],
            "symbol_ids": torch.tensor(self._symbol_ids[idx], dtype=torch.long),
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
        "expr": torch.stack([b["expr"] for b in batch]),
        "pert_pos": torch.tensor([b["pert_pos"] for b in batch], dtype=torch.long),
        "ppi_idx": torch.tensor([b["ppi_idx"] for b in batch], dtype=torch.long),
        "symbol_ids": torch.stack([b["symbol_ids"] for b in batch]),
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])
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
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

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
# Model — 4-source Cross-Attention Fusion
# ─────────────────────────────────────────────────────────────────────────────
class AIDOCell4SourceCrossAttnDEGModel(nn.Module):
    """
    4-source DEG prediction model using TransformerEncoder cross-attention fusion:

    Branch 1: AIDO.Cell-10M + LoRA (r=4, all 8 layers) → dual pool → 512-dim
    Branch 2: Frozen STRING GNN PPI embeddings → 256-dim
    Branch 3: Symbol CNN → 64-dim

    Fusion: 3-token TransformerEncoder cross-attention (nhead=8, dim_ff=384)
      - Token 0: AIDO aido_pool (512) projected to 256
      - Token 1: PPI embedding (256) — direct
      - Token 2: Symbol CNN (64) projected to 256
      All tokens fused via 3 TransformerEncoder layers → mean pool → 256-dim
      Linear(256 → 19920) → [B, 3, 6640]

    This replicates the tree-best node3-1-1-1-1-1-2 (test F1=0.5049) architecture.
    """

    def __init__(
        self,
        dropout: float = 0.4,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        fusion_n_layers: int = 3,
        fusion_nhead: int = 8,
        fusion_dim_ff: int = 384,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()

        # ── Branch 1: AIDO.Cell-10M backbone with LoRA (all 8 layers) ─────
        backbone = AutoModel.from_pretrained(
            AIDO_MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16
        )
        backbone.config.use_cache = False

        # Monkey-patch to avoid NotImplementedError
        def noop_enable_input_require_grads(self):
            pass
        backbone.enable_input_require_grads = noop_enable_input_require_grads.__get__(
            backbone, type(backbone)
        )

        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=list(range(8)),  # ALL 8 layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Cast LoRA params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Branch 2: Frozen STRING GNN PPI embeddings ─────────────────────
        ppi_model = AutoModel.from_pretrained(
            Path(STRING_MODEL_DIR), trust_remote_code=True
        )
        ppi_model.eval()
        graph = torch.load(Path(STRING_MODEL_DIR) / "graph_data.pt", weights_only=False)
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)
        with torch.no_grad():
            outputs = ppi_model(edge_index=edge_index, edge_weight=edge_weight)
            ppi_emb_matrix = outputs.last_hidden_state  # [18870, 256]
        self.register_buffer("ppi_emb_matrix", ppi_emb_matrix.float())
        del ppi_model, graph

        # Learnable fallback for genes not in STRING graph
        self.ppi_unk_embed = nn.Parameter(torch.zeros(PPI_DIM))
        nn.init.normal_(self.ppi_unk_embed, std=0.02)

        # ── Branch 3: Symbol CNN ─────────────────────────────────────────
        self.symbol_cnn = SymbolCNN(char_embed_dim=32, num_filters=64, out_dim=SYM_DIM)

        # ── Token Projection Layers ──────────────────────────────────────
        # Project each branch feature to FUSION_D_MODEL=256
        # Branch 1: 512 → 256
        self.aido_proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, FUSION_D_MODEL),
            nn.GELU(),
        )
        # Branch 2: 256 → 256 (identity-compatible, but learned)
        self.ppi_proj = nn.Sequential(
            nn.Linear(PPI_DIM, FUSION_D_MODEL),
            nn.GELU(),
        )
        # Branch 3: 64 → 256
        self.sym_proj = nn.Sequential(
            nn.Linear(SYM_DIM, FUSION_D_MODEL),
            nn.GELU(),
        )

        # ── 3-layer TransformerEncoder Fusion ────────────────────────────
        # Pre-LN (norm_first=True) for training stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FUSION_D_MODEL,
            nhead=fusion_nhead,
            dim_feedforward=fusion_dim_ff,
            dropout=fusion_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.fusion_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=fusion_n_layers,
        )
        self.fusion_norm = nn.LayerNorm(FUSION_D_MODEL)

        # ── Output Head ─────────────────────────────────────────────────
        # 256 → 19920 → [B, 3, 6640]
        self.head_dropout = nn.Dropout(dropout)
        self.head_fc = nn.Linear(FUSION_D_MODEL, N_CLASSES * N_GENES_OUT)
        nn.init.trunc_normal_(self.head_fc.weight, std=0.02)
        nn.init.zeros_(self.head_fc.bias)

        # Initialize projection layers
        for proj in [self.aido_proj, self.ppi_proj, self.sym_proj]:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        expr: torch.Tensor,
        pert_pos: torch.Tensor,
        ppi_idx: torch.Tensor,
        symbol_ids: torch.Tensor,
    ) -> torch.Tensor:
        B = expr.shape[0]
        device = expr.device

        # ── Branch 1: AIDO.Cell → dual pool → 512-dim ─────────────────
        outputs = self.backbone(
            input_ids=expr,
            attention_mask=torch.ones(B, N_GENES_MODEL, dtype=torch.long, device=device),
        )
        hidden = outputs.last_hidden_state.float()  # [B, 19266, 256]

        global_pool = hidden[:, :N_GENES_MODEL, :].mean(dim=1)  # [B, 256]
        safe_pos = pert_pos.clamp(min=0)
        pos_idx = safe_pos.view(B, 1, 1).expand(B, 1, HIDDEN_DIM)
        pert_hidden = hidden.gather(1, pos_idx).squeeze(1)  # [B, 256]
        unknown_mask = (pert_pos < 0)
        if unknown_mask.any():
            pert_hidden = pert_hidden.clone()
            pert_hidden[unknown_mask] = global_pool[unknown_mask]
        aido_pool = torch.cat([pert_hidden, global_pool], dim=1)  # [B, 512]

        # ── Branch 2: STRING GNN PPI lookup → 256-dim ─────────────────
        ppi_emb_buf = self.ppi_emb_matrix.to(device)
        safe_ppi = ppi_idx.clamp(min=0)
        ppi_emb = ppi_emb_buf[safe_ppi].clone()  # [B, 256]
        unknown_ppi = (ppi_idx < 0)
        if unknown_ppi.any():
            unk_count = unknown_ppi.sum().item()
            unk_indices = torch.where(unknown_ppi)[0]
            ppi_emb[unk_indices] = self.ppi_unk_embed.to(device).unsqueeze(0).expand(unk_count, -1)

        # ── Branch 3: Symbol CNN → 64-dim ────────────────────────────
        sym_emb = self.symbol_cnn(symbol_ids.to(device))  # [B, 64]

        # ── Project all branches to FUSION_D_MODEL=256 ────────────────
        t0 = self.aido_proj(aido_pool)   # [B, 256]
        t1 = self.ppi_proj(ppi_emb)      # [B, 256]
        t2 = self.sym_proj(sym_emb)      # [B, 256]

        # ── 3-token TransformerEncoder fusion ─────────────────────────
        # Stack tokens: [B, 3, 256]
        tokens = torch.stack([t0, t1, t2], dim=1)  # [B, 3, 256]
        fused = self.fusion_encoder(tokens)         # [B, 3, 256]
        fused = self.fusion_norm(fused)             # [B, 3, 256]

        # Mean pool over tokens → [B, 256]
        fused_pool = fused.mean(dim=1)  # [B, 256]

        # ── Output Head ────────────────────────────────────────────────
        fused_pool = self.head_dropout(fused_pool)
        logits = self.head_fc(fused_pool)          # [B, 19920]
        return logits.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# Manifold Mixup
# ─────────────────────────────────────────────────────────────────────────────
def manifold_mixup(
    features: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Manifold mixup on feature-space tensors.
    Returns mixed features, original labels, shuffled labels, and lambda.
    """
    if alpha <= 0:
        batch_size = features.size(0)
        perm = torch.randperm(batch_size, device=features.device)
        return features, labels, labels[perm], 1.0

    lam = float(np.random.beta(alpha, alpha))
    batch_size = features.size(0)
    perm = torch.randperm(batch_size, device=features.device)
    mixed = lam * features + (1.0 - lam) * features[perm]
    return mixed, labels, labels[perm], lam


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
        backbone_lr: float = 1e-4,   # CRITICAL FIX: 2e-4 → 1e-4
        head_lr: float = 6e-4,
        weight_decay: float = 0.10,  # Proven sweet spot for cross-attention
        focal_gamma: float = 1.5,    # Matches tree-best node3-1-1-1-1-1-2
        label_smoothing: float = 0.05,
        max_epochs: int = 120,
        lr_reduce_patience: int = 12,
        lr_reduce_factor: float = 0.5,
        mixup_alpha: float = 0.3,    # Manifold mixup: proven in all nodes >0.47
        warmup_epochs: int = 3,      # Linear warmup for backbone stability
        fusion_n_layers: int = 3,
        fusion_nhead: int = 8,
        fusion_dim_ff: int = 384,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[AIDOCell4SourceCrossAttnDEGModel] = None
        self.loss_fn: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []
        self._current_epoch_for_warmup: int = 0

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = AIDOCell4SourceCrossAttnDEGModel(
                dropout=self.hparams.dropout,
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                fusion_n_layers=self.hparams.fusion_n_layers,
                fusion_nhead=self.hparams.fusion_nhead,
                fusion_dim_ff=self.hparams.fusion_dim_ff,
                fusion_dropout=self.hparams.fusion_dropout,
            )
            self.loss_fn = FocalLoss(
                gamma=self.hparams.focal_gamma,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

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
        )

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        labels_perm: Optional[torch.Tensor] = None,
        lam: float = 1.0,
    ) -> torch.Tensor:
        """Compute focal loss, optionally with mixup interpolation."""
        labels_remapped = labels + 1  # {-1,0,1} → {0,1,2}
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()  # [B*G, 3]
        labels_flat = labels_remapped.reshape(-1)                       # [B*G]

        if labels_perm is not None and lam < 1.0:
            labels_perm_remapped = labels_perm + 1
            labels_perm_flat = labels_perm_remapped.reshape(-1)
            loss_a = self.loss_fn(logits_flat, labels_flat)
            loss_b = self.loss_fn(logits_flat, labels_perm_flat)
            return lam * loss_a + (1.0 - lam) * loss_b
        else:
            return self.loss_fn(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Get the model's fused representation before the head for mixup
        # We apply mixup at the feature level (manifold mixup)
        # To do this efficiently, we compute features and apply mixup before the head
        B = batch["expr"].shape[0]
        device = batch["expr"].device

        # Get intermediate features (before head) by running branches + fusion
        model = self.model

        # Forward through backbone and projections
        outputs = model.backbone(
            input_ids=batch["expr"],
            attention_mask=torch.ones(B, N_GENES_MODEL, dtype=torch.long, device=device),
        )
        hidden = outputs.last_hidden_state.float()  # [B, 19266, 256]
        global_pool = hidden[:, :N_GENES_MODEL, :].mean(dim=1)
        safe_pos = batch["pert_pos"].clamp(min=0)
        pos_idx = safe_pos.view(B, 1, 1).expand(B, 1, HIDDEN_DIM)
        pert_hidden = hidden.gather(1, pos_idx).squeeze(1)
        unknown_mask = (batch["pert_pos"] < 0)
        if unknown_mask.any():
            pert_hidden = pert_hidden.clone()
            pert_hidden[unknown_mask] = global_pool[unknown_mask]
        aido_pool = torch.cat([pert_hidden, global_pool], dim=1)

        ppi_emb_buf = model.ppi_emb_matrix.to(device)
        safe_ppi = batch["ppi_idx"].clamp(min=0)
        ppi_emb = ppi_emb_buf[safe_ppi].clone()
        unknown_ppi = (batch["ppi_idx"] < 0)
        if unknown_ppi.any():
            unk_count = unknown_ppi.sum().item()
            unk_indices = torch.where(unknown_ppi)[0]
            ppi_emb[unk_indices] = model.ppi_unk_embed.to(device).unsqueeze(0).expand(unk_count, -1)

        sym_emb = model.symbol_cnn(batch["symbol_ids"].to(device))

        t0 = model.aido_proj(aido_pool)
        t1 = model.ppi_proj(ppi_emb)
        t2 = model.sym_proj(sym_emb)

        tokens = torch.stack([t0, t1, t2], dim=1)  # [B, 3, 256]
        fused = model.fusion_encoder(tokens)
        fused = model.fusion_norm(fused)
        fused_pool = fused.mean(dim=1)  # [B, 256]

        # Apply manifold mixup at fused_pool level
        labels = batch["label"]
        mixed_pool, labels_a, labels_b, lam = manifold_mixup(
            fused_pool, labels, alpha=self.hparams.mixup_alpha
        )

        # Head on mixed features
        mixed_pool = model.head_dropout(mixed_pool)
        logits = model.head_fc(mixed_pool).view(B, N_CLASSES, N_GENES_OUT)

        loss = self._compute_loss(logits, labels_a, labels_b, lam)
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

        preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
        labels = al.view(-1, N_GENES_OUT).cpu().numpy()
        idxs = ai.view(-1).cpu().numpy()
        _, uniq = np.unique(idxs, return_index=True)
        labels_remapped = labels[uniq] + 1  # {-1,0,1} → {0,1,2}
        f1_val = compute_deg_f1(preds[uniq], labels_remapped)
        self.log("val_f1", f1_val, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        lp = torch.cat(self._test_preds, 0)
        li = torch.cat(self._test_indices, 0)
        ap = self.all_gather(lp)
        ai = self.all_gather(li)
        self._test_preds.clear()
        self._test_indices.clear()

        if self._test_labels:
            ll = torch.cat(self._test_labels, 0)
            al = self.all_gather(ll)
            self._test_labels.clear()
            has_labels = True
        else:
            has_labels = False

        if self.trainer.is_global_zero:
            preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
            idxs = ai.view(-1).cpu().numpy()
            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]
            idxs = idxs[uniq]
            order = np.argsort(idxs)
            preds = preds[order]
            idxs = idxs[order]

            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "test_pert_ids") and dm.test_pert_ids:
                pert_ids = dm.test_pert_ids
                symbols = dm.test_symbols
            else:
                pert_ids = self._test_pert_ids
                symbols = self._test_symbols

            # Compute test F1 if labels are available
            if has_labels:
                lbls = al.view(-1, N_GENES_OUT).cpu().numpy()
                lbls = lbls[uniq]
                lbls = lbls[order]
                lbls_remapped = lbls + 1  # {-1,0,1} → {0,1,2}
                test_f1 = compute_deg_f1(preds, lbls_remapped)
                self.log("test_f1", test_f1, prog_bar=True, sync_dist=True)
                self.print(f"Test F1: {test_f1:.4f}")

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
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters() if p.requires_grad
        ]
        ppi_unk_params = [self.model.ppi_unk_embed]
        symbol_cnn_params = list(self.model.symbol_cnn.parameters())
        # All projection, fusion, and head parameters
        other_params = (
            list(self.model.aido_proj.parameters()) +
            list(self.model.ppi_proj.parameters()) +
            list(self.model.sym_proj.parameters()) +
            list(self.model.fusion_encoder.parameters()) +
            list(self.model.fusion_norm.parameters()) +
            list(self.model.head_dropout.parameters()) +
            list(self.model.head_fc.parameters())
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params,    "lr": self.hparams.backbone_lr},
                {"params": ppi_unk_params,     "lr": self.hparams.head_lr},
                {"params": symbol_cnn_params,  "lr": self.hparams.head_lr},
                {"params": other_params,       "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
            eps=1e-8,
        )

        # ReduceLROnPlateau monitoring val_f1 (mode=max) — critical: NOT val_loss
        # val_loss oscillates heavily with focal loss + mixup; val_f1 is stable
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.lr_reduce_factor,
            patience=self.hparams.lr_reduce_patience,
            min_lr=1e-8,
        )

        # Note: warmup is handled separately in on_train_epoch_start via manual LR override
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self) -> None:
        """Apply linear warmup for backbone LR during first `warmup_epochs` epochs."""
        warmup_epochs = self.hparams.warmup_epochs
        current_epoch = self.current_epoch
        if warmup_epochs > 0 and current_epoch < warmup_epochs:
            warmup_factor = (current_epoch + 1) / warmup_epochs
            optimizer = self.optimizers()
            if hasattr(optimizer, 'param_groups'):
                # Only scale backbone LR (param group 0)
                target_backbone_lr = self.hparams.backbone_lr * warmup_factor
                optimizer.param_groups[0]['lr'] = target_backbone_lr
                self.log("backbone_lr_warmup", target_backbone_lr, prog_bar=False, sync_dist=True)

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
        description="Node 3-1-1-1-2-1: AIDO.Cell-10M + LoRA + STRING GNN + Symbol CNN + CrossAttn"
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "data"),
    )
    p.add_argument("--micro_batch_size",         type=int,   default=4)
    p.add_argument("--global_batch_size",        type=int,   default=32)
    p.add_argument("--max_epochs",               type=int,   default=120)
    p.add_argument("--backbone_lr",              type=float, default=1e-4)    # CRITICAL FIX
    p.add_argument("--head_lr",                  type=float, default=6e-4)
    p.add_argument("--weight_decay",             type=float, default=0.10)    # Proven sweet spot
    p.add_argument("--dropout",                  type=float, default=0.4)
    p.add_argument("--lora_r",                   type=int,   default=4)
    p.add_argument("--lora_alpha",               type=int,   default=8)
    p.add_argument("--lora_dropout",             type=float, default=0.1)
    p.add_argument("--focal_gamma",              type=float, default=1.5)     # Matches tree-best
    p.add_argument("--label_smoothing",          type=float, default=0.05)
    p.add_argument("--lr_reduce_patience",       type=int,   default=12)
    p.add_argument("--lr_reduce_factor",         type=float, default=0.5)
    p.add_argument("--early_stopping_patience",  type=int,   default=30)
    p.add_argument("--mixup_alpha",              type=float, default=0.3)     # Manifold mixup
    p.add_argument("--warmup_epochs",            type=int,   default=3)
    p.add_argument("--fusion_n_layers",          type=int,   default=3)
    p.add_argument("--fusion_nhead",             type=int,   default=8)
    p.add_argument("--fusion_dim_ff",            type=int,   default=384)
    p.add_argument("--fusion_dropout",           type=float, default=0.1)
    p.add_argument("--num_workers",              type=int,   default=0)
    p.add_argument("--val_check_interval",       type=float, default=1.0)
    p.add_argument("--debug_max_step",           type=int,   default=None)
    p.add_argument("--fast_dev_run",             action="store_true")
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
    # debug_max_step: limit training to N global steps (for rapid pipeline verification).
    # When set, training stops after N steps; validation always uses FULL val set
    # (limit_val_batches=1.0) so val_f1 is accurate for ReduceLROnPlateau.
    # max_steps=-1 means no step limit (Lightning uses max_epochs instead).
    is_debug = args.debug_max_step is not None or args.fast_dev_run
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit = args.debug_max_step if args.debug_max_step is not None else 1.0

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node3-1-1-1-2-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=3,   # Keep top-3 for potential averaging
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
        limit_val_batches=1.0,   # Always use full val set so val_f1 is accurate for LR scheduler
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
        mixup_alpha=args.mixup_alpha,
        warmup_epochs=args.warmup_epochs,
        fusion_n_layers=args.fusion_n_layers,
        fusion_nhead=args.fusion_nhead,
        fusion_dim_ff=args.fusion_dim_ff,
        fusion_dropout=args.fusion_dropout,
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
        # Use best val_f1 from checkpoint callback (test set has no labels, so test F1
        # is computed by EvaluateAgent using test_predictions.tsv + data/test.tsv)
        val_f1_best = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else float("nan")
        )
        score_path.write_text(
            f"# Node 3-1-1-1-2-1 Test Evaluation Results\n"
            f"# Model: AIDO.Cell-10M + LoRA r=4 all-8-layers + STRING GNN PPI + Symbol CNN\n"
            f"# Fusion: 3-layer TransformerEncoder cross-attention (nhead=8, dim_ff=384)\n"
            f"# Training: backbone_lr=1e-4, head_lr=6e-4, mixup_alpha=0.3, wd=0.10\n"
            f"# Loss: FocalLoss(gamma=1.5, weights=[6,1,12])\n"
            f"# Primary metric: f1_score (macro-averaged per-gene F1)\n"
            f"# Note: test_f1 computed by EvaluateAgent using test_predictions.tsv\n"
            f"val_f1_best: {val_f1_best:.6f}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
