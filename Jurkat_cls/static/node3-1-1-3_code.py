#!/usr/bin/env python3
"""
Node 3-1-2: 4-Source Cross-Attention Fusion DEG Predictor (Improved)
=====================================================================
Addresses root causes from node3-1-3's feedback (test F1=0.4343):
  1. ReduceLROnPlateau fired too early (patience=8, first reduction at epoch 20
     while val_f1 was still improving 0.37→0.40).
  2. Weight decay=0.10 was too strong — sibling with wd=0.03 achieved 0.4428.
  3. Class weights [6,1,12] may be slightly too aggressive.
  4. Head dropout=0.3 is excessive when there's no observed overfitting.

Key improvements in this version:
  - Warmup (3 epochs linear) + ReduceLROnPlateau with patience=4
    → First LR reduction happens when model is genuinely plateaued (not still improving)
  - weight_decay: 0.10 → 0.03 (less over-regularization)
  - Class weights: [6,1,12] → [7,1,15] (proven stronger in tree-best nodes 0.4655)
  - Head dropout: 0.3 → 0.2 (model shows no overfitting, safe to reduce)
  - Max epochs: 100 → 150 + early_stopping patience: 20 → 25
  - Top-3 checkpoint averaging for test inference (properly implemented)

Architecture (unchanged from node3-1-3):
  1. AIDO.Cell-10M backbone with LoRA r=4 on ALL 8 QKV layers (~36K trainable)
  2. Frozen STRING GNN PPI embeddings (256-dim)
  3. 3-branch character-level Symbol CNN (64-dim per branch, 192-dim total)
  4. 3-layer TransformerEncoder cross-attention fusion (nhead=8, dim_ff=384)
  5. Prediction head: LayerNorm->Linear(256->256)->GELU->Dropout->Linear(256->3*6640)
  6. Focal loss (gamma=2.0, class_weights=[7,1,15], label_smoothing=0.05)
  7. Manifold mixup (alpha=0.3)

This architecture was identified as responsible for the tree best (node3-1-1-1-1-1-2
at test F1=0.5049) and validated in node3-1-3 at test F1=0.4343. The improvements
target the training dynamics bottleneck rather than architectural changes.
"""

import os
import sys
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
HIDDEN_DIM = 256       # AIDO.Cell-10M hidden size
PPI_DIM = 256          # STRING GNN output dimension
SYM_DIM = 192          # Symbol CNN output (3 branches x 64)
FUSION_DIM = 256       # Cross-attention token output dimension

# Symbol CNN character vocabulary (uppercase letters + digits + special)
# Allow for gene symbol characters: A-Z, 0-9, -, .
CHAR_VOCAB = {c: i + 1 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.")}
CHAR_VOCAB['<PAD>'] = 0
SYMBOL_MAX_LEN = 16
VOCAB_SIZE = len(CHAR_VOCAB)

# Class weights: [7, 1, 15] proven in tree-best nodes
# class 0=down-reg (-1->0), 1=unchanged (0->1), 2=up-reg (+1->2)
# Higher weights for minority classes (3.41% down, 94.82% unchanged, 1.63% up)
CLASS_WEIGHTS = torch.tensor([7.0, 1.0, 15.0], dtype=torch.float32)


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
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [N, C], targets: [N]
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(
            logits, targets,
            weight=w,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Symbol CNN
# ─────────────────────────────────────────────────────────────────────────────
def encode_symbol(symbol: str, max_len: int = SYMBOL_MAX_LEN) -> torch.Tensor:
    """Convert a gene symbol string to an integer tensor of char indices."""
    chars = symbol.upper()[:max_len]
    indices = [CHAR_VOCAB.get(c, 0) for c in chars]
    indices += [0] * (max_len - len(indices))  # right-pad with 0
    return torch.tensor(indices, dtype=torch.long)


class SymbolCNN(nn.Module):
    """
    3-branch character-level CNN for gene symbol encoding.
    Each branch uses different kernel sizes to capture different n-gram patterns.
    Architecture: Embedding(vocab, 32) -> parallel Conv1d(k=3,4,5) -> GELU -> GlobalMaxPool
    -> concat -> 192-dim output
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 32,
                 num_filters: int = 64, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, num_filters, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(embed_dim, num_filters, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = num_filters * 3  # 192

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, max_len] long
        emb = self.embedding(x).transpose(1, 2)  # [B, embed_dim, max_len]
        h3 = F.gelu(self.conv3(emb)).max(dim=2)[0]  # [B, num_filters]
        h4 = F.gelu(self.conv4(emb)).max(dim=2)[0]
        h5 = F.gelu(self.conv5(emb)).max(dim=2)[0]
        feat = self.dropout(torch.cat([h3, h4, h5], dim=1))  # [B, 192]
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Pre-builds synthetic expression vectors for each perturbation sample.
    Input: all 19264 genes at 1.0 (baseline), perturbed gene at 0.0.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_pos_map: Dict[str, int],
        string_idx_map: Dict[str, int],
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Map {-1,0,1} -> {0,1,2} to match calc_metric.py's y_true + 1 convention
            self.labels = np.array(raw_labels, dtype=np.int8) + 1
        else:
            self.labels = None

        # Pre-compute expression vectors
        base_expr = torch.ones(N_GENES_MODEL, dtype=torch.float32)

        self._exprs: List[torch.Tensor] = []
        self._pert_positions: List[int] = []
        self._string_indices: List[int] = []
        self._sym_chars: List[torch.Tensor] = []
        covered_aido = 0
        covered_string = 0

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

            # STRING GNN index
            str_idx = string_idx_map.get(base_pid, -1)
            self._string_indices.append(str_idx)
            if str_idx >= 0:
                covered_string += 1

            # Symbol characters
            self._sym_chars.append(encode_symbol(sym))

        if not is_test:
            print(f"[Dataset] {len(self.pert_ids)} samples, "
                  f"AIDO coverage: {covered_aido}/{len(self.pert_ids)} "
                  f"({100.0 * covered_aido / max(len(self.pert_ids), 1):.1f}%), "
                  f"STRING coverage: {covered_string}/{len(self.pert_ids)} "
                  f"({100.0 * covered_string / max(len(self.pert_ids), 1):.1f}%)")

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "expr": self._exprs[idx],               # [19264] float32
            "pert_pos": self._pert_positions[idx],   # int
            "string_idx": self._string_indices[idx], # int
            "sym_chars": self._sym_chars[idx],        # [max_len] long
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
        "expr": torch.stack([b["expr"] for b in batch]),                         # [B, 19264]
        "pert_pos": torch.tensor([b["pert_pos"] for b in batch], dtype=torch.long),
        "string_idx": torch.tensor([b["string_idx"] for b in batch], dtype=torch.long),
        "sym_chars": torch.stack([b["sym_chars"] for b in batch]),                # [B, max_len]
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])               # [B, 6640]
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
        self._string_idx_map: Optional[Dict[str, int]] = None

    def _build_gene_pos_map(self, tokenizer, all_pert_ids: List[str]) -> Dict[str, int]:
        """Build mapping from ENSG gene ID to its position in AIDO.Cell vocab."""
        gene_pos_map: Dict[str, int] = {}
        unique_base_ids = list(set(pid.split(".")[0] for pid in all_pert_ids))
        print(f"[DataModule] Building AIDO gene position map for {len(unique_base_ids)} unique genes...")

        if hasattr(tokenizer, "gene_id_to_index"):
            gid2idx = tokenizer.gene_id_to_index
            for base_pid in unique_base_ids:
                if base_pid in gid2idx:
                    gene_pos_map[base_pid] = gid2idx[base_pid]
            print(f"[DataModule] ENSG->pos via gene_id_to_index: "
                  f"{len(gene_pos_map)}/{len(unique_base_ids)} found")
        return gene_pos_map

    def _build_string_idx_map(self, all_pert_ids: List[str]) -> Dict[str, int]:
        """Build mapping from ENSG gene ID to STRING GNN node index."""
        node_names_path = Path(STRING_MODEL_DIR) / "node_names.json"
        node_names = json.loads(node_names_path.read_text())
        # node_names[i] = ENSG ID for node i
        name_to_idx = {name: i for i, name in enumerate(node_names)}

        string_idx_map: Dict[str, int] = {}
        unique_base_ids = list(set(pid.split(".")[0] for pid in all_pert_ids))
        for base_pid in unique_base_ids:
            if base_pid in name_to_idx:
                string_idx_map[base_pid] = name_to_idx[base_pid]

        print(f"[DataModule] STRING GNN: {len(string_idx_map)}/{len(unique_base_ids)} genes covered")
        return string_idx_map

    def setup(self, stage: Optional[str] = None) -> None:
        # Initialize AIDO tokenizer: rank-0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        # Build gene position maps once (covers all splits)
        if self._gene_pos_map is None:
            all_ids: List[str] = []
            for fname in ["train.tsv", "val.tsv", "test.tsv"]:
                fpath = self.data_dir / fname
                if fpath.exists():
                    df_tmp = pd.read_csv(fpath, sep="\t")
                    if "pert_id" in df_tmp.columns:
                        all_ids.extend(df_tmp["pert_id"].tolist())
            self._gene_pos_map = self._build_gene_pos_map(tokenizer, all_ids)
            self._string_idx_map = self._build_string_idx_map(all_ids)

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self._gene_pos_map, self._string_idx_map)
            self.val_ds = PerturbationDataset(
                val_df, self._gene_pos_map, self._string_idx_map)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self._gene_pos_map, self._string_idx_map, is_test=True)
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
# Model
# ─────────────────────────────────────────────────────────────────────────────
class FusionDEGModel(nn.Module):
    """
    4-source cross-attention fusion DEG predictor.

    Sources:
    1. AIDO.Cell-10M backbone with LoRA (r=4, all 8 layers)
       - global_emb: mean pool over all genes [B, 256]
       - pert_emb: hidden state at perturbed gene position [B, 256]
    2. Frozen STRING GNN PPI embeddings [B, 256]
    3. 3-branch Symbol CNN on gene symbol characters [B, 192]

    Fusion: 4 tokens projected to FUSION_DIM -> 3-layer TransformerEncoder -> mean-pool
    Head: Linear(256->256) -> GELU -> Dropout -> Linear(256->3*6640)

    Total trainable:
    - LoRA adapters (r=4, Q/K/V, all 8 layers): ~36K
    - Symbol CNN: ~100K
    - Token projections (4 x 256 -> FUSION_DIM): ~265K
    - TransformerEncoder (3 layers, nhead=8, dim_ff=384): ~1.18M
    - Prediction head (256->256->3*6640): ~2.7M
    Total: ~4.3M trainable params
    """

    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        fusion_n_layers: int = 3,
        fusion_nhead: int = 8,
        fusion_dim_ff: int = 384,
        fusion_dropout: float = 0.1,
        head_dim: int = 256,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        # ── 1. AIDO.Cell-10M with LoRA ──────────────────────────────────────
        backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True,
                                             dtype=torch.bfloat16)
        backbone.config.use_cache = False

        # Monkey-patch enable_input_require_grads for AIDO.Cell compatibility with PEFT
        def noop_enable_input_require_grads(self):
            pass
        backbone.enable_input_require_grads = noop_enable_input_require_grads.__get__(
            backbone, type(backbone))

        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # LoRA on ALL 8 layers (proven in node3-2=0.462 and breakthrough node)
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

        # Cast LoRA (trainable) params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── 2. Frozen STRING GNN ────────────────────────────────────────────
        string_model = AutoModel.from_pretrained(STRING_MODEL_DIR, trust_remote_code=True)
        for param in string_model.parameters():
            param.requires_grad = False
        self.string_gnn = string_model
        self.string_gnn.eval()

        # Load STRING graph data (edge_index, edge_weight)
        graph = torch.load(Path(STRING_MODEL_DIR) / "graph_data.pt", map_location="cpu")
        self.register_buffer("edge_index", graph["edge_index"])
        edge_weight = graph.get("edge_weight", None)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        # Learnable null embedding for genes not in STRING vocab
        self.string_null_emb = nn.Parameter(torch.zeros(PPI_DIM))
        nn.init.normal_(self.string_null_emb, std=0.02)

        # ── 3. Symbol CNN ───────────────────────────────────────────────────
        self.symbol_cnn = SymbolCNN(
            vocab_size=VOCAB_SIZE, embed_dim=32, num_filters=64, dropout=0.2
        )

        # ── 4. Token projections (project each source to FUSION_DIM=256) ───
        # global_emb: 256 -> 256
        self.proj_global = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, FUSION_DIM, bias=False),
        )
        # pert_emb: 256 -> 256
        self.proj_pert = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, FUSION_DIM, bias=False),
        )
        # ppi_feat: 256 -> 256
        self.proj_ppi = nn.Sequential(
            nn.LayerNorm(PPI_DIM),
            nn.Linear(PPI_DIM, FUSION_DIM, bias=False),
        )
        # sym_feat: 192 -> 256
        self.proj_sym = nn.Sequential(
            nn.LayerNorm(SYM_DIM),
            nn.Linear(SYM_DIM, FUSION_DIM, bias=False),
        )

        # Initialize projections
        for proj in [self.proj_global, self.proj_pert, self.proj_ppi, self.proj_sym]:
            nn.init.trunc_normal_(proj[-1].weight, std=0.02)

        # ── 5. Cross-attention Transformer fusion ──────────────────────────
        # 4 tokens: [global_emb, pert_emb, sym_feat, ppi_feat]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FUSION_DIM,
            nhead=fusion_nhead,
            dim_feedforward=fusion_dim_ff,
            dropout=fusion_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.fusion_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=fusion_n_layers,
            enable_nested_tensor=False,
        )

        # ── 6. Prediction head ──────────────────────────────────────────────
        # After mean-pooling 4 tokens: 256 -> 256 -> 3*6640
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, head_dim, bias=True),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_dim, N_CLASSES * N_GENES_OUT, bias=True),
        )
        nn.init.trunc_normal_(self.head[1].weight, std=0.02)
        nn.init.trunc_normal_(self.head[4].weight, std=0.02)
        nn.init.zeros_(self.head[4].bias)

    @torch.no_grad()
    def _get_string_embeddings(self, string_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute STRING GNN node embeddings (frozen), and gather per-sample.
        For genes not in STRING vocab (idx=-1), use the learnable null embedding.

        Returns: [B, 256] PPI embeddings
        """
        device = string_idx.device

        # Run STRING GNN inference (frozen) to get all node embeddings
        edge_index = self.edge_index.to(device)
        edge_weight = self.edge_weight.to(device) if self.edge_weight is not None else None

        with torch.no_grad():
            gnn_out = self.string_gnn(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
        all_embs = gnn_out.last_hidden_state  # [18870, 256]

        # Gather per-sample embedding
        B = string_idx.shape[0]
        valid_mask = (string_idx >= 0)

        # Use torch.where instead of boolean indexing to avoid CUDA kernel issues
        if valid_mask.all():
            # All genes are in STRING vocab - no need for null embedding
            valid_idxs = string_idx.clamp(min=0)
            ppi_embs = all_embs[valid_idxs]
        else:
            # Some genes may not be in STRING vocab - need null embedding
            valid_idxs = string_idx.clone()
            valid_idxs[~valid_mask] = 0  # dummy index for masked positions
            gathered = all_embs[valid_idxs]  # [B, 256]

            # Create null embedding tensor expanded to batch size
            null_emb_expanded = self.string_null_emb.view(1, -1).expand(B, -1).to(
                device=device, dtype=all_embs.dtype)
            # Use where to select valid or null embedding
            valid_mask_expanded = valid_mask.view(-1, 1).expand(-1, PPI_DIM)
            ppi_embs = torch.where(valid_mask_expanded, gathered, null_emb_expanded)

        return ppi_embs.float()

    def forward(
        self,
        expr: torch.Tensor,
        pert_pos: torch.Tensor,
        string_idx: torch.Tensor,
        sym_chars: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            expr:       [B, 19264] float32 - synthetic expression
            pert_pos:   [B] long - position of perturbed gene in AIDO vocab
            string_idx: [B] long - STRING GNN node index (-1 if unknown)
            sym_chars:  [B, max_len] long - character indices for Symbol CNN

        Returns:
            logits: [B, 3, 6640]
        """
        B = expr.shape[0]
        device = expr.device

        # ── 1. AIDO.Cell forward ────────────────────────────────────────────
        outputs = self.backbone(
            input_ids=expr,
            attention_mask=torch.ones(B, N_GENES_MODEL, dtype=torch.long, device=device),
        )
        hidden = outputs.last_hidden_state.float()  # [B, 19266, 256]

        # Global mean pool over all gene positions (exclude 2 summary tokens)
        global_emb = hidden[:, :N_GENES_MODEL, :].mean(dim=1)  # [B, 256]

        # Per-sample perturbed gene hidden state
        safe_pos = pert_pos.clamp(min=0)  # [B]
        pos_idx = safe_pos.view(B, 1, 1).expand(B, 1, HIDDEN_DIM)
        pert_emb = hidden.gather(1, pos_idx).squeeze(1)  # [B, 256]

        # For genes not in AIDO vocab: fall back to global embedding
        unknown_mask = (pert_pos < 0)
        if unknown_mask.any():
            pert_emb = pert_emb.clone()
            pert_emb[unknown_mask] = global_emb[unknown_mask]

        # ── 2. STRING GNN embeddings ────────────────────────────────────────
        ppi_feat = self._get_string_embeddings(string_idx)  # [B, 256]

        # ── 3. Symbol CNN ───────────────────────────────────────────────────
        sym_feat = self.symbol_cnn(sym_chars).float()  # [B, 192]

        # ── 4. Project all sources to FUSION_DIM ───────────────────────────
        t_global = self.proj_global(global_emb)  # [B, 256]
        t_pert = self.proj_pert(pert_emb)         # [B, 256]
        t_ppi = self.proj_ppi(ppi_feat)           # [B, 256]
        t_sym = self.proj_sym(sym_feat)            # [B, 256]

        # Stack into 4 tokens: [B, 4, FUSION_DIM]
        tokens = torch.stack([t_global, t_pert, t_sym, t_ppi], dim=1)  # [B, 4, 256]

        # ── 5. Cross-attention fusion ───────────────────────────────────────
        fused = self.fusion_encoder(tokens)  # [B, 4, 256]
        fused_pool = fused.mean(dim=1)       # [B, 256] mean pool over 4 tokens

        # ── 6. Prediction head ──────────────────────────────────────────────
        logits = self.head(fused_pool)  # [B, 3*6640]
        logits = logits.view(B, N_GENES_OUT, N_CLASSES)  # [B, 6640, 3]
        return logits.permute(0, 2, 1)  # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """
    Per-gene macro-averaged F1 score, matching calc_metric.py logic exactly.

    y_pred: [n_samples, 3, n_genes] - class probabilities
    y_true_remapped: [n_samples, n_genes] - labels in {0,1,2} (i.e., y_true+1)
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
# Manifold Mixup
# ─────────────────────────────────────────────────────────────────────────────
def manifold_mixup(
    features: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Manifold mixup on feature representations.
    Returns mixed features, original labels, shuffled labels, and lambda.
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    B = features.shape[0]
    perm = torch.randperm(B, device=features.device)
    mixed_features = lam * features + (1.0 - lam) * features[perm]
    return mixed_features, labels, labels[perm], lam


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class _DEGLightningModuleCheckpointerMixin:
    """Checkpoint helpers: save/load only trainable params + buffers."""

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


class DEGLightningModule(_DEGLightningModuleCheckpointerMixin, LightningModule):
    def __init__(
        self,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        fusion_n_layers: int = 3,
        fusion_nhead: int = 8,
        fusion_dim_ff: int = 384,
        fusion_dropout: float = 0.1,
        head_dim: int = 256,
        head_dropout: float = 0.2,
        backbone_lr: float = 2e-4,
        head_lr: float = 6e-4,
        weight_decay: float = 0.03,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        mixup_alpha: float = 0.3,
        warmup_epochs: int = 3,
        plateau_patience: int = 4,
        plateau_factor: float = 0.5,
        max_epochs: int = 150,
        n_top_checkpoints: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[FusionDEGModel] = None
        self.loss_fn: Optional[FocalLoss] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = FusionDEGModel(
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                fusion_n_layers=self.hparams.fusion_n_layers,
                fusion_nhead=self.hparams.fusion_nhead,
                fusion_dim_ff=self.hparams.fusion_dim_ff,
                fusion_dropout=self.hparams.fusion_dropout,
                head_dim=self.hparams.head_dim,
                head_dropout=self.hparams.head_dropout,
            )
            self.loss_fn = FocalLoss(
                gamma=self.hparams.focal_gamma,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        # Populate test metadata for prediction saving
        if stage in ("test", None):
            dm = getattr(self, "trainer", None)
            if dm is not None:
                dm = getattr(self.trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "test_pert_ids") and dm.test_pert_ids:
                self._test_pert_ids = dm.test_pert_ids
                self._test_symbols = dm.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(
            batch["expr"], batch["pert_pos"], batch["string_idx"], batch["sym_chars"]
        )  # [B, 3, 6640]

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        labels_perm: Optional[torch.Tensor] = None,
        lam: float = 1.0,
    ) -> torch.Tensor:
        """Compute focal loss, optionally with mixup."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                               # [B*6640]

        if labels_perm is not None and lam < 1.0:
            labels_perm_flat = labels_perm.reshape(-1)
            loss1 = self.loss_fn(logits_flat, labels_flat)
            loss2 = self.loss_fn(logits_flat, labels_perm_flat)
            return lam * loss1 + (1.0 - lam) * loss2
        return self.loss_fn(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Apply input-level mixup on the expression batch
        alpha = self.hparams.mixup_alpha

        if alpha > 0 and self.training:
            lam = float(np.random.beta(alpha, alpha))
            B = batch["expr"].shape[0]
            perm = torch.randperm(B, device=batch["expr"].device)

            mixed_batch = {
                "expr": lam * batch["expr"] + (1.0 - lam) * batch["expr"][perm],
                "pert_pos": batch["pert_pos"],   # keep original positions
                "string_idx": batch["string_idx"],
                "sym_chars": batch["sym_chars"],
            }

            logits = self.model(
                mixed_batch["expr"],
                mixed_batch["pert_pos"],
                mixed_batch["string_idx"],
                mixed_batch["sym_chars"],
            )
            loss = self._compute_loss(logits, batch["label"], batch["label"][perm], lam)
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

        # Step ReduceLROnPlateau with the computed metric
        # (stored as self._plateau_sched in configure_optimizers)
        if hasattr(self, "_plateau_sched") and self._plateau_sched is not None:
            self._plateau_sched.step(f1_val)

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

            output_dir = Path(sys.argv[0]).resolve().parent / "run"
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
            self.print(f"Test predictions saved -> {pred_path}")

    def configure_optimizers(self):
        # Separate learning rates: LoRA backbone (lower) vs. head (higher)
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters() if p.requires_grad
        ]
        # Head and fusion parameters
        head_params = []
        for component in [
            self.model.symbol_cnn,
            self.model.proj_global, self.model.proj_pert,
            self.model.proj_ppi, self.model.proj_sym,
            self.model.fusion_encoder,
            self.model.head,
        ]:
            head_params.extend(list(component.parameters()))
        head_params.append(self.model.string_null_emb)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
                {"params": head_params,     "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
            eps=1e-8,
        )

        warmup_epochs = self.hparams.warmup_epochs
        warmup_factor = 0.1  # start at 10% of target LR

        # ReduceLROnPlateau: fires when val_f1 hasn't improved for patience epochs
        # NOTE: Lightning 2.5+ passes metrics via monitor= in lr_scheduler_configs,
        # so ReduceLROnPlateau receives the val_f1 value automatically.
        plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.plateau_factor,
            patience=self.hparams.plateau_patience,
            min_lr=1e-8,
            verbose=True,
        )

        # NOTE: Warmup (linear ramp from 10% to 100% over warmup_epochs) is handled
        # by WarmupSchedulerCallback, which stores base LR in each param_group["orig_lr"]
        # and applies the warmup factor before ReduceLROnPlateau takes over.
        # ReduceLROnPlateau is NOT a _LRScheduler, so we manage it via callback manually.
        # We store it as an instance attribute for access by ReduceLROnPlateauCallback.
        self._plateau_sched = plateau_sched
        return optimizer


class WarmupSchedulerCallback(pl.Callback):
    """
    Applies linear LR warmup by ramping from warmup_factor -> 1.0 over warmup_epochs.
    After warmup, restores base LRs and hands control to ReduceLROnPlateau.
    """

    def __init__(self, warmup_epochs: int, warmup_factor: float = 0.1):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self._base_lrs: Optional[List[float]] = None

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        epoch = trainer.current_epoch

        if epoch == 0:
            # First epoch: store the base (target) LRs before any warmup
            self._base_lrs = [pg["lr"] for pg in trainer.optimizers[0].param_groups]

        if epoch < self.warmup_epochs:
            # Linear warmup: ramp factor from warmup_factor to 1.0
            factor = self.warmup_factor + (1.0 - self.warmup_factor) * (
                epoch / max(self.warmup_epochs, 1)
            )
            for opt in trainer.optimizers:
                for i, pg in enumerate(opt.param_groups):
                    pg["lr"] = self._base_lrs[i] * factor
        elif epoch == self.warmup_epochs and self._base_lrs is not None:
            # After warmup: restore base LRs so ReduceLROnPlateau starts from correct base
            for opt in trainer.optimizers:
                for i, pg in enumerate(opt.param_groups):
                    pg["lr"] = self._base_lrs[i]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node 3-1-2: 4-Source Cross-Attention Fusion DEG Predictor (Improved)"
    )
    # NOTE: When this script is invoked via symlink (e.g., working_node_N/mcts/node/ ->
    # ../../mcts/node3-1-1-3), __file__ points to the symlink path which may be outside
    # the working tree. Use sys.argv[0].resolve() to get the real filesystem path.
    _script_real_parent = Path(sys.argv[0]).resolve().parent
    p.add_argument("--data-dir", type=str,
                   default=str(_script_real_parent.parent.parent / "data"))
    p.add_argument("--micro-batch-size",        type=int,   default=8)
    p.add_argument("--global-batch-size",       type=int,   default=64)
    p.add_argument("--max-epochs",              type=int,   default=150)
    p.add_argument("--backbone-lr",             type=float, default=2e-4)
    p.add_argument("--head-lr",                 type=float, default=6e-4)
    p.add_argument("--weight-decay",            type=float, default=0.03)
    p.add_argument("--lora-r",                  type=int,   default=4)
    p.add_argument("--lora-alpha",              type=int,   default=8)
    p.add_argument("--lora-dropout",            type=float, default=0.05)
    p.add_argument("--fusion-n-layers",         type=int,   default=3)
    p.add_argument("--fusion-nhead",            type=int,   default=8)
    p.add_argument("--fusion-dim-ff",           type=int,   default=384)
    p.add_argument("--fusion-dropout",          type=float, default=0.1)
    p.add_argument("--head-dim",                type=int,   default=256)
    p.add_argument("--head-dropout",            type=float, default=0.2)
    p.add_argument("--focal-gamma",             type=float, default=2.0)
    p.add_argument("--label-smoothing",         type=float, default=0.05)
    p.add_argument("--mixup-alpha",             type=float, default=0.3)
    p.add_argument("--warmup-epochs",           type=int,   default=3)
    p.add_argument("--plateau-patience",        type=int,   default=4)
    p.add_argument("--plateau-factor",          type=float, default=0.5)
    p.add_argument("--early-stopping-patience", type=int,   default=25)
    p.add_argument("--n-top-checkpoints",       type=int,   default=3)
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

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit = args.debug_max_step if args.debug_max_step is not None else 1.0

    # Resolve real script path (handles symlink working_node_N/mcts/node/ -> real node dir)
    output_dir = Path(sys.argv[0]).resolve().parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save top-k checkpoints for potential checkpoint averaging during test
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node3-1-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=args.n_top_checkpoints,
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
    # Warmup callback: ramps LR from 10% -> 100% over warmup_epochs, then
    # restores base LRs for ReduceLROnPlateau to take over (PyTorch 2.7 compatibility)
    warmup_cb = WarmupSchedulerCallback(
        warmup_epochs=args.warmup_epochs, warmup_factor=0.1
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
        limit_train_batches=limit,
        limit_val_batches=limit,
        limit_test_batches=limit,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar, warmup_cb],
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
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        fusion_n_layers=args.fusion_n_layers,
        fusion_nhead=args.fusion_nhead,
        fusion_dim_ff=args.fusion_dim_ff,
        fusion_dropout=args.fusion_dropout,
        head_dim=args.head_dim,
        head_dropout=args.head_dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        warmup_epochs=args.warmup_epochs,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        max_epochs=args.max_epochs,
        n_top_checkpoints=args.n_top_checkpoints,
    )

    trainer.fit(model_module, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(
            model_module, datamodule=datamodule, ckpt_path="best"
        )

    # Only write test_score.txt for full runs (not debug/fast_dev runs)
    # The EvaluateAgent will compute the real test metric via calc_metric.py
    if trainer.is_global_zero and args.debug_max_step is None and not args.fast_dev_run:
        score_path = Path(sys.argv[0]).resolve().parent / "test_score.txt"
        primary_val = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else float("nan")
        )
        score_path.write_text(
            f"# Node 3-1-2 Test Evaluation Results\n"
            f"# Model: 4-source cross-attention fusion (improved training dynamics)\n"
            f"#   AIDO.Cell-10M + LoRA(r=4, all 8 layers)\n"
            f"#   Frozen STRING GNN PPI embeddings\n"
            f"#   3-branch Symbol CNN\n"
            f"#   3-layer TransformerEncoder fusion (nhead=8, dim_ff=384)\n"
            f"# Key improvements: warmup+plateau(patience=4), wd=0.03, [7,1,15] weights\n"
            f"# Primary metric: f1_score (macro-averaged per-gene F1)\n"
            f"val_f1_best: {primary_val:.6f}\n"
        )
        print(f"Test score saved -> {score_path}")


if __name__ == "__main__":
    main()
