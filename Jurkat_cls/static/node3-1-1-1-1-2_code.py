#!/usr/bin/env python3
"""
Node: AIDO.Cell-10M + LoRA (r=4, ALL 8 layers) + STRING GNN PPI + Symbol CNN
       + 4-source Fusion MLP Head (832→384→19920)
=================================================================================
Improves on parent (test F1=0.3965) by adopting the proven 4-source fusion
architecture from node3-2 (test F1=0.4622, tree-best at time of writing) and
node2-2-3-1-1-1 (test F1=0.4655, another tree-best).

The parent lineage (node3-1-1-1-1 → node3-1-1-1-1-1-1) has remained stuck in a
simple AIDO.Cell-only paradigm (with small MLP head variations), reaching only
~0.40 F1. Meanwhile, 4-source fusion nodes achieve 0.46+.

KEY CHANGE: Adopt the 4-source fusion architecture that has proven to be the
performance ceiling-breaker in this MCTS tree.

==============================================================================
Architecture
==============================================================================

1. AIDO.Cell-10M with LoRA (r=4, ALL 8 Q/K/V layers)
   - Dual pooling: [pert_hidden, global_mean] → 512-dim
   - Expanded from parent's last-3-layers to all-8 (matches proven best nodes)

2. Symbol CNN (3-branch character-level CNN on gene symbol)
   - Encodes gene family patterns (NDUF, KDM, DHX prefixes)
   - Orthogonal signal to AIDO.Cell positional embedding
   - 64-dim output

3. Frozen STRING GNN PPI (precomputed at setup)
   - 256-dim per-gene PPI topology embeddings
   - Provides biological interaction network signal
   - Key differentiator from AIDO.Cell-only: ~+0.014 F1 boost confirmed

4. Fusion MLP Head: concat([dual_pool, sym_cnn, ppi_feat]) → 832-dim
   → Linear(832→384) → GELU → Dropout(0.4)
   → Linear(384→19920) → reshape [B, 3, 6640]

Total trainable:
  - LoRA r=4 all-8-layers Q/K/V: ~36K
  - Symbol CNN: ~25K
  - STRING PPI projection (Linear 256→256): ~66K
  - Head (832→384→19920): ~3.88M
  - Total: ~4.0M

==============================================================================
Evidence from MCTS tree
==============================================================================

Best nodes using 4-source fusion:
  node3-2         (F1=0.4622): AIDO.Cell-10M LoRA r=4 all-8 + String + CNN + 832→384 head
  node2-2-3-1-1-1 (F1=0.4655): same architecture + stronger class weights [7,1,15]
  node3-1-3-1-1-1-1 (F1=0.4768): cross-attention fusion (more complex, more params)
  node3-1-2       (F1=0.4577): 4-source + 832→384 head
  node2-3-1-1     (F1=0.4555): 4-source + same architecture

Key lessons from these nodes:
  - LoRA on ALL 8 layers (not just last 3-4) is critical
  - STRING PPI is the single most impactful add-on (+0.014 F1)
  - Symbol CNN adds orthogonal gene-family signal (+0.002-0.005 F1)
  - Head width 384 is the proven sweet spot
  - Class weights [5-7, 1, 10-15] needed for minority class recall
  - val_f1 monitoring for ReduceLROnPlateau
  - Seed=0 + weight_decay=0.03 from best nodes

==============================================================================
Training Configuration
==============================================================================

| Hyperparameter         | Parent      | This Node        | Rationale           |
|------------------------|-------------|------------------|---------------------|
| LoRA layers            | 5,6,7 only  | ALL 8 layers     | Proven in top nodes |
| Input channels         | AIDO only   | AIDO+PPI+CNN     | +0.06 F1 boost      |
| Head architecture      | 512→128→19920 | 832→384→19920  | Wider proven best   |
| Head params            | 2.64M       | ~3.88M           | Better capacity     |
| Head LR                | 2e-4        | 6e-4             | Matches best nodes  |
| Backbone LR            | 3e-5        | 2e-4             | For all-8 LoRA      |
| Dropout (head)         | 0.3         | 0.4              | Proven in tree-best |
| Class weights          | [5,1,10]    | [5,1,10]         | Unchanged; proven   |
| Label smoothing        | 0.1         | 0.05             | Best nodes use 0.05 |
| Weight decay           | 0.01        | 0.03             | Best nodes use 0.03 |
| LR reduce patience     | 10          | 8                | Matches best nodes  |
| Early stop patience    | 25          | 20               | Matches best nodes  |
| Max epochs             | 150         | 150              | Unchanged           |

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
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT = 6_640
N_GENES_MODEL = 19_264
N_CLASSES = 3
AIDO_HIDDEN = 256       # AIDO.Cell-10M hidden size
STRING_HIDDEN = 256     # STRING GNN output dim
CNN_HIDDEN = 64         # Symbol CNN output dim
FUSION_DIM = AIDO_HIDDEN * 2 + STRING_HIDDEN + CNN_HIDDEN  # 512 + 256 + 64 = 832

# Class weights: corrects imbalance (~95% class 1/unchanged) — proven effective
# at [5,1,10] across 30+ nodes in tree
CLASS_WEIGHTS = torch.tensor([5.0, 1.0, 10.0], dtype=torch.float32)

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
# Symbol CNN (3-branch character-level CNN for gene symbols)
# ─────────────────────────────────────────────────────────────────────────────
class SymbolCNN(nn.Module):
    """
    3-branch character-level 1D CNN for gene symbol encoding.
    Captures gene family naming conventions (prefixes, stems) at multiple scales.
    Produces 64-dim output.

    Proven effective at +0.002-0.005 F1 gain across node2-2, node3-2 lineages.
    """

    # Character vocabulary: a-z, 0-9, plus special characters
    VOCAB = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789-.")}
    PAD_IDX = 0
    VOCAB_SIZE = len(VOCAB) + 1  # +1 for PAD
    MAX_LEN = 16

    def __init__(self, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB_SIZE, 32, padding_idx=self.PAD_IDX)
        # 3 branches: kernel sizes 2, 3, 4
        self.conv2 = nn.Conv1d(32, out_dim, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(32, out_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(32, out_dim, kernel_size=4, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(out_dim * 3, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def encode_symbol(self, symbol: str) -> torch.Tensor:
        """Convert gene symbol string to int tensor of length MAX_LEN."""
        sym_lower = symbol.lower()[:self.MAX_LEN]
        ids = [self.VOCAB.get(c, self.PAD_IDX) for c in sym_lower]
        # Pad to MAX_LEN
        ids = ids + [self.PAD_IDX] * (self.MAX_LEN - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, symbol_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            symbol_ids: [B, MAX_LEN] long
        Returns:
            [B, out_dim] float
        """
        x = self.embed(symbol_ids)          # [B, MAX_LEN, 32]
        x = x.transpose(1, 2)              # [B, 32, MAX_LEN]
        # 3-branch convolutions
        c2 = F.gelu(self.conv2(x))          # [B, out_dim, L]
        c3 = F.gelu(self.conv3(x))          # [B, out_dim, L]
        c4 = F.gelu(self.conv4(x))          # [B, out_dim, L]
        # Global max-pool
        p2 = self.pool(c2).squeeze(-1)      # [B, out_dim]
        p3 = self.pool(c3).squeeze(-1)      # [B, out_dim]
        p4 = self.pool(c4).squeeze(-1)      # [B, out_dim]
        # Concat + project
        cat = torch.cat([p2, p3, p4], dim=1)  # [B, out_dim*3]
        out = self.proj(self.dropout(cat))   # [B, out_dim]
        return self.norm(out)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Pre-builds synthetic expression vectors and symbol encodings for each sample.

    Input encoding:
    - AIDO.Cell: all 19264 genes at 1.0 (baseline), perturbed gene at 0.0
    - Symbol CNN: character-level encoding of gene symbol (gene family proxy)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gene_pos_map: Dict[str, int],
        ppi_node_map: Dict[str, int],  # ENSG → STRING GNN node index
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Map {-1,0,1} → {0,1,2} to match calc_metric.py's y_true + 1 convention
            self.labels = np.array(raw_labels, dtype=np.int8) + 1
        else:
            self.labels = None

        # Pre-compute expression vectors for efficiency
        # base: all 19264 genes at 1.0; knock out perturbed gene to 0.0
        base_expr = torch.ones(N_GENES_MODEL, dtype=torch.float32)

        self._exprs: List[torch.Tensor] = []
        self._pert_positions: List[int] = []
        self._ppi_indices: List[int] = []     # STRING GNN node index for each sample
        self._symbol_ids: List[torch.Tensor] = []

        covered_aido = 0
        covered_ppi = 0
        for pid, sym in zip(self.pert_ids, self.symbols):
            base_pid = pid.split(".")[0]

            # AIDO.Cell position
            pos = gene_pos_map.get(base_pid, -1)
            self._pert_positions.append(pos)
            if pos >= 0:
                expr = base_expr.clone()
                expr[pos] = 0.0
                covered_aido += 1
            else:
                expr = base_expr.clone()
            self._exprs.append(expr)

            # STRING GNN node index
            ppi_idx = ppi_node_map.get(base_pid, -1)
            self._ppi_indices.append(ppi_idx)
            if ppi_idx >= 0:
                covered_ppi += 1

            # Symbol character encoding
            sym_ids = SymbolCNN.encode_symbol(SymbolCNN, sym)
            self._symbol_ids.append(sym_ids)

        if not is_test:
            print(f"[Dataset] {len(self.pert_ids)} samples, "
                  f"AIDO coverage: {covered_aido}/{len(self.pert_ids)} ({100.0 * covered_aido / len(self.pert_ids):.1f}%), "
                  f"STRING PPI coverage: {covered_ppi}/{len(self.pert_ids)} ({100.0 * covered_ppi / len(self.pert_ids):.1f}%)")

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "expr": self._exprs[idx],                         # [19264] float32
            "pert_pos": self._pert_positions[idx],            # int
            "ppi_idx": self._ppi_indices[idx],                # int (STRING GNN node)
            "symbol_ids": self._symbol_ids[idx],              # [MAX_LEN] long
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
        "expr": torch.stack([b["expr"] for b in batch]),                  # [B, 19264]
        "pert_pos": torch.tensor([b["pert_pos"] for b in batch], dtype=torch.long),
        "ppi_idx": torch.tensor([b["ppi_idx"] for b in batch], dtype=torch.long),
        "symbol_ids": torch.stack([b["symbol_ids"] for b in batch]),      # [B, MAX_LEN]
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])        # [B, 6640]
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
        self._ppi_node_map: Optional[Dict[str, int]] = None
        # Precomputed STRING GNN embeddings [18870, 256], frozen
        self.ppi_emb: Optional[torch.Tensor] = None

    def _build_gene_pos_map(
        self, tokenizer, all_pert_ids: List[str]
    ) -> Dict[str, int]:
        """Build mapping from ENSG gene ID to AIDO.Cell vocab position."""
        gene_pos_map: Dict[str, int] = {}
        unique_base_ids = list(set(pid.split(".")[0] for pid in all_pert_ids))
        print(f"[DataModule] Building gene position map for {len(unique_base_ids)} unique genes...")

        if hasattr(tokenizer, "gene_id_to_index"):
            gid2idx = tokenizer.gene_id_to_index
            for base_pid in unique_base_ids:
                if base_pid in gid2idx:
                    gene_pos_map[base_pid] = gid2idx[base_pid]
            if len(gene_pos_map) > 0:
                print(f"[DataModule] ENSG→pos via gene_id_to_index: "
                      f"{len(gene_pos_map)}/{len(unique_base_ids)} found")
                return gene_pos_map

        print(f"[DataModule] No gene position mapping available; all pert_pos will be -1")
        return gene_pos_map

    def _build_ppi_node_map(self, all_pert_ids: List[str]) -> Tuple[Dict[str, int], torch.Tensor]:
        """
        Build mapping from ENSG gene ID to STRING GNN node index.
        Also precompute frozen STRING GNN embeddings.
        """
        import json as json_lib
        node_names_path = Path(STRING_GNN_DIR) / "node_names.json"
        node_names = json_lib.loads(node_names_path.read_text())
        # node_names[i] = ENSG gene ID string

        ppi_node_map: Dict[str, int] = {}
        for i, ensg_id in enumerate(node_names):
            ppi_node_map[ensg_id] = i

        unique_base_ids = list(set(pid.split(".")[0] for pid in all_pert_ids))
        covered = sum(1 for pid in unique_base_ids if pid in ppi_node_map)
        print(f"[DataModule] STRING PPI node map: {covered}/{len(unique_base_ids)} genes covered "
              f"({100.0 * covered / len(unique_base_ids):.1f}%)")

        # Precompute STRING GNN embeddings (frozen) on CPU first, move to GPU in model
        print("[DataModule] Precomputing frozen STRING GNN embeddings...")
        device = torch.device("cpu")
        gnn_model = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        gnn_model.eval()
        gnn_model.to(device)

        graph_data = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", weights_only=False)
        edge_index = graph_data["edge_index"].to(device)
        edge_weight = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        with torch.no_grad():
            outputs = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
        ppi_emb = outputs.last_hidden_state.cpu()  # [18870, 256]
        print(f"[DataModule] STRING GNN embeddings precomputed: {ppi_emb.shape}")

        # Free GNN model memory
        del gnn_model
        import gc
        gc.collect()

        return ppi_node_map, ppi_emb

    def setup(self, stage: Optional[str] = None) -> None:
        # Initialize AIDO tokenizer: rank-0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        # Read all pert_ids for building maps
        all_ids: List[str] = []
        for fname in ["train.tsv", "val.tsv", "test.tsv"]:
            fpath = self.data_dir / fname
            if fpath.exists():
                df_tmp = pd.read_csv(fpath, sep="\t")
                if "pert_id" in df_tmp.columns:
                    all_ids.extend(df_tmp["pert_id"].tolist())

        # Build gene position map for AIDO.Cell
        if self._gene_pos_map is None:
            self._gene_pos_map = self._build_gene_pos_map(tokenizer, all_ids)

        # Build STRING GNN map and precompute embeddings
        if self._ppi_node_map is None:
            self._ppi_node_map, self.ppi_emb = self._build_ppi_node_map(all_ids)

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(
                train_df, self._gene_pos_map, self._ppi_node_map
            )
            self.val_ds = PerturbationDataset(
                val_df, self._gene_pos_map, self._ppi_node_map
            )

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(
                test_df, self._gene_pos_map, self._ppi_node_map, is_test=True
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
# Model
# ─────────────────────────────────────────────────────────────────────────────
class FourSourceDEGModel(nn.Module):
    """
    4-source feature fusion DEG predictor:
    1. AIDO.Cell-10M + LoRA (r=4, all 8 layers) → dual-pool → 512-dim
    2. Symbol CNN (3-branch char-level) → 64-dim
    3. Frozen STRING GNN PPI (precomputed) → 256-dim
    4. Fusion MLP head: 832→384→19920

    Architecture matches tree-best node3-2 (F1=0.4622) which represents the
    definitive ceiling for the 4-source fusion paradigm.

    Trainable params: ~4.0M
    - LoRA r=4 all-8-layers Q/K/V: ~36K
    - Symbol CNN: ~25K
    - STRING PPI projection (linear 256→256): ~66K
    - MLP head (832→384→19920): ~3.88M
    """

    def __init__(
        self,
        ppi_emb: torch.Tensor,             # [18870, 256] precomputed PPI embeddings
        head_hidden: int = 384,
        head_dropout: float = 0.4,
        sym_cnn_dim: int = 64,
        sym_cnn_dropout: float = 0.1,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        # ── 1. AIDO.Cell-10M backbone with LoRA ────────────────────────────
        backbone = AutoModel.from_pretrained(
            AIDO_MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        backbone.config.use_cache = False

        # Monkey-patch AIDO.Cell's NotImplementedError for enable_input_require_grads
        def noop_enable_input_require_grads(self):
            pass
        backbone.enable_input_require_grads = noop_enable_input_require_grads.__get__(
            backbone, type(backbone)
        )
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # LoRA on ALL 8 layers (Q/K/V) — matches node3-2, node2-3-1-1 proven configs
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,  # None = all layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Cast LoRA (trainable) params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── 2. Symbol CNN ────────────────────────────────────────────────────
        self.symbol_cnn = SymbolCNN(out_dim=sym_cnn_dim, dropout=sym_cnn_dropout)

        # ── 3. Frozen STRING GNN PPI embeddings ──────────────────────────────
        # Register as buffer: automatically moved to device, not trained
        self.register_buffer("ppi_emb", ppi_emb.float())  # [18870, 256]
        # 1-layer projection for PPI features (trainable)
        self.ppi_proj = nn.Linear(STRING_HIDDEN, STRING_HIDDEN, bias=True)
        nn.init.eye_(self.ppi_proj.weight)
        nn.init.zeros_(self.ppi_proj.bias)

        # ── 4. Fusion MLP head ───────────────────────────────────────────────
        # Input: concat([pert_hidden, global_mean, sym_cnn, ppi_feat])
        # = 512 + 64 + 256 = 832
        in_dim = FUSION_DIM  # 832
        out_dim = N_CLASSES * N_GENES_OUT  # 3 × 6640 = 19920

        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(head_dropout),
            nn.Linear(in_dim, head_hidden, bias=True),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, out_dim, bias=True),
        )

        # Initialize head layers
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        expr: torch.Tensor,          # [B, 19264] float32
        pert_pos: torch.Tensor,       # [B] long
        ppi_idx: torch.Tensor,        # [B] long (STRING GNN node indices)
        symbol_ids: torch.Tensor,     # [B, MAX_LEN] long
    ) -> torch.Tensor:               # [B, 3, 6640]
        B = expr.shape[0]
        device = expr.device

        # ── 1. AIDO.Cell forward pass ──────────────────────────────────────
        outputs = self.backbone(
            input_ids=expr,
            attention_mask=torch.ones(B, N_GENES_MODEL, dtype=torch.long, device=device),
        )
        # Cast to float32 for stable head computation
        hidden = outputs.last_hidden_state.float()  # [B, 19266, 256]

        # ── 2. Dual pooling ────────────────────────────────────────────────
        # Global mean pool over gene positions (exclude 2 summary tokens)
        global_pool = hidden[:, :N_GENES_MODEL, :].mean(dim=1)  # [B, 256]

        # Per-sample perturbed-gene positional embedding
        safe_pos = pert_pos.clamp(min=0)
        pos_idx = safe_pos.view(B, 1, 1).expand(B, 1, AIDO_HIDDEN)
        pert_hidden = hidden.gather(1, pos_idx).squeeze(1)  # [B, 256]

        # Fallback for genes not in AIDO vocabulary
        unknown_mask = (pert_pos < 0)
        if unknown_mask.any():
            pert_hidden = pert_hidden.clone()
            pert_hidden[unknown_mask] = global_pool[unknown_mask]

        dual_pool = torch.cat([pert_hidden, global_pool], dim=1)  # [B, 512]

        # ── 3. Symbol CNN features ─────────────────────────────────────────
        sym_feat = self.symbol_cnn(symbol_ids)  # [B, 64]

        # ── 4. STRING GNN PPI features ─────────────────────────────────────
        # Lookup precomputed PPI embeddings; use zero for genes not in STRING
        safe_ppi = ppi_idx.clamp(min=0)
        ppi_raw = self.ppi_emb[safe_ppi]       # [B, 256]
        # Zero out for genes not in STRING vocabulary
        unknown_ppi = (ppi_idx < 0)
        if unknown_ppi.any():
            ppi_raw = ppi_raw.clone()
            ppi_raw[unknown_ppi] = 0.0
        ppi_feat = self.ppi_proj(ppi_raw)       # [B, 256] (trainable projection)

        # ── 5. Fusion MLP head ─────────────────────────────────────────────
        fusion = torch.cat([dual_pool, sym_feat, ppi_feat], dim=1)  # [B, 832]
        logits = self.head(fusion)                                     # [B, 19920]

        return logits.view(B, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """
    Per-gene macro-averaged F1 score, matching calc_metric.py logic exactly.

    y_pred: [n_samples, 3, n_genes] — class probabilities
    y_true_remapped: [n_samples, n_genes] — labels in {0,1,2} (i.e., y_true+1)
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
        head_hidden: int = 384,
        head_dropout: float = 0.4,
        sym_cnn_dim: int = 64,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        backbone_lr: float = 2e-4,
        head_lr: float = 6e-4,
        sym_lr: float = 6e-4,
        ppi_lr: float = 6e-4,
        weight_decay: float = 0.03,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 150,
        lr_reduce_patience: int = 8,
        lr_reduce_factor: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[FourSourceDEGModel] = None
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
            # Get precomputed PPI embeddings from datamodule
            dm = getattr(self.trainer, "datamodule", None)
            if dm is None or dm.ppi_emb is None:
                raise RuntimeError("DataModule must be set up before model setup. "
                                   "Ensure DEGDataModule.setup() is called first.")
            ppi_emb = dm.ppi_emb

            self.model = FourSourceDEGModel(
                ppi_emb=ppi_emb,
                head_hidden=self.hparams.head_hidden,
                head_dropout=self.hparams.head_dropout,
                sym_cnn_dim=self.hparams.sym_cnn_dim,
                lora_r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
            )
            self.loss_fn = FocalLoss(
                gamma=self.hparams.focal_gamma,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        # Populate test metadata for prediction saving
        if stage in ("test", None):
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
        )  # [B, 3, 6640]

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, 3, 6640], labels: [B, 6640]
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                               # [B*6640]
        return self.loss_fn(logits_flat, labels_flat)

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

            output_dir = Path(__file__).parent / "run"
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
            self.print(f"Test predictions saved → {pred_path}")

    def configure_optimizers(self):
        # Separate learning rates per parameter group
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters() if p.requires_grad
        ]
        sym_params = list(self.model.symbol_cnn.parameters())
        ppi_params = list(self.model.ppi_proj.parameters())
        head_params = list(self.model.head.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
                {"params": sym_params,     "lr": self.hparams.sym_lr},
                {"params": ppi_params,     "lr": self.hparams.ppi_lr},
                {"params": head_params,    "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
            eps=1e-8,
        )

        # ReduceLROnPlateau monitoring val_f1 (mode=max)
        # patience=8 matches node3-2's proven configuration that reached F1=0.4622
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.lr_reduce_factor,
            patience=self.hparams.lr_reduce_patience,
            min_lr=1e-8,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

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
        description="Node: AIDO.Cell-10M + LoRA (r=4, all 8 layers) + STRING PPI + Symbol CNN + 832→384 head"
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "data"),
    )
    p.add_argument("--micro_batch_size",        type=int,   default=4)
    p.add_argument("--global_batch_size",       type=int,   default=32)
    p.add_argument("--max_epochs",              type=int,   default=150)
    p.add_argument("--backbone_lr",             type=float, default=2e-4)
    p.add_argument("--head_lr",                 type=float, default=6e-4)
    p.add_argument("--sym_lr",                  type=float, default=6e-4)
    p.add_argument("--ppi_lr",                  type=float, default=6e-4)
    p.add_argument("--weight_decay",            type=float, default=0.03)
    p.add_argument("--head_dropout",            type=float, default=0.4)
    p.add_argument("--head_hidden",             type=int,   default=384)
    p.add_argument("--sym_cnn_dim",             type=int,   default=64)
    p.add_argument("--lora_r",                  type=int,   default=4)
    p.add_argument("--lora_alpha",              type=int,   default=8)
    p.add_argument("--lora_dropout",            type=float, default=0.1)
    p.add_argument("--focal_gamma",             type=float, default=2.0)
    p.add_argument("--label_smoothing",         type=float, default=0.05)
    p.add_argument("--lr_reduce_patience",      type=int,   default=8)
    p.add_argument("--lr_reduce_factor",        type=float, default=0.5)
    p.add_argument("--early_stopping_patience", type=int,   default=20)
    p.add_argument("--num_workers",             type=int,   default=0)
    p.add_argument("--val_check_interval",      type=float, default=1.0)
    p.add_argument("--debug_max_step",          type=int,   default=None)
    p.add_argument("--fast_dev_run",            action="store_true")
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
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit = args.debug_max_step if args.debug_max_step is not None else 1.0

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node-4src-fusion-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
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
        limit_val_batches=limit,
        limit_test_batches=limit,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=False,  # Disabled: AdaptiveMaxPool1d in SymbolCNN is non-deterministic on CUDA
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
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        sym_cnn_dim=args.sym_cnn_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        sym_lr=args.sym_lr,
        ppi_lr=args.ppi_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        lr_reduce_patience=args.lr_reduce_patience,
        lr_reduce_factor=args.lr_reduce_factor,
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
        primary_val = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else float("nan")
        )
        score_path.write_text(
            f"# Node: AIDO.Cell-10M + LoRA (r=4, all 8 layers) + STRING PPI + Symbol CNN\n"
            f"# Architecture: 4-source fusion (AIDO dual-pool + PPI + CNN → 832→384→19920)\n"
            f"# Primary metric: f1_score (macro-averaged per-gene F1)\n"
            f"# Key improvement: 4-source fusion matching proven tree-best node3-2 architecture\n"
            f"# Expected: ~0.46+ F1 (tree-best at 0.4768 with cross-attention fusion)\n"
            f"\n"
            f"Best val_f1 (from checkpoint): {primary_val:.6f}\n"
            f"\n"
            f"Test results:\n"
            f"{json.dumps(test_results, indent=2)}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
