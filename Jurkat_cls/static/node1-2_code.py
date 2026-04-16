#!/usr/bin/env python3
"""
Node 1-2: Perturbation-Conditioned STRING_GNN + AIDO.Cell-10M Dual Encoder
===========================================================================
Key design decisions (distinct from sibling node1-1 and parent node1):

  1. STRING_GNN is run DYNAMICALLY per batch with perturbation-specific cond_emb.
     For each sample, the perturbed gene's AIDO embedding is injected into the
     STRING_GNN as a conditioning signal at the gene's PPI graph node. The GCN
     then propagates this perturbation signal through 8 message-passing layers,
     making the output embedding perturbation-specific. This is biologically
     meaningful: it simulates how a gene knockout propagates through the PPI network.

  2. AIDO.Cell-10M (10M params, 256-dim hidden) provides frozen transcriptomic context.
     Dual pooling: gene_pos_emb (256) + mean_pool (256) = 512-dim.

  3. A trainable projection projects AIDO gene_pos_emb → STRING cond_emb space.
     Only this projection (256→256) + null_emb (256) + head are trainable.

  4. Wider output head: LayerNorm(768) → Linear(768,128) → GELU → Dropout(0.4)
     → Linear(128, 3×6640). Total head params ≈ 2.65M.

  5. BOTH backbones (STRING_GNN + AIDO.Cell-10M) FULLY FROZEN. Only the:
     - cond_emb projection: 256×256 + 256 = 65,792 params
     - null_emb: 256 params
     - head layers: ~2.65M params
     → Total trainable ≈ 2.72M params (~1,813/sample)

  6. STRING_GNN is batched efficiently: each mini-batch of B samples creates a
     batched graph with B copies of the STRING graph, each with different cond_emb.
     PyG handles the batched sparse ops automatically.

  7. Strong regularization: Dropout(0.4), label_smoothing=0.12, focal_gamma=2.5,
     moderate class weights [3.0, 1.0, 7.0].

  8. ModelCheckpoint and EarlyStopping monitor val_f1 (mode="max").
     (Critical fix from parent node1's feedback.)

  9. CosineAnnealingWarmRestarts(T_0=40, T_mult=2) for multi-cycle LR schedule
     that allows escape from local minima.
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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
N_GENES = 6_640
N_CLASSES = 3

AIDO_MODEL_PATH = "/home/Models/AIDO.Cell-10M"  # 10M, 256-dim hidden
STRING_GNN_PATH = "/home/Models/STRING_GNN"

AIDO_DIM = 256          # AIDO.Cell-10M hidden dim
STRING_DIM = 256        # STRING_GNN output dim
DUAL_POOL_DIM = AIDO_DIM * 2   # 512 (gene_pos + mean_pool)
FUSION_DIM = STRING_DIM + DUAL_POOL_DIM  # 768

# Moderate class weights: balanced between parent's [2,1,5] and node2's [5,1,10]
# Train distribution: ~3.41% down-regulated (class 0), ~95.48% unchanged (class 1), ~1.10% up-regulated (class 2)
CLASS_WEIGHTS = torch.tensor([3.0, 1.0, 7.0], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss with optional class weighting and label smoothing."""

    def __init__(self, gamma: float = 2.5, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [N, C]  float32
        targets: [N]     int64
        """
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none").detach())
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper (mirrors calc_metric.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# STRING_GNN Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_string_gnn(model_path: str = STRING_GNN_PATH):
    """
    Load STRING_GNN model and graph data.

    Returns:
        gnn_model:       StringGNNModel (eval mode, FROZEN)
        edge_index:      [2, E] int64 CPU tensor
        edge_weight:     [E] float32 CPU tensor (or None)
        node_name_to_idx: dict mapping Ensembl ID -> node index in [0, 18869]
    """
    import json as _json
    model_dir = Path(model_path)

    node_names = _json.loads((model_dir / "node_names.json").read_text())
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

    gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    gnn_model.eval()
    for param in gnn_model.parameters():
        param.requires_grad = False

    graph = torch.load(str(model_dir / "graph_data.pt"), map_location="cpu")
    edge_index = graph["edge_index"]   # [2, E] int64
    edge_weight = graph.get("edge_weight", None)   # [E] float32 or None

    n_nodes = len(node_names)
    print(f"STRING_GNN loaded: {n_nodes} nodes, "
          f"{edge_index.shape[1]} directed edges, "
          f"{sum(p.numel() for p in gnn_model.parameters()):,} params (all frozen)")

    return gnn_model, edge_index, edge_weight, node_name_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# Limited DataLoader wrapper for debug modes
# ─────────────────────────────────────────────────────────────────────────────
class _HeadDataLoader(DataLoader):
    """
    A DataLoader that stops after `max_batches` iterations.
    Used to implement --fast-dev-run (1 batch) and --debug-max-step N (N batches)
    without relying on Lightning's `limit_train_batches` float semantics.
    """

    def __init__(self, dataset, max_batches: int, **kwargs):
        super().__init__(dataset, **kwargs)
        self.max_batches = max_batches

    def __iter__(self):
        iterator = super().__iter__()
        for i, batch in enumerate(iterator):
            if i >= self.max_batches:
                break
            yield batch

    def __len__(self):
        return min(len(self.dataset) // self.batch_size, self.max_batches)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    """
    Dataset that pre-tokenizes AIDO.Cell inputs and stores STRING node indices.
    STRING_GNN is run dynamically in the model forward pass (not pre-computed)
    to allow perturbation-specific graph conditioning.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        input_ids: torch.Tensor,        # [N, 19264] float32 (AIDO.Cell input)
        pert_positions: torch.Tensor,   # [N] int64 (-1 if gene not in AIDO vocab)
        string_node_indices: torch.Tensor,  # [N] int64 (-1 if gene not in STRING vocab)
        is_test: bool = False,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.input_ids = input_ids
        self.pert_positions = pert_positions
        self.string_node_indices = string_node_indices
        self.is_test = is_test

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            arr = np.array(raw_labels, dtype=np.int8) + 1  # {-1,0,1} → {0,1,2}
            self.labels = torch.from_numpy(arr).long()      # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "input_ids": self.input_ids[idx],           # [19264] float32
            "pert_pos": self.pert_positions[idx],        # int64 (-1 if unknown)
            "string_node_idx": self.string_node_indices[idx],  # int64 (-1 if unknown)
        }
        if not self.is_test:
            item["label"] = self.labels[idx]  # [6640] int64
        return item


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

        # STRING vocabulary (set in setup)
        self._string_node_to_idx: Optional[Dict[str, int]] = None

    def _init_tokenizer(self) -> AutoTokenizer:
        """Rank-safe tokenizer initialization (rank 0 downloads first if distributed)."""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return AutoTokenizer.from_pretrained(AIDO_MODEL_PATH, trust_remote_code=True)

    def _init_string_vocab(self) -> None:
        """Load STRING node vocabulary (fast, JSON only, no model weights)."""
        if self._string_node_to_idx is not None:
            return
        import json as _json
        node_names = _json.loads((Path(STRING_GNN_PATH) / "node_names.json").read_text())
        self._string_node_to_idx = {name: i for i, name in enumerate(node_names)}

    def _tokenize_and_get_positions(
        self,
        tokenizer: AutoTokenizer,
        pert_ids: List[str],
        split_name: str = "split",
    ) -> tuple:
        """Tokenize pert_ids for AIDO.Cell and get positional indices."""
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        chunk_size = 128
        all_input_ids: List[torch.Tensor] = []
        for i in range(0, len(expr_dicts), chunk_size):
            chunk = expr_dicts[i:i + chunk_size]
            toks = tokenizer(chunk, return_tensors="pt")
            all_input_ids.append(toks["input_ids"])
        input_ids = torch.cat(all_input_ids, dim=0)  # [N, 19264] float32
        non_missing = input_ids > -0.5
        has_gene = non_missing.any(dim=1)
        pert_positions = non_missing.long().argmax(dim=1)
        pert_positions[~has_gene] = -1
        coverage = 100.0 * has_gene.float().mean().item()
        print(f"  [{split_name}] AIDO vocab coverage: "
              f"{has_gene.sum().item()}/{len(pert_ids)} ({coverage:.1f}%)")
        return input_ids, pert_positions

    def _get_string_node_indices(
        self,
        pert_ids: List[str],
        split_name: str = "split",
    ) -> torch.Tensor:
        """Look up STRING node indices for pert_ids. Returns -1 for unknown genes."""
        indices = []
        n_found = 0
        for pid in pert_ids:
            idx = self._string_node_to_idx.get(pid, -1)
            indices.append(idx)
            if idx >= 0:
                n_found += 1
        coverage = 100.0 * n_found / len(pert_ids)
        print(f"  [{split_name}] STRING vocab coverage: "
              f"{n_found}/{len(pert_ids)} ({coverage:.1f}%)")
        return torch.tensor(indices, dtype=torch.long)

    def setup(self, stage: Optional[str] = None) -> None:
        self._init_string_vocab()
        tokenizer = self._init_tokenizer()

        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")

            print("Preparing train set...")
            tr_ids, tr_pos = self._tokenize_and_get_positions(
                tokenizer, train_df["pert_id"].tolist(), "train")
            tr_str_idx = self._get_string_node_indices(
                train_df["pert_id"].tolist(), "train")

            print("Preparing val set...")
            va_ids, va_pos = self._tokenize_and_get_positions(
                tokenizer, val_df["pert_id"].tolist(), "val")
            va_str_idx = self._get_string_node_indices(
                val_df["pert_id"].tolist(), "val")

            self.train_ds = PerturbationDataset(
                train_df, tr_ids, tr_pos, tr_str_idx, is_test=False)
            self.val_ds = PerturbationDataset(
                val_df, va_ids, va_pos, va_str_idx, is_test=False)

        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            print("Preparing test set...")
            te_ids, te_pos = self._tokenize_and_get_positions(
                tokenizer, test_df["pert_id"].tolist(), "test")
            te_str_idx = self._get_string_node_indices(
                test_df["pert_id"].tolist(), "test")

            self.test_ds = PerturbationDataset(
                test_df, te_ids, te_pos, te_str_idx, is_test=True)
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
        )

    # ── Limited train dataloader (used by debug wrappers) ──────────────────────
    def _limited_train_dataloader(self, num_batches: int) -> DataLoader:
        """Return a train dataloader that yields at most `num_batches` batches.
        Used for --fast-dev-run (1 batch) and --debug-max-step N (N batches).
        """
        if num_batches <= 0:
            return DataLoader(
                self.train_ds,
                batch_size=self.micro_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        return _HeadDataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            max_batches=num_batches,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Override to ensure no DistributedSampler is applied in DDP mode.
        # Each rank must process ALL test samples in the same order so that
        # all_gather + positional deduplication works correctly.
        if self.test_ds is None:
            return DataLoader([], batch_size=self.micro_batch_size)
        from torch.utils.data import SequentialSampler
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            sampler=SequentialSampler(self.test_ds),  # no DDP sharding
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model: Perturbation-Conditioned STRING_GNN + AIDO.Cell-10M Dual Encoder
# ─────────────────────────────────────────────────────────────────────────────
class PerturbationConditionedDualEncoder(nn.Module):
    """
    Architecture:
      ┌─ AIDO.Cell-10M FROZEN (256-dim) ─────────────────────────────────────┐
      │  dual pooling: [gene_pos_emb (256)] + [mean_pool (256)] = 512-dim    │
      └──────────────────────────────────────────────────────────────────────┘
                  ↓ gene_pos_emb projected to STRING space
      ┌─ STRING_GNN FROZEN (256-dim) ──────────────────────────────────────────┐
      │  Dynamic forward with per-sample cond_emb:                            │
      │    cond_emb = zeros([18870, 256])                                      │
      │    cond_emb[pert_gene_node] = CondProj(gene_pos_emb)  [trainable]      │
      │  GCN propagates perturbation signal 8 hops through PPI network        │
      │  Extract: conditioned embedding for perturbed gene → [B, 256]          │
      └──────────────────────────────────────────────────────────────────────┘
                  ↓ cat
      768-dim fusion: [conditioned_str_emb (256)] + [aido_dual_pool (512)]
                  ↓
      LayerNorm(768)
      → Linear(768, head_width) → GELU → Dropout(head_dropout)
      → LayerNorm(head_width)
      → Linear(head_width, 3×6640)
      → [B, 3, 6640] logits

    Trainable parameters (head_width=128):
      cond_proj:    256 × 256 + 256 = 65,792
      null_emb:     256
      LayerNorm(768): 1,536
      Linear(768→128): 98,560
      LayerNorm(128):  256
      Linear(128→19920): 2,549,760
      biases:         ~19,984
      Total ≈ 2.74M params (~1,827/training sample)
    """

    def __init__(self, head_width: int = 128, head_dropout: float = 0.4):
        super().__init__()
        self.head_width = head_width
        self.head_dropout = head_dropout

        # AIDO.Cell-10M backbone (initialized in initialize_aido())
        self.aido_backbone: Optional[nn.Module] = None

        # STRING_GNN backbone (initialized in initialize_string_gnn())
        self.string_gnn: Optional[nn.Module] = None
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_weight: Optional[torch.Tensor] = None
        self._n_nodes: int = 18870

        # Trainable: project AIDO gene_pos_emb → STRING cond_emb space
        # This is the only trainable part of the backbone interaction
        self.cond_proj = nn.Linear(AIDO_DIM, STRING_DIM)

        # Learnable fallback for genes not in STRING vocabulary
        self.string_null_emb = nn.Parameter(torch.zeros(STRING_DIM))

        # Wider output head (vs parent's rank=64)
        self.head: Optional[nn.Sequential] = None

    def initialize_aido(self) -> None:
        """Load AIDO.Cell-10M (frozen) on GPU."""
        self.aido_backbone = AutoModel.from_pretrained(
            AIDO_MODEL_PATH, trust_remote_code=True)
        self.aido_backbone = self.aido_backbone.to(torch.bfloat16)
        self.aido_backbone.config.use_cache = False
        for param in self.aido_backbone.parameters():
            param.requires_grad = False
        total = sum(p.numel() for p in self.aido_backbone.parameters())
        print(f"AIDO.Cell-10M: {total:,} params (all frozen)")

    def initialize_string_gnn(self) -> None:
        """Load STRING_GNN (frozen) and graph data (kept on CPU, moved to GPU in forward)."""
        self.string_gnn, edge_index, edge_weight, _ = load_string_gnn()
        # Register as buffers so they move to the correct device automatically
        self.register_buffer("_edge_index_buf", edge_index)   # [2, E]
        if edge_weight is not None:
            self.register_buffer("_edge_weight_buf", edge_weight)  # [E] float32, registered buffer
        self._n_nodes = 18870
        print(f"STRING_GNN: ready for dynamic perturbation conditioning")

    def initialize_head(self) -> None:
        """Create the output head (trainable)."""
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),                            # 768
            nn.Linear(FUSION_DIM, self.head_width),              # 768 → 128
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.LayerNorm(self.head_width),                       # 128
            nn.Linear(self.head_width, N_CLASSES * N_GENES),    # 128 → 3*6640
        )
        # Truncated-normal init
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize cond_proj with smaller std for stable start
        nn.init.trunc_normal_(self.cond_proj.weight, std=0.01)
        if self.cond_proj.bias is not None:
            nn.init.zeros_(self.cond_proj.bias)

    def _get_aido_dual_pool(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32
        pert_positions: torch.Tensor,  # [B] int64 (-1 for unknown)
    ) -> tuple:
        """
        Run AIDO.Cell-10M (frozen) and return:
          - gene_pos_emb: [B, 256] float32
          - mean_pool:    [B, 256] float32
          - dual_pool:    [B, 512] float32 = cat([gene_pos_emb, mean_pool])
        """
        backbone_device = next(self.aido_backbone.parameters()).device
        input_ids_dev = input_ids.to(backbone_device)
        attn_mask = torch.ones(
            input_ids_dev.shape[0], input_ids_dev.shape[1],
            dtype=torch.long, device=backbone_device
        )
        with torch.no_grad():
            out = self.aido_backbone(
                input_ids=input_ids_dev,
                attention_mask=attn_mask,
            )
        hidden = out.last_hidden_state  # [B, 19266, 256] bfloat16

        # Global mean-pool over gene positions (exclude 2 summary tokens)
        mean_pool = hidden[:, :19264, :].mean(dim=1).float()  # [B, 256]

        # Per-gene positional extraction
        B = hidden.size(0)
        hidden_device = hidden.device
        pert_positions_dev = pert_positions.to(hidden_device)
        valid = pert_positions_dev >= 0
        safe_pos = pert_positions_dev.clamp(min=0)
        gene_emb_raw = hidden[
            torch.arange(B, device=hidden_device), safe_pos, :
        ].float()  # [B, 256]

        # Fallback to mean_pool for unknown genes
        valid_f = valid.float().unsqueeze(-1)
        gene_emb = gene_emb_raw * valid_f + mean_pool * (1.0 - valid_f)

        dual_pool = torch.cat([gene_emb, mean_pool], dim=-1)  # [B, 512]
        return gene_emb, mean_pool, dual_pool

    def _run_conditioned_string_gnn(
        self,
        gene_pos_emb: torch.Tensor,         # [B, 256] float32 (AIDO gene embeddings)
        string_node_indices: torch.Tensor,  # [B] int64 (-1 if not in STRING vocab)
    ) -> torch.Tensor:
        """
        Run STRING_GNN with perturbation-specific conditioning.

        STRING_GNN is transductive with fixed N=18870 nodes.
        We run ONE STRING_GNN forward per BATCH (not per sample) using the
        MEAN conditioning signal across all samples in the batch:
          - cond_emb: [N, 256] where each node that appears as a perturbed gene
            gets the average of the cond_proj(gene_pos_emb) signals from all
            samples targeting that node.

        After this shared GNN forward:
          - For each sample b: extract conditioned embedding at string_node_indices[b]

        This is computationally efficient (1 GNN forward per batch) while still
        allowing the GNN output to vary per batch based on which genes are perturbed.

        The cond_proj (trainable linear 256→256) learns to project AIDO embeddings
        into the STRING conditioning space, enabling cross-modal feature interaction.

        Returns: [B, 256] conditioned STRING embeddings
        """
        B = gene_pos_emb.size(0)
        N = self._n_nodes  # 18870

        # Get STRING_GNN device
        gnn_device = next(self.string_gnn.parameters()).device

        # Prepare conditioning signals via cond_proj (trainable)
        # cond_signal[b] = projected AIDO embedding for sample b
        cond_signal = self.cond_proj(gene_pos_emb)  # [B, 256]
        cond_signal = cond_signal.to(gnn_device)
        string_node_indices_dev = string_node_indices.to(gnn_device)

        # Build cond_emb: [N, 256] averaged over samples that target each node
        # For nodes with multiple samples (rare in small batch), average their signals
        cond_emb = torch.zeros(N, STRING_DIM, device=gnn_device, dtype=torch.float32)
        valid_mask = string_node_indices_dev >= 0  # [B] bool

        # Accumulate conditioning signals per node
        count = torch.zeros(N, device=gnn_device, dtype=torch.float32)
        valid_indices = string_node_indices_dev[valid_mask]       # [n_valid] int64
        valid_signals = cond_signal[valid_mask]                   # [n_valid, 256]

        if valid_indices.numel() > 0:
            # Use scatter_add to accumulate signals per node
            # Ensure dtypes match before scatter_add
            idx_expanded = valid_indices.unsqueeze(1).expand(-1, STRING_DIM)  # [n_valid, 256]
            cond_emb.scatter_add_(0, idx_expanded, valid_signals.float())
            count.scatter_add_(0, valid_indices,
                               torch.ones(valid_indices.numel(), device=gnn_device))
            # Normalize by count (average) where count > 0
            nonzero = count > 0
            cond_emb[nonzero] = cond_emb[nonzero] / count[nonzero].unsqueeze(1)

        # Run ONE STRING_GNN forward with batch-averaged conditioning
        edge_index_dev = self._edge_index_buf.to(gnn_device)
        edge_weight_dev = (
            self._edge_weight_buf.to(gnn_device)
            if hasattr(self, '_edge_weight_buf')
            else None
        )

        with torch.no_grad():
            out = self.string_gnn(
                edge_index=edge_index_dev,
                edge_weight=edge_weight_dev,
                cond_emb=cond_emb,
            )
        node_embs = out.last_hidden_state.float()  # [N, 256]

        # Extract conditioned embedding for each sample's perturbed gene
        result = torch.zeros(B, STRING_DIM, device=gnn_device, dtype=torch.float32)
        null_emb_dev = self.string_null_emb.float().to(gnn_device)

        for b in range(B):
            if valid_mask[b]:
                result[b] = node_embs[string_node_indices_dev[b]]
            else:
                result[b] = null_emb_dev

        return result  # [B, 256]

    def forward(
        self,
        input_ids: torch.Tensor,           # [B, 19264] float32 (AIDO.Cell input)
        pert_positions: torch.Tensor,      # [B] int64 (-1 for AIDO unknown)
        string_node_indices: torch.Tensor, # [B] int64 (-1 for STRING unknown)
    ) -> torch.Tensor:
        """Returns: [B, 3, N_GENES] logits."""
        B = input_ids.size(0)

        # ── AIDO.Cell-10M: frozen dual-pool ─────────────────────────────────
        gene_pos_emb, mean_pool, aido_dual = self._get_aido_dual_pool(
            input_ids, pert_positions)  # [B, 256], [B, 256], [B, 512]

        # ── Perturbation-conditioned STRING_GNN ─────────────────────────────
        str_feat = self._run_conditioned_string_gnn(
            gene_pos_emb, string_node_indices)  # [B, 256]

        # Move all features to the head's device
        head_device = next(self.head.parameters()).device
        str_feat = str_feat.to(head_device)
        aido_dual = aido_dual.to(head_device)

        # ── Feature fusion ───────────────────────────────────────────────────
        combined = torch.cat([str_feat, aido_dual], dim=-1)  # [B, 768]

        # ── Output head ──────────────────────────────────────────────────────
        logits = self.head(combined)                   # [B, 3*6640]
        return logits.view(B, N_CLASSES, N_GENES)      # [B, 3, 6640]


# ─────────────────────────────────────────────────────────────────────────────
# LightningModule
# ─────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        head_width: int = 128,
        head_dropout: float = 0.4,
        lr_head: float = 1e-3,
        weight_decay: float = 3e-2,
        gamma_focal: float = 2.5,
        label_smoothing: float = 0.12,
        max_epochs: int = 120,
        t0_warmrestart: int = 40,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialized in setup()
        self.model: Optional[PerturbationConditionedDualEncoder] = None
        self.criterion: Optional[FocalLoss] = None

        # Accumulators (cleared each epoch)
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            self.model = PerturbationConditionedDualEncoder(
                head_width=self.hparams.head_width,
                head_dropout=self.hparams.head_dropout,
            )
            self.model.initialize_aido()
            self.model.initialize_string_gnn()
            self.model.initialize_head()

            # Cast trainable parameters to float32 for stable optimization
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.float()

            self.criterion = FocalLoss(
                gamma=self.hparams.gamma_focal,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

            # Print trainable parameter count
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            self.print(f"Trainable params: {trainable:,} / {total:,} "
                       f"({100.0 * trainable / max(total, 1):.2f}%)")
            self.print(f"Trainable params per training sample: "
                       f"{trainable / 1500:.0f}")

        if stage == "test" and hasattr(self.trainer.datamodule, "test_pert_ids"):
            self._test_pert_ids = self.trainer.datamodule.test_pert_ids
            self._test_symbols = self.trainer.datamodule.test_symbols

    def forward(
        self,
        input_ids: torch.Tensor,
        pert_positions: torch.Tensor,
        string_node_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, pert_positions, string_node_indices)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits: [B, 3, G], labels: [B, G] ({0,1,2}) → scalar loss."""
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
        labels_flat = labels.reshape(-1)                       # [B*G]
        return self.criterion(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["pert_pos"], batch["string_node_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["pert_pos"], batch["string_node_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, dim=0)
        local_labels = torch.cat(self._val_labels, dim=0)
        local_idx = torch.cat(self._val_indices, dim=0)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        world_size = self.trainer.world_size if self.trainer.world_size else 1
        if world_size > 1:
            all_preds = self.all_gather(local_preds)
            all_labels = self.all_gather(local_labels)
            all_idx = self.all_gather(local_idx)

            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
            labels_flat = all_labels.view(-1, N_GENES).cpu().numpy()
            idx_flat = all_idx.view(-1).cpu().numpy()

            unique_pos = np.unique(idx_flat, return_index=True)[1]
            preds_flat = preds_flat[unique_pos]
            labels_flat = labels_flat[unique_pos]
            order = np.argsort(idx_flat[unique_pos])
            preds_flat = preds_flat[order]
            labels_flat = labels_flat[order]
            f1 = compute_deg_f1(preds_flat, labels_flat)
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        else:
            f1 = compute_deg_f1(local_preds.numpy(), local_labels.numpy())
            self.log("val_f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["pert_pos"], batch["string_node_idx"])
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, G]
        self._test_preds.append(probs)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        world_size = self.trainer.world_size if self.trainer.world_size else 1
        # n_expected is the full test set size (used for alignment/metadata).
        # n_actual is the number of rows we actually collected in this run
        # (may be < n_expected in debug/fast modes).
        n_expected = len(self._test_pert_ids)
        n_actual_local = local_preds.shape[0]

        if world_size > 1:
            # With a SequentialSampler in test_dataloader + use_distributed_sampler=False,
            # ALL ranks process the FULL test set in the same order.
            # After all_gather we get world_size copies of the same data.
            # Deduplicate by position: keep the first n_expected rows.
            all_preds = self.all_gather(local_preds)
            if self.trainer.is_global_zero:
                preds = all_preds.view(-1, N_CLASSES, N_GENES).cpu().numpy()
                total_collected = preds.shape[0]
                # Dedup: use min(total_collected, n_expected)
                # If total_collected < n_expected, warn but still save what we have.
                if total_collected < n_expected:
                    self.print(
                        f"WARNING: Collected {total_collected} < {n_expected} "
                        f"(limit_test_batches was active). Saving {total_collected} rows."
                    )
                    n_rows = total_collected
                else:
                    n_rows = n_expected
                    preds = preds[:n_rows]  # positional dedup

                output_dir = Path(__file__).parent / "run"
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / "test_predictions.tsv"
                rows = [
                    {"idx": self._test_pert_ids[i],
                     "input": self._test_symbols[i],
                     "prediction": json.dumps(preds[i].tolist())}
                    for i in range(n_rows)
                ]
                pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
                self.print(f"Test predictions saved ({n_rows} rows) → {out_path}")
        else:
            preds = local_preds.cpu().numpy()
            n_rows = preds.shape[0]
            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"
            rows = [
                {"idx": self._test_pert_ids[i],
                 "input": self._test_symbols[i],
                 "prediction": json.dumps(preds[i].tolist())}
                for i in range(n_rows)
            ]
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"Test predictions saved ({n_rows} rows) → {out_path}")

    def configure_optimizers(self):
        # Only trainable parameters: cond_proj + null_emb + head
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.lr_head,
            weight_decay=self.hparams.weight_decay,
        )
        # CosineAnnealingWarmRestarts with T_0=40, T_mult=2
        # Restarts at epochs: 40, 120 (40+80), 360 ...
        # For max_epochs=120, this gives 3 cycles: [0,40], [40,120]
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=self.hparams.t0_warmrestart,
            T_mult=2,
            eta_min=1e-6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    # ── Checkpoint: save only trainable parameters ────────────────────────────
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
        total = sum(p.numel() for p in self.parameters())
        tr_cnt = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Checkpoint: saving {tr_cnt:,}/{total:,} params "
            f"({100.0 * tr_cnt / max(total, 1):.2f}% trainable)"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Perturbation-Conditioned STRING_GNN + AIDO.Cell-10M DEG predictor"
    )
    p.add_argument("--data-dir",               type=str,
                   default=str(Path(__file__).parent.parent.parent / "data"))
    p.add_argument("--micro-batch-size",        type=int,   default=8)
    p.add_argument("--global-batch-size",       type=int,   default=64)
    p.add_argument("--max-epochs",              type=int,   default=120)
    p.add_argument("--lr-head",                 type=float, default=1e-3)
    p.add_argument("--weight-decay",            type=float, default=3e-2)
    p.add_argument("--head-width",              type=int,   default=128)
    p.add_argument("--head-dropout",            type=float, default=0.4)
    p.add_argument("--gamma-focal",             type=float, default=2.5)
    p.add_argument("--label-smoothing",         type=float, default=0.12)
    p.add_argument("--t0-warmrestart",          type=int,   default=40)
    p.add_argument("--early-stopping-patience", type=int,   default=30)
    p.add_argument("--num-workers",             type=int,   default=4)
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

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Distributed setup ────────────────────────────────────────────────────
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    micro_batch = args.micro_batch_size
    num_train_batches = 1500 // micro_batch          # 187 for mbs=8, drop_last=True
    num_val_batches   = (167 + micro_batch - 1) // micro_batch  # ceil: 21 for mbs=8
    num_test_batches  = num_val_batches              # same as val: 21 for mbs=8

    # Batch limit strategy:
    # For --fast-dev-run and --debug-max-step N, we use _HeadDataLoader
    # to cap the number of yielded batches. Here we set limit_*=1.0 so
    # Lightning processes all yielded batches (no further fraction applied).
    # max_steps controls the hard stop (1 for fast-dev-run, N for debug).
    # For full runs: no limits, max_steps=-1 (run to max_epochs).
    if args.fast_dev_run:
        _n_train = 1
        max_steps       = 1
        limit_train     = 1.0     # capped by _HeadDataLoader to 1 batch
        limit_val       = 1.0 / num_val_batches   # exactly 1 val batch (safe fraction)
        limit_test      = 1.0 / num_test_batches  # exactly 1 test batch
    elif args.debug_max_step is not None:
        _n_train = min(args.debug_max_step, num_train_batches)
        max_steps       = args.debug_max_step
        limit_train     = 1.0     # capped by _HeadDataLoader to _n_train batches
        limit_val       = float(min(args.debug_max_step, num_val_batches)) / num_val_batches
        limit_test      = 1.0     # full test set
    else:
        _n_train = -1   # -1 = no cap (use full dataloader)
        max_steps   = -1
        limit_train = limit_val = limit_test = 1.0

    val_check_interval = args.val_check_interval if (
        args.debug_max_step is None and not args.fast_dev_run
    ) else 1.0

    # NOTE: The batched STRING_GNN forward uses cond_emb which adds conditioning
    # to model buffers (_edge_index_buf etc.); these are not "unused" params
    # per se but DDP find_unused_parameters=True is safer for frozen backbones.
    if n_gpus == 1:
        strategy: Any = SingleDeviceStrategy(device="cuda:0")
    else:
        strategy = DDPStrategy(
            find_unused_parameters=True,  # Frozen backbone params are not in grad graph
            timeout=timedelta(seconds=300),
        )

    # ── Callbacks ────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node1-2-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",    # Critical: monitor val_f1 not val_loss
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",    # Critical: monitor val_f1 not val_loss
        mode="max",
        patience=args.early_stopping_patience,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # ── Loggers ──────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # ── Trainer ──────────────────────────────────────────────────────────────
    # NOTE on fast_dev_run: we explicitly pass fast_dev_run=False to prevent
    # Lightning from overriding our limit_*_batches settings with internal
    # fast_dev_run defaults (e.g., limit_test_batches=1, num_sanity_val_steps=0).
    # Batch limits are controlled exclusively by limit_train/val/test_batches.
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
        fast_dev_run=False,  # Explicit False: let limit_* control batch counts
        gradient_clip_val=1.0,
        use_distributed_sampler=False,
    )

    # ── Data & model ─────────────────────────────────────────────────────────
    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        head_width=args.head_width,
        head_dropout=args.head_dropout,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        t0_warmrestart=args.t0_warmrestart,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    # For debug modes, override the train dataloader with a capped version.
    # We patch datamodule.train_dataloader to return limited batches.
    _orig_train_dataloader = datamodule.train_dataloader
    if _n_train > 0:
        datamodule.train_dataloader = lambda: datamodule._limited_train_dataloader(_n_train)
    elif _n_train == 0:
        datamodule.train_dataloader = lambda: datamodule._limited_train_dataloader(0)

    trainer.fit(model_module, datamodule=datamodule)

    # Restore original dataloader
    datamodule.train_dataloader = _orig_train_dataloader

    # ── Test ─────────────────────────────────────────────────────────────────
    # We use a plain DataLoader with shuffle=False. Lightning will wrap it with
    # DistributedSampler in DDP mode, partitioning samples across ranks.
    # Each rank processes ALL test samples in the same order (no shuffling).
    # In on_test_epoch_end, we use all_gather to collect all predictions and
    # deduplicate by POSITION (first 167 entries since all ranks have same order).
    from torch.utils.data import DataLoader
    test_ds = datamodule.test_ds
    # Ensure model has test_pert_ids/symbols set (may not be set after fit)
    if not model_module._test_pert_ids:
        model_module._test_pert_ids = datamodule.test_pert_ids
        model_module._test_symbols = datamodule.test_symbols
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(model_module, datamodule=datamodule, ckpt_path="best")

    # ── Save test score ───────────────────────────────────────────────────────
    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        best_val_f1 = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else None
        )
        score_path.write_text(
            f"test_results: {test_results}\n"
            f"val_f1_best: {best_val_f1}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
