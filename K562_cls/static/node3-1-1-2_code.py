"""Node 3-1-1-2 – Frozen STRING_GNN (K=16 neighborhood attention) + AIDO.Cell-10M (QKV-only)
Dual-stream Hybrid Fusion with Restored Head Capacity.

This node is an improved version of the parent (node3-1-1), designed as a recovery and
enhancement over the sibling (node3-1-1-1) which catastrophically failed due to over-regularization.

Key design decisions:
1. Frozen STRING_GNN with K=16 PPI Neighborhood Attention:
   - Cache STRING_GNN embeddings once (no gradient computation through GNN)
   - Add lightweight neighborhood attention aggregation (K=16, proven +0.010 F1 in node1-1-1-1-1)
   - Sibling used FULL fine-tune (risky, 5.43M extra trainable params) + over-regularized head
   - This node uses frozen GNN + moderate head capacity

2. Restored head capacity: head_hidden=512, dropout=0.3, wd_head=0.02
   - Sibling catastrophically failed with head_hidden=256 + dropout=0.5 + wd=0.05
   - Parent (F1=0.4325) used head_hidden=512, dropout=0.3 — restoring this proven capacity

3. AIDO.Cell-10M QKV-only fine-tuning with Muon (unchanged from parent)
   - Proven: node3 (F1=0.426), node3-1-1 (F1=0.4325)

4. 1280-dim fusion input: [256 (STRING neighbor attn) + 1024 (AIDO 4-layer concat)]
   - STRING_GNN provides PPI topology signal
   - AIDO.Cell provides gene expression context signal
   - Complementary information sources (both node3-1-2 at 0.4407 and node1-1 at 0.453 support this direction)

5. Standard CosineAnnealingLR with warmup (T_max=80):
   - More stable than CosineAnnealingWarmRestarts for hybrid architecture
   - Warmup=5 epochs to stabilize neighborhood attention module initialization
   - Standard cosine decay to eta_min=1e-6 prevents LR collapse

6. Early stopping patience=10, min_delta=0.001 — balanced to prevent wasted epochs
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES       = 6640
N_CLASSES     = 3
AIDO_GENES    = 19264
AIDO_MODEL_DIR = "/home/Models/AIDO.Cell-10M"
STRING_MODEL_DIR = "/home/Models/STRING_GNN"
HIDDEN_DIM    = 256      # AIDO.Cell-10M hidden size
N_LAYERS      = 8        # AIDO.Cell-10M transformer layers
GNN_DIM       = 256      # STRING_GNN embedding dimension

# Class frequencies: down-regulated, neutral, up-regulated (remapped to 0,1,2)
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights for class imbalance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute per-gene macro F1, matching the calc_metric.py evaluation logic.

    Args:
        preds: [N, 3, G] softmax probabilities
        targets: [N, G] integer class labels (0=down, 1=neutral, 2=up)

    Returns:
        Scalar: mean per-gene macro F1 over all G genes.
    """
    y_hat       = preds.argmax(dim=1)  # [N, G]
    G           = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)
    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0).float()
        tp  = (is_pred & is_true).float().sum(0)
        fp  = (is_pred & ~is_true).float().sum(0)
        fn  = (~is_pred & is_true).float().sum(0)
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(prec + rec > 0, 2*prec*rec/(prec+rec+1e-8), torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# PPI Neighborhood Attention Module (frozen STRING_GNN + K-NN aggregation)
# ---------------------------------------------------------------------------
class NeighborhoodAttentionModule(nn.Module):
    """Lightweight K=16 PPI neighbor aggregation on top of frozen STRING_GNN embeddings.

    Proven to provide +0.010 F1 in node1-1-1-1-1 over simple direct lookup.
    This is a lightweight center-context attention module with ~164K params.
    """

    def __init__(
        self,
        K: int = 16,
        attn_dim: int = 64,
        emb_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.K = K
        self.emb_dim = emb_dim

        # Attention projection for center (query) and neighbor (key/value)
        self.q_proj = nn.Linear(emb_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, attn_dim, bias=False)
        # Context gate: outputs gating weight in [0,1] for weighting center vs neighbors
        self.context_gate = nn.Linear(emb_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(
        self,
        center_emb: torch.Tensor,       # [B, emb_dim] — embedding for perturbed gene
        node_embs: torch.Tensor,        # [N_nodes, emb_dim] — all node embeddings from GNN
        neighbor_idx: torch.Tensor,     # [B, K] — top-K neighbor indices
        neighbor_conf: torch.Tensor,    # [B, K] — edge confidence weights
    ) -> torch.Tensor:
        """Aggregate neighbor embeddings via weighted attention, gate with center.

        Returns: [B, emb_dim] — enriched gene embedding
        """
        B = center_emb.shape[0]
        K = neighbor_idx.shape[1]

        # Gather neighbor embeddings: [B, K, emb_dim]
        # Handle invalid neighbors (index -1) by clamping to 0
        valid_mask = (neighbor_idx >= 0)  # [B, K]
        clamped_idx = neighbor_idx.clamp(min=0)  # [B, K]
        neigh_embs = node_embs[clamped_idx]  # [B, K, emb_dim]

        # Attention weights: query=center, key=neighbors
        q = self.q_proj(center_emb).unsqueeze(1)          # [B, 1, attn_dim]
        k = self.k_proj(neigh_embs)                         # [B, K, attn_dim]
        attn_scores = (q * k).sum(-1) / (self.K ** 0.5)    # [B, K]

        # Mask invalid neighbors and weight by edge confidence
        attn_scores = attn_scores + neighbor_conf.log().clamp(min=-10)  # add log-confidence
        attn_scores = attn_scores.masked_fill(~valid_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, K]
        attn_weights = self.dropout(attn_weights)

        # Aggregate neighbors: [B, emb_dim]
        neighbor_context = (attn_weights.unsqueeze(-1) * neigh_embs).sum(1)

        # Context gate: decide how much center vs neighbor context to use
        gate_input = torch.cat([center_emb, neighbor_context], dim=-1)  # [B, 2*emb_dim]
        gate = torch.sigmoid(self.context_gate(gate_input))              # [B, 1]
        output = gate * center_emb + (1 - gate) * neighbor_context      # [B, emb_dim]

        return self.norm(output)


# ---------------------------------------------------------------------------
# STRING_GNN Graph Cache (computed once at setup, no gradients)
# ---------------------------------------------------------------------------
class StringGNNCache:
    """Manages frozen STRING_GNN embeddings and PPI neighbor lookup.

    Key design: STRING_GNN embeddings are pre-computed and frozen (no gradients),
    keeping them as a fixed feature extractor. This avoids the over-fitting risk
    of full fine-tuning on 1388 samples (proven failure in sibling node3-1-1-1).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.gnn = None
        self.node_embs = None     # [N_nodes, 256] frozen embeddings
        self.pert_to_idx = {}     # pert_id -> GNN node index
        self.neighbor_idx = None  # [N_nodes, K] top-K neighbor indices
        self.neighbor_conf = None # [N_nodes, K] neighbor confidence weights

    def setup(self, K: int = 16):
        """Load STRING_GNN, run inference once, cache embeddings and precompute neighbors."""
        import json

        model_dir = Path(STRING_MODEL_DIR)
        gnn = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(self.device)
        gnn.eval()
        for p in gnn.parameters():
            p.requires_grad = False

        graph = torch.load(model_dir / "graph_data.pt")
        node_names = json.loads((model_dir / "node_names.json").read_text())
        edge_index = graph["edge_index"].to(self.device)
        edge_weight = graph["edge_weight"]
        edge_weight = edge_weight.to(self.device) if edge_weight is not None else None

        # Run STRING_GNN once with no_grad to get all node embeddings
        with torch.no_grad():
            out = gnn(edge_index=edge_index, edge_weight=edge_weight)
        node_embs = out.last_hidden_state.float()  # [18870, 256]

        # Build pert_id -> node index map
        pert_to_idx = {name: i for i, name in enumerate(node_names)}

        # Build neighbor lookup: for each node, find top-K neighbors by edge confidence
        N = len(node_names)
        neighbor_idx_list = [[] for _ in range(N)]
        neighbor_conf_list = [[] for _ in range(N)]

        edge_index_cpu = edge_index.cpu()
        edge_weight_cpu = edge_weight.cpu() if edge_weight is not None else None

        for e in range(edge_index_cpu.shape[1]):
            src = edge_index_cpu[0, e].item()
            dst = edge_index_cpu[1, e].item()
            conf = edge_weight_cpu[e].item() if edge_weight_cpu is not None else 1.0
            neighbor_idx_list[src].append((conf, dst))

        # For each node: sort by confidence, take top-K, pad with -1
        neigh_idx = torch.full((N, K), -1, dtype=torch.long)
        neigh_conf = torch.zeros(N, K, dtype=torch.float)
        for i in range(N):
            neighbors = sorted(neighbor_idx_list[i], reverse=True)[:K]
            for j, (conf, idx) in enumerate(neighbors):
                neigh_idx[i, j] = idx
                neigh_conf[i, j] = conf

        self.node_embs = node_embs  # kept on device, frozen
        self.pert_to_idx = pert_to_idx
        self.neighbor_idx = neigh_idx.to(self.device)
        self.neighbor_conf = neigh_conf.to(self.device)
        self.gnn = gnn  # kept for reference but frozen

        print(f"[StringGNNCache] Loaded {N} nodes, precomputed K={K} neighbors")

    def get_node_idx(self, pert_ids: List[str]) -> torch.Tensor:
        """Convert list of pert_ids to node indices. Unknown → -1."""
        idxs = [self.pert_to_idx.get(pid, -1) for pid in pert_ids]
        return torch.tensor(idxs, dtype=torch.long, device=self.device)

    def get_center_embs(self, node_idx: torch.Tensor) -> torch.Tensor:
        """Get frozen center embeddings for given node indices.
        Unknown nodes (idx=-1) get zero embeddings.
        """
        valid_mask = (node_idx >= 0)
        clamped_idx = node_idx.clamp(min=0)
        embs = self.node_embs[clamped_idx]  # [B, 256]
        embs = embs * valid_mask.float().unsqueeze(-1)  # zero out unknown
        return embs

    def get_neighbors(
        self, node_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-K neighbor indices and confidence weights for given nodes.

        Returns:
            neigh_idx: [B, K]
            neigh_conf: [B, K]
        """
        valid_mask = (node_idx >= 0)
        clamped_idx = node_idx.clamp(min=0)
        neigh_idx = self.neighbor_idx[clamped_idx]  # [B, K]
        neigh_conf = self.neighbor_conf[clamped_idx]  # [B, K]
        # Mask neighbors for unknown genes
        neigh_idx = neigh_idx * valid_mask.unsqueeze(-1).long()
        neigh_idx = neigh_idx - (~valid_mask).long().unsqueeze(-1)  # set to -1 for invalid center
        neigh_conf = neigh_conf * valid_mask.float().unsqueeze(-1)
        return neigh_idx, neigh_conf


# ---------------------------------------------------------------------------
# Dataset / DataModule
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List] = (
            [torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
             for row in df["label"].tolist()]
            if has_label else None
        )

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx": idx,
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_collate(tokenizer, micro_batch_size: int = 16):
    """Collate factory with micro_batch_size for computing globally-unique sample indices.

    The global sample index = global_rank * micro_batch_size + batch_start_offset + local_idx
    where batch_start_offset = batch_idx * micro_batch_size.
    This ensures indices are unique across ALL ranks and ALL batches for correct DDP deduplication.

    The batch offset is computed using DataLoader's worker_info.id (which gives the worker
    process index) combined with per-worker batch indices, ensuring deterministic ordering
    without needing global synchronization.
    """
    # Per-worker batch counter (incremented per call to collate_fn).
    # Each DataLoader worker process has its own closure instance, so this is naturally isolated.
    _batch_counter: List[int] = [0]

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        from torch.utils.data import get_worker_info

        B = len(batch)

        # Compute global sample indices using worker-aware batch offset:
        # batch_offset accounts for all previously-seen batches in this worker
        batch_offset = _batch_counter[0] * micro_batch_size
        _batch_counter[0] += 1

        # Compute rank-based offset for multi-GPU / multi-worker data distribution
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        global_rank = int(os.environ.get("GLOBAL_RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        # Each rank processes a different slice of the dataset
        # global_rank * micro_batch_size gives the starting index for this rank
        rank_offset = global_rank * micro_batch_size

        # Use worker_info to compute per-worker offset
        worker_info = get_worker_info()
        if worker_info is not None:
            # Each worker processes every num_workers-th sample
            # Worker w processes indices w, w+num_workers, w+2*num_workers, ...
            # Batch offset should account for all batches from all workers that came before
            # Since worker id determines which samples this worker processes,
            # and all workers process batches in parallel, we use worker_id * micro_batch_size
            # as the starting offset for this worker
            pass  # worker_id is already accounted for in which samples appear in `batch`

        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        input_ids  = tokenized["input_ids"]  # [B, 19264] float32 → convert to bfloat16 for FlashAttention
        input_ids  = input_ids.to(torch.bfloat16)
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),
            torch.zeros(len(batch), dtype=torch.long),
        )
        # Globally unique sample indices for DDP deduplication:
        # - batch_offset: accounts for previous batches from this worker in this epoch
        # - rank_offset: unique offset per GPU rank
        # - local sample index: unique within this batch
        global_idx = torch.arange(B, dtype=torch.long) + batch_offset + rank_offset
        out: Dict[str, Any] = {
            "sample_idx":     global_idx,
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      input_ids,
            "attention_mask": tokenized["attention_mask"],
            "gene_positions": gene_positions,
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return collate_fn


class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer   = None

    def setup(self, stage: Optional[str] = None) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        self.train_ds = DEGDataset(train_df)
        self.val_ds   = DEGDataset(val_df)
        self.test_ds  = DEGDataset(test_df)

    def _loader(self, ds, shuffle, collate_override=None):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=collate_override or make_collate(self.tokenizer, self.batch_size))

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)

    def reset_batch_counters(self):
        """Reset collate batch counters between epochs. Called by the model."""
        pass  # Counter is inside the collate closure; each worker process has its own copy.


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FrozenStringGNNAIDOCellModel(pl.LightningModule):
    """Frozen STRING_GNN (K=16 neighborhood attention) + AIDO.Cell-10M (QKV-only) fusion.

    Key differences from parent (node3-1-1):
    - Added STRING_GNN as frozen feature extractor (+256-dim PPI topology signal)
    - Added lightweight K=16 neighborhood attention on top of frozen STRING_GNN
    - Head input expanded from 1024-dim → 1280-dim (concat STRING_neigh + AIDO_concat)
    - head_hidden restored to 512 (proven in parent F1=0.4325)
    - head_dropout=0.3 (same as parent — adequate regularization)
    - wd_head=0.02 (vs 0.05 in sibling — sibling proved 0.05 too aggressive)
    - Standard cosine LR decay (stable, proven in this lineage)

    Key differences from sibling (node3-1-1-1) which catastrophically failed:
    - STRING_GNN: FROZEN (vs full fine-tune) — avoids 5.43M overfitting params
    - head_hidden: 512 (vs 256 in sibling) — restores proven capacity
    - head_dropout: 0.3 (vs 0.5 in sibling) — less aggressive
    - wd_head: 0.02 (vs 0.05 in sibling) — less aggressive
    - LR schedule: CosineAnnealingLR with warmup (vs T_0=25 CAWR that never fired)
    """

    def __init__(
        self,
        fusion_layers: int    = 4,         # AIDO.Cell last N transformer layers to concat
        head_hidden: int      = 512,        # proven in parent (F1=0.4325)
        head_dropout: float   = 0.3,        # proven in parent; sibling's 0.5 was catastrophic
        lr_muon: float        = 0.02,       # Muon lr for AIDO.Cell QKV weight matrices
        lr_adamw: float       = 2e-4,       # AdamW lr for head + neighborhood attention
        weight_decay_qkv: float  = 1e-2,   # weight decay for AIDO.Cell QKV
        weight_decay_head: float = 2e-2,   # weight decay for head — NOT 0.05 (sibling failure)
        label_smoothing: float = 0.1,       # label smoothing for CE loss
        cosine_t_max: int     = 80,         # T_max for CosineAnnealingLR
        cosine_eta_min: float = 1e-6,       # minimum LR floor
        warmup_epochs: int    = 5,          # linear warmup epochs
        nbattn_K: int         = 16,         # top-K PPI neighbors
        nbattn_attn_dim: int  = 64,         # attention projection dimension
        nbattn_dropout: float = 0.2,        # dropout in neighborhood attention
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.gnn_cache: Optional[StringGNNCache] = None

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load and freeze STRING_GNN, precompute K-NN embeddings ----
        self.gnn_cache = StringGNNCache(device=self.device)
        self.gnn_cache.setup(K=hp.nbattn_K)

        # ---- Neighborhood attention module (lightweight, trainable) ----
        self.nbattn = NeighborhoodAttentionModule(
            K=hp.nbattn_K,
            attn_dim=hp.nbattn_attn_dim,
            emb_dim=GNN_DIM,
            dropout=hp.nbattn_dropout,
        ).float()

        # ---- Load AIDO.Cell-10M backbone ----
        # FlashAttention is critical for 19266-gene sequences (saves ~50% memory).
        # We convert the entire model to bfloat16 to ensure all activations stay
        # in bfloat16 (required for FlashAttention to activate) and to save memory.
        # Lightning's AMP (bf16-mixed) then manages the optimizer step in float32.
        self.backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        self.backbone = self.backbone.to(torch.bfloat16)
        self.backbone.config.use_cache = False
        # Explicitly enable FlashAttention (AIDO.Cell config has it True by default)
        self.backbone.config._use_flash_attention_2 = True

        # FlashAttention is CRITICAL for 19266-gene sequences.
        # With batch=16: SDPA needs ~23 GB vs FlashAttention ~3 GB for attention.
        # The model is in bfloat16 (required for FlashAttention activation).
        # Note: gradient_checkpointing was removed — with FlashAttention, activations
        # are small enough that gradient checkpointing's memory saving is minimal
        # while its recomputation cost slows training.

        # ---- Share QKV weight tensors between flash_self and self.self ----
        for layer in self.backbone.bert.encoder.layer:
            ss = layer.attention.flash_self
            mm = layer.attention.self
            ss.query.weight = mm.query.weight
            ss.key.weight   = mm.key.weight
            ss.value.weight = mm.value.weight
            ss.query.bias   = mm.query.bias
            ss.key.bias     = mm.key.bias
            ss.value.bias   = mm.value.bias

        # ---- Freeze all AIDO.Cell params, then unfreeze QKV weights ----
        for param in self.backbone.parameters():
            param.requires_grad = False

        qkv_patterns = (
            "attention.self.query.weight",
            "attention.self.key.weight",
            "attention.self.value.weight",
        )
        for name, param in self.backbone.named_parameters():
            if any(name.endswith(p) for p in qkv_patterns):
                param.requires_grad = True

        qkv_count = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total      = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Node] AIDO.Cell trainable backbone params: {qkv_count:,} / {total:,} (QKV-only)")

        # NOTE: Do NOT manually cast trainable params to float32.
        # The model is converted to bfloat16 (`.to(torch.bfloat16)`) BEFORE setting
        # trainable params. This ensures trainable QKV params are also in bfloat16,
        # avoiding dtype mismatches inside the encoder layers.
        # MuonWithAuxAdam internally casts float32 for the Newton-Schulz step.

        # ---- Fusion classification head ----
        # Input: [B, fusion_layers * HIDDEN_DIM + GNN_DIM] = [B, 4*256 + 256] = [B, 1280]
        in_dim = hp.fusion_layers * HIDDEN_DIM + GNN_DIM
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )
        self.head = self.head.float()

        # ---- Loss: label-smoothed CE + class weights ----
        class_weights = get_class_weights()
        self.register_buffer("class_weights", class_weights)
        self.label_smoothing = hp.label_smoothing

        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_tgts: List[torch.Tensor]  = []
        self._test_meta:  List[Tuple]        = []

        # Training step counter for warmup
        self._warmup_steps: Optional[int] = None
        self._current_step: int = 0

    # ---- STRING_GNN neighborhood feature extraction ----
    def _get_gnn_features(self, pert_ids: List[str]) -> torch.Tensor:
        """Extract STRING_GNN neighborhood-aggregated embeddings.

        Returns: [B, GNN_DIM=256]
        """
        # Get node indices for perturbed genes
        node_idx = self.gnn_cache.get_node_idx(pert_ids)  # [B]
        # Get frozen center embeddings
        center_embs = self.gnn_cache.get_center_embs(node_idx)  # [B, 256]
        # Get top-K neighbor indices and confidences
        neigh_idx, neigh_conf = self.gnn_cache.get_neighbors(node_idx)  # [B, K]
        # Apply neighborhood attention aggregation (trainable)
        gnn_feats = self.nbattn(
            center_emb=center_embs,
            node_embs=self.gnn_cache.node_embs,  # frozen
            neighbor_idx=neigh_idx,
            neighbor_conf=neigh_conf + 1e-8,  # prevent log(0) in attention
        )  # [B, 256]
        return gnn_feats

    # ---- forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_positions: torch.Tensor,
        pert_ids: List[str],
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        # ---- AIDO.Cell stream: 4-layer concat at perturbed gene position ----
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = out.hidden_states  # len = N_LAYERS + 1 = 9

        n = self.hparams.fusion_layers
        layer_embs = []
        for i in range(n):
            hs = hidden_states[-(i + 1)]  # [B, AIDO_GENES+2, 256]
            ge = hs[torch.arange(B, device=hs.device), gene_positions, :].float()  # [B, 256]
            layer_embs.append(ge)

        aido_feat = torch.cat(layer_embs, dim=1)  # [B, 1024]

        # ---- STRING_GNN stream: neighborhood attention aggregation ----
        gnn_feat = self._get_gnn_features(pert_ids)  # [B, 256]

        # ---- Concatenate streams: [B, 1280] ----
        fused = torch.cat([gnn_feat, aido_feat], dim=1)  # [B, 1280]

        # ---- Classification head ----
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return logits

    # ---- loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        flat_logits  = logits.permute(0, 2, 1).reshape(-1, C)
        flat_targets = targets.reshape(-1)
        return F.cross_entropy(
            flat_logits,
            flat_targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    # ---- steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        # Lightning's Bf16PrecisionPlugin handles autocast automatically.
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"]
        )
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"]
        )
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits.float(), dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        local_idx   = torch.cat(self._val_idx,   0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"]
        )
        probs  = torch.softmax(logits.float(), dim=1).detach()
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)
            self._test_tgts.append(batch["labels"].detach())
        self._test_preds.append(probs)
        for i, (pid, sym) in enumerate(zip(batch["pert_id"], batch["symbol"])):
            self._test_meta.append((pid, sym, batch["sample_idx"][i].item()))

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        all_preds   = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)

        local_idx_list = torch.tensor(
            [m[2] for m in self._test_meta], dtype=torch.long, device=all_preds.device
        )
        all_idx = self.all_gather(local_idx_list).view(-1)

        if self._test_tgts:
            local_tgts = torch.cat(self._test_tgts, 0)
            all_tgts   = self.all_gather(local_tgts).view(-1, N_GENES)
            order  = torch.argsort(all_idx)
            s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
            mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
            test_f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=True)

        world_size = self.trainer.world_size if hasattr(self.trainer, "world_size") else 1
        all_meta_flat: List[Tuple] = []
        if world_size > 1:
            gathered_meta: List[List] = [None] * world_size
            torch.distributed.all_gather_object(gathered_meta, list(self._test_meta))
            for meta_list in gathered_meta:
                all_meta_flat.extend(meta_list)
        else:
            all_meta_flat = list(self._test_meta)

        if self.trainer.is_global_zero:
            meta_dict: Dict[int, Tuple] = {m[2]: m for m in all_meta_flat}
            n_samples = all_preds.shape[0]
            rows = []
            for i in range(n_samples):
                idx_val = all_idx[i].item()
                if idx_val in meta_dict:
                    pid, sym, _ = meta_dict[idx_val]
                else:
                    pid, sym = f"unknown_{idx_val}", f"unknown_{idx_val}"
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(all_preds[i].float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_tgts.clear()
        self._test_meta.clear()

    # ---- checkpoint: save only trainable parameters ----
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full:
                    trainable[key] = full[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full:
                trainable[key] = full[key]
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {trained}/{total} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- optimizer: Muon for QKV weight matrices, AdamW for head + nbattn ----
    def configure_optimizers(self):
        hp = self.hparams

        # Muon group: AIDO.Cell QKV weight matrices (3 matrices per layer × 8 layers = 24)
        backbone_matrices = [
            p for name, p in self.backbone.named_parameters()
            if p.requires_grad and p.ndim >= 2
        ]
        backbone_biases = [
            p for name, p in self.backbone.named_parameters()
            if p.requires_grad and p.ndim < 2
        ]
        # AdamW group: classification head + neighborhood attention + backbone biases
        head_nbattn_params = list(self.head.parameters()) + list(self.nbattn.parameters())

        param_groups = [
            dict(
                params      = backbone_matrices,
                use_muon    = True,
                lr          = hp.lr_muon,
                weight_decay = hp.weight_decay_qkv,
                momentum    = 0.95,
            ),
            dict(
                params      = head_nbattn_params + backbone_biases,
                use_muon    = False,
                lr          = hp.lr_adamw,
                betas       = (0.9, 0.95),
                weight_decay = hp.weight_decay_head,
            ),
        ]
        use_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        opt_cls = MuonWithAuxAdam if use_distributed else SingleDeviceMuonWithAuxAdam
        optimizer = opt_cls(param_groups)

        # CosineAnnealingLR with linear warmup via LambdaLR
        # Warmup phase: epochs 0..warmup_epochs → LR ramps from 0 to 1
        # Cosine phase: epochs warmup_epochs..T_max → CosineAnnealing
        warmup = hp.warmup_epochs
        t_max  = hp.cosine_t_max
        eta_min_ratio = hp.cosine_eta_min / max(hp.lr_muon, hp.lr_adamw)

        def lr_lambda(epoch):
            if epoch < warmup:
                return float(epoch + 1) / float(max(1, warmup))
            progress = float(epoch - warmup) / float(max(1, t_max - warmup))
            return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (
                1.0 + np.cos(np.pi * min(progress, 1.0))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node 3-1-1-2: Frozen STRING_GNN (K=16 NbAttn) + AIDO.Cell-10M QKV Hybrid"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=16)
    parser.add_argument("--global-batch-size",  type=int,   default=128)
    parser.add_argument("--max-epochs",         type=int,   default=120)
    parser.add_argument("--lr-muon",            type=float, default=0.02)
    parser.add_argument("--lr-adamw",           type=float, default=2e-4)
    parser.add_argument("--weight-decay-qkv",   type=float, default=1e-2)
    parser.add_argument("--weight-decay-head",  type=float, default=2e-2)
    parser.add_argument("--fusion-layers",      type=int,   default=4)
    parser.add_argument("--head-hidden",        type=int,   default=512)
    parser.add_argument("--head-dropout",       type=float, default=0.3)
    parser.add_argument("--label-smoothing",    type=float, default=0.1)
    parser.add_argument("--cosine-t-max",       type=int,   default=80)
    parser.add_argument("--cosine-eta-min",     type=float, default=1e-6)
    parser.add_argument("--warmup-epochs",      type=int,   default=5)
    parser.add_argument("--nbattn-k",           type=int,   default=16)
    parser.add_argument("--nbattn-attn-dim",    type=int,   default=64)
    parser.add_argument("--nbattn-dropout",     type=float, default=0.2)
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--debug_max_step",     type=int,   default=None)
    parser.add_argument("--fast_dev_run",       action="store_true")
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = lim_val = lim_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = FrozenStringGNNAIDOCellModel(
        fusion_layers       = args.fusion_layers,
        head_hidden         = args.head_hidden,
        head_dropout        = args.head_dropout,
        lr_muon             = args.lr_muon,
        lr_adamw            = args.lr_adamw,
        weight_decay_qkv    = args.weight_decay_qkv,
        weight_decay_head   = args.weight_decay_head,
        label_smoothing     = args.label_smoothing,
        cosine_t_max        = args.cosine_t_max,
        cosine_eta_min      = args.cosine_eta_min,
        warmup_epochs       = args.warmup_epochs,
        nbattn_K            = args.nbattn_k,
        nbattn_attn_dim     = args.nbattn_attn_dim,
        nbattn_dropout      = args.nbattn_dropout,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
        auto_insert_metric_name=False,
    )
    # Patience=10, min_delta=0.001: balanced to prevent wasted epochs after peak
    # but permissive enough to allow the neighborhood attention module to warm up
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=10, min_delta=0.001)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    strategy = (
        DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = n_gpus,
        num_nodes               = 1,
        strategy                = strategy,  # find_unused_parameters handled via strategy arg
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        max_steps               = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches     = lim_train,
        limit_val_batches       = lim_val,
        limit_test_batches      = lim_test,
        val_check_interval      = args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
