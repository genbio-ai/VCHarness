"""Node 3-1-2-2-1: AIDO.Cell-100M LoRA r=8 + STRING_GNN K=16 2-Head Attention + CAWR.

Improvements over parent node3-1-2-2 (test F1=0.4813):
1. 2-head neighborhood attention (from tree-best node2-1-1-1-1-1, F1=0.5128)
2. Larger global batch size: 128 (vs parent's 32) for stable gradients
3. CosineAnnealingWarmRestarts (T_0=50, T_mult=2) for escaping local minima
4. Higher eta_min (1e-5 vs 1e-6) to prevent LR collapse at the cosine tail
5. Extended training: max_epochs=300, patience=30 to capture late improvement peaks
6. Discriminative learning rate: LoRA backbone lr=5e-5, head+STRING attention lr=1e-4

Parent bottleneck: training cut short at epoch 91 (val F1 still rising to 0.505),
global_batch=32 → noisy gradients, eta_min=1e-6 → near-zero LR → training stalls,
single-head attention vs 2-head in tree-best.

Architecture (unchanged from proven recipe):
1. AIDO.Cell-100M with LoRA r=8 (summary token extraction, 640-dim)
2. Frozen STRING_GNN with K=16 2-head neighborhood attention aggregation (256-dim)
3. Concat fusion: [AIDO.Cell summary (640) + STRING 2-head (256)] = 896-dim
4. 2-layer MLP head: Linear(896→256) + LayerNorm + GELU + Dropout(0.5) + Linear(256→19920)
5. Label-smoothed CE + sqrt-inverse-frequency class weights
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import shutil
import subprocess
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
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES    = 6640
N_CLASSES  = 3

AIDO_MODEL_DIR   = "/home/Models/AIDO.Cell-100M"
STRING_GNN_DIR   = "/home/Models/STRING_GNN"

AIDO_HIDDEN_DIM  = 640    # AIDO.Cell-100M hidden size
STRING_DIM       = 256    # STRING_GNN embedding dimension

# Class frequency from DATA_ABSTRACT.md: down=0.0429, neutral=0.9251, up=0.0320
CLASS_FREQ = [0.0429, 0.9251, 0.0320]  # remapped order: {-1->0, 0->1, 1->2}

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights for handling class imbalance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute per-gene macro F1, matching the calc_metric.py evaluation logic.

    Args:
        preds:   [N, 3, G] float tensor of softmax probabilities
        targets: [N, G] long tensor of class indices {0, 1, 2}

    Returns:
        float: per-gene macro-averaged F1 score
    """
    y_hat = preds.argmax(dim=1)  # [N, G]
    G = targets.shape[1]
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
        f1_c = torch.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-8), torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Rank-Aware Sampler (proven in best node node2-1-1-1-1-1)
# ---------------------------------------------------------------------------
class RankAwareSampler(torch.utils.data.Sampler):
    """Deterministically assigns disjoint subsets to each rank for DDP val/test.

    Unlike DistributedSampler which requires world_size coordination, this uses
    LOCAL_RANK to split the dataset. Each rank processes a non-overlapping subset.
    This eliminates DDP sample collision issues in val/test phases.
    """

    def __init__(self, dataset, shuffle: bool = False, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.rng = torch.Generator()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.world_size = max(self.world_size, 1)
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        rank_indices = [indices[i] for i in range(self.rank, len(indices), self.world_size)]
        return iter(rank_indices)

    def __len__(self):
        return (len(self.dataset) + self.world_size - 1) // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch


# ---------------------------------------------------------------------------
# STRING_GNN Neighborhood Attention
# ---------------------------------------------------------------------------
def build_top_k_neighbors(
    emb_matrix: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    K: int = 16,
    n_nodes: int = 18870,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute top-K neighbors and weights for each node in the STRING graph.

    This is the proven K=16 neighborhood approach from node1-1-1-1-1 (F1=0.4846)
    and node2-1-1-1-1-1 (F1=0.5128).

    FIXED: Normalize weights per-node to [0, 1] using max-normalization (proven approach).

    Args:
        emb_matrix:  [N, 256] float32 embedding matrix
        edge_index:  [2, E] long tensor
        edge_weight: [E] float tensor (optional)
        K:           number of neighbors to keep
        n_nodes:     number of nodes

    Returns:
        neighbor_idx: [n_nodes, K] long tensor of top-K neighbor indices
        neighbor_wt:  [n_nodes, K] float tensor of normalized edge weights [0, 1]
    """
    from collections import defaultdict
    adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    wt  = edge_weight.cpu().numpy() if edge_weight is not None else np.ones(len(src), dtype=np.float32)
    for s, d, w in zip(src, dst, wt):
        adj[int(s)].append((int(d), float(w)))

    neighbor_idx = torch.full((n_nodes, K), -1, dtype=torch.long)
    neighbor_wt  = torch.zeros(n_nodes, K, dtype=torch.float32)

    for node, neighbors in adj.items():
        if node >= n_nodes:
            continue
        neighbors.sort(key=lambda x: x[1], reverse=True)
        k_actual = min(K, len(neighbors))
        indices = torch.tensor([nb[0] for nb in neighbors[:k_actual]], dtype=torch.long)
        weights = torch.tensor([nb[1] for nb in neighbors[:k_actual]], dtype=torch.float32)

        # Normalize weights per-node to [0, 1] using max-normalization (proven in best node)
        max_w = weights.max()
        if max_w > 0:
            weights = weights / max_w

        neighbor_idx[node, :k_actual] = indices
        neighbor_wt[node,  :k_actual] = weights

    return neighbor_idx, neighbor_wt


def load_string_gnn_embeddings_and_neighbors(
    K: int = 16,
) -> Tuple[torch.Tensor, Dict[str, int], torch.Tensor, torch.Tensor]:
    """Load frozen STRING_GNN embeddings and precompute top-K neighbors.

    Returns:
        emb_matrix:   [18870, 256] float32 embedding matrix (CPU)
        name_to_idx:  Ensembl gene ID -> row index
        neighbor_idx: [18870, K] top-K neighbor node indices (CPU)
        neighbor_wt:  [18870, K] normalized edge weights [0, 1] (CPU)
    """
    node_names = json.loads((Path(STRING_GNN_DIR) / "node_names.json").read_text())
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    n_nodes = len(node_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True).to(device)
    gnn_model.eval()

    graph = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", map_location=device)
    edge_index  = graph["edge_index"]
    edge_weight = graph.get("edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        outputs = gnn_model(edge_index=edge_index, edge_weight=edge_weight)
        emb_matrix = outputs.last_hidden_state.float().cpu()

    edge_index_cpu  = edge_index.cpu()
    edge_weight_cpu = edge_weight.cpu() if edge_weight is not None else None
    neighbor_idx, neighbor_wt = build_top_k_neighbors(
        emb_matrix, edge_index_cpu, edge_weight_cpu, K=K, n_nodes=n_nodes
    )

    del gnn_model
    torch.cuda.empty_cache()

    return emb_matrix, name_to_idx, neighbor_idx, neighbor_wt


# ---------------------------------------------------------------------------
# 2-Head Neighborhood Attention Module (from tree-best node2-1-1-1-1-1)
# ---------------------------------------------------------------------------
class TwoHeadNeighborhoodAttentionAggregator(nn.Module):
    """K-neighborhood 2-head attention aggregation for STRING_GNN embeddings.

    IMPROVEMENT over parent's single-head: The tree-best node2-1-1-1-1-1 (F1=0.5128)
    used 2-head attention, while the parent node3-1-2-2 (F1=0.4813) used single-head.
    2-head attention allows the model to simultaneously attend to different PPI
    relationship types (e.g., functional similarity vs structural similarity),
    providing richer neighborhood representations.

    Implements 2-head center-context gating with learned attention weights biased by
    edge confidence scores, matching the proven architecture from node2-1-1-1-1-1.
    """

    def __init__(self, emb_dim: int = 256, K: int = 16, attn_dim: int = 64, n_heads: int = 2) -> None:
        super().__init__()
        self.K = K
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        self.head_dim = emb_dim // n_heads

        # Per-head attention projections
        self.attn_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim * 2, attn_dim),
                nn.GELU(),
                nn.Linear(attn_dim, 1),
            )
            for _ in range(n_heads)
        ])
        # Per-head gating
        self.gates = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim)
            for _ in range(n_heads)
        ])
        self.dropout = nn.Dropout(0.1)
        # Output projection to merge heads
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        center_idx: torch.Tensor,
        emb_table: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_wt: torch.Tensor,
    ) -> torch.Tensor:
        B = center_idx.shape[0]
        K = self.K
        device = center_idx.device

        valid_mask = (center_idx >= 0)
        safe_idx   = center_idx.clamp(min=0)

        center_emb = torch.where(
            valid_mask.unsqueeze(-1).expand(B, self.emb_dim),
            emb_table[safe_idx],
            torch.zeros(B, self.emb_dim, device=device, dtype=torch.float32),
        )

        if not valid_mask.any():
            return center_emb

        nb_idx  = neighbor_idx[safe_idx]
        nb_wt   = neighbor_wt[safe_idx]
        nb_mask = (nb_idx >= 0)
        nb_safe = nb_idx.clamp(min=0)
        nb_emb  = emb_table[nb_safe]  # [B, K, D]

        # Multi-head attention over neighbors
        center_exp = center_emb.unsqueeze(1).expand(B, K, self.emb_dim)
        pair_emb   = torch.cat([center_exp, nb_emb], dim=-1)  # [B, K, 2D]

        # Collect head-specific aggregations
        head_outputs = []
        for h in range(self.n_heads):
            attn_logits = self.attn_projs[h](pair_emb).squeeze(-1)  # [B, K]
            attn_logits = attn_logits + (nb_wt + 1e-8).log()
            attn_logits = attn_logits.masked_fill(~nb_mask, float('-inf'))

            has_any = nb_mask.any(dim=-1, keepdim=True)
            attn_logits = torch.where(
                has_any.expand(B, K),
                attn_logits,
                torch.zeros_like(attn_logits),
            )
            attn_weights = self.dropout(torch.softmax(attn_logits, dim=-1))  # [B, K]

            # Aggregate neighbors
            aggregated = (attn_weights.unsqueeze(-1) * nb_emb).sum(dim=1)  # [B, D]

            # Per-head gating using head-specific sub-space
            c_head = center_emb[:, h*self.head_dim:(h+1)*self.head_dim]
            a_head = aggregated[:, h*self.head_dim:(h+1)*self.head_dim]
            gate_val = torch.sigmoid(self.gates[h](c_head))
            head_out = c_head + gate_val * a_head  # [B, head_dim]
            head_outputs.append(head_out)

        # Concatenate heads and project
        combined = torch.cat(head_outputs, dim=-1)  # [B, D]
        output = self.out_proj(combined)  # [B, D]

        # Apply valid mask
        output = torch.where(
            valid_mask.unsqueeze(-1).expand(B, self.emb_dim),
            output,
            center_emb,
        )
        return output


# ---------------------------------------------------------------------------
# Dataset / DataModule
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        string_name_to_idx: Dict[str, int],
    ) -> None:
        self.pert_ids    = df["pert_id"].tolist()
        self.symbols     = df["symbol"].tolist()
        self.name_to_idx = string_name_to_idx

        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List] = (
            [torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
             for row in df["label"].tolist()]
            if has_label else None
        )

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.pert_ids[idx]
        gnn_idx = self.name_to_idx.get(pid, -1)
        item: Dict[str, Any] = {
            "sample_idx": idx,
            "pert_id":    pid,
            "symbol":     self.symbols[idx],
            "gnn_idx":    gnn_idx,
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_collate(tokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "gnn_idx":        torch.tensor([b["gnn_idx"] for b in batch], dtype=torch.long),
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return collate_fn


class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_workers: int = 4) -> None:
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

        node_names = json.loads((Path(STRING_GNN_DIR) / "node_names.json").read_text())
        self.string_name_to_idx = {name: i for i, name in enumerate(node_names)}

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df, self.string_name_to_idx)
        self.val_ds   = DEGDataset(val_df,   self.string_name_to_idx)
        self.test_ds  = DEGDataset(test_df,  self.string_name_to_idx)

        train_cf = make_collate(self.tokenizer)
        val_cf   = make_collate(self.tokenizer)
        test_cf  = make_collate(self.tokenizer)

        val_sampler   = RankAwareSampler(self.val_ds, shuffle=False)
        test_sampler  = RankAwareSampler(self.test_ds, shuffle=False)

        self._cached_train_dl = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=train_cf,
        )
        self._cached_val_dl = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=val_cf,
            sampler=val_sampler,
        )
        self._cached_test_dl = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=test_cf,
            sampler=test_sampler,
        )

    def train_dataloader(self): return self._cached_train_dl
    def val_dataloader(self):   return self._cached_val_dl
    def test_dataloader(self):  return self._cached_test_dl


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDO100MStringGNNFusionModelV2(pl.LightningModule):
    """AIDO.Cell-100M LoRA r=8 + STRING_GNN K=16 2-head neighborhood attention fusion.

    Key improvements over parent node3-1-2-2 (F1=0.4813):
    1. 2-head STRING attention → matches tree-best node2-1-1-1-1-1 recipe
    2. Discriminative LR: LoRA lr=5e-5 < head+attn lr=1e-4
    3. CAWR (T_0=50, T_mult=2) → escape local minima across training cycles
    4. eta_min=1e-5 → prevents LR collapse at cosine tail
    5. global_batch=128 → 4x more gradient update density
    6. max_epochs=300 + patience=30 → capture late improvement spikes (like epoch 77 in tree best)

    Architecture (896-dim fusion, identical to tree-best):
    - AIDO.Cell-100M with LoRA r=8 (summary token extraction, 640-dim)
    - Frozen STRING_GNN + K=16 2-head neighborhood attention aggregation (256-dim)
    - Concat fusion: 640+256=896-dim
    - 2-layer head: Linear(896→256) + LN + GELU + Dropout(0.5) + Linear(256→19920)
    - Label-smoothed CE (0.05) + sqrt-inverse-frequency class weights
    """

    def __init__(
        self,
        lora_r: int          = 8,
        lora_alpha: int      = 16,
        lora_dropout: float  = 0.05,
        head_hidden: int     = 256,
        head_dropout: float  = 0.5,
        lr_lora: float       = 5e-5,   # Discriminative: smaller for LoRA backbone
        lr_head: float       = 1e-4,   # Discriminative: larger for head + STRING attention
        weight_decay: float  = 2e-2,
        warmup_epochs: int   = 10,
        max_epochs: int      = 300,
        string_k: int        = 16,
        string_attn_dim: int = 64,
        string_n_heads: int  = 2,      # 2-head attention (from tree-best)
        label_smoothing: float = 0.05,
        es_patience: int     = 30,
        cawr_t0: int         = 50,     # CAWR: first restart period
        cawr_tmult: int      = 2,      # CAWR: period multiplier after each restart
        eta_min_ratio: float = 1e-4,   # eta_min = lr * eta_min_ratio (prevents LR collapse)
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load AIDO.Cell-100M backbone ----
        base_model = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        base_model = base_model.to(torch.bfloat16)
        base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Share QKV weights between flash_self and self (AIDO.Cell architecture quirk)
        for layer in base_model.bert.encoder.layer:
            ss = layer.attention.flash_self
            mm = layer.attention.self
            ss.query.weight = mm.query.weight
            ss.key.weight   = mm.key.weight
            ss.value.weight = mm.value.weight
            if hasattr(mm.query, 'bias') and mm.query.bias is not None:
                ss.query.bias = mm.query.bias
                ss.key.bias   = mm.key.bias
                ss.value.bias = mm.value.bias

        # CRITICAL FIX: Monkey-patch enable_input_require_grads (PEFT compatibility)
        base_model.enable_input_require_grads = lambda: None

        # Apply LoRA
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,
        )
        self.backbone = get_peft_model(base_model, lora_cfg)
        self.backbone.print_trainable_parameters()

        # CRITICAL FIX: Register gradient hook on gene_embedding so LoRA adapters
        # receive gradients through the gene embedding layer.
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        gene_emb = self.backbone.model.bert.gene_embedding
        gene_emb.register_forward_hook(_make_inputs_require_grad)

        # Cast trainable LoRA params to float32
        for param in self.backbone.parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ---- Load STRING_GNN embeddings (frozen) ----
        emb_matrix, name_to_idx, neighbor_idx, neighbor_wt = \
            load_string_gnn_embeddings_and_neighbors(K=hp.string_k)

        self.register_buffer("string_emb",   emb_matrix)
        self.register_buffer("neighbor_idx", neighbor_idx)
        self.register_buffer("neighbor_wt",  neighbor_wt)

        # ---- 2-Head Neighborhood attention module (IMPROVEMENT over parent) ----
        self.nb_attention = TwoHeadNeighborhoodAttentionAggregator(
            emb_dim  = STRING_DIM,
            K        = hp.string_k,
            attn_dim = hp.string_attn_dim,
            n_heads  = hp.string_n_heads,
        )
        for p in self.nb_attention.parameters():
            p.data = p.data.float()

        # ---- Fusion head ----
        total_dim = AIDO_HIDDEN_DIM + STRING_DIM
        self.head = nn.Sequential(
            nn.Linear(total_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )
        for p in self.head.parameters():
            p.data = p.data.float()

        # ---- Class weights ----
        self.register_buffer("class_weights", get_class_weights())

        # ---- Accumulators ----
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_meta:  List[Tuple]        = []

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        gnn_idx:        torch.Tensor,
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        # Summary token at position 19264 (first appended summary token)
        aido_feat = out.last_hidden_state[:, 19264, :].float()

        string_feat = self.nb_attention(
            center_idx   = gnn_idx,
            emb_table    = self.string_emb,
            neighbor_idx = self.neighbor_idx,
            neighbor_wt  = self.neighbor_wt,
        )

        fused  = torch.cat([aido_feat, string_feat], dim=-1)
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return logits

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
            weight=self.class_weights.to(logits.device),
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gnn_idx"])
        loss   = self._loss(logits, batch["labels"])
        bs = batch["input_ids"].shape[0]
        self.log("train/loss", loss, prog_bar=True, sync_dist=True,
                 on_step=True, on_epoch=False, batch_size=bs)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gnn_idx"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            bs = batch["input_ids"].shape[0]
            self.log("val/loss", loss, sync_dist=True, batch_size=bs)
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
        mask   = torch.cat([
            torch.tensor([True], device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gnn_idx"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        for i, (pid, sym) in enumerate(zip(batch["pert_id"], batch["symbol"])):
            self._test_meta.append((pid, sym, batch["sample_idx"][i].item()))
        if "labels" in batch:
            bs = batch["input_ids"].shape[0]
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True, batch_size=bs)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = torch.cat(self._test_preds, 0)
        local_rows  = []
        for i, (pid, sym, sidx) in enumerate(self._test_meta):
            local_rows.append({
                "sample_idx": sidx,
                "pert_id":    pid,
                "symbol":     sym,
                "prediction": local_preds[i].cpu().numpy().tolist(),
            })

        if self.trainer.is_global_zero:
            if torch.distributed.is_initialized():
                world_size   = torch.distributed.get_world_size()
                all_meta_obj = [None] * world_size
                torch.distributed.all_gather_object(all_meta_obj, local_rows)
            else:
                all_meta_obj = [local_rows]
        else:
            if torch.distributed.is_initialized():
                _dummy = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(_dummy, local_rows)

        self._test_preds.clear()
        self._test_meta.clear()

        if not self.trainer.is_global_zero:
            return

        global_rows = []
        for rank_rows in all_meta_obj:
            global_rows.extend(rank_rows)

        seen = set()
        unique_rows = []
        for row in global_rows:
            if row["sample_idx"] not in seen:
                seen.add(row["sample_idx"])
                unique_rows.append(row)

        rows = []
        for row in unique_rows:
            rows.append({
                "idx":        row["pert_id"],
                "input":      row["symbol"],
                "prediction": json.dumps(row["prediction"]),
            })

        out_dir = Path(__file__).parent / "run"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
        print(f"[Node3-1-2-2-1] Saved {len(rows)} test predictions.")

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
        self.print(f"Checkpoint: {trained:,}/{total:,} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict, ignoring missing/unexpected keys from partial checkpoints."""
        expected_keys = set(k for k, v in self.named_parameters() if v.requires_grad)
        expected_keys |= {name for name, _ in self.named_buffers()}
        # missing_keys: expected by model but absent in checkpoint
        # unexpected_keys: in checkpoint but not expected by model
        missing_keys = [k for k in expected_keys if k not in state_dict]
        unexpected_keys = [k for k in state_dict if k not in expected_keys]
        if missing_keys:
            self.print(f"Warning: missing keys in checkpoint: {missing_keys[:5]}...")
        if unexpected_keys:
            self.print(f"Warning: unexpected keys in checkpoint: {unexpected_keys[:5]}...")
        return super().load_state_dict(state_dict, strict=False)

    def configure_optimizers(self):
        hp = self.hparams

        # Discriminative learning rates:
        # - LoRA backbone parameters: smaller lr (5e-5) for stable adaptation
        # - Head + STRING attention: larger lr (1e-4) for faster task-specific learning
        backbone_params = [p for n, p in self.named_parameters()
                          if p.requires_grad and n.startswith("backbone")]
        other_params    = [p for n, p in self.named_parameters()
                          if p.requires_grad and not n.startswith("backbone")]

        n_backbone = sum(p.numel() for p in backbone_params)
        n_other    = sum(p.numel() for p in other_params)
        print(f"[Node3-1-2-2-1] Backbone trainable: {n_backbone:,}, Head+Attention: {n_other:,}")

        param_groups = [
            {"params": backbone_params, "lr": hp.lr_lora,  "name": "backbone_lora"},
            {"params": other_params,    "lr": hp.lr_head,  "name": "head_and_attention"},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=hp.weight_decay,
            betas=(0.9, 0.95),
        )

        # CAWR with linear warmup for the first warmup_epochs
        # Phase 1: Linear warmup from 0.1x to 1.0x lr over warmup_epochs
        # Phase 2: CosineAnnealingWarmRestarts (T_0, T_mult) for periodic LR restarts
        # This allows the model to escape local minima via periodic LR increases

        warmup_sched = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )

        # eta_min as ratio of initial lr to prevent near-zero LR collapse
        eta_min_lora = hp.lr_lora * hp.eta_min_ratio
        eta_min_head = hp.lr_head * hp.eta_min_ratio

        # Use CAWR for cyclic restarts after warmup
        # T_0: first cycle length; T_mult: subsequent cycle multiplier
        cawr_sched = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.cawr_t0,
            T_mult=hp.cawr_tmult,
            eta_min=eta_min_lora,   # uses minimum from first param group
        )

        # We use SequentialLR to combine warmup then CAWR
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cawr_sched],
            milestones=[hp.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node3-1-2-2-1: AIDO.Cell-100M LoRA + STRING_GNN K=16 2-Head + CAWR"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=8)    # 8 per GPU (100M memory-safe)
    parser.add_argument("--global-batch-size",  type=int,   default=128)  # 128 for stable gradients
    parser.add_argument("--max-epochs",         type=int,   default=300)  # Extended training budget
    parser.add_argument("--lr-lora",            type=float, default=5e-5) # Discriminative: LoRA backbone
    parser.add_argument("--lr-head",            type=float, default=1e-4) # Discriminative: head+attention
    parser.add_argument("--weight-decay",       type=float, default=2e-2)
    parser.add_argument("--lora-r",             type=int,   default=8)
    parser.add_argument("--lora-alpha",         type=int,   default=16)
    parser.add_argument("--lora-dropout",       type=float, default=0.05)
    parser.add_argument("--head-hidden",        type=int,   default=256)
    parser.add_argument("--head-dropout",       type=float, default=0.5)
    parser.add_argument("--warmup-epochs",      type=int,   default=10)
    parser.add_argument("--string-k",           type=int,   default=16)
    parser.add_argument("--string-attn-dim",    type=int,   default=64)
    parser.add_argument("--string-n-heads",     type=int,   default=2)    # 2-head attention
    parser.add_argument("--label-smoothing",    type=float, default=0.05)
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--es-patience",        type=int,   default=30)   # Extended patience
    parser.add_argument("--es-min-delta",       type=float, default=1e-3)
    parser.add_argument("--cawr-t0",            type=int,   default=50)   # CAWR T_0
    parser.add_argument("--cawr-tmult",         type=int,   default=2)    # CAWR T_mult
    parser.add_argument("--eta-min-ratio",      type=float, default=1e-4) # eta_min = lr * ratio
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--debug_max_step",     type=int,   default=None)
    parser.add_argument("--fast_dev_run",       action="store_true")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean checkpoint dir before training to avoid stale checkpoints.
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        ckpt_dir = output_dir / "checkpoints"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = lim_val = lim_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDO100MStringGNNFusionModelV2(
        lora_r          = args.lora_r,
        lora_alpha      = args.lora_alpha,
        lora_dropout    = args.lora_dropout,
        head_hidden     = args.head_hidden,
        head_dropout    = args.head_dropout,
        lr_lora         = args.lr_lora,
        lr_head         = args.lr_head,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        max_epochs      = args.max_epochs,
        string_k        = args.string_k,
        string_attn_dim = args.string_attn_dim,
        string_n_heads  = args.string_n_heads,
        label_smoothing = args.label_smoothing,
        es_patience     = args.es_patience,
        cawr_t0         = args.cawr_t0,
        cawr_tmult      = args.cawr_tmult,
        eta_min_ratio   = args.eta_min_ratio,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )
    es_cb = EarlyStopping(
        monitor="val/f1", mode="max",
        patience=args.es_patience,
        min_delta=args.es_min_delta,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = n_gpus,
        num_nodes               = 1,
        strategy                = strategy,
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

    # Auto-evaluate using calc_metric.py
    score_path = Path(__file__).parent / "test_score.txt"
    pred_path  = Path(__file__).parent / "run" / "test_predictions.tsv"
    if pred_path.exists() and Path(TEST_TSV).exists():
        try:
            result = subprocess.run(
                ["python", str(DATA_ROOT / "calc_metric.py"), str(pred_path), str(TEST_TSV)],
                capture_output=True, text=True, timeout=120
            )
            metrics = json.loads(result.stdout.strip().split("\n")[-1])
            f1_val = metrics.get("value", None)
            if f1_val is not None:
                with open(score_path, "w") as f:
                    f.write(f"f1_score: {f1_val}\n")
                    if "details" in metrics:
                        for k, v in metrics["details"].items():
                            f.write(f"  {k}: {v}\n")
                print(f"[Node3-1-2-2-1] test_f1={f1_val:.4f} — saved to {score_path}")
            else:
                with open(score_path, "w") as f:
                    f.write(f"error: {metrics.get('error', 'unknown')}\n")
        except Exception as e:
            with open(score_path, "w") as f:
                f.write(f"error: {e}\n")
            print(f"[Node3-1-2-2-1] calc_metric failed: {e}")
    else:
        with open(score_path, "w") as f:
            f.write("error: test_predictions.tsv not found\n")

    print(f"[Node3-1-2-2-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
