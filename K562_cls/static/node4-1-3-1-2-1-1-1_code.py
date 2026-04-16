"""Node4-1-3-1-2-1-1-1: AIDO.Cell-100M (LoRA r=16, ALL 18 layers) + STRING_GNN K=16 2-head
+ Summary Token ONLY + Narrow Head (896->256->19920) + Full Cosine Decay (patience=200, max_epochs=400).

## Summary of Design

This node makes TWO targeted improvements to the parent (node4-1-3-1-2-1-1, test F1=0.5128),
guided by the parent's feedback analysis and the tree-wide architectural ceiling evidence.

### Change 1: Patience 15 → 200 + max_epochs 300 → 400 (ALLOW FULL COSINE DECAY):
   The parent stopped at epoch 92 with LR ~8.2e-5 — only 28% into the cosine schedule
   (T_max=290). The entire second 72% of the cosine decay (epochs 93-300, LR 8e-5 → 1e-6)
   has NEVER been explored by any node in this lineage. The feedback explicitly marks this
   as "high priority": "Allowing training to continue to epochs 150-200 (LR ~5-6e-5) or
   even 250-280 (LR ~2-4e-5) may reveal whether the plateau can be escaped at lower LR."

   With patience=200 and min_delta=0.0, early stopping is effectively disabled (improvements
   every 5-10 epochs during plateau reset the counter), and training runs to max_epochs=400,
   completing the full cosine schedule and a brief eta_min stabilization phase.

### Change 2: LoRA r=8 → r=16, alpha=16 → 32, lora_dropout=0.05 → 0.10, wd=2e-2 → 3e-2:
   Three independent runs (node2-1-1-1-1-1, node4-1-3-1-2-1, node4-1-3-1-2-1-1) reached
   F1=0.5128 with r=8, confirming a TRUE ARCHITECTURAL CEILING. Increasing LoRA rank to 16
   (with proportional regularization) is explicitly recommended in the feedback as
   "medium priority — untested at top performance tier with weighted CE." The previous r=16
   failure (node1-1-1-3-1-2, F1=0.4542) was caused by focal loss, NOT r=16.

### Critical guardrails preserved vs. failed sibling (node4-1-3-1-1, F1=0.4705):
   - eta_min=1e-6 (NOT 1e-5): prevents late-epoch drift with higher LoRA capacity
   - Focal loss AVOIDED: proven catastrophic (-0.05-0.06 F1) in all AIDO.Cell nodes
   - Summary token ONLY: zero val-test generalization gap confirmed in all parent nodes

### What is preserved (proven correct from parent):
   - Summary token ONLY at position 19264 (zero val-test generalization gap confirmed)
   - Narrow head 896->256->19920 (~5.3M params, matches tree best)
   - K=16 2-head STRING_GNN neighborhood attention (frozen)
   - eta_min=1e-6, lr=1e-4
   - label_smoothing=0.05, sqrt-inv-freq class weights
   - LoRA on ALL 18 layers (all-18 broke the 0.50 ceiling)
   - min_delta=0.0 (any improvement resets counter, critical for oscillatory plateau)
   - gradient checkpointing

## Architecture

  AIDO.Cell-100M LoRA(r=16, ALL 18 layers):
    → summary_emb = lhs[:, 19264, :]     [B, 640]   (global context token ONLY)

  STRING_GNN (frozen):
    → K=16 PPI neighbors via 2-head multi-head attention
    → string_emb                          [B, 256]

  Fusion: cat([summary_emb, string_emb])   [B, 896]

  Head (2-layer):
    Linear(896->256) -> LayerNorm -> GELU -> Dropout(0.5)
    Linear(256->19920) -> view([B, 3, 6640])

## Node Comparison

| Node                    | Test F1 | LoRA r | patience | max_epochs |
|-------------------------|---------|--------|----------|------------|
| node4-1-3-1-2-1-1       | 0.5128  | 8      | 15       | 300        |
| node4-1-3-1-2-1         | 0.5128  | 8      | 10       | 300        |
| node2-1-1-1-1-1         | 0.5128  | 8      | 10       | 300        |
| node4-1-3-1-2-1-1-1     | target  | 16     | 200      | 400        |
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES          = 6640
N_CLASSES        = 3
AIDO_GENES       = 19264        # AIDO.Cell gene vocabulary size
AIDO_MODEL_DIR   = "/home/Models/AIDO.Cell-100M"
STRING_MODEL_DIR = "/home/Models/STRING_GNN"
AIDO_HIDDEN      = 640          # AIDO.Cell-100M hidden size
STRING_HIDDEN    = 256          # STRING_GNN hidden size
NEIGHBOR_K       = 16           # Proven optimal K for PPI neighborhood
ATTN_DIM         = 64           # Attention projection dimension per head
N_ATTN_HEADS     = 2            # 2-head attention (proven in node2-1-1-1-1-1)

# Class frequency: [down(-1->0), neutral(0->1), up(+1->2)]
CLASS_FREQ      = [0.0429, 0.9251, 0.0320]
LABEL_SMOOTHING = 0.05

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic."""
    y_hat = preds.argmax(dim=1)   # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0).float()
        tp = (is_pred & is_true).float().sum(0)
        fp = (is_pred & ~is_true).float().sum(0)
        fn = (~is_pred & is_true).float().sum(0)
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(
            prec + rec > 0,
            2 * prec * rec / (prec + rec + 1e-8),
            torch.zeros_like(prec),
        )
        f1_per_gene += f1_c * present
        n_present   += present

    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        has_label = "label" in df.columns and df["label"].notna().all()
        if has_label:
            self.labels: Optional[List[torch.Tensor]] = [
                torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
                for row in df["label"].tolist()
            ]
        else:
            self.labels = None

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


class RankAwareSampler(torch.utils.data.Sampler):
    """Deterministically assigns disjoint subsets of the dataset to each rank.

    Eliminates DDP sample collision issues in val/test phases by assigning each
    rank a non-overlapping subset.
    """

    def __init__(self, dataset, shuffle: bool = False, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = max(int(os.environ.get("WORLD_SIZE", 1)), 1)

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            n = len(self.dataset)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        rank_indices = [indices[i] for i in range(self.rank, len(indices), self.world_size)]
        return iter(rank_indices)

    def __len__(self):
        n = len(self.dataset)
        return (n + self.world_size - 1) // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch


def make_collate(tokenizer):
    """Factory for collate_fn with AIDO.Cell tokenizer.

    Note: We extract summary token only — no gene_positions needed.
    """
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]

        # Tokenize: each sample gets only its perturbed gene with expression=1.0
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")  # input_ids: [B, 19264] float32

        dataset_indices = torch.tensor(
            [b["sample_idx"] for b in batch], dtype=torch.long
        )

        out: Dict[str, Any] = {
            "sample_idx":         dataset_indices,
            "local_dataset_idx":  dataset_indices,
            "pert_id":            pert_ids,
            "symbol":             symbols,
            "input_ids":          tokenized["input_ids"],
            "attention_mask":     tokenized["attention_mask"],
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out

    return collate_fn


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 4, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer: Optional[Any] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Rank-0 downloads first, then all ranks load
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

        collate_fn = make_collate(self.tokenizer)
        val_sampler  = RankAwareSampler(self.val_ds,  shuffle=False)
        test_sampler = RankAwareSampler(self.test_ds, shuffle=False)

        self._cached_train_dl = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn,
        )
        self._cached_val_dl = DataLoader(
            self.val_ds, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=collate_fn, sampler=val_sampler,
        )
        self._cached_test_dl = DataLoader(
            self.test_ds, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=collate_fn, sampler=test_sampler,
        )

    def train_dataloader(self) -> DataLoader:
        return self._cached_train_dl

    def val_dataloader(self) -> DataLoader:
        return self._cached_val_dl

    def test_dataloader(self) -> DataLoader:
        return self._cached_test_dl


# ---------------------------------------------------------------------------
# Multi-Head Neighborhood Attention Module (K=16, 2 heads)
# ---------------------------------------------------------------------------
class MultiHeadNeighborhoodAttentionModule(nn.Module):
    """Multi-head K-hop neighborhood attention for PPI graph context.

    Proven optimal in node2-1-1-1-1-1 (F1=0.5128, K=16, 2-head).
    Takes frozen STRING_GNN embeddings and computes context-aware
    representation using top-16 PPI neighbors.

    Architecture per head h:
        q_h = W_q_h(center_emb)              [B, attn_dim]
        k_h = W_k_h(neigh_embs)              [B, K, attn_dim]
        attn_h = softmax(q_h @ k_h.T / sqrt(attn_dim) + log(edge_conf))  [B, K]
        context_h = attn_h @ neigh_embs      [B, 256]

    Final output:
        multi_context = concat(context_h for each head)   [B, n_heads*256]
        projected = W_out(multi_context)                  [B, 256]
        gate = sigmoid(W_gate([center, projected]))
        output = gate * center + (1-gate) * projected     [B, 256]
    """

    def __init__(
        self,
        emb_dim: int   = 256,
        attn_dim: int  = 64,
        n_heads: int   = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb_dim  = emb_dim
        self.attn_dim = attn_dim
        self.n_heads  = n_heads

        self.W_q    = nn.ModuleList([nn.Linear(emb_dim, attn_dim, bias=False) for _ in range(n_heads)])
        self.W_k    = nn.ModuleList([nn.Linear(emb_dim, attn_dim, bias=False) for _ in range(n_heads)])
        self.W_out  = nn.Linear(emb_dim * n_heads, emb_dim, bias=True)
        self.W_gate = nn.Linear(emb_dim * 2, emb_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        center_emb: torch.Tensor,   # [B, 256]
        neigh_embs: torch.Tensor,   # [B, K, 256]
        neigh_conf: torch.Tensor,   # [B, K]  (normalized STRING edge confidence)
        valid_mask: torch.Tensor,   # [B] bool  (True = gene in STRING vocabulary)
    ) -> torch.Tensor:              # [B, 256]
        B, K, D = neigh_embs.shape

        log_conf = (neigh_conf + 1e-8).log()  # [B, K]

        head_contexts = []
        for h in range(self.n_heads):
            q_h = self.W_q[h](center_emb)                                                  # [B, attn_dim]
            k_h = self.W_k[h](neigh_embs.reshape(-1, D)).reshape(B, K, self.attn_dim)     # [B, K, attn_dim]

            attn_h = (q_h.unsqueeze(1) @ k_h.transpose(1, 2)) / (self.attn_dim ** 0.5)   # [B, 1, K]
            attn_h = attn_h.squeeze(1)                                                     # [B, K]
            attn_h = attn_h + log_conf
            attn_h = torch.softmax(attn_h, dim=-1)
            attn_h = self.dropout(attn_h)

            context_h = (attn_h.unsqueeze(1) @ neigh_embs).squeeze(1)   # [B, 256]
            head_contexts.append(context_h)

        multi_context = torch.cat(head_contexts, dim=-1)   # [B, n_heads*256]
        projected = self.W_out(multi_context)               # [B, 256]
        gate = torch.sigmoid(self.W_gate(torch.cat([center_emb, projected], dim=-1)))
        output = gate * center_emb + (1 - gate) * projected

        # For genes not in STRING vocabulary, return center_emb unchanged
        output = torch.where(valid_mask.unsqueeze(-1), output, center_emb)
        return output


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCellStringAllLayersModelV3(pl.LightningModule):
    """AIDO.Cell-100M (LoRA r=16, ALL 18 layers) + STRING_GNN K=16 2-head +
    Summary Token ONLY + Narrow Head (896->256->19920) + Full Cosine Decay.

    Key changes vs parent (node4-1-3-1-2-1-1, test F1=0.5128):
    1. LoRA r: 8 → 16, alpha: 16 → 32 — increased backbone expressivity
       Three independent r=8 runs converged to the same F1=0.5128 ceiling.
       r=16 + weighted CE is untested at the top tier (previous r=16 failure
       was with focal loss, which is proven catastrophic for this task).
    2. lora_dropout: 0.05 → 0.10 — stronger adapter regularization for r=16
    3. weight_decay: 2e-2 → 3e-2 — stronger parameter regularization for r=16
    4. patience: 15 → 200 — allow full cosine decay (only 28% explored so far)
    5. max_epochs: 300 → 400 — buffer for eta_min stabilization

    Critical guardrail vs. failed sibling (node4-1-3-1-1, F1=0.4705):
    - eta_min=1e-6 (NOT 1e-5): prevents late-epoch drift with higher LoRA capacity
    - Summary token ONLY: preserves zero val-test generalization gap
    """

    def __init__(
        self,
        lora_r: int             = 16,    # KEY CHANGE: 8 → 16 (increased backbone capacity)
        lora_alpha: int         = 32,    # KEY CHANGE: 16 → 32 (standard 2× rule)
        lora_dropout: float     = 0.10,  # KEY CHANGE: 0.05 → 0.10 (stronger for r=16)
        lora_last_n_layers: int = 18,    # ALL 18 layers (proven in parent lineage)
        head_hidden: int        = 256,   # 896->256->19920 (narrow, matching tree best)
        head_dropout: float     = 0.5,
        lr: float               = 1e-4,
        weight_decay: float     = 3e-2,  # KEY CHANGE: 2e-2 → 3e-2 (stronger for r=16)
        warmup_epochs: int      = 10,
        max_epochs: int         = 400,   # KEY CHANGE: 300 → 400 (full cosine + stabilization)
        eta_min: float          = 1e-6,  # Critical: NOT 1e-5 (sibling's mistake)
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load STRING_GNN backbone (frozen) and pre-compute embeddings ----
        string_model = AutoModel.from_pretrained(STRING_MODEL_DIR, trust_remote_code=True)
        string_model.eval()

        graph_data   = torch.load(Path(STRING_MODEL_DIR) / "graph_data.pt", map_location="cpu")
        node_names   = json.loads((Path(STRING_MODEL_DIR) / "node_names.json").read_text())
        self._string_node_names = node_names

        edge_index  = graph_data["edge_index"]
        edge_weight = graph_data.get("edge_weight", None)

        # Pre-compute frozen STRING_GNN embeddings once (no grad needed)
        with torch.no_grad():
            out = string_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
                output_hidden_states=False,
            )
            string_embs = out.last_hidden_state.float()  # [18870, 256]

        # Build lookup: Ensembl ID -> row index
        string_id_to_idx = {eid: i for i, eid in enumerate(node_names)}
        self._string_id_to_idx = string_id_to_idx
        n_nodes = string_embs.shape[0]

        # Register as non-trainable buffer
        self.register_buffer("string_embs", string_embs)  # [18870, 256]

        # ---- Build K=16 neighborhood lookup ----
        K = NEIGHBOR_K
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1])

        edge_index_cpu  = edge_index
        edge_weight_cpu = edge_weight.float()

        adj = [[] for _ in range(n_nodes)]
        for i in range(edge_index_cpu.shape[1]):
            src = edge_index_cpu[0, i].item()
            dst = edge_index_cpu[1, i].item()
            w   = edge_weight_cpu[i].item()
            adj[src].append((dst, w))

        topk_neighbors = torch.zeros(n_nodes, K, dtype=torch.long)
        topk_weights   = torch.zeros(n_nodes, K, dtype=torch.float32)

        for i, neighbors in enumerate(adj):
            if len(neighbors) == 0:
                topk_neighbors[i] = i
                topk_weights[i]   = 1.0
            else:
                neighbors_sorted = sorted(neighbors, key=lambda x: x[1], reverse=True)[:K]
                k_actual = len(neighbors_sorted)
                for j, (nid, nw) in enumerate(neighbors_sorted):
                    topk_neighbors[i, j] = nid
                    topk_weights[i, j]   = nw
                for j in range(k_actual, K):
                    topk_neighbors[i, j] = i
                    topk_weights[i, j]   = 0.0
                max_w = topk_weights[i].max()
                if max_w > 0:
                    topk_weights[i] = topk_weights[i] / max_w

        self.register_buffer("topk_neighbors", topk_neighbors)  # [18870, K]
        self.register_buffer("topk_weights",   topk_weights)    # [18870, K]

        # ---- Multi-head neighborhood attention (K=16, 2 heads) ----
        self.neighborhood_attn = MultiHeadNeighborhoodAttentionModule(
            emb_dim=STRING_HIDDEN,
            attn_dim=ATTN_DIM,
            n_heads=N_ATTN_HEADS,
            dropout=0.1,
        )

        # Learnable fallback embedding for genes not in STRING vocabulary
        self.register_parameter(
            "fallback_emb",
            nn.Parameter(torch.zeros(1, STRING_HIDDEN))
        )
        nn.init.normal_(self.fallback_emb, std=0.01)

        # ---- Load AIDO.Cell backbone ----
        backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # AIDO.Cell has a custom GeneEmbedding (no standard word_embeddings).
        # PEFT's get_peft_model calls enable_input_require_grads(), which internally
        # calls get_input_embeddings() -> raises NotImplementedError.
        # Monkey-patch to a no-op; we register a forward hook below instead.
        backbone.enable_input_require_grads = lambda: None

        # ---- Apply LoRA to AIDO.Cell ALL transformer layers ----
        # ALL 18 layers with r=16 (increased from r=8 in parent):
        # ~1.1M trainable backbone params (~1.1% of 99.6M total)
        # r=16 + weighted CE is untested at top performance tier — r=8 ceiling confirmed.
        total_layers = 18  # AIDO.Cell-100M has 18 transformer layers
        lora_layers = list(range(total_layers - hp.lora_last_n_layers, total_layers))
        self.print(f"[node4-1-3-1-2-1-1-1] Applying LoRA r={hp.lora_r} to layers: {lora_layers}")

        lora_cfg = LoraConfig(
            task_type            = TaskType.FEATURE_EXTRACTION,
            r                    = hp.lora_r,
            lora_alpha           = hp.lora_alpha,
            lora_dropout         = hp.lora_dropout,
            target_modules       = ["query", "key", "value"],
            layers_to_transform  = lora_layers,
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Forward hook: ensure gradients flow through gene_embedding to LoRA adapters
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        gene_emb = self.backbone.model.bert.gene_embedding
        gene_emb.register_forward_hook(_make_inputs_require_grad)

        # Cast LoRA params to float32 for numerically stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ---- Classification head ----
        # Input: [summary_emb(640) | string_emb(256)] = 896-dim (summary token ONLY)
        # Matching tree best (node2-1-1-1-1-1, F1=0.5128): 896->256->19920
        in_dim = AIDO_HIDDEN + STRING_HIDDEN  # 640 + 256 = 896
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),              # 896 -> 256
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),                    # 0.5 (proven optimal)
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES), # 256 -> 19920
        )

        head_params = sum(p.numel() for p in self.head.parameters())
        self.print(f"[node4-1-3-1-2-1-1-1] Head: {head_params:,} params (896->{hp.head_hidden}->19920)")

        # ---- Loss weights ----
        self.register_buffer("class_weights", get_class_weights())

        # ---- Accumulators ----
        self._val_preds:      List[torch.Tensor] = []
        self._val_tgts:       List[torch.Tensor] = []
        self._val_idx:        List[torch.Tensor] = []
        self._test_preds:     List[torch.Tensor] = []
        self._test_idx:       List[torch.Tensor] = []
        self._test_local_idx: List[torch.Tensor] = []

    def _get_string_neighbor_embs(self, pert_ids: List[str]):
        """Retrieve STRING_GNN neighborhood embeddings for a batch.

        Returns:
            center_emb [B, 256]: single-node embedding (or fallback)
            neigh_embs [B, K, 256]: top-K neighbor embeddings
            neigh_conf [B, K]: normalized PPI edge confidence weights
            valid_mask [B]: True if gene in STRING vocabulary
        """
        B = len(pert_ids)
        K = NEIGHBOR_K
        device = self.string_embs.device

        indices = [self._string_id_to_idx.get(pid, -1) for pid in pert_ids]
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        valid_mask  = (idx_tensor >= 0)
        idx_clamped = idx_tensor.clamp(min=0)

        center_emb = torch.zeros(B, STRING_HIDDEN, dtype=torch.float32, device=device)
        if valid_mask.any():
            center_emb[valid_mask] = self.string_embs[idx_clamped[valid_mask]]
        if (~valid_mask).any():
            center_emb[~valid_mask] = self.fallback_emb.expand((~valid_mask).sum(), -1).float()

        neigh_idx  = torch.zeros(B, K, dtype=torch.long, device=device)
        neigh_conf = torch.ones(B, K, dtype=torch.float32, device=device) / K

        if valid_mask.any():
            valid_src = idx_clamped[valid_mask]
            neigh_idx[valid_mask]  = self.topk_neighbors[valid_src]
            neigh_conf[valid_mask] = self.topk_weights[valid_src]

        flat_neigh_idx = neigh_idx.reshape(-1)
        flat_neigh_emb = self.string_embs[flat_neigh_idx]
        neigh_embs = flat_neigh_emb.reshape(B, K, STRING_HIDDEN)

        return center_emb, neigh_embs, neigh_conf, valid_mask

    # ---- Forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids: List[str],
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        # 1. AIDO.Cell forward pass (with LoRA r=16 on ALL 18 layers)
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        # last_hidden_state: [B, AIDO_GENES + 2, 640] = [B, 19266, 640]
        lhs = out.last_hidden_state

        # Summary token (position 19264): aggregates global transcriptome context
        # This is the ONLY AIDO.Cell embedding used — no gene-position token.
        # Proven: zero val-test generalization gap in all parent nodes.
        summary_emb = lhs[:, AIDO_GENES, :].float()           # [B, 640]

        # 2. STRING_GNN neighborhood embeddings (K=16, frozen)
        center_emb, neigh_embs, neigh_conf, valid_mask = self._get_string_neighbor_embs(pert_ids)

        # 3. Apply 2-head neighborhood attention
        string_emb = self.neighborhood_attn(
            center_emb, neigh_embs, neigh_conf, valid_mask
        )  # [B, 256]

        # 4. Concatenate: [summary | string_gnn] -> [B, 896]
        emb = torch.cat([summary_emb, string_emb], dim=-1)  # [B, 896]

        # 5. Head (2-layer): 896->256->19920
        logits = self.head(emb).view(B, N_CLASSES, N_GENES)  # [B, 3, G]
        return logits

    # ---- Loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),   # [B*G, 3]
            targets.reshape(-1),                        # [B*G]
            weight=self.class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )

    # ---- Training step ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["pert_id"]
        )
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    # ---- Validation step ----
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["pert_id"]
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
        self._val_preds.clear()
        self._val_tgts.clear()
        self._val_idx.clear()

        # Gather across all GPUs
        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        # Sort and deduplicate
        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]
        s_pred = all_preds[order]
        s_tgt  = all_tgts[order]
        mask   = torch.cat([
            torch.ones(1, dtype=torch.bool, device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    # ---- Test step ----
    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["pert_id"]
        )
        probs = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        self._test_local_idx.append(batch["local_dataset_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        local_idx   = torch.cat(self._test_idx,   0)
        local_local = torch.cat(self._test_local_idx, 0)

        # Gather from all GPUs
        all_preds  = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_idx    = self.all_gather(local_idx).view(-1)
        all_local  = self.all_gather(local_local).view(-1)

        if self.trainer.is_global_zero:
            # Sort and deduplicate
            order   = torch.argsort(all_idx)
            s_idx   = all_idx[order]
            s_pred  = all_preds[order]
            s_local = all_local[order]
            mask    = torch.cat([
                torch.ones(1, dtype=torch.bool, device=s_idx.device),
                s_idx[1:] != s_idx[:-1],
            ])
            s_local = s_local[mask]
            s_pred  = s_pred[mask]

            test_ds = self.trainer.datamodule.test_ds
            rows = []
            for i in range(len(s_local)):
                idx = s_local[i].item()
                pid = test_ds.pert_ids[idx]
                sym = test_ds.symbols[idx]
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(s_pred[i].float().cpu().numpy().tolist()),
                })

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"[node4-1-3-1-2-1-1-1] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()
        self._test_local_idx.clear()

    # ---- Checkpoint: save only trainable params ----
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_sd = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable_sd[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                trainable_sd[key] = full_sd[key]
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"[node4-1-3-1-2-1-1-1] Checkpoint: {trained}/{total} params ({100 * trained / total:.2f}%)")
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer + scheduler ----
    def configure_optimizers(self):
        hp = self.hparams
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )

        warmup_epochs = hp.warmup_epochs
        total_epochs  = hp.max_epochs

        # Linear warmup from 10% LR to 100% LR over warmup_epochs
        warmup_sched = LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        # Cosine annealing from 100% LR to eta_min over remaining epochs
        # T_max=290: same as parent (proven schedule shape)
        # eta_min=1e-6 (critical — NOT 1e-5): prevents late-epoch drift with higher LoRA capacity
        # With max_epochs=400, the model runs ~100 extra epochs at eta_min after cosine completes.
        cosine_t_max = max(min(total_epochs - warmup_epochs, 290), 1)  # 290 proven, cap at 290
        cosine_sched = CosineAnnealingLR(
            opt,
            T_max=cosine_t_max,
            eta_min=hp.eta_min,
        )
        scheduler = SequentialLR(
            opt,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        return {
            "optimizer": opt,
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
        description="Node4-1-3-1-2-1-1-1: AIDO.Cell-100M + STRING_GNN K=16 2-head + "
                    "Summary Only + LoRA r=16 ALL 18 layers + patience=200 full cosine decay"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=4)
    parser.add_argument("--global-batch-size",  type=int,   default=32)
    parser.add_argument("--max-epochs",         type=int,   default=400)   # KEY CHANGE: 300 → 400
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--weight-decay",       type=float, default=3e-2)  # KEY CHANGE: 2e-2 → 3e-2
    parser.add_argument("--lora-r",             type=int,   default=16,    # KEY CHANGE: 8 → 16
                        dest="lora_r")
    parser.add_argument("--lora-alpha",         type=int,   default=32,    # KEY CHANGE: 16 → 32
                        dest="lora_alpha")
    parser.add_argument("--lora-dropout",       type=float, default=0.10,  # KEY CHANGE: 0.05 → 0.10
                        dest="lora_dropout")
    parser.add_argument("--lora-last-n-layers", type=int,   default=18,    dest="lora_last_n_layers")
    parser.add_argument("--head-hidden",        type=int,   default=256)
    parser.add_argument("--head-dropout",       type=float, default=0.5)
    parser.add_argument("--warmup-epochs",      type=int,   default=10)
    parser.add_argument("--eta-min",            type=float, default=1e-6,  dest="eta_min")
    parser.add_argument("--patience",           type=int,   default=200)   # KEY CHANGE: 15 → 200
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--val-check-interval", type=float, default=1.0,   dest="val_check_interval")
    parser.add_argument("--debug-max-step",     type=int,   default=None,  dest="debug_max_step")
    parser.add_argument("--fast-dev-run",       action="store_true",       dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean checkpoint directory before training (rank-0 only to avoid race conditions)
    import shutil as _shutil
    ckpt_dir = output_dir / "checkpoints"
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        if ckpt_dir.exists():
            _shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = lim_val = lim_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm = DEGDataModule(
        batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model = AIDOCellStringAllLayersModelV3(
        lora_r             = args.lora_r,
        lora_alpha         = args.lora_alpha,
        lora_dropout       = args.lora_dropout,
        lora_last_n_layers = args.lora_last_n_layers,
        head_hidden        = args.head_hidden,
        head_dropout       = args.head_dropout,
        lr                 = args.lr,
        weight_decay       = args.weight_decay,
        warmup_epochs      = args.warmup_epochs,
        max_epochs         = args.max_epochs,
        eta_min            = args.eta_min,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
        save_on_train_epoch_end = False,
        auto_insert_metric_name = False,
    )
    es_cb = EarlyStopping(
        monitor   = "val/f1",
        mode      = "max",
        patience  = args.patience,  # KEY CHANGE: 200 (from 15) — allow full cosine decay
        min_delta = 0.0,            # Any improvement resets counter (proven in parent)
        # With patience=200 and min_delta=0.0, early stopping is effectively disabled:
        # improvements occur every 5-10 epochs during the oscillatory plateau, resetting
        # the counter. Training runs to max_epochs=400, completing the full cosine schedule.
        # The low-LR regime (epochs 93-300, LR 8e-5 → 1e-6) is the primary new exploration.
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

    # Only rank-0 writes test_score.txt to avoid race conditions in DDP
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        score_path = Path(__file__).parent / "test_score.txt"
        with open(score_path, "w") as f:
            f.write(f"test_results: {test_results}\n")
            if test_results:
                for k, v in test_results[0].items():
                    f.write(f"  {k}: {v}\n")
        print(f"[node4-1-3-1-2-1-1-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
