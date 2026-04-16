"""Node2-1-2: AIDO.Cell-100M (LoRA) + STRING_GNN 2-Head K=16 Neighborhood Attention
            + Discriminative LRs + Extended Training (300 epochs)

Key improvements over parent node2-1-1 (AIDO.Cell-100M + raw STRING_GNN, F1=0.4535)
and differentiating from sibling node2-1-1-1 (K=16 single-head, F1=0.5059):

1. 2-Head Neighborhood Attention (K=16) — matching tree-best node2-1-1-1-1-1 (F1=0.5128)
   Multiple attention heads capture different aspects of PPI neighborhood topology
   Each head: W_q[256→64], W_k[256→64]; heads concatenated → W_proj[512→256]; gating W_gate[512→256]

2. Discriminative Learning Rates: backbone_lr=5e-5, head_lr=2e-4
   Conservative backbone adaptation preserves AIDO.Cell pretrained knowledge on 1,388 samples
   Proven in node2-1-2 (F1=0.4921) with zero val-test gap

3. Extended Training: max_epochs=300, patience=15
   Sibling stopped at epoch 63 WITHOUT early stopping — model was still improving
   Expected peak at epoch 100-150; patience=15 captures late-epoch improvements

4. Restored head_hidden=256 (from parent's 128), matching sibling's fix
5. All proven regularization retained: weighted CE + label smoothing, weight_decay=2e-2
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
AIDO_GENES       = 19264       # AIDO.Cell gene vocabulary size
AIDO_MODEL_DIR   = "/home/Models/AIDO.Cell-100M"
STRING_MODEL_DIR = "/home/Models/STRING_GNN"
AIDO_HIDDEN      = 640         # AIDO.Cell-100M hidden size
STRING_HIDDEN    = 256         # STRING_GNN hidden size
ATTN_DIM         = 64          # Per-head attention projection dimension
N_ATTN_HEADS     = 2           # Number of attention heads
N_NEIGHBORS      = 16          # Top-K PPI neighbors

# Class frequency: [down(-1→0), neutral(0→1), up(+1→2)]
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

    Unlike DistributedSampler which requires world_size coordination and may
    add padding, this sampler uses LOCAL_RANK to split the dataset directly.
    Each rank processes a different, non-overlapping subset of the data.
    This eliminates DDP sample collision issues in val/test phases and
    avoids the padding-induced duplicate issue in small datasets.
    """

    def __init__(self, dataset, shuffle: bool = False, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.rng = torch.Generator()

        # Get rank info
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
            n = len(self.dataset)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Assign a disjoint subset to this rank
        rank_indices = [indices[i] for i in range(self.rank, len(indices), self.world_size)]
        return iter(rank_indices)

    def __len__(self):
        n = len(self.dataset)
        return (n + self.world_size - 1) // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch


def make_collate(tokenizer):
    """Factory for collate_fn with AIDO.Cell tokenizer."""

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]

        # Tokenize: each sample gets only its perturbed gene with expression=1.0;
        # all other 19,263 genes receive -1.0 (missing).
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized = tokenizer(expr_dicts, return_tensors="pt")  # input_ids: [B, 19264] float32

        # Find the gene position for each sample (position where input_ids > -1.0)
        input_ids = tokenized["input_ids"]   # [B, 19264]
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)                        # [B]
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),
            torch.zeros(len(batch), dtype=torch.long),
        )   # [B]

        # Compute GLOBAL sample indices for proper DDP deduplication.
        # Dataset index is unique across all ranks since each rank processes
        # a different disjoint subset of the dataset.
        dataset_indices = torch.tensor(
            [b["sample_idx"] for b in batch], dtype=torch.long
        )

        out: Dict[str, Any] = {
            "sample_idx":          dataset_indices,
            "local_dataset_idx":   dataset_indices,  # For pert_id lookup post-gather
            "pert_id":             pert_ids,
            "symbol":              symbols,
            "input_ids":           input_ids,
            "attention_mask":      tokenized["attention_mask"],
            "gene_positions":      gene_positions,
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

    def _loader(self, ds: Dataset, shuffle: bool, sampler: Optional[torch.utils.data.Sampler] = None) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate(self.tokenizer),
            sampler=sampler,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        val_sampler  = RankAwareSampler(self.val_ds,  shuffle=False)
        return self._loader(self.val_ds,  shuffle=False, sampler=val_sampler)

    def test_dataloader(self) -> DataLoader:
        test_sampler = RankAwareSampler(self.test_ds, shuffle=False)
        return self._loader(self.test_ds, shuffle=False, sampler=test_sampler)


# ---------------------------------------------------------------------------
# Neighborhood Attention Module (2-Head)
# ---------------------------------------------------------------------------
class NeighborhoodAttention2Head(nn.Module):
    """2-head neighborhood attention over STRING_GNN PPI graph.

    For each perturbed gene:
    1. Retrieve center embedding (frozen STRING_GNN node) → [B, 256]
    2. Retrieve top-K PPI neighbor embeddings (frozen) → [B, K, 256]
    3. Apply 2-head cross-attention with PPI confidence weighting
    4. Fuse center and context via learnable gating
    5. Output: context-aware PPI embedding → [B, 256]

    Architecture:
    - Head 1: W_q1 [256→64], W_k1 [256→64] → attn_1 [B, K] → context_1 [B, 256]
    - Head 2: W_q2 [256→64], W_k2 [256→64] → attn_2 [B, K] → context_2 [B, 256]
    - Multi-head context: W_proj [512→256] (concat(context_1, context_2))
    - Gating: W_gate [512→256] (concat(center, context)) → sigmoid → gate [B, 256]
    - Output: gate * center + (1 - gate) * context → [B, 256]
    """

    def __init__(
        self,
        embed_dim: int = 256,
        attn_dim: int = 64,
        n_heads: int = 2,
        n_neighbors: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim   = embed_dim
        self.attn_dim    = attn_dim
        self.n_heads     = n_heads
        self.n_neighbors = n_neighbors
        self.scale       = attn_dim ** -0.5

        # Per-head projection matrices
        self.W_q = nn.ModuleList([
            nn.Linear(embed_dim, attn_dim, bias=False)
            for _ in range(n_heads)
        ])
        self.W_k = nn.ModuleList([
            nn.Linear(embed_dim, attn_dim, bias=False)
            for _ in range(n_heads)
        ])

        # Multi-head context projection: concat(n_heads * context) → embed_dim
        self.W_proj = nn.Linear(n_heads * embed_dim, embed_dim, bias=False)

        # Gating: concat(center, context) → gate
        self.W_gate = nn.Linear(2 * embed_dim, embed_dim, bias=False)

        # Fallback for genes not in STRING vocabulary (non-trainable, float32 buffer)
        self.register_buffer("fallback_emb", torch.zeros(embed_dim, dtype=torch.float32))

    def forward(
        self,
        center_embs: torch.Tensor,    # [B, 256]
        neigh_embs: torch.Tensor,     # [B, K, 256]
        neigh_confs: torch.Tensor,    # [B, K]
        valid_mask: torch.Tensor,     # [B] bool, True if in STRING vocab
    ) -> torch.Tensor:               # [B, 256]
        B, K, D = neigh_embs.shape
        assert D == self.embed_dim

        # For genes not in vocab, use fallback embedding
        fb = self.fallback_emb.unsqueeze(0).expand(B, -1).to(center_embs.dtype)  # [B, 256]
        center = torch.where(
            valid_mask.unsqueeze(1).expand_as(center_embs),
            center_embs,
            fb,
        )  # [B, 256]

        # Log-domain PPI confidence weighting: [B, 1, K]
        log_conf = torch.log(neigh_confs.clamp(min=1e-8)).unsqueeze(1)  # [B, 1, K]

        # Compute per-head attention and context
        head_contexts = []
        for h in range(self.n_heads):
            q = self.W_q[h](center)           # [B, attn_dim]
            k = self.W_k[h](neigh_embs)       # [B, K, attn_dim]

            # Scaled dot-product: [B, 1, attn_dim] x [B, attn_dim, K] = [B, 1, K]
            scores = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)) * self.scale  # [B, 1, K]
            scores = scores + log_conf        # Add PPI confidence in log-domain
            attn   = torch.softmax(scores, dim=-1)  # [B, 1, K]

            # Weighted sum of neighbor embeddings: [B, 256]
            context_h = torch.bmm(attn, neigh_embs).squeeze(1)  # [B, 256]
            head_contexts.append(context_h)

        # Multi-head context: concat and project
        multi_context = torch.cat(head_contexts, dim=-1)       # [B, n_heads * 256]
        context       = self.W_proj(multi_context)             # [B, 256]

        # Gating: adaptive combination of center and context
        gate_input = torch.cat([center, context], dim=-1)     # [B, 512]
        gate       = torch.sigmoid(self.W_gate(gate_input))   # [B, 256]
        out        = gate * center + (1.0 - gate) * context   # [B, 256]

        return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCellString2HeadFusionModel(pl.LightningModule):
    """AIDO.Cell-100M (LoRA) + STRING_GNN 2-Head K=16 Neighborhood Attention Fusion.

    Architecture:
      AIDO.Cell backbone (LoRA r=8):
        → last_hidden_state[:, AIDO_GENES, :] → summary_token [B, 640]

      STRING_GNN K=16 2-Head Neighborhood Attention (frozen GNN embeddings, trainable attention):
        → context-aware PPI embedding [B, 256]

      Fusion: concat([summary_token, string_context]) → [B, 896]
      Head: Linear(896→256) → LayerNorm → GELU → Dropout(0.5) → Linear(256→19920)
            → view([B, 3, 6640])

    Discriminative LRs:
      - AIDO.Cell LoRA params: backbone_lr (5e-5 default)
      - Neighborhood attention + head params: head_lr (2e-4 default)
    """

    def __init__(
        self,
        lora_r: int         = 8,
        lora_alpha: int     = 16,
        lora_dropout: float = 0.05,
        head_hidden: int    = 256,
        head_dropout: float = 0.5,
        backbone_lr: float  = 5e-5,
        head_lr: float      = 2e-4,
        weight_decay: float = 2e-2,
        warmup_epochs: int  = 10,
        max_epochs: int     = 300,
        n_neighbors: int    = 16,
        attn_dim: int       = 64,
        n_attn_heads: int   = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load STRING_GNN backbone (frozen) and pre-compute all embeddings ----
        import json as _json
        string_model = AutoModel.from_pretrained(STRING_MODEL_DIR, trust_remote_code=True)
        string_model.eval()

        graph_data   = torch.load(Path(STRING_MODEL_DIR) / "graph_data.pt", map_location="cpu")
        node_names   = _json.loads((Path(STRING_MODEL_DIR) / "node_names.json").read_text())
        self._string_node_names = node_names  # list of Ensembl IDs

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

        # Build lookup: Ensembl ID → row index in string_embs
        string_id_to_idx = {eid: i for i, eid in enumerate(node_names)}
        self._string_id_to_idx = string_id_to_idx

        # Register STRING embeddings as a non-trainable buffer
        self.register_buffer("string_embs", string_embs)  # [18870, 256], float32

        # ---- Pre-compute top-K neighbors for all nodes ----
        N_nodes = len(node_names)
        K       = hp.n_neighbors

        # Build adjacency from edge_index + edge_weight
        # edge_index: [2, E], edge_weight: [E] (normalized STRING combined_score)
        ei  = edge_index   # [2, E] — already CPU
        ew  = edge_weight  # [E]
        if ew is None:
            ew = torch.ones(ei.shape[1])
        ew = ew.float()

        # For each node, collect (neighbor_idx, weight) then take top-K
        from collections import defaultdict
        neigh_dict: Dict[int, List] = defaultdict(list)
        for e_idx in range(ei.shape[1]):
            src = ei[0, e_idx].item()
            dst = ei[1, e_idx].item()
            w   = ew[e_idx].item()
            neigh_dict[src].append((dst, w))
            # STRING GNN uses directed edges for message passing;
            # add both directions for neighbor lookup
            neigh_dict[dst].append((src, w))

        topk_neighbors = torch.zeros(N_nodes, K, dtype=torch.long)   # [N, K]
        topk_weights   = torch.zeros(N_nodes, K, dtype=torch.float32)  # [N, K]

        for node_i in range(N_nodes):
            neighbors = neigh_dict.get(node_i, [])
            if not neighbors:
                # No neighbors: self-loop fallback
                topk_neighbors[node_i, :] = node_i
                topk_weights[node_i, :] = 1.0
            else:
                # Sort by weight (descending) and take top-K
                neighbors_sorted = sorted(neighbors, key=lambda x: -x[1])[:K]
                for j, (nb_idx, nb_w) in enumerate(neighbors_sorted):
                    topk_neighbors[node_i, j] = nb_idx
                    topk_weights[node_i, j]   = nb_w
                # If fewer than K neighbors, pad with self-loops
                n_actual = len(neighbors_sorted)
                if n_actual < K:
                    for j in range(n_actual, K):
                        topk_neighbors[node_i, j] = node_i
                        topk_weights[node_i, j]   = 1.0

        # Normalize weights to [0, 1] range (already normalized per STRING score)
        # but clamp to prevent zeros
        topk_weights = topk_weights.clamp(min=1e-8)

        self.register_buffer("topk_neighbors", topk_neighbors)  # [N, K]
        self.register_buffer("topk_weights",   topk_weights)    # [N, K]

        # ---- Load AIDO.Cell backbone ----
        backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # AIDO.Cell has a custom GeneEmbedding (no standard word_embeddings).
        # PEFT's get_peft_model calls enable_input_require_grads(), which internally
        # calls get_input_embeddings() → raises NotImplementedError.
        # Monkey-patch to a no-op; we register a forward hook below instead.
        backbone.enable_input_require_grads = lambda: None

        # ---- Apply LoRA to AIDO.Cell ----
        lora_cfg = LoraConfig(
            task_type      = TaskType.FEATURE_EXTRACTION,
            r              = hp.lora_r,
            lora_alpha     = hp.lora_alpha,
            lora_dropout   = hp.lora_dropout,
            target_modules = ["query", "key", "value"],
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

        # ---- 2-Head Neighborhood Attention (trainable, float32) ----
        self.neigh_attn = NeighborhoodAttention2Head(
            embed_dim   = STRING_HIDDEN,  # 256
            attn_dim    = hp.attn_dim,    # 64
            n_heads     = hp.n_attn_heads, # 2
            n_neighbors = hp.n_neighbors, # 16
        )
        # Ensure all attention module params (and registered buffers) are float32
        for name, param in self.neigh_attn.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()
        # Buffers (fallback_emb, etc.) should already be float32 from register_buffer

        # ---- Classification head ----
        # Input: concat(AIDO summary_token [640], STRING context [256]) = 896-dim
        in_dim = AIDO_HIDDEN + STRING_HIDDEN  # 896
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # ---- Loss weights ----
        self.register_buffer("class_weights", get_class_weights())

        # ---- Accumulators ----
        self._val_preds: List[torch.Tensor] = []
        self._val_tgts:  List[torch.Tensor] = []
        self._val_idx:   List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []
        self._test_local_idx: List[torch.Tensor] = []

    def _get_string_context_for_batch(self, pert_ids: List[str]) -> torch.Tensor:
        """Compute STRING_GNN 2-head neighborhood attention context for a batch.

        Returns [B, 256] tensor. Uses fallback_emb for genes not in STRING vocabulary.
        """
        B = len(pert_ids)
        indices = [self._string_id_to_idx.get(pid, -1) for pid in pert_ids]
        idx_tensor  = torch.tensor(indices, dtype=torch.long, device=self.string_embs.device)
        valid_mask  = (idx_tensor >= 0)  # [B]
        safe_idx    = idx_tensor.clamp(min=0)  # [B]

        # Gather center embeddings
        center_embs = self.string_embs[safe_idx]  # [B, 256]

        # Gather neighbor indices and weights
        # topk_neighbors: [N, K], topk_weights: [N, K]
        neigh_idx    = self.topk_neighbors[safe_idx]  # [B, K]
        neigh_w      = self.topk_weights[safe_idx]    # [B, K]

        # Gather neighbor embeddings: [B, K, 256]
        K            = neigh_idx.shape[1]                     # = n_neighbors (from hparam)
        flat_idx     = neigh_idx.view(-1)                     # [B*K]
        flat_embs    = self.string_embs[flat_idx]             # [B*K, 256]
        neigh_embs   = flat_embs.view(B, K, STRING_HIDDEN)    # [B, K, 256]

        # Run neighborhood attention (trainable params in float32)
        context = self.neigh_attn(
            center_embs,  # [B, 256]
            neigh_embs,   # [B, K, 256]
            neigh_w,      # [B, K]
            valid_mask,   # [B]
        )  # [B, 256]

        return context

    # ---- Forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_positions: torch.Tensor,
        pert_ids: List[str],
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        # 1. AIDO.Cell forward pass
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        # last_hidden_state: [B, AIDO_GENES + 2, HIDDEN_DIM] = [B, 19266, 640]
        lhs = out.last_hidden_state

        # Summary token (position AIDO_GENES=19264): aggregates all gene context
        summary_emb = lhs[:, AIDO_GENES, :].float()  # [B, 640]

        # 2. STRING_GNN 2-head neighborhood attention context (trainable attention)
        string_context = self._get_string_context_for_batch(pert_ids)  # [B, 256], float32

        # 3. Concatenate: [summary_token | STRING_GNN_context] → [B, 896]
        emb = torch.cat([summary_emb, string_context], dim=-1)  # [B, 896]

        # 4. Head
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
            batch["gene_positions"], batch["pert_id"]
        )
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    # ---- Validation step ----
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
        self._val_preds.clear()
        self._val_tgts.clear()
        self._val_idx.clear()

        # Gather across all GPUs
        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        # Sort and deduplicate (handles DDP duplicates from DistributedSampler padding)
        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]
        s_pred = all_preds[order]
        s_tgt  = all_tgts[order]
        mask   = torch.cat([
            torch.tensor([True], device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    # ---- Test step ----
    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"]
        )
        probs  = torch.softmax(logits.float(), dim=1).detach()
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
        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)
        all_local = self.all_gather(local_local).view(-1)

        if self.trainer.is_global_zero:
            # Sort and deduplicate by global sample index
            order  = torch.argsort(all_idx)
            s_idx  = all_idx[order]
            s_pred = all_preds[order]
            s_local = all_local[order]
            mask   = torch.cat([
                torch.tensor([True], device=s_idx.device),
                s_idx[1:] != s_idx[:-1],
            ])
            s_local = s_local[mask]
            s_pred  = s_pred[mask]

            # Retrieve pert_id / symbol from DataModule's test dataset using local_dataset_idx
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
            self.print(f"[node2-1-2] Saved {len(rows)} test predictions.")

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
        self.print(f"Checkpoint: {trained}/{total} params ({100 * trained / total:.2f}%)")
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer + scheduler (discriminative LRs) ----
    def configure_optimizers(self):
        hp = self.hparams

        # Collect backbone (LoRA) vs head+attention parameters
        backbone_params = [p for n, p in self.backbone.named_parameters() if p.requires_grad]
        head_params     = (
            list(self.neigh_attn.parameters()) +
            list(self.head.parameters())
        )

        param_groups = [
            {"params": backbone_params, "lr": hp.backbone_lr, "name": "backbone_lora"},
            {"params": head_params,     "lr": hp.head_lr,     "name": "head_and_attn"},
        ]

        opt = torch.optim.AdamW(
            param_groups,
            weight_decay=hp.weight_decay,
        )

        warmup_epochs = hp.warmup_epochs
        total_epochs  = hp.max_epochs
        T_max         = max(total_epochs - warmup_epochs, 1)

        # Create one scheduler per param group for independent LR scaling
        # SequentialLR applies the same milestone-based schedule to all groups
        warmup_sched = LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched = CosineAnnealingLR(
            opt,
            T_max=T_max,
            eta_min=1e-6,
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
        description="Node2-1-2 – AIDO.Cell-100M + STRING_GNN 2-Head K=16 Neighborhood Attention"
                    " + Discriminative LRs + Extended Training"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=4)
    parser.add_argument("--global-batch-size",  type=int,   default=32)
    parser.add_argument("--max-epochs",         type=int,   default=300)
    parser.add_argument("--backbone-lr",        type=float, default=5e-5)
    parser.add_argument("--head-lr",            type=float, default=2e-4)
    parser.add_argument("--weight-decay",       type=float, default=2e-2)
    parser.add_argument("--lora-r",             type=int,   default=8)
    parser.add_argument("--lora-alpha",         type=int,   default=16)
    parser.add_argument("--lora-dropout",       type=float, default=0.05)
    parser.add_argument("--head-hidden",        type=int,   default=256)
    parser.add_argument("--head-dropout",       type=float, default=0.5)
    parser.add_argument("--n-neighbors",        type=int,   default=16)
    parser.add_argument("--attn-dim",           type=int,   default=64)
    parser.add_argument("--n-attn-heads",       type=int,   default=2)
    parser.add_argument("--warmup-epochs",      type=int,   default=10)
    parser.add_argument("--patience",           type=int,   default=15)
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--debug-max-step",     type=int,   default=None, dest="debug_max_step")
    parser.add_argument("--fast-dev-run",       action="store_true", dest="fast_dev_run")
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

    dm = DEGDataModule(
        batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model = AIDOCellString2HeadFusionModel(
        lora_r        = args.lora_r,
        lora_alpha    = args.lora_alpha,
        lora_dropout  = args.lora_dropout,
        head_hidden   = args.head_hidden,
        head_dropout  = args.head_dropout,
        backbone_lr   = args.backbone_lr,
        head_lr       = args.head_lr,
        weight_decay  = args.weight_decay,
        warmup_epochs = args.warmup_epochs,
        max_epochs    = args.max_epochs,
        n_neighbors   = args.n_neighbors,
        attn_dim      = args.attn_dim,
        n_attn_heads  = args.n_attn_heads,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
    )
    es_cb = EarlyStopping(
        monitor   = "val/f1",
        mode      = "max",
        patience  = args.patience,
        min_delta = 1e-3,
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
        val_check_interval      = 1.0 if (args.debug_max_step is not None or fast_dev_run) else args.val_check_interval,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 5,
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
    print(f"[node2-1-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
