"""Node 3-1-3-2-1: AIDO.Cell-100M (LoRA r=8) + STRING_GNN K=16 **2-Head** Neighborhood Attention.

PRIMARY CHANGE from parent node3-1-3-2 (test F1=0.4884):
  Switch from single-head (1-head, attn_dim=64) to 2-head (2-head, attn_dim=64 per head)
  neighborhood attention for STRING_GNN.
  The parent feedback explicitly recommends: "Increase to 2-head neighborhood attention —
  this is the single most impactful change to pursue. The tree-best node (node2-1-1-1-1-1,
  F1=0.5128) uses 2-head attention. The single-head approach was chosen for conservatism,
  but the data shows 2-head is both safe and beneficial."

SECONDARY CHANGE: Increase global batch size from 32 to 64.
  Parent feedback: "Small effective batch size (32) may affect the optimization trajectory.
  Larger batches often provide more stable gradients and better convergence on small datasets."

TERTIARY CHANGE: Increase patience from 15 to 20, min_delta from 0.001 to 0.0005.
  tree-best node2-1-1-1-1-1 showed a late spike to its peak at epoch 77 after spending
  epochs 10-65 in a plateau (0.49-0.51). More lenient stopping allows such late spikes.

ALL ELSE RETAINED from parent (proven optimal configuration):
  - LoRA r=8 (all 18 layers): proven; r=16 caused catastrophic failure in node2-1-1-1-1-1-1-1
  - Summary token extraction: global context more robust than gene-position embedding
  - head_dropout=0.5, label_smoothing=0.05: matches tree-best
  - lr=1e-4, 10-epoch warmup, weight_decay=2e-2: proven for AIDO.Cell LoRA
  - max_epochs=200: allow full convergence
  - gradient_clip_val=1.0: stable LoRA training

Architecture:
- AIDO.Cell-100M (640-dim): LoRA r=8, α=16, targeting query/key/value in all 18 layers
  * ~0.55M trainable params
  * FlashAttention-2 naturally handled by PEFT
  * Gradient checkpointing for memory efficiency
  * Summary token extraction: last_hidden_state[:, 19264, :] → [B, 640]
- STRING_GNN: frozen, K=16 **2-head** neighborhood attention (attn_dim=64 per head)
  * ~250K trainable params
  * Two independent attention heads capture diverse PPI relationship types
  * center_emb + weighted context from top-16 PPI neighbors → [B, 256]
- Fusion: simple concat [AIDO_640 + STRING_256] = 896-dim
- Head: 896 → 256 (LN + GELU + Dropout=0.5) → 19920 → reshape [B, 3, 6640]
- Loss: label-smoothed CE (ε=0.05) + sqrt-inverse-freq class weights
- Optimizer: AdamW (lr=1e-4, weight_decay=2e-2) with LinearLR warmup + CosineAnnealingLR
- Global batch size: 64 (up from parent's 32; more stable gradients on 1388 samples)
- Patience: 20, min_delta=0.0005 (allows capturing late convergence spikes)
- Total trainable params: ~0.55M (LoRA) + ~250K (STRING attn) + ~5.1M (head) = ~5.9M

Expected F1: 0.50-0.515 (closing gap to tree-best 0.5128)
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
STRING_GNN_DIR   = "/home/Models/STRING_GNN"
AIDO_HIDDEN      = 640         # AIDO.Cell-100M hidden dimension
STRING_HIDDEN    = 256         # STRING_GNN hidden dimension
NEIGHBOR_K       = 16          # K=16: proven optimal in node2-1-1-1 (F1=0.5059)
ATTN_DIM         = 64          # Per-head attention dim; 2-head = 128 total attn capacity
N_HEADS          = 2           # 2-head: tree-best (node2-1-1-1-1-1, F1=0.5128) uses 2-head

# Class frequency: [down(-1→0), neutral(0→1), up(+1→2)]
CLASS_FREQ      = [0.0429, 0.9251, 0.0320]
LABEL_SMOOTHING = 0.05  # Matches tree-best nodes

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights for 92.5% neutral class imbalance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1, exactly matching data/calc_metric.py logic.

    Args:
        preds:   [N, 3, G] softmax probabilities
        targets: [N, G]    integer class labels in {0,1,2}
    Returns:
        scalar F1 averaged over all G genes
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
        f1_c = torch.where(
            prec + rec > 0,
            2 * prec * rec / (prec + rec + 1e-8),
            torch.zeros_like(prec),
        )
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# K=16 Multi-Head Neighborhood Attention Module (2-head)
# ---------------------------------------------------------------------------
class MultiHeadNeighborhoodAttentionModule(nn.Module):
    """K=16 PPI neighborhood attention with n_heads independent attention heads.

    Each head uses center gene embedding as query against top-K=16 PPI neighbors.
    2-head matches tree-best configuration (node2-1-1-1-1-1, F1=0.5128).

    Architecture (per head h in [0, n_heads-1]):
        q_h = W_q_h(center_emb)                          # [B, attn_dim]
        k_h = W_k_h(neigh_embs)                          # [B, K, attn_dim]
        attn_h = softmax(q_h @ k_h.T / sqrt(attn_dim) + log(edge_conf))  # [B, K]
        ctx_h = attn_h @ neigh_embs                       # [B, 256]

    Aggregate:
        context = mean(ctx_h for h in heads)              # [B, 256]
        gate = sigmoid(W_gate([center, context]))          # [B, 256]
        output = gate * center + (1-gate) * context       # [B, 256]

    For genes not in STRING vocabulary: returns center_emb (fallback) unchanged.
    """

    def __init__(
        self,
        emb_dim:  int   = 256,
        n_heads:  int   = 2,
        attn_dim: int   = 64,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.emb_dim  = emb_dim
        self.n_heads  = n_heads
        self.attn_dim = attn_dim

        # Per-head Q/K projection matrices
        self.W_qs = nn.ModuleList([nn.Linear(emb_dim, attn_dim, bias=False) for _ in range(n_heads)])
        self.W_ks = nn.ModuleList([nn.Linear(emb_dim, attn_dim, bias=False) for _ in range(n_heads)])

        # Gating between center identity and aggregated multi-head context
        self.W_gate = nn.Linear(emb_dim * 2, emb_dim, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        center_emb: torch.Tensor,   # [B, 256]
        neigh_embs: torch.Tensor,   # [B, K, 256]
        neigh_conf: torch.Tensor,   # [B, K] (STRING edge confidence weights, normalized)
        valid_mask: torch.Tensor,   # [B] bool (True = gene in STRING vocabulary)
    ) -> torch.Tensor:              # [B, 256]
        B, K, D = neigh_embs.shape

        # Log-domain PPI confidence bias (adds preference for high-confidence neighbors)
        log_conf = (neigh_conf + 1e-8).log()  # [B, K]

        # Multi-head attention: each head independently queries the neighborhood
        head_contexts = []
        for h in range(self.n_heads):
            # Compute per-head attention scores
            q_h = self.W_qs[h](center_emb)                                        # [B, attn_dim]
            k_h = self.W_ks[h](neigh_embs.reshape(-1, D)).reshape(B, K, self.attn_dim)  # [B, K, attn_dim]

            # Scaled dot-product attention with PPI confidence bias
            attn_h = (q_h.unsqueeze(1) @ k_h.transpose(1, 2)) / (self.attn_dim ** 0.5)  # [B, 1, K]
            attn_h = attn_h.squeeze(1) + log_conf   # [B, K]
            attn_h = torch.softmax(attn_h, dim=-1)  # [B, K]
            attn_h = self.dropout(attn_h)

            # Weighted neighbor context for this head
            ctx_h = (attn_h.unsqueeze(1) @ neigh_embs).squeeze(1)  # [B, 256]
            head_contexts.append(ctx_h)

        # Average context across all heads
        # Mean aggregation provides stable, diverse neighborhood representation
        context = torch.stack(head_contexts, dim=0).mean(dim=0)  # [B, 256]

        # Learnable gating between center identity and neighborhood context
        gate   = torch.sigmoid(self.W_gate(torch.cat([center_emb, context], dim=-1)))  # [B, 256]
        output = gate * center_emb + (1 - gate) * context  # [B, 256]

        # For genes not in STRING vocabulary: return center_emb (no neighborhood context)
        output = torch.where(valid_mask.unsqueeze(-1), output, center_emb)
        return output


# ---------------------------------------------------------------------------
# Dataset / DataModule
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


def make_collate(tokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]

        # Set expression=1.0 for the perturbed gene only; all others default to -1.0 (missing)
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        input_ids  = tokenized["input_ids"]  # [B, 19264] float32

        # Find the position of the perturbed gene in the fixed 19264-gene sequence
        # (needed for gene_positions API compatibility; we use summary token in forward)
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)    # [B] bool
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),     # position of perturbed gene
            torch.zeros(len(batch), dtype=torch.long),    # fallback: position 0
        )

        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
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
    def __init__(self, batch_size: int = 4, num_workers: int = 4) -> None:
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

        self.train_ds = DEGDataset(pd.read_csv(TRAIN_TSV, sep="\t"))
        self.val_ds   = DEGDataset(pd.read_csv(VAL_TSV,   sep="\t"))
        self.test_ds  = DEGDataset(pd.read_csv(TEST_TSV,  sep="\t"))

    def _loader(self, ds: DEGDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=make_collate(self.tokenizer),
        )

    def train_dataloader(self) -> DataLoader: return self._loader(self.train_ds, True)
    def val_dataloader(self)   -> DataLoader: return self._loader(self.val_ds,   False)
    def test_dataloader(self)  -> DataLoader: return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# Main Model: AIDO.Cell-100M (LoRA) + STRING_GNN K=16 2-Head Attention + 2-Layer Head
# ---------------------------------------------------------------------------
class AIDOLoRAString2HeadK16Model(pl.LightningModule):
    """AIDO.Cell-100M (LoRA r=8) + frozen STRING_GNN K=16 2-head neighborhood attention.

    Architecture:
      AIDO.Cell-100M (LoRA r=8, all 18 layers' Q/K/V):
        gene expression (pert gene=1.0, rest=-1.0) → 18-layer transformer
        last_hidden_state[:, AIDO_GENES, :] → summary_token [B, 640]
        (summary token at position 19264 = global transcriptome context)

      STRING_GNN (frozen, K=16 **2-head** neighborhood attention):
        pert_id → STRING node index → center_emb [B, 256]
        + top-16 PPI neighbor embeddings [B, 16, 256]
        → MultiHeadNeighborhoodAttentionModule (2 heads) → context_emb [B, 256]

      Fusion: concat([summary_token, context_emb]) → [B, 896]

      Head (2-layer): 896 → 256 (LN+GELU+Dropout=0.5) → 19920 → [B, 3, 6640]

    Primary change from parent (node3-1-3-2):
    - 2-head attention (vs single-head): tree-best node2-1-1-1-1-1 uses 2-head → F1=0.5128
    - larger global batch (64 vs 32): more stable gradients on 1388 samples
    - patience=20 (vs 15): allow capturing late convergence spikes
    """

    def __init__(
        self,
        lora_r:         int   = 8,
        lora_alpha:     int   = 16,
        lora_dropout:   float = 0.05,
        head_hidden:    int   = 256,
        head_dropout:   float = 0.5,     # strong regularization; matches tree-best
        lr:             float = 1e-4,
        weight_decay:   float = 2e-2,
        warmup_epochs:  int   = 10,      # linear warmup; proven for AIDO.Cell LoRA
        max_epochs:     int   = 200,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load STRING_GNN (frozen) and pre-compute all embeddings ----
        string_model = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        string_model.eval()

        graph_data   = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", map_location="cpu")
        node_names   = json.loads((Path(STRING_GNN_DIR) / "node_names.json").read_text())

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

        # Build Ensembl gene ID → STRING node index mapping
        self._string_id_to_idx: Dict[str, int] = {eid: i for i, eid in enumerate(node_names)}
        n_nodes = string_embs.shape[0]

        # Register STRING embeddings as a non-trainable buffer
        self.register_buffer("string_embs", string_embs)  # [18870, 256], float32

        # ---- Build K=16 neighborhood lookup from graph_data ----
        K = NEIGHBOR_K  # 16

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1])

        edge_weight_cpu = edge_weight.float()  # [E]

        # Build adjacency dict: node → [(neighbor_idx, weight)]
        adj: List[List] = [[] for _ in range(n_nodes)]
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            w   = edge_weight_cpu[i].item()
            adj[src].append((dst, w))

        # For each node, select top-K neighbors by weight; pad with self-loop
        topk_neighbors = torch.zeros(n_nodes, K, dtype=torch.long)    # [18870, K]
        topk_weights   = torch.zeros(n_nodes, K, dtype=torch.float32) # [18870, K]

        for i, neighbors in enumerate(adj):
            if len(neighbors) == 0:
                # No neighbors: self-loop fallback
                topk_neighbors[i] = i
                topk_weights[i]   = 1.0
            else:
                # Sort by weight descending, take top K
                neighbors_sorted = sorted(neighbors, key=lambda x: x[1], reverse=True)[:K]
                k_actual = len(neighbors_sorted)
                for j, (nid, nw) in enumerate(neighbors_sorted):
                    topk_neighbors[i, j] = nid
                    topk_weights[i, j]   = nw
                # Pad remaining slots with self-loop (weight=0 → won't be attended)
                for j in range(k_actual, K):
                    topk_neighbors[i, j] = i
                    topk_weights[i, j]   = 0.0
                # Normalize weights to [0,1] so confidence bias is bounded
                max_w = topk_weights[i].max()
                if max_w > 0:
                    topk_weights[i] = topk_weights[i] / max_w

        # Register as non-trainable buffers (moved to GPU automatically)
        self.register_buffer("topk_neighbors", topk_neighbors)  # [18870, K]
        self.register_buffer("topk_weights",   topk_weights)    # [18870, K], normalized

        # ---- 2-Head neighborhood attention module (K=16) ----
        # PRIMARY CHANGE: 2-head (vs single-head in parent)
        # Two independent attention heads capture diverse PPI relationship types
        # Exactly matching tree-best node2-1-1-1-1-1 configuration (F1=0.5128)
        self.neighborhood_attn = MultiHeadNeighborhoodAttentionModule(
            emb_dim=STRING_HIDDEN,
            n_heads=N_HEADS,
            attn_dim=ATTN_DIM,
            dropout=0.1,
        )

        # Learnable fallback embedding for genes not in STRING vocabulary
        self.register_parameter(
            "fallback_emb",
            nn.Parameter(torch.zeros(1, STRING_HIDDEN))
        )
        nn.init.normal_(self.fallback_emb, std=0.01)

        attn_params = sum(p.numel() for p in self.neighborhood_attn.parameters())
        print(
            f"[Node3-1-3-2-1] STRING_GNN loaded: {n_nodes} nodes, K={K}, "
            f"{N_HEADS}-head neighborhood attn params: {attn_params:,}"
        )

        # ---- Load AIDO.Cell-100M backbone ----
        backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        # Gradient checkpointing: critical for 100M model with 19266-length sequences
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Monkey-patch to avoid get_input_embeddings() → NotImplementedError in PEFT
        backbone.enable_input_require_grads = lambda: None

        # ---- Apply LoRA to all 18 layers' Q/K/V (r=8, α=16) ----
        # LoRA r=8: proven optimal; r=16 caused catastrophic failure in node2-1-1-1-1-1-1-1
        # Targets: bert.encoder.layer.*.attention.self.{query,key,value}
        # flash_self shares weight tensors with self, so LoRA on self applies to both
        lora_cfg = LoraConfig(
            task_type      = TaskType.FEATURE_EXTRACTION,
            r              = hp.lora_r,
            lora_alpha     = hp.lora_alpha,
            lora_dropout   = hp.lora_dropout,
            target_modules = ["query", "key", "value"],
            layers_to_transform = None,  # all 18 layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Forward hook: ensure gradients flow through GeneEmbedding to LoRA adapters
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        gene_emb = self.backbone.model.bert.gene_embedding
        gene_emb.register_forward_hook(_make_inputs_require_grad)

        # Cast LoRA params to float32 for stable gradient flow
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ---- Classification head (2-layer) ----
        # Input: concat(AIDO summary_token [640], STRING neighborhood emb [256]) = 896-dim
        in_dim = AIDO_HIDDEN + STRING_HIDDEN  # 896
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),  # → 19920
        )
        # Cast head to float32 for stable gradient flow
        for param in self.head.parameters():
            param.data = param.data.float()

        # Loss weights
        self.register_buffer("class_weights", get_class_weights())

        # Metric accumulation buffers for DDP
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_tgts:  List[torch.Tensor] = []   # accumulate targets for test F1
        self._test_idx:   List[torch.Tensor] = []

    def _get_string_neighbor_embs(self, pert_ids: List[str]):
        """Retrieve STRING_GNN neighborhood embeddings for a batch.

        Returns:
            center_emb: [B, 256] — single-node embedding or fallback
            neigh_embs: [B, K, 256] — top-K neighbor embeddings
            neigh_conf: [B, K] — normalized PPI edge confidence weights
            valid_mask: [B] bool — True if gene in STRING vocabulary
        """
        B      = len(pert_ids)
        K      = NEIGHBOR_K
        device = self.string_embs.device

        indices = [self._string_id_to_idx.get(pid, -1) for pid in pert_ids]
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        valid_mask = (idx_tensor >= 0)
        idx_clamped = idx_tensor.clamp(min=0)

        # Center embeddings (fallback for unknown genes)
        center_emb = torch.zeros(B, STRING_HIDDEN, dtype=torch.float32, device=device)
        if valid_mask.any():
            center_emb[valid_mask] = self.string_embs[idx_clamped[valid_mask]]
        if (~valid_mask).any():
            center_emb[~valid_mask] = self.fallback_emb.expand(
                (~valid_mask).sum(), -1
            ).float()

        # Neighbor lookups (default: uniform weights for invalid genes)
        neigh_idx  = torch.zeros(B, K, dtype=torch.long,    device=device)
        neigh_conf = torch.ones(B, K, dtype=torch.float32,  device=device) / K

        if valid_mask.any():
            valid_src = idx_clamped[valid_mask]
            neigh_idx[valid_mask]  = self.topk_neighbors[valid_src]  # [n_valid, K]
            neigh_conf[valid_mask] = self.topk_weights[valid_src]    # [n_valid, K]

        # Gather neighbor embeddings
        flat_neigh_emb = self.string_embs[neigh_idx.reshape(-1)]     # [B*K, 256]
        neigh_embs = flat_neigh_emb.reshape(B, K, STRING_HIDDEN)     # [B, K, 256]

        return center_emb, neigh_embs, neigh_conf, valid_mask

    # ---- Forward pass ----
    def forward(
        self,
        input_ids:      torch.Tensor,  # [B, 19264] float32
        attention_mask: torch.Tensor,  # [B, 19264] int64
        gene_positions: torch.Tensor,  # [B] int64 (not used; kept for API compatibility)
        pert_ids:       List[str],
    ) -> torch.Tensor:
        """
        Returns:
            logits: [B, N_CLASSES, N_GENES] = [B, 3, 6640]
        """
        B = input_ids.shape[0]

        # ---- AIDO.Cell-100M forward (LoRA fine-tuned) ----
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        # last_hidden_state: [B, 19266, 640] (19264 gene positions + 2 summary tokens appended)
        # Summary token at position AIDO_GENES=19264 aggregates global transcriptome context
        summary_emb = out.last_hidden_state[:, AIDO_GENES, :].float()  # [B, 640]

        # ---- STRING_GNN: K=16 2-head neighborhood attention ----
        center_emb, neigh_embs, neigh_conf, valid_mask = self._get_string_neighbor_embs(pert_ids)
        context_emb = self.neighborhood_attn(center_emb, neigh_embs, neigh_conf, valid_mask)  # [B, 256]

        # ---- Fusion: simple concatenation ----
        fused  = torch.cat([summary_emb, context_emb], dim=-1)   # [B, 896]
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)    # [B, 3, 6640]
        return logits

    # ---- Loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G   = logits.shape
        flat_log  = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        flat_tgt  = targets.reshape(-1)                      # [B*G]
        return F.cross_entropy(
            flat_log, flat_tgt,
            weight=self.class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )

    # ---- Lightning steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"],
        )
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"],
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

        # Gather from all DDP ranks and deduplicate by sample index
        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        dedup  = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[dedup], s_tgt[dedup])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch["pert_id"],
        )
        probs = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)
            self._test_tgts.append(batch["labels"].detach())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        local_idx   = torch.cat(self._test_idx,   0)
        local_tgts  = torch.cat(self._test_tgts, 0) if self._test_tgts else None
        self._test_preds.clear(); self._test_tgts.clear(); self._test_idx.clear()

        # Compute local F1 first (before all_gather which can cause NCCL hangs in DDP test)
        test_f1 = float("nan")
        if local_tgts is not None:
            test_f1 = compute_per_gene_f1(local_preds, local_tgts)
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=True)

        # Write per-rank predictions to temp files to avoid NCCL all_gather deadlock
        # in DDP test (NCCL seq nums conflict between train→test transition)
        out_dir = Path(__file__).parent / "run"
        out_dir.mkdir(parents=True, exist_ok=True)
        rank = self.trainer.global_rank
        tmp_file = out_dir / f"_test_preds_rank{rank}.pt"

        # Save (idx, pred) pairs - dedup happens at merge time
        torch.save({
            "preds": local_preds.cpu(),
            "idx": local_idx.cpu(),
        }, tmp_file)

        # Barrier: ensure all ranks write their files
        torch.distributed.barrier()

        # Rank 0 merges all rank files and writes final predictions
        if self.trainer.is_global_zero:
            all_preds_list = []
            all_idx_list   = []
            world_size = self.trainer.world_size
            for r in range(world_size):
                f = out_dir / f"_test_preds_rank{r}.pt"
                data = torch.load(f, map_location="cpu")
                all_preds_list.append(data["preds"])
                all_idx_list.append(data["idx"])
                try:
                    os.remove(f)
                except OSError:
                    pass

            all_preds = torch.cat(all_preds_list, dim=0)   # [N_total, 3, 6640]
            all_idx   = torch.cat(all_idx_list,   dim=0)   # [N_total]

            # De-duplicate by sample index
            order  = torch.argsort(all_idx)
            s_idx  = all_idx[order]
            s_pred = all_preds[order]
            dedup  = torch.cat([
                torch.tensor([True]),
                s_idx[1:] != s_idx[:-1],
            ])
            s_idx  = s_idx[dedup]
            s_pred = s_pred[dedup]

            # Log merged F1 for reference
            if local_tgts is not None:
                merged_f1 = compute_per_gene_f1(s_pred, s_tgts) if "s_tgts" in dir() else test_f1
                self.log("test/f1_merged", merged_f1, prog_bar=False, sync_dist=False)

            test_ds = self.trainer.datamodule.test_ds
            rows = []
            for i, idx in enumerate(s_idx.cpu().tolist()):
                pid = test_ds.pert_ids[idx]
                sym = test_ds.symbols[idx]
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(s_pred[i].float().cpu().numpy().tolist()),
                })

            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node3-1-3-2-1] Saved {len(rows)} test predictions, test_f1={test_f1:.4f}")

    # ---- Checkpoint: save only trainable parameters + buffers ----
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
        self.print(
            f"Checkpoint: {trained:,}/{total:,} params ({100 * trained / total:.2f}%)"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer: AdamW with linear warmup + cosine annealing ----
    def configure_optimizers(self):
        hp = self.hparams

        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=hp.lr,
            weight_decay=hp.weight_decay,
            betas=(0.9, 0.95),
        )

        warmup_epochs = hp.warmup_epochs
        total_epochs  = hp.max_epochs

        # Linear warmup: LR 10% → 100% over warmup_epochs
        # Proven to stabilize early training for AIDO.Cell LoRA fine-tuning
        warmup_sched = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        # Cosine annealing: LR 100% → eta_min=1e-6 over remaining epochs
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=1e-6,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node3-1-3-2-1: AIDO.Cell-100M (LoRA r=8) + STRING_GNN K=16 2-Head Neighborhood Attn"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=4)
    # SECONDARY CHANGE: global_batch=64 (up from parent's 32); more stable gradients
    parser.add_argument("--global-batch-size",  type=int,   default=64)
    parser.add_argument("--max-epochs",         type=int,   default=200)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--weight-decay",       type=float, default=2e-2)
    parser.add_argument("--lora-r",             type=int,   default=8)
    parser.add_argument("--lora-alpha",         type=int,   default=16)
    parser.add_argument("--lora-dropout",       type=float, default=0.05)
    parser.add_argument("--head-hidden",        type=int,   default=256)
    parser.add_argument("--head-dropout",       type=float, default=0.5)
    parser.add_argument("--warmup-epochs",      type=int,   default=10)
    # TERTIARY CHANGE: patience=20 (up from 15); allow capturing late spikes
    parser.add_argument("--patience",           type=int,   default=20)
    # TERTIARY CHANGE: min_delta=0.0005 (down from 0.001); capture smaller gains
    parser.add_argument("--min-delta",          type=float, default=0.0005)
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--debug_max_step",     type=int,   default=None)
    parser.add_argument("--fast_dev_run",       action="store_true")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = lim_val = args.debug_max_step
        max_steps = args.debug_max_step
        # For debug runs: limit test to the same step count, but ensure
        # both DDP ranks process the same number of batches (ceil division)
        # so that all_gather in on_test_epoch_end doesn't hang.
        # Use 1.0 (full test set) to keep DDP ranks in sync.
        lim_test = 1.0
    else:
        lim_train = lim_val = 1.0
        lim_test = 1.0
        max_steps = -1

    # Gradient accumulation to reach global batch size
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    print(
        f"[Node3-1-3-2-1] n_gpus={n_gpus}, micro_bs={args.micro_batch_size}, "
        f"global_bs={args.global_batch_size}, accumulate={accumulate}, "
        f"n_heads={N_HEADS} (2-head attn)"
    )

    model = AIDOLoRAString2HeadK16Model(
        lora_r        = args.lora_r,
        lora_alpha    = args.lora_alpha,
        lora_dropout  = args.lora_dropout,
        head_hidden   = args.head_hidden,
        head_dropout  = args.head_dropout,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        warmup_epochs = args.warmup_epochs,
        max_epochs    = args.max_epochs,
    )
    datamodule = DEGDataModule(
        batch_size  = args.micro_batch_size,
        num_workers = args.num_workers,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath        = str(output_dir / "checkpoints"),
        filename       = "best-{epoch:03d}-{val/f1:.4f}",
        monitor        = "val/f1",
        mode           = "max",
        save_top_k     = 1,
        save_last      = True,
        auto_insert_metric_name = False,
    )
    early_stop_callback = EarlyStopping(
        monitor    = "val/f1",
        mode       = "max",
        patience   = args.patience,    # 20: allows capturing late-improvement spikes
        min_delta  = args.min_delta,   # 0.0005: capture even smaller gains than parent
        verbose    = True,
    )
    lr_monitor   = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger         = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tensorboard_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy: DDP with find_unused_parameters=True for LoRA + neighborhood attn compatibility
    # Increased timeout to 300s for safety during long test evaluations
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=300))

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = n_gpus,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        max_steps               = max_steps,
        accumulate_grad_batches = accumulate,
        limit_train_batches     = lim_train,
        limit_val_batches       = lim_val,
        limit_test_batches      = lim_test,
        val_check_interval      = (
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps    = 2,
        callbacks               = [checkpoint_callback, early_stop_callback, lr_monitor, progress_bar],
        logger                  = [csv_logger, tensorboard_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,  # gradient clipping for stable LoRA training
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Test on best checkpoint
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Save test score (F1 for FeedbackAgent evaluation)
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        f1_val = test_results[0].get("test/f1", float("nan"))
        loss_val = test_results[0].get("test/loss", float("nan"))
        # Note: final F1 score is computed by EvaluateAgent from test_predictions.tsv
        # Here we write the checkpoint-computed F1 for quick reference
        score_path.write_text(f"test_f1={f1_val:.6f}\ntest_loss={loss_val:.6f}\n")
        print(f"[Node3-1-3-2-1] Test results: {test_results}")


if __name__ == "__main__":
    main()
