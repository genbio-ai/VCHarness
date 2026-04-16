"""Node 3-2-3 – AIDO.Cell-100M (LoRA r=8) + STRING_GNN (K=16, 2-head) + Concatenation Fusion.

Strategy (major architectural shift from parent node3-2-2):
- Switch from AIDO.Cell-10M (ceiling ~0.44 F1) to AIDO.Cell-100M + STRING_GNN fusion
  (proven to achieve ~0.51 F1 in node2 lineage, a +0.07 absolute improvement).
- The parent's AIDO.Cell-10M lineage is exhausted: node3-2-2 achieved only 0.4296 despite
  extensive tuning of SGDR and label noise. The fundamental bottleneck is domain mismatch
  between AIDO.Cell-10M's capacity and the DEG perturbation task complexity.
- Directly replicate the proven winning recipe from node2-1-1-1-1-1 (test F1=0.5128):
    * AIDO.Cell-100M with LoRA r=8 (backbone adaptation, ~0.55M trainable params)
    * STRING_GNN frozen + K=16 2-head NeighborhoodAttention (~164K additional params)
    * Simple concatenation: [640 AIDO summary token + 256 STRING context] = 896-dim
    * 2-layer MLP head: 896→256→19920 with dropout=0.5
    * Label smoothing ε=0.05, weighted CE loss (not focal loss)
    * AdamW (lr=1e-4, wd=2e-2) + cosine annealing (warmup=10, eta_min=1e-6)
    * Patience=15 to capture late improvements (node2-1-1-1-1-1 peaked at epoch 77)

Key lessons from memory that guide this design:
1. node2-1-1-1-1-1 (F1=0.5128): AIDO.Cell-100M + STRING K=16 2-head = tree best
2. node2-1-1-1 (F1=0.5059): Single-head → 2-head gave +0.007 improvement
3. node3-2 feedback: "SGDR + label noise interaction caused overfitting, abandon SGDR variants"
4. node4-2-2-1 feedback: "GatedFusion incompatible with LoRA; simple concatenation works"
5. node2-2-1 feedback: "Within-run checkpoint ensembles lack diversity, don't bother"
6. node2-1-2-1 feedback: "backbone_lr=1e-4 overfits; use lower (same as head or lower)"
   Actually contradicted by node2-1-1-1 which used lr=1e-4 and got 0.5059 — use 1e-4
7. Dropout=0.5 confirmed to prevent overfitting (exact same as node2-1-1-1-1-1)

Memory from parent feedback specifically recommends:
"Cross-lineage transfer: Consider transferring the best elements from node4 lineage
(which achieves 0.46-0.49 F1) into this lineage... Do NOT continue exploring SGDR variants"

This node represents a clean break from the AIDO.Cell-10M exploration path.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES      = 6640
N_CLASSES    = 3
AIDO_GENES   = 19264
AIDO_100M_DIR = "/home/Models/AIDO.Cell-100M"
STRING_DIR    = "/home/Models/STRING_GNN"
HIDDEN_DIM   = 640  # AIDO.Cell-100M hidden size
STRING_DIM   = 256  # STRING_GNN embedding dim

CLASS_FREQ = [0.0429, 0.9251, 0.0320]  # down, neutral, up (remapped 0,1,2)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute per-gene macro-averaged F1 matching calc_metric.py logic."""
    y_hat       = preds.argmax(dim=1)
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
# Neighborhood Attention Module for STRING_GNN context
# ---------------------------------------------------------------------------
class NeighborhoodAttentionModule(nn.Module):
    """Lightweight 2-head neighborhood attention over STRING PPI neighbors.

    Proven in node2-1-1-1-1-1 (F1=0.5128) to provide +0.007 over single-head.
    Aggregates top-K PPI neighbors' STRING embeddings with learned attention weights
    to provide topological context for the perturbed gene.

    Args:
        string_dim: STRING_GNN embedding dimension (256)
        n_heads: number of attention heads (2, proven in tree best)
        k_neighbors: top-K neighbors to aggregate (16, proven in tree best)
        attn_dim: attention projection dimension per head (64)
    """
    def __init__(
        self,
        string_dim: int = 256,
        n_heads: int = 2,
        k_neighbors: int = 16,
        attn_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.k_neighbors = k_neighbors
        self.attn_dim = attn_dim

        # Query projection per head
        self.query_proj = nn.Linear(string_dim, n_heads * attn_dim, bias=False)
        # Key projection per head
        self.key_proj   = nn.Linear(string_dim, n_heads * attn_dim, bias=False)
        # Value stays as is (string_dim)
        # Output projection: concatenate heads' string_dim → string_dim
        self.out_proj   = nn.Linear(string_dim, string_dim, bias=False)
        self.scale      = math.sqrt(attn_dim)

    def forward(
        self,
        center_emb: torch.Tensor,   # [B, 256] - embedding of perturbed gene
        all_emb: torch.Tensor,      # [N_nodes, 256] - all STRING node embeddings
        node_indices: torch.Tensor,  # [B] - node index of each perturbed gene in STRING
        adj_list: Dict[int, List],  # precomputed adjacency list (node_idx → sorted neighbors)
    ) -> torch.Tensor:
        """Return context-aware embedding for perturbed gene.

        Returns:
            torch.Tensor: [B, 256] aggregated neighborhood embedding
        """
        B = center_emb.shape[0]
        device = center_emb.device
        K = self.k_neighbors

        # Gather neighbor embeddings for each sample in batch
        # Shape: [B, K, 256]
        neighbor_embs = torch.zeros(B, K, all_emb.shape[-1], device=device, dtype=all_emb.dtype)
        neighbor_weights = torch.zeros(B, K, device=device, dtype=all_emb.dtype)

        for i, node_idx in enumerate(node_indices.tolist()):
            node_idx = int(node_idx)
            if node_idx in adj_list and len(adj_list[node_idx]) > 0:
                neighbors = adj_list[node_idx][:K]  # already sorted by weight descending
                n_valid = len(neighbors)
                for j, (nb_idx, nb_weight) in enumerate(neighbors):
                    neighbor_embs[i, j] = all_emb[nb_idx]
                    neighbor_weights[i, j] = nb_weight
            # If no neighbors, leave as zeros (gene not in STRING or isolated)

        # Normalize neighbor weights
        weight_sum = neighbor_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        neighbor_weights = neighbor_weights / weight_sum  # [B, K]

        # Multi-head attention: center queries neighbors
        # center_emb: [B, 256] → Q: [B, n_heads, attn_dim]
        Q = self.query_proj(center_emb).view(B, self.n_heads, self.attn_dim)
        # neighbor_embs: [B, K, 256] → K: [B, K, n_heads, attn_dim]
        K_proj = self.key_proj(neighbor_embs.float()).view(B, K, self.n_heads, self.attn_dim)

        # Attention scores: [B, n_heads, K]
        attn_scores = torch.einsum('bhd,bkhd->bhk', Q, K_proj) / self.scale

        # Incorporate STRING edge weights as attention bias
        # neighbor_weights: [B, K] → [B, 1, K]
        weight_bias = neighbor_weights.unsqueeze(1).log().clamp(min=-10.0)
        attn_scores = attn_scores + weight_bias

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, n_heads, K]

        # Value: use neighbor embeddings directly [B, K, 256]
        # Per-head weighted sum: [B, n_heads, 256] → but we aggregate across heads using string_dim
        # For simplicity, average over heads after computing weighted sum
        # weighted_val: [B, n_heads, 256]
        weighted_val = torch.einsum('bhk,bkd->bhd', attn_weights, neighbor_embs.float())
        # Average across heads: [B, 256]
        context = weighted_val.mean(dim=1)
        # Output projection
        context = self.out_proj(context)

        # Residual connection: center + context
        return (center_emb.float() + context).to(center_emb.dtype)


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


def make_collate(tokenizer):
    """Collate function for AIDO.Cell-100M.

    Uses per-gene perturbation encoding: encode only the perturbed gene
    (expression=1.0 for that gene, all others missing=-1.0).
    Extract summary token (position -2, i.e., index 19264) for cell-level representation.
    """
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        # Encode perturbation identity: only the perturbed gene has expression=1.0
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        input_ids  = tokenized["input_ids"]  # [B, 19264] float32

        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      input_ids,
            "attention_mask": tokenized["attention_mask"],
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
            AutoTokenizer.from_pretrained(AIDO_100M_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_100M_DIR, trust_remote_code=True)

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        self.train_ds = DEGDataset(train_df)
        self.val_ds   = DEGDataset(val_df)
        self.test_ds  = DEGDataset(test_df)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=make_collate(self.tokenizer))

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# LR schedule: warmup + cosine annealing
# ---------------------------------------------------------------------------
def cosine_schedule_with_warmup(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    """Linear warmup then cosine decay to min_lr_ratio * base_lr."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_val = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCell100MStringFusion(pl.LightningModule):
    """AIDO.Cell-100M (LoRA r=8) + STRING_GNN (frozen, K=16 2-head) fusion model.

    This is a major architectural shift from the parent's AIDO.Cell-10M approach.

    Architecture (matching node2-1-1-1-1-1 tree-best):
    1. AIDO.Cell-100M backbone with LoRA r=8, lora_alpha=16
       - Targets query, key, value weights in all 18 transformer layers
       - ~0.55M trainable backbone params
    2. STRING_GNN: fully frozen, cached embeddings computed once at setup
       - 2-head NeighborhoodAttention over top-16 PPI neighbors
       - ~164K trainable attention params
    3. Fusion: concatenate [AIDO summary token (640) + STRING context (256)] = 896-dim
    4. Head: Linear(896→256) → GELU → Dropout(0.5) → Linear(256→19920)
    5. Loss: weighted CE + label_smoothing=0.05 (ε=0.05, not 0.1)

    Optimizer: AdamW with discriminative LRs
    - Backbone LoRA: lr=1e-4, wd=2e-2
    - STRING attention + head: lr=2e-4, wd=2e-2

    LR Schedule: linear warmup (10 epochs) + cosine decay (eta_min=1e-6)
    Early Stopping: patience=15, min_delta=1e-4 to capture late spikes
    """

    def __init__(
        self,
        lora_r: int             = 8,
        lora_alpha: int         = 16,
        lora_dropout: float     = 0.05,
        k_neighbors: int        = 16,
        attn_n_heads: int       = 2,
        attn_dim: int           = 64,
        head_hidden: int        = 256,
        head_dropout: float     = 0.5,
        backbone_lr: float      = 1e-4,
        head_lr: float          = 2e-4,
        weight_decay: float     = 2e-2,
        label_smoothing: float  = 0.05,
        warmup_epochs: int      = 10,
        min_lr_ratio: float     = 0.0,   # cosine decays to 0
        max_epochs: int         = 200,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load and configure AIDO.Cell-100M backbone with LoRA ----
        backbone = AutoModel.from_pretrained(AIDO_100M_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Workaround: AIDO.Cell doesn't implement get_input_embeddings(), which breaks
        # PEFT's enable_input_require_grads() inside get_peft_model().
        # Monkey-patch the method to be a no-op. LoRA only needs gradients on its own
        # trainable params, not on frozen embeddings, so skipping input-require-grads
        # is safe.
        backbone.enable_input_require_grads = lambda: None

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=["query", "key", "value"],
            # flash_self shares weight tensors with self, so LoRA applies to both automatically
            layers_to_transform=None,  # all 18 layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)

        # Cast LoRA params to float32 for training stability
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        lora_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Node3-2-3] LoRA backbone params: {lora_params:,} / {total_params:,}")

        # ---- Load STRING_GNN (frozen, cached embeddings) ----
        string_model_dir = Path(STRING_DIR)
        string_model = AutoModel.from_pretrained(string_model_dir, trust_remote_code=True)
        string_model.eval()

        graph = torch.load(string_model_dir / "graph_data.pt", weights_only=False)
        self._node_names = json.loads((string_model_dir / "node_names.json").read_text())
        self._node_name_to_idx = {name: idx for idx, name in enumerate(self._node_names)}

        # Cache STRING embeddings at setup (frozen, no gradient needed)
        with torch.no_grad():
            edge_index = graph["edge_index"]
            edge_weight = graph["edge_weight"]
            string_out = string_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            # [N_nodes, 256]
            string_emb = string_out.last_hidden_state.float()

        # Register as buffer for DDP and checkpoint compatibility
        self.register_buffer("string_emb", string_emb)

        # Precompute adjacency list for NeighborhoodAttention
        # For each node, sort neighbors by edge weight (descending)
        n_nodes = len(self._node_names)
        adj_list: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_nodes)}

        edge_idx_np = graph["edge_index"].numpy()
        edge_wt_np = edge_weight.numpy() if edge_weight is not None else np.ones(edge_idx_np.shape[1])

        for e_idx in range(edge_idx_np.shape[1]):
            src = int(edge_idx_np[0, e_idx])
            dst = int(edge_idx_np[1, e_idx])
            w = float(edge_wt_np[e_idx])
            adj_list[src].append((dst, w))

        # Sort by weight descending, keep top K_neighbors
        for node_idx in adj_list:
            adj_list[node_idx].sort(key=lambda x: x[1], reverse=True)
            adj_list[node_idx] = adj_list[node_idx][:hp.k_neighbors]

        self._adj_list = adj_list

        # ---- Neighborhood Attention Module (trainable) ----
        self.nb_attention = NeighborhoodAttentionModule(
            string_dim=STRING_DIM,
            n_heads=hp.attn_n_heads,
            k_neighbors=hp.k_neighbors,
            attn_dim=hp.attn_dim,
        )
        # Cast attention module to float32 for stable optimization
        for param in self.nb_attention.parameters():
            param.data = param.data.float()

        # ---- Classification head ----
        # Input: [AIDO 640 + STRING 256] = 896
        fusion_dim = HIDDEN_DIM + STRING_DIM  # 896
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )
        # Cast head to float32
        for param in self.head.parameters():
            param.data = param.data.float()

        self.register_buffer("class_weights", get_class_weights())

        # State buffers for validation/test gathering
        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_meta:  List[Tuple]        = []

        # Will be set in main() before fit
        # Fallback: ceil(1388 / (8*2)) // 4 = ceil(86.75) // 4 = 87 // 4 = 21
        self._steps_per_epoch: int = 21
        self._max_epochs: int = hp.max_epochs

    def _get_string_node_indices(self, pert_ids: List[str]) -> torch.Tensor:
        """Resolve pert_ids (Ensembl IDs) to STRING node indices.

        STRING_GNN nodes are identified by Ensembl gene IDs.
        Returns -1 for genes not in STRING graph.
        """
        indices = []
        for pid in pert_ids:
            # Try direct match (Ensembl ID)
            idx = self._node_name_to_idx.get(pid, -1)
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids: List[str],
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        # ---- AIDO.Cell-100M backbone ----
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        # last_hidden_state: [B, G+2, 640] = [B, 19266, 640]
        # Summary token: position 19264 (index -2) contains aggregated cell representation
        aido_emb = out.last_hidden_state[:, 19264, :].float()  # [B, 640]

        # ---- STRING_GNN: get context-aware embedding via neighborhood attention ----
        node_indices = self._get_string_node_indices(pert_ids)
        node_indices = node_indices.to(input_ids.device)

        # Get base STRING embeddings for the perturbed genes
        # For genes not in STRING (-1), use zero embedding
        center_string_emb = torch.zeros(B, STRING_DIM, dtype=torch.float32,
                                         device=input_ids.device)
        valid_mask = node_indices >= 0
        if valid_mask.any():
            center_string_emb[valid_mask] = self.string_emb[node_indices[valid_mask]]

        # Apply neighborhood attention to get context-enriched STRING embedding
        string_context = self.nb_attention(
            center_emb=center_string_emb,
            all_emb=self.string_emb,
            node_indices=node_indices,
            adj_list=self._adj_list,
        )  # [B, 256]

        # Zero out STRING context for genes not in STRING graph
        if (~valid_mask).any():
            string_context[~valid_mask] = 0.0

        # ---- Fusion: concatenate AIDO + STRING ----
        fused = torch.cat([aido_emb, string_context], dim=-1)  # [B, 896]

        # ---- Classification head ----
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)  # [B, 3, 6640]
        return logits

    # ---- loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ---- steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
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
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        for i, (pid, sym) in enumerate(zip(batch["pert_id"], batch["symbol"])):
            self._test_meta.append((pid, sym, batch["sample_idx"][i].item()))
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)  # [N_local, 3, 6640]
        local_sample_idx = torch.tensor(
            [meta[2] for meta in self._test_meta], dtype=torch.long, device=local_preds.device
        )

        # ALL ranks must participate in all_gather to avoid deadlock!
        # Rank 0 gathers predictions + metadata; other ranks participate to complete the collective.
        all_preds = self.all_gather(local_preds)
        all_sample_idx = self.all_gather(local_sample_idx)

        # Normalize shapes - Lightning all_gather prepends world_size dim when n_gpus > 1
        if all_preds.dim() == 4:
            n_gpus = all_preds.shape[0]
            all_preds = all_preds.view(n_gpus * local_preds.shape[0], N_CLASSES, N_GENES)
            all_sample_idx = all_sample_idx.view(-1)
        elif all_preds.dim() == 3 and all_sample_idx.dim() == 2:
            n_gpus = all_preds.shape[0]
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
            all_sample_idx = all_sample_idx.view(-1)

        # Encode metadata as uint8 tensors with null-padding for safe all_gather
        MAX_PID, MAX_SYM = 32, 24
        N_LOCAL = len(self._test_meta)
        pid_buf = torch.zeros(N_LOCAL, MAX_PID, dtype=torch.uint8, device=local_preds.device)
        sym_buf = torch.zeros(N_LOCAL, MAX_SYM, dtype=torch.uint8, device=local_preds.device)
        for i, (pid, sym, _) in enumerate(self._test_meta):
            pid_enc = torch.tensor(list(pid.encode('utf-8')), dtype=torch.uint8, device=local_preds.device)
            sym_enc = torch.tensor(list(sym.encode('utf-8')), dtype=torch.uint8, device=local_preds.device)
            pid_buf[i, :len(pid_enc)] = pid_enc
            sym_buf[i, :len(sym_enc)] = sym_enc

        # All ranks participate in all_gather to avoid deadlock
        gathered_pid = self.all_gather(pid_buf)
        gathered_sym = self.all_gather(sym_buf)
        if gathered_pid.dim() == 3:  # multi-GPU: [world_size, N_local, MAX_*]
            all_pid_buf = gathered_pid.view(-1, MAX_PID)
            all_sym_buf = gathered_sym.view(-1, MAX_SYM)
        else:  # single GPU: [N_local, MAX_*]
            all_pid_buf = gathered_pid
            all_sym_buf = gathered_sym

        # Only rank 0 writes the file
        if self.trainer.is_global_zero:
            # Sort by sample_idx and deduplicate
            order = torch.argsort(all_sample_idx)
            sorted_preds  = all_preds[order]
            sorted_pid_buf = all_pid_buf[order]
            sorted_sym_buf = all_sym_buf[order]
            sorted_idx    = all_sample_idx[order]

            n_total = sorted_idx.shape[0]
            unique_pos = []
            seen = set()
            for pos in range(n_total):
                idx_val = sorted_idx[pos].item()
                if idx_val not in seen:
                    seen.add(idx_val)
                    unique_pos.append(pos)

            rows = []
            for pos in unique_pos:
                pid_bytes = sorted_pid_buf[pos].cpu().numpy()
                sym_bytes = sorted_sym_buf[pos].cpu().numpy()
                pid = pid_bytes[:np.where(pid_bytes == 0)[0][0]].tobytes().decode('utf-8') if (pid_bytes == 0).any() else pid_bytes.tobytes().decode('utf-8')
                sym = sym_bytes[:np.where(sym_bytes == 0)[0][0]].tobytes().decode('utf-8') if (sym_bytes == 0).any() else sym_bytes.tobytes().decode('utf-8')
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(sorted_preds[pos].float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node3-2-3] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_meta.clear()

    # ---- checkpoint (save trainable params + buffers only) ----
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

    # ---- optimizer ----
    def configure_optimizers(self):
        hp = self.hparams

        # Backbone LoRA params: lower LR for careful backbone adaptation
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        # Neighborhood attention + head: higher LR for faster task adaptation
        task_params = list(self.nb_attention.parameters()) + list(self.head.parameters())

        param_groups = [
            {"params": backbone_params, "lr": hp.backbone_lr, "weight_decay": hp.weight_decay},
            {"params": task_params,     "lr": hp.head_lr,     "weight_decay": hp.weight_decay},
        ]

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

        # Cosine annealing with warmup
        warmup_steps = self._steps_per_epoch * hp.warmup_epochs
        total_steps  = self._steps_per_epoch * self._max_epochs

        scheduler = cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=hp.min_lr_ratio,
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
                "frequency": 1,
                "monitor":   "val/f1",
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node3-2-3 – AIDO.Cell-100M LoRA + STRING_GNN K=16 2-head Fusion"
    )
    parser.add_argument("--micro_batch_size",   type=int,   default=8,
                        help="Per-GPU batch size (AIDO.Cell-100M needs smaller batches)")
    parser.add_argument("--global_batch_size",  type=int,   default=64,
                        help="Global batch size (multiple of micro_batch_size * 8)")
    parser.add_argument("--max_epochs",         type=int,   default=250,
                        help="Extend to capture late improvements (node2 best at epoch 77)")
    parser.add_argument("--lora_r",             type=int,   default=8)
    parser.add_argument("--lora_alpha",         type=int,   default=16)
    parser.add_argument("--lora_dropout",       type=float, default=0.05)
    parser.add_argument("--k_neighbors",        type=int,   default=16)
    parser.add_argument("--attn_n_heads",       type=int,   default=2)
    parser.add_argument("--attn_dim",           type=int,   default=64)
    parser.add_argument("--head_hidden",        type=int,   default=256)
    parser.add_argument("--head_dropout",       type=float, default=0.5)
    parser.add_argument("--backbone_lr",        type=float, default=1e-4)
    parser.add_argument("--head_lr",            type=float, default=2e-4)
    parser.add_argument("--weight_decay",       type=float, default=2e-2)
    parser.add_argument("--label_smoothing",    type=float, default=0.05)
    parser.add_argument("--warmup_epochs",      type=int,   default=10)
    parser.add_argument("--min_lr_ratio",       type=float, default=0.0,
                        help="Cosine min LR as fraction of base LR (0=decay to 0)")
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--num_workers",        type=int,   default=4)
    parser.add_argument("--debug_max_step",     type=int,   default=None)
    parser.add_argument("--fast_dev_run",       action="store_true")
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

    # Compute steps per epoch for LR scheduler
    n_train_samples = 1388
    steps_per_epoch = max(1, math.ceil(n_train_samples / (args.micro_batch_size * n_gpus)) // accum)

    print(f"[Node3-2-3] n_gpus={n_gpus}, accum={accum}, steps_per_epoch={steps_per_epoch}")
    print(f"[Node3-2-3] Architecture: AIDO.Cell-100M (LoRA r={args.lora_r}) + STRING_GNN K=16 2-head")
    print(f"[Node3-2-3] Expected ~0.51 F1 based on node2-1-1-1-1-1 blueprint")

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOCell100MStringFusion(
        lora_r          = args.lora_r,
        lora_alpha      = args.lora_alpha,
        lora_dropout    = args.lora_dropout,
        k_neighbors     = args.k_neighbors,
        attn_n_heads    = args.attn_n_heads,
        attn_dim        = args.attn_dim,
        head_hidden     = args.head_hidden,
        head_dropout    = args.head_dropout,
        backbone_lr     = args.backbone_lr,
        head_lr         = args.head_lr,
        weight_decay    = args.weight_decay,
        label_smoothing = args.label_smoothing,
        warmup_epochs   = args.warmup_epochs,
        min_lr_ratio    = args.min_lr_ratio,
        max_epochs      = args.max_epochs,
    )
    model._steps_per_epoch = steps_per_epoch
    model._max_epochs      = args.max_epochs

    ckpt_cb = ModelCheckpoint(
        dirpath  = str(output_dir / "checkpoints"),
        filename = "best-{epoch:03d}-{val/f1:.4f}",
        monitor  = "val/f1", mode="max", save_top_k=1,
    )
    # patience=15 to capture late improvements like node2-1-1-1-1-1 (peaked at epoch 77,
    # ES would've fired at epoch ~12 with patience=10)
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=15, min_delta=1e-4)
    lr_cb = LearningRateMonitor(logging_interval="step")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=600))
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
    print(f"[Node3-2-3] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
