"""Node 3-1-1-3: AIDO.Cell-100M LoRA (r=8) + STRING_GNN K=16 2-head NbAttn + 2-layer MLP Head.

This node is an improvement of the parent (node3-1-1), which reached F1=0.4325 using
AIDO.Cell-10M QKV+Output fine-tuning + Muon optimizer — hitting the fundamental ceiling
of the AIDO.Cell-10M paradigm (~0.43 F1).

Key architectural change: Upgrade from AIDO.Cell-10M (QKV fine-tuning) to the
best-in-tree architecture: AIDO.Cell-100M (LoRA r=8) + frozen STRING_GNN (K=16 2-head
neighborhood attention), which achieved F1=0.5128 in node2-1-1-1-1-1.

This is distinct from both siblings:
- node3-1-1-1 (sibling 1): STRING_GNN full fine-tune + AIDO.Cell-10M QKV → F1=0.3989 (FAILED)
- node3-1-1-2 (sibling 2): Frozen STRING_GNN K=16 NbAttn + AIDO.Cell-10M QKV → F1=0.3864 (FAILED)

Both siblings are bottlenecked by AIDO.Cell-10M's fundamental capability ceiling at ~0.43.
This node breaks through by using AIDO.Cell-100M which provides 10× richer transcriptomic
representations, combined with the proven STRING_GNN neighborhood aggregation for PPI topology.

Design decisions based on cross-tree memory synthesis:
1. AIDO.Cell-100M LoRA r=8 (summary token extraction, 640-dim): The key upgrade from -10M.
   node2-1-1-1-1-1 proved this combination reaches F1=0.5128 — 19% above AIDO.Cell-10M ceiling.
2. Frozen STRING_GNN + K=16 2-head neighborhood attention: Provides complementary PPI signal.
   The frozen approach avoids overfitting on 1,388 samples (proven superior to fine-tuning).
3. Concatenation fusion (640+256=896-dim): Simple and proven effective in the best-in-tree.
4. 2-layer MLP head (896→256→19920), dropout=0.5: Exactly matches the best-in-tree recipe.
5. Single lr=1e-4 for all trainable params: Proven in best-in-tree. Discriminative LR
   (backbone_lr=5e-5 vs head_lr) failed in node2-1-2, node2-2-1-1-1, node3-1-3-3-2, etc.
6. global_batch_size=32: CRITICAL. node1-3-3-3 showed that gbs=256 with this architecture
   yields only ~6 optimizer steps/epoch vs gbs=32's ~44 steps/epoch, causing severe
   underconvergence and F1=0.4446 vs the target 0.51+.
7. CosineAnnealingLR (T_max=200, warmup=10, eta_min=1e-6): Matches best-in-tree recipe.
8. patience=20, max_epochs=300: The best-in-tree had its late spike at epoch 77 (val/f1=0.5128),
   patience=10 barely captured it. patience=20 ensures we don't miss late improvements.
9. label_smoothing=0.05, weight_decay=2e-2: Proven in best-in-tree.
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
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES     = 6640
N_CLASSES   = 3
AIDO_GENES  = 19264

AIDO_MODEL_DIR   = "/home/Models/AIDO.Cell-100M"
STRING_MODEL_DIR = "/home/Models/STRING_GNN"
AIDO_HIDDEN_DIM  = 640      # AIDO.Cell-100M hidden size
STRING_DIM       = 256      # STRING_GNN embedding dim
FUSION_DIM       = AIDO_HIDDEN_DIM + STRING_DIM  # 896

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
    """Sqrt-inverse-frequency class weights for class imbalance handling."""
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
# STRING_GNN K=16 2-head Neighborhood Attention Aggregator
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Aggregate top-K PPI neighbors via 2-head attention.

    This is the proven K=16 2-head neighborhood attention from node2-1-1-1-1-1
    (F1=0.5128), which delivered +0.0524 improvement over the parent node.

    Architecture:
    - Project center gene and neighbor genes to 2 attention head subspaces
    - Compute attention weights (with STRING confidence weighting)
    - Concatenate 2-head outputs → project back to STRING_DIM
    - Gate between center and aggregated context

    Args:
        embed_dim: Input/output dimension (256 from STRING_GNN)
        num_heads: Number of attention heads (default: 2)
        attn_dim: Per-head attention dimension (default: 64)
        k: Number of top-K PPI neighbors to use (default: 16)
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 2,
                 attn_dim: int = 64, k: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dim  = attn_dim
        self.k         = k

        total_attn_dim = num_heads * attn_dim
        self.q_proj = nn.Linear(embed_dim, total_attn_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, total_attn_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, total_attn_dim, bias=False)
        self.out_proj = nn.Linear(total_attn_dim, embed_dim, bias=False)
        self.gate = nn.Linear(embed_dim * 2, 1, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        center_embs: torch.Tensor,       # [B, embed_dim]
        all_embs: torch.Tensor,          # [N_nodes, embed_dim]
        neighbor_indices: torch.Tensor,  # [B, K] — top-K neighbor node indices
        neighbor_weights: torch.Tensor,  # [B, K] — STRING confidence weights
    ) -> torch.Tensor:
        """Returns aggregated + gated embedding of shape [B, embed_dim]."""
        B, K = neighbor_indices.shape

        # Gather neighbor embeddings: [B, K, embed_dim]
        nb_embs = all_embs[neighbor_indices.view(-1)].view(B, K, self.embed_dim)

        # Multi-head attention projection
        Q = self.q_proj(center_embs).view(B, 1, self.num_heads, self.attn_dim)   # [B, 1, H, d]
        Ke = self.k_proj(nb_embs).view(B, K, self.num_heads, self.attn_dim)       # [B, K, H, d]
        V = self.v_proj(nb_embs).view(B, K, self.num_heads, self.attn_dim)        # [B, K, H, d]

        # Compute attention scores [B, H, 1, K]
        Q = Q.permute(0, 2, 1, 3)    # [B, H, 1, d]
        Ke = Ke.permute(0, 2, 1, 3)  # [B, H, K, d]
        V = V.permute(0, 2, 1, 3)    # [B, H, K, d]

        scale = float(self.attn_dim) ** -0.5
        attn_scores = torch.matmul(Q, Ke.transpose(-2, -1)) * scale  # [B, H, 1, K]

        # Add log-confidence weighting (avoid -inf for zero weights)
        log_conf = torch.log(neighbor_weights.clamp(min=1e-6) + 1e-6)  # [B, K]
        log_conf = log_conf.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K]
        attn_scores = attn_scores + log_conf

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, 1, K]

        # Aggregate context: [B, H, 1, d] → [B, H*d]
        context = torch.matmul(attn_weights, V)  # [B, H, 1, d]
        context = context.squeeze(2)  # [B, H, d]
        context = context.reshape(B, -1)  # [B, H*d]
        context = self.out_proj(context)  # [B, embed_dim]

        # Gate between center embedding and aggregated context
        gate_input = torch.cat([center_embs, context], dim=-1)  # [B, 2*embed_dim]
        gate = torch.sigmoid(self.gate(gate_input))  # [B, 1]
        aggregated = gate * center_embs + (1.0 - gate) * context  # [B, embed_dim]

        return self.norm(aggregated)


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
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        input_ids  = tokenized["input_ids"]  # [B, 19264] float32
        # Find position in vocab for each sample's perturbed gene
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),
            torch.zeros(len(batch), dtype=torch.long),
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
# Model
# ---------------------------------------------------------------------------
class AIDO100MStringGNNModel(pl.LightningModule):
    """AIDO.Cell-100M (LoRA r=8) + Frozen STRING_GNN (K=16 2-head NbAttn) + 2-layer MLP head.

    This is a direct implementation of the best-in-tree architecture (node2-1-1-1-1-1, F1=0.5128),
    applied as an improvement over the parent node3-1-1 (AIDO.Cell-10M, F1=0.4325).

    Architecture:
    - AIDO.Cell-100M: LoRA r=8 on QKV across all 18 layers. Summary token at position 19264
      extracted for 640-dim perturbation-aware embedding.
    - STRING_GNN: Frozen, pre-computed node embeddings. K=16 2-head neighborhood attention
      aggregates PPI context into a 256-dim embedding per perturbed gene.
    - Fusion: Concatenation of [640-dim AIDO + 256-dim STRING] = 896-dim.
    - Head: 2-layer MLP (896→256→19920) with dropout=0.5.

    Key hyperparameters (matching best-in-tree):
    - lr=1e-4 for ALL trainable parameters (LoRA + neighborhood attention + head)
    - global_batch_size=32 for sufficient optimizer steps (~44/epoch on 1388 samples)
    - weight_decay=2e-2, label_smoothing=0.05
    - CosineAnnealingLR T_max=200 with 10-epoch warmup
    - patience=20 to capture late-spike improvements (best-in-tree peaked at epoch 77)
    """

    def __init__(
        self,
        lora_r: int             = 8,
        lora_alpha: int         = 16,
        k_neighbors: int        = 16,
        num_attn_heads: int     = 2,
        attn_dim: int           = 64,
        head_hidden: int        = 256,
        head_dropout: float     = 0.5,
        lr: float               = 1e-4,
        weight_decay: float     = 2e-2,
        label_smoothing: float  = 0.05,
        t_max: int              = 200,
        eta_min: float          = 1e-6,
        warmup_epochs: int      = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load AIDO.Cell-100M with LoRA ----
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Load backbone in bf16
        backbone = AutoModel.from_pretrained(AIDO_MODEL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        # Apply LoRA to Q/K/V in all 18 transformer layers
        # NOTE: LoRA must be applied BEFORE enabling gradient checkpointing.
        # Enabling GC before get_peft_model causes PEFT to call enable_input_require_grads()
        # which calls get_input_embeddings() — not implemented in AIDO.Cell (NotImplementedError).
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
            # flash_self shares the same weight tensors as self, so LoRA on self.query
            # automatically applies to both paths (see AIDO.Cell skill documentation)
            layers_to_transform=None,  # Apply to all 18 layers
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_cfg)

        # Enable gradient checkpointing AFTER LoRA (per AIDO.Cell skill template)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        trainable_backbone = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_backbone = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Node3-1-1-3] AIDO.Cell-100M LoRA: {trainable_backbone:,} / {total_backbone:,} params")

        # ---- Load STRING_GNN (frozen) ----
        import json as _json
        string_model = AutoModel.from_pretrained(STRING_MODEL_DIR, trust_remote_code=True)
        string_model.eval()
        for param in string_model.parameters():
            param.requires_grad = False

        graph = torch.load(f"{STRING_MODEL_DIR}/graph_data.pt")
        _edge_index_cpu = graph["edge_index"]
        _edge_weight_cpu = graph.get("edge_weight", None)

        # Pre-compute frozen STRING_GNN node embeddings (cached as buffer)
        # Run inference on CPU, then register as buffer (will move to GPU with model)
        with torch.no_grad():
            out = string_model(
                edge_index=_edge_index_cpu,
                edge_weight=_edge_weight_cpu,
            )
            frozen_embs = out.last_hidden_state.detach().float()  # [18870, 256]
        self.register_buffer("string_embs", frozen_embs)
        print(f"[Node3-1-1-3] STRING_GNN frozen embeddings: {frozen_embs.shape}")

        # Build gene-ID → STRING_GNN node index mapping
        node_names = _json.loads(Path(f"{STRING_MODEL_DIR}/node_names.json").read_text())
        # node_names[i] is the Ensembl gene ID for node i
        self._string_node_lookup: Dict[str, int] = {name: i for i, name in enumerate(node_names)}
        print(f"[Node3-1-1-3] STRING_GNN node count: {len(node_names)}")

        # Pre-build adjacency list for neighborhood selection
        # For each node, find top-K neighbors by edge weight
        n_nodes = len(node_names)
        self._k = hp.k_neighbors
        self._build_neighbor_lookup(_edge_index_cpu, _edge_weight_cpu, n_nodes)

        # ---- Neighborhood Attention Aggregator ----
        self.nb_attn = NeighborhoodAttentionAggregator(
            embed_dim=STRING_DIM,
            num_heads=hp.num_attn_heads,
            attn_dim=hp.attn_dim,
            k=hp.k_neighbors,
        )
        # Cast to float32 for stable optimization
        for param in self.nb_attn.parameters():
            param.data = param.data.float()

        # ---- Classification Head ----
        # 2-layer MLP: [896 → 256 → 19920] with dropout=0.5
        # This exactly matches the best-in-tree recipe (node2-1-1-1-1-1)
        self.head = nn.Sequential(
            nn.Linear(FUSION_DIM, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )
        # Cast to float32 for stable optimization
        for param in self.head.parameters():
            param.data = param.data.float()

        # ---- Loss: label-smoothed CE + class weights ----
        class_weights = get_class_weights()
        self.register_buffer("class_weights", class_weights)
        self.label_smoothing = hp.label_smoothing

        # Storage for validation and test accumulation
        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_tgts:  List[torch.Tensor] = []
        self._test_meta:  List[Tuple]        = []

    def _build_neighbor_lookup(self, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor], n_nodes: int) -> None:
        """Build top-K neighbor lookup tables (buffers) from graph edge data."""
        K = self._k
        # edge_index: [2, E] — CPU tensor
        # edge_weight: [E] or None — CPU tensor

        # Build adjacency: src → list of (dst, weight)
        adj: Dict[int, List[Tuple[int, float]]] = {}
        E = edge_index.shape[1]
        for e_idx in range(E):
            src = edge_index[0, e_idx].item()
            dst = edge_index[1, e_idx].item()
            w = float(edge_weight[e_idx].item()) if edge_weight is not None else 1.0
            if src not in adj:
                adj[src] = []
            adj[src].append((dst, w))

        # Build top-K lookup arrays
        topk_indices = torch.zeros(n_nodes, K, dtype=torch.long)
        topk_weights = torch.zeros(n_nodes, K, dtype=torch.float32)

        for node_idx in range(n_nodes):
            if node_idx in adj and len(adj[node_idx]) > 0:
                neighbors = adj[node_idx]
                # Sort by weight descending, take top K
                neighbors.sort(key=lambda x: x[1], reverse=True)
                neighbors = neighbors[:K]
                for j, (nb_idx, nb_w) in enumerate(neighbors):
                    topk_indices[node_idx, j] = nb_idx
                    topk_weights[node_idx, j] = nb_w
                # If fewer than K neighbors, pad with self-loop
                if len(neighbors) < K:
                    for j in range(len(neighbors), K):
                        topk_indices[node_idx, j] = node_idx
                        topk_weights[node_idx, j] = 0.0
            else:
                # No neighbors: pad with self (zero weights)
                topk_indices[node_idx, :] = node_idx
                topk_weights[node_idx, :] = 0.0

        self.register_buffer("topk_indices", topk_indices)  # [N_nodes, K]
        self.register_buffer("topk_weights", topk_weights)  # [N_nodes, K]
        print(f"[Node3-1-1-3] Built top-{K} neighbor lookup for {n_nodes} nodes")

    def _get_string_embeddings(self, pert_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Look up STRING_GNN embeddings and neighbor info for a batch of perturbation IDs.

        Returns:
            center_embs: [B, 256] frozen STRING embeddings for perturbed genes
            neighbor_indices: [B, K] indices into string_embs
            neighbor_weights: [B, K] STRING confidence weights
        """
        B = len(pert_ids)
        K = self._k
        device = self.string_embs.device

        node_indices = []
        for pid in pert_ids:
            # Strip version suffix if present (e.g. ENSG00000000003.14 → ENSG00000000003)
            pid_clean = pid.split(".")[0]
            idx = self._string_node_lookup.get(pid_clean, -1)
            node_indices.append(idx)

        # Build batch lookup tensors
        valid_indices = torch.tensor(
            [i if i >= 0 else 0 for i in node_indices],
            dtype=torch.long, device=device
        )

        center_embs = self.string_embs[valid_indices].clone()  # [B, 256] — detached copy

        # Get top-K neighbors and weights
        neighbor_indices = self.topk_indices[valid_indices]  # [B, K]
        neighbor_weights = self.topk_weights[valid_indices]  # [B, K]

        # Zero out embeddings for genes not in STRING vocab
        not_in_vocab = torch.tensor(
            [idx < 0 for idx in node_indices],
            dtype=torch.bool, device=device
        )
        if not_in_vocab.any():
            center_embs = center_embs.clone()
            center_embs[not_in_vocab] = 0.0

        return center_embs, neighbor_indices, neighbor_weights

    # ---- forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_positions: torch.Tensor,
        pert_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        # ---- AIDO.Cell-100M LoRA forward ----
        # Extract summary token at position 19264 (the first appended summary position)
        # The AIDO.Cell tokenizer appends 2 summary tokens after all 19264 gene positions.
        # Position 19264 is the first summary token (0-indexed in the 19266-length output).
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last_hidden_state: [B, 19266, 640]
        # Use position 19264 as the summary token for the whole-transcriptome perturbation embedding
        aido_emb = out.last_hidden_state[:, 19264, :].float()  # [B, 640]

        # ---- STRING_GNN Neighborhood Attention ----
        if pert_ids is not None:
            center_embs, neighbor_indices, neighbor_weights = self._get_string_embeddings(pert_ids)
            # Apply 2-head neighborhood attention aggregation
            string_emb = self.nb_attn(
                center_embs.float(),
                self.string_embs.float(),
                neighbor_indices,
                neighbor_weights,
            )  # [B, 256]
        else:
            # Fallback: zero STRING embeddings (for fast_dev_run or unit tests)
            string_emb = torch.zeros(B, STRING_DIM, device=aido_emb.device, dtype=torch.float32)

        # ---- Concatenation Fusion ----
        fused = torch.cat([aido_emb, string_emb], dim=1)  # [B, 896]

        # ---- Classification Head ----
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)  # [B, 3, 6640]
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
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch.get("pert_id")
        )
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch.get("pert_id")
        )
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits.float(), dim=1).detach()
            self._val_preds.append(probs.cpu())
            self._val_tgts.append(batch["labels"].detach().cpu())
            self._val_idx.append(batch["sample_idx"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        local_idx   = torch.cat(self._val_idx,   0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        # Move to device for gathering
        local_preds = local_preds.to(self.device)
        local_tgts  = local_tgts.to(self.device)
        local_idx   = local_idx.to(self.device)

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        # Deduplicate and sort by sample index
        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        # sync_dist=False is correct: f1 is already computed from globally gathered data
        # All ranks log the same value (all_gather ensures full dataset is available on every rank)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["input_ids"], batch["attention_mask"],
            batch["gene_positions"], batch.get("pert_id")
        )
        probs = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs.cpu())
        for i, (pid, sym) in enumerate(zip(batch["pert_id"], batch["symbol"])):
            self._test_meta.append((pid, sym, batch["sample_idx"][i].item()))
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)
            self._test_tgts.append(batch["labels"].detach().cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0).to(self.device)
        all_preds   = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)

        local_idx_list = torch.tensor(
            [m[2] for m in self._test_meta], dtype=torch.long, device=self.device
        )
        all_idx = self.all_gather(local_idx_list).view(-1)

        # Compute test F1 if targets available
        if self._test_tgts:
            local_tgts = torch.cat(self._test_tgts, 0).to(self.device)
            all_tgts   = self.all_gather(local_tgts).view(-1, N_GENES)
            order  = torch.argsort(all_idx)
            s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
            mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
            test_f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=True)

        # Gather metadata from all ranks
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

            # Deduplicate: keep only unique sample indices
            seen_indices = set()
            rows = []
            order = torch.argsort(all_idx)
            for i in order.tolist():
                idx_val = all_idx[i].item()
                if idx_val in seen_indices:
                    continue
                seen_indices.add(idx_val)
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
            print(f"[Node3-1-1-3] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_tgts.clear()
        self._test_meta.clear()

    # ---- Checkpoint: save only trainable parameters ----
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
        total_bufs = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Checkpoint: {trained}/{total} params ({100*trained/total:.2f}%), "
            f"plus {total_bufs} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer ----
    def configure_optimizers(self):
        hp = self.hparams

        # Single learning rate for ALL trainable parameters (proven in best-in-tree)
        # Discriminative LR (backbone_lr < head_lr) failed in multiple nodes
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=hp.lr,
            betas=(0.9, 0.999),
            weight_decay=hp.weight_decay,
        )

        # Linear warmup + cosine annealing (matching best-in-tree recipe)
        def lr_lambda(epoch):
            if epoch < hp.warmup_epochs:
                return float(epoch + 1) / float(hp.warmup_epochs)
            progress = min(1.0, float(epoch - hp.warmup_epochs) / float(max(1, hp.t_max - hp.warmup_epochs)))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            floor = hp.eta_min / hp.lr
            return max(floor, cosine_decay)

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
        description="Node3-1-1-3: AIDO.Cell-100M LoRA + Frozen STRING_GNN K=16 2-head"
    )
    # Batch sizes: global_batch_size=32 is CRITICAL for sufficient optimizer steps
    # (node1-3-3-3 proved that gbs=256 with this architecture yields only ~6 steps/epoch
    # vs gbs=32's ~44 steps/epoch, causing severe underconvergence and F1=0.4446 vs 0.51+)
    parser.add_argument("--micro-batch-size",   type=int,   default=4)
    parser.add_argument("--global-batch-size",  type=int,   default=32)
    parser.add_argument("--max-epochs",         type=int,   default=300)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--weight-decay",       type=float, default=2e-2)
    parser.add_argument("--lora-r",             type=int,   default=8)
    parser.add_argument("--lora-alpha",         type=int,   default=16)
    parser.add_argument("--k-neighbors",        type=int,   default=16)
    parser.add_argument("--num-attn-heads",     type=int,   default=2)
    parser.add_argument("--attn-dim",           type=int,   default=64)
    parser.add_argument("--head-hidden",        type=int,   default=256)
    parser.add_argument("--head-dropout",       type=float, default=0.5)
    parser.add_argument("--label-smoothing",    type=float, default=0.05)
    parser.add_argument("--t-max",              type=int,   default=200)
    parser.add_argument("--eta-min",            type=float, default=1e-6)
    parser.add_argument("--warmup-epochs",      type=int,   default=10)
    parser.add_argument("--patience",           type=int,   default=20)
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

    # accumulate_grad_batches = global_batch_size / (micro_batch_size * n_gpus)
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    print(f"[Node3-1-1-3] n_gpus={n_gpus}, micro_batch={args.micro_batch_size}, "
          f"global_batch={args.global_batch_size}, accum={accum}")

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDO100MStringGNNModel(
        lora_r          = args.lora_r,
        lora_alpha      = args.lora_alpha,
        k_neighbors     = args.k_neighbors,
        num_attn_heads  = args.num_attn_heads,
        attn_dim        = args.attn_dim,
        head_hidden     = args.head_hidden,
        head_dropout    = args.head_dropout,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        label_smoothing = args.label_smoothing,
        t_max           = args.t_max,
        eta_min         = args.eta_min,
        warmup_epochs   = args.warmup_epochs,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
        auto_insert_metric_name=False,
    )
    # patience=20: captures late-spike improvements (best-in-tree peaked at epoch 77)
    es_cb = EarlyStopping(monitor="val/f1", mode="max",
                          patience=args.patience, min_delta=5e-4)
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

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node3-1-1-3] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
