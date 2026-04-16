"""Node 1-1-1-1-3 – Frozen STRING_GNN + K=16 Neighborhood Attention + GenePriorBias.

Combines the two independently-validated innovations from sibling nodes:
  1. K=16 PPI Neighborhood Context Aggregation (node1-1-1-1-1, Test F1=0.4846, +0.010 vs parent)
     Enriches perturbation embeddings by attending over top-16 STRING PPI neighbors,
     with PPI edge-confidence priors and learnable gating. attn_dim=32 (reduced from
     sibling1's 64) to limit overfitting capacity in the combined model.
  2. GenePriorBias calibration (node1-1-1-1-2, Test F1=0.4814, +0.007 vs parent)
     Learnable [3, 6640] per-gene-per-class logit bias, zero-gradient for 50 epochs
     then activated via a persistent register_buffer to ensure correct test inference.

These innovations address orthogonal bottlenecks:
  - Neighborhood attention: enriches the *input* perturbation representation (PPI context)
  - GenePriorBias: calibrates the *output* logits (gene-specific DEG tendencies)

Additional improvements over parent (node1-1-1-1, F1=0.4746):
  - weight_decay: 2e-2 → 3e-2 (consensus from both siblings + parent feedback)
  - dropout: 0.35 → 0.40 (consensus from both siblings + parent feedback)
  - T_max: 100 → 150 (more LR budget during post-bias-activation phase)
  - patience: 7 → 25 (allow GenePriorBias to learn after activation at epoch 50)
  - max_epochs: 150 → 250 (safety margin for GenePriorBias convergence)
  - FIXED: test_predictions.tsv saved in correct [3, 6640] format (parent bug: wrote [6640, 3])

Expected Test F1: ~0.488–0.492 (additive benefit of neighborhood + gene prior calibration)
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
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3

# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

STRING_DIM = 256  # STRING_GNN hidden dimension


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights for weighted cross-entropy."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def load_string_gnn_mapping() -> Dict[str, int]:
    """Load STRING_GNN node_names.json → Ensembl-ID to node-index mapping."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    return {name: idx for idx, name in enumerate(node_names)}


def compute_topk_neighbors(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    n_nodes: int,
    K: int = 16,
) -> tuple:
    """Pre-compute top-K neighbor indices and softmax-normalized weights for each node.

    Args:
        edge_index: [2, E] long tensor of directed edges
        edge_weight: [E] float tensor of edge confidence scores
        n_nodes: total number of nodes in the graph
        K: number of top neighbors to retain per node

    Returns:
        topk_neighbors: [n_nodes, K] long tensor
        topk_weights:   [n_nodes, K] float tensor (softmax-normalized per node)
    """
    src = edge_index[0]   # [E]
    dst = edge_index[1]   # [E]
    wts = edge_weight     # [E]

    # Sort all edges by source node index (stable)
    perm = torch.argsort(src, stable=True)
    src_sorted = src[perm]
    dst_sorted = dst[perm]
    wts_sorted = wts[perm]
    E = src.shape[0]

    # Find start index for each node's outgoing edges using searchsorted
    node_ids = torch.arange(n_nodes, dtype=torch.long)
    boundaries = torch.searchsorted(src_sorted.contiguous(), node_ids.contiguous())
    # Append sentinel for the final node
    boundaries = torch.cat([boundaries, torch.tensor([E], dtype=torch.long)])

    topk_neighbors = torch.zeros(n_nodes, K, dtype=torch.long)
    topk_weights   = torch.full((n_nodes, K), 1.0 / K, dtype=torch.float32)

    for i in range(n_nodes):
        start = int(boundaries[i].item())
        end   = int(boundaries[i + 1].item())

        if start == end:
            # No outgoing edges: use self-loop padding
            topk_neighbors[i] = torch.full((K,), i, dtype=torch.long)
            topk_weights[i]   = torch.full((K,), 1.0 / K, dtype=torch.float32)
            continue

        node_dsts = dst_sorted[start:end]
        node_wts  = wts_sorted[start:end]
        n_nbrs    = node_dsts.shape[0]

        if n_nbrs > K:
            # Take top-K by confidence weight
            top_idx  = torch.argsort(node_wts, descending=True)[:K]
            node_dsts = node_dsts[top_idx]
            node_wts  = node_wts[top_idx]
            n_nbrs    = K

        # Pad to K if fewer than K neighbors (repeat last neighbor)
        if n_nbrs < K:
            pad = K - n_nbrs
            node_dsts = torch.cat([node_dsts, node_dsts[-1:].expand(pad)])
            node_wts  = torch.cat([node_wts,  node_wts[-1:].expand(pad)])

        topk_neighbors[i] = node_dsts
        topk_weights[i]   = torch.softmax(node_wts.float(), dim=0)

    return topk_neighbors, topk_weights


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0, 1, 2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)           # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)           # [N, G]
        is_pred = (y_hat == c)             # [N, G]
        present = is_true.any(dim=0)       # [G]

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
        f1_per_gene += f1_c * present.float()
        n_present   += present.float()

    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    """K562 DEG prediction dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_map: Dict[str, int],
    ) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()

        # STRING_GNN node index for each sample (-1 means not in STRING)
        self.string_node_indices = torch.tensor(
            [string_map.get(p, -1) for p in self.pert_ids], dtype=torch.long
        )

        has_label = "label" in df.columns and df["label"].notna().all()
        if has_label:
            self.labels = [
                torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
                for row in df["label"].tolist()
            ]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx":      idx,
            "pert_id":         self.pert_ids[idx],
            "symbol":          self.symbols[idx],
            "string_node_idx": self.string_node_indices[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]  # [G] in {0, 1, 2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx":      torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":         [b["pert_id"]  for b in batch],
        "symbol":          [b["symbol"]   for b in batch],
        "string_node_idx": torch.stack([b["string_node_idx"] for b in batch]),
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.string_map: Optional[Dict[str, int]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.string_map is None:
            self.string_map = load_string_gnn_mapping()

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df, self.string_map)
        self.val_ds   = DEGDataset(val_df,   self.string_map)
        self.test_ds  = DEGDataset(test_df,  self.string_map)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
            sampler=SequentialSampler(self.test_ds),
        )


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Lightweight attention-based PPI neighborhood aggregation.

    Given a center node embedding and its top-K neighbor embeddings,
    computes an attention-weighted context that blends with the center
    via a learnable gate. The attention scores incorporate PPI edge
    confidence as a prior.

    Total parameters:
        W_q: Linear(embed_dim, attn_dim, bias=False) = embed_dim * attn_dim
        W_k: Linear(embed_dim, attn_dim, bias=False) = embed_dim * attn_dim
        W_gate: Linear(embed_dim*2, embed_dim)       = embed_dim*2*embed_dim + embed_dim
    For embed_dim=256, attn_dim=32:
        W_q: 8,192  |  W_k: 8,192  |  W_gate: 131,328  →  total ~147,712 params
    """

    def __init__(
        self,
        embed_dim: int = 256,
        attn_dim:  int = 32,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn_dim = attn_dim
        # Low-rank attention projections (bias=False → fewer params, less overfitting)
        self.W_q    = nn.Linear(embed_dim, attn_dim, bias=False)
        self.W_k    = nn.Linear(embed_dim, attn_dim, bias=False)
        # Gate: blends center embedding with neighborhood context
        self.W_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        center_emb: torch.Tensor,   # [B, D]
        neigh_embs: torch.Tensor,   # [B, K, D]
        neigh_wts:  torch.Tensor,   # [B, K] — pre-softmaxed PPI confidence weights
    ) -> torch.Tensor:              # [B, D]
        B, K, D = neigh_embs.shape

        # Project center and neighbors into attention space
        q = self.W_q(center_emb)                                          # [B, attn_dim]
        k = self.W_k(neigh_embs.reshape(-1, D)).reshape(B, K, self.attn_dim)  # [B, K, attn_dim]

        # Scaled dot-product attention
        scale = self.attn_dim ** 0.5
        attn = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)) / scale      # [B, 1, K]

        # Weight by PPI edge confidence before softmax
        attn = attn * neigh_wts.unsqueeze(1)                              # [B, 1, K]
        attn = torch.softmax(attn, dim=-1)                                # [B, 1, K]
        attn = self.attn_dropout(attn)

        # Weighted sum over neighbors
        context = torch.bmm(attn, neigh_embs).squeeze(1)                  # [B, D]

        # Learnable gating: learn how much neighborhood context to incorporate
        gate = torch.sigmoid(self.W_gate(torch.cat([center_emb, context], dim=-1)))  # [B, D]
        fused = gate * center_emb + (1.0 - gate) * context               # [B, D]
        return fused


class GenePriorBias(nn.Module):
    """Per-gene-per-class learnable logit bias with gradient warmup.

    During warmup phase (epochs 0 to bias_warmup_epochs-1), the bias
    is NOT added to logits (not in computation graph → zero gradients).
    At epoch bias_warmup_epochs, activate() is called to enable the bias.

    Critical design: bias_active is a persistent register_buffer so it
    is saved/loaded with every checkpoint, ensuring correct test-time behavior
    when the best checkpoint is loaded with ckpt_path='best'.
    (This fixes the inference bug in older nodes that used current_epoch=0 fallback.)
    """

    def __init__(self, n_classes: int = 3, n_genes: int = 6640) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_classes, n_genes))
        # persistent=True (default) ensures this is saved in checkpoints
        self.register_buffer("bias_active", torch.tensor(False))

    def activate(self) -> None:
        """Enable the bias term. Called at epoch bias_warmup_epochs."""
        self.bias_active.fill_(True)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply per-gene-per-class bias if active.

        Args:
            logits: [B, 3, G]
        Returns:
            [B, 3, G] with bias added (if active)
        """
        if bool(self.bias_active.item()):
            return logits + self.bias.unsqueeze(0)   # [1, 3, G] broadcasts to [B, 3, G]
        return logits


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CombinedStringGNNModel(pl.LightningModule):
    """Frozen STRING_GNN + K=16 Neighborhood Attention + GenePriorBias.

    Architecture:
        1. STRING_GNN run ONCE at setup() → pre-computed node embeddings [18870, 256]
           + pre-computed top-16 neighbor indices and softmax weights.
           Backbone stays completely frozen (no gradient flow).
        2. For each perturbed gene:
           a. Lookup center embedding: node_embeddings[string_node_idx]  → [B, 256]
           b. Lookup neighbor embeddings + edge weights                  → [B, 16, 256], [B, 16]
           c. NeighborhoodAttentionAggregator: fuses center + context    → [B, 256]
           (Fallback: learnable embedding for genes not in STRING graph)
        3. 2-layer MLP projection head:
             LayerNorm(256) → Linear(256→256) → GELU → Dropout(0.40)
             → LayerNorm(256) → Linear(256→256) → GELU → Dropout(0.40)
        4. Bilinear output:
             logits[b,c,g] = h[b] · gene_class_emb[c,g]     → [B, 3, G]
        5. GenePriorBias (inactive for first 50 epochs):
             final_logits = logits + gene_prior.bias          → [B, 3, G]
        6. Weighted cross-entropy + mild label smoothing (eps=0.05)
    """

    def __init__(
        self,
        bilinear_dim:        int   = 256,
        dropout:             float = 0.40,
        lr:                  float = 3e-4,
        weight_decay:        float = 3e-2,
        warmup_epochs:       int   = 20,
        T_max:               int   = 150,
        label_smoothing:     float = 0.05,
        K:                   int   = 16,
        attn_dim:            int   = 32,
        bias_warmup_epochs:  int   = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Model layers initialized in setup()

    def setup(self, stage: Optional[str] = None) -> None:
        # Guard against repeated setup calls (DDP calls setup on each rank)
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        hp = self.hparams

        # ----------------------------------------------------------------
        # 1. Pre-compute STRING_GNN embeddings and neighborhood lookup tables
        # ----------------------------------------------------------------
        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        graph        = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index   = graph["edge_index"].long()
        edge_weight  = graph["edge_weight"].float()
        n_nodes      = backbone.config.num_nodes if hasattr(backbone.config, "num_nodes") else None

        with torch.no_grad():
            gnn_out  = backbone(edge_index=edge_index, edge_weight=edge_weight)
            node_emb = gnn_out.last_hidden_state.float().detach()  # [18870, 256]

        if n_nodes is None:
            n_nodes = node_emb.shape[0]

        # Fixed lookup table — registered as buffer so Lightning moves to GPU
        self.register_buffer("node_embeddings", node_emb)

        # Pre-compute top-K neighborhood for each node
        print(f"[setup] Pre-computing top-K={hp.K} neighbors for {n_nodes} nodes ...")
        topk_nbrs, topk_wts = compute_topk_neighbors(
            edge_index=edge_index,
            edge_weight=edge_weight,
            n_nodes=n_nodes,
            K=hp.K,
        )
        self.register_buffer("topk_neighbors", topk_nbrs)  # [n_nodes, K]
        self.register_buffer("topk_weights",   topk_wts)   # [n_nodes, K]
        print(f"[setup] Neighborhood buffers registered: {topk_nbrs.shape}, {topk_wts.shape}")

        del backbone, graph, edge_index, edge_weight, gnn_out

        # ----------------------------------------------------------------
        # 2. Learnable fallback for unknown pert_ids (~6.4% of training data)
        # ----------------------------------------------------------------
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 3. Neighborhood attention aggregator
        # ----------------------------------------------------------------
        self.nbr_agg = NeighborhoodAttentionAggregator(
            embed_dim    = STRING_DIM,
            attn_dim     = hp.attn_dim,   # 32 (reduced vs sibling1's 64)
            attn_dropout = 0.1,
        )

        # ----------------------------------------------------------------
        # 4. 2-layer MLP head
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(STRING_DIM),
            nn.Linear(STRING_DIM, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
            nn.LayerNorm(hp.bilinear_dim),
            nn.Linear(hp.bilinear_dim, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # ----------------------------------------------------------------
        # 5. Bilinear gene-class embedding (no gene_bias — removed in node1-1-1-1)
        # ----------------------------------------------------------------
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # ----------------------------------------------------------------
        # 6. GenePriorBias module (activated at epoch bias_warmup_epochs)
        # ----------------------------------------------------------------
        self.gene_prior = GenePriorBias(n_classes=N_CLASSES, n_genes=N_GENES)

        # Class weights for weighted CE
        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        for _, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators for val/test (cleared each epoch)
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_ids:   List[str]          = []
        self._test_syms:  List[str]          = []
        self._test_idx:   List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def on_train_epoch_start(self) -> None:
        """Activate GenePriorBias at epoch bias_warmup_epochs."""
        if self.current_epoch == self.hparams.bias_warmup_epochs:
            self.gene_prior.activate()
            self.print(
                f"[Epoch {self.current_epoch}] GenePriorBias activated — "
                f"per-gene calibration bias now in computation graph."
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _get_pert_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Neighborhood-aggregated perturbation embeddings.

        For known pert_ids: center embedding + top-K neighbor attention.
        For unknown pert_ids: learnable fallback embedding (no neighborhood).

        Args:
            string_node_idx: [B] long tensor, -1 for pert_ids not in STRING.
        Returns:
            [B, STRING_DIM] float32 perturbation embeddings.
        """
        B    = string_node_idx.shape[0]
        known   = string_node_idx >= 0
        unknown = ~known

        emb = torch.zeros(B, STRING_DIM, dtype=torch.float32,
                          device=self.node_embeddings.device)

        if known.any():
            known_idx = string_node_idx[known]                        # [n_known]

            # Center embeddings for known genes
            center = self.node_embeddings[known_idx].float()          # [n_known, D]

            # Neighbor embeddings: lookup neighbors, then their embeddings
            nbr_idx  = self.topk_neighbors[known_idx]                 # [n_known, K]
            nbr_embs = self.node_embeddings[nbr_idx].float()          # [n_known, K, D]
            nbr_wts  = self.topk_weights[known_idx].float()           # [n_known, K]

            # Neighborhood-aware representation
            emb[known] = self.nbr_agg(center, nbr_embs, nbr_wts)

        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(unknown.sum(), dtype=torch.long,
                            device=self.node_embeddings.device)
            ).float()
            emb[unknown] = fb

        return emb

    def forward(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        pert_emb = self._get_pert_embeddings(string_node_idx)          # [B, D]
        h        = self.head(pert_emb)                                  # [B, bilinear_dim]
        logits   = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb) # [B, 3, G]
        logits   = self.gene_prior(logits)                              # bias (if active)
        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, 3]
            targets.reshape(-1),                       # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["string_node_idx"])
        loss   = self._loss(logits, batch["labels"])
        # NOTE: sync_dist=True is intentionally omitted in training step.
        # DDP gradient all-reduce is the only collective needed.
        # sync_dist in training_step can deadlock with AMP loss scaler on H100/NCCL.
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits, dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, dim=0)   # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  dim=0)   # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   dim=0)   # [N_local]
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        # Gather across all DDP ranks
        all_preds = self.all_gather(local_preds)           # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)            # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)             # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        # De-duplicate (DDP padding may introduce repeated samples)
        order  = torch.argsort(idx_flat)
        s_idx  = idx_flat[order]
        s_pred = preds_flat[order]
        s_tgt  = tgts_flat[order]
        mask   = torch.cat([
            torch.tensor([True], device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        preds_dedup = s_pred[mask]
        tgts_dedup  = s_tgt[mask]

        f1 = compute_per_gene_f1(preds_dedup, tgts_dedup)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        # Store pert_id and symbol lists for output file
        self._test_ids.extend(batch["pert_id"])
        self._test_syms.extend(batch["symbol"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)  # [N_local]

        all_preds = self.all_gather(local_preds)           # [W, N_local, 3, G]
        all_idx   = self.all_gather(local_idx)             # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            # De-duplicate by sample index
            pred_map: Dict[int, torch.Tensor] = {}
            for i in range(len(idx_flat)):
                gid = int(idx_flat[i].item())
                if gid not in pred_map:
                    pred_map[gid] = preds_flat[i]

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            rows = []
            for i in range(len(test_df)):
                if i not in pred_map:
                    continue
                pid  = test_df.iloc[i]["pert_id"]
                sym  = test_df.iloc[i]["symbol"]
                # IMPORTANT: save as [3, 6640] (NOT transposed) so calc_metric.py works correctly.
                # Parent node (node1-1-1-1) had a bug: .T.tolist() → [6640, 3] which failed
                # calc_metric.py's shape[1]==3 check. This is fixed here.
                pred = pred_map[i].float().cpu().numpy().tolist()  # [3, 6640]
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            print(f"[Node1-1-1-1-3] Saved {len(rows)} test predictions to {out_path}")

        self._test_preds.clear()
        self._test_ids.clear()
        self._test_syms.clear()
        self._test_idx.clear()

    # ------------------------------------------------------------------
    # Checkpoint helpers — save only trainable params + buffers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        saved = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full:
                    saved[key] = full[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full:
                saved[key] = full[key]
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {trainable}/{total} params ({100 * trainable / total:.1f}%)")
        return saved

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: AdamW + linear warmup + CosineAnnealingLR
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=hp.lr, weight_decay=hp.weight_decay)

        # Phase 1: linear warmup from 0.1×lr to lr over warmup_epochs
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        # Phase 2: simple CosineAnnealingLR (T_max=150 provides more budget for post-bias phase)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=hp.T_max,
            eta_min=1e-6,
        )
        # Sequential: warmup first, then cosine decay
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[hp.warmup_epochs],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-1-1-1-3 – Frozen STRING_GNN + K=16 Neighborhood Attention + GenePriorBias"
    )
    parser.add_argument("--micro-batch-size",     type=int,   default=32)
    parser.add_argument("--global-batch-size",    type=int,   default=256)
    parser.add_argument("--max-epochs",           type=int,   default=250)
    parser.add_argument("--lr",                   type=float, default=3e-4)
    parser.add_argument("--weight-decay",         type=float, default=3e-2)
    parser.add_argument("--bilinear-dim",         type=int,   default=256)
    parser.add_argument("--dropout",              type=float, default=0.40)
    parser.add_argument("--warmup-epochs",        type=int,   default=20)
    parser.add_argument("--t-max",                type=int,   default=150, dest="t_max")
    parser.add_argument("--label-smoothing",      type=float, default=0.05,
                        dest="label_smoothing")
    parser.add_argument("--patience",             type=int,   default=25)
    parser.add_argument("--k-neighbors",          type=int,   default=16, dest="k_neighbors")
    parser.add_argument("--attn-dim",             type=int,   default=32, dest="attn_dim")
    parser.add_argument("--bias-warmup-epochs",   type=int,   default=50,
                        dest="bias_warmup_epochs")
    parser.add_argument("--num-workers",          type=int,   default=4)
    parser.add_argument("--debug-max-step",       type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",         action="store_true", dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Limit / debug logic
    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        lim_test  = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        lim_test  = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    # Model
    model = CombinedStringGNNModel(
        bilinear_dim       = args.bilinear_dim,
        dropout            = args.dropout,
        lr                 = args.lr,
        weight_decay       = args.weight_decay,
        warmup_epochs      = args.warmup_epochs,
        T_max              = args.t_max,
        label_smoothing    = args.label_smoothing,
        K                  = args.k_neighbors,
        attn_dim           = args.attn_dim,
        bias_warmup_epochs = args.bias_warmup_epochs,
    )

    # Callbacks
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
        patience  = args.patience,   # 25 — enough for GenePriorBias to activate at epoch 50
        min_delta = 1e-4,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy: use SingleDeviceStrategy for fast_dev_run to avoid DDP+AMP deadlocks
    # on H100/NCCL 2.23 + Lightning 2.5 + PyTorch 2.7 single-step runs.
    use_ddp = n_gpus > 1 and not fast_dev_run
    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if use_ddp else "auto"
    )
    devices_for_trainer = 1 if (fast_dev_run and n_gpus > 1) else n_gpus

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = devices_for_trainer,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        max_steps               = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches     = lim_train,
        limit_val_batches       = lim_val,
        limit_test_batches      = lim_test,
        val_check_interval      = 1.0,
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

    # Save test score
    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1-1-1-1-3] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
