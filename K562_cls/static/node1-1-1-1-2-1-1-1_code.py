"""Node 1-1-1-1-2-1-1-1 – STRING-only K=24, Two-Stage GenePriorBias, Noise Augmentation.

KEY DECISION: Abandon AIDO.Cell fusion — return to proven STRING-only architecture.

Parent (node1-1-1-1-2-1-1, F1=0.4123) attempted AIDO.Cell-100M LoRA + STRING fusion but
suffered severe regression due to:
  1. weight_decay=0.02 (too low for 6M-param model on 1388 samples → gene_class_emb overfit)
  2. gradient accumulation=32 (too high → slow convergence, trapped in local minimum)
  3. AIDO.Cell fusion created 896-dim optimization space that head could not learn effectively

Grandparent (node1-1-1-1-2-1, F1=0.4913) with STRING-only + GenePriorBias is the most
reliable baseline in this lineage. This node returns to that architecture with improvements.

Improvements over grandparent (node1-1-1-1-2-1, F1=0.4913):
1. K=24 neighborhood attention (vs K=16): more PPI context without instability
   - Feedback recommends: "Increase K from 16 to 24 with attn_dim=64"
2. Two-stage GenePriorBias training (bias activation at epoch 80):
   - Stage 1 (epoch 0-79): Head trains freely, bias frozen (LR=0.0)
   - Stage 2 (epoch 80+): Bias activates with lr=1e-5
   - Feedback recommends: "Two-stage bias training: head first 80 epochs, then bias"
3. Embedding noise augmentation during training (σ=0.05):
   - Adds small Gaussian noise to STRING embeddings during training
   - Prevents overfitting on STRING topology patterns
   - Data regularizer for only 1388 training samples
4. Hyperparameters aligned with grandparent's proven values:
   - lr=3e-4 (vs 1e-4 in parent that regressed)
   - weight_decay=4e-2 (vs 2e-2 in parent → 2× stronger regularization)
   - dropout=0.40 (vs 0.50 in parent)
   - patience=25, max_epochs=300, warmup=20
   - micro_batch=8 (vs 4), global_batch=256 (accum=16 on 2 GPUs)

Memory connections:
  - node1-1-1-1-2-1 (grandparent, F1=0.4913): proven STRING-only baseline to build on
  - node1-1-1-1-2-1-1 (parent, F1=0.4123): AIDO.Cell fusion failure → abandon path
  - parent feedback: "Return to STRING-only + GenePriorBias; K=24, two-stage training"
  - node2-1-1-1-1-1 (best in tree, F1=0.5128): K=16 2-head STRING neighborhood attention
    works extremely well in STRING-rich architectures
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
N_GENES       = 6640
N_CLASSES     = 3
STRING_HIDDEN = 256  # STRING_GNN hidden dimension

# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
DATA_ROOT      = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV      = DATA_ROOT / "train.tsv"
VAL_TSV        = DATA_ROOT / "val.tsv"
TEST_TSV       = DATA_ROOT / "test.tsv"


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


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0, 1, 2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)          # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)          # [N, G]
        is_pred = (y_hat == c)            # [N, G]
        present = is_true.any(dim=0)      # [G]

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
    """K562 DEG prediction dataset (STRING-only, no AIDO tokenization needed)."""

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
    """Collate without AIDO tokenization (STRING-only)."""
    out: Dict[str, Any] = {
        "sample_idx":      torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":         [b["pert_id"] for b in batch],
        "symbol":          [b["symbol"] for b in batch],
        "string_node_idx": torch.stack([b["string_node_idx"] for b in batch]),
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_workers: int = 4) -> None:
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
# Multi-Head Neighborhood Attention Module (STRING_GNN)
# ---------------------------------------------------------------------------
class MultiHeadNeighborhoodAttentionModule(nn.Module):
    """K-hop neighborhood attention with multiple heads for PPI graph context.

    For each perturbed gene, aggregates top-K PPI neighbor embeddings using
    independent attention heads, then projects the concatenated head outputs
    back to the STRING hidden dimension.

    Architecture (per head h):
        q_h = W_q_h(center_emb)                                [B, attn_dim]
        k_h = W_k_h(neigh_embs.reshape(-1, D)).reshape(B, K, attn_dim)
        attn_h = softmax(q_h @ k_h.T / sqrt(attn_dim)) * neigh_weights  [B, K]
        context_h = attn_h @ neigh_embs                        [B, D]
    multi_context = concat([context_h for each head])          [B, n_heads*D]
    output = W_out(multi_context)                              [B, D]

    Used with K=24 in this node (vs K=16 in grandparent/parent).
    """

    def __init__(
        self,
        emb_dim: int = STRING_HIDDEN,
        attn_dim: int = 64,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads  = n_heads
        self.attn_dim = attn_dim
        self.W_q = nn.ModuleList([nn.Linear(emb_dim, attn_dim, bias=False) for _ in range(n_heads)])
        self.W_k = nn.ModuleList([nn.Linear(emb_dim, attn_dim, bias=False) for _ in range(n_heads)])
        self.W_out   = nn.Linear(emb_dim * n_heads, emb_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        center_emb: torch.Tensor,    # [B, D]
        neigh_embs: torch.Tensor,    # [B, K, D]
        neigh_weights: torch.Tensor, # [B, K]
        valid_mask: torch.Tensor,    # [B] bool
    ) -> torch.Tensor:
        """Return context-enriched embeddings [B, D]."""
        B, D = center_emb.shape
        K = neigh_embs.shape[1]

        head_contexts = []
        for h in range(self.n_heads):
            q = self.W_q[h](center_emb)   # [B, attn_dim]
            k_flat = self.W_k[h](neigh_embs.reshape(-1, D))
            k = k_flat.reshape(B, K, self.attn_dim)   # [B, K, attn_dim]

            # Scaled dot-product attention: [B, 1, K]
            attn = (q.unsqueeze(1) @ k.transpose(1, 2)) / (self.attn_dim ** 0.5)
            attn = attn.squeeze(1)   # [B, K]

            # Modulate by STRING confidence weights and re-normalize
            attn = F.softmax(attn, dim=-1) * neigh_weights
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            attn = self.dropout(attn)

            ctx = (attn.unsqueeze(1) @ neigh_embs).squeeze(1)  # [B, D]
            head_contexts.append(ctx)

        # Concatenate [B, n_heads*D] → project → [B, D]
        multi_ctx = torch.cat(head_contexts, dim=-1)
        out = self.W_out(multi_ctx)   # [B, D]

        # For unknown pert_ids (no STRING neighbors), return center_emb unchanged
        out = torch.where(valid_mask.unsqueeze(-1), out, center_emb)
        return out


# ---------------------------------------------------------------------------
# GenePriorBias: per-gene per-class learnable bias
# ---------------------------------------------------------------------------
class GenePriorBias(nn.Module):
    """Learnable per-gene, per-class bias [3, G].

    Initially frozen (requires_grad=False), activated at epoch 80
    with a lower learning rate (lr=1e-5) via BiasActivationCallback.
    This prevents the bias from interfering with the head's optimization
    during early training when gene representations are still unstable.
    """

    def __init__(self, n_classes: int = N_CLASSES, n_genes: int = N_GENES) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_classes, n_genes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Add per-gene per-class bias to logits [B, 3, G]."""
        return logits + self.bias.unsqueeze(0)  # [B, 3, G] + [1, 3, G]


# ---------------------------------------------------------------------------
# BiasActivationCallback: two-stage training control
# ---------------------------------------------------------------------------
class BiasActivationCallback(pl.Callback):
    """Activates GenePriorBias at a specified epoch with a lower LR.

    Stage 1 (epoch 0 to activation_epoch-1): GenePriorBias frozen (LR=0.0).
      The main head optimizes freely without per-gene calibration interference.
    Stage 2 (epoch activation_epoch onwards): bias requires_grad=True, LR=bias_lr.
      The bias refines per-gene class distributions on top of the learned head.
    """

    def __init__(self, activation_epoch: int, bias_lr: float) -> None:
        self.activation_epoch = activation_epoch
        self.bias_lr = bias_lr
        self._activated = False

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self._activated and trainer.current_epoch >= self.activation_epoch:
            self._activated = True
            # Enable gradient flow for the bias parameter
            pl_module.gene_prior_bias.bias.requires_grad_(True)
            # Update the 'bias' param group LR in the optimizer
            opt = trainer.optimizers[0]
            for pg in opt.param_groups:
                if pg.get("name") == "bias":
                    pg["lr"] = self.bias_lr
                    pl_module.print(
                        f"[Epoch {trainer.current_epoch}] GenePriorBias activated "
                        f"with lr={self.bias_lr:.2e}"
                    )


# ---------------------------------------------------------------------------
# Main Model: STRING-only with K=24, GenePriorBias, Noise Augmentation
# ---------------------------------------------------------------------------
class STRINGDEGModel(pl.LightningModule):
    """STRING-only DEG prediction model with K=24 2-head neighborhood attention.

    Architecture:
        1. STRING_GNN (frozen, pre-computed): lookup + K=24 2-head neighborhood attention
           → enriched perturbation context [B, 256]
           Optional Gaussian noise (σ=0.05) added during training
        2. 2-layer MLP head: 256 → head_hidden with dropout
        3. Bilinear output: einsum("bd,cgd->bcg", h, gene_class_emb) → [B, 3, G]
        4. GenePriorBias: learnable [3, G] bias, frozen until epoch 80, then lr=1e-5
        5. Weighted cross-entropy + label smoothing ε=0.05
    """

    def __init__(
        self,
        head_hidden: int = 256,
        head_dropout: float = 0.40,
        lr: float = 3e-4,
        bias_lr: float = 1e-5,
        weight_decay: float = 4e-2,
        warmup_epochs: int = 20,
        T_max: int = 200,
        label_smoothing: float = 0.05,
        k_neighbors: int = 24,
        attn_dim: int = 64,
        n_attn_heads: int = 2,
        attn_dropout: float = 0.1,
        emb_noise_std: float = 0.05,
        bias_activation_epoch: int = 80,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def _precompute_topk_neighbors(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        n_nodes: int,
        k: int,
    ):
        """Pre-compute top-K neighbors per node sorted by edge weight."""
        edge_src = edge_index[0].numpy()
        edge_dst = edge_index[1].numpy()
        edge_wt  = edge_weight.numpy()

        order = np.argsort(edge_src, kind="stable")
        edge_src_s = edge_src[order]
        edge_dst_s = edge_dst[order]
        edge_wt_s  = edge_wt[order]

        counts  = np.bincount(edge_src_s, minlength=n_nodes)
        offsets = np.concatenate([[0], np.cumsum(counts)])

        topk_idx_np = np.zeros((n_nodes, k), dtype=np.int64)
        topk_wts_np = np.zeros((n_nodes, k), dtype=np.float32)
        for i in range(n_nodes):
            topk_idx_np[i] = i  # default: self-loop

        for i in range(n_nodes):
            start, end = int(offsets[i]), int(offsets[i + 1])
            if start == end:
                continue
            nbr_dst = edge_dst_s[start:end]
            nbr_wt  = edge_wt_s[start:end]
            n_nbr   = len(nbr_dst)
            k_actual = min(k, n_nbr)
            if n_nbr <= k:
                idx = np.argsort(-nbr_wt)[:k_actual]
            else:
                part_idx = np.argpartition(-nbr_wt, k_actual)[:k_actual]
                idx = part_idx[np.argsort(-nbr_wt[part_idx])]
            topk_idx_np[i, :k_actual] = nbr_dst[idx]
            topk_wts_np[i, :k_actual] = nbr_wt[idx]

        return (
            torch.from_numpy(topk_idx_np).long(),
            torch.from_numpy(topk_wts_np).float(),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        hp = self.hparams

        # ----------------------------------------------------------------
        # 1. Pre-compute STRING_GNN embeddings (frozen backbone)
        # ----------------------------------------------------------------
        string_backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        string_backbone.eval()
        for p in string_backbone.parameters():
            p.requires_grad = False

        graph       = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph["edge_index"].long()
        edge_weight = graph["edge_weight"].float()

        with torch.no_grad():
            gnn_out  = string_backbone(edge_index=edge_index, edge_weight=edge_weight)
            node_emb = gnn_out.last_hidden_state.float().detach()  # [18870, 256]

        self.register_buffer("node_embeddings", node_emb)  # non-trainable buffer

        # ----------------------------------------------------------------
        # 2. Pre-compute top-K=24 neighbor indices and weights
        # ----------------------------------------------------------------
        n_nodes = node_emb.shape[0]
        topk_idx, topk_wts = self._precompute_topk_neighbors(
            edge_index, edge_weight, n_nodes, hp.k_neighbors
        )
        self.register_buffer("topk_idx", topk_idx)
        self.register_buffer("topk_wts", topk_wts)

        del string_backbone, graph, edge_index, edge_weight, gnn_out

        # ----------------------------------------------------------------
        # 3. Learnable fallback for unknown pert_ids
        # ----------------------------------------------------------------
        self.fallback_string_emb = nn.Embedding(1, STRING_HIDDEN)
        nn.init.normal_(self.fallback_string_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 4. Multi-head neighborhood attention (K=24, 2 heads)
        # ----------------------------------------------------------------
        self.neighborhood_attn = MultiHeadNeighborhoodAttentionModule(
            emb_dim=STRING_HIDDEN,
            attn_dim=hp.attn_dim,
            n_heads=hp.n_attn_heads,
            dropout=hp.attn_dropout,
        )

        # ----------------------------------------------------------------
        # 5. 2-layer MLP head + bilinear gene-class embedding
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(STRING_HIDDEN),
            nn.Linear(STRING_HIDDEN, hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
        )
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.head_hidden) * 0.02
        )

        # ----------------------------------------------------------------
        # 6. GenePriorBias — frozen initially, activated at epoch 80
        # ----------------------------------------------------------------
        self.gene_prior_bias = GenePriorBias()
        # Start frozen: bias will not receive gradients until activation
        self.gene_prior_bias.bias.requires_grad_(False)

        # ----------------------------------------------------------------
        # 7. Class weights for weighted CE
        # ----------------------------------------------------------------
        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        for _, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators for val/test gathering
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    def _get_string_embeddings(
        self,
        string_node_idx: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        """Lookup STRING_GNN embeddings + K=24 2-head neighborhood attention.

        During training, optionally adds Gaussian noise for regularization.

        Args:
            string_node_idx: [B] long, -1 for pert_ids not in STRING.
            training:        bool, whether to apply embedding noise.
        Returns:
            [B, STRING_HIDDEN] float32 context embeddings.
        """
        B = string_node_idx.shape[0]
        device = self.node_embeddings.device

        known   = string_node_idx >= 0
        unknown = ~known

        emb = torch.zeros(B, STRING_HIDDEN, dtype=torch.float32, device=device)

        if known.any():
            known_idx  = string_node_idx[known]
            center_emb = self.node_embeddings[known_idx].float()

            neigh_idx  = self.topk_idx[known_idx]
            neigh_idx  = neigh_idx.clamp(0, self.node_embeddings.shape[0] - 1)
            neigh_embs = self.node_embeddings[neigh_idx.reshape(-1)].float()
            neigh_embs = neigh_embs.reshape(
                known_idx.shape[0], self.hparams.k_neighbors, STRING_HIDDEN
            )
            neigh_wts  = self.topk_wts[known_idx].float()
            valid_mask = torch.ones(known_idx.shape[0], dtype=torch.bool, device=device)

            enriched = self.neighborhood_attn(
                center_emb=center_emb,
                neigh_embs=neigh_embs,
                neigh_weights=neigh_wts,
                valid_mask=valid_mask,
            )

            # Embedding noise augmentation during training (σ=0.05)
            if training and self.hparams.emb_noise_std > 0:
                noise_scale = self.hparams.emb_noise_std * enriched.detach().std().clamp(min=1e-6)
                enriched = enriched + torch.randn_like(enriched) * noise_scale

            emb[known] = enriched

        if unknown.any():
            fb = self.fallback_string_emb(
                torch.zeros(unknown.sum(), dtype=torch.long, device=device)
            ).to(torch.float32)
            emb[unknown] = fb

        return emb

    def forward(
        self,
        string_node_idx: torch.Tensor,  # [B] long
        training: bool = False,
    ) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        # 1. STRING_GNN K=24 2-head neighborhood attention
        string_emb = self._get_string_embeddings(string_node_idx, training=training)  # [B, 256]

        # 2. MLP head → [B, head_hidden]
        h = self.head(string_emb)

        # 3. Bilinear interaction: h · gene_class_emb → [B, 3, G]
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)

        # 4. Add per-gene per-class bias (zero in stage 1, learned in stage 2)
        logits = self.gene_prior_bias(logits)

        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy + mild label smoothing."""
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["string_node_idx"], training=True)
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"], training=False)
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
        local_preds = torch.cat(self._val_preds, dim=0)
        local_tgts  = torch.cat(self._val_tgts,  dim=0)
        local_idx   = torch.cat(self._val_idx,   dim=0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds)
        all_tgts  = self.all_gather(local_tgts)
        all_idx   = self.all_gather(local_idx)

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

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
        logits = self(batch["string_node_idx"], training=False)
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        """Save test predictions using a standalone sequential dataloader.

        Lightning's trainer.test() wraps dataloaders with DistributedSampler
        in DDP mode, potentially missing some test samples. This method bypasses
        Lightning's test dataloader by iterating the dataset directly with a
        SequentialSampler, guaranteeing all 154 samples are processed.
        """
        from torch.utils.data import DataLoader, SequentialSampler

        self.print("[on_test_epoch_end] Starting sequential test loop")
        self.eval()

        dm = self.trainer.datamodule
        test_ds = dm.test_ds
        test_dl = DataLoader(
            test_ds,
            batch_size=8,
            shuffle=False,
            sampler=SequentialSampler(test_ds),
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        self.print(f"[on_test_epoch_end] {len(test_dl)} batches, {len(test_ds)} samples")

        all_preds: List[torch.Tensor] = []
        all_idx:   List[torch.Tensor] = []

        device = next(self.parameters()).device
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for batch in test_dl:
                batch_gpu = {
                    "string_node_idx": batch["string_node_idx"].to(device, non_blocking=True),
                }
                logits = self(batch_gpu["string_node_idx"], training=False)
                probs  = torch.softmax(logits, dim=1).float()
                all_preds.append(probs.cpu())
                all_idx.append(batch["sample_idx"])

        local_preds_t = torch.cat(all_preds, dim=0)
        local_idx_t   = torch.cat(all_idx,   dim=0)

        gathered_preds = self.all_gather(local_preds_t)
        gathered_idx   = self.all_gather(local_idx_t)

        if self.trainer.is_global_zero:
            preds_flat = gathered_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = gathered_idx.view(-1)

            # Deduplicate: keep first occurrence of each sample index
            pred_map: Dict[int, torch.Tensor] = {}
            for i in range(len(idx_flat)):
                gid = int(idx_flat[i].item())
                if gid not in pred_map:
                    pred_map[gid] = preds_flat[i]

            self.print(f"[on_test_epoch_end] Unique predictions: {len(pred_map)}")

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            rows = []
            for i in range(len(test_df)):
                if i not in pred_map:
                    self.print(f"[on_test_epoch_end] WARNING: missing prediction for idx {i}")
                    continue
                pid  = test_df.iloc[i]["pert_id"]
                sym  = test_df.iloc[i]["symbol"]
                pred = pred_map[i].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
            self.print(f"[on_test_epoch_end] Saved {len(rows)} test predictions to {pred_path}")

        self._test_preds.clear()
        self._test_idx.clear()

    # ------------------------------------------------------------------
    # Checkpoint helpers — save trainable params + bias + buffers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable = {}

        # Save all parameters that require grad
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full:
                    trainable[key] = full[key]

        # Also save gene_prior_bias even when frozen (it may have been activated
        # in stage 2 and contains learned per-gene calibration values)
        for name, p in self.named_parameters():
            if "gene_prior_bias" in name:
                key = prefix + name
                if key in full:
                    trainable[key] = full[key]

        # Save all buffers (node_embeddings, topk_idx, topk_wts, class_weights)
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full:
                trainable[key] = full[key]

        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {train}/{total} params ({100 * train / total:.1f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable params and buffers from a partial checkpoint."""
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: AdamW + linear warmup + CosineAnnealingLR
    # Two param groups: main (head + attn) and bias (frozen until epoch 80)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        # Separate bias params from main params
        bias_params     = list(self.gene_prior_bias.parameters())
        bias_param_ids  = {id(p) for p in bias_params}
        main_params     = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in bias_param_ids
        ]

        # Two param groups: main params get lr=3e-4, bias starts at lr=0.0
        # BiasActivationCallback updates bias LR to 1e-5 at epoch 80
        param_groups = [
            {"params": main_params, "lr": hp.lr,   "weight_decay": hp.weight_decay, "name": "main"},
            {"params": bias_params, "lr": 0.0,      "weight_decay": hp.weight_decay, "name": "bias"},
        ]
        opt = torch.optim.AdamW(param_groups)

        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=hp.warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=hp.T_max, eta_min=1e-6,
        )
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
        description="Node1-1-1-1-2-1-1-1 – STRING-only K=24 + Two-Stage GenePriorBias"
    )
    parser.add_argument("--micro-batch-size",       type=int,   default=8)
    parser.add_argument("--global-batch-size",      type=int,   default=256)
    parser.add_argument("--max-epochs",             type=int,   default=300)
    parser.add_argument("--lr",                     type=float, default=3e-4)
    parser.add_argument("--bias-lr",                type=float, default=1e-5,  dest="bias_lr")
    parser.add_argument("--weight-decay",           type=float, default=4e-2)
    parser.add_argument("--head-hidden",            type=int,   default=256,   dest="head_hidden")
    parser.add_argument("--head-dropout",           type=float, default=0.40,  dest="head_dropout")
    parser.add_argument("--warmup-epochs",          type=int,   default=20)
    parser.add_argument("--t-max",                  type=int,   default=200,   dest="t_max")
    parser.add_argument("--label-smoothing",        type=float, default=0.05,  dest="label_smoothing")
    parser.add_argument("--k-neighbors",            type=int,   default=24,    dest="k_neighbors")
    parser.add_argument("--attn-dim",               type=int,   default=64,    dest="attn_dim")
    parser.add_argument("--n-attn-heads",           type=int,   default=2,     dest="n_attn_heads")
    parser.add_argument("--attn-dropout",           type=float, default=0.1,   dest="attn_dropout")
    parser.add_argument("--emb-noise-std",          type=float, default=0.05,  dest="emb_noise_std")
    parser.add_argument("--bias-activation-epoch",  type=int,   default=80,    dest="bias_activation_epoch")
    parser.add_argument("--patience",               type=int,   default=25)
    parser.add_argument("--num-workers",            type=int,   default=4)
    parser.add_argument("--debug-max-step",         type=int,   default=None,  dest="debug_max_step")
    parser.add_argument("--fast-dev-run",           action="store_true",        dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    model = STRINGDEGModel(
        head_hidden            = args.head_hidden,
        head_dropout           = args.head_dropout,
        lr                     = args.lr,
        bias_lr                = args.bias_lr,
        weight_decay           = args.weight_decay,
        warmup_epochs          = args.warmup_epochs,
        T_max                  = args.t_max,
        label_smoothing        = args.label_smoothing,
        k_neighbors            = args.k_neighbors,
        attn_dim               = args.attn_dim,
        n_attn_heads           = args.n_attn_heads,
        attn_dropout           = args.attn_dropout,
        emb_noise_std          = args.emb_noise_std,
        bias_activation_epoch  = args.bias_activation_epoch,
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
        patience  = args.patience,
        min_delta = 1e-4,
    )
    lr_cb   = LearningRateMonitor(logging_interval="epoch")
    pg_cb   = TQDMProgressBar(refresh_rate=10)
    bias_cb = BiasActivationCallback(
        activation_epoch = args.bias_activation_epoch,
        bias_lr          = args.bias_lr,
    )

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy: DDP for multi-GPU, auto for single-GPU / fast_dev_run
    use_ddp = n_gpus > 1 and not fast_dev_run
    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=600))
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
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb, bias_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    # Run test using trainer.test() which triggers on_test_epoch_end where we
    # run the sequential test loop (bypasses Lightning's DistributedSampler).
    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    if ckpt_path:
        test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)
    else:
        test_results = trainer.test(model, datamodule=dm)

    # Save test score to run/ directory AND node root for EvaluateAgent
    for score_path in [
        output_dir / "test_score.txt",
        Path(__file__).parent / "test_score.txt",
    ]:
        with open(score_path, "w") as f:
            f.write(f"test_results: {test_results}\n")
            if test_results:
                for k, v in test_results[0].items():
                    f.write(f"  {k}: {v}\n")

    print(f"[Node1-1-1-1-2-1-1-1] Done. Results: {test_results}")


if __name__ == "__main__":
    main()
