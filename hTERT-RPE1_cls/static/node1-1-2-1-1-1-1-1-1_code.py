"""
Node 1-1-2-1-1-1-1-1-1 – STRING_GNN (Frozen) + Cross-Attention Output Head

Architecture:
  - STRING_GNN backbone (FULLY FROZEN, precomputed once at setup)
    * No backbone fine-tuning (per feedback: mps.6+7+post_mp gave no measurable benefit)
    * Learnable OOV embedding (mean of frozen full embeddings, trainable)
  - Cross-attention output head (REPLACES bilinear head from all prior nodes):
    * Perturbation encoder: LayerNorm + Linear + 4 ResidualBlocks -> [B, 512]
    * Context expansion: Linear(512, K*512) -> [B, K=4, 512]  (K=4 context tokens)
    * Learnable gene position queries: nn.Embedding(6640, 512) -> [B, 6640, 512]
    * 2-layer cross-attention (pre-norm + residual, 8 heads, attn_dim=512):
        Q = gene_queries [B, 6640, 512]
        K = V = ctx     [B, 4,    512]
        -> attended gene repr [B, 6640, 512]
    * Per-gene classification: LayerNorm -> Linear(512, 3) -> [B, 6640, 3] -> [B, 3, 6640]
  - Loss: class-weighted focal cross-entropy (gamma=2.0, weights=[2.0, 0.5, 4.0], ls=0.05)
  - Optimizer:
    * Muon (lr=0.005): 2D hidden matrices — ResidualBlocks + ctx_proj + MHA in/out_proj
    * AdamW (lr=5e-4, wd=2e-3): proj_in linear, gene_queries, classifier, norms, biases
    * AdamW (lr=1e-4, wd=1e-4): learnable OOV embedding
  - LR schedule: single cosine (T_max=80 epochs) with eta_min=1e-6 (NO warm restarts)
    * eta_min prevents hard LR=0 freeze (parent had 47 wasted epochs at LR=0)
  - EarlyStopping patience=50 (parent best at epoch 56; patience=80 wasted 80+ epochs)

Key changes from parent (node1-1-2-1-1-1-1-1, F1=0.5030):
  1. CROSS-ATTENTION HEAD replaces bilinear head — first architectural diversity in 5+ nodes
     Each of 6640 gene positions independently attends to 4 learned perturbation context
     tokens, enabling per-gene dynamic weighting of perturbation signal aspects.
  2. FULLY FROZEN backbone — backbone fine-tuning removed (per feedback recommendation)
  3. eta_min=1e-6 in cosine schedule — prevents total LR freeze observed in parent
  4. patience=50 (from 80) — parent's best at epoch 56; patience=80 wasted time
  5. 4 ResidualBlocks (from 6) — reduce capacity to fight overfitting on 1416 samples

Memory influences:
  - parent feedback (primary): "Replace bilinear with cross-attention head"
  - parent feedback: "eta_min=1e-6", "patience=50"
  - parent feedback: "fully frozen backbone (F1=0.5029 in earlier node)"
  - tree-wide: Muon lr=0.005 validated optimal across all nodes
  - node1-2-3: class weights [2.0, 0.5, 4.0] + gamma=2.0 proven best
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

# ─── Constants ────────────────────────────────────────────────────────────────

N_GENES_OUT = 6640
N_CLASSES = 3

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
STRING_GNN_DIM = 256

# Derive project root from script path so relative data paths work in DDP
# Path(__file__).resolve() -> .../mcts/node1-xxx/main.py
# .parent.parent.parent   -> project root (containing data/ and mcts/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "data")


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Exact per-gene macro F1 matching calc_metric.py logic.

    Args:
        pred_logits_np: [N, 3, G] float (logits or probabilities)
        labels_np:      [N, G]    int   (class indices 0/1/2)

    Returns:
        Mean per-gene F1 score (float).
    """
    pred_classes = pred_logits_np.argmax(axis=1)  # [N, G]
    n_genes = labels_np.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = labels_np[:, g]
        yh = pred_classes[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Loss ─────────────────────────────────────────────────────────────────────

def class_weighted_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Class-weighted focal cross-entropy loss.

    Weights [down=2.0, neutral=0.5, up=4.0] address label imbalance (88.9% neutral).
    Focal term (1-pt)^gamma down-weights easy neutral predictions.
    """
    ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )  # [B, G]
    with torch.no_grad():
        log_probs = F.log_softmax(logits, dim=1)  # [B, C, G]
        probs = log_probs.exp()
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B, G]
    focal = (1.0 - pt) ** gamma * ce
    return focal.mean()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        pert_id_to_gnn_idx: Dict[str, int],
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gnn_indices: List[int] = [
            pert_id_to_gnn_idx.get(pid, -1) for pid in self.pert_ids
        ]
        self.has_labels = has_labels
        if has_labels and "label" in df.columns:
            rows = [[x + 1 for x in json.loads(lbl_str)] for lbl_str in df["label"]]
            self.labels = torch.tensor(rows, dtype=torch.long)  # [N, G]
        else:
            self.has_labels = False

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int):
        item = {
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "gnn_idx": self.gnn_indices[idx],
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch: List[dict]) -> dict:
    result = {
        "gnn_idx": torch.tensor([item["gnn_idx"] for item in batch], dtype=torch.long),
        "pert_id": [item["pert_id"] for item in batch],
        "symbol": [item["symbol"] for item in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([item["label"] for item in batch], dim=0)
    return result


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule for perturbation DEG prediction."""

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.pert_id_to_gnn_idx: Dict[str, int] = {}

    def setup(self, stage: Optional[str] = None):
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.pert_id_to_gnn_idx = {name: i for i, name in enumerate(node_names)}

        dfs = {
            split: pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")
            for split in ("train", "val", "test")
        }
        self.train_ds = PerturbationDataset(dfs["train"], self.pert_id_to_gnn_idx, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   self.pert_id_to_gnn_idx, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  self.pert_id_to_gnn_idx, True)

        oov_train = sum(1 for idx in self.train_ds.gnn_indices if idx == -1)
        oov_val   = sum(1 for idx in self.val_ds.gnn_indices   if idx == -1)
        print(f"[DataModule] OOV genes — train: {oov_train}/{len(self.train_ds)}, "
              f"val: {oov_val}/{len(self.val_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            collate_fn=collate_fn, persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
            collate_fn=collate_fn, persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
            collate_fn=collate_fn, persistent_workers=self.num_workers > 0,
        )


# ─── Model Components ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout + skip."""

    def __init__(self, hidden_dim: int, expand: int = 4, dropout: float = 0.3):
        super().__init__()
        inner = hidden_dim * expand
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class GNNCrossAttentionHead(nn.Module):
    """
    Cross-attention based gene perturbation predictor.

    Replaces the bilinear head used in all prior nodes with a multi-head cross-attention
    mechanism. Each of the 6,640 output gene positions has a learnable query vector that
    attends to K=4 context tokens derived from the perturbation embedding, enabling each
    gene to selectively weight different aspects of the perturbation signal.

    Architecture:
        gnn_emb [B, 256]
          -> proj_in: LayerNorm + Linear(256->512) + GELU + Dropout   [B, 512]
          -> 4x ResidualBlock(512, expand=4, dropout=0.3)             [B, 512]
          -> norm_pert: LayerNorm                                      [B, 512]
          -> ctx_proj: Linear(512, K*512)  reshape  [B, K=4, 512]    (context tokens)

        gene_queries: Embedding(6640, 512)  expanded  [B, 6640, 512]

        For each of 2 cross-attention layers (pre-norm + residual):
            normed_q  = LayerNorm(gene_repr)            [B, 6640, 512]
            attn_out  = MHA(Q=normed_q, K=ctx, V=ctx)  [B, 6640, 512]
            gene_repr = gene_repr + attn_out            [B, 6640, 512]

        -> out_norm: LayerNorm                          [B, 6640, 512]
        -> classifier: Linear(512, 3)                  [B, 6640, 3]
        -> permute(0,2,1)                               [B, 3, 6640]

    Memory footprint (B=16, bf16):
        gene_repr: 16 x 6640 x 512 x 2 = 109 MB per layer
        attn weights: 16 x 8 x 6640 x 4 x 4 = 13.6 MB (trivial)
        Total: well under 1 GB
    """

    def __init__(
        self,
        gnn_dim: int = STRING_GNN_DIM,
        hidden_dim: int = 512,
        n_resblocks: int = 4,
        expand: int = 4,
        dropout: float = 0.3,
        attn_dim: int = 512,
        n_ctx_tokens: int = 4,
        n_heads: int = 8,
        n_attn_layers: int = 2,
        attn_dropout: float = 0.1,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()

        # ── Perturbation encoder ──────────────────────────────────────────────
        self.proj_in = nn.Sequential(
            nn.LayerNorm(gnn_dim),
            nn.Linear(gnn_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.resblocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
              for _ in range(n_resblocks)]
        )
        self.norm_pert = nn.LayerNorm(hidden_dim)

        # ── Context expansion: hidden_dim -> K context tokens ─────────────────
        self.ctx_proj = nn.Linear(hidden_dim, n_ctx_tokens * attn_dim)
        self.n_ctx_tokens = n_ctx_tokens
        self.attn_dim = attn_dim

        # ── Learnable gene position queries ───────────────────────────────────
        # These represent each gene's "question" about the perturbation.
        # Shape [G, attn_dim] learned during training.
        self.gene_queries = nn.Embedding(n_genes_out, attn_dim)
        nn.init.normal_(self.gene_queries.weight, std=0.02)

        # ── Cross-attention layers (pre-norm architecture) ────────────────────
        # Gene queries attend to perturbation context tokens.
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=attn_dim,
                num_heads=n_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            for _ in range(n_attn_layers)
        ])
        self.attn_pre_norms = nn.ModuleList([
            nn.LayerNorm(attn_dim) for _ in range(n_attn_layers)
        ])

        # ── Output classification ─────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(attn_dim)
        self.classifier = nn.Linear(attn_dim, n_classes, bias=True)

        # Initialize projections
        nn.init.xavier_uniform_(self.ctx_proj.weight)
        nn.init.zeros_(self.ctx_proj.bias)
        nn.init.zeros_(self.classifier.bias)

        self.n_genes_out = n_genes_out
        self.n_classes = n_classes

    def forward(self, gnn_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gnn_emb: [B, 256] STRING_GNN frozen embeddings

        Returns:
            logits: [B, 3, 6640]
        """
        B = gnn_emb.shape[0]

        # 1. Encode perturbation into a fixed-size context
        h = self.proj_in(gnn_emb)          # [B, hidden_dim]
        h = self.resblocks(h)               # [B, hidden_dim]
        h = self.norm_pert(h)               # [B, hidden_dim]

        # 2. Expand to K context tokens (K different 'views' of the perturbation)
        ctx = self.ctx_proj(h)              # [B, K * attn_dim]
        ctx = ctx.view(B, self.n_ctx_tokens, self.attn_dim)  # [B, K, attn_dim]

        # 3. Initialize gene queries — each gene independently queries the perturbation
        # .contiguous() ensures proper memory layout for MHA
        gene_repr = self.gene_queries.weight.unsqueeze(0).expand(B, -1, -1).contiguous()
        # gene_repr: [B, 6640, attn_dim]

        # 4. Cross-attention: gene queries attend to perturbation context
        # Pre-norm architecture with residual connections (stable training)
        for attn_layer, pre_norm in zip(self.cross_attn_layers, self.attn_pre_norms):
            normed_q = pre_norm(gene_repr)           # [B, 6640, attn_dim]
            attn_out, _ = attn_layer(
                query=normed_q, key=ctx, value=ctx,  # Q=genes, K=V=perturbation
            )                                        # [B, 6640, attn_dim]
            gene_repr = gene_repr + attn_out          # residual  [B, 6640, attn_dim]

        # 5. Per-gene classification
        gene_repr = self.out_norm(gene_repr)            # [B, 6640, attn_dim]
        logits = self.classifier(gene_repr)             # [B, 6640, 3]
        logits = logits.permute(0, 2, 1).contiguous()  # [B, 3, 6640]

        return logits


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(
    local_preds: torch.Tensor,
    local_labels: torch.Tensor,
    device: torch.device,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather variable-length tensors from all DDP ranks with padding."""
    local_size = torch.tensor([local_preds.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = int(max(s.item() for s in all_sizes))

    pad = max_size - local_preds.shape[0]
    p = local_preds.to(device)
    l = local_labels.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], dim=0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], dim=0)

    g_preds  = [torch.zeros_like(p) for _ in range(world_size)]
    g_labels = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(g_preds, p)
    dist.all_gather(g_labels, l)

    real_preds  = torch.cat([g_preds[i][:all_sizes[i].item()].cpu()  for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][:all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """
    LightningModule for gene perturbation DEG prediction.

    Key changes from parent (node1-1-2-1-1-1-1-1, F1=0.5030):
    1. CROSS-ATTENTION HEAD: 4-block ResidualMLP encoder + K=4 context tokens +
       2-layer MHA cross-attention (8 heads, dim=512) — replaces bilinear head.
       Each of 6640 gene positions independently attends to perturbation context.
    2. FULLY FROZEN backbone: precomputed frozen_full_embs, no backbone fine-tuning.
    3. eta_min=1e-6 in cosine schedule: prevents LR=0 hard freeze at epoch ~89.
    4. patience=50: parent best at epoch 56; 80-epoch patience wasted 80+ epochs.
    5. 4 ResidualBlocks (from 6): reduce overfitting risk on 1416 training samples.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        n_resblocks: int = 4,
        expand: int = 4,
        dropout: float = 0.3,
        attn_dim: int = 512,
        n_ctx_tokens: int = 4,
        n_heads: int = 8,
        n_attn_layers: int = 2,
        attn_dropout: float = 0.1,
        lr: float = 5e-4,
        muon_lr: float = 0.005,
        oov_lr: float = 1e-4,
        weight_decay: float = 2e-3,
        warmup_steps: int = 100,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        class_weight_down: float = 2.0,
        class_weight_neutral: float = 0.5,
        class_weight_up: float = 4.0,
        cosine_max_steps: int = 880,
        eta_min_ratio: float = 0.002,  # eta_min / base_lr: 0.002 * 5e-4 = 1e-6 for AdamW
        grad_clip_norm: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams

        # ── Load STRING_GNN and precompute FROZEN full embeddings ────────────
        # No backbone fine-tuning: backbone is 100% frozen, embeddings precomputed once.
        # This simplifies the model, reduces trainable params, and prevents overfitting
        # from unnecessary backbone adaptation (per feedback: no benefit from fine-tuning).
        gnn_model = AutoModel.from_pretrained(
            str(STRING_GNN_DIR), trust_remote_code=True
        )
        gnn_model.eval()

        graph = torch.load(
            STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False
        )
        edge_index  = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)

        with torch.no_grad():
            outputs = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
                output_hidden_states=False,
            )
        frozen_full_embs = outputs.last_hidden_state.detach()  # [18870, 256]
        self.register_buffer("frozen_full_embs", frozen_full_embs)

        # ── Learnable OOV embedding ──────────────────────────────────────────
        # Initialized with mean of all frozen embeddings; trained during finetuning.
        # Used for ~6.4% of training genes not present in STRING_GNN.
        oov_init = frozen_full_embs.mean(dim=0)  # [256]
        self.oov_emb = nn.Parameter(oov_init.float(), requires_grad=True)

        # ── Register class weights as buffer ─────────────────────────────────
        class_weights = torch.tensor(
            [hp.class_weight_down, hp.class_weight_neutral, hp.class_weight_up],
            dtype=torch.float32,
        )
        self.register_buffer("class_weights", class_weights)

        # ── Build cross-attention prediction head ─────────────────────────────
        self.model = GNNCrossAttentionHead(
            gnn_dim=STRING_GNN_DIM,
            hidden_dim=hp.hidden_dim,
            n_resblocks=hp.n_resblocks,
            expand=hp.expand,
            dropout=hp.dropout,
            attn_dim=hp.attn_dim,
            n_ctx_tokens=hp.n_ctx_tokens,
            n_heads=hp.n_heads,
            n_attn_layers=hp.n_attn_layers,
            attn_dropout=hp.attn_dropout,
        )

        # Cast trainable parameters to float32 for stable optimization
        for param in self.model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    def _get_gnn_emb(self, gnn_idx: torch.Tensor) -> torch.Tensor:
        """Get frozen STRING_GNN embeddings. OOV genes use learnable OOV embedding.

        Backbone is fully frozen: uses precomputed frozen_full_embs.
        """
        valid_mask = gnn_idx >= 0
        safe_idx = gnn_idx.clone()
        safe_idx[~valid_mask] = 0

        emb = self.frozen_full_embs[safe_idx].float()  # [B, 256]

        if (~valid_mask).any():
            emb = emb.clone()
            oov_count = int((~valid_mask).sum().item())
            oov_expanded = self.oov_emb.unsqueeze(0).expand(oov_count, -1)
            emb[~valid_mask] = oov_expanded.to(emb.dtype)

        return emb

    def forward(self, gnn_idx: torch.Tensor) -> torch.Tensor:
        emb = self._get_gnn_emb(gnn_idx)
        return self.model(emb)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return class_weighted_focal_loss(
            logits,
            labels,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["gnn_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["gnn_idx"])
        if "label" in batch:
            loss = self._compute_loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())
        return logits

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        local_p = torch.cat(self._val_preds, dim=0)
        local_l = torch.cat(self._val_labels, dim=0)

        if self.trainer.world_size > 1:
            all_p, all_l = _gather_tensors(local_p, local_l, self.device, self.trainer.world_size)
        else:
            all_p, all_l = local_p, local_l

        f1 = compute_per_gene_f1(all_p.numpy(), all_l.numpy())
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["gnn_idx"])
        probs = torch.softmax(logits, dim=1)  # [B, 3, 6640]
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            if not hasattr(self, "_test_labels"):
                self._test_labels = []
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs = torch.cat(self._test_preds, dim=0)
        dummy_labels = torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        if hasattr(self, "_test_labels") and self._test_labels:
            dummy_labels = torch.cat(self._test_labels, dim=0)
            del self._test_labels

        if self.trainer.world_size > 1:
            all_probs, all_labels = _gather_tensors(
                local_probs, dummy_labels, self.device, self.trainer.world_size
            )
            all_pert = [None] * self.trainer.world_size
            all_syms = [None] * self.trainer.world_size
            dist.all_gather_object(all_pert, self._test_pert_ids)
            dist.all_gather_object(all_syms, self._test_symbols)
            all_pert = [p for sub in all_pert for p in sub]
            all_syms = [s for sub in all_syms for s in sub]
        else:
            all_probs  = local_probs
            all_labels = dummy_labels
            all_pert   = self._test_pert_ids
            all_syms   = self._test_symbols

        if self.trainer.is_global_zero:
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            seen_ids: set = set()
            dedup_probs: list = []
            dedup_labels: list = []
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for i, (pert_id, symbol, probs) in enumerate(
                    zip(all_pert, all_syms, all_probs.numpy())
                ):
                    if pert_id not in seen_ids:
                        seen_ids.add(pert_id)
                        fh.write(f"{pert_id}\t{symbol}\t{json.dumps(probs.tolist())}\n")
                        dedup_probs.append(probs)
                        dedup_labels.append(all_labels[i].numpy())

            self.print(
                f"[Node1-1-2-1-1-1-1-1-1] Saved test predictions -> {pred_path} "
                f"({len(seen_ids)} unique samples)"
            )

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-1-2-1-1-1-1-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        """
        Three parameter groups:
          1. Muon (lr=0.005): 2D hidden weight matrices in ResidualBlocks, ctx_proj,
             MHA in_proj_weight, MHA out_proj.weight
          2. AdamW (lr=5e-4, wd=2e-3): proj_in linear, gene_queries, classifier,
             LayerNorm weights/biases, all other biases
          3. AdamW (lr=1e-4, wd=1e-4): learnable OOV embedding

        LR schedule: single cosine with eta_min (no warm restarts, no hard clamp at 0)
          - Prevents hard LR=0 freeze seen in parent node (47 wasted epochs at LR=0)
          - eta_min = eta_min_ratio * base_lr = 0.002 * 5e-4 = 1e-6 for AdamW group
        """
        hp = self.hparams

        muon_params = []
        adamw_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            is_2d_matrix = param.ndim >= 2
            is_norm_param = "norm" in name         # LayerNorm weight/bias
            is_bias = "bias" in name               # any bias
            is_input_proj = "proj_in" in name      # input projection (boundary w/ backbone)
            is_output_cls = "classifier" in name   # final classification output
            is_embedding  = "gene_queries" in name # embedding table (not suitable for Muon)

            # Muon: 2D hidden matrices that are not input/output boundaries or embeddings
            if (is_2d_matrix and not is_norm_param and not is_bias
                    and not is_input_proj and not is_output_cls and not is_embedding):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        print(f"[Optimizer] Muon params: {sum(p.numel() for p in muon_params):,} "
              f"(covers: ResidualBlock weights, ctx_proj.weight, MHA in/out_proj.weight)")
        print(f"[Optimizer] AdamW head params: {sum(p.numel() for p in adamw_params):,} "
              f"(covers: proj_in, gene_queries, classifier, norms, biases)")
        print(f"[Optimizer] OOV emb: {self.oov_emb.numel()} params "
              f"(lr={hp.oov_lr}, wd=1e-4)")

        param_groups = [
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            dict(
                params=adamw_params,
                use_muon=False,
                lr=hp.lr,
                betas=(0.9, 0.999),
                eps=1e-10,
                weight_decay=hp.weight_decay,
            ),
            dict(
                params=[self.oov_emb],
                use_muon=False,
                lr=hp.oov_lr,
                betas=(0.9, 0.999),
                eps=1e-10,
                weight_decay=1e-4,
            ),
        ]

        if dist.is_available() and dist.is_initialized():
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        # Single cosine schedule with eta_min (NO warm restarts, NO hard clamp at 0).
        # lr_lambda returns a multiplier in [eta_min_ratio, 1.0].
        # Actual eta_min per group = eta_min_ratio * group_base_lr.
        # For AdamW group: eta_min = 0.002 * 5e-4 = 1e-6  (prevents 47-epoch LR=0 freeze)
        # For Muon group:  eta_min = 0.002 * 0.005 = 1e-5
        cosine_max_steps = hp.cosine_max_steps
        eta_min_ratio    = hp.eta_min_ratio

        def lr_lambda_cosine_eta_min(current_step: int) -> float:
            if current_step < hp.warmup_steps:
                # Linear warmup: 0 -> 1
                return float(current_step) / max(1, hp.warmup_steps)
            step_after_warmup = current_step - hp.warmup_steps
            # Single cosine decay, hard-clamped at progress=1.0 (no restart)
            progress = min(1.0, float(step_after_warmup) / max(1, cosine_max_steps))
            cosine_val = 0.5 * (1.0 + np.cos(np.pi * progress))  # [1.0 -> 0.0]
            # Shift range from [0, 1] to [eta_min_ratio, 1.0]
            return eta_min_ratio + (1.0 - eta_min_ratio) * max(0.0, cosine_val)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_cosine_eta_min)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable params + buffers ───────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        trainable_sd = {
            k: v for k, v in full_sd.items()
            if k in trainable_keys or k in buffer_keys
        }
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 1-1-2-1-1-1-1-1-1 – STRING_GNN Frozen + Cross-Attention Head"
    )
    # Data
    p.add_argument("--data-dir",             type=str,   default=_DEFAULT_DATA_DIR)
    # Architecture
    p.add_argument("--hidden-dim",           type=int,   default=512)
    p.add_argument("--n-resblocks",          type=int,   default=4)
    p.add_argument("--expand",               type=int,   default=4)
    p.add_argument("--dropout",              type=float, default=0.3)
    p.add_argument("--attn-dim",             type=int,   default=512)
    p.add_argument("--n-ctx-tokens",         type=int,   default=4)
    p.add_argument("--n-heads",              type=int,   default=8)
    p.add_argument("--n-attn-layers",        type=int,   default=2)
    p.add_argument("--attn-dropout",         type=float, default=0.1)
    # Optimizer
    p.add_argument("--lr",                   type=float, default=5e-4)
    p.add_argument("--muon-lr",              type=float, default=0.005)
    p.add_argument("--oov-lr",               type=float, default=1e-4)
    p.add_argument("--weight-decay",         type=float, default=2e-3)
    p.add_argument("--warmup-steps",         type=int,   default=100)
    p.add_argument("--focal-gamma",          type=float, default=2.0)
    p.add_argument("--label-smoothing",      type=float, default=0.05)
    p.add_argument("--class-weight-down",    type=float, default=2.0)
    p.add_argument("--class-weight-neutral", type=float, default=0.5)
    p.add_argument("--class-weight-up",      type=float, default=4.0)
    p.add_argument("--grad-clip-norm",       type=float, default=1.0)
    p.add_argument("--eta-min-ratio",        type=float, default=0.002)
    # Training schedule
    p.add_argument("--micro-batch-size",     type=int,   default=16)
    p.add_argument("--global-batch-size",    type=int,   default=128)
    p.add_argument("--cosine-tmax-epochs",   type=int,   default=80)
    p.add_argument("--max-epochs",           type=int,   default=150)
    p.add_argument("--patience",             type=int,   default=50)
    p.add_argument("--num-workers",          type=int,   default=4)
    p.add_argument("--val-check-interval",   type=float, default=1.0)
    # Debug
    p.add_argument("--debug-max-step",       type=int,   default=None)
    p.add_argument("--fast-dev-run",         action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # Compute cosine T_max in effective steps
    _train_df_size = pd.read_csv(
        Path(args.data_dir) / "train.tsv", sep="\t", usecols=["pert_id"]
    ).shape[0]
    steps_per_epoch = _train_df_size // (args.micro_batch_size * n_gpus)
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)
    cosine_max_steps = effective_steps_per_epoch * args.cosine_tmax_epochs

    print(f"[Main] effective_steps_per_epoch={effective_steps_per_epoch}, "
          f"cosine_max_steps={cosine_max_steps} (T_max={args.cosine_tmax_epochs} epochs)")
    print(f"[Main] max_epochs={args.max_epochs}, patience={args.patience}, "
          f"eta_min_ratio={args.eta_min_ratio}")

    lit = PerturbationLitModule(
        hidden_dim=args.hidden_dim,
        n_resblocks=args.n_resblocks,
        expand=args.expand,
        dropout=args.dropout,
        attn_dim=args.attn_dim,
        n_ctx_tokens=args.n_ctx_tokens,
        n_heads=args.n_heads,
        n_attn_layers=args.n_attn_layers,
        attn_dropout=args.attn_dropout,
        lr=args.lr,
        muon_lr=args.muon_lr,
        oov_lr=args.oov_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        class_weight_down=args.class_weight_down,
        class_weight_neutral=args.class_weight_neutral,
        class_weight_up=args.class_weight_up,
        cosine_max_steps=max(cosine_max_steps, 1),
        eta_min_ratio=args.eta_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-4)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    # Debug / fast-dev-run settings
    max_steps: int            = -1
    limit_train_batches       = 1.0
    limit_val_batches         = 1.0
    limit_test_batches        = 1.0
    fast_dev_run              = False

    if args.debug_max_step is not None:
        max_steps           = args.debug_max_step
        limit_train_batches = args.debug_max_step
        limit_val_batches   = 2
        limit_test_batches  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        gradient_clip_val=args.grad_clip_norm,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=args.val_check_interval if (
            args.debug_max_step is None and not args.fast_dev_run
        ) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"Node 1-1-2-1-1-1-1-1-1 – STRING_GNN Frozen + Cross-Attention Head\n"
            f"Test results: {test_results}\n"
            f"Architecture: hidden_dim={args.hidden_dim}, n_resblocks={args.n_resblocks}, "
            f"attn_dim={args.attn_dim}, n_ctx_tokens={args.n_ctx_tokens}, "
            f"n_heads={args.n_heads}, n_attn_layers={args.n_attn_layers}, "
            f"attn_dropout={args.attn_dropout}\n"
            f"Optimizer: muon_lr={args.muon_lr}, lr={args.lr}, oov_lr={args.oov_lr}, "
            f"weight_decay={args.weight_decay}\n"
            f"Loss: focal_gamma={args.focal_gamma}, label_smoothing={args.label_smoothing}, "
            f"class_weights=[{args.class_weight_down},{args.class_weight_neutral},{args.class_weight_up}]\n"
            f"Schedule: cosine_tmax={args.cosine_tmax_epochs} epochs, "
            f"eta_min_ratio={args.eta_min_ratio}, patience={args.patience}\n"
        )


if __name__ == "__main__":
    main()
