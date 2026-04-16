"""
Node 2-1-1-1-1-1-1-1 — Frozen STRING_GNN + 6-layer rank=256 head + Stronger Weight Decay

This node addresses the primary bottleneck of its parent (node2-1-1-1-1-1-1, F1=0.4705):
reducing the residual depth from 8 → 6 layers (proven optimal by clean isolated ablation),
while adding stronger weight decay (3e-3 vs node1-2's 1e-3) to push beyond the tree best.

Architecture:
  - Frozen STRING_GNN backbone (all 5.43M params frozen):
      * Precomputed frozen forward pass in DataModule.setup()
      * Zero gradient pressure on backbone — avoids catastrophic forgetting
      * Proven best foundation (node1-2: F1=0.4912, tree best)

  - NO conditioning MLP (abandoned per feedback consensus from all prior nodes)

  - OOV embedding (learned): fallback for ~6.4% out-of-vocabulary genes

  - 6-layer residual bilinear MLP head (rank=256):
      * LayerNorm + Input projection (256→512)
      * 6x ResidualBlock (expand=4, dropout=0.2) — proven optimal depth from node1-2
      * Bilinear rank=256: [B,3,256] × [6640,256]^T → [B, 3, 6640]
      * Random init for out_gene_emb (std=0.02) — same as node1-2 proven approach

  - Pure focal loss only (gamma=2.0, label_smoothing=0.0):
      * Hard-target gradient signal for minority classes
      * Focal term handles the 88.9% neutral class imbalance

Key changes over parent (node2-1-1-1-1-1-1, F1=0.4705):
  1. n_residual_layers 8 → 6: PRIMARY FIX — isolated ablation proves 6 layers is
     definitively better than 8 for this 1,416-sample task (expected +0.015-0.020 F1)
  2. weight_decay 1e-3 → 3e-3: KEY DIFFERENTIATION from node1-2 — the parent still
     showed loss ratio 2.276 (moderate overfitting); stronger WD addresses this
  3. patience 80 → 50: aligned with node1-2's proven optimal; avoids 80-epoch plateau
     at minimum LR (parent's best epoch was 77, patience=80 → 80 wasted epochs)
  4. LR schedule: target ~90 epochs (better aligned with patience=50 convergence profile
     vs prior nodes targeting 120 epochs that expired before best checkpoint)

Differentiation from node1-2 (tree best, F1=0.4912):
  - weight_decay: 3e-3 vs node1-2's 1e-3 (3x stronger regularization)
  - This is the primary avenue to surpass node1-2: node1-2 shows train/val F1 gap
    indicating mild overfitting; stronger WD should close this gap

Expected performance:
  - Recovery from parent: +0.015-0.020 F1 (from 0.4705 to ~0.488-0.491) by fixing depth
  - Improvement over node1-2: +0.000-0.008 F1 (from 0.4912 to ~0.491-0.499) via stronger WD

MCTS Tree context:
  - node1-2 (F1=0.4912): frozen STRING_GNN + 6-layer rank=256, NO smoothing → tree best
  - node2-1-1-1-1-1-1 (F1=0.4705, parent): 8-layer rank=256, no smoothing → depth bottleneck
  - node1-2-1-1 (F1=0.4900): frozen + pert_matrix conditioning → slightly worse
  - node3-1-1-1-1-1-1: 8 layers + rank=512 + dropout=0.3 + wd=3e-3 → depth issue persists
  - node2-1-1-1-1-1 (F1=0.4780): 8 layers + rank=512 + label_smoothing=0.05 → rank sat + smooth
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Required for deterministic=True with CUDA >= 10.2

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

# ─── Constants ────────────────────────────────────────────────────────────────

N_GENES_OUT = 6640
N_CLASSES = 3
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")


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


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset.

    Stores precomputed frozen STRING_GNN embeddings and their vocab membership flags.
    The frozen embeddings are used directly for in-vocab genes; OOV genes get a learned fallback.
    No conditioning is applied here — the model handles OOV at forward time.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gnn_embeddings: np.ndarray,       # [N_nodes, 256] - frozen base embeddings
        node_name_to_idx: Dict[str, int],
        embed_dim: int = 256,
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.has_labels = has_labels

        n_samples = len(df)
        base_embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)
        in_vocab = []

        for i, pert_id in enumerate(self.pert_ids):
            if pert_id in node_name_to_idx:
                node_idx = node_name_to_idx[pert_id]
                base_embeddings[i] = gnn_embeddings[node_idx]
                in_vocab.append(True)
            else:
                # OOV: base embedding stays as zeros (model will replace with oov_embedding)
                in_vocab.append(False)

        self.base_embeddings = torch.from_numpy(base_embeddings)  # [N, 256]
        self.in_vocab = torch.tensor(in_vocab, dtype=torch.bool)  # [N]

        if has_labels and "label" in df.columns:
            rows = []
            for lbl_str in df["label"]:
                rows.append([x + 1 for x in json.loads(lbl_str)])
            self.labels = torch.tensor(rows, dtype=torch.long)  # [N, G]
        else:
            self.has_labels = False

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int):
        item = {
            "pert_id":        self.pert_ids[idx],
            "symbol":         self.symbols[idx],
            "base_embedding": self.base_embeddings[idx],   # [256]
            "in_vocab":       self.in_vocab[idx],           # bool
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule using precomputed frozen STRING_GNN embeddings."""

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

    def setup(self, stage: Optional[str] = None):
        # Guard against double initialization
        if hasattr(self, "train_ds"):
            return

        # Run STRING_GNN ONCE in completely frozen mode to get base embeddings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[DataModule] Computing STRING_GNN base embeddings (frozen forward pass)...")
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", weights_only=False)

        model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        edge_index = graph["edge_index"].to(device)
        edge_weight = graph["edge_weight"].to(device) if graph.get("edge_weight") is not None else None

        with torch.no_grad():
            outputs = model(edge_index=edge_index, edge_weight=edge_weight)

        gnn_embeddings = outputs.last_hidden_state.float().cpu().numpy()  # [N_nodes, 256]
        node_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(node_names)}

        self.gnn_embeddings = gnn_embeddings
        self.node_name_to_idx = node_name_to_idx
        self.n_gnn_nodes = len(node_names)

        del model
        torch.cuda.empty_cache()

        print(f"[DataModule] STRING_GNN base embeddings shape: {gnn_embeddings.shape}")

        # Load all splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        n_train_cov = sum(p in node_name_to_idx for p in dfs["train"]["pert_id"])
        n_val_cov = sum(p in node_name_to_idx for p in dfs["val"]["pert_id"])
        print(f"[DataModule] Coverage: {n_train_cov}/{len(dfs['train'])} train genes, "
              f"{n_val_cov}/{len(dfs['val'])} val genes in STRING_GNN")

        embed_dim = gnn_embeddings.shape[1]
        self.train_ds = PerturbationDataset(dfs["train"], gnn_embeddings, node_name_to_idx, embed_dim, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   gnn_embeddings, node_name_to_idx, embed_dim, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  gnn_embeddings, node_name_to_idx, embed_dim, True)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )


# ─── Model Components ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout + skip."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FrozenStringGNNBaseline(nn.Module):
    """Frozen STRING_GNN backbone (no conditioning).

    Retrieves precomputed frozen PPI embeddings for in-vocab genes.
    OOV genes (~6.4%) receive a learned fallback embedding.

    Proven best foundation: node1-2 achieved F1=0.4912 (tree best) using this exact backbone.
    No conditioning is applied — both transductive and inductive conditioning variants
    (node1-3-1, node2-1-1-1-1) proved net negative against unconditioned baseline.
    """

    def __init__(self, gnn_dim: int = 256):
        super().__init__()
        self.gnn_dim = gnn_dim

        # OOV fallback embedding for genes not in STRING_GNN vocab (~6.4% of samples)
        self.oov_embedding = nn.Parameter(torch.empty(gnn_dim))
        nn.init.normal_(self.oov_embedding, std=0.02)

    def forward(
        self,
        base_embedding: torch.Tensor,  # [B, 256] - frozen precomputed embeddings
        in_vocab: torch.Tensor,        # [B] bool
    ) -> torch.Tensor:
        """
        Returns frozen embedding for in-vocab genes; learned OOV embedding otherwise.

        Args:
            base_embedding: [B, gnn_dim] precomputed frozen STRING_GNN embeddings
            in_vocab: [B] boolean mask, True = gene is in STRING_GNN vocabulary

        Returns:
            emb: [B, gnn_dim] embeddings ready for the bilinear head
        """
        base_embedding = base_embedding.float()
        emb = base_embedding.clone()

        oov_mask = ~in_vocab
        if oov_mask.any():
            # Replace OOV zero-filled embeddings with the learned OOV parameter
            emb[oov_mask] = self.oov_embedding.unsqueeze(0).expand(oov_mask.sum(), -1).to(emb.device)

        return emb  # [B, 256]


class GNNBilinearHead(nn.Module):
    """6-layer prediction head with bilinear interaction (rank=256).

    KEY CHANGE from parent: n_residual_layers=8 → 6 (proven optimal depth from node1-2).
    The clean isolated ablation in the parent (node2-1-1-1-1-1-1) conclusively demonstrated
    that 8 residual layers is worse than 6 for this 1,416-sample task.

    Mechanisms explaining why 6 > 8 layers:
      - Over-parameterization: ~19M params vs ~16M for 6 layers, with only 1,416 samples
      - Cumulative dropout: 0.8^8 ≈ 16.8% active vs 0.8^6 ≈ 26.2% — over-regularizes
      - Noisier gradients: deeper networks accumulate more gradient noise on small data

    Left side: GNN embedding -> 6-layer MLP -> [B, 3, rank]
    Right side: randomly initialized learnable output gene embeddings [n_genes_out, rank]
    Interaction: einsum("bcr,gr->bcg") -> logits [B, 3, n_genes_out]
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 256,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.2,
        n_residual_layers: int = 6,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.rank = rank

        # Input normalization
        self.input_norm = nn.LayerNorm(gnn_dim)

        # Projection: gnn_dim -> hidden_dim
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)

        # 6-layer residual MLP (REDUCED from 8 — proven optimal depth from node1-2)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim -> n_classes * rank
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank)

        # Output gene embeddings: [n_genes_out, rank] — random init (node1-2 proven approach)
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, rank))

        # Head dropout
        self.head_dropout = nn.Dropout(dropout)

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[GNNBilinearHead] {n_residual_layers} residual blocks, rank={rank}: {n_params:,} params")

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)
        nn.init.normal_(self.out_gene_emb, std=0.02)  # same as node1-2 (random init)

    def forward(self, gnn_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gnn_emb: [B, gnn_dim] - frozen STRING_GNN embeddings
        Returns:
            logits: [B, 3, n_genes_out]
        """
        B = gnn_emb.shape[0]

        x = self.input_norm(gnn_emb)
        x = self.proj_in(x)   # [B, hidden_dim]

        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)   # [B, hidden_dim]

        x = self.head_dropout(x)
        pert_proj = self.proj_bilinear(x)                          # [B, n_classes * rank]
        pert_proj = pert_proj.view(B, self.n_classes, self.rank)   # [B, 3, rank]

        # Bilinear interaction
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, n_genes_out]
        return logits


# ─── Focal Loss (Pure — No Label Smoothing) ───────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Pure focal loss — NO label smoothing.

    Hard-target CE provides maximal gradient signal for minority classes (down/up-regulated)
    that the macro-averaged per-gene F1 metric rewards. Focal weight down-weights the
    dominant neutral class (88.9% of labels) to focus training on hard examples.

    Args:
        logits: [B, C, G] logits
        labels: [B, G] long class indices (0/1/2)
        gamma:  focal exponent (2.0)

    Returns:
        Scalar loss
    """
    B, C, G = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
    labels_flat = labels.reshape(-1)                        # [B*G]

    # Pure CE — no label smoothing (restores hard-target gradient signal)
    ce = F.cross_entropy(
        logits_flat, labels_flat,
        reduction="none",
    )  # [B*G]

    # Focal weight from unsmoothed probabilities — focuses on hard examples
    with torch.no_grad():
        probs = F.softmax(logits_flat, dim=1)
        pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt).pow(gamma)

    return (focal_weight * ce).mean()


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
    dist.all_gather(g_preds,  p)
    dist.all_gather(g_labels, l)

    real_preds  = torch.cat([g_preds[i][:all_sizes[i].item()].cpu()  for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][:all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """LightningModule for gene-perturbation DEG prediction.

    Frozen STRING_GNN backbone (no conditioning) + 6-layer rank=256 bilinear head.
    Pure focal loss (no label smoothing): hard-target gradient signal for minority classes.
    Stronger weight decay (3e-3) as the key differentiation from node1-2 (1e-3).

    Changes from parent (node2-1-1-1-1-1-1, F1=0.4705):
      1. n_residual_layers=6 (was 8): PRIMARY FIX — isolated depth ablation proves 6 is optimal
      2. weight_decay=3e-3 (was 1e-3): KEY DIFFERENTIATION — stronger regularization to
         close the persistent train/val overfitting gap observed across all STRING_GNN nodes
      3. patience=50 (was 80): aligned with proven node1-2 setting; avoids 80-epoch plateau
      4. total_steps calibrated to ~90 epochs (better alignment with patience=50)
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 256,
        n_residual_layers: int = 6,
        dropout: float = 0.2,
        lr: float = 5e-4,
        weight_decay: float = 3e-3,
        focal_gamma: float = 2.0,
        warmup_steps: int = 50,
        total_steps: int = 2000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams

        # Build frozen backbone (no conditioning)
        self.backbone = FrozenStringGNNBaseline(gnn_dim=hp.gnn_dim)

        # Build 6-layer rank=256 bilinear prediction head
        # KEY CHANGES: n_residual_layers=6 (was 8); dropout=0.2 (unchanged)
        self.head = GNNBilinearHead(
            gnn_dim=hp.gnn_dim,
            hidden_dim=hp.hidden_dim,
            rank=hp.rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
        )

        # Cast all trainable parameters to float32 for stable optimization
        for p in self.backbone.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        for p in self.head.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Count parameters
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        total_trainable = backbone_trainable + head_trainable
        print(f"[Setup] Trainable params: "
              f"backbone (oov_embedding)={backbone_trainable:,}, "
              f"head={head_trainable:,}, "
              f"total={total_trainable:,}")
        print(f"[Setup] All STRING_GNN backbone params are FROZEN (no conditioning MLP)")
        print(f"[Setup] Head: {hp.n_residual_layers} layers, rank={hp.rank}, "
              f"dropout={hp.dropout}, label_smoothing=0.0, "
              f"focal_gamma={hp.focal_gamma}, weight_decay={hp.weight_decay}")

    def forward(
        self,
        base_embedding: torch.Tensor,
        in_vocab: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.backbone(base_embedding, in_vocab)  # [B, 256]
        logits = self.head(emb)                         # [B, 3, 6640]
        return logits

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Pure focal loss — no label smoothing
        return focal_loss(
            logits, labels,
            gamma=self.hparams.focal_gamma,
        )

    def training_step(self, batch, batch_idx):
        logits = self(
            batch["base_embedding"].float(),
            batch["in_vocab"],
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(
            batch["base_embedding"].float(),
            batch["in_vocab"],
        )
        if "label" in batch:
            loss = self._compute_loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())
        return logits

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        local_p = torch.cat(self._val_preds,  dim=0)
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
        logits = self(
            batch["base_embedding"].float(),
            batch["in_vocab"],
        )
        probs = torch.softmax(logits, dim=1)  # [B, 3, 6640]
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs = torch.cat(self._test_preds, dim=0)
        dummy_labels = torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        if self._test_labels:
            dummy_labels = torch.cat(self._test_labels, dim=0)

        if self.trainer.world_size > 1:
            all_probs, all_labels = _gather_tensors(local_probs, dummy_labels, self.device, self.trainer.world_size)
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

            self.print(f"[Node2-1-1-1-1-1-1-1] Saved test predictions → {pred_path} ({len(seen_ids)} samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node2-1-1-1-1-1-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Single optimizer group: OOV embedding + 6-layer rank=256 bilinear head
        # (STRING_GNN backbone is fully frozen — no backbone parameters here)
        # KEY CHANGE: weight_decay=3e-3 (was 1e-3) — 3x stronger regularization
        # to address persistent overfitting (loss ratio 2.276 in parent despite dropout=0.2)
        all_trainable = (
            list(self.backbone.parameters()) +
            list(self.head.parameters())
        )

        optimizer = torch.optim.AdamW(
            all_trainable,
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )

        # Cosine annealing with linear warmup
        # total_steps calibrated to ~90 epochs (better aligned with patience=50)
        def lr_lambda(step: int) -> float:
            if step < hp.warmup_steps:
                return float(step) / max(1, hp.warmup_steps)
            progress = (step - hp.warmup_steps) / max(1, hp.total_steps - hp.warmup_steps)
            return max(0.01, 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable params ─────────────────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        trainable_sd = {k: v for k, v in full_sd.items() if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)")
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-1-1-1-1-1-1-1 – Frozen STRING_GNN + 6-layer rank=256 head + stronger WD"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--gnn-dim",            type=int,   default=256)
    p.add_argument("--hidden-dim",         type=int,   default=512)
    p.add_argument("--rank",               type=int,   default=256,
                   help="Bilinear rank (matches STRING_GNN's 256-dim input bottleneck)")
    p.add_argument("--n-residual-layers",  type=int,   default=6,
                   help="Number of residual blocks (6 = proven optimal from node1-2; 8 was worse)")
    p.add_argument("--dropout",            type=float, default=0.2,
                   help="Dropout rate (same as node1-2 proven value)")
    p.add_argument("--lr",                 type=float, default=5e-4,
                   help="Learning rate (proven effective across best nodes)")
    p.add_argument("--weight-decay",       type=float, default=3e-3,
                   help="Weight decay (INCREASED from 1e-3 → 3e-3: key differentiation from node1-2)")
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--warmup-steps",       type=int,   default=50)
    p.add_argument("--micro-batch-size",   type=int,   default=16)
    p.add_argument("--global-batch-size",  type=int,   default=64)
    p.add_argument("--max-epochs",         type=int,   default=300)
    p.add_argument("--patience",           type=int,   default=50,
                   help="Patience=50 (REDUCED from 80): aligned with node1-2's proven optimal")
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataModule
    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    # Estimate total training steps for LR schedule
    # CRITICAL: calibrate total_steps to actual training length for proper LR decay
    # With patience=50, early stopping expected ~50-100 epochs after best.
    # Targeting ~90 epochs for schedule alignment (better matched to patience=50
    # vs prior nodes targeting 120 epochs, where LR expired before best epoch).
    steps_per_epoch = max(1, len(dm.train_ds) // (args.micro_batch_size * n_gpus))
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)
    # Calibrate to ~90 epochs — better aligned with patience=50 expected convergence
    calibrated_total_steps = effective_steps_per_epoch * 90
    # Ensure a minimum of 1000 steps regardless of GPU count
    total_steps = max(1000, calibrated_total_steps)

    print(f"[Main] effective_steps_per_epoch={effective_steps_per_epoch}, "
          f"calibrated total_steps={total_steps} "
          f"(targeting ~90 epochs for LR schedule alignment with patience=50)")
    print(f"[Main] 6-layer rank=256 head (NO label smoothing, weight_decay=3e-3): "
          f"n_residual_layers={args.n_residual_layers}, rank={args.rank}, "
          f"dropout={args.dropout}, focal_gamma={args.focal_gamma}, patience={args.patience}")
    print(f"[Main] Key changes from parent: 8→6 layers (fix), wd 1e-3→3e-3 (stronger regularization), patience 80→50")

    # LightningModule
    lit = PerturbationLitModule(
        gnn_dim=args.gnn_dim,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        n_residual_layers=args.n_residual_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=1, save_last=True,
    )
    es_cb = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.patience,
        min_delta=1e-5,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    pb_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    # Debug / fast-dev-run settings
    max_steps:           int | None   = -1
    limit_train_batches: float | int  = 1.0
    limit_val_batches:   float | int  = 1.0
    limit_test_batches:  float | int  = 1.0
    fast_dev_run = False

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
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
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
            f"Node 2-1-1-1-1-1-1-1 – Frozen STRING_GNN + 6-layer rank=256 head + weight_decay=3e-3\n"
            f"Test results from trainer: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
