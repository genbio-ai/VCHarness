"""
Node 1-1-2-1-1-2 – STRING_GNN Partial Fine-Tuning + Bilinear MLP + SGDR Warm Restarts + SWA

Architecture:
  - Partially fine-tuned STRING_GNN backbone (mps.0-6 frozen, mps.7+post_mp trainable at lr=1e-5)
  - Per-step GNN forward pass (NOT precomputed) to enable backbone gradient flow
  - Deep 6-layer residual bilinear MLP head (hidden_dim=512, expand=4, rank=512, dropout=0.3)
  - Bilinear interaction: pert_emb [B, 256] → head [B, 512] → [B, 3*512] → [B, 3, 512]
                          × out_gene_emb [6640, 512] → logits [B, 3, 6640]
  - Class-weighted focal cross-entropy loss (gamma=2.0, weights=[2.0(down), 0.5(neutral), 4.0(up)])
  - Three-group optimizer:
      * Muon (lr=0.005) for hidden 2D matrices in ResidualBlocks
      * AdamW (lr=5e-4, wd=2e-3) for head projections, embeddings, norms, biases
      * AdamW (lr=1e-5, wd=1e-3) for trainable backbone params (mps.7+post_mp)
  - Cosine Annealing with Warm Restarts (SGDR, T_0=600 steps ≈ 27 epochs/cycle, T_mult=1)
  - Gradient clipping (max_norm=1.0)
  - Quality-filtered SWA: top-K checkpoints by val_f1 saved to disk, averaged at test time

Key improvements over Parent (node1-1-2-1-1, frozen backbone, F1=0.5023):
  1. Partial STRING_GNN fine-tuning (mps.7+post_mp, ~132K backbone params at lr=1e-5):
     All top-performing nodes (>0.51 F1) use partial backbone fine-tuning. The frozen backbone
     (PPI topology only) is the representational ceiling per node1-1-2-1-1-1 feedback:
     "the STRING_GNN frozen backbone is the fundamental ceiling." Unfreezing the last 2 layers
     at a very low lr (1e-5) allows task-specific PPI adaptation without catastrophic forgetting.
  2. SGDR warm restarts (T_0=600 steps, ≈27 epochs/cycle):
     Node1-2-2-2-1 (F1=0.5099) and node1-2-2-3 (F1=0.5101) used warm restarts with T_0=600
     and achieved staircase improvement across cycles. Periodic LR resets help escape local minima
     formed after plateau convergence.
  3. Fixed cosine LR (CosineAnnealingWarmRestarts with eta_min=1e-6):
     Sibling node1-1-2-1-1-1 had an unclamped cosine lambda causing unintended second cycles.
     This node uses the standard scheduler with eta_min=1e-6 to prevent hard LR=0 freeze.
  4. Moderate regularization (dropout=0.3, wd=2e-3):
     From sibling node1-1-2-1-1-1 feedback, these settings pushed best epoch from 31 to 114.
  5. Quality-filtered SWA (top-K checkpoints with val_f1 >= threshold):
     Node2-1-1-1-2-1-1-1-1 (F1=0.5124) showed SWA provided +0.004 over best single checkpoint.
     Saves checkpoints to disk with their val_f1 scores, loads and averages top-K at test time.
  6. rank=512 bilinear head (doubled from parent's 256):
     Validated by node1-2-3 (+0.0057 gain rank=256→512) and all top-performing nodes.

Differentiation from sibling (node1-1-2-1-1-1, F1=0.5029):
  - Sibling: fully frozen backbone, accidental 2nd LR cycle (unclamped lambda), no SWA
  - This node: partial backbone fine-tuning (primary differentiator), intentional SGDR,
    disk-based SWA with quality filtering

Key insights from collected_memory:
  - STRING_GNN partial fine-tuning (mps.7+post_mp) = key to >0.51 F1 across tree
  - Muon lr=0.005 for ResBlock hidden matrices = validated optimal across many nodes
  - Class-weighted focal [down=2.0, neutral=0.5, up=4.0] + gamma=2.0 = proven best config
  - SGDR T_0=600 steps drives staircase improvement (node1-2-2-2-1: 0.5099)
  - Quality-filtered SWA provides +0.003-0.004 rescue (node2-1-1-1-2-1-1-1-1: 0.5124)
  - dropout=0.3, wd=2e-3 proven to push best epoch later (sibling: epoch 31→114)
  - rank=512 consistently better than rank=256
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
STRING_GNN_DIM = 256      # STRING_GNN hidden dimension


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

    Applies per-class weights to address label imbalance (88.9% neutral, 8.1% down, 3.0% up).
    Weights: [down=2.0, neutral=0.5, up=4.0] — proven optimal from node1-2-3 (F1=0.4969).

    Args:
        logits:         [B, C, G] float32 – per-class logits
        targets:        [B, G]    long    – class indices 0..C-1
        class_weights:  [C]       float32 – per-class weight tensor
        gamma:          focusing parameter (0 = standard CE)
        label_smoothing: label smoothing epsilon

    Returns:
        Scalar loss.
    """
    # Standard cross-entropy with class weights and label smoothing
    ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )  # [B, G]

    # Focal modulation: compute pt for focal weighting
    with torch.no_grad():
        log_probs = F.log_softmax(logits, dim=1)  # [B, C, G]
        probs = log_probs.exp()                    # [B, C, G]
        # Gather probability at the true class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B, G]

    focal = (1.0 - pt) ** gamma * ce
    return focal.mean()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset. Labels are optionally present."""

    def __init__(
        self,
        df: pd.DataFrame,
        pert_id_to_gnn_idx: Dict[str, int],
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        # Map pert_id (Ensembl gene ID) to STRING_GNN node index; -1 for unknown
        self.gnn_indices: List[int] = [
            pert_id_to_gnn_idx.get(pid, -1) for pid in self.pert_ids
        ]
        self.has_labels = has_labels
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
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "gnn_idx": self.gnn_indices[idx],
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch: List[dict]) -> dict:
    """Simple collate: stack gnn_idx, labels; keep lists for strings."""
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
    """Single-fold DataModule for perturbation DEG prediction with STRING_GNN."""

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
        # Load STRING_GNN node names to build pert_id → node index mapping
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.pert_id_to_gnn_idx = {name: i for i, name in enumerate(node_names)}

        # Load splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        self.train_ds = PerturbationDataset(dfs["train"], self.pert_id_to_gnn_idx, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   self.pert_id_to_gnn_idx, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  self.pert_id_to_gnn_idx, True)

        # Log OOV statistics
        oov_train = sum(1 for idx in self.train_ds.gnn_indices if idx == -1)
        oov_val   = sum(1 for idx in self.val_ds.gnn_indices   if idx == -1)
        print(f"[DataModule] OOV genes — train: {oov_train}/{len(self.train_ds)}, "
              f"val: {oov_val}/{len(self.val_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
        )


# ─── Model Components ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout + skip."""

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


class GNNBilinearHead(nn.Module):
    """
    Bilinear MLP head on top of STRING_GNN embeddings.

    Architecture:
        gnn_emb [B, 256] → proj_in [B, hidden_dim=512] → 6×ResidualBlock → [B, 512]
        → norm_out → proj_bilinear [B, 3 * rank=512] → reshape [B, 3, 512]
        × out_gene_emb [6640, 512] → logits [B, 3, 6640]
    """

    def __init__(
        self,
        gnn_dim: int = STRING_GNN_DIM,
        hidden_dim: int = 512,
        n_resblocks: int = 6,
        expand: int = 4,
        dropout: float = 0.3,
        rank: int = 512,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()

        # Input projection: LayerNorm + Linear + GELU + Dropout
        self.proj_in = nn.Sequential(
            nn.LayerNorm(gnn_dim),
            nn.Linear(gnn_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Deep residual MLP: 6 blocks (proven optimal depth from node1-2)
        self.resblocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
              for _ in range(n_resblocks)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear interaction head (rank=512 for expanded gene-space capacity)
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank, bias=True)
        self.out_gene_emb = nn.Embedding(n_genes_out, rank)
        nn.init.normal_(self.out_gene_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)

        self.n_classes = n_classes
        self.rank = rank

    def forward(self, gnn_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gnn_emb: [B, 256] STRING_GNN embeddings

        Returns:
            logits: [B, 3, 6640]
        """
        B = gnn_emb.shape[0]

        h = self.proj_in(gnn_emb)               # [B, hidden_dim]
        h = self.resblocks(h)                    # [B, hidden_dim]
        h = self.norm_out(h)                     # [B, hidden_dim]

        proj = self.proj_bilinear(h)             # [B, 3 * rank]
        proj = proj.view(B, self.n_classes, self.rank)  # [B, 3, rank]
        out_emb = self.out_gene_emb.weight       # [6640, rank]
        logits = torch.einsum("bcr,gr->bcg", proj, out_emb)  # [B, 3, 6640]
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
    """LightningModule for gene-perturbation DEG prediction (Node 1-1-2-1-1-2).

    Key innovations over parent (node1-1-2-1-1, F1=0.5023):
    1. Partial STRING_GNN fine-tuning (mps.7+post_mp, ~132K backbone params at lr=1e-5)
    2. Per-step GNN forward pass (not precomputed) to enable backbone gradient flow
    3. SGDR warm restarts (T_0=600 steps) for staircase improvement across cycles
    4. Disk-based quality-filtered SWA of top-K checkpoints (saved by epoch val_f1)
    5. dropout=0.3, wd=2e-3 (from sibling's successful regularization)
    6. rank=512 bilinear head
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        n_resblocks: int = 6,
        expand: int = 4,
        dropout: float = 0.3,
        rank: int = 512,
        lr: float = 5e-4,
        muon_lr: float = 0.005,
        backbone_lr: float = 1e-5,
        weight_decay: float = 2e-3,
        backbone_weight_decay: float = 1e-3,
        warmup_steps: int = 100,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        # Class weights: [down=-1, neutral=0, up=+1] → class indices [0,1,2]
        class_weight_down: float = 2.0,
        class_weight_neutral: float = 0.5,
        class_weight_up: float = 4.0,
        sgdr_t0: int = 600,          # Steps per SGDR cycle (~27 epochs/cycle)
        sgdr_t_mult: int = 1,        # Cycle multiplier (keep constant)
        sgdr_eta_min: float = 1e-6,  # Non-zero to prevent hard LR freeze
        grad_clip_norm: float = 1.0,
        # SWA: disk-based quality-filtered checkpoint averaging
        swa_start_epoch: int = 30,
        swa_freq: int = 10,
        swa_top_k: int = 10,
        swa_threshold: float = 0.49,
        swa_dir: str = "",  # Set in main()
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

        # SWA state: list of (val_f1, path) for quality-filtered averaging
        # Stored on the module to persist across train/test phases
        self._swa_pool: List[Tuple[float, str]] = []
        self._current_val_f1: float = 0.0

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams

        # ── Load STRING_GNN with partial fine-tuning ───────────────────────
        # Key change: load full model, freeze early layers, unfreeze mps.7+post_mp
        self.gnn_model = AutoModel.from_pretrained(
            str(STRING_GNN_DIR), trust_remote_code=True
        )

        # Freeze all backbone params first
        for param in self.gnn_model.parameters():
            param.requires_grad = False

        # Selectively unfreeze mps.7 and post_mp for task-specific adaptation
        for name, param in self.gnn_model.named_parameters():
            if "mps.7" in name or "post_mp" in name:
                param.requires_grad = True
                param.data = param.data.float()  # Ensure float32 for stable optimization

        backbone_trainable = sum(
            p.numel() for p in self.gnn_model.parameters() if p.requires_grad
        )
        print(f"[Setup] STRING_GNN backbone: {backbone_trainable:,} trainable params "
              f"(mps.7+post_mp only; mps.0-6+emb frozen)")

        # Load graph data once (fixed topology; only the GNN weights are updated)
        graph = torch.load(
            STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False
        )
        self.register_buffer("edge_index", graph["edge_index"])
        ew = graph.get("edge_weight", None)
        if ew is not None:
            self.register_buffer("edge_weight", ew)
        # If no edge_weight, self.edge_weight won't exist; _run_gnn_forward checks with hasattr

        # ── Register class weights as buffer ───────────────────────────────
        class_weights = torch.tensor(
            [hp.class_weight_down, hp.class_weight_neutral, hp.class_weight_up],
            dtype=torch.float32,
        )
        self.register_buffer("class_weights", class_weights)

        # ── Build prediction head (rank=512 for expanded gene capacity) ─────
        self.model = GNNBilinearHead(
            gnn_dim=STRING_GNN_DIM,
            hidden_dim=hp.hidden_dim,
            n_resblocks=hp.n_resblocks,
            expand=hp.expand,
            dropout=hp.dropout,
            rank=hp.rank,
        )

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Setup SWA directory
        if hp.swa_dir:
            self._swa_dir = Path(hp.swa_dir)
        else:
            self._swa_dir = Path(__file__).parent / "run" / "swa_pool"
        self._swa_dir.mkdir(parents=True, exist_ok=True)

    def _run_gnn_forward(self) -> torch.Tensor:
        """Run the STRING_GNN forward pass to get per-node embeddings.

        The frozen layers (mps.0-6, emb) are run without gradient tracking.
        The trainable layers (mps.7, post_mp) are run with gradient tracking.

        Returns:
            all_embs: [18870, 256] float32 tensor of ALL node embeddings
        """
        edge_weight = self.edge_weight if hasattr(self, 'edge_weight') else None
        outputs = self.gnn_model(
            edge_index=self.edge_index,
            edge_weight=edge_weight,
        )
        return outputs.last_hidden_state  # [18870, 256]

    def _get_gnn_emb(self, gnn_idx: torch.Tensor, all_embs: torch.Tensor) -> torch.Tensor:
        """Look up STRING_GNN embeddings for each sample.

        Args:
            gnn_idx: [B] long tensor of GNN node indices; -1 for OOV
            all_embs: [18870, 256] full GNN embedding matrix

        Returns:
            emb: [B, 256] float32 tensor
        """
        valid_mask = gnn_idx >= 0  # [B]
        safe_idx = gnn_idx.clone()
        safe_idx[~valid_mask] = 0

        emb = all_embs[safe_idx]  # [B, 256]

        # Replace OOV entries with mean embedding
        if (~valid_mask).any():
            oov_emb = all_embs.mean(dim=0, keepdim=True).expand(
                (~valid_mask).sum(), -1
            )
            emb = emb.clone()
            emb[~valid_mask] = oov_emb.to(emb.dtype)

        return emb.float()

    def forward(self, gnn_idx: torch.Tensor) -> torch.Tensor:
        all_embs = self._run_gnn_forward()
        emb = self._get_gnn_emb(gnn_idx, all_embs)
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
        self._current_val_f1 = f1

        # --- Disk-based quality-filtered SWA pool management ---
        # Save snapshots at regular intervals if quality threshold met
        hp = self.hparams
        current_epoch = self.current_epoch
        if (
            self.trainer.is_global_zero
            and current_epoch >= hp.swa_start_epoch
            and (current_epoch - hp.swa_start_epoch) % hp.swa_freq == 0
            and f1 >= hp.swa_threshold
        ):
            ckpt_path = self._swa_dir / f"swa_epoch{current_epoch:04d}_f1{f1:.4f}.pt"
            # Save only trainable params (backbone mps.7+post_mp + head)
            swa_state = {}
            for name, param in self.gnn_model.named_parameters():
                if param.requires_grad:
                    swa_state[f"gnn_model.{name}"] = param.data.cpu().clone()
            for name, param in self.model.named_parameters():
                swa_state[f"model.{name}"] = param.data.cpu().clone()
            torch.save({"f1": f1, "epoch": current_epoch, "state": swa_state},
                       str(ckpt_path))
            self._swa_pool.append((f1, str(ckpt_path)))
            # Keep only top-K by val_f1 — remove low-quality checkpoints from disk
            self._swa_pool.sort(key=lambda x: x[0], reverse=True)
            if len(self._swa_pool) > hp.swa_top_k:
                # Remove the lowest-quality checkpoint
                removed_f1, removed_path = self._swa_pool.pop()
                try:
                    Path(removed_path).unlink(missing_ok=True)
                except Exception:
                    pass
            print(f"[SWA] Epoch {current_epoch}: saved checkpoint (val_f1={f1:.4f}), "
                  f"pool size={len(self._swa_pool)}, "
                  f"pool range: {self._swa_pool[-1][0]:.4f}–{self._swa_pool[0][0]:.4f}")

        self._val_preds.clear()
        self._val_labels.clear()

    def apply_swa(self):
        """Load and average all SWA pool checkpoints into the current model.

        This is called before test evaluation to apply the SWA averaged weights.
        Uses uniform averaging across all quality-filtered checkpoints.
        """
        if not self._swa_pool:
            print("[SWA] No SWA pool available — using current weights as-is")
            return

        # Load all checkpoints from disk
        loaded = []
        for f1_score_val, path in self._swa_pool:
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                loaded.append((f1_score_val, data["state"]))
            except Exception as e:
                print(f"[SWA] Warning: failed to load {path}: {e}")

        if not loaded:
            print("[SWA] All checkpoint loads failed, skipping SWA")
            return

        print(f"[SWA] Averaging {len(loaded)} checkpoints "
              f"(val_f1 range: {loaded[-1][0]:.4f}–{loaded[0][0]:.4f})")

        # Compute element-wise mean across all checkpoints
        avg_state = {}
        for key in loaded[0][1].keys():
            stacked = torch.stack([state[key].float() for _, state in loaded], dim=0)
            avg_state[key] = stacked.mean(dim=0)

        # Load averaged parameters into models
        for name, param in self.gnn_model.named_parameters():
            key = f"gnn_model.{name}"
            if key in avg_state and param.requires_grad:
                param.data.copy_(avg_state[key].to(param.device).to(param.dtype))

        for name, param in self.model.named_parameters():
            key = f"model.{name}"
            if key in avg_state:
                param.data.copy_(avg_state[key].to(param.device).to(param.dtype))

        print("[SWA] SWA weight averaging applied successfully")

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
                f"[Node1-1-2-1-1-2] Saved test predictions → {pred_path} "
                f"({len(seen_ids)} unique samples)"
            )

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-1-2-1-1-2] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        """
        Three-group optimizer: Muon + AdamW (head) + AdamW (backbone).

        Group 1 - Muon (lr=0.005): Hidden 2D weight matrices in ResidualBlocks
          Orthogonalized momentum for MLP hidden layers.
          Validated as optimal across many nodes (node1-1-2-1-1: 0.5023, etc.).

        Group 2 - AdamW (lr=5e-4, wd=2e-3): Head projections, embeddings, norms, biases

        Group 3 - AdamW (lr=1e-5, wd=1e-3): Backbone trainable params (mps.7+post_mp)
          Very low lr to prevent catastrophic forgetting while enabling task adaptation.
          Validated in node1-2-2-2-1 (F1=0.5099), node1-2-2-3 (F1=0.5101).

        LR Schedule: CosineAnnealingWarmRestarts (SGDR)
          T_0=600 steps, T_mult=1, eta_min=1e-6
          Warm restarts enable staircase improvement; eta_min prevents hard LR=0 freeze
          (fixing the bug in sibling node1-1-2-1-1-1 where progress was not clamped).
        """
        hp = self.hparams

        # ── Identify three parameter groups ──────────────────────────────
        muon_params = []      # Muon: ResidualBlock 2D hidden matrices
        adamw_head_params = [] # AdamW: all other head params
        adamw_backbone_params = []  # AdamW: backbone mps.7+post_mp

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_resblock_linear_weight = (
                "resblocks" in name
                and "weight" in name
                and param.ndim >= 2
                and "norm" not in name
            )
            if is_resblock_linear_weight:
                muon_params.append(param)
            else:
                adamw_head_params.append(param)

        for name, param in self.gnn_model.named_parameters():
            if param.requires_grad:
                adamw_backbone_params.append(param)

        print(f"[Optimizer] Muon params: {sum(p.numel() for p in muon_params):,} "
              f"(lr={hp.muon_lr})")
        print(f"[Optimizer] AdamW head params: {sum(p.numel() for p in adamw_head_params):,} "
              f"(lr={hp.lr})")
        print(f"[Optimizer] AdamW backbone params: "
              f"{sum(p.numel() for p in adamw_backbone_params):,} "
              f"(lr={hp.backbone_lr})")

        param_groups = [
            # Muon group: hidden weight matrices in ResidualBlocks
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            # AdamW group: head projections, embeddings, norms, biases
            dict(
                params=adamw_head_params,
                use_muon=False,
                lr=hp.lr,
                betas=(0.9, 0.999),
                eps=1e-10,
                weight_decay=hp.weight_decay,
            ),
            # AdamW group: backbone trainable params (mps.7 + post_mp)
            dict(
                params=adamw_backbone_params,
                use_muon=False,
                lr=hp.backbone_lr,
                betas=(0.9, 0.999),
                eps=1e-10,
                weight_decay=hp.backbone_weight_decay,
            ),
        ]

        # Use distributed-aware optimizer
        if dist.is_available() and dist.is_initialized():
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        # SGDR: Cosine Annealing with Warm Restarts
        # T_0=600 steps ≈ 27 epochs/cycle with effective_steps ≈ 22/epoch (2 GPUs)
        # eta_min=1e-6 prevents hard LR=0 freeze (key fix from sibling's feedback)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.sgdr_t0,
            T_mult=hp.sgdr_t_mult,
            eta_min=hp.sgdr_eta_min,
        )

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
        description="Node 1-1-2-1-1-2 – STRING_GNN Partial Fine-Tuning + SGDR + SWA"
    )
    p.add_argument("--data-dir",                  type=str,   default="data")
    p.add_argument("--hidden-dim",                type=int,   default=512)
    p.add_argument("--n-resblocks",               type=int,   default=6)
    p.add_argument("--expand",                    type=int,   default=4)
    p.add_argument("--dropout",                   type=float, default=0.3)
    p.add_argument("--rank",                      type=int,   default=512)
    p.add_argument("--lr",                        type=float, default=5e-4)
    p.add_argument("--muon-lr",                   type=float, default=0.005)
    p.add_argument("--backbone-lr",               type=float, default=1e-5)
    p.add_argument("--weight-decay",              type=float, default=2e-3)
    p.add_argument("--backbone-weight-decay",     type=float, default=1e-3)
    p.add_argument("--warmup-steps",              type=int,   default=100)
    p.add_argument("--focal-gamma",               type=float, default=2.0)
    p.add_argument("--label-smoothing",           type=float, default=0.05)
    p.add_argument("--class-weight-down",         type=float, default=2.0)
    p.add_argument("--class-weight-neutral",      type=float, default=0.5)
    p.add_argument("--class-weight-up",           type=float, default=4.0)
    p.add_argument("--grad-clip-norm",            type=float, default=1.0)
    # SGDR warm restart params
    p.add_argument("--sgdr-t0",                   type=int,   default=600)
    p.add_argument("--sgdr-t-mult",               type=int,   default=1)
    p.add_argument("--sgdr-eta-min",              type=float, default=1e-6)
    # SWA params
    p.add_argument("--swa-start-epoch",           type=int,   default=30)
    p.add_argument("--swa-freq",                  type=int,   default=10)
    p.add_argument("--swa-top-k",                 type=int,   default=10)
    p.add_argument("--swa-threshold",             type=float, default=0.49)
    p.add_argument("--micro-batch-size",          type=int,   default=16)
    p.add_argument("--global-batch-size",         type=int,   default=128)
    p.add_argument("--max-epochs",                type=int,   default=300)
    p.add_argument("--patience",                  type=int,   default=100)
    p.add_argument("--num-workers",               type=int,   default=4)
    p.add_argument("--val-check-interval",        type=float, default=1.0)
    p.add_argument("--debug-max-step",            type=int,   default=None)
    p.add_argument("--fast-dev-run",              action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)
    swa_dir = str(out_dir / "swa_pool")

    # DataModule
    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # Compute effective batch info for logging
    _train_df_size = pd.read_csv(
        Path(args.data_dir) / "train.tsv", sep="\t", usecols=["pert_id"]
    ).shape[0]
    steps_per_epoch = _train_df_size // (args.micro_batch_size * n_gpus)
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)

    print(f"[Main] GPUs={n_gpus}, effective_steps_per_epoch={effective_steps_per_epoch}")
    print(f"[Main] SGDR T_0={args.sgdr_t0} steps "
          f"≈ {args.sgdr_t0/effective_steps_per_epoch:.1f} epochs/cycle")
    print(f"[Main] Partial backbone: mps.7+post_mp at backbone_lr={args.backbone_lr}")

    lit = PerturbationLitModule(
        hidden_dim=args.hidden_dim,
        n_resblocks=args.n_resblocks,
        expand=args.expand,
        dropout=args.dropout,
        rank=args.rank,
        lr=args.lr,
        muon_lr=args.muon_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        backbone_weight_decay=args.backbone_weight_decay,
        warmup_steps=args.warmup_steps,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        class_weight_down=args.class_weight_down,
        class_weight_neutral=args.class_weight_neutral,
        class_weight_up=args.class_weight_up,
        sgdr_t0=args.sgdr_t0,
        sgdr_t_mult=args.sgdr_t_mult,
        sgdr_eta_min=args.sgdr_eta_min,
        grad_clip_norm=args.grad_clip_norm,
        swa_start_epoch=args.swa_start_epoch,
        swa_freq=args.swa_freq,
        swa_top_k=args.swa_top_k,
        swa_threshold=args.swa_threshold,
        swa_dir=swa_dir,
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
    max_steps: int           = -1
    limit_train_batches: float | int = 1.0
    limit_val_batches:   float | int = 1.0
    limit_test_batches:  float | int = 1.0
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

    # ── Test phase: apply SWA (if available) then evaluate ─────────────────
    # Strategy:
    # - If SWA pool is available AND not debug mode: apply SWA then test without ckpt_path
    # - If no SWA pool: load best checkpoint and test as usual
    #
    # Rationale: SWA averaging is applied IN-PLACE on the model.
    # The best checkpoint is implicitly represented in the SWA pool (checkpoints are
    # saved during training when val_f1 >= threshold). Applying SWA gives a better
    # ensemble solution than any single checkpoint.
    # ── Synchronize use_swa decision across all DDP ranks ────────────────────
    # Only rank 0 populates _swa_pool (guarded by is_global_zero in training).
    # We must broadcast this boolean before the if-branch so all ranks take the
    # same path; otherwise ranks diverge, causing a DDP deadlock/NCCL timeout.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    _cuda_device = torch.device(f"cuda:{local_rank}")

    _use_swa_local = (
        len(lit._swa_pool) > 0
        and not fast_dev_run
        and args.debug_max_step is None
    )
    if dist.is_available() and dist.is_initialized():
        # Broadcast the decision from rank 0 (the only rank with the truth)
        _use_swa_tensor = torch.tensor(
            [int(_use_swa_local)], dtype=torch.int32, device=_cuda_device
        )
        dist.broadcast(_use_swa_tensor, src=0)
        use_swa = bool(_use_swa_tensor.item())
    else:
        use_swa = _use_swa_local

    if use_swa:
        # Apply SWA on rank 0 only; other ranks will receive weights via broadcast
        if trainer.is_global_zero:
            print(f"\n[Main] Applying SWA over {len(lit._swa_pool)} checkpoints "
                  f"before test evaluation...")
            # Ensure the model is on the correct CUDA device before averaging,
            # because PL may move the model to CPU after trainer.fit() completes.
            lit.gnn_model.to(_cuda_device)
            lit.model.to(_cuda_device)
            lit.apply_swa()

        # Broadcast SWA-averaged parameters from rank 0 to all other DDP ranks.
        # IMPORTANT: tensors MUST be on CUDA for NCCL; move them first if needed.
        if trainer.world_size > 1 and dist.is_available() and dist.is_initialized():
            for param in lit.gnn_model.parameters():
                if param.requires_grad:
                    if param.data.device.type != "cuda":
                        param.data = param.data.to(_cuda_device)
                    dist.broadcast(param.data, src=0)
            for param in lit.model.parameters():
                if param.data.device.type != "cuda":
                    param.data = param.data.to(_cuda_device)
                dist.broadcast(param.data, src=0)

        # Test with SWA-averaged weights (no ckpt_path — use current model state)
        test_results = trainer.test(lit, datamodule=dm, ckpt_path=None)
        if trainer.is_global_zero:
            print(f"[Main] SWA test results: {test_results}")
    else:
        # No SWA: load best checkpoint
        ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
        test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt_path)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"Node 1-1-2-1-1-2 – STRING_GNN Partial Fine-Tuning + SGDR + SWA\n"
            f"Architecture: mps.7+post_mp trainable (lr={args.backbone_lr}), "
            f"rank=512 bilinear head, dropout=0.3\n"
            f"Optimizer: Muon lr={args.muon_lr} + AdamW lr={args.lr} + "
            f"backbone AdamW lr={args.backbone_lr}\n"
            f"Schedule: SGDR T_0={args.sgdr_t0} steps, T_mult={args.sgdr_t_mult}, "
            f"eta_min={args.sgdr_eta_min}\n"
            f"SWA: pool={len(lit._swa_pool)} checkpoints\n"
            f"Best-ckpt test results: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
