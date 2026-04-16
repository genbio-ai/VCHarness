"""
Node 1-1-2-1-1-1 – STRING_GNN + Deep Bilinear MLP Head + Stronger Regularization + Rank=512

Architecture:
  - Frozen STRING_GNN backbone (18,870 nodes, 256-dim PPI topology embeddings)
  - Pre-computed once at setup; no per-step GNN forward pass
  - Deep 6-layer residual bilinear MLP head (hidden_dim=512, expand=4, rank=512)
  - Bilinear interaction: pert_emb [B, 256] → head [B, 512] → [B, 3*512] → [B, 3, 512]
                          × out_gene_emb [6640, 512] → logits [B, 3, 6640]
  - Class-weighted focal cross-entropy loss (gamma=2.0, weights=[2.0(down), 0.5(neutral), 4.0(up)])
  - Muon optimizer (lr=0.005) for hidden MLP weight matrices
  - AdamW (lr=5e-4, weight_decay=2e-3) for embeddings/norms/biases/input_output layers
  - Cosine annealing LR with 100-step warmup, total_steps calibrated to 80-epoch horizon
  - Gradient clipping (max_norm=1.0)

Key improvements over Parent (Node 1-1-2-1-1, F1=0.5023 — tree best):
  1. Stronger regularization — dropout=0.2→0.3 + weight_decay=1e-3→2e-3:
     Parent peaked at epoch 31 out of 250, with train/val loss gap widening from -0.109
     to -0.218 over 80 epochs. This is the primary bottleneck: insufficient regularization
     causing early-onset overfitting. Dropout=0.3 slows memorization; weight_decay=2e-3
     applies stronger L2 to AdamW params (embeddings, projections). Expected to push best
     epoch from 31 → 50-80 range, improving peak val F1 by +0.002-0.005.

  2. Increased rank (256→512):
     Out gene embedding [6640, 256→512] doubles the gene-space representational capacity.
     Inspired by node1-2-3 (F1=0.4969) which used rank=512 with AdamW-only and was the
     previous tree best. The bilinear decomposition with rank=512 can capture more complex
     perturbation-to-gene interactions. Adds ~2.1M trainable params (total ~16.9M).

  3. Cosine schedule calibrated to 80-epoch horizon (vs parent's 150):
     Parent peaked at epoch 31; with stronger regularization, expected peak is 50-80.
     Calibrating max_steps_total to 80×11=880 steps ensures near-zero LR exactly at the
     expected peak zone, enabling the secondary low-LR fine-tuning phase to be most
     effectively utilized in the productive learning window.

  4. Mild label smoothing (0.0→0.05):
     The parent's progressive overfitting manifests in overconfident probability calibration
     (val F1 stable while val loss increases). Mild label smoothing (0.05) encourages more
     calibrated probabilities without degrading per-gene argmax classification decisions.
     Works synergistically with class weights to guide minority class calibration.

Key insights from collected_memory:
  - STRING_GNN frozen backbone + deep bilinear head = proven tree architecture (F1=0.4912→0.5023)
  - Class-weighted focal [down=2.0, neutral=0.5, up=4.0] + gamma=2.0 → node1-2-3 achieved 0.4969
  - Muon lr=0.005 is now validated correct (parent eliminates oscillation, confirms this is optimal)
  - Dropout + weight_decay are the most actionable fix for early-onset overfitting
  - node1-2-3 (rank=512) achieved 0.4969 vs node1-2 (rank=256, 0.4912): rank helps
  - 6 residual layers is optimal (8 layers hurt performance)
  - patience=80-100 captures full secondary improvement phase
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
    Weights reduce gradient dominance of neutral class and amplify minority class gradients.

    Inspired by node1-2-3 (tree best: 0.4969) which used class_weights=[down=2.0, neutral=0.5, up=4.0]
    with gamma=2.0 to great effect. Combined with mild label smoothing (0.05) to reduce
    overconfident probability calibration observed in parent node1-1-2-1-1.

    Args:
        logits:         [B, C, G] float32 – per-class logits
        targets:        [B, G]    long    – class indices 0..C-1
        class_weights:  [C]       float32 – per-class weight tensor
        gamma:          focusing parameter (0 = standard CE)
        label_smoothing: label smoothing epsilon

    Returns:
        Scalar loss.
    """
    # Compute standard cross-entropy with label smoothing (reduction='none')
    # class_weight applied via weight parameter of cross_entropy
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
        # node_names[i] is the Ensembl gene ID (pert_id) for STRING_GNN node i
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
    Pure STRING_GNN-based perturbation predictor (256-dim input).

    Architecture (improved from node1-1-2-1-1 with rank=256 → rank=512):
        gnn_emb [B, 256] → proj_in [B, hidden_dim=512] → 6×ResidualBlock → [B, 512]
        → norm_out → proj_bilinear [B, 3 * rank=512] → reshape [B, 3, 512]
        × out_gene_emb [6640, 512] → logits [B, 3, 6640]

    Rank=512 doubles gene-space capacity vs parent's rank=256.
    Inspired by node1-2-3 (rank=512, F1=0.4969 = previous tree best).
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

        # Bilinear interaction head (rank=512 vs parent's 256)
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

        # Project and process through residual MLP
        h = self.proj_in(gnn_emb)               # [B, hidden_dim]
        h = self.resblocks(h)                    # [B, hidden_dim]
        h = self.norm_out(h)                     # [B, hidden_dim]

        # Bilinear interaction: [B, hidden_dim] → [B, 3, rank] × [6640, rank].T
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
    """LightningModule for gene-perturbation DEG prediction (Node 1-1-2-1-1-1).

    Key innovations over parent (node1-1-2-1-1, F1=0.5023):
    - Stronger regularization: dropout=0.3 (vs 0.2) + weight_decay=2e-3 (vs 1e-3)
    - Increased rank=512 (vs 256): doubles gene-space bilinear capacity
    - Cosine schedule calibrated to 80-epoch horizon (vs 150)
    - Mild label smoothing=0.05 (vs 0.0) for better probability calibration
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        n_resblocks: int = 6,
        expand: int = 4,
        dropout: float = 0.3,   # Increased from parent's 0.2
        rank: int = 512,         # Increased from parent's 256
        lr: float = 5e-4,
        muon_lr: float = 0.005,
        weight_decay: float = 2e-3,  # Increased from parent's 1e-3
        warmup_steps: int = 100,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,  # Mild smoothing (vs parent's 0.0)
        # Class weights: [down=-1, neutral=0, up=+1] mapped to [0,1,2]
        # Based on node1-2-3 (F1=0.4969): [down=2.0, neutral=0.5, up=4.0]
        class_weight_down: float = 2.0,
        class_weight_neutral: float = 0.5,
        class_weight_up: float = 4.0,
        max_steps_total: int = 880,  # Calibrated to ~80 epochs × 11 effective steps
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

        # ── Load frozen STRING_GNN embeddings ──────────────────────────────
        gnn_model = AutoModel.from_pretrained(
            str(STRING_GNN_DIR), trust_remote_code=True
        )
        gnn_model.eval()
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)

        with torch.no_grad():
            outputs = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
        gnn_embs = outputs.last_hidden_state.detach()  # [18870, 256]

        # Register frozen GNN embeddings as a buffer (no grad)
        self.register_buffer("gnn_embs", gnn_embs)

        # Compute mean embedding for OOV fallback
        oov_emb = gnn_embs.mean(dim=0, keepdim=True)  # [1, 256]
        self.register_buffer("oov_emb", oov_emb)

        # ── Register class weights as buffer ───────────────────────────────
        # Class mapping: 0=down-regulated, 1=unchanged, 2=up-regulated
        class_weights = torch.tensor(
            [hp.class_weight_down, hp.class_weight_neutral, hp.class_weight_up],
            dtype=torch.float32,
        )
        self.register_buffer("class_weights", class_weights)

        # ── Build prediction head ───────────────────────────────────────────
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

    def _get_gnn_emb(self, gnn_idx: torch.Tensor) -> torch.Tensor:
        """Look up STRING_GNN embeddings for each sample.

        Args:
            gnn_idx: [B] long tensor of STRING_GNN node indices; -1 for OOV

        Returns:
            emb: [B, 256] float32 tensor
        """
        valid_mask = gnn_idx >= 0  # [B]

        # Safe index (clamp -1 to 0, overwrite later)
        safe_idx = gnn_idx.clone()
        safe_idx[~valid_mask] = 0

        emb = self.gnn_embs[safe_idx]  # [B, 256]

        # Replace OOV entries with mean fallback
        if (~valid_mask).any():
            emb = emb.clone()
            emb[~valid_mask] = self.oov_emb.expand(
                (~valid_mask).sum(), -1
            ).to(emb.dtype)

        return emb.float()

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
                f"[Node1-1-2-1-1-1] Saved test predictions → {pred_path} "
                f"({len(seen_ids)} unique samples)"
            )

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-1-2-1-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        """
        Muon + AdamW optimizer configuration.

        UNCHANGED from parent (node1-1-2-1-1):
        - Muon LR=0.005 (validated as correct; eliminates oscillation that plagued lr=0.02)
        - AdamW LR=5e-4

        CHANGED from parent:
        - weight_decay: 1e-3 → 2e-3 (stronger L2 regularization to combat early-onset overfitting)

        Muon is applied to hidden weight matrices (2D params in ResidualBlocks).
        AdamW is applied to all other parameters:
          - LayerNorm gains/biases (1D)
          - proj_in (first layer — not suitable for Muon per skill docs)
          - out_gene_emb (embedding — not suitable for Muon)
          - proj_bilinear (output head — not suitable for Muon)
          - all biases
        """
        hp = self.hparams

        # Identify Muon vs AdamW parameters
        # Muon: hidden 2D weight matrices in ResidualBlocks (NOT first/last layers)
        muon_params = []
        adamw_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Muon: hidden 2D weight matrices in residual blocks (expand and contract)
            # NOT: proj_in (input projection), proj_bilinear (output), out_gene_emb, norms, biases
            is_resblock_linear_weight = (
                "resblocks" in name
                and "weight" in name
                and param.ndim >= 2
                and "norm" not in name
            )
            if is_resblock_linear_weight:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        print(f"[Optimizer] Muon params: {sum(p.numel() for p in muon_params):,} "
              f"(lr={hp.muon_lr})")
        print(f"[Optimizer] AdamW params: {sum(p.numel() for p in adamw_params):,} "
              f"(lr={hp.lr}, weight_decay={hp.weight_decay})")

        param_groups = [
            # Muon group: hidden weight matrices in ResidualBlocks
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,       # 0.005 (validated optimal from parent)
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            # AdamW group: input projection, bilinear head, embeddings, norms, biases
            dict(
                params=adamw_params,
                use_muon=False,
                lr=hp.lr,
                betas=(0.9, 0.999),
                eps=1e-10,
                weight_decay=hp.weight_decay,  # 2e-3 (increased from parent's 1e-3)
            ),
        ]

        # Use MuonWithAuxAdam for distributed training, SingleDeviceMuonWithAuxAdam otherwise
        if dist.is_available() and dist.is_initialized():
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        # Cosine annealing LR schedule with linear warmup
        # CHANGE: total_steps calibrated to ~80 epochs (expected peak with stronger regularization)
        # Parent peaked at epoch 31; with dropout=0.3 + weight_decay=2e-3, expect peak at 50-80
        # Calibrating to 80 epochs ensures near-zero LR at the expected peak zone
        def lr_lambda(current_step: int):
            if current_step < hp.warmup_steps:
                return float(current_step) / max(1, hp.warmup_steps)
            progress = float(current_step - hp.warmup_steps) / max(
                1, hp.max_steps_total - hp.warmup_steps
            )
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
        description="Node 1-1-2-1-1-1 – STRING_GNN + Deep Bilinear MLP Head + Stronger Regularization + Rank=512"
    )
    p.add_argument("--data-dir",              type=str,   default="data")
    p.add_argument("--hidden-dim",            type=int,   default=512)
    p.add_argument("--n-resblocks",           type=int,   default=6)
    p.add_argument("--expand",                type=int,   default=4)
    # CHANGE: dropout=0.3 (increased from parent's 0.2) — primary regularization fix
    p.add_argument("--dropout",               type=float, default=0.3)
    # CHANGE: rank=512 (increased from parent's 256) — expanded gene embedding capacity
    p.add_argument("--rank",                  type=int,   default=512)
    p.add_argument("--lr",                    type=float, default=5e-4)
    # Muon LR=0.005 (unchanged from parent — validated optimal)
    p.add_argument("--muon-lr",               type=float, default=0.005)
    # CHANGE: weight_decay=2e-3 (increased from parent's 1e-3) — stronger L2 regularization
    p.add_argument("--weight-decay",          type=float, default=2e-3)
    p.add_argument("--warmup-steps",          type=int,   default=100)
    # Class-weighted focal loss parameters (unchanged from parent — proven optimal)
    p.add_argument("--focal-gamma",           type=float, default=2.0)
    # CHANGE: label_smoothing=0.05 (vs parent's 0.0) — mild smoothing for better calibration
    p.add_argument("--label-smoothing",       type=float, default=0.05)
    p.add_argument("--class-weight-down",     type=float, default=2.0)
    p.add_argument("--class-weight-neutral",  type=float, default=0.5)
    p.add_argument("--class-weight-up",       type=float, default=4.0)
    p.add_argument("--grad-clip-norm",        type=float, default=1.0)
    p.add_argument("--micro-batch-size",      type=int,   default=16)
    p.add_argument("--global-batch-size",     type=int,   default=128)
    # Extended training budget — stronger regularization allows longer convergence
    p.add_argument("--max-epochs",            type=int,   default=300)
    p.add_argument("--patience",              type=int,   default=100)
    p.add_argument("--num-workers",           type=int,   default=4)
    p.add_argument("--val-check-interval",    type=float, default=1.0)
    p.add_argument("--debug-max-step",        type=int,   default=None)
    p.add_argument("--fast-dev-run",          action="store_true", default=False)
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

    # Compute total training steps for LR scheduler
    # CHANGE: Calibrated to ~80 epochs (expected peak region with stronger regularization)
    # Parent peaked at epoch 31 (dropout=0.2, weight_decay=1e-3).
    # With dropout=0.3 and weight_decay=2e-3, the model is expected to converge
    # more slowly, peaking around epoch 50-80. Setting max_steps_total to 80×steps/epoch
    # ensures the cosine schedule reaches near-zero precisely at the expected peak zone,
    # enabling the secondary low-LR fine-tuning phase to be maximally effective.
    _train_df_size = pd.read_csv(
        Path(args.data_dir) / "train.tsv", sep="\t", usecols=["pert_id"]
    ).shape[0]
    steps_per_epoch = _train_df_size // (args.micro_batch_size * n_gpus)
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)

    # Calibrate total_steps to 80-epoch realistic training horizon
    # (based on expected peak of regularized model)
    target_epochs_for_schedule = min(80, args.max_epochs)
    max_steps_total = effective_steps_per_epoch * target_epochs_for_schedule

    print(f"[Main] effective_steps_per_epoch={effective_steps_per_epoch}, "
          f"max_steps_total={max_steps_total} (calibrated to {target_epochs_for_schedule} epochs)")

    lit = PerturbationLitModule(
        hidden_dim=args.hidden_dim,
        n_resblocks=args.n_resblocks,
        expand=args.expand,
        dropout=args.dropout,
        rank=args.rank,
        lr=args.lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        class_weight_down=args.class_weight_down,
        class_weight_neutral=args.class_weight_neutral,
        class_weight_up=args.class_weight_up,
        max_steps_total=max(max_steps_total, 1),
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

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"Node 1-1-2-1-1-1 – STRING_GNN + Deep Bilinear MLP Head + Stronger Regularization + Rank=512\n"
            f"Test results from trainer: {test_results}\n"
            f"Hyperparameters: dropout={args.dropout}, rank={args.rank}, "
            f"weight_decay={args.weight_decay}, label_smoothing={args.label_smoothing}, "
            f"muon_lr={args.muon_lr}, lr={args.lr}, "
            f"class_weights=[{args.class_weight_down}, {args.class_weight_neutral}, {args.class_weight_up}], "
            f"max_steps_total={lit.hparams.max_steps_total}\n"
        )
        print(f"[Main] Test score saved → {score_path}")


if __name__ == "__main__":
    main()
