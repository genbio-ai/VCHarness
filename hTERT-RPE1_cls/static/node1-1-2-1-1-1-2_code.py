"""
Node 1-2 – STRING_GNN (mps.6+7+post_mp fine-tuned) + Deep Bilinear MLP Head + Post-hoc SWA

Architecture:
  - STRING_GNN partial backbone: mps.6+mps.7+post_mp trainable at LR=1e-5 (~198K params)
    All earlier layers (emb + mps.0-5) frozen; intermediate buffer precomputed once.
  - Deep 6-layer residual bilinear MLP head (hidden_dim=512, expand=4, rank=512, dropout=0.3)
  - Bilinear interaction: pert_emb [B, 256] → head [B, 512] → [B, 3*512] → [B, 3, 512]
                          × out_gene_emb [6640, 512] → logits [B, 3, 6640]
  - Class-weighted focal cross-entropy loss (gamma=2.0, weights=[2.0(down), 0.5(neutral), 4.0(up)])
  - Three-group optimizer:
      - Muon (lr=0.005) for hidden MLP weight matrices
      - AdamW (lr=5e-4, weight_decay=2e-3) for embeddings/norms/biases/input_output layers
      - AdamW (lr=1e-5, weight_decay=1e-4) for backbone mps.6+7+post_mp
  - Single cosine annealing LR with 100-step warmup, clamped at T_max=130 epochs
    (fixes the unclamped progress bug in parent nodes, prevents LR surge after T_max)
  - Periodic checkpoint collection (every 5 epochs, threshold=val_f1>=0.49)
  - Post-hoc quality-filtered SWA over top-15 checkpoints (temp=3.0)
  - Gradient clipping (max_norm=1.0)
  - Learnable OOV embedding

Key improvements over Parent (Node 1-1-2-1-1-1, F1=0.5029):
  1. Extended partial backbone fine-tuning: mps.6+mps.7+post_mp (~198K backbone params)
     vs. parent's frozen backbone. Sibling (node1-1-2-1-1-1-1) tried mps.7+post_mp (132K)
     and got +0.0006; adding mps.6 doubles the backbone trainable capacity per STRING_GNN
     skill docs recommendation: "freeze emb table, tune mps.6.*, mps.7.*, post_mp.*"

  2. Single cosine schedule with CLAMPED progress (fixes the critical bug in parent):
     The parent's lr_lambda computes progress without clamping at 1.0, causing an
     unintended second LR cycle. This was accidentally beneficial (best epoch 114 at
     second cycle ascending ramp) but caused overfitting surge after epoch 115.
     This node: clamps progress=min(progress,1.0) AND sets T_max=130 epochs, which
     calibrates the LR minimum to the observed productive window end (~epoch 114-130).
     Single-cycle with no warm restarts (sibling showed warm restarts are disruptive).

  3. Post-hoc quality-filtered SWA (inspired by node2-1-1-1-2-1-1-1-1-1-1, F1=0.5180):
     The tree's best SWA node gained +0.0065 over its best single checkpoint. SWA works
     by exponentially weighted averaging of diverse periodic checkpoints (top-15 by val_f1,
     temperature=3.0, threshold=0.490, collected every 5 epochs from epoch 30).
     Key fix over node1-1-2-1-1-2 (F1=0.5069) which used 10-epoch intervals: collecting
     every 5 epochs prevents missing the best checkpoint (which occurred at epoch 59 in
     node1-1-2-1-1-2, causing its SWA pool to contain only sub-best checkpoints).

  4. Learnable OOV embedding (minor improvement):
     Replace static mean-embedding fallback for ~6.4% OOV genes with a learnable
     nn.Parameter initialized to the mean GNN embedding, allowing the optimizer to find
     an optimal OOV representation. Low-risk improvement recommended by sibling feedback.

Key insights from collected_memory:
  - STRING_GNN frozen/partial-FT backbone + bilinear head = proven tree architecture
  - mps.6+7+post_mp is the standard partial FT strategy per STRING_GNN skill docs
  - SWA with SGDR diversity is highly effective: node2-1-1-1-2-1-1-1-1-1-1 at F1=0.5180
  - Warm restarts (sibling node1-1-2-1-1-1-1) caused disruption, single cosine is better
  - Clamping progress prevents unintended LR surge after T_max
  - Muon lr=0.005, AdamW lr=5e-4, dropout=0.3, rank=512 all validated optimal
  - Class weights [2.0, 0.5, 4.0] + gamma=2.0 proven best configuration
  - SWA collection interval must be fine-grained (every 5 epochs) to capture best epoch
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
import copy
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
    STRING_GNN-based perturbation predictor (256-dim input).

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

        # Deep residual MLP: 6 blocks (proven optimal depth)
        self.resblocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
              for _ in range(n_resblocks)]
        )

        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear interaction head (rank=512)
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


# ─── SWA Checkpoint Collection Callback ───────────────────────────────────────

class SWACheckpointCallback(pl.Callback):
    """Collect periodic checkpoints for post-hoc quality-filtered SWA.

    Saves a checkpoint every `swa_freq` epochs when val_f1 >= threshold.
    Checkpoints are saved to <out_dir>/swa_pool/ and tracked in memory.

    Key design to avoid the node1-1-2-1-1-2 bug (collection at 10-epoch
    intervals missed best epoch at 59): use 5-epoch intervals starting from
    epoch 20 to ensure fine-grained coverage of the best-epoch window.
    """

    def __init__(
        self,
        out_dir: Path,
        swa_freq: int = 5,
        start_epoch: int = 20,
        val_f1_threshold: float = 0.490,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.swa_freq = swa_freq
        self.start_epoch = start_epoch
        self.val_f1_threshold = val_f1_threshold
        self.checkpoint_pool: List[Tuple[float, Path]] = []  # (val_f1, ckpt_path)
        self._last_val_f1: float = 0.0

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Capture val_f1 at the END of validation (so epoch-level F1 is ready)."""
        metrics = trainer.callback_metrics
        val_f1 = float(metrics.get("val_f1", 0.0))
        self._last_val_f1 = val_f1

        epoch = trainer.current_epoch
        # Save checkpoint if: past start epoch, on collection interval, meets threshold
        # CRITICAL: trainer.save_checkpoint() internally calls strategy.barrier() — a
        # collective requiring ALL ranks to participate. We must call it on ALL ranks
        # (not just rank 0). Previously the entire block was guarded by
        # `if trainer.is_global_zero:`, which caused rank 0 to call the barrier while
        # rank 1 skipped it. This desynchronized ranks: rank 1 moved ahead into
        # LightningModule.on_validation_epoch_end() and called dist.all_gather(), while
        # rank 0 was stuck at the barrier. The mismatched NCCL collectives corrupted
        # all_gather output on rank 1 (garbage max_size -> storage overflow crash).
        if (
            epoch >= self.start_epoch
            and epoch % self.swa_freq == 0
            and val_f1 >= self.val_f1_threshold
        ):
            # Prepare checkpoint path (needed on all ranks for trainer.save_checkpoint)
            swa_pool_dir = self.out_dir / "swa_pool"
            ckpt_path = swa_pool_dir / f"swa_epoch={epoch:04d}_val_f1={val_f1:.4f}.ckpt"
            if trainer.is_global_zero:
                swa_pool_dir.mkdir(parents=True, exist_ok=True)
            # Call on ALL ranks: DDPStrategy.save_checkpoint only writes on rank 0,
            # but the internal barrier must be participated by all ranks.
            trainer.save_checkpoint(str(ckpt_path))
            if trainer.is_global_zero:
                self.checkpoint_pool.append((val_f1, ckpt_path))
                pl_module.print(
                    f"[SWA Pool] Saved checkpoint: epoch={epoch}, val_f1={val_f1:.4f} "
                    f"(pool size={len(self.checkpoint_pool)})"
                )


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """LightningModule for gene-perturbation DEG prediction (Node 1-2).

    Key innovations over parent (node1-1-2-1-1-1, F1=0.5029):
    - Extended partial backbone fine-tuning: mps.6+mps.7+post_mp (~198K params)
    - Single cosine schedule with CLAMPED progress (fixes unintended LR cycling bug)
    - T_max=130 epochs (calibrated to parent's observed productive window)
    - Learnable OOV embedding (initialized to mean GNN embedding)
    - Post-hoc quality-filtered SWA via SWACheckpointCallback
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
        backbone_weight_decay: float = 1e-4,
        warmup_steps: int = 100,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        class_weight_down: float = 2.0,
        class_weight_neutral: float = 0.5,
        class_weight_up: float = 4.0,
        max_steps_total: int = 1430,  # Calibrated to ~130 epochs horizon
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

        # ── Load STRING_GNN and extract partial backbone ────────────────────
        gnn_model = AutoModel.from_pretrained(
            str(STRING_GNN_DIR), trust_remote_code=True
        )
        gnn_model.eval()
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
        self.edge_index = graph["edge_index"]   # stored as attribute for live GNN forward
        self.edge_weight = graph.get("edge_weight", None)

        # Precompute frozen intermediate (output after mps.5, before mps.6)
        # This avoids recomputing layers 0-5 at every step
        with torch.no_grad():
            x = gnn_model.emb.weight.clone()  # [18870, 256]
            for i in range(6):  # mps.0 through mps.5
                mp = gnn_model.mps[i]
                x = mp(x, self.edge_index, self.edge_weight) + x  # residual
        frozen_intermediate = x.detach()  # [18870, 256]
        self.register_buffer("frozen_intermediate", frozen_intermediate)

        # Also store full GNN embeddings for reference (used when computing OOV emb)
        with torch.no_grad():
            outputs = gnn_model(
                edge_index=self.edge_index,
                edge_weight=self.edge_weight,
            )
        gnn_embs_full = outputs.last_hidden_state.detach()  # [18870, 256]

        # Learnable OOV embedding initialized to mean of full GNN embeddings
        # Placed in backbone AdamW group to prevent large gradient steps
        oov_emb_init = gnn_embs_full.mean(dim=0)  # [256]
        self.oov_emb = nn.Parameter(oov_emb_init.clone())

        # ── Extract the two fine-tuned backbone modules ─────────────────────
        # We fine-tune mps.6, mps.7, and post_mp (per STRING_GNN skill recommendation)
        self.gnn_layer6 = gnn_model.mps[6]
        self.gnn_layer7 = gnn_model.mps[7]
        self.gnn_post_mp = gnn_model.post_mp

        # Ensure these modules are in train mode for fine-tuning
        self.gnn_layer6.train()
        self.gnn_layer7.train()
        self.gnn_post_mp.train()

        # ── Register class weights as buffer ───────────────────────────────
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
        for param in self.gnn_layer6.parameters():
            param.data = param.data.float()
        for param in self.gnn_layer7.parameters():
            param.data = param.data.float()
        for param in self.gnn_post_mp.parameters():
            param.data = param.data.float()

    def _get_gnn_emb(self, gnn_idx: torch.Tensor) -> torch.Tensor:
        """Compute GNN embeddings with partial backbone fine-tuning.

        Forward pass:
            frozen_intermediate [18870, 256] (precomputed, no grad)
            → mps.6 (trainable, gradient flows) + residual
            → mps.7 (trainable, gradient flows) + residual
            → post_mp (trainable, gradient flows)
            → index by gnn_idx → [B, 256]

        Args:
            gnn_idx: [B] long tensor of STRING_GNN node indices; -1 for OOV

        Returns:
            emb: [B, 256] float32 tensor
        """
        device = gnn_idx.device

        # Move edge_index and edge_weight to correct device for live GNN forward
        edge_idx = self.edge_index.to(device)
        edge_wt = self.edge_weight.to(device) if self.edge_weight is not None else None

        # Live forward through fine-tuned backbone layers
        # frozen_intermediate is a registered buffer (auto moved to device)
        x_inter = self.frozen_intermediate  # [18870, 256], no grad

        # mps.6: trainable (gradient flows)
        x_after6 = self.gnn_layer6(x_inter, edge_idx, edge_wt) + x_inter  # residual
        # mps.7: trainable (gradient flows)
        x_after7 = self.gnn_layer7(x_after6, edge_idx, edge_wt) + x_after6  # residual
        # post_mp: trainable (gradient flows, no residual for post_mp per STRING_GNN arch)
        x_final = self.gnn_post_mp(x_after7)  # [18870, 256]

        # Index embeddings for each sample in batch
        valid_mask = gnn_idx >= 0  # [B]
        safe_idx = gnn_idx.clone()
        safe_idx[~valid_mask] = 0

        emb = x_final[safe_idx]  # [B, 256]

        # Replace OOV entries with learnable OOV embedding
        if (~valid_mask).any():
            emb = emb.clone()
            n_oov = (~valid_mask).sum()
            emb[~valid_mask] = self.oov_emb.unsqueeze(0).expand(n_oov, -1).to(emb.dtype)

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
                f"[Node1-2] Saved test predictions → {pred_path} "
                f"({len(seen_ids)} unique samples)"
            )

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-2] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        """
        Three-group optimizer configuration (Muon + AdamW head + AdamW backbone).

        Groups:
          1. Muon: hidden 2D weight matrices in ResidualBlocks (lr=0.005, validated optimal)
          2. AdamW head: embeddings, norms, biases, proj_in, proj_bilinear, OOV emb
             (lr=5e-4, weight_decay=2e-3)
          3. AdamW backbone: mps.6, mps.7, post_mp fine-tuned layers
             (lr=1e-5, weight_decay=1e-4 — conservative to preserve pretrained structure)

        LR Schedule:
          Single cosine annealing with CLAMPED progress (critical fix):
            progress = min(1.0, (step - warmup) / (max_steps - warmup))
          This prevents the unintended second LR cycle seen in parent nodes.
          T_max = 130 epochs (calibrated to observed productive window end).
        """
        hp = self.hparams

        # ── Separate parameter groups ──────────────────────────────────────
        muon_params = []
        adamw_head_params = []
        adamw_backbone_params = []

        # Head parameters (MLP head model)
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

        # OOV embedding (learnable parameter, placed in head group)
        adamw_head_params.append(self.oov_emb)

        # Backbone parameters (fine-tuned GNN layers)
        for module in [self.gnn_layer6, self.gnn_layer7, self.gnn_post_mp]:
            for param in module.parameters():
                if param.requires_grad:
                    adamw_backbone_params.append(param)

        print(f"[Optimizer] Muon params: {sum(p.numel() for p in muon_params):,} "
              f"(lr={hp.muon_lr})")
        print(f"[Optimizer] AdamW head params: {sum(p.numel() for p in adamw_head_params):,} "
              f"(lr={hp.lr}, wd={hp.weight_decay})")
        print(f"[Optimizer] AdamW backbone params: {sum(p.numel() for p in adamw_backbone_params):,} "
              f"(lr={hp.backbone_lr}, wd={hp.backbone_weight_decay})")

        param_groups = [
            # Group 1: Muon for ResidualBlock hidden weight matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            # Group 2: AdamW for head projections, embeddings, norms, biases, OOV
            dict(
                params=adamw_head_params,
                use_muon=False,
                lr=hp.lr,
                betas=(0.9, 0.999),
                eps=1e-10,
                weight_decay=hp.weight_decay,
            ),
            # Group 3: AdamW for fine-tuned backbone (very low LR)
            dict(
                params=adamw_backbone_params,
                use_muon=False,
                lr=hp.backbone_lr,
                betas=(0.9, 0.999),
                eps=1e-10,
                weight_decay=hp.backbone_weight_decay,
            ),
        ]

        if dist.is_available() and dist.is_initialized():
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        # Single cosine annealing with CLAMPED progress
        # CRITICAL FIX: clamp progress = min(1.0, progress) to prevent unintended
        # second LR cycle seen in parent (node1-1-2-1-1-1) and grandparent nodes.
        # T_max calibrated to 130 epochs = observed best-epoch productive window end.
        def lr_lambda(current_step: int):
            if current_step < hp.warmup_steps:
                return float(current_step) / max(1, hp.warmup_steps)
            progress = float(current_step - hp.warmup_steps) / max(
                1, hp.max_steps_total - hp.warmup_steps
            )
            # CRITICAL: Clamp progress to [0, 1] to prevent unintended second cycle
            progress = min(1.0, progress)
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


# ─── Post-hoc SWA ─────────────────────────────────────────────────────────────

def apply_swa_and_predict(
    checkpoint_pool: List[Tuple[float, Path]],
    lit_module: PerturbationLitModule,
    datamodule: PerturbationDataModule,
    out_dir: Path,
    top_k: int = 15,
    temperature: float = 3.0,
    n_gpus: int = 1,
) -> Optional[float]:
    """Apply post-hoc quality-filtered SWA over collected checkpoints.

    Strategy (inspired by node2-1-1-1-2-1-1-1-1-1-1, which gained +0.0065 with SWA):
    1. Sort checkpoints by val_f1 descending
    2. Take top_k checkpoints
    3. Compute exponential weights: w_i = exp(val_f1_i * temperature)
    4. Weighted average of parameters from all top_k checkpoints
    5. Run test inference with the averaged model

    Args:
        checkpoint_pool: List of (val_f1, ckpt_path) tuples
        lit_module: the trained LightningModule (reference for architecture)
        datamodule: the DataModule for test inference
        out_dir: output directory for SWA predictions
        top_k: number of top checkpoints to average
        temperature: exponential weighting temperature (higher = more weight to best)
        n_gpus: number of GPUs for SWA inference

    Returns:
        SWA test F1 if successful, None otherwise.
    """
    if len(checkpoint_pool) < 2:
        print(f"[SWA] Not enough checkpoints (need >=2, got {len(checkpoint_pool)}). Skipping SWA.")
        return None

    # Sort by val_f1 descending, take top_k
    pool_sorted = sorted(checkpoint_pool, key=lambda x: x[0], reverse=True)[:top_k]
    print(f"[SWA] Using {len(pool_sorted)} checkpoints (top_k={top_k})")
    for i, (f1, path) in enumerate(pool_sorted):
        print(f"  [{i+1}] val_f1={f1:.4f} — {path.name}")

    # Compute exponential weights
    f1_vals = np.array([f1 for f1, _ in pool_sorted])
    log_weights = f1_vals * temperature
    log_weights -= log_weights.max()  # numerical stability
    weights = np.exp(log_weights)
    weights /= weights.sum()
    print(f"[SWA] Weights: min={weights.min():.4f}, max={weights.max():.4f}, "
          f"best ckpt weight={weights[0]*100:.1f}%")

    # Compute weighted average of state dicts
    averaged_sd = {}
    for i, (_, ckpt_path) in enumerate(pool_sorted):
        w = float(weights[i])
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
        except Exception as e:
            print(f"[SWA] Failed to load {ckpt_path}: {e}. Skipping.")
            continue

        if i == 0:
            averaged_sd = {k: v.float() * w for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if k in averaged_sd:
                    averaged_sd[k] = averaged_sd[k] + v.float() * w
                else:
                    averaged_sd[k] = v.float() * w

    if not averaged_sd:
        print("[SWA] Empty averaged state dict. Skipping SWA.")
        return None

    # Create a new LightningModule instance and load averaged weights
    swa_module = PerturbationLitModule(
        hidden_dim=lit_module.hparams.hidden_dim,
        n_resblocks=lit_module.hparams.n_resblocks,
        expand=lit_module.hparams.expand,
        dropout=lit_module.hparams.dropout,
        rank=lit_module.hparams.rank,
        lr=lit_module.hparams.lr,
        muon_lr=lit_module.hparams.muon_lr,
        backbone_lr=lit_module.hparams.backbone_lr,
        weight_decay=lit_module.hparams.weight_decay,
        backbone_weight_decay=lit_module.hparams.backbone_weight_decay,
        warmup_steps=lit_module.hparams.warmup_steps,
        focal_gamma=lit_module.hparams.focal_gamma,
        label_smoothing=lit_module.hparams.label_smoothing,
        class_weight_down=lit_module.hparams.class_weight_down,
        class_weight_neutral=lit_module.hparams.class_weight_neutral,
        class_weight_up=lit_module.hparams.class_weight_up,
        max_steps_total=lit_module.hparams.max_steps_total,
        grad_clip_norm=lit_module.hparams.grad_clip_norm,
    )

    # Override test output directory for SWA predictions
    swa_out_dir = out_dir / "swa"
    swa_out_dir.mkdir(parents=True, exist_ok=True)

    # Patch on_test_epoch_end to write to SWA directory
    original_test_epoch_end = swa_module.on_test_epoch_end

    def swa_test_epoch_end(self):
        # Temporarily redirect output to SWA directory
        original_path = Path(__file__).parent / "run"
        # Use a monkey-patched approach: the on_test_epoch_end writes to
        # Path(__file__).parent / "run" — we rely on this and copy afterward
        original_test_epoch_end()

    # Run SWA inference with a minimal single-device trainer
    swa_trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,   # SWA always runs on single device to avoid gather complexity
        num_nodes=1,
        strategy="auto",
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        max_epochs=1,
    )

    # Setup the SWA module to initialize architecture
    swa_trainer.strategy.connect(swa_module)
    swa_module.setup("test")

    # Move to GPU and load averaged weights
    swa_module = swa_module.cuda()
    load_result = swa_module.load_state_dict(averaged_sd, strict=False)
    print(f"[SWA] State dict loaded. Missing keys: {len(load_result.missing_keys)}, "
          f"Unexpected: {len(load_result.unexpected_keys)}")

    swa_module.eval()

    # Run test on validation set first to assess SWA quality
    print("[SWA] Running test inference with SWA weights...")

    # We need to run forward passes manually to avoid trainer complexity
    device = torch.device("cuda:0")
    swa_module = swa_module.to(device)

    all_probs_list = []
    all_labels_list = []
    all_pert_ids = []
    all_symbols = []

    datamodule.setup("test")

    with torch.no_grad():
        # Validation set assessment
        val_loader = datamodule.val_dataloader()
        val_preds = []
        val_labels = []
        for batch in val_loader:
            gnn_idx = batch["gnn_idx"].to(device)
            logits = swa_module(gnn_idx)
            probs = torch.softmax(logits, dim=1).cpu().float()
            val_preds.append(probs)
            if "label" in batch:
                val_labels.append(batch["label"].cpu())

        if val_preds and val_labels:
            val_probs_np = torch.cat(val_preds, dim=0).numpy()
            val_labels_np = torch.cat(val_labels, dim=0).numpy()
            swa_val_f1 = compute_per_gene_f1(val_probs_np, val_labels_np)
            print(f"[SWA] Val F1 with SWA weights: {swa_val_f1:.4f} "
                  f"(best single ckpt val_f1={pool_sorted[0][0]:.4f})")

        # Test set predictions
        test_loader = datamodule.test_dataloader()
        for batch in test_loader:
            gnn_idx = batch["gnn_idx"].to(device)
            logits = swa_module(gnn_idx)
            probs = torch.softmax(logits, dim=1).cpu().float()
            all_probs_list.append(probs)
            all_pert_ids.extend(batch["pert_id"])
            all_symbols.extend(batch["symbol"])
            if "label" in batch:
                all_labels_list.append(batch["label"].cpu())

    all_probs_np = torch.cat(all_probs_list, dim=0).numpy()
    if all_labels_list:
        all_labels_np = torch.cat(all_labels_list, dim=0).numpy()
    else:
        all_labels_np = np.zeros((all_probs_np.shape[0], N_GENES_OUT), dtype=np.int64)

    # Save SWA test predictions
    swa_pred_path = swa_out_dir / "test_predictions.tsv"
    seen_ids: set = set()
    dedup_probs = []
    dedup_labels = []
    with open(swa_pred_path, "w") as fh:
        fh.write("idx\tinput\tprediction\n")
        for i, (pert_id, symbol, probs) in enumerate(
            zip(all_pert_ids, all_symbols, all_probs_np)
        ):
            if pert_id not in seen_ids:
                seen_ids.add(pert_id)
                fh.write(f"{pert_id}\t{symbol}\t{json.dumps(probs.tolist())}\n")
                dedup_probs.append(probs)
                dedup_labels.append(all_labels_np[i])
    print(f"[SWA] Saved SWA test predictions → {swa_pred_path} ({len(seen_ids)} samples)")

    swa_test_f1 = None
    if dedup_probs and dedup_labels:
        dedup_probs_np = np.stack(dedup_probs, axis=0)
        dedup_labels_np = np.stack(dedup_labels, axis=0)
        if dedup_labels_np.any():
            swa_test_f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
            print(f"[SWA] Self-computed SWA test F1 = {swa_test_f1:.4f}")

    return swa_test_f1


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 1-2 – STRING_GNN (mps.6+7+post_mp FT) + Deep Bilinear MLP Head + SWA"
    )
    p.add_argument("--data-dir",              type=str,   default="data")
    p.add_argument("--hidden-dim",            type=int,   default=512)
    p.add_argument("--n-resblocks",           type=int,   default=6)
    p.add_argument("--expand",                type=int,   default=4)
    p.add_argument("--dropout",               type=float, default=0.3)
    p.add_argument("--rank",                  type=int,   default=512)
    p.add_argument("--lr",                    type=float, default=5e-4)
    p.add_argument("--muon-lr",               type=float, default=0.005)
    # Backbone LR: very conservative for partial fine-tuning (10x lower than pretrain LR)
    p.add_argument("--backbone-lr",           type=float, default=1e-5)
    p.add_argument("--weight-decay",          type=float, default=2e-3)
    p.add_argument("--backbone-weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps",          type=int,   default=100)
    p.add_argument("--focal-gamma",           type=float, default=2.0)
    p.add_argument("--label-smoothing",       type=float, default=0.05)
    p.add_argument("--class-weight-down",     type=float, default=2.0)
    p.add_argument("--class-weight-neutral",  type=float, default=0.5)
    p.add_argument("--class-weight-up",       type=float, default=4.0)
    p.add_argument("--grad-clip-norm",        type=float, default=1.0)
    p.add_argument("--micro-batch-size",      type=int,   default=16)
    p.add_argument("--global-batch-size",     type=int,   default=128)
    # Target 130 epochs for cosine schedule calibration
    p.add_argument("--target-epochs-for-schedule", type=int, default=130)
    p.add_argument("--max-epochs",            type=int,   default=300)
    p.add_argument("--patience",              type=int,   default=100)
    p.add_argument("--num-workers",           type=int,   default=4)
    p.add_argument("--val-check-interval",    type=float, default=1.0)
    # SWA hyperparameters
    p.add_argument("--swa-freq",              type=int,   default=5)
    p.add_argument("--swa-start-epoch",       type=int,   default=20)
    p.add_argument("--swa-threshold",         type=float, default=0.490)
    p.add_argument("--swa-top-k",             type=int,   default=15)
    p.add_argument("--swa-temperature",       type=float, default=3.0)
    p.add_argument("--no-swa",                action="store_true", default=False)
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
    # Calibrate to target_epochs_for_schedule (130 epochs by default)
    # This calibration ensures the cosine LR reaches minimum at the end of the expected
    # productive training window, providing a strong signal throughout training.
    _train_df_size = pd.read_csv(
        Path(args.data_dir) / "train.tsv", sep="\t", usecols=["pert_id"]
    ).shape[0]
    steps_per_epoch = _train_df_size // (args.micro_batch_size * n_gpus)
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)

    target_epochs = min(args.target_epochs_for_schedule, args.max_epochs)
    max_steps_total = effective_steps_per_epoch * target_epochs

    print(f"[Main] effective_steps_per_epoch={effective_steps_per_epoch}, "
          f"max_steps_total={max_steps_total} (calibrated to {target_epochs} epochs)")

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

    # SWA checkpoint collection callback
    # Collect every 5 epochs starting from epoch 20, threshold=0.490
    swa_cb = SWACheckpointCallback(
        out_dir=out_dir,
        swa_freq=args.swa_freq,
        start_epoch=args.swa_start_epoch,
        val_f1_threshold=args.swa_threshold,
    )

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

    callbacks_list = [ckpt_cb, es_cb, lr_cb, pb_cb]
    # Only add SWA callback when not in debug mode
    if not fast_dev_run and args.debug_max_step is None and not args.no_swa:
        callbacks_list.append(swa_cb)

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
        callbacks=callbacks_list,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    # ── Post-hoc SWA on rank 0 only ────────────────────────────────────────
    swa_f1 = None
    if (
        trainer.is_global_zero
        and not fast_dev_run
        and args.debug_max_step is None
        and not args.no_swa
        and len(swa_cb.checkpoint_pool) >= 2
    ):
        print(f"\n[Main] Starting post-hoc SWA with {len(swa_cb.checkpoint_pool)} checkpoints "
              f"in pool...")
        try:
            swa_f1 = apply_swa_and_predict(
                checkpoint_pool=swa_cb.checkpoint_pool,
                lit_module=lit,
                datamodule=dm,
                out_dir=out_dir,
                top_k=args.swa_top_k,
                temperature=args.swa_temperature,
                n_gpus=n_gpus,
            )
        except Exception as e:
            print(f"[SWA] SWA failed with exception: {e}. Falling back to single-checkpoint test.")
            import traceback
            traceback.print_exc()
    elif trainer.is_global_zero and not fast_dev_run and args.debug_max_step is None and not args.no_swa:
        print(f"[Main] SWA pool too small ({len(swa_cb.checkpoint_pool)} checkpoints). "
              f"Need >= 2. Skipping SWA.")

    if trainer.is_global_zero:
        # Copy the best test_predictions.tsv (SWA if available and better, else single best)
        best_pred_path = out_dir / "test_predictions.tsv"
        swa_pred_path = out_dir / "swa" / "test_predictions.tsv"

        # If SWA predictions exist and appear reasonable, use them as the primary output
        # (SWA is designed to be better; if it ran successfully, it should be preferred)
        if swa_pred_path.exists() and swa_f1 is not None:
            import shutil
            shutil.copy2(str(swa_pred_path), str(best_pred_path))
            print(f"[Main] Using SWA predictions as primary output (SWA test F1={swa_f1:.4f})")
        else:
            print(f"[Main] Using single best checkpoint predictions as primary output")

        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"Node 1-2 – STRING_GNN (mps.6+7+post_mp FT) + Deep Bilinear MLP Head + SWA\n"
            f"Test results from trainer: {test_results}\n"
            f"SWA test F1: {swa_f1}\n"
            f"SWA pool size: {len(swa_cb.checkpoint_pool) if not fast_dev_run and args.debug_max_step is None else 0}\n"
            f"Hyperparameters: dropout={args.dropout}, rank={args.rank}, "
            f"weight_decay={args.weight_decay}, backbone_lr={args.backbone_lr}, "
            f"label_smoothing={args.label_smoothing}, "
            f"muon_lr={args.muon_lr}, lr={args.lr}, "
            f"class_weights=[{args.class_weight_down}, {args.class_weight_neutral}, {args.class_weight_up}], "
            f"max_steps_total={lit.hparams.max_steps_total}, "
            f"target_epochs={target_epochs}\n"
        )
        print(f"[Main] Test score saved → {score_path}")


if __name__ == "__main__":
    main()
