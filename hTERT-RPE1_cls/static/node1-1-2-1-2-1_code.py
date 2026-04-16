"""
Node 1-2 – Frozen STRING_GNN + Bilinear MLP Head (rank=512) + Muon + SGDR Warm Restarts

Architecture:
  - Fully frozen STRING_GNN backbone:
      * All layers frozen; embeddings pre-computed once at setup as buffer [18870, 256]
      * No gradient flow through backbone (eliminates early-phase instability from parent)
  - Deep 6-layer residual bilinear MLP head (hidden_dim=512, expand=4, rank=512)
  - Bilinear interaction: pert_emb [B, 256] → head [B, 512] → [B, 3*512] → [B, 3, 512]
                          × out_gene_emb [6640, 512] → logits [B, 3, 6640]
  - Class-weighted focal cross-entropy loss (gamma=2.0, weights=[down=2.0, neutral=0.5, up=4.0])
  - Two-group optimizer:
      * Muon (lr=0.005): hidden 2D weight matrices in ResidualBlocks (~12.6M params)
      * AdamW (lr=5e-4): head other params (proj_in, proj_bilinear, out_gene_emb, norms, biases)
  - SGDR cosine warm restarts (T_0=300 steps, T_mult=1, eta_min=1e-6) with 100-step linear warmup
  - Gradient clipping (max_norm=1.0)
  - Strong regularization: dropout=0.3, weight_decay=2e-3

Key improvements over Parent (Node 1-1-2-1-2, F1=0.4986):
  1. Fully frozen backbone — eliminates early-phase embedding instability
     (parent's val_f1 at epoch 10 was 0.4678 vs sibling's 0.4794 due to backbone friction)
  2. SGDR warm restarts (T_0=300 steps, T_mult=1, eta_min=1e-6)
     — replaces parent's single clamped cosine (80-epoch horizon) that decayed LR to 56% of peak
       by epoch 42 and then froze at 0% for 23 epochs
     — eta_min=1e-6 prevents frozen-model wasted epochs
     — T_0=300 (~27 epochs/cycle) matches the T_0=600 cycle length in node1-2-2-2-1 (F1=0.5099)
  3. Extended training budget (max_epochs=250, patience=100) to accommodate 5-6 SGDR cycles
  4. Simplified two-group optimizer (no backbone AdamW group)

Key insights from collected_memory:
  - node1-1-2-1-2 (parent, F1=0.4986): 80-epoch cosine horizon → premature LR decay; backbone FT → instability
  - node1-1-2-1-1 (Muon lr=0.005 + class weights + frozen, F1=0.5023): frozen backbone + Muon proven
  - node1-1-2-1-1-1 (rank=512 + dropout=0.3 + wd=2e-3 + frozen, F1=0.5029): stronger regularization works
  - node1-1-2-1-1-1-1 (partial backbone FT, F1=0.5035): best in parent lineage
  - node1-2-2-2-1 (T_0=600 steps, T_mult=1, partial FT, F1=0.5099): SGDR ascending staircase proven
  - node1-2-2-3 (T_0=1200, F1=0.5101): LR restart at cycle 1→2 boundary was decisive
  - SGDR warm restarts consistently outperform single cosine when architecture is stable (frozen backbone)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
import math
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
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Focal cross-entropy loss with per-class weights.

    Combines focal loss modulation with class-level gradient weighting to address
    the severe class imbalance (88.9% neutral / 8.1% down / 3.0% up).

    Args:
        logits:        [B, C, G] float32 – per-class logits
        targets:       [B, G]    long    – class indices 0..C-1
        gamma:         focusing parameter (0 = standard CE)
        class_weights: [C]       float32 – per-class weight

    Returns:
        Scalar loss.
    """
    # Standard weighted CE (reduction='none') with class weights
    ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        reduction="none",
    )  # [B, G]
    # Focal modulation: use plain prob for pt
    with torch.no_grad():
        log_probs = F.log_softmax(logits, dim=1)  # [B, C, G]
        probs = log_probs.exp()                    # [B, C, G]
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
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.pert_id_to_gnn_idx = {name: i for i, name in enumerate(node_names)}

        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        self.train_ds = PerturbationDataset(dfs["train"], self.pert_id_to_gnn_idx, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   self.pert_id_to_gnn_idx, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  self.pert_id_to_gnn_idx, True)

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
    STRING_GNN-based perturbation predictor with rank=512 bilinear head.

    Architecture:
        gnn_emb [B, 256] → proj_in [B, hidden_dim=512] → 6×ResidualBlock → [B, 512]
        → norm_out → proj_bilinear [B, 3 * rank=512] → reshape [B, 3, 512]
        × out_gene_emb [6640, 512] → logits [B, 3, 6640]

    Key improvements over parent:
    - rank=512 (same as parent): larger bilinear interaction space (more expressive)
    - dropout=0.3 (same as parent): stronger regularization to delay overfitting
    - Frozen backbone (new): eliminates early-phase instability from partial FT
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

        # Deep residual MLP: 6 blocks (proven optimal from tree best nodes)
        self.resblocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
              for _ in range(n_resblocks)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear interaction head with rank=512
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
    """
    LightningModule for gene-perturbation DEG prediction.

    Node 1-2: Frozen STRING_GNN + rank=512 Bilinear MLP Head + Muon + SGDR Warm Restarts

    Primary distinction from parent (node1-1-2-1-2, F1=0.4986):
    - Fully frozen backbone (no partial FT) → eliminates early-phase instability
    - SGDR warm restarts with T_0=300 steps, eta_min=1e-6 → multi-cycle staircase improvement
    - Extended training (max_epochs=250, patience=100) → full 5-6 SGDR cycles
    - Two-group optimizer (no backbone group)
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
        weight_decay: float = 2e-3,
        warmup_steps: int = 100,
        focal_gamma: float = 2.0,
        class_weight_down: float = 2.0,
        class_weight_neutral: float = 0.5,
        class_weight_up: float = 4.0,
        sgdr_t0: int = 300,
        sgdr_t_mult: int = 1,
        sgdr_eta_min_lr: float = 1e-6,
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

        # ── Load STRING_GNN backbone and pre-compute frozen embeddings ─────────
        gnn_model = AutoModel.from_pretrained(
            str(STRING_GNN_DIR), trust_remote_code=True
        )
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)

        # Run the full GNN forward pass once to get final embeddings
        # (last_hidden_state = post_mp output, which is what we want)
        gnn_model.eval()
        with torch.no_grad():
            outputs = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
                output_hidden_states=False,  # Only need the final output
            )
            # last_hidden_state is post_mp(mps[-1](x)), the final 256-dim embedding
            pre_embs = outputs.last_hidden_state.detach()  # [18870, 256]

        # Register frozen embeddings as buffer for fast lookup
        self.register_buffer("pre_embs", pre_embs)  # [18870, 256]

        # Static OOV fallback: mean of all node embeddings
        oov_emb = pre_embs.mean(dim=0, keepdim=True)  # [1, 256]
        self.register_buffer("oov_emb", oov_emb)

        # Free gnn_model memory (backbone fully frozen, no trainable params)
        del gnn_model

        # ── Build prediction head ─────────────────────────────────────────────
        self.model = GNNBilinearHead(
            gnn_dim=STRING_GNN_DIM,
            hidden_dim=hp.hidden_dim,
            n_resblocks=hp.n_resblocks,
            expand=hp.expand,
            dropout=hp.dropout,
            rank=hp.rank,
        )

        # Register class weights as buffer for loss computation
        cw = torch.tensor(
            [hp.class_weight_down, hp.class_weight_neutral, hp.class_weight_up],
            dtype=torch.float32,
        )
        self.register_buffer("class_weights", cw)

        # Cast trainable parameters to float32 for stable optimization
        for param in self.model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    def _get_gnn_emb(self, gnn_idx: torch.Tensor) -> torch.Tensor:
        """
        Retrieve STRING_GNN embeddings for a batch of gene indices.

        Uses the fully pre-computed frozen embedding buffer.
        OOV genes (index -1) are replaced with the static mean embedding.

        Args:
            gnn_idx: [B] long tensor of STRING_GNN node indices; -1 for OOV

        Returns:
            emb: [B, 256] float32 tensor
        """
        valid_mask = gnn_idx >= 0  # [B]

        safe_idx = gnn_idx.clone()
        safe_idx[~valid_mask] = 0

        emb = self.pre_embs[safe_idx]  # [B, 256]

        # Replace OOV entries with static mean fallback
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
            gamma=self.hparams.focal_gamma,
            class_weights=self.class_weights,
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
        Two-group optimizer configuration with SGDR warm restarts.

        Group 1 — Muon: hidden 2D weight matrices in ResidualBlocks
            Muon lr=0.005 (proven calibration from multiple nodes)
            These are the ~12.6M parameters in the 6-layer deep MLP (expand=4, rank=512)
            Linear(512→2048) and Linear(2048→512) in each block.

        Group 2 — AdamW (head): proj_in, proj_bilinear, out_gene_emb, norms, biases
            lr=5e-4, weight_decay=2e-3 (proven from node1-2 and multiple successors)

        LR schedule: Linear warmup for warmup_steps steps, then SGDR cosine warm restarts.
        SGDR parameters: T_0=sgdr_t0 steps, T_mult=sgdr_t_mult, eta_min=sgdr_eta_min_lr.
        Using eta_min > 0 prevents the "frozen model" problem (LR=0 for 23 epochs in parent).

        Implementation: A single LambdaLR that encodes both warmup and SGDR behavior.
        The SGDR cycle position is computed mathematically from the current step.
        """
        hp = self.hparams

        muon_params = []
        adamw_head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Muon: hidden 2D weight matrices in ResidualBlocks (not norms, not embeddings)
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

        print(f"[Optimizer] Muon params:       {sum(p.numel() for p in muon_params):,}")
        print(f"[Optimizer] AdamW head params: {sum(p.numel() for p in adamw_head_params):,}")

        param_groups = [
            # Group 1: Muon for hidden MLP weight matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            # Group 2: AdamW for head input/output/embedding/norm params
            dict(
                params=adamw_head_params,
                use_muon=False,
                lr=hp.lr,
                betas=(0.9, 0.999),
                eps=1e-10,
                weight_decay=hp.weight_decay,
            ),
        ]

        # Use MuonWithAuxAdam when distributed is initialized,
        # SingleDeviceMuonWithAuxAdam for single GPU or fast_dev_run
        if dist.is_available() and dist.is_initialized():
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        # LR schedule: linear warmup + SGDR cosine warm restarts
        # SGDR: CosineAnnealingWarmRestarts behavior implemented via LambdaLR
        # - T_0 = sgdr_t0 steps per cycle
        # - T_mult = sgdr_t_mult (cycle multiplier; 1 = equal cycles)
        # - eta_min = sgdr_eta_min_lr (prevents LR=0 freeze)
        # Combined with linear warmup over warmup_steps steps.
        #
        # Formula for SGDR fraction in [0, 1]:
        #   Find cycle i and step within cycle, compute cosine decay:
        #   lr_scale = eta_min + 0.5*(1 - eta_min)*(1 + cos(pi * progress_in_cycle))
        # where progress_in_cycle = steps_in_cycle / T_i

        def compute_sgdr_scale(step: int) -> float:
            """Compute LR scale factor for SGDR at given global step."""
            t0 = hp.sgdr_t0
            t_mult = hp.sgdr_t_mult
            eta_min_frac = hp.sgdr_eta_min_lr / hp.muon_lr  # As fraction of peak LR

            # Find which cycle and position within that cycle
            if t_mult == 1:
                # Equal cycles of length t0
                cycle = step // t0
                t_cur = step % t0
                t_i = t0
            else:
                # Geometrically growing cycles
                t_i = t0
                t_cur = step
                cycle = 0
                while t_cur >= t_i:
                    t_cur -= t_i
                    t_i = int(t_i * t_mult)
                    cycle += 1

            # Cosine annealing within cycle
            progress = t_cur / max(1, t_i)
            cosine_scale = eta_min_frac + 0.5 * (1.0 - eta_min_frac) * (
                1.0 + math.cos(math.pi * progress)
            )
            return cosine_scale

        def lr_lambda(current_step: int) -> float:
            if current_step < hp.warmup_steps:
                # Linear warmup
                return float(current_step) / max(1, hp.warmup_steps)
            # SGDR cosine warm restarts after warmup
            post_warmup_step = current_step - hp.warmup_steps
            return compute_sgdr_scale(post_warmup_step)

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
        description="Node 1-2 – Frozen STRING_GNN + Bilinear MLP Head (rank=512) + Muon + SGDR"
    )
    p.add_argument("--data-dir",                    type=str,   default="data")
    p.add_argument("--hidden-dim",                  type=int,   default=512)
    p.add_argument("--n-resblocks",                 type=int,   default=6)
    p.add_argument("--expand",                      type=int,   default=4)
    p.add_argument("--dropout",                     type=float, default=0.3)
    p.add_argument("--rank",                        type=int,   default=512)
    p.add_argument("--lr",                          type=float, default=5e-4)
    p.add_argument("--muon-lr",                     type=float, default=0.005)
    p.add_argument("--weight-decay",                type=float, default=2e-3)
    p.add_argument("--warmup-steps",                type=int,   default=100)
    p.add_argument("--focal-gamma",                 type=float, default=2.0)
    p.add_argument("--class-weight-down",           type=float, default=2.0)
    p.add_argument("--class-weight-neutral",        type=float, default=0.5)
    p.add_argument("--class-weight-up",             type=float, default=4.0)
    p.add_argument("--sgdr-t0",                     type=int,   default=300)
    p.add_argument("--sgdr-t-mult",                 type=int,   default=1)
    p.add_argument("--sgdr-eta-min-lr",             type=float, default=1e-6)
    p.add_argument("--grad-clip-norm",              type=float, default=1.0)
    p.add_argument("--micro-batch-size",            type=int,   default=16)
    p.add_argument("--global-batch-size",           type=int,   default=128)
    p.add_argument("--max-epochs",                  type=int,   default=250)
    p.add_argument("--patience",                    type=int,   default=100)
    p.add_argument("--num-workers",                 type=int,   default=4)
    p.add_argument("--val-check-interval",          type=float, default=1.0)
    p.add_argument("--debug-max-step",              type=int,   default=None)
    p.add_argument("--fast-dev-run",                action="store_true", default=False)
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

    # Compute gradient accumulation
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    print(f"[Main] n_gpus={n_gpus}, micro_batch_size={args.micro_batch_size}, "
          f"global_batch_size={args.global_batch_size}, accum={accum}")
    print(f"[Main] SGDR: T_0={args.sgdr_t0} steps, T_mult={args.sgdr_t_mult}, "
          f"eta_min={args.sgdr_eta_min_lr}, warmup={args.warmup_steps} steps")

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
        class_weight_down=args.class_weight_down,
        class_weight_neutral=args.class_weight_neutral,
        class_weight_up=args.class_weight_up,
        sgdr_t0=args.sgdr_t0,
        sgdr_t_mult=args.sgdr_t_mult,
        sgdr_eta_min_lr=args.sgdr_eta_min_lr,
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
            f"Node 1-2 – Frozen STRING_GNN + Bilinear MLP (rank=512) + Muon + SGDR Warm Restarts\n"
            f"Test results from trainer: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
