"""
Node 2-1-1-2-1 — STRING_GNN Partial Fine-Tuning + Rank-768 Bilinear Head
                  with Label Smoothing, Lighter Regularization, and Deeper Head

Architecture:
  - STRING_GNN partial fine-tuning (last 2 GCN layers: mps.6, mps.7 + post_mp)
    ~198K trainable backbone params out of 5.43M total
  - Precomputed frozen embeddings as buffer (efficient: GNN forward pass once, then lookup)
  - 8-layer deep residual bilinear MLP head (rank=768):
    LayerNorm(256) → Linear(256→512) → 8x ResidualBlock (expand=4) → Linear(512→2304)
    → reshape [B, 3, 768] → einsum × out_gene_emb [6640, 768] → [B, 3, 6640]
  - Class-weighted focal loss with label smoothing (gamma=2.0, weights=[2.0, 0.5, 5.0])
  - Cosine LR with total_steps=1200 (calibrated to ~100 epoch actual training)
  - Patience=60 for full improvement phase

Key changes from parent (node2-1-1-2, F1=0.5000):
  - bilinear_rank: 512 → 768 (+50% head expressivity)
  - n_residual_blocks: 6 → 8 (deeper head for better feature extraction)
  - label_smoothing: 0.0 → 0.05 (reduces probability overconfidence, addresses calibration overfitting)
  - dropout: 0.25 → 0.20 (match node2-1-3's proven lighter reg that achieved 0.5047)
  - weight_decay: 1.5e-3 → 1e-3 (match node2-1-3's proven config)
  - class_weight_up: 4.0 → 5.0 (slightly more emphasis on rarest class, per feedback)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional

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

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_HIDDEN     = 256   # STRING_GNN hidden size
BILINEAR_RANK  = 768   # Increased from 512 for more expressivity


# ─── Focal Loss with Class Weights and Label Smoothing ────────────────────────

class ClassWeightedFocalLossWithSmoothing(nn.Module):
    """
    Focal loss with explicit class weights and label smoothing.

    Key change from parent node:
    - Added label_smoothing: prevents overconfident predictions, directly addresses
      the calibration overfitting observed in parent (val/train loss 3.28x).
    - Class weight for up-regulated increased to 5.0 (from 4.0): rarest class (3.0%)
      gets more emphasis per feedback analysis.

    Configuration:
      - gamma=2.0 for focal weighting (unchanged from parent)
      - class weights=[2.0, 0.5, 5.0] for (down, neutral, up)
      - label_smoothing=0.05 for calibration improvement
    """
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.n_classes = n_classes
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, C] (2D, already reshaped)
        targets: [N] long
        """
        # Standard cross-entropy with class weights and label smoothing
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        # Get probability of the true class for focal weighting
        # Note: use unsmoothed CE to compute pt for focal weighting
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction='none'))
        # Focal weight: down-weight easy examples
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Per-gene macro F1 matching calc_metric.py.  pred_np: [N,3,G], labels_np: [N,G]."""
    pred_cls = pred_np.argmax(axis=1)
    f1_vals = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    """
    Stores pert_ids, symbols, labels, and precomputed STRING_GNN node indices.
    """

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        gnn_node_indices: List[int],   # index in STRING_GNN node_names or -1 for OOV
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long or None
    ):
        self.pert_ids         = pert_ids
        self.symbols          = symbols
        self.gnn_node_indices = gnn_node_indices
        self.labels           = labels

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":      self.pert_ids[idx],
            "symbol":       self.symbols[idx],
            "gnn_node_idx": self.gnn_node_indices[idx],  # int
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    """Simple collate: stack gnn_node_idx and labels."""
    pert_ids      = [b["pert_id"]      for b in batch]
    symbols       = [b["symbol"]       for b in batch]
    gnn_node_idxs = [b["gnn_node_idx"] for b in batch]

    out = {
        "pert_id":      pert_ids,
        "symbol":       symbols,
        "gnn_node_idx": torch.tensor(gnn_node_idxs, dtype=torch.long),  # [B]
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])  # [B, 6640]
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class GNNDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage: Optional[str] = None):
        # ── Build STRING_GNN node name → index lookup ──────────────────────
        node_names = json.loads((Path(STRING_GNN_DIR) / "node_names.json").read_text())
        # node_names[i] = Ensembl gene ID (e.g., "ENSG00000000003")
        self.gnn_node_name_to_idx = {name: i for i, name in enumerate(node_names)}

        # ── Helper: load a split ───────────────────────────────────────────
        def load_split(fname: str, has_label: bool):
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            pert_ids = df["pert_id"].tolist()
            symbols  = df["symbol"].tolist()

            # Map pert_ids to STRING_GNN node indices (pert_id is Ensembl gene ID)
            gnn_node_indices = [
                self.gnn_node_name_to_idx.get(pid, -1)  # -1 = OOV
                for pid in pert_ids
            ]

            labels = None
            if has_label and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)

            return PerturbDataset(pert_ids, symbols, gnn_node_indices, labels)

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  False)

    def _loader(self, ds, shuffle, drop_last=False):
        return DataLoader(
            ds, batch_size=self.micro_batch_size,
            shuffle=shuffle, collate_fn=collate_fn,
            num_workers=self.num_workers, pin_memory=True,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True,  drop_last=True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False, drop_last=False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False, drop_last=False)


# ─── Residual Block ───────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Pre-norm residual block: LayerNorm → Linear(d→d*expand) → GELU → Dropout → Linear(d*expand→d) → Dropout + skip.
    Proven effective in node1-2 (F1=0.4912), node2-1-3 (F1=0.5047).
    """
    def __init__(self, d: int, expand: int = 4, dropout: float = 0.20):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net  = nn.Sequential(
            nn.Linear(d, d * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * expand, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


# ─── Bilinear Head ────────────────────────────────────────────────────────────

class GNNBilinearHead(nn.Module):
    """
    8-layer deep residual bilinear MLP head (rank=768).

    Key changes from parent:
    1. bilinear_rank: 512 → 768 (+50% interaction expressivity)
    2. n_residual_blocks: 6 → 8 (deeper representation learning)
    3. dropout: 0.25 → 0.20 (lighter regularization, matching node2-1-3's proven config)

    Architecture:
      LayerNorm(256) → Linear(256→512) → [B, 512]
      8x ResidualBlock(512, expand=4, dropout=0.20)
      LayerNorm(512) + Dropout(0.20)
      Linear(512 → 3×768) → reshape [B, 3, 768]
      einsum("bcr,gr→bcg", [B,3,768], [6640,768]) → [B, 3, 6640]
    """
    def __init__(
        self,
        input_dim: int = GNN_HIDDEN,
        hidden_dim: int = 512,
        n_classes: int = N_CLASSES,
        n_genes_out: int = N_GENES_OUT,
        bilinear_rank: int = BILINEAR_RANK,
        dropout: float = 0.20,
        n_residual_blocks: int = 8,
    ):
        super().__init__()
        self.bilinear_rank = bilinear_rank
        self.n_classes     = n_classes

        # Input normalization + projection to hidden dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Deep residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=4, dropout=dropout)
            for _ in range(n_residual_blocks)
        ])

        # Output normalization + bilinear projection
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, n_classes * bilinear_rank)

        # Learnable output gene embeddings [n_genes_out, bilinear_rank]
        # Initialized with small noise (std=0.02) - clean initialization
        self.out_gene_emb = nn.Parameter(
            torch.randn(n_genes_out, bilinear_rank) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim] (GNN node embeddings for perturbed genes)
        Returns: logits [B, 3, 6640]
        """
        B = x.shape[0]

        # Input projection
        h = self.input_proj(self.input_norm(x))  # [B, 512]

        # Deep residual processing
        for block in self.residual_blocks:
            h = block(h)

        # Output: project to bilinear representation
        h = self.out_drop(self.out_norm(h))          # [B, 512]
        h = self.out_proj(h)                          # [B, 3*768]
        h = h.view(B, self.n_classes, self.bilinear_rank)  # [B, 3, 768]

        # Bilinear interaction with output gene embeddings
        # [B, 3, 768] × [6640, 768]^T → [B, 3, 6640]
        logits = torch.einsum("bcr,gr->bcg", h, self.out_gene_emb)
        return logits


# ─── Main Model ───────────────────────────────────────────────────────────────

class GNNPerturbModel(nn.Module):
    """
    STRING_GNN partial fine-tuning + deep bilinear head.

    Unchanged from parent:
    - Partial fine-tuning of last 2 GCN layers + post_mp (proven best strategy)
    - Precomputed frozen embeddings buffer for efficiency
    - OOV embedding for genes not in STRING_GNN

    Changes from parent:
    - Head: rank=768, 8 residual blocks, dropout=0.20
    """

    def __init__(
        self,
        bilinear_rank: int = BILINEAR_RANK,
        head_dropout: float = 0.20,
        n_residual_blocks: int = 8,
    ):
        super().__init__()

        # ── STRING_GNN backbone: partial fine-tuning ──────────────────────
        self.gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        # Freeze emb + first 6 layers (mps.0 through mps.5)
        for name, param in self.gnn.named_parameters():
            if name.startswith("emb") or any(f"mps.{i}." in name for i in range(6)):
                param.requires_grad = False
            else:
                # trainable: mps.6.*, mps.7.*, post_mp.*
                param.requires_grad = True
                param.data = param.data.float()  # float32 for stable training

        # ── Load graph data ────────────────────────────────────────────────
        graph_data  = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", weights_only=False)
        self.register_buffer("edge_index",  graph_data["edge_index"].long())   # [2, E]
        edge_weight = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight.float())
        else:
            self.register_buffer("edge_weight", None)

        # ── Precomputed frozen buffer for intermediate GNN states ──────────
        # Will be populated in setup() via _precompute_frozen_embeddings()
        # Shape: [18870, 256] — intermediate node states after mps.0-5
        self.register_buffer("frozen_node_states", None)

        # ── OOV embedding for genes not in STRING_GNN ─────────────────────
        self.gnn_oov_emb = nn.Parameter(torch.randn(GNN_HIDDEN) * 0.02)

        # ── Deep bilinear MLP head ─────────────────────────────────────────
        self.head = GNNBilinearHead(
            input_dim=GNN_HIDDEN,
            hidden_dim=512,
            n_classes=N_CLASSES,
            n_genes_out=N_GENES_OUT,
            bilinear_rank=bilinear_rank,
            dropout=head_dropout,
            n_residual_blocks=n_residual_blocks,
        )

    def precompute_frozen_embeddings(self, device: torch.device):
        """
        Precompute node states after the frozen portion of the GNN (emb + mps.0-5).
        This is called once during setup and stored as a buffer.
        Avoids running the full frozen portion on every training step.
        """
        with torch.no_grad():
            gnn_cpu = self.gnn.cpu()
            edge_index_cpu  = self.edge_index.cpu()
            edge_weight_cpu = self.edge_weight.cpu() if self.edge_weight is not None else None

            # Get intermediate hidden states using output_hidden_states=True
            # hidden_states: (emb_init, after_mps.0, ..., after_mps.7) → 9 tensors total
            out = gnn_cpu(
                edge_index=edge_index_cpu,
                edge_weight=edge_weight_cpu,
                output_hidden_states=True,
            )
            # after mps.0-5 = hidden_states[6] (index 0=emb, 1=after_mps.0, ..., 6=after_mps.5)
            # We want the state entering mps.6, which is after mps.5 = hidden_states[6]
            frozen_states = out.hidden_states[6].float()  # [18870, 256]

            self.frozen_node_states = frozen_states.to(device)
            print(f"[Node2-1-1-2-1] Precomputed frozen node states: {frozen_states.shape}")

    def forward_trainable_gnn_layers(self, device: torch.device) -> torch.Tensor:
        """
        Run only the trainable GNN layers (mps.6, mps.7, post_mp) using the
        precomputed frozen states as input.

        Returns: all_node_embs [18870, 256]
        """
        edge_index  = self.edge_index.to(device)
        edge_weight = self.edge_weight.to(device) if self.edge_weight is not None else None

        # Start from precomputed frozen states (state after mps.5 + residual)
        x = self.frozen_node_states  # [18870, 256]

        # Run mps.6 with residual connection (matching GNN forward implementation)
        x = self.gnn.mps[6](x, edge_index, edge_weight) + x   # [18870, 256]
        # Run mps.7 with residual connection
        x = self.gnn.mps[7](x, edge_index, edge_weight) + x   # [18870, 256]
        # Apply post_mp projection (no residual — matches GNN forward)
        x = self.gnn.post_mp(x)                                # [18870, 256]

        return x.float()

    def forward(
        self,
        gnn_node_idxs: torch.Tensor,   # [B] long — STRING_GNN node index (-1 = OOV)
    ) -> torch.Tensor:
        """Returns logits [B, 3, 6640]."""
        B      = gnn_node_idxs.shape[0]
        device = gnn_node_idxs.device

        # ── A. Run trainable GNN layers ────────────────────────────────────
        all_node_embs = self.forward_trainable_gnn_layers(device)  # [18870, 256]

        # ── B. Extract per-sample embeddings ──────────────────────────────
        in_vocab_mask = (gnn_node_idxs >= 0)  # [B] bool
        node_embs = torch.zeros(B, GNN_HIDDEN, dtype=torch.float32, device=device)

        if in_vocab_mask.any():
            valid_idxs = gnn_node_idxs[in_vocab_mask]       # [n_valid]
            node_embs[in_vocab_mask] = all_node_embs[valid_idxs]

        # OOV genes get the learnable OOV embedding
        oov_mask = ~in_vocab_mask
        if oov_mask.any():
            node_embs[oov_mask] = self.gnn_oov_emb.unsqueeze(0).expand(
                oov_mask.sum(), -1
            ).to(dtype=torch.float32)

        # ── C. Deep bilinear head → logits ────────────────────────────────
        logits = self.head(node_embs)  # [B, 3, 6640]
        return logits


# ─── DDP Tensor Gathering ─────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))

    pad = max_sz - local_p.shape[0]
    p = local_p.to(device)
    l = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)

    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p)
    dist.all_gather(gl, l)

    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class GNNLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_backbone: float = 5e-5,    # for partial GNN fine-tuning (proven optimal)
        lr_head: float = 5e-4,         # for fresh bilinear MLP head
        weight_decay: float = 1e-3,    # lighter regularization matching node2-1-3's proven config
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,  # [down, neutral, up] weights
        head_dropout: float = 0.20,    # lighter dropout matching node2-1-3's proven config
        bilinear_rank: int = BILINEAR_RANK,
        n_residual_blocks: int = 8,    # increased from 6 for deeper representation
        label_smoothing: float = 0.05, # new: reduces calibration overconfidence
        warmup_steps: int = 100,
        max_steps_total: int = 1200,   # calibrated to actual training duration (~100 epochs)
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []
        self._test_labels:   List[torch.Tensor] = []

    def setup(self, stage: Optional[str] = None):
        # Initialize model
        self.model = GNNPerturbModel(
            bilinear_rank=self.hparams.bilinear_rank,
            head_dropout=self.hparams.head_dropout,
            n_residual_blocks=self.hparams.n_residual_blocks,
        )

        # Precompute frozen GNN embeddings (once, during setup)
        self.model.precompute_frozen_embeddings(self.device)

        # Initialize focal loss with class weights and label smoothing
        if self.hparams.class_weights is not None:
            cw = torch.tensor(self.hparams.class_weights, dtype=torch.float32)
        else:
            # Default weights: slightly more emphasis on up-regulated (rarest class)
            cw = torch.tensor([2.0, 0.5, 5.0], dtype=torch.float32)

        self.loss_fn = ClassWeightedFocalLossWithSmoothing(
            gamma=self.hparams.focal_gamma,
            class_weights=cw,
            label_smoothing=self.hparams.label_smoothing,
            n_classes=N_CLASSES,
        )

    def forward(self, gnn_node_idxs):
        return self.model(gnn_node_idxs)

    def _loss(self, logits, labels):
        # logits: [B, 3, 6640] -> [B*6640, 3];  labels: [B, 6640] -> [B*6640]
        logits_2d = logits.float().permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_1d = labels.reshape(-1)
        return self.loss_fn(logits_2d, labels_1d)

    def training_step(self, batch, batch_idx):
        logits = self(batch["gnn_node_idx"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["gnn_node_idx"])
        if "label" in batch:
            loss = self._loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        lp = torch.cat(self._val_preds,  0)
        ll = torch.cat(self._val_labels, 0)
        if self.trainer.world_size > 1:
            lp, ll = _gather_tensors(lp, ll, self.device, self.trainer.world_size)
        f1 = compute_per_gene_f1(lp.numpy(), ll.numpy())
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["gnn_node_idx"])
        probs  = torch.softmax(logits, dim=1)
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (
            torch.cat(self._test_labels, 0)
            if self._test_labels
            else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        )

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
            all_probs, all_labels = local_probs, dummy_labels
            all_pert, all_syms    = self._test_pert_ids, self._test_symbols

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id (DDP may pad with duplicates)
            seen_pids:    set = set()
            dedup_indices: List[int] = []
            for i, pid in enumerate(all_pert):
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    dedup_indices.append(i)

            all_probs_np = all_probs.numpy()
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for i in dedup_indices:
                    fh.write(
                        f"{all_pert[i]}\t{all_syms[i]}\t"
                        f"{json.dumps(all_probs_np[i].tolist())}\n"
                    )
            self.print(f"[Node2-1-1-2-1] Saved {len(dedup_indices)} test predictions → {pred_path}")

            if self._test_labels:
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-1-1-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Two non-overlapping parameter groups:
        # 1. STRING_GNN trainable backbone (mps.6.*, mps.7.*, post_mp.*) — lower LR
        # 2. All other trainable params (bilinear head, OOV emb, output gene emb) — higher LR
        backbone_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and n.startswith("gnn.")
        ]
        backbone_set = set(id(p) for p in backbone_params)
        head_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and id(p) not in backbone_set
        ]

        self.print(f"[Node2-1-1-2-1] Backbone params: {sum(p.numel() for p in backbone_params):,}")
        self.print(f"[Node2-1-1-2-1] Head params:     {sum(p.numel() for p in head_params):,}")

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": hp.lr_backbone, "weight_decay": hp.weight_decay},
                {"params": head_params,     "lr": hp.lr_head,     "weight_decay": hp.weight_decay},
            ]
        )

        # Cosine annealing with linear warmup
        warmup = hp.warmup_steps
        total  = hp.max_steps_total

        def lr_lambda(current_step: int):
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            # Clamp progress at 1.0 to avoid unintended second cycle
            progress = min(
                1.0,
                float(current_step - warmup) / float(max(1, total - warmup))
            )
            return max(1e-6, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items()
              if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} params ({100*trained/total:.2f}%)")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-1-1-2-1 — STRING_GNN Partial Fine-Tuning + Rank-768 Bilinear Head with Label Smoothing"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--lr-backbone",       type=float, default=5e-5,
                   help="LR for trainable GNN backbone layers (mps.6, mps.7, post_mp)")
    p.add_argument("--lr-head",           type=float, default=5e-4,
                   help="LR for bilinear MLP head + output gene embeddings")
    p.add_argument("--weight-decay",      type=float, default=1e-3)
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--class-weight-down", type=float, default=2.0,
                   help="Class weight for down-regulated class (-1, class 0)")
    p.add_argument("--class-weight-neutral", type=float, default=0.5,
                   help="Class weight for neutral class (0, class 1)")
    p.add_argument("--class-weight-up",   type=float, default=5.0,
                   help="Class weight for up-regulated class (+1, class 2); increased from 4.0")
    p.add_argument("--head-dropout",      type=float, default=0.20)
    p.add_argument("--bilinear-rank",     type=int,   default=768)
    p.add_argument("--n-residual-blocks", type=int,   default=8)
    p.add_argument("--label-smoothing",   type=float, default=0.05)
    p.add_argument("--warmup-steps",      type=int,   default=100)
    p.add_argument("--micro-batch-size",  type=int,   default=16)
    p.add_argument("--global-batch-size", type=int,   default=128,
                   help="Must be multiple of micro_batch_size * 8 for up to 8 GPUs")
    p.add_argument("--max-epochs",        type=int,   default=300)
    p.add_argument("--patience",          type=int,   default=60)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",    type=int,   default=None)
    p.add_argument("--fast-dev-run",      action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()

    # Ensure deterministic behavior
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute gradient accumulation steps
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # Estimate total training steps for LR schedule
    # Calibrated to ~100 epochs (actual expected training duration based on tree history)
    train_size          = 1416
    steps_per_epoch     = max(1, train_size // (args.micro_batch_size * n_gpus))
    eff_steps_per_epoch = max(1, steps_per_epoch // accum)
    # Target: 100 epoch horizon for cosine schedule (calibrated fix from parent node2-1-1-2)
    max_steps_total     = eff_steps_per_epoch * 100

    class_weights = [
        args.class_weight_down,
        args.class_weight_neutral,
        args.class_weight_up,
    ]

    dm  = GNNDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = GNNLitModule(
        lr_backbone      = args.lr_backbone,
        lr_head          = args.lr_head,
        weight_decay     = args.weight_decay,
        focal_gamma      = args.focal_gamma,
        class_weights    = class_weights,
        head_dropout     = args.head_dropout,
        bilinear_rank    = args.bilinear_rank,
        n_residual_blocks = args.n_residual_blocks,
        label_smoothing  = args.label_smoothing,
        warmup_steps     = args.warmup_steps,
        max_steps_total  = max_steps_total,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb   = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
    lr_cb   = LearningRateMonitor(logging_interval="step")
    pb_cb   = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps_trainer: int = -1
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps_trainer = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2
        limit_test  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps_trainer,
        accumulate_grad_batches=accum,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 2-1-1-2-1 — STRING_GNN Partial Fine-Tuning + Rank-768 Bilinear Head with Label Smoothing\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
