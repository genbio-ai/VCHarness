"""
Node 2-2-1-1 – Partial STRING_GNN Fine-Tuning + rank=512 Bilinear Head + Calibrated LR Schedule

This node represents a complete pivot from the AIDO.Cell lineage (parent node2-2-1, F1=0.4421)
to the proven STRING_GNN architecture, specifically targeting the known bottleneck in tree-best
node2-1-3 (F1=0.5047): the LR schedule mismatch and insufficient regularization.

Key improvements over parent's AIDO.Cell approach AND over node2-1-3 (tree best):
  1. BACKBONE PIVOT: Replace AIDO.Cell-100M (F1 ceiling ~0.44) with partial STRING_GNN
     fine-tuning — proven to achieve F1=0.5047 (tree best) on this exact task.
  2. CALIBRATED LR SCHEDULE: total_steps=1200 (vs node2-1-3's 6600 which barely decayed).
     At ~11 steps/epoch with global_batch_size=128 on 8 GPUs, 1200 steps ~ 109 epochs,
     ensuring meaningful cosine annealing through the actual training window.
  3. STRONGER REGULARIZATION: dropout=0.3 (vs 0.2), weight_decay=2e-3 (vs 1e-3) —
     addressing node2-1-3's observed overfitting (train/val loss ratio of 7.2x at epoch 82).
  4. INTERMEDIATE CLASS WEIGHTS [1.5, 0.5, 3.0]: Synthesizing node2-1-2's mild weights
     [1.5, 0.8, 2.5] (F1=0.5011) and node2-1-3's stronger weights [2.0, 0.5, 4.0] (F1=0.5047),
     retaining neutral suppression (0.5) while moderating up-regulated amplification (3.0 vs 4.0)
     to reduce calibration degradation.
  5. RANK=512 bilinear head retained (proven to improve over rank=256 in the tree).
  6. PARTIAL BACKBONE: Freeze emb.weight + mps.0-5; train mps.6, mps.7, post_mp (~198K params).

Expected F1: 0.505–0.515 (above node2-1-3's tree best of 0.5047 via LR fix + regularization).
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

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_DIM        = 256    # STRING_GNN hidden dimension


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

class StringGNNPerturbDataset(Dataset):
    """Stores pert_ids, symbols, node_indices, and labels."""

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        node_indices: List[int],   # index into STRING_GNN's 18870-node vocab; -1 = OOV
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long (class indices 0/1/2)
    ):
        self.pert_ids     = pert_ids
        self.symbols      = symbols
        self.node_indices = node_indices
        self.labels       = labels

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
            "node_index": self.node_indices[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    B = len(batch)
    pert_ids    = [b["pert_id"]    for b in batch]
    symbols     = [b["symbol"]     for b in batch]
    node_index  = torch.tensor([b["node_index"] for b in batch], dtype=torch.long)  # [B]
    out = {
        "pert_id":    pert_ids,
        "symbol":     symbols,
        "node_index": node_index,
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])  # [B, 6640]
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class StringGNNDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 16,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage: Optional[str] = None):
        # Load STRING_GNN node names for pert_id -> node_index mapping
        node_names_path = Path(STRING_GNN_DIR) / "node_names.json"
        node_names: List[str] = json.loads(node_names_path.read_text())
        # node_names[i] is the Ensembl gene ID for node index i
        pert_id_to_idx: Dict[str, int] = {pid: i for i, pid in enumerate(node_names)}

        def get_node_index(pert_id: str) -> int:
            """Return the node index for a pert_id; -1 for OOV."""
            return pert_id_to_idx.get(str(pert_id), -1)

        def load_split(fname: str, has_label: bool) -> StringGNNPerturbDataset:
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            pert_ids     = df["pert_id"].astype(str).tolist()
            symbols      = df["symbol"].tolist()
            node_indices = [get_node_index(pid) for pid in pert_ids]
            labels = None
            if has_label and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return StringGNNPerturbDataset(pert_ids, symbols, node_indices, labels)

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

        # Compute OOV rates
        train_oov = sum(1 for i in self.train_ds.node_indices if i == -1)
        print(f"[DataModule] Train OOV: {train_oov}/{len(self.train_ds.node_indices)}")

    def _loader(self, ds: StringGNNPerturbDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Focal Loss with Class Weights ────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss with optional class weights. No label smoothing
    (per collective memory: label smoothing consistently hurts per-gene macro F1).

    Input: logits [N, C], targets [N].
    class_weights [C] — per-class multiplier to address class imbalance.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs    = F.log_softmax(logits, dim=-1)          # [N, C]
        ce_per_sample = F.nll_loss(log_probs, targets, reduction="none")  # [N]
        probs_true   = log_probs.gather(-1, targets.unsqueeze(-1)).exp().squeeze(-1)
        focal_weight = (1.0 - probs_true.detach()) ** self.gamma  # [N]
        loss         = focal_weight * ce_per_sample                # [N]
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]           # [N]
            loss = loss * sample_weights
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─── Residual Block for MLP Head ──────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Pre-norm residual block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> residual add.
    expand=4 gives hidden_dim * 4 in the intermediate layer.
    """

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.3):
        super().__init__()
        inner_dim = dim * expand
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ─── Deep Residual Bilinear Head ──────────────────────────────────────────────

class GNNBilinearHead(nn.Module):
    """
    6-layer deep residual bilinear MLP head.
    Architecture:
      proj_in: Linear(gnn_dim, hidden_dim)
      6 x ResidualBlock(hidden_dim, expand=4, dropout=dropout)
      proj_out: Linear(hidden_dim, n_classes * bilinear_rank)
      reshape: [B, n_classes, bilinear_rank]
      einsum("bcr,gr->bcg", proj, out_gene_emb) -> logits [B, n_classes, n_genes_out]

    Param count (rank=512, hidden_dim=512):
      proj_in: 256 * 512 ≈ 0.13M
      6 ResidualBlocks: 6 * (512*2048 + 2048*512) ≈ 12.6M
      proj_out: 512 * 1536 ≈ 0.79M
      out_gene_emb: 6640 * 512 ≈ 3.40M
      Total: ~16.9M (same proven architecture as node2-1-3)
    """

    def __init__(
        self,
        gnn_dim: int        = GNN_DIM,
        hidden_dim: int     = 512,
        bilinear_rank: int  = 512,
        n_classes: int      = N_CLASSES,
        n_genes_out: int    = N_GENES_OUT,
        n_blocks: int       = 6,
        dropout: float      = 0.3,   # increased from 0.2 in node2-1-3
    ):
        super().__init__()
        self.n_classes     = n_classes
        self.bilinear_rank = bilinear_rank
        self.n_genes_out   = n_genes_out

        # Project from GNN embedding dim to hidden_dim
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)

        # Deep residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=4, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Project from hidden_dim to bilinear space (n_classes * bilinear_rank)
        self.proj_out = nn.Linear(hidden_dim, n_classes * bilinear_rank)

        # Learnable output gene embeddings [n_genes_out, bilinear_rank]
        self.out_gene_emb = nn.Embedding(n_genes_out, bilinear_rank)
        nn.init.normal_(self.out_gene_emb.weight, std=0.02)

    def forward(self, pert_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pert_emb: [B, GNN_DIM=256] perturbation embedding from STRING_GNN
        Returns:
            logits: [B, n_classes, n_genes_out]
        """
        B = pert_emb.shape[0]

        x = self.proj_in(pert_emb)          # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)                    # [B, hidden_dim]

        proj = self.proj_out(x)             # [B, n_classes * bilinear_rank]
        proj = proj.view(B, self.n_classes, self.bilinear_rank)  # [B, 3, bilinear_rank]

        # Bilinear interaction: [B, 3, rank] x [n_genes_out, rank]^T -> [B, 3, n_genes_out]
        out_emb = self.out_gene_emb.weight   # [n_genes_out, bilinear_rank]
        logits  = torch.einsum("bcr,gr->bcg", proj, out_emb)  # [B, 3, n_genes_out]
        return logits


# ─── Full Model: Partial STRING_GNN + Bilinear Head ───────────────────────────

class StringGNNPerturbModel(nn.Module):
    """
    Partial STRING_GNN fine-tuning + rank=512 deep residual bilinear head.

    Partial fine-tuning strategy (from STRING_GNN skill documentation):
    - Freeze: emb.weight, mps.0-5 (early GNN layers)
    - Train: mps.6, mps.7, post_mp (~198K trainable backbone params)
    - Pre-compute: activations after mps.0-5 stored as a buffer (partial_emb)
      This makes each forward pass efficient — only 2 GCN layers run differentially.

    OOV handling: genes not found in STRING_GNN vocabulary use a learnable oov_emb vector.
    """

    def __init__(
        self,
        head_hidden_dim: int    = 512,
        head_bilinear_rank: int = 512,
        head_dropout: float     = 0.3,
        head_n_blocks: int      = 6,
    ):
        super().__init__()

        # ── Load full STRING_GNN model ──────────────────────────────────────
        model_dir  = Path(STRING_GNN_DIR)
        gnn_model  = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        gnn_model  = gnn_model.float()

        # Load graph data (edge_index, edge_weight)
        graph_data = torch.load(model_dir / "graph_data.pt", map_location="cpu")
        edge_index  = graph_data["edge_index"]          # [2, E] long
        edge_weight = graph_data.get("edge_weight", None)  # [E] float or None
        if edge_weight is not None:
            edge_weight = edge_weight.float()

        # Register graph data as non-trainable buffers
        self.register_buffer("edge_index",  edge_index)
        self.register_buffer("edge_weight", edge_weight if edge_weight is not None
                             else torch.ones(edge_index.shape[1], dtype=torch.float32))

        # ── Pre-compute partial embeddings (after mps.0-5) ──────────────────
        # This avoids re-running the frozen layers at every forward pass.
        gnn_model.eval()
        with torch.no_grad():
            # Full GNN forward to get intermediate hidden states
            full_out = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight if edge_weight is not None else
                    torch.ones(edge_index.shape[1], dtype=torch.float32),
                output_hidden_states=True,
            )
            # hidden_states[0] = initial emb, [1..8] = after each GNN layer
            # After mps.0-5: hidden_states[6] (0-indexed as emb + 6 layers)
            partial_emb = full_out.hidden_states[6].float().clone()  # [N, 256] after mps.0-5

        # Store partial embedding as a non-trainable buffer
        self.register_buffer("partial_emb", partial_emb)  # [18870, 256]

        # ── Store trainable GNN components (mps.6, mps.7, post_mp) ─────────
        # Extract GNN layer modules for the trainable final layers
        self.gnn_mps_6   = gnn_model.mps[6]    # GNNLayer (GCNConv + LayerNorm + GELU + Dropout)
        self.gnn_mps_7   = gnn_model.mps[7]    # GNNLayer
        self.gnn_post_mp = gnn_model.post_mp   # Linear(256, 256)

        # Ensure trainable GNN components are in float32
        self.gnn_mps_6.float()
        self.gnn_mps_7.float()
        self.gnn_post_mp.float()

        # Mark as trainable (will be set by named_parameters via requires_grad)
        for p in self.gnn_mps_6.parameters():
            p.requires_grad = True
        for p in self.gnn_mps_7.parameters():
            p.requires_grad = True
        for p in self.gnn_post_mp.parameters():
            p.requires_grad = True

        # ── OOV embedding for genes not in STRING_GNN vocabulary ────────────
        self.oov_emb = nn.Parameter(torch.zeros(1, GNN_DIM))

        # ── Prediction head ─────────────────────────────────────────────────
        self.head = GNNBilinearHead(
            gnn_dim=GNN_DIM,
            hidden_dim=head_hidden_dim,
            bilinear_rank=head_bilinear_rank,
            n_classes=N_CLASSES,
            n_genes_out=N_GENES_OUT,
            n_blocks=head_n_blocks,
            dropout=head_dropout,
        )

    def _get_adapted_embeddings(self) -> torch.Tensor:
        """
        Run the trainable tail of STRING_GNN (mps.6, mps.7, post_mp) starting
        from the frozen pre-computed partial_emb buffer.

        Returns: adapted_emb [N, 256] — final per-node embeddings.

        The residual addition in the original StringGNNModel.forward is:
            x = mp(x, edge_index, edge_weight) + x
        where mp is each GNNLayer (which internally applies norm, conv, act, dropout).
        We replicate this structure here.
        """
        device = self.partial_emb.device
        ei = self.edge_index.to(device)
        ew = self.edge_weight.to(device)

        # Start from frozen pre-computed partial embedding
        x = self.partial_emb.float()

        # Trainable layers: apply each GNNLayer + residual
        # All outputs explicitly cast to float32 to avoid bf16-mixed autocast dtype mismatches
        x = self.gnn_mps_6(x, ei, ew).float() + x   # [N, 256]
        x = self.gnn_mps_7(x, ei, ew).float() + x   # [N, 256]
        x = self.gnn_post_mp(x).float()               # [N, 256] explicit float32
        return x

    def forward(
        self,
        node_indices: torch.Tensor,   # [B] long — STRING_GNN node index; -1 = OOV
    ) -> torch.Tensor:
        """Returns logits [B, 3, 6640]."""
        # Get adapted node embeddings from the trainable GNN tail
        adapted_emb = self._get_adapted_embeddings()  # [18870, 256]

        B = node_indices.shape[0]
        device = adapted_emb.device

        # Gather per-sample embeddings; replace OOV indices with oov_emb
        oov_mask = (node_indices == -1)  # [B]
        safe_indices = node_indices.clone()
        safe_indices[oov_mask] = 0  # temporary placeholder

        pert_emb = adapted_emb[safe_indices, :]  # [B, 256]
        if oov_mask.any():
            # Cast oov_emb to match pert_emb dtype to avoid bf16-mixed autocast mismatches
            pert_emb[oov_mask] = self.oov_emb.to(dtype=pert_emb.dtype).expand(oov_mask.sum(), -1)

        # Prediction head
        logits = self.head(pert_emb)   # [B, 3, 6640]
        return logits


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p: torch.Tensor, local_l: torch.Tensor,
                    device, world_size: int):
    """All-gather variable-length tensors for DDP metric aggregation."""
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

class StringGNNLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_backbone: float        = 5e-5,    # conservative LR for pretrained GNN layers
        lr_head: float            = 5e-4,    # proven optimal for the fresh head
        weight_decay: float       = 2e-3,    # increased from 1e-3 in node2-1-3 for regularization
        focal_gamma: float        = 2.0,
        class_weight_down: float  = 1.5,     # intermediate between node2-1-2 and node2-1-3
        class_weight_neutral: float = 0.5,   # retain neutral suppression
        class_weight_up: float    = 3.0,     # moderate amplification (vs 4.0 in node2-1-3)
        head_hidden_dim: int      = 512,
        head_bilinear_rank: int   = 512,
        head_dropout: float       = 0.3,     # increased from 0.2 for regularization
        head_n_blocks: int        = 6,
        warmup_steps: int         = 100,
        max_steps_total: int      = 1200,    # calibrated to ~109 epochs (fixed from node2-1-3's 6600)
        gradient_clip_val: float  = 1.0,
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
        self.model = StringGNNPerturbModel(
            head_hidden_dim=self.hparams.head_hidden_dim,
            head_bilinear_rank=self.hparams.head_bilinear_rank,
            head_dropout=self.hparams.head_dropout,
            head_n_blocks=self.hparams.head_n_blocks,
        )

        # Cast all trainable parameters to float32 for stable optimization
        # and to avoid dtype mismatches with bf16-mixed autocast
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Class-weighted focal loss (no label smoothing per collective memory guidance)
        cw = torch.tensor([
            self.hparams.class_weight_down,
            self.hparams.class_weight_neutral,
            self.hparams.class_weight_up,
        ], dtype=torch.float32)
        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            class_weights=cw,
        )

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        return self.model(node_indices)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Reshape to 2D for focal loss. logits: [B,3,6640] -> [B*6640, 3], labels: [B,6640] -> [B*6640]."""
        logits_2d = logits.float().permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_1d = labels.reshape(-1)
        return self.focal_loss(logits_2d, labels_1d)

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
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
        logits = self(batch["node_index"])
        probs  = torch.softmax(logits, dim=1)
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (torch.cat(self._test_labels, 0) if self._test_labels
                        else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long))

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
            all_pert, all_syms   = self._test_pert_ids, self._test_symbols

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id (DDP sampler may pad with duplicates)
            seen_pids: set = set()
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
            self.print(f"[Node2-2-1-1] Saved {len(dedup_indices)} test predictions -> {pred_path}")
            if all_labels.any():
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-2-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Two parameter groups: trainable backbone tail (mps.6, mps.7, post_mp, oov_emb)
        # vs prediction head
        backbone_params = []
        head_params     = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if (name.startswith("gnn_mps_6.")
                    or name.startswith("gnn_mps_7.")
                    or name.startswith("gnn_post_mp.")
                    or name == "oov_emb"):
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": hp.lr_backbone},
                {"params": head_params,     "lr": hp.lr_head},
            ],
            weight_decay=hp.weight_decay,
        )

        # Cosine annealing with linear warmup
        # total_steps=1200 calibrated to ~109 epochs (fixed from node2-1-3's 6600 mismatch)
        warmup = hp.warmup_steps
        total  = hp.max_steps_total

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total - warmup))
            progress = min(progress, 1.0)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers to minimize checkpoint size."""
        full_sd        = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items() if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} params ({100*trained/total:.2f}%)")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-2-1-1 – Partial STRING_GNN + rank=512 Bilinear Head + Calibrated LR"
    )
    p.add_argument("--data-dir",              type=str,   default="data")
    p.add_argument("--lr-backbone",           type=float, default=5e-5)
    p.add_argument("--lr-head",               type=float, default=5e-4)
    p.add_argument("--weight-decay",          type=float, default=2e-3)
    p.add_argument("--focal-gamma",           type=float, default=2.0)
    p.add_argument("--class-weight-down",     type=float, default=1.5)
    p.add_argument("--class-weight-neutral",  type=float, default=0.5)
    p.add_argument("--class-weight-up",       type=float, default=3.0)
    p.add_argument("--head-hidden-dim",       type=int,   default=512)
    p.add_argument("--head-bilinear-rank",    type=int,   default=512)
    p.add_argument("--head-dropout",          type=float, default=0.3)
    p.add_argument("--head-n-blocks",         type=int,   default=6)
    p.add_argument("--warmup-steps",          type=int,   default=100)
    p.add_argument("--micro-batch-size",      type=int,   default=16)
    p.add_argument("--global-batch-size",     type=int,   default=128)
    p.add_argument("--max-epochs",            type=int,   default=200)
    p.add_argument("--patience",              type=int,   default=50)
    p.add_argument("--num-workers",           type=int,   default=2)
    p.add_argument("--gradient-clip-val",     type=float, default=1.0)
    p.add_argument("--val-check-interval",    type=float, default=1.0)
    p.add_argument("--debug-max-step",        type=int,   default=None,
                   help="Limit training/val/test steps for quick debugging")
    p.add_argument("--fast-dev-run",          action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calibrate LR schedule to actual training duration
    # With global_batch_size=128 and 1,416 training samples:
    #   steps_per_epoch = ceil(1416 / 128) = 12 (with 8 GPUs: 128 samples/step)
    #   1200 steps / 11 steps_per_epoch ~ 109 epochs — aligned with patience-based early stopping
    steps_per_epoch = max(1, int(np.ceil(1416 / args.global_batch_size)))
    max_steps_total = steps_per_epoch * 100  # ~100 epochs horizon for cosine schedule
    if args.debug_max_step is not None:
        max_steps_total = args.debug_max_step

    dm  = StringGNNDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = StringGNNLitModule(
        lr_backbone          = args.lr_backbone,
        lr_head              = args.lr_head,
        weight_decay         = args.weight_decay,
        focal_gamma          = args.focal_gamma,
        class_weight_down    = args.class_weight_down,
        class_weight_neutral = args.class_weight_neutral,
        class_weight_up      = args.class_weight_up,
        head_hidden_dim      = args.head_hidden_dim,
        head_bilinear_rank   = args.head_bilinear_rank,
        head_dropout         = args.head_dropout,
        head_n_blocks        = args.head_n_blocks,
        warmup_steps         = args.warmup_steps,
        max_steps_total      = max_steps_total,
        gradient_clip_val    = args.gradient_clip_val,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.patience, min_delta=1e-5,
    )
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps:   int | None = None
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps   = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2
        limit_test  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    accum    = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        accumulate_grad_batches=accum,
        gradient_clip_val=args.gradient_clip_val,
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
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 2-2-1-1 – Partial STRING_GNN + rank=512 Bilinear Head + Calibrated LR\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
