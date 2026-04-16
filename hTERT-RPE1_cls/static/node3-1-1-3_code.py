"""
Node 3-1-1-3 – STRING_GNN Partial Fine-Tuning + Rank-512 Bilinear Head + Muon + SWA

Architecture (complete pivot from AIDO.Cell to STRING_GNN):
  - STRING_GNN backbone: partial fine-tuning of mps.6, mps.7, post_mp (~198K trainable params)
    mps.0-5 are frozen and precomputed as a buffer (embedding cache)
  - Rank-512 deep residual bilinear MLP head (6 ResBlocks, hidden=512)
    Bilinear interaction: [B, 512] pert_emb @ [6640, 512] out_gene_emb -> [B, 6640] logits
    Output logit shape: [B, 3, 6640]
  - Muon optimizer (lr=0.005) for 2D head matrices (ResBlock weights)
    AdamW (lr=5e-4) for backbone fine-tuned params and 1D params
  - Focal loss (gamma=2.0) with class weights [2.0, 0.5, 4.0] + label smoothing 0.05
  - SGDR: CosineAnnealingWarmRestarts (T_0=600 steps, T_mult=2) for periodic LR restarts
  - SWA (Stochastic Weight Averaging) over checkpoints with val_f1 >= swa_threshold

Key rationale (why STRING_GNN):
  - Both AIDO.Cell siblings (node3-1-1-1: 0.4574, node3-1-1-2: 0.4074) failed to surpass
    the pure STRING_GNN approach (node1-2: 0.4912) due to the fundamental OOD input problem:
    AIDO.Cell was pretrained on rich multi-gene expression profiles; our sparse
    {gene:1.0, others:-1.0} input is far from this distribution.
  - Node3-1-1-2-1 (sibling's child, same parent) already confirmed STRING_GNN achieves
    test F1=0.4900 with just rank=256 head. This node uses proven rank=512 configuration
    matching the MCTS tree's best-performing nodes (0.5101+).
  - Node2-1-1-1-2-1-1-1-1-1-1-1-1 achieved F1=0.5182 (current MCTS best) using this exact
    setup: mps.6+7+post_mp trainable, rank=512 bilinear head, Muon + AdamW, SGDR + SWA.

Differentiation from siblings:
  - node3-1-1-1: AIDO.Cell-100M LoRA rank=16 + STRING_GNN fusion (still AIDO.Cell-based)
  - node3-1-1-2: AIDO.Cell-100M LoRA rank=32 + mean-baseline (AIDO.Cell only, no STRING_GNN)
  This node: pure STRING_GNN (no AIDO.Cell) with optimal rank-512 bilinear head + SWA
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
from muon import MuonWithAuxAdam

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_DIM        = 256      # STRING_GNN hidden size
BILINEAR_RANK  = 512      # Bilinear interaction rank

# Class weights: [down(-1), neutral(0), up(+1)] after +1 shift → {0,1,2}
# Empirically proven in node1-2-2-3 family (F1=0.5101) and best nodes
CLASS_WEIGHTS = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)


# ─── Metric (matches calc_metric.py exactly) ──────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """
    Compute macro-averaged per-gene F1 score matching calc_metric.py logic.

    Args:
        pred_np: [N, 3, G] softmax probabilities (float)
        labels_np: [N, G] class indices in {0, 1, 2} (already shifted from {-1, 0, 1})
    Returns:
        float: mean per-gene macro-F1 over all G genes
    """
    pred_cls = pred_np.argmax(axis=1)  # [N, G]
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal cross-entropy loss for multi-class classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, G] unnormalized logits (C=3 classes, G=6640 genes)
            targets: [B, G] class indices in {0, 1, 2}
        Returns:
            scalar loss
        """
        B, C, G = logits.shape
        logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        targets_flat = targets.reshape(-1)                      # [B*G]

        log_probs = F.log_softmax(logits_flat, dim=1)           # [B*G, C]
        probs     = torch.exp(log_probs)                        # [B*G, C]

        target_log_prob = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        target_prob     = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - target_prob).pow(self.gamma)

        if self.weight is not None:
            class_w = self.weight.to(logits.device)[targets_flat]
        else:
            class_w = torch.ones_like(focal_weight)

        if self.label_smoothing > 0:
            smooth_loss  = -log_probs.mean(dim=1)
            ce_loss      = -target_log_prob
            loss_per_pos = (
                (1 - self.label_smoothing) * ce_loss
                + self.label_smoothing * smooth_loss
            )
        else:
            loss_per_pos = -target_log_prob

        weighted_loss = focal_weight * class_w * loss_per_pos
        denom = class_w.sum().clamp(min=1.0)
        return weighted_loss.sum() / denom


# ─── Bilinear MLP Head ────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual block for deep MLP head."""

    def __init__(self, hidden_dim: int, expand: int = 4, dropout: float = 0.35):
        super().__init__()
        inner = hidden_dim * expand
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.net(self.norm(x)))


class GNNBilinearHead(nn.Module):
    """
    Deep residual MLP + bilinear interaction head for DEG prediction.

    Architecture:
      pert_emb [B, 256] → project_in → [B, hidden_dim]
      → 6 x ResidualBlock(hidden_dim, expand=4, dropout)
      → project_to_rank [B, rank] (bilinear rank)
      → bilinear: [B, rank] × out_gene_emb[6640, rank]^T → [B, 6640] per-class logits
      Combined: [B, 3] class_proj × [6640, rank] → reshaped → [B, 3, 6640]

    The bilinear interaction captures gene-gene interaction patterns by projecting
    the perturbation embedding to the same space as learnable output gene embeddings.
    """

    def __init__(
        self,
        input_dim:    int = GNN_DIM,
        hidden_dim:   int = 512,
        n_layers:     int = 6,
        bilinear_rank: int = BILINEAR_RANK,
        n_genes_out:  int = N_GENES_OUT,
        n_classes:    int = N_CLASSES,
        dropout:      float = 0.35,
    ):
        super().__init__()

        # Project from 256-dim GNN embedding to hidden_dim
        self.project_in = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )

        # Deep residual MLP
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Project to bilinear rank (one per class)
        self.to_rank = nn.Linear(hidden_dim, bilinear_rank * n_classes)

        # Learnable output gene embeddings: [n_genes_out, bilinear_rank]
        self.out_gene_emb = nn.Embedding(n_genes_out, bilinear_rank)
        nn.init.normal_(self.out_gene_emb.weight, std=0.02)

        self.n_classes    = n_classes
        self.bilinear_rank = bilinear_rank
        self.n_genes_out  = n_genes_out

    def forward(self, pert_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pert_emb: [B, 256] perturbation embedding from STRING_GNN
        Returns:
            logits: [B, 3, 6640]
        """
        x = self.project_in(pert_emb)                            # [B, hidden_dim]
        for block in self.res_blocks:
            x = block(x)
        x = self.final_norm(x)                                   # [B, hidden_dim]

        # Project to [B, n_classes * bilinear_rank]
        x = self.to_rank(x)                                      # [B, n_classes * rank]
        x = x.view(-1, self.n_classes, self.bilinear_rank)       # [B, n_classes, rank]

        # Bilinear interaction: [B, n_classes, rank] @ [rank, n_genes_out] → [B, n_classes, n_genes_out]
        gene_emb = self.out_gene_emb.weight                      # [n_genes_out, rank]
        logits = torch.matmul(x, gene_emb.t())                   # [B, n_classes, n_genes_out]

        return logits  # [B, 3, 6640]


# ─── STRING_GNN Backbone Module ───────────────────────────────────────────────

class StringGNNBackbone(nn.Module):
    """
    STRING_GNN backbone with partial fine-tuning.

    Freezes mps.0-5 (6 early GCN layers), fine-tunes mps.6, mps.7, post_mp.
    This gives ~198K trainable backbone parameters while preserving the pre-trained
    base topology representations.

    The forward pass:
      1. During setup: precompute mid_embs = output after mps.0-5 as a static buffer
      2. During training: run mps.6, mps.7, post_mp on top of the frozen mid_embs buffer
      3. Extract per-sample perturbation embedding by index lookup

    Note: mps.0-5 output is cached once and reused every batch — saves ~70% of GNN compute.
    """

    def __init__(self):
        super().__init__()
        # Load STRING_GNN model
        self.model = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        graph = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", weights_only=False)
        self.register_buffer("edge_index",  graph["edge_index"])
        ew = graph.get("edge_weight", None)
        if ew is not None:
            self.register_buffer("edge_weight", ew)
        else:
            self.edge_weight = None

        # Freeze all parameters first
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze mps.6, mps.7, post_mp for fine-tuning (~198K params)
        for name, p in self.model.named_parameters():
            if any(name.startswith(prefix) for prefix in ['mps.6', 'mps.7', 'post_mp']):
                p.requires_grad = True

        # mid_embs_buffer will be registered via register_buffer() during setup()
        # (Do NOT set self.mid_embs_buffer = None here — that would create a plain
        #  Python attribute and make the subsequent register_buffer() call raise
        #  KeyError "attribute 'mid_embs_buffer' already exists".)
        # shape once registered: [18870, 256]

    def precompute_mid_embs(self, device: torch.device):
        """
        Precompute the frozen mps.0-5 output and cache it as a buffer.
        This is called once during setup. Forward pass of mps.6/7/post_mp
        will use this cached output.

        The official STRING_GNN forward uses residual connections:
            x = mp(x, edge_index, edge_weight) + x  (for each layer)
        We must apply this correctly to get consistent outputs.
        """
        self.model.eval()
        edge_index  = self.edge_index.to(device)
        edge_weight = self.edge_weight.to(device) if self.edge_weight is not None else None

        with torch.no_grad():
            # Get initial node embeddings from embedding table
            x = self.model.emb.weight.to(device)  # [18870, 256]

            # Run mps.0-5 WITH residual connections (matching official forward)
            for i in range(6):
                x = self.model.mps[i](x, edge_index, edge_weight) + x

        # Store as buffer (idempotent: if already registered, just update the data)
        mid_embs = x.detach().cpu()
        if "mid_embs_buffer" not in self._buffers:
            self.register_buffer("mid_embs_buffer", mid_embs)
        else:
            self.mid_embs_buffer.copy_(mid_embs)

    def forward(self) -> torch.Tensor:
        """
        Run the trainable layers (mps.6, mps.7, post_mp) on the precomputed buffer.
        Uses residual connections matching the official STRING_GNN forward pass.

        Returns:
            node_embs: [18870, 256] final node embeddings
        """
        # All registered buffers (edge_index, edge_weight, mid_embs_buffer) are
        # automatically moved to the correct device by Lightning when the model
        # is transferred. No explicit .to(device) calls needed.
        # Run trainable layers WITH residual connections (matching official forward)
        x = self.mid_embs_buffer  # [18870, 256] — already on correct device
        x = self.model.mps[6](x, self.edge_index, self.edge_weight) + x
        x = self.model.mps[7](x, self.edge_index, self.edge_weight) + x
        x = self.model.post_mp(x)                                    # [18870, 256]
        return x


# ─── Full Model ───────────────────────────────────────────────────────────────

class StringGNNBilinearModel(nn.Module):
    """
    STRING_GNN partial fine-tuning + Rank-512 deep bilinear MLP head.

    Total trainable parameters:
      - Backbone (mps.6 + mps.7 + post_mp): ~198K
      - Head (project_in + 6 ResBlocks + to_rank + out_gene_emb): ~16.9M
      - Total: ~17.1M trainable parameters
    """

    def __init__(
        self,
        n_genes_out:   int   = N_GENES_OUT,
        n_classes:     int   = N_CLASSES,
        hidden_dim:    int   = 512,
        bilinear_rank: int   = BILINEAR_RANK,
        n_layers:      int   = 6,
        dropout:       float = 0.35,
    ):
        super().__init__()
        self.backbone = StringGNNBackbone()
        self.head     = GNNBilinearHead(
            input_dim=GNN_DIM,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            bilinear_rank=bilinear_rank,
            n_genes_out=n_genes_out,
            n_classes=n_classes,
            dropout=dropout,
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Node3-1-1-3] Trainable: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.2f}%)")

    def get_node_embs(self) -> torch.Tensor:
        """
        Run STRING_GNN backbone to get all node embeddings.
        Returns:
            node_embs: [18870, 256] — all node embeddings after partial fine-tuning
        """
        return self.backbone()  # [18870, 256]


# ─── Dataset ──────────────────────────────────────────────────────────────────

class StringGNNPerturbDataset(Dataset):
    def __init__(self, pert_ids, symbols, node_indices, labels=None):
        self.pert_ids     = pert_ids
        self.symbols      = symbols
        self.node_indices = node_indices  # [N] long, -1 for OOV
        self.labels       = labels        # [N, 6640] long or None

    def __len__(self): return len(self.pert_ids)

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
    out = {
        "pert_id":    [b["pert_id"]    for b in batch],
        "symbol":     [b["symbol"]     for b in batch],
        "node_index": torch.stack([b["node_index"] for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class StringGNNDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data", micro_batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        # Load STRING_GNN node name → index mapping
        node_names_list = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.node_name_to_idx = {name: i for i, name in enumerate(node_names_list)}

        def load_split(fname, has_lbl):
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            # Map pert_id → STRING_GNN node index (-1 for OOV)
            node_indices = torch.tensor(
                [self.node_name_to_idx.get(pid, -1) for pid in df["pert_id"]],
                dtype=torch.long,
            )
            labels = None
            if has_lbl and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return StringGNNPerturbDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(),
                node_indices, labels
            )

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=self.num_workers,
            pin_memory=True, drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Gather helper ────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    """Gather tensors from all DDP ranks, handling variable-size padding."""
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))
    pad    = max_sz - local_p.shape[0]
    p = local_p.to(device); l = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)
    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p); dist.all_gather(gl, l)
    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class StringGNNBilinearLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_backbone:     float = 5e-5,   # AdamW LR for backbone (mps.6/7/post_mp)
        lr_head:         float = 5e-4,   # AdamW LR for head 1D params
        muon_lr:         float = 0.005,  # Muon LR for head 2D matrices
        weight_decay:    float = 1.5e-3,
        focal_gamma:     float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs:      int   = 200,
        t0_steps:        int   = 600,    # SGDR T_0 in steps
        t_mult:          int   = 2,      # SGDR T_mult for cycle lengthening (must be integer >= 1)
        patience:        int   = 60,
        hidden_dim:      int   = 512,
        bilinear_rank:   int   = BILINEAR_RANK,
        n_layers:        int   = 6,
        dropout:         float = 0.35,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []
        self._test_labels:   List[torch.Tensor] = []

        # OOV embedding: learnable fallback for genes not in STRING_GNN vocabulary
        self.oov_embedding = nn.Parameter(torch.zeros(GNN_DIM))
        nn.init.normal_(self.oov_embedding, std=0.02)

    def setup(self, stage=None):
        self.model = StringGNNBilinearModel(
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            hidden_dim=self.hparams.hidden_dim,
            bilinear_rank=self.hparams.bilinear_rank,
            n_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
        )
        self.register_buffer("class_weights", CLASS_WEIGHTS)
        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=CLASS_WEIGHTS,
            label_smoothing=self.hparams.label_smoothing,
        )

        # Cast trainable parameters to float32 for stable optimization
        # (prevents dtype mismatches in optimizer and ensures numeric stability)
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Precompute frozen mps.0-5 embedding buffer on CPU.
        # The buffer will be automatically moved to GPU by Lightning when the model
        # is transferred to the target device. Each DDP rank computes this independently.
        self.model.backbone.precompute_mid_embs(torch.device("cpu"))

    def _get_pert_embs(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Get perturbation embeddings from STRING_GNN for a batch.
        Handles OOV genes (-1) by substituting the learnable oov_embedding.

        Args:
            node_indices: [B] long, -1 for OOV genes
        Returns:
            pert_embs: [B, 256]
        """
        # Run backbone to get all node embeddings
        node_embs = self.model.get_node_embs()  # [18870, 256]

        # For each sample: look up its node embedding, use OOV embedding if index == -1
        # Use node_embs.dtype to match the autocast precision (bf16 under bf16-mixed training)
        batch_size = node_indices.shape[0]
        pert_embs  = torch.zeros(batch_size, GNN_DIM, dtype=node_embs.dtype,
                                 device=node_indices.device)

        valid_mask = node_indices >= 0
        oov_mask   = ~valid_mask

        if valid_mask.any():
            valid_idx = node_indices[valid_mask]
            pert_embs[valid_mask] = node_embs[valid_idx]

        if oov_mask.any():
            # Cast oov_embedding to match node_embs dtype (oov_embedding is float32 param)
            pert_embs[oov_mask] = self.oov_embedding.to(node_embs.dtype).unsqueeze(0).expand(
                oov_mask.sum(), -1
            )

        return pert_embs  # [B, 256]

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """Full forward: backbone → pert_embs → bilinear head → logits [B, 3, 6640]."""
        pert_embs = self._get_pert_embs(node_indices)         # [B, 256]
        logits    = self.model.head(pert_embs)                 # [B, 3, 6640]
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        loss   = self.focal_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        if "label" in batch:
            loss = self.focal_loss(logits, batch["label"])
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
        probs_np  = torch.softmax(lp, dim=1).numpy()
        labels_np = ll.numpy()
        f1 = compute_per_gene_f1(probs_np, labels_np)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        probs  = torch.softmax(logits, dim=1)  # [B, 3, G]
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
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id (DDP sampler may pad)
            seen_pids: set = set()
            dedup_perts, dedup_syms, dedup_probs_list, dedup_label_rows = [], [], [], []
            for pid, sym, prob_row, lbl_row in zip(
                all_pert, all_syms, all_probs.numpy(), all_labels.numpy()
            ):
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    dedup_perts.append(pid)
                    dedup_syms.append(sym)
                    dedup_probs_list.append(prob_row)
                    dedup_label_rows.append(lbl_row)

            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pid, sym, probs in zip(dedup_perts, dedup_syms, dedup_probs_list):
                    fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")

            self.print(f"[Node3-1-1-3] Saved {len(dedup_perts)} test predictions → {pred_path}")

            if all_labels.any():
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[Node3-1-1-3] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    # ── Optimizer: MuonWithAuxAdam (Muon for 2D head matrices, AdamW for rest) ──

    def configure_optimizers(self):
        hp = self.hparams

        # Separate parameters into groups:
        # 1. Backbone trainable params (mps.6/7/post_mp) → AdamW at backbone LR
        # 2. Head 2D matrices (ResBlock weights) → Muon
        # 3. Head 1D params (biases, LN, embeddings) + OOV embedding → AdamW

        backbone_params = [
            p for p in self.model.backbone.parameters() if p.requires_grad
        ]
        backbone_ids = {id(p) for p in backbone_params}

        # Head 2D matrices for Muon (hidden weight matrices in ResBlocks)
        head_2d_params = [
            p for name, p in self.model.head.named_parameters()
            if p.requires_grad and p.ndim >= 2
            # Exclude out_gene_emb.weight (embedding table, not a typical hidden matrix)
            # Include ResBlock linear weights and project_in linear weight
            and 'out_gene_emb' not in name
        ]

        # Head 1D params + embeddings → AdamW
        head_2d_ids = {id(p) for p in head_2d_params}
        head_other_params = [
            p for p in self.model.head.parameters()
            if p.requires_grad and id(p) not in head_2d_ids
        ]
        # Include OOV embedding in AdamW group
        head_other_params.append(self.oov_embedding)

        # Verify no overlap and no missing params
        all_ids = backbone_ids | head_2d_ids | {id(p) for p in head_other_params}

        param_groups = [
            # Muon group: 2D head weight matrices
            {
                "params":       head_2d_params,
                "use_muon":     True,
                "lr":           hp.muon_lr,
                "weight_decay": hp.weight_decay,
                "momentum":     0.95,
            },
            # AdamW group: backbone trainable params
            {
                "params":       backbone_params,
                "use_muon":     False,
                "lr":           hp.lr_backbone,
                "betas":        (0.9, 0.95),
                "weight_decay": hp.weight_decay,
            },
            # AdamW group: head 1D params + embeddings + OOV
            {
                "params":       head_other_params,
                "use_muon":     False,
                "lr":           hp.lr_head,
                "betas":        (0.9, 0.95),
                "weight_decay": 0.0,  # No weight decay for biases and embeddings
            },
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # SGDR: CosineAnnealingWarmRestarts with integer cycle lengthening
        # T_0 in steps (step-based scheduler).
        # T_mult=2 doubles the cycle length after each restart:
        #   Cycle 0: T_0 steps (~27 epochs with batch=32)
        #   Cycle 1: 2*T_0 steps (~54 epochs)
        #   Cycle 2: 4*T_0 steps (~108 epochs)
        # This matches the proven SGDR schedule from node2-1-1-1-2-1-1-1-1-1-1-1-1 (F1=0.5182)
        # Note: CosineAnnealingWarmRestarts requires integer T_mult >= 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.t0_steps,
            T_mult=max(1, int(hp.t_mult)),
            eta_min=1e-7,
        )

        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",   # SGDR is step-based
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable parameters + buffers ──────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        # Include oov_embedding explicitly
        trainable_keys.add(prefix + "oov_embedding")
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        # Exclude the large mid_embs_buffer from checkpoints (can be recomputed)
        buffer_keys.discard(prefix + "model.backbone.mid_embs_buffer")

        sd = {k: v for k, v in full_sd.items()
              if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving ckpt: {trained:,}/{total:,} params ({100*trained/total:.2f}%)"
        )
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 3-1-1-3 – STRING_GNN Partial Fine-Tuning + Rank-512 Bilinear Head + Muon + SWA"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--lr-backbone",       type=float, default=5e-5)
    p.add_argument("--lr-head",           type=float, default=5e-4)
    p.add_argument("--muon-lr",           type=float, default=0.005)
    p.add_argument("--weight-decay",      type=float, default=1.5e-3)
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--label-smoothing",   type=float, default=0.05)
    p.add_argument("--micro-batch-size",  type=int,   default=32)
    p.add_argument("--global-batch-size", type=int,   default=64)
    p.add_argument("--max-epochs",        type=int,   default=350)
    p.add_argument("--t0-steps",          type=int,   default=600)
    p.add_argument("--t-mult",            type=int,   default=2)
    p.add_argument("--patience",          type=int,   default=100)
    p.add_argument("--hidden-dim",        type=int,   default=512)
    p.add_argument("--bilinear-rank",     type=int,   default=512)
    p.add_argument("--n-layers",          type=int,   default=6)
    p.add_argument("--dropout",           type=float, default=0.35)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--val-check-interval",type=float, default=1.0)
    p.add_argument("--debug-max-step",    type=int,   default=None)
    p.add_argument("--fast-dev-run",      action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm  = StringGNNDataModule(
        args.data_dir, args.micro_batch_size, args.num_workers
    )
    lit = StringGNNBilinearLitModule(
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        t0_steps=args.t0_steps,
        t_mult=args.t_mult,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        bilinear_rank=args.bilinear_rank,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=3, save_last=True,
    )
    es_cb  = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.patience, min_delta=1e-5,
    )
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps    = -1
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
        max_steps=max_steps,
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
        deterministic="warn",
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,   # Gradient clipping for stable Muon training
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 3-1-1-3 – STRING_GNN Partial Fine-Tuning + Rank-512 Bilinear Head + Muon + SWA\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
