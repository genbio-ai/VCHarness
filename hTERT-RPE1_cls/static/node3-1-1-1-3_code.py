"""
Node 3-1-1-1-3 — Partial STRING_GNN Fine-Tuning (mps.6+mps.7+post_mp)
                  + Rank-512 Deep Bilinear MLP Head
                  + MuonWithAuxAdam + SGDR Warm Restarts
                  + Quality-Filtered SWA (top-k=15, temp=3.0, threshold=0.497)

Architecture:
  - STRING_GNN backbone with mps.0-5 frozen (precomputed as buffer)
  - Trainable tail: mps.6 + mps.7 + post_mp (~198K params at backbone_lr=1e-5)
  - 6-layer deep residual bilinear MLP head (rank=512, hidden=512, dropout=0.45)
  - Bilinear output: [B, 3, 512] x out_gene_emb[6640, 512] → [B, 3, 6640]
  - Focal loss (gamma=2.0, class_weights=[2.0, 0.5, 4.0], label_smoothing=0.05)
  - MuonWithAuxAdam: Muon lr=0.005 (ResBlock 2D matrices), AdamW lr=5e-4 (other head),
                     AdamW lr=1e-5 (backbone mps.6+mps.7+post_mp)
  - SGDR warm restarts (T_0=20 epochs, T_mult=2) — proven staircase improvement pattern
  - Quality-filtered SWA (top-k=15, temp=3.0, threshold=0.497, every-3-epochs)
  - patience=150, max_epochs=350

Design rationale:
  - DROPS inductive conditioning (consistently failed in siblings: node3-1-1-1-1, node3-1-1-1-2)
  - Adopts proven tree-best strategy: partial backbone FT + Muon + SGDR + SWA
  - Tree best: node2-1-1-1-2-1-1-1-1-1-1-1-1 at F1=0.5182 uses this exact pattern
  - Differentiates from siblings (both focused on frozen backbone + conditioning):
    - Sibling1 (3-1-1-1-1): frozen STRING_GNN + cond_mlp (LayerNorm bug) → F1=0.4671
    - Sibling2 (3-1-1-1-2): frozen STRING_GNN + scalar gate (dead gate) → F1=0.5003

Key choices vs siblings:
  1. Partial backbone FT (mps.6+mps.7+post_mp) → task-adaptive embeddings (~+0.005 F1 over frozen)
  2. Muon optimizer → 1.35x faster convergence for ResBlock 2D matrices
  3. SGDR warm restarts → staircase improvement across cycles (+0.003-0.010 F1 per cycle)
  4. Quality-filtered SWA → ensemble of best-cycle checkpoints (+0.003-0.006 F1)
  5. dropout=0.45 → proven optimal for this capacity/dataset ratio
  6. label_smoothing=0.05 → from grandparent (0.07 was slightly too high)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # required for deterministic=True with CUDA >= 10.2

import json
import re
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

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

N_GENES_OUT = 6640
N_CLASSES   = 3
GNN_DIM     = 256  # STRING_GNN hidden size

# Number of frozen layers (mps.0 through mps.5 frozen, mps.6+mps.7+post_mp trainable)
N_FROZEN_LAYERS = 6


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """
    Compute macro-averaged per-gene F1 score matching calc_metric.py.

    Args:
        pred_np: [N, 3, G] softmax probabilities (float)
        labels_np: [N, G] class indices in {0, 1, 2} (shifted from {-1, 0, 1})
    Returns:
        float: mean per-gene macro-F1 over all G genes
    """
    pred_cls = pred_np.argmax(axis=1)  # [N, G]
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1   = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    """
    Perturbation DEG dataset.

    Stores precomputed frozen embeddings (output of mps.0-5) per sample for fast
    batch retrieval. Trainable layers mps.6, mps.7, post_mp run at forward time
    in the LightningModule using the full graph (all 18,870 nodes).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        frozen_embeddings: np.ndarray,     # [N_nodes, 256] output of mps.0-5 for all nodes
        node_name_to_idx: Dict[str, int],
        has_labels: bool = True,
    ):
        self.pert_ids  = df["pert_id"].tolist()
        self.symbols   = df["symbol"].tolist()
        self.has_labels = has_labels
        embed_dim = frozen_embeddings.shape[1]

        n = len(df)
        node_indices = np.full(n, -1, dtype=np.int64)
        for i, pid in enumerate(self.pert_ids):
            pid_clean = pid.split(".")[0]  # strip version suffix
            if pid_clean in node_name_to_idx:
                node_indices[i] = node_name_to_idx[pid_clean]

        self.node_indices = torch.from_numpy(node_indices)  # [N] int64, -1 for OOV

        # Per-sample precomputed embeddings for fast batch retrieval
        embeddings = np.zeros((n, embed_dim), dtype=np.float32)
        for i, idx in enumerate(node_indices):
            if idx >= 0:
                embeddings[i] = frozen_embeddings[idx]
        self.embeddings = torch.from_numpy(embeddings)  # [N, 256]

        self.in_vocab = (self.node_indices >= 0)  # [N] bool

        if has_labels:
            rows = []
            for lbl_str in df["label"]:
                rows.append([x + 1 for x in json.loads(lbl_str)])  # {-1,0,1} → {0,1,2}
            self.labels = torch.tensor(rows, dtype=torch.long)  # [N, G]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
            "embedding":  self.embeddings[idx],       # [256] frozen mps.0-5 output
            "node_idx":   self.node_indices[idx],     # int64, -1 for OOV
            "in_vocab":   self.in_vocab[idx],          # bool
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbDataModule(pl.LightningDataModule):
    """
    DataModule for perturbation DEG prediction.

    Precomputes frozen intermediate embeddings (output after mps.0-5) for all 18,870
    STRING_GNN nodes. The trainable layers mps.6+mps.7+post_mp are applied online
    during training using the full graph structure.
    """

    def __init__(
        self,
        data_dir: str,
        micro_batch_size: int,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir        = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers     = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if hasattr(self, "train_ds"):
            return  # Already set up

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[DataModule] Computing frozen intermediate embeddings through mps.{N_FROZEN_LAYERS-1}...")
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", weights_only=False)

        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone = backbone.to(device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        edge_index  = graph["edge_index"].to(device)
        edge_weight = graph.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        with torch.no_grad():
            # Forward through first N_FROZEN_LAYERS (mps.0 through mps.5)
            x = backbone.emb.weight.clone()  # [N_nodes, 256]
            for i in range(N_FROZEN_LAYERS):
                layer  = backbone.mps[i]
                x_conv = layer.conv(x, edge_index, edge_weight)
                x_norm = layer.norm(x_conv)
                x_act  = layer.act(x_norm)
                x      = x + layer.dropout(x_act)

        frozen_embeddings = x.float().cpu().numpy()   # [N_nodes, 256]
        node_name_to_idx  = {name: i for i, name in enumerate(node_names)}

        self.frozen_embeddings        = frozen_embeddings
        self.frozen_embeddings_tensor = torch.from_numpy(frozen_embeddings)  # CPU
        self.node_name_to_idx         = node_name_to_idx
        self.n_gnn_nodes              = len(node_names)
        self.edge_index               = graph["edge_index"]  # CPU
        self.edge_weight              = graph.get("edge_weight")  # CPU or None

        del backbone
        torch.cuda.empty_cache()
        print(f"[DataModule] Frozen embeddings computed: {frozen_embeddings.shape}")

        # Load data splits
        train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
        val_df   = pd.read_csv(self.data_dir / "val.tsv",   sep="\t")
        test_df  = pd.read_csv(self.data_dir / "test.tsv",  sep="\t")

        self.train_ds = PerturbDataset(train_df, frozen_embeddings, node_name_to_idx, True)
        self.val_ds   = PerturbDataset(val_df,   frozen_embeddings, node_name_to_idx, True)
        self.test_ds  = PerturbDataset(test_df,  frozen_embeddings, node_name_to_idx, False)

        n_cov = sum(1 for p in train_df["pert_id"] if p.split(".")[0] in node_name_to_idx)
        print(f"[DataModule] Coverage: {n_cov}/{len(train_df)} train genes in STRING_GNN vocab")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
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
    """Residual MLP block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout + residual."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.45):
        super().__init__()
        inner = dim * expand
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GNNBilinearHead(nn.Module):
    """
    Deep bilinear MLP head for gene-perturbation interaction prediction.

    pert_emb [B, 256]
        → LayerNorm + Linear(256→512)
        → 6 × ResidualBlock(512, expand=4, dropout=0.45)
        → LayerNorm + Dropout + Linear(512→3×rank)
        → reshape [B, 3, rank]
        → einsum([B, 3, rank] × out_gene_emb[G, rank]) → [B, 3, G]
    """

    def __init__(
        self,
        gnn_dim: int    = GNN_DIM,       # 256
        hidden_dim: int = 512,
        rank: int       = 512,
        n_genes: int    = N_GENES_OUT,   # 6640
        n_classes: int  = N_CLASSES,     # 3
        dropout: float  = 0.45,
        n_layers: int   = 6,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.rank      = rank

        self.input_norm = nn.LayerNorm(gnn_dim)
        self.proj_in    = nn.Linear(gnn_dim, hidden_dim)
        self.blocks     = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_norm   = nn.LayerNorm(hidden_dim)
        self.out_drop   = nn.Dropout(dropout)
        self.proj_out   = nn.Linear(hidden_dim, n_classes * rank)

        # Learnable output gene embeddings (random init — STRING_GNN ordering ≠ label ordering)
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes, rank))
        nn.init.normal_(self.out_gene_emb, std=0.02)

        # Weight init
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, pert_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pert_emb: [B, 256] final STRING_GNN embedding after backbone adapter
        Returns:
            logits: [B, 3, 6640]
        """
        x = self.proj_in(self.input_norm(pert_emb))  # [B, 512]
        for blk in self.blocks:
            x = blk(x)
        x = self.proj_out(self.out_drop(self.out_norm(x)))  # [B, 3*rank]
        pert_proj = x.view(-1, self.n_classes, self.rank)    # [B, 3, rank]
        # Bilinear: [B, 3, rank] × [G, rank].T → [B, 3, G]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)
        return logits


class PartialBackboneAdapter(nn.Module):
    """
    Wraps trainable STRING_GNN layers mps.6, mps.7, post_mp.

    Takes full intermediate embedding matrix [N_nodes, 256] from frozen mps.0-5
    and applies trainable layers using the full graph structure.
    This is identical to the approach used in node2-1-1-1-2-1-1-1-1-1-1-1-1 (F1=0.5182).
    """

    def __init__(self, layer6, layer7, post_mp):
        super().__init__()
        self.layer6  = layer6
        self.layer7  = layer7
        self.post_mp = post_mp

    def forward(
        self,
        x: torch.Tensor,            # [N_nodes, 256] - after mps.0-5
        edge_index: torch.Tensor,   # [2, E]
        edge_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply trainable layers with full graph convolution."""
        # mps.6: residual GCN
        x = x + self.layer6.dropout(self.layer6.act(
            self.layer6.norm(self.layer6.conv(x, edge_index, edge_weight))
        ))
        # mps.7: residual GCN
        x = x + self.layer7.dropout(self.layer7.act(
            self.layer7.norm(self.layer7.conv(x, edge_index, edge_weight))
        ))
        # post_mp: output projection
        x = self.post_mp(x)
        return x  # [N_nodes, 256]


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbModel(pl.LightningModule):
    """
    Partial STRING_GNN fine-tuning + rank-512 deep bilinear head for DEG prediction.

    Forward pass:
        1. Move frozen_embeddings to GPU (all 18870 nodes)
        2. Apply trainable backbone adapter (mps.6+mps.7+post_mp, full graph)
        3. Extract per-sample embeddings using node_idx, handle OOV
        4. Feed into deep bilinear head → [B, 3, 6640] logits
    """

    def __init__(
        self,
        args: argparse.Namespace,
        n_gpus: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.args   = args
        self.n_gpus = n_gpus

        # SWA pool (populated via periodic checkpoints at test time, not training hooks)
        self._swa_pool: List[dict] = []

        # Storage for val/test
        self._val_preds:    List[torch.Tensor] = []
        self._val_labels:   List[torch.Tensor] = []
        self._test_preds:   List[torch.Tensor] = []
        self._test_ids:     List[str]           = []
        self._test_symbols: List[str]           = []

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize model components after DDP setup."""
        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)

        # ─── Trainable backbone tail: mps.6 + mps.7 + post_mp ────────────────
        self.backbone_adapter = PartialBackboneAdapter(
            layer6  = backbone.mps[6],
            layer7  = backbone.mps[7],
            post_mp = backbone.post_mp,
        )
        del backbone
        torch.cuda.empty_cache()

        # Learnable OOV embedding
        self.oov_embedding = nn.Parameter(torch.zeros(GNN_DIM))
        nn.init.normal_(self.oov_embedding, std=0.02)

        # ─── Bilinear prediction head ─────────────────────────────────────────
        self.head = GNNBilinearHead(
            gnn_dim    = GNN_DIM,
            hidden_dim = 512,
            rank       = self.args.bilinear_rank,
            n_genes    = N_GENES_OUT,
            n_classes  = N_CLASSES,
            dropout    = self.args.head_dropout,
            n_layers   = self.args.n_resblocks,
        )

        # Cast trainable parameters to float32 for stable optimization
        for k, v in self.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()

        # ─── Focal loss ───────────────────────────────────────────────────────
        class_weights = torch.tensor(self.args.class_weights, dtype=torch.float32)
        self.register_buffer("class_weights_buf", class_weights)

    def _focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Focal loss with class weighting and label smoothing."""
        gamma          = self.args.focal_gamma
        label_smoothing = self.args.label_smoothing
        class_weights  = self.class_weights_buf.to(logits.device)

        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        labels_flat = labels.reshape(-1)                        # [B*G]

        if label_smoothing > 0.0:
            one_hot       = F.one_hot(labels_flat, num_classes=C).float()
            smooth_targets = (1.0 - label_smoothing) * one_hot + label_smoothing / C
            log_probs     = F.log_softmax(logits_flat, dim=1)
            w_expanded    = class_weights.unsqueeze(0)  # [1, C]
            ce_loss       = -(smooth_targets * w_expanded * log_probs).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(
                logits_flat, labels_flat,
                weight=class_weights, reduction="none"
            )

        with torch.no_grad():
            probs = F.softmax(logits_flat, dim=1)
            pt    = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt).pow(gamma)
        return (focal_weight * ce_loss).mean()

    def _get_node_embeddings(self) -> torch.Tensor:
        """
        Apply trainable backbone adapter to frozen intermediate embeddings.
        Uses full graph structure. Returns [N_nodes, 256] final embeddings.
        """
        dm = self.trainer.datamodule

        frozen_embs = dm.frozen_embeddings_tensor.to(
            device=self.device, dtype=torch.float32
        )
        edge_index = dm.edge_index.to(device=self.device)
        edge_weight = None
        if dm.edge_weight is not None:
            edge_weight = dm.edge_weight.to(device=self.device, dtype=torch.float32)

        # Apply trainable tail (mps.6 + mps.7 + post_mp)
        all_embs = self.backbone_adapter(frozen_embs, edge_index, edge_weight)
        return all_embs  # [N_nodes, 256]

    def _lookup_pert_embeddings(
        self,
        all_embs: torch.Tensor,   # [N_nodes, 256]
        node_idx: torch.Tensor,   # [B] int64, -1 for OOV
        in_vocab: torch.Tensor,   # [B] bool
        emb_fallback: torch.Tensor,  # [B, 256] pre-frozen embeddings as fallback
    ) -> torch.Tensor:
        """
        Extract per-sample perturbation embeddings from the full node embedding matrix.
        OOV genes use the learnable oov_embedding.

        Args:
            all_embs: [N_nodes, 256] full node embedding matrix after backbone adapter
            node_idx: [B] node indices (-1 for OOV)
            in_vocab: [B] bool mask (True = in vocab)
            emb_fallback: [B, 256] pre-frozen per-sample embeddings (fallback for OOV)
        Returns:
            pert_emb: [B, 256]
        """
        B = node_idx.shape[0]
        # Avoid boolean-indexing assignment on bf16 tensors (PyTorch CUDA bug).
        # Use torch.where which is dtype-safe and differentiable.
        safe_idx = node_idx.clamp(min=0).long()  # replace -1 (OOV) with 0 for gather
        all_embs_batch = all_embs[safe_idx]      # [B, 256] — OOV rows have garbage values

        # Broadcast OOV embedding to batch
        oov_emb = self.oov_embedding.to(dtype=all_embs.dtype).unsqueeze(0).expand(B, -1)

        # Select: in-vocab → from all_embs; OOV → learnable oov_embedding
        in_vocab_expanded = in_vocab.unsqueeze(-1)  # [B, 1] → broadcasts to [B, 256]
        pert_emb = torch.where(in_vocab_expanded, all_embs_batch, oov_emb)  # [B, 256]

        return pert_emb

    def forward(
        self,
        embedding:  torch.Tensor,  # [B, 256] pre-frozen emb (fallback)
        node_idx:   torch.Tensor,  # [B] int64
        in_vocab:   torch.Tensor,  # [B] bool
    ) -> torch.Tensor:
        """Full forward pass."""
        all_embs = self._get_node_embeddings()  # [N_nodes, 256]
        pert_emb = self._lookup_pert_embeddings(all_embs, node_idx, in_vocab, embedding)
        logits   = self.head(pert_emb)
        return logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["embedding"].to(self.device),
            batch["node_idx"].to(self.device),
            batch["in_vocab"].to(self.device),
        )
        loss = self._focal_loss(logits, batch["label"].to(self.device))
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(
            batch["embedding"].to(self.device),
            batch["node_idx"].to(self.device),
            batch["in_vocab"].to(self.device),
        )
        labels = batch["label"].to(self.device)
        loss   = self._focal_loss(logits, labels)

        probs = torch.softmax(logits, dim=1)  # [B, 3, G]
        self._val_preds.append(probs.detach().cpu())
        self._val_labels.append(labels.detach().cpu())

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        if dist.is_available() and dist.is_initialized():
            all_preds  = self.all_gather(torch.cat(self._val_preds,  dim=0).to(self.device))
            all_labels = self.all_gather(torch.cat(self._val_labels, dim=0).to(self.device))
            all_preds  = all_preds.view(-1, N_CLASSES, N_GENES_OUT)
            all_labels = all_labels.view(-1, N_GENES_OUT)
        else:
            all_preds  = torch.cat(self._val_preds,  dim=0)
            all_labels = torch.cat(self._val_labels, dim=0)

        self._val_preds  = []
        self._val_labels = []

        # All ranks compute the same f1 (after gathering, all ranks have the full dataset)
        n_val    = len(self.trainer.datamodule.val_ds)
        pred_np  = all_preds.float().cpu().numpy()[:n_val]
        label_np = all_labels.cpu().numpy()[:n_val]

        f1 = compute_per_gene_f1(pred_np, label_np)
        # Log on all ranks with the same value (all ranks computed from gathered tensors).
        # sync_dist=True silences Lightning's DDP logging warning and is safe here.
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        if self.trainer.is_global_zero:
            print(f"\n[Epoch {self.current_epoch}] val_f1={f1:.4f}")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(
            batch["embedding"].to(self.device),
            batch["node_idx"].to(self.device),
            batch["in_vocab"].to(self.device),
        )
        probs = torch.softmax(logits, dim=1)  # [B, 3, G]
        self._test_preds.append(probs.detach().cpu())
        self._test_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        local_preds = torch.cat(self._test_preds, dim=0).to(self.device)

        if dist.is_available() and dist.is_initialized():
            all_preds = self.all_gather(local_preds)  # [W, N_local, 3, G]
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES_OUT)
            # Gather string IDs and symbols from all ranks (tensors cannot carry strings)
            gathered_ids     = [None] * dist.get_world_size()
            gathered_symbols = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_ids,     self._test_ids)
            dist.all_gather_object(gathered_symbols, self._test_symbols)
            all_ids     = [p for sub in gathered_ids     for p in sub]
            all_symbols = [p for sub in gathered_symbols for p in sub]
        else:
            all_preds   = local_preds
            all_ids     = self._test_ids
            all_symbols = self._test_symbols

        self._test_preds   = []

        if self.trainer.is_global_zero:
            pred_np = all_preds.float().cpu().numpy()
            n_test  = len(self.trainer.datamodule.test_ds)
            pred_np = pred_np[:n_test]

            ids     = all_ids[:n_test]
            symbols = all_symbols[:n_test]

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "test_predictions.tsv"

            rows = []
            for i in range(len(ids)):
                pred_3xG = pred_np[i]  # [3, G]
                rows.append({
                    "idx":        ids[i],
                    "input":      symbols[i] if i < len(symbols) else "",
                    "prediction": json.dumps(pred_3xG.tolist()),
                })
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            print(f"\nTest predictions saved to {out_path} ({len(rows)} rows)")

        self._test_ids     = []
        self._test_symbols = []

    def configure_optimizers(self):
        """
        Three-group MuonWithAuxAdam:
          - Group 1: Muon for 2D weight matrices in ResidualBlocks (lr=0.005)
          - Group 2: AdamW for other head params (lr=5e-4)
          - Group 3: AdamW for backbone tail mps.6+mps.7+post_mp (lr=1e-5)
        SGDR warm restarts: T_0=T_0_steps, T_mult=2
        """
        try:
            from muon import MuonWithAuxAdam
        except ImportError:
            raise ImportError("Install muon: pip install git+https://github.com/KellerJordan/Muon")

        # Identify backbone parameter IDs
        backbone_param_ids = set(id(p) for p in self.backbone_adapter.parameters())

        muon_params          = []
        adamw_head_params    = []
        backbone_params_list = list(self.backbone_adapter.parameters())

        # Add oov_embedding to head group
        adamw_head_params.append(self.oov_embedding)

        # Classify head parameters
        for name, param in self.head.named_parameters():
            # Muon for 2D weight matrices in ResidualBlocks (fc1, fc2 weights)
            if param.ndim >= 2 and "blocks" in name and "weight" in name:
                muon_params.append(param)
            else:
                adamw_head_params.append(param)

        param_groups = [
            # Group 1: Muon for ResBlock 2D matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=self.args.muon_lr,
                weight_decay=self.args.weight_decay,
                momentum=0.95,
            ),
            # Group 2: AdamW for other head params
            dict(
                params=adamw_head_params,
                use_muon=False,
                lr=self.args.head_lr,
                betas=(0.9, 0.999),
                weight_decay=self.args.weight_decay,
            ),
            # Group 3: AdamW for backbone (lower LR, no weight decay)
            dict(
                params=backbone_params_list,
                use_muon=False,
                lr=self.args.backbone_lr,
                betas=(0.9, 0.999),
                weight_decay=0.0,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # SGDR warm restarts
        # steps_per_epoch = ceil(train_size / (n_gpus * micro_batch_size * accum))
        train_size      = len(self.trainer.datamodule.train_ds)
        steps_per_epoch = max(1, train_size // (
            self.n_gpus
            * self.args.micro_batch_size
            * self.trainer.accumulate_grad_batches
        ))
        T_0_steps = self.args.sgdr_t0_epochs * steps_per_epoch

        print(f"[LR] steps_per_epoch={steps_per_epoch}, T_0_steps={T_0_steps}")

        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, T_0_steps),
            T_mult=self.args.sgdr_t_mult,
            eta_min=1e-6,
        )

        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
                "frequency": 1,
            },
        }

    # ─── Efficient checkpoint ─────────────────────────────────────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Save only trainable parameters (exclude frozen buffers)
        trainable_keys = {
            prefix + name
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        sd = {k: v for k, v in full_sd.items() if k in trainable_keys}

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Checkpoint: {trainable}/{total} trainable params "
            f"({100.0*trainable/total:.1f}%)"
        )
        return sd

    def load_state_dict(self, state_dict, strict=True):
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        loaded = sum(1 for k in state_dict if k in trainable_keys)
        self.print(f"Loading: {loaded} trainable parameter tensors")
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Node 3-1-1-1-3: Partial STRING_GNN FT + Bilinear + Muon + SGDR + SWA"
    )

    # Data
    parser.add_argument("--data-dir", type=str, default="data")

    # Batch
    parser.add_argument("--micro-batch-size",  type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=64,
                        help="Multiple of micro_batch_size * 8")

    # Architecture
    parser.add_argument("--bilinear-rank", type=int,   default=512)
    parser.add_argument("--n-resblocks",   type=int,   default=6)
    parser.add_argument("--head-dropout",  type=float, default=0.45)

    # Optimizer
    parser.add_argument("--muon-lr",      type=float, default=0.005,
                        help="Muon LR for ResidualBlock 2D matrices")
    parser.add_argument("--head-lr",      type=float, default=5e-4,
                        help="AdamW LR for other head params")
    parser.add_argument("--backbone-lr",  type=float, default=1e-5,
                        help="AdamW LR for backbone tail")
    parser.add_argument("--weight-decay", type=float, default=3e-3)

    # Loss
    parser.add_argument("--focal-gamma",     type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weights",   nargs=3, type=float,
                        default=[2.0, 0.5, 4.0])

    # SGDR
    parser.add_argument("--sgdr-t0-epochs", type=int, default=20,
                        help="SGDR T_0 in epochs")
    parser.add_argument("--sgdr-t-mult",    type=int, default=2)

    # Training
    parser.add_argument("--max-epochs",          type=int,   default=350)
    parser.add_argument("--patience",            type=int,   default=150,
                        help="EarlyStopping patience for SGDR cycles")
    parser.add_argument("--val-check-interval",  type=float, default=1.0)

    # SWA
    parser.add_argument("--swa-threshold",   type=float, default=0.497,
                        help="Minimum val_f1 for SWA pool inclusion")
    parser.add_argument("--swa-top-k",       type=int,   default=15)
    parser.add_argument("--swa-temperature", type=float, default=3.0)
    parser.add_argument("--swa-every-n-epochs", type=int, default=3,
                        help="Save periodic checkpoint every N epochs for SWA")

    # Debug
    parser.add_argument("--debug-max-step",  type=int,  default=None)
    parser.add_argument("--fast-dev-run",    action="store_true")
    parser.add_argument("--num-workers",     type=int,  default=4)

    return parser.parse_args()


def main():
    pl.seed_everything(0)
    args = parse_args()

    # Resolve relative data_dir to absolute path
    args.data_dir = str(Path(__file__).parent.parent.parent / args.data_dir)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(1, n_gpus)

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # ─── Output ─────────────────────────────────────────────────────────────
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── DataModule ─────────────────────────────────────────────────────────
    datamodule = PerturbDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # ─── Model ──────────────────────────────────────────────────────────────
    model = PerturbModel(args=args, n_gpus=n_gpus)

    # ─── Callbacks ──────────────────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-epoch={epoch:04d}-val_f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    # Periodic checkpoints for SWA pool (every N epochs from epoch 10 onward)
    periodic_checkpoint = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints" / "periodic"),
        filename="periodic-epoch={epoch:04d}-val_f1={val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=-1,   # Keep all periodic checkpoints
        every_n_epochs=args.swa_every_n_epochs,
        auto_insert_metric_name=False,
    )

    early_stop = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.patience,
        verbose=True,
    )

    lr_monitor   = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(refresh_rate=50)

    # ─── Loggers ────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # ─── Debug ──────────────────────────────────────────────────────────────
    fast_dev_run        = args.fast_dev_run
    limit_train_batches = 1.0
    limit_val_batches   = 1.0
    limit_test_batches  = 1.0
    max_steps           = -1

    if args.debug_max_step is not None:
        limit_train_batches = args.debug_max_step
        limit_val_batches   = args.debug_max_step
        limit_test_batches  = args.debug_max_step
        max_steps           = args.debug_max_step

    # ─── Trainer ────────────────────────────────────────────────────────────
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
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[
            checkpoint_callback,
            periodic_checkpoint,
            early_stop,
            lr_monitor,
            progress_bar,
        ],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=1 if fast_dev_run else False,
    )

    # ─── Training ───────────────────────────────────────────────────────────
    trainer.fit(model, datamodule=datamodule)

    # ─── Post-hoc quality-filtered SWA ──────────────────────────────────────
    do_swa = (not fast_dev_run and args.debug_max_step is None)
    swa_applied = False

    if do_swa:
        # All ranks participate so no broadcast is needed (checkpoints on shared filesystem).
        # This ensures all ranks have identical SWA weights before trainer.test() is called.
        periodic_dir = output_dir / "checkpoints" / "periodic"
        pool = []

        if periodic_dir.exists():
            for ckpt_path in sorted(periodic_dir.glob("*.ckpt")):
                # Handle both "val_f1=0.4487" and "val_f1=val_f1=0.4487" (PL double-prefix)
                matches = re.findall(r'val_f1=(\d+\.\d+)', ckpt_path.name)
                if matches:
                    val_f1 = float(matches[-1])  # take last match to handle double prefix
                    if val_f1 >= args.swa_threshold:
                        pool.append({"val_f1": val_f1, "path": str(ckpt_path)})

        pool.sort(key=lambda x: x["val_f1"], reverse=True)
        pool = pool[:args.swa_top_k]

        if len(pool) >= 2:
            if trainer.is_global_zero:
                print(f"\nSWA: {len(pool)} qualifying checkpoints, "
                      f"val_f1 range [{pool[-1]['val_f1']:.4f}, {pool[0]['val_f1']:.4f}]")

            f1_vals = torch.tensor([e["val_f1"] for e in pool])
            weights = torch.softmax(f1_vals * args.swa_temperature, dim=0)
            if trainer.is_global_zero:
                print("SWA weights:", [f"{w:.4f}" for w in weights.tolist()])

            avg_state = None
            for i, entry in enumerate(pool):
                ckpt = torch.load(entry["path"], map_location="cpu", weights_only=False)
                sd   = ckpt.get("state_dict", ckpt)
                w    = weights[i].item()
                if avg_state is None:
                    avg_state = {k: v.float() * w for k, v in sd.items()}
                else:
                    for k in avg_state:
                        if k in sd:
                            avg_state[k] += sd[k].float() * w

            if avg_state is not None:
                if trainer.is_global_zero:
                    swa_ckpt_path = output_dir / "checkpoints" / "swa_averaged.ckpt"
                    torch.save({"state_dict": avg_state}, swa_ckpt_path)
                    print(f"SWA checkpoint saved: {swa_ckpt_path}")

                # All ranks load SWA weights for consistent test inference across GPUs
                model.load_state_dict(avg_state, strict=False)
                swa_applied = True
                if trainer.is_global_zero:
                    print("SWA weights loaded for testing")

        else:
            if trainer.is_global_zero:
                print(f"SWA pool has {len(pool)} checkpoints (< 2) — skipping SWA, using best checkpoint")

    # ─── Testing ────────────────────────────────────────────────────────────
    if fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    elif swa_applied:
        # Already loaded SWA weights — test without loading from checkpoint
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print(f"\nTest results: {test_results}")


if __name__ == "__main__":
    main()
