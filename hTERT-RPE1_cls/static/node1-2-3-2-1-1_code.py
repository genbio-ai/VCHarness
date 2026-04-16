"""
Node 1-2-3-2-1-1: Partial STRING_GNN Fine-Tuning (mps.7+post_mp) + MuonWithAuxAdam + Reverted Regularization

Architecture:
  - STRING_GNN partial fine-tuning: freeze emb+mps.0-6, fine-tune mps.7+post_mp (~67K backbone params)
    at very small LR (1e-5 via AdamW). The full GNN forward pass is run once per step on all
    18,870 nodes to produce updated per-node embeddings.
  - Deep residual MLP (6 blocks, hidden_dim=512): transforms per-sample embedding to perturbation repr
  - Bilinear interaction head (rank=512):
      [B, 3, 512] x [6640, 512]^T -> [B, 3, 6640]
  - MuonWithAuxAdam: Muon lr=0.005 for ResBlock 2D weight matrices, AdamW for rest
  - Class-weighted focal loss [down=1.5, neutral=0.8, up=3.0] + gamma=2.0
  - No label smoothing (reverted - found to hurt minority-class F1 in parent)
  - Cosine LR schedule (total_steps=6600, warmup=50) + patience=60

Key improvements over parent (node1-2-3-2-1, F1=0.4984):
  1. Partial STRING_GNN backbone unfreezing (mps.7+post_mp at backbone_lr=1e-5) to break the
     frozen backbone ceiling (~0.499-0.500) -- primary high-leverage intervention.
     Evidence: tree best nodes using partial FT achieve 0.5047-0.5182 vs ~0.499 frozen ceiling.
  2. MuonWithAuxAdam optimizer for ResBlock hidden matrices (Muon lr=0.005). This optimizer
     has proven highly effective across the MCTS tree (nodes node1-1-2-1-1-1, node2-1-2-2-2-1,
     node1-2-2-2, etc.), typically adding +0.005-0.015 F1 over plain AdamW.
  3. Remove label smoothing (reverted to 0.0). Parent feedback explicitly noted that epsilon=0.05
     softens minority-class (down/up) predictions in this 88.9% neutral dataset, reducing
     argmax F1 for rare classes. Parent achieved val/train ratio 2.25x but F1 was lower than
     its own parent's 3.03x ratio -- confirming over-regularization from label smoothing.
  4. Revert dropout 0.25 -> 0.20 (parent's sweet spot). Parent node1-2-3-2 with dropout=0.20
     achieved 0.4996; this node with 0.25 got 0.4984. The intermediate val/train ratio of 2.25x
     may indicate slightly too much regularization suppressing model capacity.
  5. Revert weight_decay 2e-3 -> 1e-3 for MLP body (parent's proven sweet spot).
  6. Patience reduced 80 -> 60. Parent found that patience=80 added 80 epochs after best
     with no improvement -- patience=60 is more efficient.

How partial backbone FT works (efficient transductive approach):
  - Run FULL STRING_GNN forward once per training EPOCH (not per step) with gradient
    enabled only for mps.7+post_mp
  - Cache the resulting [18870, 256] node embeddings for lookup during batch processing
  - At inference (val/test), similarly refresh embeddings at epoch start
  - This gives gradients through mps.7+post_mp without running GNN every step

Expected: 0.503-0.508 F1 (breaking the 0.499-0.500 frozen backbone ceiling)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # required for deterministic=True with CUDA >= 10.2

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

N_GENES_OUT = 6640
N_CLASSES = 3
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Exact per-gene macro F1 matching calc_metric.py logic."""
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

    Each sample provides its pert_id for embedding lookup in the shared GNN node table.
    The actual embeddings are looked up from the module's cached node embeddings at
    inference/training time (after the GNN forward pass).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        node_name_to_idx: Dict[str, int],
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.has_labels = has_labels

        # Pre-compute which samples are in STRING_GNN vocab
        self.in_vocab = torch.tensor(
            [pid in node_name_to_idx for pid in self.pert_ids],
            dtype=torch.bool,
        )
        # Pre-compute node indices (use 0 as placeholder for OOV)
        self.node_indices = torch.tensor(
            [node_name_to_idx.get(pid, 0) for pid in self.pert_ids],
            dtype=torch.long,
        )

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
            "node_idx": self.node_indices[idx],   # STRING_GNN node index (for embedding lookup)
            "in_vocab": self.in_vocab[idx],        # bool: True if gene is in STRING_GNN vocab
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule for partial STRING_GNN fine-tuning approach."""

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

        # Load STRING_GNN graph topology and node names (no model forward needed here)
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.node_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(node_names)}

        # Store graph topology for use by the model's GNN forward pass
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", weights_only=False)
        self.edge_index = graph["edge_index"].cpu()
        self.edge_weight = graph["edge_weight"].cpu() if graph.get("edge_weight") is not None else None

        # Load all splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        print(f"[DataModule] STRING_GNN coverage: "
              f"{sum(p in self.node_name_to_idx for p in dfs['train']['pert_id'])} / "
              f"{len(dfs['train'])} train genes in STRING_GNN vocabulary")

        self.train_ds = PerturbationDataset(dfs["train"], self.node_name_to_idx, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   self.node_name_to_idx, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  self.node_name_to_idx, True)

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


# ─── Model ────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout + skip."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.20):
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


class GNNBilinearHead(nn.Module):
    """Prediction head using STRING_GNN per-sample embeddings as input.

    Architecture:
      - OOV embedding for genes not in STRING_GNN vocabulary
      - Input normalization (LayerNorm)
      - 6 residual layers (dropout=0.20 reverted to parent's sweet spot)
      - rank=512 bilinear interaction (proven +0.0057 from grandparent)
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.20,          # REVERTED to parent's sweet spot
        n_residual_layers: int = 6,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.rank = rank

        # OOV embedding for genes not in STRING_GNN (fallback)
        self.oov_embedding = nn.Embedding(1, gnn_dim)

        # Input normalization
        self.input_norm = nn.LayerNorm(gnn_dim)

        # Projection: gnn_dim -> hidden_dim
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)

        # Deep residual MLP (6 layers, dropout REVERTED to 0.20)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim -> n_classes * rank
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank)

        # Output gene embeddings: learnable [n_genes_out, rank]
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, rank))

        # Head dropout (reverted to 0.20)
        self.head_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.oov_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)
        nn.init.normal_(self.out_gene_emb, std=0.02)

    def forward(
        self,
        gnn_emb: torch.Tensor,   # [B, 256] GNN embeddings for batch samples
        in_vocab: torch.Tensor,  # [B] bool mask - True if gene is in STRING_GNN vocab
    ) -> torch.Tensor:
        """
        Args:
            gnn_emb:  [B, gnn_dim] - per-sample embeddings from GNN forward pass
            in_vocab: [B] bool - True if gene is in STRING_GNN vocabulary
        Returns:
            logits: [B, 3, 6640]
        """
        B = gnn_emb.shape[0]

        # Replace OOV embeddings with learned fallback
        oov_emb = self.oov_embedding(torch.zeros(B, dtype=torch.long, device=gnn_emb.device))
        in_vocab_f = in_vocab.unsqueeze(1).float()
        x = gnn_emb * in_vocab_f + oov_emb * (1.0 - in_vocab_f)  # [B, gnn_dim]

        # Input normalization
        x = self.input_norm(x)

        # Projection to hidden dim
        x = self.proj_in(x)   # [B, hidden_dim]

        # Deep residual MLP
        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)   # [B, hidden_dim]

        # Bilinear head
        x = self.head_dropout(x)
        pert_proj = self.proj_bilinear(x)                        # [B, n_classes * rank]
        pert_proj = pert_proj.view(B, self.n_classes, self.rank) # [B, 3, rank]

        # Bilinear interaction: [B, 3, rank] x [rank, n_genes_out] -> [B, 3, n_genes_out]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, 6640]
        return logits


# ─── Focal Loss ───────────────────────────────────────────────────────────────

def class_weighted_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Focal loss with mild class weights and NO label smoothing.

    Label smoothing is explicitly disabled (reverted from parent's 0.05).
    Parent feedback confirmed epsilon=0.05 hurts minority-class argmax F1
    on this heavily imbalanced (88.9% neutral) dataset.
    """
    B, C, G = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
    labels_flat = labels.reshape(-1)                        # [B*G]

    ce_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        weight=class_weights,
        reduction="none",
        label_smoothing=0.0,  # NO label smoothing (reverted)
    )  # [B*G]

    with torch.no_grad():
        probs = F.softmax(logits_flat, dim=1)
        pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt).pow(gamma)

    loss = (focal_weight * ce_loss).mean()
    return loss


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(
    local_preds: torch.Tensor,
    local_labels: torch.Tensor,
    device: torch.device,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    real_preds  = torch.cat([g_preds[i][: all_sizes[i].item()].cpu()  for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][: all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """LightningModule for gene-perturbation DEG prediction (Node 1-2-3-2-1-1).

    Key changes from parent (node1-2-3-2-1, F1=0.4984):
      1. Partial STRING_GNN backbone unfreezing (mps.7+post_mp at backbone_lr=1e-5)
      2. MuonWithAuxAdam optimizer (Muon lr=0.005 for ResBlock 2D matrices)
      3. No label smoothing (reverted from parent's 0.05)
      4. Dropout reverted 0.25 -> 0.20
      5. Weight decay reverted 2e-3 -> 1e-3 for MLP body
      6. Patience reduced 80 -> 60

    GNN strategy: The full STRING_GNN forward pass is run ONCE per training step
    (not per epoch). At each training_step, the full [18870, 256] node embedding
    matrix is computed with gradients through mps.7+post_mp. Per-sample embeddings
    are then indexed from this matrix for the batch. This ensures proper gradient
    flow through the trainable backbone layers.

    During validation/test, the forward pass runs in eval mode (no gradients).
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_residual_layers: int = 6,
        dropout: float = 0.20,             # REVERTED from parent's 0.25
        lr: float = 3e-4,                  # AdamW head LR
        muon_lr: float = 5e-3,             # Muon LR for ResBlock matrices
        backbone_lr: float = 1e-5,         # Very small LR for trainable GNN tail
        lr_emb_multiplier: float = 0.5,    # embedding LR = lr * multiplier = 1.5e-4
        weight_decay: float = 1e-3,        # REVERTED from parent's 2e-3
        focal_gamma: float = 2.0,
        class_weight_down: float = 1.5,
        class_weight_neutral: float = 0.8,
        class_weight_up: float = 3.0,
        warmup_steps: int = 50,
        total_steps: int = 6600,
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

        # Load STRING_GNN model for partial fine-tuning
        self.gnn_model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)

        # Freeze all GNN parameters first
        for p in self.gnn_model.parameters():
            p.requires_grad = False

        # Selectively unfreeze mps.7 and post_mp
        for name, param in self.gnn_model.named_parameters():
            if name.startswith("mps.7.") or name.startswith("post_mp."):
                param.requires_grad = True
                param.data = param.data.float()  # ensure float32

        # Count trainable backbone params
        backbone_trainable = sum(p.numel() for p in self.gnn_model.parameters() if p.requires_grad)
        print(f"[Model] Trainable backbone params (mps.7+post_mp): {backbone_trainable:,}")

        # Prediction head
        self.head = GNNBilinearHead(
            gnn_dim=hp.gnn_dim,
            hidden_dim=hp.hidden_dim,
            rank=hp.rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
        )

        # Ensure float32 for all trainable head parameters
        for p in self.head.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Class weights tensor
        self._class_weights_cpu = torch.tensor(
            [hp.class_weight_down, hp.class_weight_neutral, hp.class_weight_up],
            dtype=torch.float32,
        )

        # Store graph topology from datamodule
        dm = self.trainer.datamodule if self.trainer is not None else None
        if dm is not None and hasattr(dm, "edge_index"):
            self._edge_index_cpu = dm.edge_index  # [2, E] CPU tensor
            self._edge_weight_cpu = dm.edge_weight  # [E] CPU tensor or None
        else:
            graph = torch.load(STRING_GNN_DIR / "graph_data.pt", weights_only=False)
            self._edge_index_cpu = graph["edge_index"].cpu()
            ew = graph.get("edge_weight")
            self._edge_weight_cpu = ew.cpu() if ew is not None else None

    def _run_gnn_forward(self) -> torch.Tensor:
        """Run full STRING_GNN forward pass to get [N_nodes, 256] node embeddings.

        For training_step: runs with gradient enabled for mps.7+post_mp.
        For val/test steps: runs in eval mode without gradients.

        Returns:
            node_emb: [N_nodes, 256] float32 tensor on self.device
        """
        device = self.device
        edge_index = self._edge_index_cpu.to(device)
        edge_weight = self._edge_weight_cpu.to(device) if self._edge_weight_cpu is not None else None

        # Run GNN model with gradient only for trainable params
        outputs = self.gnn_model(
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        node_emb = outputs.last_hidden_state  # [N_nodes, 256]
        return node_emb.float()

    def forward(
        self,
        gnn_emb: torch.Tensor,  # [B, 256]
        in_vocab: torch.Tensor, # [B] bool
    ) -> torch.Tensor:
        return self.head(gnn_emb, in_vocab)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cw = self._class_weights_cpu.to(logits.device)
        return class_weighted_focal_loss(
            logits, labels,
            gamma=self.hparams.focal_gamma,
            class_weights=cw,
        )

    def training_step(self, batch, batch_idx):
        # Run full GNN forward pass with gradient tracking for trainable backbone
        node_emb = self._run_gnn_forward()  # [N_nodes, 256], gradients through mps.7+post_mp

        # Index per-sample embeddings from full node embedding matrix
        node_idx = batch["node_idx"].to(self.device)   # [B] node indices
        in_vocab  = batch["in_vocab"].to(self.device)  # [B] bool
        gnn_emb = node_emb[node_idx]                   # [B, 256]

        logits = self(gnn_emb, in_vocab)
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Run full GNN in eval mode (no gradient needed)
        with torch.no_grad():
            node_emb = self._run_gnn_forward()  # [N_nodes, 256]

        node_idx = batch["node_idx"].to(self.device)
        in_vocab  = batch["in_vocab"].to(self.device)
        gnn_emb = node_emb[node_idx]  # [B, 256]

        logits = self(gnn_emb, in_vocab)
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
        with torch.no_grad():
            node_emb = self._run_gnn_forward()

        node_idx = batch["node_idx"].to(self.device)
        in_vocab  = batch["in_vocab"].to(self.device)
        gnn_emb = node_emb[node_idx]

        logits = self(gnn_emb, in_vocab)
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
            self.print(f"[Node1-2-3-2-1-1] Saved test predictions -> {pred_path} ({len(seen_ids)} unique samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-2-3-2-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams
        lr_emb = hp.lr * hp.lr_emb_multiplier  # 3e-4 * 0.5 = 1.5e-4

        # ── Collect parameters by group ──────────────────────────────────────
        # Group 1 (Muon): 2D weight matrices in ResBlocks
        # Group 2 (AdamW, full LR): non-matrix head params + proj_in + proj_bilinear + norms/biases
        # Group 3 (AdamW, lower LR): embedding-like params (out_gene_emb, oov_embedding)
        # Group 4 (AdamW, very small LR): backbone trainable params (mps.7, post_mp)

        muon_params: List[torch.Tensor] = []
        adamw_head_params: List[torch.Tensor] = []
        adamw_emb_params: List[torch.Tensor] = []
        adamw_backbone_params: List[torch.Tensor] = []

        muon_ids: set = set()
        emb_names = {"head.out_gene_emb", "head.oov_embedding.weight"}

        # Collect ResBlock 2D matrices -> Muon group
        for name, param in self.head.res_blocks.named_parameters():
            if param.requires_grad and param.ndim >= 2:
                muon_params.append(param)
                muon_ids.add(id(param))

        # Separate remaining head params and backbone params
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in muon_ids:
                continue  # already in Muon group
            if name in emb_names:
                adamw_emb_params.append(param)
            elif name.startswith("gnn_model."):
                adamw_backbone_params.append(param)
            else:
                adamw_head_params.append(param)

        # ── Build MuonWithAuxAdam param groups ───────────────────────────────
        param_groups = []

        if muon_params:
            param_groups.append({
                "params": muon_params,
                "use_muon": True,
                "lr": hp.muon_lr,
                "weight_decay": hp.weight_decay,
                "momentum": 0.95,
            })

        if adamw_head_params:
            param_groups.append({
                "params": adamw_head_params,
                "use_muon": False,
                "lr": hp.lr,
                "weight_decay": hp.weight_decay,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            })

        if adamw_emb_params:
            param_groups.append({
                "params": adamw_emb_params,
                "use_muon": False,
                "lr": lr_emb,
                "weight_decay": 0.0,   # No L2 on embedding tables
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            })

        if adamw_backbone_params:
            param_groups.append({
                "params": adamw_backbone_params,
                "use_muon": False,
                "lr": hp.backbone_lr,  # Very small LR for backbone
                "weight_decay": 0.0,   # No weight decay on backbone
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            })

        optimizer = MuonWithAuxAdam(param_groups)

        # Cosine annealing with linear warmup
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
        trainable_sd = {}
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        for k, v in full_sd.items():
            if k in trainable_keys or k in buffer_keys:
                trainable_sd[k] = v
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)")
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 1-2-3-2-1-1 – Partial STRING_GNN FT (mps.7+post_mp) + MuonWithAuxAdam"
    )
    p.add_argument("--data-dir",              type=str,   default="data")
    p.add_argument("--gnn-dim",               type=int,   default=256)
    p.add_argument("--hidden-dim",            type=int,   default=512)
    p.add_argument("--rank",                  type=int,   default=512)
    p.add_argument("--n-residual-layers",     type=int,   default=6)
    p.add_argument("--dropout",               type=float, default=0.20)    # REVERTED to 0.20
    p.add_argument("--lr",                    type=float, default=3e-4)    # AdamW LR for non-Muon head params
    p.add_argument("--muon-lr",               type=float, default=5e-3)    # Muon LR for ResBlock 2D matrices
    p.add_argument("--backbone-lr",           type=float, default=1e-5)    # Very small LR for trainable GNN tail
    p.add_argument("--lr-emb-multiplier",     type=float, default=0.5)     # embedding LR = lr * 0.5 = 1.5e-4
    p.add_argument("--weight-decay",          type=float, default=1e-3)    # REVERTED to 1e-3
    p.add_argument("--focal-gamma",           type=float, default=2.0)
    p.add_argument("--class-weight-down",     type=float, default=1.5)
    p.add_argument("--class-weight-neutral",  type=float, default=0.8)
    p.add_argument("--class-weight-up",       type=float, default=3.0)
    p.add_argument("--warmup-steps",          type=int,   default=50)
    p.add_argument("--total-steps",           type=int,   default=6600)
    p.add_argument("--micro-batch-size",      type=int,   default=16)
    p.add_argument("--global-batch-size",     type=int,   default=64)
    p.add_argument("--max-epochs",            type=int,   default=300)
    p.add_argument("--patience",              type=int,   default=60)      # Reduced from parent's 80
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
    dm.setup()

    # Compute accumulation factor
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    total_steps = args.total_steps

    # LightningModule
    lit = PerturbationLitModule(
        gnn_dim=args.gnn_dim,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        n_residual_layers=args.n_residual_layers,
        dropout=args.dropout,
        lr=args.lr,
        muon_lr=args.muon_lr,
        backbone_lr=args.backbone_lr,
        lr_emb_multiplier=args.lr_emb_multiplier,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        class_weight_down=args.class_weight_down,
        class_weight_neutral=args.class_weight_neutral,
        class_weight_up=args.class_weight_up,
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
    es_cb = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
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
            f"Node 1-2-3-2-1-1 – Partial STRING_GNN FT (mps.7+post_mp) + MuonWithAuxAdam + Reverted Regularization\n"
            f"Test results from trainer: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
