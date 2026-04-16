"""
Node 1-2-2-1: Clean STRING_GNN + Deep Bilinear Head + Class-Weighted Focal Loss

Key improvements over parent node1-2-2 (F1=0.4664):

1. Remove InductiveConditioningModule (PRIMARY FIX):
   The module was proven net harmful across four experiments (node1-2-1, node1-2-1-1,
   node1-3-1, node1-2-2), consistently degrading performance. Returning to the clean
   frozen STRING_GNN + bilinear head architecture of node1-2 (F1=0.4912).

2. Remove label smoothing (SECONDARY FIX):
   Label smoothing (ε=0.05) improved calibration in node1-2-2 (val/train ratio 2.19x vs
   3.71x) but F1 dropped 0.0248 below parent. This is because label smoothing softens
   probability distributions, reducing the probability gap between "neutral" and
   "non-neutral" predictions, causing some argmax decisions to shift back toward neutral
   for the already-difficult minority classes. Removing it.

3. Class-weighted focal loss (PRIMARY INNOVATION):
   The label distribution is 88.9% neutral / 8.1% down / 3.0% up, a ~30x imbalance
   between neutral and up-regulated. Pure focal loss (gamma=2.0) down-weights easy
   neutral examples, but may not sufficiently up-weight the rare up-regulated class.
   Explicit per-class weights in the loss function directly target the imbalance:
     - Class 0 (down-regulated, -1): weight = 3.0 (moderate up-weight)
     - Class 1 (neutral, 0):         weight = 0.3 (strong down-weight)
     - Class 2 (up-regulated, +1):   weight = 7.0 (strong up-weight, rarest class)
   These weights are applied as multiplicative factors in the cross-entropy loss BEFORE
   focal weighting, so focal weighting and class weighting work synergistically.
   Theoretical motivation: the per-gene macro F1 averages across present classes. If
   the model is more biased toward correctly classifying the rare up-regulated (+1)
   examples, the improvement in F1 at those gene positions will outweigh any F1 drop
   at positions where the rare class is already predicted correctly.

4. Increased focal gamma from 2.0 to 3.0 (SECONDARY INNOVATION):
   With class weights now compensating for the imbalance arithmetically, increasing
   gamma from 2.0 to 3.0 provides additional dynamic down-weighting of easy neutral
   examples during training. Combined with class weights, this should substantially
   reduce the model's tendency to predict neutral for ambiguous cases.

5. Restore total_steps=6600 (LR SCHEDULE RESTORATION):
   node1-2's success (F1=0.4912) was built on a secondary improvement phase (epochs
   70-98) enabled by the cosine LR decaying at a moderate pace with total_steps=6600.
   node1-2-2's total_steps=4000 caused premature LR decay but also limited early-phase
   exploration. Restoring to 6600 ensures the LR profile matches the proven schedule
   that yielded the current best F1.

6. Restore patience=30 (EARLY STOPPING RESTORATION):
   node1-2 achieved its best val F1 at epoch 98 with patience=30, triggering early
   stopping at epoch 128. Returning to the proven setting.

Architecture (identical to node1-2 — proven baseline):
  - STRING_GNN (frozen): one-time forward pass → 256-dim PPI embeddings
  - OOV fallback embedding (learnable, shared)
  - Input normalization: LayerNorm(256)
  - Projection: Linear(256 → 512)
  - 6x ResidualBlock (expand=4, GELU, Dropout=0.2)
  - Output normalization + dropout
  - Bilinear head: Linear(512 → 3*256) → reshape [B, 3, 256]
    einsum("bcr,gr→bcg", ...) with learnable out_gene_emb [6640, 256]
  - Class-weighted focal loss (gamma=3.0, class_weights=[3.0, 0.3, 7.0])

Memory influences:
  - node1-2-2 feedback: InductiveConditioningModule is net harmful → remove entirely
  - node1-2-2 feedback: label smoothing hurts F1 despite improving calibration → remove
  - node1-2-2 feedback: focal gamma tuning + class weighting recommended
  - node1-2 success: clean frozen STRING_GNN architecture is the proven baseline
  - DATA_ABSTRACT.md: 88.9%/8.1%/3.0% class distribution justifies strong class weights
  - node1-3-1-1 memory: increasing rank/hidden_dim not beneficial → keep proven 512/256
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


# ─── STRING_GNN Embedding Extraction ─────────────────────────────────────────

def build_string_gnn_embeddings(
    model_dir: Path,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build pretrained STRING_GNN embeddings for all nodes.

    Runs one frozen forward pass to get [N_nodes, 256] embeddings.
    Returns (embeddings_np [N, 256], node_name_to_idx dict).
    """
    node_names = json.loads((model_dir / "node_names.json").read_text())
    graph = torch.load(model_dir / "graph_data.pt", weights_only=False)

    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    model = model.to(device)
    model.eval()

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    edge_index = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"].to(device) if graph.get("edge_weight") is not None else None

    with torch.no_grad():
        outputs = model(
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

    embeddings = outputs.last_hidden_state.float().cpu().numpy()  # [N_nodes, 256]
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

    del model
    torch.cuda.empty_cache()

    return embeddings, node_name_to_idx


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset using precomputed STRING_GNN embeddings."""

    def __init__(
        self,
        df: pd.DataFrame,
        gnn_embeddings: np.ndarray,          # [N_nodes, 256]
        node_name_to_idx: Dict[str, int],
        embed_dim: int = 256,
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.has_labels = has_labels

        # Build embeddings tensor for this dataset: [N_samples, 256]
        n_samples = len(df)
        embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)
        in_vocab = []
        for i, pert_id in enumerate(self.pert_ids):
            if pert_id in node_name_to_idx:
                node_idx = node_name_to_idx[pert_id]
                embeddings[i] = gnn_embeddings[node_idx]
                in_vocab.append(True)
            else:
                in_vocab.append(False)

        self.embeddings = torch.from_numpy(embeddings)  # [N, 256]
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
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "embedding": self.embeddings[idx],   # [256]
            "in_vocab": self.in_vocab[idx],       # bool
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule using STRING_GNN embeddings."""

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 8,
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[DataModule] Computing STRING_GNN embeddings (frozen forward pass)...")
        self.gnn_embeddings, self.node_name_to_idx = build_string_gnn_embeddings(
            STRING_GNN_DIR, device
        )
        print(f"[DataModule] STRING_GNN embeddings shape: {self.gnn_embeddings.shape}")

        # Load all splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        print(f"[DataModule] Coverage: "
              f"{sum(p in self.node_name_to_idx for p in dfs['train']['pert_id'])} / "
              f"{len(dfs['train'])} train genes in STRING_GNN")

        embed_dim = self.gnn_embeddings.shape[1]
        self.train_ds = PerturbationDataset(dfs["train"], self.gnn_embeddings, self.node_name_to_idx, embed_dim, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   self.gnn_embeddings, self.node_name_to_idx, embed_dim, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  self.gnn_embeddings, self.node_name_to_idx, embed_dim, True)

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


class GNNBilinearHead(nn.Module):
    """Prediction head using STRING_GNN embeddings as input features.

    Identical to the proven parent node1-2 architecture (InductiveConditioningModule
    removed as it was net harmful across 4 experiments).

    Architecture:
      1. OOV fallback embedding (learnable, for genes not in STRING_GNN)
      2. Input normalization: LayerNorm(256)
      3. Input projection: 256 -> hidden_dim (512)
      4. Deep residual MLP (n_residual_layers blocks, each expand=4)
      5. Output LayerNorm + Dropout
      6. Bilinear projection: hidden -> n_classes * rank
         Bilinear interaction: [B, rank] x [6640, rank]^T -> [B, 3, 6640]

    This is the exact architecture of node1-2 (F1=0.4912), restored after
    node1-2-2's regression (F1=0.4664) caused by InductiveConditioningModule.
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

        # OOV embedding for genes not in STRING_GNN (fallback)
        self.oov_embedding = nn.Embedding(1, gnn_dim)  # shared OOV token

        # Input normalization (on raw PPI embeddings — no conditioning module)
        self.input_norm = nn.LayerNorm(gnn_dim)

        # Projection: gnn_dim -> hidden_dim
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)

        # Deep residual MLP
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim -> n_classes * rank
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank)

        # Output gene embeddings: learnable [n_genes_out, rank]
        # Randomly initialized — no GNN mapping is available for output positions
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, rank))

        # Head dropout
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
        gnn_emb: torch.Tensor,   # [B, 256] frozen STRING_GNN embeddings
        in_vocab: torch.Tensor,  # [B] bool mask
    ) -> torch.Tensor:
        """
        Args:
            gnn_emb:  [B, gnn_dim] - precomputed frozen STRING_GNN embeddings
            in_vocab: [B] bool - True if gene is in STRING_GNN vocabulary
        Returns:
            logits: [B, 3, 6640]
        """
        B = gnn_emb.shape[0]

        # Step 1: OOV handling — replace out-of-vocab embeddings with learned fallback
        oov_emb = self.oov_embedding(torch.zeros(B, dtype=torch.long, device=gnn_emb.device))
        in_vocab_f = in_vocab.unsqueeze(1).float()  # [B, 1]
        x = gnn_emb * in_vocab_f + oov_emb * (1.0 - in_vocab_f)  # [B, gnn_dim]

        # Step 2: Input normalization + projection to hidden dim
        # (No InductiveConditioningModule — returns clean frozen embedding directly)
        x = self.input_norm(x)
        x = self.proj_in(x)   # [B, hidden_dim]

        # Step 3: Deep residual MLP
        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)   # [B, hidden_dim]

        # Step 4: Bilinear interaction head
        x = self.head_dropout(x)
        pert_proj = self.proj_bilinear(x)             # [B, n_classes * rank]
        pert_proj = pert_proj.view(B, self.n_classes, self.rank)  # [B, 3, rank]

        # Bilinear: [B, 3, rank] x [n_genes_out, rank]^T -> [B, 3, n_genes_out]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, 6640]

        return logits


# ─── Loss ─────────────────────────────────────────────────────────────────────

def class_weighted_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 3.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Class-weighted focal loss for multi-class classification.

    Combines focal loss (handles dynamic instance-level difficulty) with
    class-level static weighting to address the strong label imbalance:
      - Class 0 (down-regulated, -1): 8.1% → weight=3.0
      - Class 1 (neutral, 0):        88.9% → weight=0.3
      - Class 2 (up-regulated, +1):  3.0%  → weight=7.0

    The class weights are applied BEFORE focal weighting, so the final
    weight for each sample is: class_weight * (1 - p_t)^gamma.

    This is an improvement over node1-2's pure focal loss (gamma=2.0, no class
    weights) and node1-2-2's focal loss with label smoothing (which hurt F1):
    - Pure focal loss: dynamically down-weights easy examples but doesn't
      explicitly up-weight the rare up-regulated (+1) class
    - Label smoothing: improved calibration but softened argmax boundaries,
      reducing minority class F1 by shifting ambiguous predictions to neutral
    - Class weighting (this node): explicitly amplifies gradient signal from
      rare minority classes without softening the prediction boundaries

    Args:
        logits:        [B, 3, G] raw logits
        labels:        [B, G] integer class labels (0, 1, 2)
        gamma:         focal loss focusing parameter (3.0 > 2.0 for stronger focus)
        class_weights: [3] per-class weights tensor (on GPU). If None, use uniform.

    Returns:
        Scalar loss value.
    """
    B, C, G = logits.shape

    # Reshape to [B*G, 3] and [B*G]
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
    labels_flat = labels.reshape(-1)                        # [B*G]

    # Compute cross-entropy without reduction (no label smoothing — removed)
    ce_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        reduction="none",
    )  # [B*G]

    # Focal weighting: (1 - p_t)^gamma where p_t = softmax probability of true class
    with torch.no_grad():
        probs = F.softmax(logits_flat, dim=1)           # [B*G, 3]
        pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [B*G]
        focal_weight = (1.0 - pt).pow(gamma)            # [B*G]

    # Class weight: apply per-class multiplier to amplify minority class loss
    if class_weights is not None:
        sample_class_weight = class_weights[labels_flat]  # [B*G]
        loss = (sample_class_weight * focal_weight * ce_loss).mean()
    else:
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
    """LightningModule for gene-perturbation DEG prediction (Node 1-2-2-1).

    Key differences from parent node1-2-2:
    - NO InductiveConditioningModule (removed — proven net harmful)
    - NO label smoothing (removed — improved calibration but hurt F1)
    - Class-weighted focal loss (down-weight neutral, up-weight rare classes)
    - Higher focal gamma (3.0 vs 2.0 — stronger focus on hard examples)
    - total_steps=6600 (restored from node1-2's proven schedule)
    - patience=30 (restored from node1-2's proven setting)
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 256,
        n_residual_layers: int = 6,
        dropout: float = 0.2,
        lr: float = 5e-4,
        weight_decay: float = 1e-3,
        focal_gamma: float = 3.0,
        # Class weights: [down_weight, neutral_weight, up_weight]
        # down=-1 (class 0): 8.1% → 3.0x
        # neutral=0 (class 1): 88.9% → 0.3x
        # up=+1 (class 2): 3.0% → 7.0x
        class_weight_down: float = 3.0,
        class_weight_neutral: float = 0.3,
        class_weight_up: float = 7.0,
        warmup_steps: int = 50,
        total_steps: int = 6600,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Accumulation buffers
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams
        self.model = GNNBilinearHead(
            gnn_dim=hp.gnn_dim,
            hidden_dim=hp.hidden_dim,
            rank=hp.rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
        )
        # Register class weights as a buffer (moves with device)
        class_weights = torch.tensor(
            [hp.class_weight_down, hp.class_weight_neutral, hp.class_weight_up],
            dtype=torch.float32,
        )
        self.register_buffer("class_weights", class_weights)

        # Ensure float32 for all trainable parameters (stable optimization)
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    def forward(
        self,
        gnn_emb: torch.Tensor,
        in_vocab: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(gnn_emb, in_vocab)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return class_weighted_focal_loss(
            logits, labels,
            gamma=self.hparams.focal_gamma,
            class_weights=self.class_weights,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["embedding"].float(), batch["in_vocab"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["embedding"].float(), batch["in_vocab"])
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
        logits = self(batch["embedding"].float(), batch["in_vocab"])
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
            self.print(f"[Node1-2-2-1] Saved test predictions → {pred_path} ({len(seen_ids)} unique samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-2-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )

        # Cosine annealing with linear warmup
        # total_steps=6600 restored from node1-2's proven schedule
        # (node1-2 achieved F1=0.4912 with this exact LR configuration)
        def lr_lambda(step: int) -> float:
            if step < hp.warmup_steps:
                return float(step) / max(1, hp.warmup_steps)
            # Cosine decay after warmup
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
    p = argparse.ArgumentParser(description="Node 1-2-2-1 – STRING_GNN Frozen + Class-Weighted Focal Loss")
    p.add_argument("--data-dir",               type=str,   default="data")
    p.add_argument("--gnn-dim",                type=int,   default=256)
    p.add_argument("--hidden-dim",             type=int,   default=512)
    p.add_argument("--rank",                   type=int,   default=256)
    p.add_argument("--n-residual-layers",      type=int,   default=6)
    p.add_argument("--dropout",                type=float, default=0.2)
    p.add_argument("--lr",                     type=float, default=5e-4)
    p.add_argument("--weight-decay",           type=float, default=1e-3)
    p.add_argument("--focal-gamma",            type=float, default=3.0,
                   help="Focal loss gamma (3.0 > 2.0 for stronger focus on hard examples)")
    p.add_argument("--class-weight-down",      type=float, default=3.0,
                   help="Weight for down-regulated class (class 0, 8.1% of labels)")
    p.add_argument("--class-weight-neutral",   type=float, default=0.3,
                   help="Weight for neutral class (class 1, 88.9% of labels)")
    p.add_argument("--class-weight-up",        type=float, default=7.0,
                   help="Weight for up-regulated class (class 2, 3.0% of labels)")
    p.add_argument("--warmup-steps",           type=int,   default=50)
    p.add_argument("--total-steps",            type=int,   default=6600,
                   help="Total steps for cosine LR schedule. Restored to node1-2's proven "
                        "value of 6600 that enabled the secondary improvement phase.")
    p.add_argument("--micro-batch-size",       type=int,   default=16)
    p.add_argument("--global-batch-size",      type=int,   default=64)
    p.add_argument("--max-epochs",             type=int,   default=300)
    p.add_argument("--patience",               type=int,   default=30,
                   help="EarlyStopping patience (restored to node1-2's proven value of 30)")
    p.add_argument("--num-workers",            type=int,   default=4)
    p.add_argument("--val-check-interval",     type=float, default=1.0)
    p.add_argument("--debug-max-step",         type=int,   default=None)
    p.add_argument("--fast-dev-run",           action="store_true", default=False)
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
        class_weight_down=args.class_weight_down,
        class_weight_neutral=args.class_weight_neutral,
        class_weight_up=args.class_weight_up,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
    )

    # gradient accumulation
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

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
        gradient_clip_val=1.0,   # gradient clipping for stable training
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
            f"Node 1-2-2-1 – STRING_GNN Frozen + Class-Weighted Focal Loss (gamma=3.0)\n"
            f"Test results from trainer: {test_results}\n"
            f"Key settings: focal_gamma={args.focal_gamma}, "
            f"class_weights=[{args.class_weight_down}, {args.class_weight_neutral}, {args.class_weight_up}], "
            f"total_steps={args.total_steps}, patience={args.patience}\n"
        )
        print(f"[Node1-2-2-1] Test score written to {score_path}")


if __name__ == "__main__":
    main()
