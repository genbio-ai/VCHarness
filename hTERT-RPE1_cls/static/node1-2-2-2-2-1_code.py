"""
Node node1-2-2-2-2-1: Partial STRING_GNN FT (mps.7+post_mp) + Rank-512 Bilinear Head
                       + Muon Optimizer + Aggressive Regularization
                       + Class-Weighted Focal Loss + Label Smoothing (ε=0.10)
                       + Cosine Warm Restarts (T_0=600, T_mult=1)

Course correction from parent node1-2-2-2-2 (test F1=0.5031, WORSE than parent at 0.5060):
  - Parent failed because rank=1024 on 1,416 training samples → severe overfitting (val/train 10.41×)
  - Frozen backbone prevented warm restarts from being effective (no staircase improvement)

Key changes from parent (node1-2-2-2-2):
1. Revert bilinear rank 1024 → 512 (CRITICAL FIX)
   Root cause of parent failure: rank=1024 added 7M params on 1,416 samples → memorization
   Tree-wide validation: rank=512 is optimal (proven across node1-2-2-2-1: 0.5099, node1-2-2-3: 0.5101)

2. Add partial backbone fine-tuning (mps.7 + post_mp, ~67K params at lr=1e-5)
   Tree evidence: node1-2-2-2-1 added partial FT to frozen parent → +0.0039 (0.5060→0.5099)
   Critical insight: partial FT creates richer loss landscape enabling warm restart staircase
   Without partial FT (parent), warm restarts caused F1 drops instead of staircase improvement

3. Aggressive regularization: dropout=0.35, label_smoothing=0.10, weight_decay=3e-3
   - Dropout 0.25→0.35: uncle (node1-2-2-2-1) had 4.77× val/train with 0.30, feedback said insufficient
   - Label smoothing 0.05→0.10: uncle recommended ε=0.05–0.10; parent's ε=0.05 failed at rank=1024
   - Weight decay 2e-3→3e-3: additional L2 regularization for partial FT + head combined capacity

4. Warm restarts T_0=600, T_mult=1 (staircase mechanism)
   Uncle (node1-2-2-2-1): T_0=600 × 6 cycles → staircase 0.488→0.496→0.501→0.502→0.503→0.510 → 0.5099
   Parent used T_0=1200, T_mult=2 but FROZEN backbone → no staircase (restarts caused drops)
   Node1-2-2-3 feedback: "restart benefit derives from FREQUENCY of resets rather than cycle depth"
   T_0=600 (27 epochs/cycle) × 6 cycles = ~162 epochs → replicates uncle's successful staircase

5. Restore grad_clip=1.0 (vs parent's 0.5)
   Parent used 0.5 for rank=1024 head. With rank=512, standard 1.0 is appropriate (proven in uncle).

Architecture:
  STRING_GNN mps.0-6 (frozen): one-time forward pass → intermediate embeddings
  STRING_GNN mps.7 + post_mp (trainable, lr=1e-5): adapted final 256-dim PPI embeddings
  OOV fallback: shared learnable embedding for OOV genes
  GNNBilinearHead: LayerNorm(256) → Linear(256→512) → 6x ResBlocks(expand=4, dropout=0.35)
    → LayerNorm(512) + Dropout(0.35) → Linear(512→3*512) → bilinear with out_gene_emb[6640,512]
  Class-weighted focal loss with label smoothing (gamma=2.0, weights=[2.0, 0.5, 4.0], ε=0.10)
  MuonWithAuxAdam:
    - Backbone group: mps.7 + post_mp → AdamW, lr=1e-5, wd=0
    - Muon group: ResBlock 2D matrices → Muon, lr=0.005
    - Head AdamW group: other head params → AdamW, lr=5e-4, wd=3e-3
  Warm Restarts: T_0=600 steps, T_mult=1, eta_min=1e-5, warmup=50 steps

Memory influences:
  - Parent (node1-2-2-2-2): rank=1024 + frozen GNN → F1=0.5031 (FAILED)
    feedback: "abandon rank=1024+frozen; use sibling's partial FT + rank=512 + stronger regularization"
  - Uncle (node1-2-2-2-1): partial FT + rank=512 + T_0=600 → F1=0.5099
    feedback: "T_0=600 staircase works; dropout=0.30 insufficient; label smoothing needed"
  - node1-2-2-3 (cousin): partial FT + rank=512 + T_0=1200 → F1=0.5101 (tree best)
    feedback: "frequency of restarts > depth; T_0=600 better than T_0=1200 for staircase"
  - node1-2-2-3-1-1: dropout=0.40 improved val/train ratio to 3.05× but F1=0.5088
    → dropout=0.35 is safer intermediate
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

# Muon optimizer (MuonWithAuxAdam handles both Muon and AdamW in one optimizer)
from muon import MuonWithAuxAdam

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


# ─── STRING_GNN Partial Fine-Tuning ──────────────────────────────────────────

def load_string_gnn_for_partial_ft(
    model_dir: Path,
    device: torch.device,
) -> Tuple[nn.Module, torch.Tensor, Optional[torch.Tensor], Dict[str, int]]:
    """Load STRING_GNN model for partial fine-tuning.

    Freezes all layers except mps.7 and post_mp.
    Returns (model, edge_index, edge_weight, node_name_to_idx).

    Unlike the frozen embedding approach (build_string_gnn_embeddings), this function
    keeps the model on device and returns graph tensors for use in each forward pass.
    The model computes embeddings dynamically each step, allowing gradients to flow
    through mps.7 and post_mp.
    """
    node_names = json.loads((model_dir / "node_names.json").read_text())
    graph = torch.load(model_dir / "graph_data.pt", weights_only=False)

    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)

    # Freeze all layers first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze mps.7 (last message-passing layer) and post_mp (output projection)
    for name, p in model.named_parameters():
        if name.startswith("mps.7.") or name.startswith("post_mp."):
            p.requires_grad = True

    # Move to device
    model = model.to(device)

    # Prepare graph tensors
    edge_index = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"].to(device) if graph.get("edge_weight") is not None else None

    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

    n_trainable_backbone = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_backbone = sum(p.numel() for p in model.parameters())
    print(f"[StringGNN] Partial FT: {n_trainable_backbone}/{n_total_backbone} backbone params trainable "
          f"(mps.7 + post_mp)")

    return model, edge_index, edge_weight, node_name_to_idx


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset for partial backbone FT.

    Instead of precomputed embeddings, stores pert_id indices into the STRING_GNN
    node vocabulary (or OOV flag) for lookup during forward pass.
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

        n_samples = len(df)
        # For each sample: store STRING_GNN node index (or -1 for OOV)
        node_indices = []
        for pert_id in self.pert_ids:
            if pert_id in node_name_to_idx:
                node_indices.append(node_name_to_idx[pert_id])
            else:
                node_indices.append(-1)

        self.node_indices = torch.tensor(node_indices, dtype=torch.long)  # [N], -1 for OOV
        self.in_vocab = (self.node_indices >= 0)  # [N] bool

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
            "node_idx": self.node_indices[idx],   # int (STRING_GNN node idx, or -1)
            "in_vocab": self.in_vocab[idx],        # bool
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """DataModule for partial backbone FT.

    Stores node_name_to_idx for dataset construction. The STRING_GNN model and
    graph tensors are managed by the LightningModule (on the correct GPU device).
    """

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

        # Load node names for vocabulary mapping
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.node_name_to_idx = {name: i for i, name in enumerate(node_names)}

        # Load all splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        in_vocab_train = sum(1 for p in dfs["train"]["pert_id"] if p in self.node_name_to_idx)
        print(f"[DataModule] Coverage: {in_vocab_train}/{len(dfs['train'])} train genes in STRING_GNN")

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
    """Residual MLP block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout + skip.

    The 2D weight matrices inside this block are optimized by Muon.
    Per Muon skill: Muon is designed for hidden 2D weight matrices in MLP blocks.
    """

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.35):
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

    Key improvements over parent (node1-2-2-2-2):
      - Reverted bilinear rank: 1024 → 512 (fixes overfitting)
      - Stronger dropout: 0.25 → 0.35 (reduces val/train loss ratio)
      - Stronger label smoothing: ε=0.05 → 0.10 (suppresses overconfident predictions)
      - Partial backbone FT: mps.7+post_mp at lr=1e-5 (enables staircase improvement)

    Architecture:
      1. OOV fallback embedding (learnable, for genes not in STRING_GNN)
      2. Input normalization + projection: gnn_dim → hidden_dim
      3. Deep residual MLP (6 blocks, expand=4, dropout=0.35)
      4. Bilinear interaction: [B, rank] x [6640, rank]^T -> [B, 3, 6640]
         where rank=512
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.35,
        n_residual_layers: int = 6,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.rank = rank

        # OOV embedding for genes not in STRING_GNN (fallback)
        self.oov_embedding = nn.Embedding(1, gnn_dim)  # shared OOV token

        # Input normalization (layer norm on the PPI embeddings)
        self.input_norm = nn.LayerNorm(gnn_dim)

        # Projection: gnn_dim -> hidden_dim (AdamW, not Muon — input projection)
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)

        # Deep residual MLP (Linear layers inside ResidualBlocks → Muon)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim -> n_classes * rank (AdamW, output projection)
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank)

        # Output gene embeddings: learnable [n_genes_out, rank]
        # rank=512: [6640, 512] = 3.4M params
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
        # Standard std for rank=512 (same as grandparent node1-2-2-2)
        nn.init.normal_(self.out_gene_emb, std=0.02)

    def forward(
        self,
        gnn_emb: torch.Tensor,   # [B, 256] STRING_GNN embeddings (from partial FT or OOV)
        in_vocab: torch.Tensor,  # [B] bool mask
    ) -> torch.Tensor:
        """
        Args:
            gnn_emb:  [B, gnn_dim] - STRING_GNN embeddings (partial FT output)
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
        x = self.input_norm(x)
        x = self.proj_in(x)   # [B, hidden_dim]

        # Step 3: Deep residual MLP (Muon-optimized hidden matrices)
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

def class_weighted_focal_loss_with_smoothing(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.10,
) -> torch.Tensor:
    """Class-weighted focal loss with label smoothing.

    Label smoothing converts hard one-hot targets to soft targets:
      hard target (one-hot):  [0, 1, 0]  →  soft target: [ε/2, 1-ε, ε/2]
      where ε = label_smoothing = 0.10

    This suppresses overconfident predictions and reduces val/train loss ratio
    (uncle node1-2-2-2-1 reached 4.77× val/train ratio with no label smoothing).

    Focal weight is still computed from hard true class probability (not smoothed),
    so focal weighting remains meaningful while targets are regularized.

    Args:
        logits: [B, 3, G] raw logits
        labels: [B, G] integer class labels (0, 1, 2)
        class_weights: [3] per-class weights tensor
        gamma: focal loss focusing parameter (2.0)
        label_smoothing: smoothing factor ε (0.10)

    Returns:
        Scalar loss value.
    """
    B, C, G = logits.shape

    # Reshape to [B*G, 3] and [B*G]
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
    labels_flat = labels.reshape(-1)                        # [B*G]

    # Compute log-softmax for numerical stability
    log_probs = F.log_softmax(logits_flat, dim=1)  # [B*G, 3]

    # Label smoothing: create soft targets
    if label_smoothing > 0.0:
        smooth_val = label_smoothing / (C - 1)
        soft_targets = torch.full_like(logits_flat, smooth_val)
        soft_targets.scatter_(1, labels_flat.unsqueeze(1), 1.0 - label_smoothing)
        ce_loss = -(soft_targets * log_probs).sum(dim=1)  # [B*G]
    else:
        ce_loss = -log_probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [B*G]

    # Focal weighting: computed from hard true class probability (not smoothed)
    with torch.no_grad():
        probs = logits_flat.softmax(dim=1)                 # [B*G, 3]
        pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [B*G]
        focal_weight = (1.0 - pt).pow(gamma)               # [B*G]

    # Class weighting: per-sample static multiplier based on true class
    sample_class_weight = class_weights[labels_flat]        # [B*G]

    loss = (sample_class_weight * focal_weight * ce_loss).mean()
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
    """LightningModule for gene-perturbation DEG prediction (node1-2-2-2-2-1).

    Key differences from parent (node1-2-2-2-2):
    - Rank=512 bilinear head (reverted from 1024 — fixes overfitting)
    - Partial STRING_GNN backbone FT (mps.7+post_mp at lr=1e-5) — enables staircase improvement
    - Stronger dropout (0.35 vs 0.25)
    - Stronger label smoothing (ε=0.10 vs 0.05)
    - Higher weight decay for head (3e-3 vs 2e-3)
    - Warm restarts T_0=600, T_mult=1 (staircase mechanism proven in uncle node1-2-2-2-1)
    - Restored grad_clip=1.0 (appropriate for rank=512)
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,                      # REVERTED: 1024 → 512
        n_residual_layers: int = 6,
        dropout: float = 0.35,                # INCREASED: 0.25 → 0.35
        lr: float = 5e-4,                     # AdamW LR for head params
        muon_lr: float = 0.005,               # Muon LR for ResBlock 2D matrices
        backbone_lr: float = 1e-5,            # NEW: AdamW LR for backbone (mps.7+post_mp)
        weight_decay: float = 3e-3,           # INCREASED: 2e-3 → 3e-3
        focal_gamma: float = 2.0,
        class_weights: List[float] = None,    # [2.0, 0.5, 4.0] (down, neutral, up)
        label_smoothing: float = 0.10,        # INCREASED: 0.05 → 0.10
        warmup_steps: int = 50,
        t0_steps: int = 600,                  # CHANGED: 1200 → 600 (staircase mechanism)
        t_mult: int = 1,                      # CHANGED: 2 → 1 (fixed-length cycles)
        cosine_eta_min: float = 1e-5,         # Non-zero LR floor between restarts
    ):
        super().__init__()
        if class_weights is None:
            class_weights = [2.0, 0.5, 4.0]
        self.save_hyperparameters()
        self._class_weights_list = class_weights
        # Accumulation buffers
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams
        # Use LOCAL_RANK to determine the correct GPU device
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        # Load STRING_GNN for partial fine-tuning
        # mps.7 + post_mp are trainable; all other layers frozen
        print(f"[LitModule] Loading STRING_GNN for partial fine-tuning (mps.7 + post_mp) on {device}...")
        self.backbone, edge_index, edge_weight, node_name_to_idx = \
            load_string_gnn_for_partial_ft(STRING_GNN_DIR, device)

        # Store graph tensors as buffers (will follow model device via Lightning's .to())
        self.register_buffer("edge_index_buf", edge_index.long())
        if edge_weight is not None:
            self.register_buffer("edge_weight_buf", edge_weight.float())
            self._has_edge_weight = True
        else:
            self.register_buffer("edge_weight_buf", torch.zeros(1))  # dummy buffer
            self._has_edge_weight = False

        # Initialize bilinear prediction head
        self.model = GNNBilinearHead(
            gnn_dim=hp.gnn_dim,
            hidden_dim=hp.hidden_dim,
            rank=hp.rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
        )

        # Ensure float32 for all trainable parameters (stable optimization)
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        for p in self.backbone.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Register class weights as buffer
        self.register_buffer(
            "class_weights_buf",
            torch.tensor(self._class_weights_list, dtype=torch.float32)
        )

        total_trainable = (
            sum(p.numel() for p in self.model.parameters() if p.requires_grad) +
            sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        )
        print(f"[LitModule] Total trainable params: {total_trainable:,}")

    def _get_backbone_embeddings(self, node_idx: torch.Tensor, in_vocab: torch.Tensor) -> torch.Tensor:
        """Run STRING_GNN forward pass and extract embeddings for batch samples.

        Args:
            node_idx: [B] long tensor with STRING_GNN node indices (-1 for OOV)
            in_vocab: [B] bool tensor

        Returns:
            [B, gnn_dim] embeddings — OOV samples will be handled in the head
        """
        # Run STRING_GNN forward (mps.7+post_mp are trainable, rest frozen)
        edge_w = self.edge_weight_buf if self._has_edge_weight else None
        outputs = self.backbone(
            edge_index=self.edge_index_buf,
            edge_weight=edge_w,
        )
        all_embeddings = outputs.last_hidden_state  # [N_nodes, 256]

        B = node_idx.shape[0]
        gnn_dim = all_embeddings.shape[1]

        # Build batch embeddings: in-vocab genes get their STRING_GNN embedding
        # OOV genes get zeros (will be replaced by learned OOV token in the head)
        batch_emb = torch.zeros(B, gnn_dim, dtype=all_embeddings.dtype, device=all_embeddings.device)
        in_vocab_mask = in_vocab.bool()
        if in_vocab_mask.any():
            valid_indices = node_idx[in_vocab_mask].long()
            # Clamp to valid range for safety
            valid_indices = valid_indices.clamp(0, all_embeddings.shape[0] - 1)
            batch_emb[in_vocab_mask] = all_embeddings[valid_indices]

        return batch_emb

    def forward(
        self,
        node_idx: torch.Tensor,
        in_vocab: torch.Tensor,
    ) -> torch.Tensor:
        gnn_emb = self._get_backbone_embeddings(node_idx, in_vocab)
        return self.model(gnn_emb.float(), in_vocab)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return class_weighted_focal_loss_with_smoothing(
            logits, labels,
            class_weights=self.class_weights_buf,
            gamma=self.hparams.focal_gamma,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_idx"], batch["in_vocab"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_idx"], batch["in_vocab"])
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
        logits = self(batch["node_idx"], batch["in_vocab"])
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
            self.print(f"[node1-2-2-2-2-1] Saved test predictions → {pred_path} ({len(seen_ids)} samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[node1-2-2-2-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # ─── Three-group parameter configuration ─────────────────────────────
        # Group 1: Backbone (mps.7 + post_mp) → AdamW, lr=1e-5, wd=0
        #   Very conservative LR to avoid disrupting pretrained PPI representations
        #   No weight decay (backbone params are already regularized by pretraining)
        # Group 2: ResidualBlock 2D weight matrices → Muon, lr=0.005
        #   Per Muon skill: 2D hidden matrices benefit from orthogonalized gradients
        # Group 3: All other head params → AdamW, lr=5e-4, wd=3e-3
        #   Standard head learning rate; higher WD than parent (2e-3→3e-3)

        backbone_params = []
        muon_params = []
        adamw_params = []

        # Backbone trainable params (mps.7 + post_mp)
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)

        # Head params: Muon for ResBlock 2D matrices, AdamW for everything else
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "res_blocks" in name and param.ndim >= 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        param_groups = [
            # Group 1: Backbone (AdamW, very low LR, no WD)
            dict(
                params=backbone_params,
                use_muon=False,
                lr=hp.backbone_lr,           # 1e-5 — very conservative
                betas=(0.9, 0.95),
                weight_decay=0.0,            # No backbone weight decay
            ),
            # Group 2: Muon for ResBlock hidden 2D weight matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,              # 0.005
                momentum=0.95,
                weight_decay=0.0,           # No weight decay for Muon
            ),
            # Group 3: AdamW for all other head parameters
            dict(
                params=adamw_params,
                use_muon=False,
                lr=hp.lr,                   # 5e-4
                betas=(0.9, 0.95),
                weight_decay=hp.weight_decay,  # 3e-3
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # ─── Cosine Annealing with Warm Restarts (T_0=600, T_mult=1) ─────────
        # T_0=600 steps (~27 epochs with 22 steps/epoch at bs=64 on 1416 samples):
        #   Cycle 1: warmup (50 steps ≈ 2 epochs) + cosine (600 steps ≈ 27 epochs) → ~29 epochs
        #   Cycle 2: 600 steps ≈ 27 epochs (from step 650 to 1250)
        #   Cycle 3: 600 steps ≈ 27 epochs (from step 1250 to 1850)
        #   ...up to 6 cycles in ~162 epochs
        #
        # Evidence: uncle node1-2-2-2-1 achieved staircase improvement with T_0=600:
        #   0.488 → 0.496 → 0.501 → 0.502 → 0.503 → 0.510 (over 6 cycles → 0.5099)
        # Node1-2-2-3 feedback: "frequency of restarts benefits > depth of convergence"
        #
        # With partial backbone FT (unlike frozen parent), the richer loss landscape
        # enables productive escape from local minima at each restart.
        #
        # T_mult=1: fixed-length cycles (unlike parent's T_mult=2 geometric growth)
        # This preserves the staircase mechanism (equal-length cycles in node1-2-2-2-1).
        #
        # Implementation via LambdaLR:
        #   - Warmup phase (steps < warmup_steps): linear from 0 to 1
        #   - After warmup: T_mult=1 simplifies to modular arithmetic

        def lr_lambda(step: int) -> float:
            # Warmup phase
            if step < hp.warmup_steps:
                return float(step) / max(1, hp.warmup_steps)

            # Step after warmup
            step_adj = step - hp.warmup_steps
            t0 = float(hp.t0_steps)

            # T_mult=1: all cycles have equal length T_0
            # cycle position is simply step_adj modulo T_0
            cycle_pos = float(step_adj % int(t0)) / t0

            # Cosine decay within current cycle
            eta_min_frac = hp.cosine_eta_min / hp.lr
            return eta_min_frac + (1.0 - eta_min_frac) * 0.5 * (1.0 + np.cos(np.pi * cycle_pos))

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
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        trainable_sd = {k: v for k, v in full_sd.items()
                        if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)")
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="node1-2-2-2-2-1 – Partial STRING_GNN FT + Rank-512 + Muon + "
                    "Aggressive Regularization + Warm Restarts (T_0=600)"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--gnn-dim",            type=int,   default=256)
    p.add_argument("--hidden-dim",         type=int,   default=512)
    p.add_argument("--rank",               type=int,   default=512,
                   help="Bilinear rank. REVERTED from 1024 to 512 (proven optimal for 1416 samples).")
    p.add_argument("--n-residual-layers",  type=int,   default=6)
    p.add_argument("--dropout",            type=float, default=0.35,
                   help="Dropout. INCREASED from 0.25 to 0.35. Uncle (node1-2-2-2-1) with 0.30 "
                        "had 4.77x val/train ratio; 0.35 provides stronger regularization.")
    p.add_argument("--lr",                 type=float, default=5e-4,
                   help="AdamW LR for head parameters (non-Muon, non-backbone).")
    p.add_argument("--muon-lr",            type=float, default=0.005,
                   help="Muon LR for ResidualBlock 2D hidden weight matrices.")
    p.add_argument("--backbone-lr",        type=float, default=1e-5,
                   help="AdamW LR for backbone (mps.7+post_mp). Very conservative to avoid "
                        "disrupting pretrained PPI representations. Proven in node1-2-2-2-1.")
    p.add_argument("--weight-decay",       type=float, default=3e-3,
                   help="AdamW weight decay for head. INCREASED from 2e-3 to 3e-3 for "
                        "additional L2 regularization on combined head+backbone FT capacity.")
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--class-weights",      type=float, nargs=3,
                   default=[2.0, 0.5, 4.0],
                   metavar=("DOWN_W", "NEUTRAL_W", "UP_W"),
                   help="Per-class focal loss weights [down, neutral, up]. Validated across tree.")
    p.add_argument("--label-smoothing",    type=float, default=0.10,
                   help="Label smoothing ε. INCREASED from 0.05 to 0.10. Uncle had 4.77x "
                        "val/train ratio with 0.0; parent's ε=0.05 was insufficient at rank=1024. "
                        "ε=0.10 is standard NLP value, conservative for this task.")
    p.add_argument("--warmup-steps",       type=int,   default=50)
    p.add_argument("--t0-steps",           type=int,   default=600,
                   help="T_0 for cosine warm restarts. CHANGED from 1200 to 600. "
                        "Uncle (node1-2-2-2-1) achieved staircase improvement with T_0=600: "
                        "0.488→0.496→0.501→0.502→0.503→0.510 over 6 cycles. "
                        "Node1-2-2-3 feedback: frequency of restarts > depth of convergence.")
    p.add_argument("--t-mult",             type=int,   default=1,
                   help="T_mult for warm restart cycles. CHANGED from 2 to 1 (fixed-length cycles). "
                        "Uncle's T_mult=1 created consistent staircase; T_mult=2 (parent) failed.")
    p.add_argument("--cosine-eta-min",     type=float, default=1e-5,
                   help="Minimum LR floor between warm restarts.")
    p.add_argument("--micro-batch-size",   type=int,   default=16)
    p.add_argument("--global-batch-size",  type=int,   default=64)
    p.add_argument("--max-epochs",         type=int,   default=300,
                   help="Max epochs. 300 allows ~11 cycles at T_0=600 (27 epochs/cycle). "
                        "Uncle ran 251 epochs (6 cycles) to reach F1=0.5099.")
    p.add_argument("--patience",           type=int,   default=80,
                   help="EarlyStopping patience. 80 epochs allows ~3 additional warm restart "
                        "cycles after best checkpoint (matching uncle's successful configuration).")
    p.add_argument("--grad-clip",          type=float, default=1.0,
                   help="Gradient clipping. RESTORED to 1.0 (parent used 0.5 for rank=1024). "
                        "With rank=512, standard 1.0 is appropriate (proven in uncle).")
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
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
        muon_lr=args.muon_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        class_weights=args.class_weights,
        label_smoothing=args.label_smoothing,
        warmup_steps=args.warmup_steps,
        t0_steps=args.t0_steps,
        t_mult=args.t_mult,
        cosine_eta_min=args.cosine_eta_min,
    )

    # Gradient accumulation
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
        gradient_clip_val=args.grad_clip,  # 1.0 (restored for rank=512)
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
            f"Node node1-2-2-2-2-1 – Partial STRING_GNN FT + Rank-512 + Muon + "
            f"Aggressive Regularization (dropout=0.35, ε=0.10, wd=3e-3) + "
            f"Warm Restarts (T_0=600, T_mult=1)\n"
            f"Test results from trainer: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
