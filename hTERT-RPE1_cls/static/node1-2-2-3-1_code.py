"""
Node 1-2-2-3-1: STRING_GNN Partial Fine-tuning + Rank-512 Bilinear Head + Muon
              + Hybrid Cosine Warm Restarts (T_0=600, T_mult=1.5) + Stronger Weight Decay

Key improvements over parent (node1-2-2-3, F1=0.5101):

1. Hybrid cosine warm restarts: T_0=600 steps (≈27 epochs/cycle), T_mult=1.5 (progressive).
   Analysis of parent: T_0=1200 (uniform long cycles) produced only 3 restart events in 179
   epochs — too few events for the staircase accumulation mechanism (each restart's escape jump
   is ~0.007-0.010 F1, so 3 events ≈ same ceiling as 1). The staircase needs FREQUENCY of
   escapes first, THEN depth.

   Prior art:
   - node1-2-2-2-1 (T_0=600, T_mult=1, 6 cycles): ascending staircase 0.488→0.510 in 251 epochs
   - parent node1-2-2-3 (T_0=1200, T_mult=1, 3 cycles): single spike at epoch 58 (0.510),
     then declining staircase (0.503, 0.499) over 179 epochs

   This node's T_mult=1.5 compromise (feedback primary recommendation):
   - Cycle 1: 600 steps ≈ 27 epochs → rapid initial escape (matches node1-2-2-2-1's pace)
   - Cycle 2: 900 steps ≈ 41 epochs → deeper convergence in second cycle
   - Cycle 3: 1350 steps ≈ 61 epochs → even deeper in third cycle
   - Cycle 4: 2025 steps ≈ 92 epochs → deepest convergence in fourth cycle
   - ~4 full cycles in 350 epochs with progressive deepening

   Expected: combine the staircase frequency of T_0=600 (fast initial gains) with the deeper
   per-cycle convergence of T_0=1200 (deeper later gains). Each cycle can contribute +0.003-0.007
   to the staircase before deeper and deeper minima are reached.

2. Stronger head weight_decay=4e-3 (increased from 2e-3 in parent).
   Parent observed: val/train loss ratio grew from 3.05× at epoch 58 to 6.21× at epoch 178,
   indicating progressive overfitting unchecked by dropout=0.30 alone. The feedback specifically
   recommends "increase AdamW head weight_decay from 2e-3 to 4e-3 or 5e-3" to combat the growing
   ratio. With longer individual cycles (up to 92 epochs in cycle 4), stronger weight decay is
   necessary to prevent overfitting within each cycle. Backbone weight_decay kept at 1e-3.

3. Adjusted patience=100 (from 120 in parent).
   With T_mult=1.5, cycle lengths grow progressively: 27, 41, 61, 92 epochs. Patience=100
   covers 1+ complete cycle for cycle 3 (61 epochs) and cycle 4 (92 epochs), while still
   allowing early stopping if no improvement across a full cycle. Prevents excessive time in
   degraded later cycles if staircase plateaus early.

4. All proven ingredients inherited from parent unchanged:
   - Partial STRING_GNN fine-tuning (mps.7 + post_mp, ~67K params, backbone_lr=1e-5)
   - Muon optimizer (lr=0.005) for ResBlock 2D hidden weight matrices
   - Rank=512 bilinear head (6× ResidualBlocks, hidden=512, dropout=0.30)
   - Class-weighted focal loss (gamma=2.0, weights=[2.0, 0.5, 4.0])
   - micro_batch_size=16, global_batch_size=64 (accum=4 per GPU)
   - max_epochs=350

Differentiation from parent (node1-2-2-3) and grandparent (node1-2-2-2-1):
- node1-2-2-2-1 (F1=0.5099): T_0=600, T_mult=1 (uniform short cycles, 6 total) — staircase
  with diminishing returns after cycle 3; no progressive deepening
- node1-2-2-3 (F1=0.5101): T_0=1200, T_mult=1 (uniform long cycles, 3 total) — single spike
  at restart 1, then declining; insufficient restart frequency
- THIS NODE: T_0=600, T_mult=1.5 (progressive from short → long) — combines restart frequency
  (early cycles match T_0=600) with progressive deepening (later cycles match T_0=1200+)
  PLUS stronger weight_decay=4e-3 for overfitting suppression

Architecture:
  - STRING_GNN (partial fine-tune): mps.7 + post_mp at backbone_lr=1e-5 (AdamW)
    All other backbone layers frozen. Live forward pass each training step.
  - GNNBilinearHead: LayerNorm(256) → Linear(256→512) → 6x ResBlocks(expand=4, dropout=0.30)
    → LayerNorm(512) + Dropout(0.30) → Linear(512→3*512) → bilinear with out_gene_emb[6640,512]
  - Loss: class-weighted focal loss (gamma=2.0, weights=[2.0, 0.5, 4.0])
  - Optimizer: 3-group MuonWithAuxAdam
    * Group 1 (Muon, lr=0.005): ResBlock 2D weight matrices
    * Group 2 (AdamW, lr=5e-4, wd=4e-3): Head non-ResBlock params [INCREASED WD]
    * Group 3 (AdamW, lr=1e-5, wd=1e-3): STRING_GNN mps.7 + post_mp backbone params
  - Schedule: LambdaLR with T_0=600 steps, T_mult=1.5 (progressive cycle lengthening)
    + 50-step linear warmup; LR floor at 1% of peak

Memory influences:
  - node1-2-2-3 (parent, F1=0.5101): proven combination of Muon+rank-512+partial FT+warm restarts
    → Inherited as foundation; T_0 and T_mult changed based on parent's feedback
  - node1-2-2-3 feedback (PRIMARY RECOMMENDATION): "T_mult=1.5 with shorter T_0=600 initial cycle
    combines staircase frequency of T_0=600 with progressive cycle lengthening for deeper exploitation"
    → Implemented as the core innovation
  - node1-2-2-3 feedback: "increase AdamW head weight_decay from 2e-3 to 4e-3 or 5e-3"
    → Increased head weight_decay to 4e-3
  - node1-2-2-2-1 (F1=0.5099, T_0=600 T_mult=1): proven 6-cycle staircase from 0.488→0.510
    → T_0=600 initial cycle preserves this proven rapid initial phase
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

# Muon optimizer (MuonWithAuxAdam handles Muon and AdamW in one optimizer)
# MuonWithAuxAdam handles gradient synchronization for distributed training.
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


# ─── STRING_GNN Loading ───────────────────────────────────────────────────────

def load_string_gnn_for_finetuning(
    model_dir: Path,
    device: torch.device,
    backbone_layers: List[str] = None,
) -> Tuple[nn.Module, Dict[str, int], torch.Tensor, Optional[torch.Tensor]]:
    """Load STRING_GNN with partial unfreezing for fine-tuning.

    Unfreezes only the specified backbone layers (default: mps.7 + post_mp).
    All other layers are frozen.

    Args:
        model_dir: Path to STRING_GNN model directory
        device: Target device
        backbone_layers: List of layer name prefixes to unfreeze (e.g., ['mps.7', 'post_mp'])

    Returns:
        (model, node_name_to_idx, edge_index, edge_weight)
    """
    if backbone_layers is None:
        backbone_layers = ["mps.7", "post_mp"]

    node_names = json.loads((model_dir / "node_names.json").read_text())
    graph = torch.load(model_dir / "graph_data.pt", weights_only=False)

    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    model = model.to(device)

    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze specified backbone layers
    n_unfrozen = 0
    for name, param in model.named_parameters():
        for layer_prefix in backbone_layers:
            if name.startswith(layer_prefix):
                param.requires_grad = True
                n_unfrozen += 1
                break

    print(f"[STRING_GNN] Unfrozen {n_unfrozen} parameters in layers: {backbone_layers}")

    edge_index = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"].to(device) if graph.get("edge_weight") is not None else None

    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

    return model, node_name_to_idx, edge_index, edge_weight


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset using per-sample STRING_GNN node indices for live embedding."""

    def __init__(
        self,
        df: pd.DataFrame,
        node_name_to_idx: Dict[str, int],
        n_nodes: int,
        embed_dim: int = 256,
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.has_labels = has_labels
        self.embed_dim = embed_dim

        # Store node indices for each sample (for indexing into live backbone embeddings)
        self.node_indices: List[int] = []
        self.in_vocab: List[bool] = []
        for pert_id in self.pert_ids:
            if pert_id in node_name_to_idx:
                self.node_indices.append(node_name_to_idx[pert_id])
                self.in_vocab.append(True)
            else:
                self.node_indices.append(0)  # placeholder; masked by in_vocab=False
                self.in_vocab.append(False)

        self.node_indices_t = torch.tensor(self.node_indices, dtype=torch.long)  # [N]
        self.in_vocab_t = torch.tensor(self.in_vocab, dtype=torch.bool)          # [N]

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
            "node_idx": self.node_indices_t[idx],   # int: index into backbone embeddings
            "in_vocab": self.in_vocab_t[idx],        # bool
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule with partial STRING_GNN fine-tuning support."""

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

        # We only need node_name_to_idx for the dataset (node indices for live forward pass)
        # The actual embeddings are computed live by the STRING_GNN backbone in the model
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.node_name_to_idx = {name: i for i, name in enumerate(node_names)}
        self.n_nodes = len(node_names)
        print(f"[DataModule] STRING_GNN node count: {self.n_nodes}")

        # Load all splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        print(f"[DataModule] Coverage: "
              f"{sum(p in self.node_name_to_idx for p in dfs['train']['pert_id'])} / "
              f"{len(dfs['train'])} train genes in STRING_GNN")

        self.train_ds = PerturbationDataset(dfs["train"], self.node_name_to_idx, self.n_nodes, 256, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   self.node_name_to_idx, self.n_nodes, 256, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  self.node_name_to_idx, self.n_nodes, 256, True)

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

    The 2D weight matrices (Linear layers) inside this block are optimized by Muon.
    Per Muon skill: Muon is designed for hidden 2D weight matrices in MLP blocks.
    """

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.30):
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
    """Prediction head using live STRING_GNN backbone embeddings as input features.

    Improvements over parent (node1-2-2-3):
      - Identical architecture — improvements are in training schedule and regularization
      - Rank-512 bilinear head (proven +0.006-0.010 F1 over rank-256)
      - Increased dropout=0.30 (stronger regularization)
      - Receives live backbone embeddings (not precomputed static ones)

    Architecture:
      1. OOV fallback embedding (learnable, for genes not in STRING_GNN)
      2. Input normalization + projection: gnn_dim(256) -> hidden_dim(512)
      3. Deep residual MLP (6 blocks, expand=4, dropout=0.30) [Muon-optimized]
      4. Bilinear interaction: [B, rank=512] x [6640, rank=512]^T -> [B, 3, 6640]
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.30,
        n_residual_layers: int = 6,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.rank = rank

        # OOV embedding for genes not in STRING_GNN (fallback)
        self.oov_embedding = nn.Embedding(1, gnn_dim)

        # Input normalization + projection
        self.input_norm = nn.LayerNorm(gnn_dim)
        # proj_in: input projection → AdamW (first layer, not hidden)
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)

        # Deep residual MLP: 2D weights → Muon
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear projection: output layer → AdamW
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank)

        # Output gene embeddings: [n_genes_out, rank] → AdamW
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
        gnn_emb: torch.Tensor,   # [B, 256] live STRING_GNN embeddings
        in_vocab: torch.Tensor,  # [B] bool mask
    ) -> torch.Tensor:
        """
        Args:
            gnn_emb:  [B, gnn_dim] - live STRING_GNN backbone embeddings
            in_vocab: [B] bool - True if gene is in STRING_GNN vocabulary
        Returns:
            logits: [B, 3, 6640]
        """
        B = gnn_emb.shape[0]

        # OOV handling
        oov_emb = self.oov_embedding(torch.zeros(B, dtype=torch.long, device=gnn_emb.device))
        in_vocab_f = in_vocab.unsqueeze(1).float()  # [B, 1]
        x = gnn_emb * in_vocab_f + oov_emb * (1.0 - in_vocab_f)  # [B, gnn_dim]

        # Input normalization + projection
        x = self.input_norm(x)
        x = self.proj_in(x)   # [B, hidden_dim]

        # Deep residual MLP (Muon-optimized hidden 2D matrices)
        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)   # [B, hidden_dim]

        # Bilinear interaction head
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
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Class-weighted focal loss for multi-class classification.

    Combines:
    - Static class weighting: explicit per-class multipliers for imbalance
    - Dynamic focal weighting: down-weights easy examples
    - No label smoothing: proven better for F1 in this task

    Class weights validated as optimal across multiple tree branches:
    - down-regulated (class 0, 8.1%):   weight = 2.0
    - neutral (class 1, 88.9%):          weight = 0.5 (down-weight)
    - up-regulated (class 2, 3.0%):      weight = 4.0 (up-weight)

    Args:
        logits: [B, 3, G] raw logits
        labels: [B, G] integer class labels (0, 1, 2)
        class_weights: [3] per-class weights tensor on the correct device
        gamma: focal loss focusing parameter

    Returns:
        Scalar loss value.
    """
    B, C, G = logits.shape

    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
    labels_flat = labels.reshape(-1)                        # [B*G]

    ce_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        reduction="none",
    )  # [B*G]

    with torch.no_grad():
        probs = F.softmax(logits_flat, dim=1)           # [B*G, 3]
        pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [B*G]
        focal_weight = (1.0 - pt).pow(gamma)            # [B*G]

    sample_class_weight = class_weights[labels_flat]    # [B*G]
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
    """LightningModule for gene-perturbation DEG prediction (Node 1-2-2-3-1).

    Key differences from parent (node1-2-2-3):
    - Hybrid cosine warm restarts: T_0=600 steps, T_mult=1.5 (progressive cycle lengthening)
      vs parent's T_0=1200, T_mult=1 (uniform long cycles)
    - Stronger head weight_decay=4e-3 (vs 2e-3 in parent)
    - Adjusted patience=100 (vs 120 in parent)
    - Everything else identical: partial FT, Muon, rank-512, dropout=0.30, class weights
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_residual_layers: int = 6,
        dropout: float = 0.30,
        lr: float = 5e-4,
        muon_lr: float = 0.005,
        backbone_lr: float = 1e-5,
        weight_decay: float = 4e-3,          # CHANGED: 2e-3 → 4e-3 for stronger head regularization
        backbone_wd: float = 1e-3,
        focal_gamma: float = 2.0,
        class_weights: List[float] = None,
        warmup_steps: int = 50,
        t0_steps: int = 600,                 # CHANGED: 1200 → 600 for shorter initial cycles
        t_mult: float = 1.5,                 # NEW: progressive cycle lengthening (T_mult=1.5)
        backbone_layers: List[str] = None,
    ):
        super().__init__()
        if class_weights is None:
            class_weights = [2.0, 0.5, 4.0]
        if backbone_layers is None:
            backbone_layers = ["mps.7", "post_mp"]
        self.save_hyperparameters()
        self._class_weights_list = class_weights
        self._backbone_layers = backbone_layers
        # Accumulation buffers
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        # Guard against double initialization (setup is called for 'fit' and 'test')
        if hasattr(self, "backbone"):
            return

        hp = self.hparams
        device = self.device if self.device.type != "cpu" else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load STRING_GNN with partial unfreezing
        print(f"[Model] Loading STRING_GNN with partial fine-tuning: {self._backbone_layers}")
        self.backbone, self.node_name_to_idx, self.edge_index, self.edge_weight = \
            load_string_gnn_for_finetuning(STRING_GNN_DIR, device, self._backbone_layers)

        # Keep edge_index/edge_weight on device for forward pass
        # These are registered as buffers for correct device tracking
        self.register_buffer("_edge_index", self.edge_index)
        if self.edge_weight is not None:
            self.register_buffer("_edge_weight", self.edge_weight)
        else:
            self._edge_weight = None

        # Build prediction head
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

        n_backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        n_head_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model] Backbone trainable params: {n_backbone_trainable:,}")
        print(f"[Model] Head trainable params: {n_head_trainable:,}")

    def _get_live_embeddings(self, node_indices: torch.Tensor, in_vocab: torch.Tensor) -> torch.Tensor:
        """Run live STRING_GNN forward pass and index into embeddings.

        Args:
            node_indices: [B] int64 - index into STRING_GNN node embeddings
            in_vocab: [B] bool - True if gene is in STRING_GNN vocabulary

        Returns:
            [B, 256] embeddings
        """
        # Run forward pass through (partially fine-tuned) backbone
        outputs = self.backbone(
            edge_index=self._edge_index,
            edge_weight=self._edge_weight if self._edge_weight is not None else None,
        )
        all_emb = outputs.last_hidden_state  # [N_nodes, 256]

        # Index into embeddings for each sample in the batch
        # For OOV genes (in_vocab=False), node_idx=0 (placeholder), will be masked by head
        gnn_emb = all_emb[node_indices]  # [B, 256]
        return gnn_emb

    def forward(
        self,
        node_indices: torch.Tensor,
        in_vocab: torch.Tensor,
    ) -> torch.Tensor:
        gnn_emb = self._get_live_embeddings(node_indices, in_vocab)
        return self.model(gnn_emb, in_vocab)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return class_weighted_focal_loss(
            logits, labels,
            class_weights=self.class_weights_buf,
            gamma=self.hparams.focal_gamma,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_idx"], batch["in_vocab"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
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
        with torch.no_grad():
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

            self.print(f"[Node1-2-2-3-1] Saved test predictions → {pred_path} ({len(seen_ids)} unique samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-2-2-3-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # ─── 3-Group MuonWithAuxAdam ──────────────────────────────────────────
        # Group 1 (Muon): 2D hidden weight matrices inside ResidualBlocks
        # Group 2 (AdamW, lr=5e-4, wd=4e-3): Head non-ResBlock params [STRONGER WD vs parent's 2e-3]
        # Group 3 (AdamW, lr=1e-5, wd=1e-3): STRING_GNN backbone fine-tuned params (mps.7, post_mp)

        muon_params = []
        adamw_head_params = []
        adamw_backbone_params = []

        # Head parameter grouping
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Muon: 2D weight matrices in ResidualBlock hidden layers
            if "res_blocks" in name and param.ndim >= 2:
                muon_params.append(param)
            else:
                # AdamW: input_norm, oov_embedding, proj_in, norm_out, proj_bilinear,
                #        out_gene_emb, all biases, all 1D params
                adamw_head_params.append(param)

        # Backbone parameter grouping (low LR, no Muon per skill: not a hidden matrix)
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                adamw_backbone_params.append(param)

        param_groups = [
            # Group 1: Muon for ResBlock hidden 2D matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,         # 0.005 (higher than AdamW, per Muon skill)
                momentum=0.95,
                weight_decay=0.0,      # No WD for Muon (orthogonal momentum handles it)
            ),
            # Group 2: AdamW for head non-ResBlock params
            # CHANGED: weight_decay=4e-3 (vs 2e-3 in parent) — stronger regularization
            # to suppress the progressive overfitting (val/train ratio 3.05→6.21 in parent)
            dict(
                params=adamw_head_params,
                use_muon=False,
                lr=hp.lr,              # 5e-4
                betas=(0.9, 0.95),
                weight_decay=hp.weight_decay,  # 4e-3 (INCREASED)
            ),
            # Group 3: AdamW for backbone fine-tuned layers (low LR to preserve pretrained)
            dict(
                params=adamw_backbone_params,
                use_muon=False,
                lr=hp.backbone_lr,     # 1e-5 (proven in node1-2-2-2-1)
                betas=(0.9, 0.95),
                weight_decay=hp.backbone_wd,   # 1e-3
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # ─── Progressive Cosine Warm Restarts with Linear Warmup ──────────────
        # T_0=600 steps (≈27 epochs/cycle), T_mult=1.5 (progressive cycle lengthening)
        #
        # Cycle lengths (steps → epochs at ~22 steps/epoch):
        #   Cycle 1: 600 steps ≈ 27 epochs  (rapid initial escape, matches T_0=600 node)
        #   Cycle 2: 900 steps ≈ 41 epochs  (slightly deeper)
        #   Cycle 3: 1350 steps ≈ 61 epochs  (deeper convergence)
        #   Cycle 4: 2025 steps ≈ 92 epochs  (deep final convergence)
        # Total for 4 cycles: 4875 steps ≈ 222 epochs → well within 350 max_epochs budget
        #
        # Rationale vs parent (T_0=1200, T_mult=1):
        # - Parent: 3 cycles of 55 epochs each → insufficient restart frequency
        # - This node: 4+ cycles with progressive deepening → more escape events + deeper minima
        # - T_mult=1.5 combines the staircase frequency of T_0=600 with progressive depth
        #
        # Implementation: LambdaLR with progressive cycle tracking

        t0 = hp.t0_steps        # 600
        t_mult = hp.t_mult      # 1.5
        warmup = hp.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                # Linear warmup
                return float(step) / max(1, warmup)
            # After warmup: progressive cosine warm restarts
            t = step - warmup
            # Find current cycle and position within it
            t_cur = float(t)
            t_i = float(t0)
            while t_cur >= t_i:
                t_cur -= t_i
                t_i = t_i * t_mult
            # t_cur: steps into current cycle, t_i: current cycle length
            progress = t_cur / max(1.0, t_i)
            # Cosine annealing: 1.0 → 0 with 1% floor
            return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

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
        # Collect trainable param names from both model (head) and backbone
        trainable_keys = set()
        for n, p in self.named_parameters():
            if p.requires_grad:
                trainable_keys.add(prefix + n)
        buffer_keys = {prefix + n for n, _ in self.named_buffers()}
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
    p = argparse.ArgumentParser(description="Node 1-2-2-3-1 – STRING_GNN Partial FT + Rank-512 + Muon + Progressive Warm Restarts (T_mult=1.5)")
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--gnn-dim",            type=int,   default=256)
    p.add_argument("--hidden-dim",         type=int,   default=512)
    p.add_argument("--rank",               type=int,   default=512,
                   help="Bilinear rank=512 (proven in node1-2-2-2, F1=0.5060).")
    p.add_argument("--n-residual-layers",  type=int,   default=6)
    p.add_argument("--dropout",            type=float, default=0.30,
                   help="Dropout=0.30 (validated in node1-2-2-2-1 and parent node1-2-2-3).")
    p.add_argument("--lr",                 type=float, default=5e-4,
                   help="AdamW LR for head non-ResBlock parameters.")
    p.add_argument("--muon-lr",            type=float, default=0.005,
                   help="Muon LR for ResBlock 2D hidden weight matrices (proven at 0.005).")
    p.add_argument("--backbone-lr",        type=float, default=1e-5,
                   help="AdamW LR for backbone fine-tuned layers (mps.7+post_mp). "
                        "Proven in node1-2-2-2-1: +0.0039 F1 over frozen backbone.")
    p.add_argument("--weight-decay",       type=float, default=4e-3,
                   help="AdamW weight decay for head parameters. INCREASED to 4e-3 (vs 2e-3 in "
                        "parent) to suppress progressive overfitting (val/train ratio grew 3x→6x "
                        "in parent node1-2-2-3).")
    p.add_argument("--backbone-wd",        type=float, default=1e-3,
                   help="AdamW weight decay for backbone parameters.")
    p.add_argument("--focal-gamma",        type=float, default=2.0,
                   help="Focal loss gamma=2.0 (proven optimal in node1-2, node2-1-3).")
    p.add_argument("--class-weights",      type=float, nargs=3, default=[2.0, 0.5, 4.0],
                   help="Class weights [down, neutral, up] = [2.0, 0.5, 4.0] (proven tree optimum).")
    p.add_argument("--warmup-steps",       type=int,   default=50,
                   help="Linear warmup steps before cosine schedule starts.")
    p.add_argument("--t0-steps",           type=int,   default=600,
                   help="Initial steps per cosine warm restart cycle. T_0=600 (≈27 epochs/cycle). "
                        "Shorter than parent's T_0=1200 to enable more restart events.")
    p.add_argument("--t-mult",             type=float, default=1.5,
                   help="Progressive cycle multiplier (T_mult=1.5). Each subsequent cycle is "
                        "1.5× longer than the previous, combining T_0=600 staircase frequency "
                        "with progressively deeper convergence in later cycles.")
    p.add_argument("--micro-batch-size",   type=int,   default=16)
    p.add_argument("--global-batch-size",  type=int,   default=64)
    p.add_argument("--max-epochs",         type=int,   default=350,
                   help="Max epochs. With T_0=600, T_mult=1.5: ~4 full cycles in 222 epochs, "
                        "350 epochs budget allows ~4-5 cycles total.")
    p.add_argument("--patience",           type=int,   default=100,
                   help="EarlyStopping patience=100. Covers 1-2 full cycles at cycle 3-4 lengths "
                        "(61-92 epochs/cycle), while still stopping if staircase plateaus.")
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
        backbone_wd=args.backbone_wd,
        focal_gamma=args.focal_gamma,
        class_weights=args.class_weights,
        warmup_steps=args.warmup_steps,
        t0_steps=args.t0_steps,
        t_mult=args.t_mult,
        backbone_layers=["mps.7", "post_mp"],
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
            f"Node 1-2-2-3-1 – STRING_GNN Partial FT + Rank-512 + Muon + Progressive Warm Restarts (T_mult=1.5)\n"
            f"Test results from trainer: {test_results}\n"
            f"Backbone layers fine-tuned: mps.7 + post_mp (backbone_lr={args.backbone_lr})\n"
            f"Bilinear rank: {args.rank}\n"
            f"Muon LR: {args.muon_lr}, AdamW LR: {args.lr}, Backbone LR: {args.backbone_lr}\n"
            f"Class weights: {args.class_weights}, Focal gamma: {args.focal_gamma}\n"
            f"Dropout: {args.dropout}, Weight decay: {args.weight_decay} (head), {args.backbone_wd} (backbone)\n"
            f"T0 steps (initial cycle): {args.t0_steps}, T_mult: {args.t_mult}, Patience: {args.patience}\n"
        )


if __name__ == "__main__":
    main()
