"""
Node 1-2-2-2-1-2: Stronger Regularization + Per-Group LR Eta_Min Fix + Longer Patience
                    (Preserve Parent's Proven T_0=600 Staircase + Enhance Overfitting Control)

Key improvements over parent (node1-2-2-2-1, test F1=0.5099, tree best):

1. Increase dropout 0.30 → 0.35 (HIGH PRIORITY)
   The parent's val/train loss ratio reached 4.77× by epoch 220, with the ascending
   staircase stopping after cycle 6. Stronger dropout delays overfitting onset, potentially
   allowing cycles 7-9 to also show improvement and pushing beyond the epoch-170 peak.
   Evidence: node1-2-2-3-1-1 used dropout=0.40 and still achieved F1=0.5088, confirming
   higher dropout does not break the architecture.

2. Increase head weight decay 2e-3 → 3e-3 (MEDIUM PRIORITY)
   Complementary regularization to dropout. Multiple nodes (node2-1-3, node1-2-2-3-2) showed
   improved generalization with WD in the 2e-3–3e-3 range. Combined with dropout=0.35, this
   should reduce the overfitting rate to ~3.0–3.5× (vs parent's 4.77×).

3. Fix per-group LR schedule eta_min (TECHNICAL FIX)
   Parent bug: a single LambdaLR scales all groups uniformly with eta_min_frac=0.02.
   This gives backbone effective eta_min = 1e-5 × 0.02 = 2e-7 (essentially zero — backbone
   effectively freezes at cycle minima). Fix: per-group lambdas with group-specific eta_min:
   - Backbone AdamW: peak=1e-5, eta_min=1e-6 (10% floor, keeps backbone fine-tuning active)
   - Head Muon:      peak=0.005, eta_min=5e-5 (1% floor, standard for Muon)
   - Head AdamW:     peak=5e-4, eta_min=5e-6 (1% floor, same ratio as parent)
   This ensures the backbone continues adapting throughout all restart cycles.

4. patience=100, max_epochs=400 (EXTENDED TRAINING)
   Parent stopped at epoch 251 (best at epoch 170, patience=80). With stronger regularization
   delaying the overfitting ceiling, staircase cycles 7-9 may be productive. 400 epochs
   allows ~9-10 cycles (at T_0=600 steps, ~22 steps/epoch → ~27 epochs/cycle). patience=100
   gives the extended training budget time to find new peaks in later cycles.

5. T_0=600, T_mult=1 PRESERVED (parent-proven ascending staircase mechanism)
   Siblings that changed T_0/T_mult regressed:
   - Sibling (node1-2-2-2-1-1): T_0=1400, T_mult=2 → F1=0.5017 (only 2 cycles, killed staircase)
   - node1-2-2-2-1-1-1: T_0=900, T_mult=1 → F1=0.5062 (declining staircase, suboptimal freq)
   T_0=600 gives ~9-10 cycles in 400 epochs, matching the parent's proven restart frequency.

Architecture (unchanged from parent):
  STRING_GNN (partially fine-tuned):
    Frozen: emb, mps.0-mps.6 (precomputed once to h6_all_buf)
    Trainable: mps.7, post_mp (~67K params, AdamW lr=1e-5)
  GNNBilinearHead:
    OOV handling → LayerNorm → Linear(256→512) → 6×ResBlock(512, expand=4, dropout=0.35)
    → LayerNorm → Dropout → Linear(512→3×512) → einsum([B,3,512], [6640,512]) → [B,3,6640]
  Loss: Class-weighted focal (gamma=2.0, weights=[2.0, 0.5, 4.0])

Memory influences:
  - node1-2-2-2-1 feedback: "Bottleneck: progressive overfitting (4.77× ratio). T_0=600
    staircase works (6 cycles: 0.488→0.510) but stops after cycle 6 due to overfitting."
  - node1-2-2-2-1-1 (sibling): T_mult=2 + mps.6 + LS=0.07 → 0.5017 — confirmed T_0=600
    must be preserved; extended backbone FT and label smoothing both harmful.
  - node1-2-2-2-1-1-1 (from memory): T_0=900 + LS=0.03 → 0.5062 — declining staircase;
    confirms T_0=600 (parent frequency) is optimal, not longer cycles.
  - node1-2-2-3-1-1: dropout=0.40 achieved F1=0.5088 — confirms higher dropout doesn't break
    the architecture; 0.35 is a moderate/safe increase.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # required for deterministic=True with CUDA >= 10.2

import json
import math
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


# ─── STRING_GNN Precomputation ────────────────────────────────────────────────

def precompute_frozen_h6(
    model_dir: Path,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, int], torch.Tensor, Optional[torch.Tensor]]:
    """Precompute frozen STRING_GNN embeddings up to mps.6 (layers 0-6 inclusive).

    Returns:
        h6_np: [N_nodes, 256] numpy array of intermediate embeddings
        node_name_to_idx: mapping gene Ensembl ID → node index
        edge_index: [2, E] long tensor (for use in trainable mps.7 forward pass)
        edge_weight: [E] float tensor or None
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

    # Run forward through mps.0 to mps.6 (7 layers, indices 0-6)
    with torch.no_grad():
        h = model.emb.weight.float()  # [N, 256] initial node embeddings
        for i in range(7):  # layers 0, 1, 2, 3, 4, 5, 6
            h = model.mps[i](h, edge_index, edge_weight)
    # h is now output of mps.6, before mps.7

    h6_np = h.float().cpu().numpy()  # [N_nodes, 256]
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

    # Keep edge_index and edge_weight on CPU for later use
    edge_index_cpu = graph["edge_index"].cpu()
    edge_weight_cpu = (graph["edge_weight"].cpu()
                       if graph.get("edge_weight") is not None else None)

    del model
    torch.cuda.empty_cache()

    return h6_np, node_name_to_idx, edge_index_cpu, edge_weight_cpu


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset.

    Stores precomputed intermediate STRING_GNN embeddings (after mps.0-6).
    The trainable mps.7 + post_mp forward pass is applied in the LightningModule.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        h6_embeddings: np.ndarray,        # [N_nodes, 256] after mps.0-6
        node_name_to_idx: Dict[str, int],
        embed_dim: int = 256,
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.has_labels = has_labels

        n_samples = len(df)
        embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)
        in_vocab = []
        node_indices = []  # node index in STRING_GNN graph (-1 for OOV)

        for i, pert_id in enumerate(self.pert_ids):
            if pert_id in node_name_to_idx:
                node_idx = node_name_to_idx[pert_id]
                embeddings[i] = h6_embeddings[node_idx]
                in_vocab.append(True)
                node_indices.append(node_idx)
            else:
                # OOV: leave embedding as zeros; -1 indicates no graph node
                in_vocab.append(False)
                node_indices.append(-1)

        self.embeddings = torch.from_numpy(embeddings)  # [N, 256]
        self.in_vocab = torch.tensor(in_vocab, dtype=torch.bool)  # [N]
        self.node_indices = torch.tensor(node_indices, dtype=torch.long)  # [N]

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
            "h6_emb": self.embeddings[idx],     # [256] after mps.0-6
            "in_vocab": self.in_vocab[idx],      # bool
            "node_idx": self.node_indices[idx],  # int or -1
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[DataModule] Precomputing frozen STRING_GNN embeddings (mps.0-6)...")
        (
            self.h6_embeddings,
            self.node_name_to_idx,
            self.edge_index,
            self.edge_weight,
        ) = precompute_frozen_h6(STRING_GNN_DIR, device)
        print(f"[DataModule] h6_embeddings shape: {self.h6_embeddings.shape}")

        # Load all splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        print(f"[DataModule] Coverage: "
              f"{sum(p in self.node_name_to_idx for p in dfs['train']['pert_id'])} / "
              f"{len(dfs['train'])} train genes in STRING_GNN")

        embed_dim = self.h6_embeddings.shape[1]
        self.train_ds = PerturbationDataset(dfs["train"], self.h6_embeddings, self.node_name_to_idx, embed_dim, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   self.h6_embeddings, self.node_name_to_idx, embed_dim, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  self.h6_embeddings, self.node_name_to_idx, embed_dim, True)

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


# ─── Backbone: Trainable STRING_GNN Tail ─────────────────────────────────────

class StringGNNTail(nn.Module):
    """Trainable tail of STRING_GNN: mps.7 + post_mp.

    Initialized from the pretrained STRING_GNN weights, then fine-tuned at low LR.
    Applies to the full graph (all N_nodes) using precomputed h6 embeddings as input.

    Forward:
        h6_all: [N_nodes, 256] frozen embeddings from mps.0-6
        edge_index: [2, E] graph edges
        edge_weight: [E] or None
    Returns:
        adapted: [N_nodes, 256] after mps.7 + post_mp
    """

    def __init__(self, model_dir: Path):
        super().__init__()
        # Load full model to extract mps.7 and post_mp
        full_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
        # Copy the modules we need (they become part of this module's parameters)
        self.gnn_layer7 = full_model.mps[7]   # GNNLayer (trainable)
        self.post_mp = full_model.post_mp       # Linear(256, 256) (trainable)
        del full_model
        torch.cuda.empty_cache()

    def forward(
        self,
        h6_all: torch.Tensor,            # [N, 256]
        edge_index: torch.Tensor,         # [2, E]
        edge_weight: Optional[torch.Tensor],  # [E] or None
    ) -> torch.Tensor:
        h7 = self.gnn_layer7(h6_all, edge_index, edge_weight)  # [N, 256]
        out = self.post_mp(h7)                                   # [N, 256]
        return out


# ─── Model ────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout + skip."""

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
    """Prediction head using adapted STRING_GNN embeddings as input."""

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

        self.oov_embedding = nn.Embedding(1, gnn_dim)
        self.input_norm = nn.LayerNorm(gnn_dim)
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank)
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, rank))
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
        gnn_emb: torch.Tensor,   # [B, 256] adapted embeddings
        in_vocab: torch.Tensor,  # [B] bool mask
    ) -> torch.Tensor:
        B = gnn_emb.shape[0]

        # OOV handling
        oov_emb = self.oov_embedding(torch.zeros(B, dtype=torch.long, device=gnn_emb.device))
        in_vocab_f = in_vocab.unsqueeze(1).float()  # [B, 1]
        x = gnn_emb * in_vocab_f + oov_emb * (1.0 - in_vocab_f)  # [B, gnn_dim]

        # Input normalization + projection
        x = self.input_norm(x)
        x = self.proj_in(x)  # [B, hidden_dim]

        # Deep residual MLP
        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)

        # Bilinear interaction head
        x = self.head_dropout(x)
        pert_proj = self.proj_bilinear(x)                    # [B, n_classes * rank]
        pert_proj = pert_proj.view(B, self.n_classes, self.rank)  # [B, 3, rank]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, 6640]
        return logits


# ─── Loss ─────────────────────────────────────────────────────────────────────

def class_weighted_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Class-weighted focal loss. Proven optimal: gamma=2.0, weights=[2.0, 0.5, 4.0]."""
    B, C, G = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
    labels_flat = labels.reshape(-1)                        # [B*G]

    ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction="none")  # [B*G]

    with torch.no_grad():
        probs = F.softmax(logits_flat, dim=1)
        pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt).pow(gamma)

    sample_class_weight = class_weights[labels_flat]
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


# ─── Per-Group LR Lambda Factory ─────────────────────────────────────────────

def make_warmup_cosine_restart_lambda(
    peak_lr: float,
    eta_min: float,
    warmup_steps: int,
    T_0: int,
) -> callable:
    """Create a per-group LR lambda for cosine warm restarts with correct group-specific eta_min.

    This fixes the parent's implementation where a single shared LambdaLR used
    eta_min_frac = head_eta_min / head_lr, causing the backbone's effective eta_min
    to be negligibly small (2e-7 instead of 1e-6). Each group now gets its own
    eta_min fraction computed relative to its own peak LR.

    Args:
        peak_lr: Peak LR for this parameter group
        eta_min: Absolute minimum LR floor at cosine minima
        warmup_steps: Number of linear warmup steps
        T_0: Cosine restart period in steps

    Returns:
        A lambda(step) -> float multiplier (relative to peak_lr)
    """
    eta_min_frac = eta_min / max(peak_lr, 1e-12)  # fraction of peak LR

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        step_adj = step - warmup_steps
        cycle_step = step_adj % T_0
        progress = float(cycle_step) / float(max(1, T_0))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_frac + (1.0 - eta_min_frac) * cosine_factor

    return lr_lambda


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """LightningModule for perturbation DEG prediction with partial STRING_GNN fine-tuning.

    Key improvements over parent:
    - dropout=0.35 (vs 0.30) for stronger regularization
    - head weight_decay=3e-3 (vs 2e-3) for additional regularization
    - Per-group LR lambdas with correct eta_min per group (backbone: 1e-6, head: 5e-6)
    - patience=100, max_epochs=400 for extended staircase exploration

    Architecture:
    1. StringGNNTail (trainable, mps.7+post_mp): applied globally to h6_all buffer
    2. GNNBilinearHead: 6x ResBlocks(dropout=0.35) + bilinear → logits [B, 3, 6640]
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_residual_layers: int = 6,
        dropout: float = 0.35,
        backbone_lr: float = 1e-5,
        backbone_wd: float = 1e-4,
        backbone_eta_min: float = 1e-6,
        lr: float = 5e-4,
        lr_eta_min: float = 5e-6,
        muon_lr: float = 0.005,
        muon_eta_min: float = 5e-5,
        weight_decay: float = 3e-3,
        focal_gamma: float = 2.0,
        class_weights: List[float] = None,
        warmup_steps: int = 50,
        cosine_t0: int = 600,
        # The following are passed at runtime (not saved as hparams)
        h6_embeddings_np: Optional[np.ndarray] = None,
        edge_index_cpu: Optional[torch.Tensor] = None,
        edge_weight_cpu: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if class_weights is None:
            class_weights = [2.0, 0.5, 4.0]

        # Save only the hyperparameters (not the large numpy arrays)
        self.save_hyperparameters(ignore=["h6_embeddings_np", "edge_index_cpu", "edge_weight_cpu"])
        self._class_weights_list = class_weights

        # Store graph data for setup() — will be registered as buffers there
        self._h6_embeddings_np = h6_embeddings_np
        self._edge_index_cpu = edge_index_cpu
        self._edge_weight_cpu = edge_weight_cpu

        # These will be built in setup()
        self.gnn_tail = None
        self.model = None

        # Accumulation buffers
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        if self.model is not None:
            return  # Already initialized

        hp = self.hparams

        # ── Build trainable STRING_GNN tail (mps.7 + post_mp) ───────────────
        self.gnn_tail = StringGNNTail(STRING_GNN_DIR)

        # Ensure float32 for stable optimization
        for p in self.gnn_tail.parameters():
            p.data = p.data.float()

        # ── Register frozen graph data as buffers ────────────────────────────
        if self._h6_embeddings_np is not None:
            h6_tensor = torch.from_numpy(self._h6_embeddings_np).float()
            self.register_buffer("h6_all_buf", h6_tensor)
        if self._edge_index_cpu is not None:
            self.register_buffer("edge_index_buf", self._edge_index_cpu.long())
        if self._edge_weight_cpu is not None:
            self.register_buffer("edge_weight_buf", self._edge_weight_cpu.float())
        else:
            self.register_buffer("edge_weight_buf", None)

        # ── Build prediction head ────────────────────────────────────────────
        self.model = GNNBilinearHead(
            gnn_dim=hp.gnn_dim,
            hidden_dim=hp.hidden_dim,
            rank=hp.rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
        )

        # Cast all trainable head params to float32
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Register class weights buffer
        self.register_buffer(
            "class_weights_buf",
            torch.tensor(self._class_weights_list, dtype=torch.float32)
        )

        # Log parameter counts
        backbone_trainable = sum(p.numel() for p in self.gnn_tail.parameters())
        head_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print(f"[Setup] Trainable backbone params (mps.7+post_mp): {backbone_trainable:,}")
        self.print(f"[Setup] Trainable head params: {head_trainable:,}")
        self.print(f"[Setup] Total trainable: {backbone_trainable + head_trainable:,}")

    def _apply_backbone_and_gather(
        self,
        node_idx: torch.Tensor,   # [B] long, -1 for OOV
        in_vocab: torch.Tensor,   # [B] bool
    ) -> torch.Tensor:
        """Apply trainable gnn_tail globally, gather batch embeddings."""
        h6_all = self.h6_all_buf          # [N, 256]
        ei = self.edge_index_buf           # [2, E]
        ew = self.edge_weight_buf          # [E] or None

        # Single GCN forward: sparse matmul on ~786K edges — cheap
        adapted_all = self.gnn_tail(h6_all, ei, ew)  # [N, 256]

        # Gather batch samples by node index
        B = node_idx.shape[0]
        batch_emb = torch.zeros(B, adapted_all.shape[1], device=adapted_all.device, dtype=adapted_all.dtype)

        # Only gather embeddings for in-vocab samples (valid node indices)
        in_vocab_and_valid = in_vocab & (node_idx >= 0)
        if in_vocab_and_valid.any():
            valid_node_idx = node_idx[in_vocab_and_valid]
            batch_emb[in_vocab_and_valid] = adapted_all[valid_node_idx]

        return batch_emb

    def forward(
        self,
        h6_emb: torch.Tensor,    # [B, 256] (precomputed, used for OOV only)
        in_vocab: torch.Tensor,  # [B] bool
        node_idx: torch.Tensor,  # [B] long
    ) -> torch.Tensor:
        """Full forward pass: backbone adaptation + prediction head."""
        adapted_emb = self._apply_backbone_and_gather(node_idx, in_vocab)  # [B, 256]
        return self.model(adapted_emb, in_vocab)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return class_weighted_focal_loss(
            logits, labels,
            class_weights=self.class_weights_buf,
            gamma=self.hparams.focal_gamma,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["h6_emb"].float(), batch["in_vocab"], batch["node_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["h6_emb"].float(), batch["in_vocab"], batch["node_idx"])
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
        logits = self(batch["h6_emb"].float(), batch["in_vocab"], batch["node_idx"])
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
            self.print(f"[Node1-2-2-2-1-2] Saved test predictions → {pred_path} ({len(seen_ids)} unique samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-2-2-2-1-2] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # ─── 3-Group Optimizer ────────────────────────────────────────────────
        # Group 0 (backbone AdamW): trainable STRING_GNN tail (mps.7 + post_mp)
        # Group 1 (head Muon): 2D hidden weight matrices in ResidualBlocks
        # Group 2 (head AdamW): all other head parameters

        backbone_params = list(self.gnn_tail.parameters())

        muon_params = []
        adamw_head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "res_blocks" in name and param.ndim >= 2:
                muon_params.append(param)
            else:
                adamw_head_params.append(param)

        param_groups = [
            # Group 0: Backbone fine-tuning (AdamW, very small LR)
            dict(
                params=backbone_params,
                use_muon=False,
                lr=hp.backbone_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.backbone_wd,
            ),
            # Group 1: Head hidden 2D matrices (Muon)
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                momentum=0.95,
                weight_decay=0.0,
            ),
            # Group 2: Head other params (AdamW)
            dict(
                params=adamw_head_params,
                use_muon=False,
                lr=hp.lr,
                betas=(0.9, 0.95),
                weight_decay=hp.weight_decay,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # ─── Per-Group Cosine Warm Restarts with Correct Eta_Min ─────────────
        # KEY FIX vs parent: each group has its own lambda with group-specific eta_min.
        # Parent used a single lambda with eta_min_frac = head_eta_min / head_lr = 0.02,
        # which made backbone effective eta_min = 1e-5 × 0.02 = 2e-7 (effectively zero).
        # Fix: backbone eta_min = 1e-6 (10% of peak), Muon eta_min = 5e-5 (1%),
        #      head AdamW eta_min = 5e-6 (1%).

        warmup = hp.warmup_steps
        T_0 = hp.cosine_t0

        # Per-group LR lambdas
        backbone_lambda = make_warmup_cosine_restart_lambda(
            peak_lr=hp.backbone_lr,
            eta_min=hp.backbone_eta_min,
            warmup_steps=warmup,
            T_0=T_0,
        )
        muon_lambda = make_warmup_cosine_restart_lambda(
            peak_lr=hp.muon_lr,
            eta_min=hp.muon_eta_min,
            warmup_steps=warmup,
            T_0=T_0,
        )
        head_lambda = make_warmup_cosine_restart_lambda(
            peak_lr=hp.lr,
            eta_min=hp.lr_eta_min,
            warmup_steps=warmup,
            T_0=T_0,
        )

        # LambdaLR with list of lambdas: one per parameter group
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[backbone_lambda, muon_lambda, head_lambda]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable params + buffers ───────────────────

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
        description="Node 1-2-2-2-1-2 – Stronger Regularization + Per-Group LR Fix + Extended Patience"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--gnn-dim",            type=int,   default=256)
    p.add_argument("--hidden-dim",         type=int,   default=512)
    p.add_argument("--rank",               type=int,   default=512)
    p.add_argument("--n-residual-layers",  type=int,   default=6)
    p.add_argument("--dropout",            type=float, default=0.35,
                   help="Dropout increased from 0.30 (parent) to 0.35 for stronger regularization.")
    p.add_argument("--backbone-lr",        type=float, default=1e-5,
                   help="LR for trainable STRING_GNN layers (mps.7+post_mp). Conservative.")
    p.add_argument("--backbone-wd",        type=float, default=1e-4,
                   help="Weight decay for backbone. Light to prevent catastrophic forgetting.")
    p.add_argument("--backbone-eta-min",   type=float, default=1e-6,
                   help="Absolute eta_min for backbone at cosine minima (10%% of peak 1e-5).")
    p.add_argument("--lr",                 type=float, default=5e-4,
                   help="AdamW LR for non-Muon head parameters.")
    p.add_argument("--lr-eta-min",         type=float, default=5e-6,
                   help="Absolute eta_min for head AdamW at cosine minima (1%% of peak 5e-4).")
    p.add_argument("--muon-lr",            type=float, default=0.005,
                   help="Muon LR for ResidualBlock 2D hidden weight matrices.")
    p.add_argument("--muon-eta-min",       type=float, default=5e-5,
                   help="Absolute eta_min for Muon group at cosine minima (1%% of peak 0.005).")
    p.add_argument("--weight-decay",       type=float, default=3e-3,
                   help="Head AdamW weight decay. Increased from 2e-3 (parent) for regularization.")
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--class-weights",      type=float, nargs=3,
                   default=[2.0, 0.5, 4.0],
                   metavar=("DOWN_W", "NEUTRAL_W", "UP_W"))
    p.add_argument("--warmup-steps",       type=int,   default=50)
    p.add_argument("--cosine-t0",          type=int,   default=600,
                   help="Cosine restart period T_0 in steps. Preserved at 600 (parent-proven "
                        "staircase frequency: ~27 epochs/cycle, 9-10 cycles in 400 epochs).")
    p.add_argument("--micro-batch-size",   type=int,   default=16)
    p.add_argument("--global-batch-size",  type=int,   default=64)
    p.add_argument("--max-epochs",         type=int,   default=400,
                   help="Extended from 300 to 400 to allow 9-10 restart cycles.")
    p.add_argument("--patience",           type=int,   default=100,
                   help="Extended from 80 to 100 to capture potential late staircase peaks.")
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

    # DataModule: precomputes frozen h6 embeddings and loads datasets
    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()  # Run setup to get h6_embeddings, edge_index, edge_weight

    # LightningModule: pass graph data so setup() can register them as buffers
    lit = PerturbationLitModule(
        gnn_dim=args.gnn_dim,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        n_residual_layers=args.n_residual_layers,
        dropout=args.dropout,
        backbone_lr=args.backbone_lr,
        backbone_wd=args.backbone_wd,
        backbone_eta_min=args.backbone_eta_min,
        lr=args.lr,
        lr_eta_min=args.lr_eta_min,
        muon_lr=args.muon_lr,
        muon_eta_min=args.muon_eta_min,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        class_weights=args.class_weights,
        warmup_steps=args.warmup_steps,
        cosine_t0=args.cosine_t0,
        # Pass graph data for setup() to register as buffers
        h6_embeddings_np=dm.h6_embeddings,
        edge_index_cpu=dm.edge_index,
        edge_weight_cpu=dm.edge_weight,
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
        with open(score_path, "w") as f:
            f.write(f"test_results: {test_results}\n")
            if test_results:
                for k, v in test_results[0].items():
                    f.write(f"{k}: {v}\n")
        print(f"[Main] Test results saved to {score_path}")


if __name__ == "__main__":
    main()
