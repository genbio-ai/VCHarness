"""
Node 1-1-2-1-1-1-1-2 – STRING_GNN Deeper Partial Fine-tuning (mps.6+7+post_mp) + Single Cosine Cycle

Architecture:
  - STRING_GNN backbone (18,870 nodes, 256-dim PPI topology embeddings)
    * EXTENDED PARTIAL FINE-TUNING: last 2 GNN layers (mps.6 + mps.7) and post_mp at LR=1e-5
    * All earlier layers remain frozen (embeddings + mps.0–5)
    * Precomputed frozen intermediate up to mps.5 output; live forward for mps.6+mps.7+post_mp
  - Deep 6-layer residual bilinear MLP head (hidden_dim=512, expand=4, rank=512, dropout=0.3)
  - Bilinear interaction: pert_emb [B, 256] → head [B, 512] → [B, 3*512] → [B, 3, 512]
                          × out_gene_emb [6640, 512] → logits [B, 3, 6640]
  - Class-weighted focal cross-entropy loss (gamma=2.0, weights=[2.0(down), 0.5(neutral), 4.0(up)])
  - Two-group optimizer:
    * Muon (lr=0.005): hidden MLP weight matrices in ResidualBlocks
    * AdamW (lr=5e-4, wd=2e-3): head input/output layers, embeddings, norms, biases
    * AdamW (lr=1e-5, wd=1e-4): unfrozen GNN layers (mps.6, mps.7, post_mp) + OOV emb
  - Single cosine annealing schedule (NO warm restarts), T_max calibrated to ~80 epochs
    → Empirical evidence from parent (best epoch=69) shows the model peaks in the first 70-80
      epochs; warm restarts did not help and caused disruption. Use a single cycle targeting the
      ~70-80 epoch convergence window.
  - Gradient clipping (max_norm=1.0)
  - Learnable OOV embedding

Key improvements over Parent (Node 1-1-2-1-1-1-1, F1=0.5035 — tree best):

  1. Extend partial backbone fine-tuning from 2 to 3 modules (mps.6 + mps.7 + post_mp):
     Parent feedback explicitly identified that only 132K backbone params (2 modules) were
     insufficient to break the representational ceiling. Adding mps.6 doubles the backbone
     adaptation capacity (~198K backbone trainable params), directly following the STRING_GNN
     skill documentation's recommended partial fine-tuning strategy:
     "freeze emb table, tune mps.6.*, mps.7.*, post_mp.*"
     node2-1-2-1 (F1=0.5016): used last 2 layers + post_mp at LR=1e-5 (our reference).
     This node adds mps.6 to that configuration (one more layer, higher capacity).

  2. Remove warm restarts; use single cosine cycle with T_max=80 epochs:
     Parent feedback: "The warm restart caused disruption and zero improvement over the first
     cycle's best (epoch 69). A single cosine cycle with no restart would prevent this degradation."
     The parent's best epoch was 69, corresponding to the first ~70 epochs. Setting T_max=80
     (slightly beyond the observed best) gives the model just enough time to converge without
     the unproductive tail of a 130-epoch cycle. Single cosine with hard clamp at progress=1.0
     (eta_min=0) is equivalent to the parent's "accidental" single-cycle variant.

  3. Reduce max_epochs and patience to match observed best-epoch window:
     Parent ran to epoch 189 with best at epoch 69 (wasted 120 epochs of compute).
     This node sets max_epochs=150, patience=80 — enough budget for the model to converge
     within the 70-80 epoch window and allow some buffer for later improvements.
     The patience=80 is calibrated to detect no improvement over 80 epochs (since the model
     consistently plateaus after ~70–80 epochs based on parent training dynamics).

  4. Retained proven configuration from parent:
     - dropout=0.3, wd=2e-3, rank=512 (all validated in parent)
     - muon_lr=0.005, AdamW lr=5e-4 (validated optimal)
     - class_weights=[2.0, 0.5, 4.0] + gamma=2.0 + label_smoothing=0.05
     - 6 ResidualBlocks (proven optimal depth)
     - Learnable OOV embedding

Memory influences:
  - node1-1-2-1-1-1-1 feedback (parent, F1=0.5035):
    * PRIMARY: "extend partial fine-tuning to mps.6 + mps.7 + post_mp"
    * PRIMARY: "Remove warm restarts; single cosine cycle with T_0=70-80 epochs"
    * "Set max_epochs=150, patience=80, T_0=70"
  - STRING_GNN skill documentation: "freeze emb table, tune mps.6.*, mps.7.*, post_mp.*"
  - node2-1-2-1 (F1=0.5016): last 2 layers + post_mp at LR=1e-5 works well
  - Tree-wide pattern: Muon lr=0.005 is validated optimal
  - node1-2-3 (F1=0.4969): class weights [2.0, 0.5, 4.0] + gamma=2.0 proven best
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
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

# ─── Constants ────────────────────────────────────────────────────────────────

N_GENES_OUT = 6640
N_CLASSES = 3

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
STRING_GNN_DIM = 256      # STRING_GNN hidden dimension

# Derive the project root from this script's resolved location so that relative
# data paths work correctly in DDP where each rank may have a different CWD.
# Path(__file__).resolve() -> .../mcts/node1-xxx/main.py
# .parent.parent.parent   -> project root (containing data/ and mcts/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "data")


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
    class_weights: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Class-weighted focal cross-entropy loss.

    Applies per-class weights to address label imbalance (88.9% neutral, 8.1% down, 3.0% up).
    Weights reduce gradient dominance of neutral class and amplify minority class gradients.

    Class weights [down=2.0, neutral=0.5, up=4.0] + gamma=2.0 proven from node1-2-3 (F1=0.4969)
    and maintained through all subsequent improvements to the tree best.

    Args:
        logits:         [B, C, G] float32 – per-class logits
        targets:        [B, G]    long    – class indices 0..C-1
        class_weights:  [C]       float32 – per-class weight tensor
        gamma:          focusing parameter (0 = standard CE)
        label_smoothing: label smoothing epsilon

    Returns:
        Scalar loss.
    """
    # Compute standard cross-entropy with label smoothing (reduction='none')
    ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )  # [B, G]

    # Focal modulation: compute pt for focal weighting
    with torch.no_grad():
        log_probs = F.log_softmax(logits, dim=1)  # [B, C, G]
        probs = log_probs.exp()                    # [B, C, G]
        # Gather probability at the true class
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
        # Map pert_id (Ensembl gene ID) to STRING_GNN node index; -1 for unknown
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
        # Load STRING_GNN node names to build pert_id → node index mapping
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        # node_names[i] is the Ensembl gene ID (pert_id) for STRING_GNN node i
        self.pert_id_to_gnn_idx = {name: i for i, name in enumerate(node_names)}

        # Load splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        self.train_ds = PerturbationDataset(dfs["train"], self.pert_id_to_gnn_idx, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   self.pert_id_to_gnn_idx, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  self.pert_id_to_gnn_idx, True)

        # Log OOV statistics
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
    STRING_GNN-based perturbation predictor (256-dim input).

    Architecture (unchanged from parent):
        gnn_emb [B, 256] → proj_in [B, hidden_dim=512] → 6×ResidualBlock → [B, 512]
        → norm_out → proj_bilinear [B, 3 * rank=512] → reshape [B, 3, 512]
        × out_gene_emb [6640, 512] → logits [B, 3, 6640]
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

        # Deep residual MLP: 6 blocks (proven optimal depth from node1-2)
        self.resblocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
              for _ in range(n_resblocks)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear interaction head (rank=512)
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

        # Project and process through residual MLP
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
    """LightningModule for gene-perturbation DEG prediction.

    Key changes from parent (node1-1-2-1-1-1-1, F1=0.5035):
    1. Extended partial STRING_GNN fine-tuning: unfreeze mps.6 + mps.7 + post_mp at LR=1e-5
       - ~198K backbone trainable params (vs parent's 132K with only mps.7 + post_mp)
       - Follows STRING_GNN skill documentation: "freeze emb, tune mps.6.*, mps.7.*, post_mp.*"
       - Precomputed frozen intermediate through mps.5; live forward for mps.6, mps.7, post_mp
    2. Single cosine cycle (NO warm restarts), T_max=80 epochs
       - Parent evidence: warm restart at epoch 130 caused disruption (second cycle max=0.500
         vs best=0.5034 in first cycle). Single cosine eliminates this disruption.
       - T_max=80 epochs matches the observed best-epoch window (parent best at epoch 69)
    3. Reduced max_epochs=150, patience=80
       - Parent best at epoch 69, ran to 189 (wasted 120 compute epochs)
       - max_epochs=150 gives full budget; patience=80 detects convergence efficiently
    4. Learnable OOV embedding retained from parent
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
        backbone_lr: float = 1e-5,             # Very low LR for fine-tuned GNN layers
        weight_decay: float = 2e-3,
        backbone_weight_decay: float = 1e-4,   # Lighter WD for backbone
        warmup_steps: int = 100,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        class_weight_down: float = 2.0,
        class_weight_neutral: float = 0.5,
        class_weight_up: float = 4.0,
        cosine_max_steps: int = 880,           # T_max for single cosine (in steps)
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

        # ── Load STRING_GNN backbone ────────────────────────────────────────
        # KEY CHANGE: Extended partial fine-tuning to mps.6 + mps.7 + post_mp
        # Pre-compute frozen intermediate through mps.5 (output after mps.5 = hidden_states[6])
        # This is index 6 in the hidden_states tuple:
        #   hidden_states[0] = initial embedding table
        #   hidden_states[1] = after mps.0
        #   ...
        #   hidden_states[6] = after mps.5
        #   hidden_states[7] = after mps.6
        #   hidden_states[8] = after mps.7 (= last_hidden_state before post_mp)
        #
        # We freeze: emb + mps.0–5 → precompute once as frozen_intermediate
        # We fine-tune: mps.6 + mps.7 + post_mp at LR=1e-5

        gnn_model = AutoModel.from_pretrained(
            str(STRING_GNN_DIR), trust_remote_code=True
        )
        gnn_model.eval()

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
        self._edge_index = graph["edge_index"]
        self._edge_weight = graph.get("edge_weight", None)

        # ── Compute frozen intermediate embeddings (through mps.5) ──────────
        # We run the GNN forward pass with output_hidden_states=True and store
        # hidden_states[6] which is the output after mps.5 (before mps.6).
        # Rationale: we unfreeze mps.6, mps.7, post_mp, so the frozen intermediate
        # ends at mps.5 output.
        with torch.no_grad():
            outputs_frozen = gnn_model(
                edge_index=self._edge_index,
                edge_weight=self._edge_weight,
                output_hidden_states=True,
            )
        # hidden_states tuple structure:
        #   [0]: initial emb (before any GNN layer)
        #   [k+1]: after mps[k-1] (k=1..8)
        #   So hidden_states[6] = after mps.5 = before mps.6 (correct frozen intermediate)
        frozen_intermediate = outputs_frozen.hidden_states[6].detach()  # [18870, 256]
        self.register_buffer("frozen_intermediate", frozen_intermediate)

        # Also store full frozen embeddings for OOV initialization
        frozen_full_embs = outputs_frozen.last_hidden_state.detach()  # [18870, 256]
        self.register_buffer("frozen_full_embs", frozen_full_embs)

        # ── Extract the fine-tunable GNN layers ─────────────────────────────
        # EXTENDED vs parent: now unfreeze mps.6 AND mps.7 AND post_mp
        self.gnn_layer6 = gnn_model.mps[6]      # Second-to-last GNN layer (NEW)
        self.gnn_layer7 = gnn_model.mps[7]      # Last GNN layer
        self.gnn_post_mp = gnn_model.post_mp    # Final projection

        # Store edge_index and edge_weight as buffers for use during fine-tuning
        self.register_buffer("edge_index", self._edge_index)
        if self._edge_weight is not None:
            self.register_buffer("edge_weight", self._edge_weight)
        else:
            self.edge_weight = None

        # Cast fine-tunable backbone layers to float32 for stable optimization
        for param in self.gnn_layer6.parameters():
            param.data = param.data.float()
            param.requires_grad = True
        for param in self.gnn_layer7.parameters():
            param.data = param.data.float()
            param.requires_grad = True
        for param in self.gnn_post_mp.parameters():
            param.data = param.data.float()
            param.requires_grad = True

        # Set fine-tunable layers to train mode (they were set to eval by gnn_model.eval())
        self.gnn_layer6.train()
        self.gnn_layer7.train()
        self.gnn_post_mp.train()

        # ── Learnable OOV embedding ──────────────────────────────────────────
        # Initialized with mean of frozen full embeddings; learnable during training
        oov_init = frozen_full_embs.mean(dim=0)  # [256]
        self.oov_emb = nn.Parameter(oov_init.float(), requires_grad=True)

        # ── Register class weights as buffer ───────────────────────────────
        class_weights = torch.tensor(
            [hp.class_weight_down, hp.class_weight_neutral, hp.class_weight_up],
            dtype=torch.float32,
        )
        self.register_buffer("class_weights", class_weights)

        # ── Build prediction head ───────────────────────────────────────────
        self.model = GNNBilinearHead(
            gnn_dim=STRING_GNN_DIM,
            hidden_dim=hp.hidden_dim,
            n_resblocks=hp.n_resblocks,
            expand=hp.expand,
            dropout=hp.dropout,
            rank=hp.rank,
        )

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

    def _get_gnn_emb(self, gnn_idx: torch.Tensor) -> torch.Tensor:
        """Get STRING_GNN embeddings for each sample, applying fine-tuned layers.

        EXTENDED vs parent: run mps.6 + mps.7 + post_mp (3 fine-tunable modules) on the
        precomputed frozen intermediate embeddings (output of mps.0-5).

        For OOV genes: use the learnable OOV embedding.

        Args:
            gnn_idx: [B] long tensor of STRING_GNN node indices; -1 for OOV

        Returns:
            emb: [B, 256] float32 tensor
        """
        valid_mask = gnn_idx >= 0  # [B]

        # Full graph forward through fine-tunable layers (transductive GNN)
        # frozen_intermediate: [18870, 256] — output after mps.5
        x_inter = self.frozen_intermediate  # [N_nodes, 256]

        edge_idx = self.edge_index
        edge_wt = self.edge_weight if hasattr(self, 'edge_weight') and self.edge_weight is not None else None

        # GNNLayer forward does NOT include residual — outer loop adds it: x = mp(x, ...) + x
        # Apply mps.6 (GNNLayer: norm → conv → act → dropout, WITHOUT residual)
        x_after6 = self.gnn_layer6(x_inter, edge_idx, edge_wt) + x_inter  # [N_nodes, 256]

        # Apply mps.7 (same structure)
        x_after7 = self.gnn_layer7(x_after6, edge_idx, edge_wt) + x_after6  # [N_nodes, 256]

        # Apply post_mp projection
        x_final = self.gnn_post_mp(x_after7)  # [N_nodes, 256]

        # Index by gnn_idx to get per-sample embeddings
        safe_idx = gnn_idx.clone()
        safe_idx[~valid_mask] = 0

        emb = x_final[safe_idx]  # [B, 256]

        # Replace OOV entries with learnable OOV embedding
        if (~valid_mask).any():
            emb = emb.clone()
            oov_expanded = self.oov_emb.unsqueeze(0).expand(
                (~valid_mask).sum(), -1
            ).to(emb.dtype)
            emb[~valid_mask] = oov_expanded

        return emb.float()

    def forward(self, gnn_idx: torch.Tensor) -> torch.Tensor:
        emb = self._get_gnn_emb(gnn_idx)
        return self.model(emb)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return class_weighted_focal_loss(
            logits,
            labels,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
            label_smoothing=self.hparams.label_smoothing,
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
                f"[Node1-1-2-1-1-1-1-2] Saved test predictions → {pred_path} "
                f"({len(seen_ids)} unique samples)"
            )

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-1-2-1-1-1-1-2] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        """
        Three-group optimizer configuration:
        1. Muon (lr=0.005): hidden 2D weight matrices in ResidualBlocks
        2. AdamW (lr=5e-4, wd=2e-3): head input/output layers, embeddings, norms/biases
        3. AdamW (lr=1e-5, wd=1e-4): fine-tuned GNN layers (mps.6, mps.7, post_mp) + OOV emb

        CHANGE: Extended backbone group to include mps.6 (was only mps.7 + post_mp in parent).
        CHANGE: Single cosine annealing (NO warm restarts). T_max=cosine_max_steps.
                Warm restarts in the parent caused disruption; single cycle is empirically better.
        """
        hp = self.hparams

        # ── Classify head parameters ────────────────────────────────────────
        muon_params = []
        adamw_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_resblock_linear_weight = (
                "resblocks" in name
                and "weight" in name
                and param.ndim >= 2
                and "norm" not in name
            )
            if is_resblock_linear_weight:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        # ── Backbone parameters (mps.6 + mps.7 + post_mp + learnable OOV emb) ──
        backbone_params = (
            list(self.gnn_layer6.parameters())   # NEW: mps.6 added
            + list(self.gnn_layer7.parameters())
            + list(self.gnn_post_mp.parameters())
            + [self.oov_emb]  # learnable OOV embedding in backbone group
        )

        print(f"[Optimizer] Muon params: {sum(p.numel() for p in muon_params):,} "
              f"(lr={hp.muon_lr})")
        print(f"[Optimizer] AdamW head params: {sum(p.numel() for p in adamw_params):,} "
              f"(lr={hp.lr}, weight_decay={hp.weight_decay})")
        print(f"[Optimizer] AdamW backbone params: {sum(p.numel() for p in backbone_params):,} "
              f"(lr={hp.backbone_lr}, weight_decay={hp.backbone_weight_decay})")
        print(f"[Optimizer] Backbone includes: mps.6 + mps.7 + post_mp + OOV emb")

        param_groups = [
            # Muon group: hidden weight matrices in ResidualBlocks
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            # AdamW group: input projection, bilinear head, embeddings, norms, biases
            dict(
                params=adamw_params,
                use_muon=False,
                lr=hp.lr,
                betas=(0.9, 0.999),
                eps=1e-10,
                weight_decay=hp.weight_decay,
            ),
            # AdamW backbone group: fine-tuned GNN layers at very low LR
            dict(
                params=backbone_params,
                use_muon=False,
                lr=hp.backbone_lr,
                betas=(0.9, 0.999),
                eps=1e-10,
                weight_decay=hp.backbone_weight_decay,
            ),
        ]

        # Use MuonWithAuxAdam for distributed training, SingleDeviceMuonWithAuxAdam otherwise
        if dist.is_available() and dist.is_initialized():
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        # Single cosine annealing schedule with linear warmup (NO warm restarts)
        #
        # CHANGE from parent: parent used CosineAnnealingWarmRestarts (T_0=130 epochs)
        # which empirically caused disruption at epoch 130 (second cycle max=0.500,
        # below first cycle best=0.5034). This node uses a SINGLE cosine cycle:
        #
        # - Steps 0 to warmup_steps: linear warmup
        # - Steps > warmup_steps: cosine decay from 1.0 to 0.0 over cosine_max_steps
        # - After cosine_max_steps: LR stays at 0.0 (clamped)
        #
        # The clamping at progress=1.0 prevents any unintended second-cycle behavior.
        # cosine_max_steps = T_max in steps, calibrated to ~80 epochs (parent best at epoch 69)
        cosine_max_steps = hp.cosine_max_steps

        def lr_lambda_single_cosine(current_step: int):
            if current_step < hp.warmup_steps:
                # Linear warmup
                return float(current_step) / max(1, hp.warmup_steps)
            # After warmup: single cosine cycle with hard clamp at 1.0 (no restart)
            step_after_warmup = current_step - hp.warmup_steps
            # Clamp progress at 1.0 to prevent any LR rebound after cosine ends
            progress = min(1.0, float(step_after_warmup) / max(1, cosine_max_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_single_cosine)
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
        description="Node 1-1-2-1-1-1-1-2 – Extended STRING_GNN Fine-tuning (mps.6+7+post_mp) + Single Cosine"
    )
    p.add_argument("--data-dir",              type=str,   default=_DEFAULT_DATA_DIR)
    p.add_argument("--hidden-dim",            type=int,   default=512)
    p.add_argument("--n-resblocks",           type=int,   default=6)
    p.add_argument("--expand",                type=int,   default=4)
    p.add_argument("--dropout",               type=float, default=0.3)
    p.add_argument("--rank",                  type=int,   default=512)
    p.add_argument("--lr",                    type=float, default=5e-4)
    p.add_argument("--muon-lr",               type=float, default=0.005)
    p.add_argument("--backbone-lr",           type=float, default=1e-5)
    p.add_argument("--weight-decay",          type=float, default=2e-3)
    p.add_argument("--backbone-weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps",          type=int,   default=100)
    p.add_argument("--focal-gamma",           type=float, default=2.0)
    p.add_argument("--label-smoothing",       type=float, default=0.05)
    p.add_argument("--class-weight-down",     type=float, default=2.0)
    p.add_argument("--class-weight-neutral",  type=float, default=0.5)
    p.add_argument("--class-weight-up",       type=float, default=4.0)
    p.add_argument("--grad-clip-norm",        type=float, default=1.0)
    p.add_argument("--micro-batch-size",      type=int,   default=16)
    p.add_argument("--global-batch-size",     type=int,   default=128)
    # Single cosine T_max in epochs (no warm restarts)
    p.add_argument("--cosine-tmax-epochs",    type=int,   default=80)
    p.add_argument("--max-epochs",            type=int,   default=150)
    p.add_argument("--patience",              type=int,   default=80)
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

    # Compute cosine T_max in effective steps
    # Each epoch has train_size / (micro_batch_size * n_gpus) steps,
    # further reduced by gradient accumulation
    _train_df_size = pd.read_csv(
        Path(args.data_dir) / "train.tsv", sep="\t", usecols=["pert_id"]
    ).shape[0]
    steps_per_epoch = _train_df_size // (args.micro_batch_size * n_gpus)
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)

    # T_max in effective steps for single cosine
    # Calibrated to ~80 epochs (parent's first-cycle peak was at epoch 69)
    cosine_max_steps = effective_steps_per_epoch * args.cosine_tmax_epochs

    print(f"[Main] effective_steps_per_epoch={effective_steps_per_epoch}, "
          f"cosine_max_steps={cosine_max_steps} (calibrated to {args.cosine_tmax_epochs} epochs)")
    print(f"[Main] max_epochs={args.max_epochs}, patience={args.patience}")

    lit = PerturbationLitModule(
        hidden_dim=args.hidden_dim,
        n_resblocks=args.n_resblocks,
        expand=args.expand,
        dropout=args.dropout,
        rank=args.rank,
        lr=args.lr,
        muon_lr=args.muon_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        backbone_weight_decay=args.backbone_weight_decay,
        warmup_steps=args.warmup_steps,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        class_weight_down=args.class_weight_down,
        class_weight_neutral=args.class_weight_neutral,
        class_weight_up=args.class_weight_up,
        cosine_max_steps=max(cosine_max_steps, 1),
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
            f"Node 1-1-2-1-1-1-1-2 – Extended STRING_GNN Fine-tuning (mps.6+7+post_mp) + Single Cosine\n"
            f"Test results from trainer: {test_results}\n"
            f"Hyperparameters: dropout={args.dropout}, rank={args.rank}, "
            f"weight_decay={args.weight_decay}, label_smoothing={args.label_smoothing}, "
            f"muon_lr={args.muon_lr}, lr={args.lr}, backbone_lr={args.backbone_lr}, "
            f"backbone_weight_decay={args.backbone_weight_decay}, "
            f"class_weights=[{args.class_weight_down}, {args.class_weight_neutral}, {args.class_weight_up}], "
            f"cosine_tmax_epochs={args.cosine_tmax_epochs}, "
            f"cosine_max_steps={lit.hparams.cosine_max_steps}, "
            f"max_epochs={args.max_epochs}, patience={args.patience}\n"
        )
        print(f"[Main] Test score saved → {score_path}")


if __name__ == "__main__":
    main()
