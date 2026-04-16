"""
Node 1-2-3-1-2: Partial STRING_GNN Unfreezing (last 2 GNN layers + post_mp) + 6-Layer MLP +
                Rank-512 + Class-Weighted Focal Loss [1.5, 0.8, 4.0] + Extended LR Schedule (9000)

Architecture:
  - STRING_GNN (partial fine-tuning): layers mps.0-5 frozen, mps.6-7 + post_mp trainable
    Intermediate state at layer-5 output is cached; only mps.6, mps.7, post_mp run per step
  - 6-layer residual MLP (hidden_dim=512, expand=4): transforms embedding to perturbation repr
  - Bilinear interaction head (rank=512): [B, 3, 512] x [6640, 512]^T -> [B, 3, 6640]
  - Class-weighted focal loss (gamma=2.0, weights=[1.5, 0.8, 4.0]) for class imbalance
  - Extended cosine annealing (total_steps=9000) + LR=3e-4 for deeper LR territory utilization
  - Three LR groups: GNN tail (1e-5), MLP body (3e-4), embeddings (1.5e-4, no wd)

Key differences from parent (node1-2-3-1):
  1. PARTIAL GNN FINE-TUNING: unfreeze last 2 GNN layers + post_mp (breaks frozen backbone ceiling)
     - Cached intermediate state eliminates expensive full GNN forward per step
     - Only mps.6, mps.7, post_mp run per step (efficient, ~200K additional GNN params)
  2. REVERT to 6 residual layers (from parent's 8 — reduces overfitting on 1,416 samples)
  3. CLASS WEIGHTS [1.5, 0.8, 4.0]: uses grandparent's proven up=4.0 + safe neutral=0.8
     (neutral=0.8 prevents the premature convergence that grandparent's neutral=0.5 caused)
  4. EXTENDED SCHEDULE total_steps=9000 (vs 6600) for deeper LR territory exploration
  5. PATIENCE=80 (vs 50) to allow the model to utilize the extended cosine schedule
  6. Keep LR=3e-4, separate LR groups (embeddings 0.5x), dropout=0.25

Differentiation from sibling (node1-2-3-1-1):
  - UNFREEZES STRING_GNN tail (sibling keeps backbone fully frozen)
  - Up-class weight 4.0 vs sibling's 3.0 (stronger minority class optimization)
  - Extended total_steps=9000 vs sibling's 6600

Root cause addressed:
  - Frozen 256-dim STRING_GNN embeddings are the primary information bottleneck
    (~0.497 F1 ceiling). Unfreezing mps.6, mps.7, post_mp injects gradient signal
    from the downstream task directly into the GNN, allowing per-perturbation feature
    adaptation beyond what fixed PPI topology encodes.
  - The efficient "split caching" approach (cache layer-5 output, run only mps.6-7 per step)
    ensures this is computationally feasible without per-step full GNN forward passes.
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


# ─── STRING_GNN Partial Embedding Extraction ──────────────────────────────────

def build_string_gnn_partial(
    model_dir: Path,
    device: torch.device,
    n_frozen_layers: int = 6,  # Freeze layers 0 to n_frozen_layers-1
) -> Tuple[np.ndarray, Dict[str, int], nn.Module, torch.Tensor, Optional[torch.Tensor]]:
    """Build precomputed STRING_GNN intermediate embeddings (through frozen layers).

    Runs the first n_frozen_layers frozen, caches intermediate state.
    Returns the cached intermediate state + the trainable GNN tail module.

    Args:
        model_dir: Path to STRING_GNN model directory
        device: compute device
        n_frozen_layers: how many GNN layers to freeze (from the start), default 6 (of 8)

    Returns:
        - cached_mid_emb_np: [N_nodes, 256] numpy array of intermediate GNN embeddings
          (state after layer n_frozen_layers-1, before layer n_frozen_layers)
        - node_name_to_idx: dict mapping Ensembl gene ID to node index
        - gnn_tail: nn.ModuleList containing the trainable GNN layers + post_mp
          (to be used during training for mps.6, mps.7 and post_mp)
        - edge_index_cpu: edge index tensor (CPU) for GNN tail forward passes
        - edge_weight_cpu: optional edge weight tensor (CPU)
    """
    node_names = json.loads((model_dir / "node_names.json").read_text())
    graph = torch.load(model_dir / "graph_data.pt", weights_only=False)

    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    model = model.to(device)
    model.eval()

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    edge_index = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"].to(device) if graph.get("edge_weight") is not None else None

    # Run the first n_frozen_layers (0 to n_frozen_layers-1) frozen to cache intermediate state
    # hidden_states[0] = initial emb.weight
    # hidden_states[i+1] = output after mps.i
    # We want hidden_states[n_frozen_layers] as input to the trainable tail
    with torch.no_grad():
        outputs = model(
            edge_index=edge_index,
            edge_weight=edge_weight,
            output_hidden_states=True,
        )

    # cached_mid is the state after mps[n_frozen_layers-1] = hidden_states[n_frozen_layers]
    hidden_states = outputs.hidden_states  # tuple of length 9 (1 initial + 8 layers)
    cached_mid_emb = hidden_states[n_frozen_layers].float().cpu()  # [N_nodes, 256]
    cached_mid_emb_np = cached_mid_emb.numpy()

    # Extract the trainable GNN tail layers
    # These are mps.n_frozen_layers, mps.n_frozen_layers+1, ..., mps.7, and post_mp
    gnn_tail_layers = nn.ModuleList()
    n_total_layers = len(model.mps)
    for layer_idx in range(n_frozen_layers, n_total_layers):
        gnn_tail_layers.append(model.mps[layer_idx])
    # post_mp is the final projection layer
    gnn_post_mp = model.post_mp

    # Create a small module that holds the trainable tail
    class GNNTail(nn.Module):
        def __init__(self, tail_layers: nn.ModuleList, post_mp: nn.Linear):
            super().__init__()
            self.layers = tail_layers
            self.post_mp = post_mp

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x, edge_index, edge_weight=edge_weight) + x
            return self.post_mp(x)

    gnn_tail = GNNTail(gnn_tail_layers, gnn_post_mp)

    # Move to CPU for storage, will be moved to device during training
    gnn_tail = gnn_tail.cpu()
    edge_index_cpu = edge_index.cpu()
    edge_weight_cpu = edge_weight.cpu() if edge_weight is not None else None

    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

    # Free the full model (we only kept the tail modules)
    del model
    torch.cuda.empty_cache()

    return cached_mid_emb_np, node_name_to_idx, gnn_tail, edge_index_cpu, edge_weight_cpu


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset using node indices (not precomputed embeddings).

    Stores node indices into the STRING_GNN vocabulary instead of precomputed
    embeddings, since the GNN tail runs during training for gradient flow.
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

        # Store node index (into STRING_GNN vocab) + in_vocab flag
        n_samples = len(df)
        node_indices = []
        in_vocab = []
        for pert_id in self.pert_ids:
            if pert_id in node_name_to_idx:
                node_indices.append(node_name_to_idx[pert_id])
                in_vocab.append(True)
            else:
                node_indices.append(0)  # dummy index (in_vocab=False, will use OOV embedding)
                in_vocab.append(False)

        self.node_indices = torch.tensor(node_indices, dtype=torch.long)  # [N]
        self.in_vocab = torch.tensor(in_vocab, dtype=torch.bool)          # [N]

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
            "node_idx": self.node_indices[idx],   # int index into STRING_GNN vocab
            "in_vocab": self.in_vocab[idx],         # bool
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule using node indices (GNN fine-tuning mode)."""

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
        # node_name_to_idx is populated during setup
        self.node_name_to_idx: Dict[str, int] = {}

    def setup(self, stage: Optional[str] = None):
        # Guard against double initialization
        if hasattr(self, "train_ds"):
            return

        # Build node_name_to_idx from STRING_GNN node names
        import json as _json
        node_names = _json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.node_name_to_idx = {name: i for i, name in enumerate(node_names)}

        # Load all splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        print(f"[DataModule] Coverage: "
              f"{sum(p in self.node_name_to_idx for p in dfs['train']['pert_id'])} / "
              f"{len(dfs['train'])} train genes in STRING_GNN")

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

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.25):
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
    """Prediction head: takes 256-dim embedding -> deep MLP -> bilinear output head.

    Key features:
      - 6 residual layers (proven optimal for 1,416-sample dataset)
      - Dropout=0.25 for regularization
      - Bilinear rank=512 (proven beneficial)
      - OOV handling via learnable fallback embedding
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.25,
        n_residual_layers: int = 6,  # Reverted to 6 (proven for this dataset size)
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.rank = rank

        # OOV embedding for genes not in STRING_GNN (fallback)
        self.oov_embedding = nn.Embedding(1, gnn_dim)  # shared OOV token

        # Input normalization (layer norm on GNN embeddings)
        self.input_norm = nn.LayerNorm(gnn_dim)

        # Projection: gnn_dim -> hidden_dim
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)

        # Residual MLP (6 blocks, same as grandparent — proven optimal)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim -> n_classes * rank (WIDE RANK)
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank)

        # Output gene embeddings: learnable [n_genes_out, rank]
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, rank))

        # Head dropout (applied after output norm)
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
        gnn_emb: torch.Tensor,   # [B, 256] current STRING_GNN embeddings (after tail forward)
        in_vocab: torch.Tensor,  # [B] bool mask
    ) -> torch.Tensor:
        B = gnn_emb.shape[0]

        # Replace OOV embeddings with learned fallback
        oov_emb = self.oov_embedding(torch.zeros(B, dtype=torch.long, device=gnn_emb.device))
        in_vocab_f = in_vocab.unsqueeze(1).float()
        x = gnn_emb * in_vocab_f + oov_emb * (1.0 - in_vocab_f)  # [B, gnn_dim]

        # Input normalization
        x = self.input_norm(x)

        # Projection to hidden dim
        x = self.proj_in(x)   # [B, hidden_dim]

        # Deep residual MLP (6 blocks)
        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)   # [B, hidden_dim]

        # Bilinear head
        x = self.head_dropout(x)
        pert_proj = self.proj_bilinear(x)                   # [B, n_classes * rank]
        pert_proj = pert_proj.view(B, self.n_classes, self.rank)  # [B, 3, rank]

        # Bilinear interaction: [B, 3, rank] x [rank, n_genes_out] -> [B, 3, n_genes_out]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, 6640]

        return logits


# ─── Class-Weighted Focal Loss ────────────────────────────────────────────────

def class_weighted_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Class-weighted focal loss for multi-class classification.

    Uses weights [1.5, 0.8, 4.0] for [down, neutral, up] classes:
    - up weight 4.0: stronger signal for the rarest class (3.0% of labels)
      Same as grandparent node1-2-3's proven up=4.0
    - neutral weight 0.8: mild downweight that avoids the premature convergence
      caused by grandparent's neutral=0.5 (harsh suppression)
    - down weight 1.5: moderate upweight for the down-regulated class (8.1%)

    Args:
        logits: [B, 3, G] raw logits
        labels: [B, G] integer class labels (0, 1, 2)
        class_weights: [3] per-class weights tensor
        gamma: focal focusing parameter (default 2.0)

    Returns:
        Scalar loss value.
    """
    B, C, G = logits.shape

    # Reshape to [B*G, 3] and [B*G]
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
    labels_flat = labels.reshape(-1)                        # [B*G]

    # Class-weighted cross entropy per element
    ce_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        weight=class_weights,
        reduction="none",
    )  # [B*G]

    # Focal weighting: down-weight easy examples
    with torch.no_grad():
        probs = F.softmax(logits_flat, dim=1)
        pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [B*G]
        focal_weight = (1.0 - pt).pow(gamma)                       # [B*G]

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
    """LightningModule for gene-perturbation DEG prediction (Node 1-2-3-1-2).

    Key differences from parent node1-2-3-1:
      1. Partial STRING_GNN fine-tuning: mps.6, mps.7, post_mp unfrozen (new gradient signal)
         - Cached intermediate embeddings (after frozen mps.0-5) for efficient training
         - Only the GNN tail (2 layers + post_mp, ~200K params) runs per step
      2. Reverted to 6 residual layers (from parent's 8 — proven optimal for 1,416 samples)
      3. Class weights [1.5, 0.8, 4.0]: grandparent's proven up=4.0 + safe neutral=0.8
      4. Extended total_steps=9000 (vs parent's 6600) for deeper LR territory
      5. Patience=80 (vs parent's 50) to allow extended schedule utilization
      6. Three LR groups: GNN tail (1e-5), MLP body (3e-4), embeddings (1.5e-4, no wd)
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_residual_layers: int = 6,          # REVERTED from parent's 8
        dropout: float = 0.25,
        lr: float = 3e-4,
        lr_emb_multiplier: float = 0.5,
        lr_gnn_tail_multiplier: float = 1/30,  # 1e-5 / 3e-4 ≈ 1/30
        weight_decay: float = 1e-3,
        focal_gamma: float = 2.0,
        class_weights_list: List[float] = None,  # [1.5, 0.8, 4.0]
        warmup_steps: int = 50,
        total_steps: int = 9000,               # EXTENDED for deeper LR territory
        n_frozen_gnn_layers: int = 6,           # Freeze mps.0-5, train mps.6-7 + post_mp
    ):
        super().__init__()
        if class_weights_list is None:
            class_weights_list = [1.5, 0.8, 4.0]
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
        if hasattr(self, 'model'):
            return

        # Build STRING_GNN partial: cache frozen intermediate embeddings + keep trainable tail
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[LitModule] Computing STRING_GNN partial embeddings (frozen layers 0-5)...")
        (
            cached_mid_emb_np,
            node_name_to_idx,
            gnn_tail,
            edge_index_cpu,
            edge_weight_cpu,
        ) = build_string_gnn_partial(STRING_GNN_DIR, device, n_frozen_layers=hp.n_frozen_gnn_layers)

        print(f"[LitModule] Cached mid-GNN embedding shape: {cached_mid_emb_np.shape}")

        # Store cached intermediate embeddings as a non-trainable buffer
        # Shape: [N_nodes, 256]
        cached_mid_emb_tensor = torch.from_numpy(cached_mid_emb_np).float()
        self.register_buffer("cached_mid_emb", cached_mid_emb_tensor)

        # Store the trainable GNN tail
        self.gnn_tail = gnn_tail

        # Store graph structure for GNN tail forward passes
        self.register_buffer("edge_index", edge_index_cpu)
        if edge_weight_cpu is not None:
            self.register_buffer("edge_weight", edge_weight_cpu)
        else:
            self.edge_weight = None

        # Store node name to index mapping (for coverage reporting)
        self._node_name_to_idx = node_name_to_idx

        # Build the MLP head
        self.model = GNNBilinearHead(
            gnn_dim=hp.gnn_dim,
            hidden_dim=hp.hidden_dim,
            rank=hp.rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
        )

        # Class weights as a buffer (device-safe)
        cw = torch.tensor(hp.class_weights_list, dtype=torch.float32)
        self.register_buffer("class_weights_buf", cw)

        # CRITICAL FIX: Re-enable gradients for the GNN tail.
        # build_string_gnn_partial() freezes ALL model parameters (requires_grad=False)
        # before extracting the tail layers, so we must explicitly re-enable gradient
        # for the trainable tail (mps.6, mps.7, post_mp) here.
        for p in self.gnn_tail.parameters():
            p.requires_grad = True

        # Ensure all trainable parameters are float32 for stable optimization
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        for p in self.gnn_tail.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    def _get_current_gnn_embeddings(self) -> torch.Tensor:
        """Run the trainable GNN tail over the full graph using cached intermediate state.

        Returns:
            current_gnn_emb: [N_nodes, 256] current node embeddings after GNN tail
        """
        # cached_mid_emb is [N_nodes, 256] and lives on the correct device
        x = self.cached_mid_emb  # [N_nodes, 256]
        ei = self.edge_index
        ew = self.edge_weight if hasattr(self, 'edge_weight') and self.edge_weight is not None else None

        # Run through trainable GNN tail
        current_gnn_emb = self.gnn_tail(x, ei, ew)  # [N_nodes, 256]
        return current_gnn_emb

    def forward(
        self,
        node_indices: torch.Tensor,  # [B] long indices into STRING_GNN vocab
        in_vocab: torch.Tensor,      # [B] bool
    ) -> torch.Tensor:
        """Full forward pass: GNN tail + MLP head."""
        # Get current embeddings from trainable GNN tail
        current_emb_all = self._get_current_gnn_embeddings()  # [N_nodes, 256]

        # Select batch-relevant embeddings
        batch_gnn_emb = current_emb_all[node_indices]  # [B, 256]

        # Run MLP head
        logits = self.model(batch_gnn_emb, in_vocab)  # [B, 3, 6640]
        return logits

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
            self.print(f"[Node1-2-3-1-2] Saved test predictions → {pred_path} ({len(seen_ids)} unique samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-2-3-1-2] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Three-group parameter separation:
        # 1. GNN tail (mps.6, mps.7, post_mp): very low LR to prevent catastrophic forgetting
        # 2. MLP body (proj_in, res_blocks, proj_bilinear, norms): standard LR
        # 3. Embedding-like (out_gene_emb, oov_embedding): lower LR, no weight decay

        gnn_tail_params = []
        embedding_param_names = {"model.out_gene_emb", "model.oov_embedding.weight"}
        embedding_params = []
        mlp_params = []

        # Collect GNN tail parameters
        for name, param in self.gnn_tail.named_parameters():
            if param.requires_grad:
                gnn_tail_params.append(param)

        # Collect MLP head parameters (separate embeddings from MLP body)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            full_name = f"model.{name}"
            if full_name in embedding_param_names:
                embedding_params.append(param)
            else:
                mlp_params.append(param)

        gnn_lr = hp.lr * hp.lr_gnn_tail_multiplier  # default: 3e-4 * (1/30) ≈ 1e-5
        emb_lr = hp.lr * hp.lr_emb_multiplier        # default: 3e-4 * 0.5 = 1.5e-4

        param_groups = [
            {"params": mlp_params,       "weight_decay": hp.weight_decay, "lr": hp.lr},
            {"params": embedding_params, "weight_decay": 0.0,              "lr": emb_lr},
            {"params": gnn_tail_params,  "weight_decay": 0.0,              "lr": gnn_lr},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=hp.lr,  # default (overridden per group above)
        )

        # Extended cosine annealing with linear warmup
        # total_steps=9000 for deeper LR territory exploration
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
    p = argparse.ArgumentParser(
        description="Node 1-2-3-1-2 – Partial STRING_GNN Fine-tuning (mps.6-7 + post_mp) + "
                    "6-Layer MLP + Rank-512 + Class-Weighted Focal [1.5, 0.8, 4.0] + "
                    "Extended LR Schedule (total_steps=9000) + Patience=80"
    )
    p.add_argument("--data-dir",                type=str,   default="data")
    p.add_argument("--gnn-dim",                 type=int,   default=256)
    p.add_argument("--hidden-dim",              type=int,   default=512)
    p.add_argument("--rank",                    type=int,   default=512)
    p.add_argument("--n-residual-layers",       type=int,   default=6)       # Reverted to 6
    p.add_argument("--dropout",                 type=float, default=0.25)
    p.add_argument("--lr",                      type=float, default=3e-4)
    p.add_argument("--lr-emb-multiplier",       type=float, default=0.5)     # emb gets 0.5x LR
    p.add_argument("--lr-gnn-tail-multiplier",  type=float, default=1/30)    # GNN tail ~1e-5
    p.add_argument("--weight-decay",            type=float, default=1e-3)
    p.add_argument("--focal-gamma",             type=float, default=2.0)
    p.add_argument("--class-weight-down",       type=float, default=1.5)
    p.add_argument("--class-weight-neutral",    type=float, default=0.8)
    p.add_argument("--class-weight-up",         type=float, default=4.0)     # Grandparent's proven up=4.0
    p.add_argument("--warmup-steps",            type=int,   default=50)
    p.add_argument("--total-steps",             type=int,   default=9000)    # EXTENDED
    p.add_argument("--n-frozen-gnn-layers",     type=int,   default=6)       # Freeze mps.0-5
    p.add_argument("--micro-batch-size",        type=int,   default=16)
    p.add_argument("--global-batch-size",       type=int,   default=64)
    p.add_argument("--max-epochs",              type=int,   default=300)
    p.add_argument("--patience",                type=int,   default=80)      # INCREASED for extended schedule
    p.add_argument("--num-workers",             type=int,   default=4)
    p.add_argument("--val-check-interval",      type=float, default=1.0)
    p.add_argument("--debug-max-step",          type=int,   default=None)
    p.add_argument("--fast-dev-run",            action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataModule (loads just data, no embedding precomputation)
    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # Use extended total_steps from args (default=9000)
    total_steps = args.total_steps

    # LightningModule
    lit = PerturbationLitModule(
        gnn_dim=args.gnn_dim,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        n_residual_layers=args.n_residual_layers,
        dropout=args.dropout,
        lr=args.lr,
        lr_emb_multiplier=args.lr_emb_multiplier,
        lr_gnn_tail_multiplier=args.lr_gnn_tail_multiplier,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        class_weights_list=[args.class_weight_down, args.class_weight_neutral, args.class_weight_up],
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        n_frozen_gnn_layers=args.n_frozen_gnn_layers,
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
            f"Node 1-2-3-1-2 – Partial STRING_GNN Fine-tuning (mps.6-7 + post_mp) + "
            f"6-Layer MLP + Rank-512 + Class-Weighted Focal [1.5, 0.8, 4.0]\n"
            f"Extended LR Schedule (total_steps=9000) + Patience=80\n"
            f"Test results from trainer: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
