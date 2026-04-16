"""
Node 2-1-1-1-2-1-1 — Partial STRING_GNN + Deep Bilinear MLP + Checkpoint Ensemble
                     + Dropout=0.4 + Label Smoothing=0.10 + Extended Cosine Schedule

Architecture (same backbone as parent):
  - STRING_GNN backbone:
      * mps.0-5: frozen (precomputed embeddings through layer 5 stored in dataset)
      * mps.6, mps.7, post_mp: trainable (online graph forward with full graph,
        using precomputed mps.0-5 output as starting point, backbone_lr=1e-5 AdamW)
  - Output gene embedding initialization: STRING_GNN mps.5 embeddings (93.2% coverage)
  - 6-layer deep residual bilinear MLP head (rank=512, hidden=512, expand=4, dropout=0.4)
  - MuonWithAuxAdam optimizer:
      * Muon (lr=0.005) for hidden weight matrices in ResidualBlocks
      * AdamW (lr=5e-4) for head projections, embeddings, norms
      * AdamW (lr=1e-5, wd=2e-4) for trainable backbone layers (mps.6, mps.7, post_mp)
  - Class-weighted focal loss: gamma=2.0, weights=[2.0, 0.5, 4.0] for (down, neutral, up)
  - Label smoothing epsilon=0.10 (increased from parent's 0.05)
  - Single cosine cycle schedule calibrated to ~225 epochs (total_steps≈2475 for 2 GPUs)
  - Patience=100 (unchanged from parent)

Key differences from parent (node2-1-1-1-2-1, F1=0.5088):
  1. Increase dropout: 0.3 → 0.4
     (Parent feedback Priority 1: persistent overfitting despite 0.3; train/val ratio 4:1)
  2. Increase label smoothing: 0.05 → 0.10
     (Parent feedback Priority 1: stronger smoothing for 157-sample val F1 noise)
  3. Extended cosine schedule: target 150 → 225 epochs (total_steps ~2475 for 2 GPUs)
     (Parent cosine expired at epoch ~75; extending allows slower convergence with partial FT)
  4. Checkpoint ensemble at test time: save top-5 checkpoints, average softmax probs
     (Parent had wide oscillations with many good peaks; averaging captures multiple optima)
     Expected: +0.002-0.005 F1 from smoothing oscillation noise (feedback recommendation)

Rationale:
  - Parent achieved F1=0.5088, within 0.0011 of tree best (0.5099)
  - Main bottleneck: moderate overfitting (train/val loss ratio 4:1 at termination)
  - Dropout=0.3 wasn't sufficient; 17M params / 1416 samples needs stronger regularization
  - Extended schedule allows cosine to decay more slowly, preserving useful LR longer
  - Checkpoint ensemble is the highest-impact novel change: parent had oscillations at
    epochs 24 (0.508), 33 (0.507), 44 (0.508), 52 (0.509), 56 (0.506), 59 (0.506),
    64 (0.505) — averaging these captures a better consensus prediction
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
LABEL_GENES_FILE = Path("data/label_genes.txt")

# Tree-validated class weights (softer than inverse-freq, proven at F1>=0.50 nodes)
CLASS_WEIGHTS_LIST = [2.0, 0.5, 4.0]  # [down, neutral, up]

# Partial backbone fine-tuning: unfreeze the last 2 GNN layers + output projection
# Forward structure: emb.weight -> mps.0-5 (frozen) -> mps.6, mps.7, post_mp (trainable)
N_FROZEN_LAYERS = 6  # mps.0 through mps.5 are frozen


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

    Stores precomputed frozen embeddings (output of mps.0-5) for fast batch
    retrieval. Trainable layers mps.6, mps.7, post_mp run at forward time in
    the LightningModule using the full graph.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        frozen_embeddings: np.ndarray,     # [N_nodes, 256] - output of mps.0-5 for all STRING nodes
        node_name_to_idx: Dict[str, int],
        embed_dim: int = 256,
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.has_labels = has_labels

        n_samples = len(df)
        # Store node indices for retrieval (not per-sample embeddings)
        # We need the full node embedding matrix in the module for graph-based ops
        node_indices = np.full(n_samples, -1, dtype=np.int64)  # -1 for OOV
        for i, pert_id in enumerate(self.pert_ids):
            if pert_id in node_name_to_idx:
                node_indices[i] = node_name_to_idx[pert_id]

        self.node_indices = torch.from_numpy(node_indices)  # [N] int64, -1 for OOV

        # Also precompute per-sample embeddings for fast batch retrieval
        embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)
        for i, idx in enumerate(node_indices):
            if idx >= 0:
                embeddings[i] = frozen_embeddings[idx]
        self.embeddings = torch.from_numpy(embeddings)   # [N, 256]

        # In-vocab mask
        in_vocab = [node_name_to_idx.get(p, -1) >= 0 for p in self.pert_ids]
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
            "pert_id":      self.pert_ids[idx],
            "symbol":       self.symbols[idx],
            "embedding":    self.embeddings[idx],       # [256] - pre-frozen mps.0-5 output
            "node_idx":     self.node_indices[idx],     # int64 - STRING_GNN node index (-1=OOV)
            "in_vocab":     self.in_vocab[idx],          # bool
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule with partial backbone computation.

    Precomputes frozen mps.0-5 embeddings for all STRING nodes.
    Stores the full node embedding matrix for online partial FT inference.
    """

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
        if hasattr(self, "train_ds"):
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[DataModule] Computing intermediate embeddings through mps.{N_FROZEN_LAYERS-1} (frozen)...")
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", weights_only=False)

        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone = backbone.to(device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        edge_index = graph["edge_index"].to(device)
        edge_weight = graph["edge_weight"].to(device) if graph.get("edge_weight") is not None else None

        with torch.no_grad():
            # Forward through first N_FROZEN_LAYERS layers only
            x = backbone.emb.weight  # [N_nodes, 256]
            for i in range(N_FROZEN_LAYERS):  # mps.0 through mps.5
                layer = backbone.mps[i]
                # GNNLayer forward: x = x + act(norm(conv(x, edge_index, edge_weight)))
                x_conv = layer.conv(x, edge_index, edge_weight)
                x_norm = layer.norm(x_conv)
                x_act = layer.act(x_norm)
                x = x + layer.dropout(x_act)
            # x is now the intermediate embedding after N_FROZEN_LAYERS frozen layers

        frozen_embeddings = x.float().cpu().numpy()  # [N_nodes, 256]

        node_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(node_names)}

        self.frozen_embeddings = frozen_embeddings
        self.frozen_embeddings_tensor = torch.from_numpy(frozen_embeddings)  # for module use
        self.node_name_to_idx = node_name_to_idx
        self.n_gnn_nodes = len(node_names)
        self.edge_index = graph["edge_index"]  # CPU tensor for module storage
        self.edge_weight = graph["edge_weight"] if graph.get("edge_weight") is not None else None

        del backbone
        torch.cuda.empty_cache()

        print(f"[DataModule] Frozen intermediate embeddings shape: {frozen_embeddings.shape}")
        print(f"[DataModule] mps.{N_FROZEN_LAYERS}+, post_mp will be applied online (trainable)")

        # Load data splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        n_train_cov = sum(p in node_name_to_idx for p in dfs["train"]["pert_id"])
        print(f"[DataModule] Coverage: {n_train_cov}/{len(dfs['train'])} train genes in STRING_GNN")

        embed_dim = frozen_embeddings.shape[1]
        self.train_ds = PerturbationDataset(dfs["train"], frozen_embeddings, node_name_to_idx, embed_dim, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   frozen_embeddings, node_name_to_idx, embed_dim, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  frozen_embeddings, node_name_to_idx, embed_dim, True)

        # Load label gene mappings for output gene embedding initialization
        label_genes_path = self.data_dir / "label_genes.txt"
        self.label_gene_ids: List[str] = []
        if label_genes_path.exists():
            with open(label_genes_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        self.label_gene_ids.append(parts[0])  # Ensembl ID
        print(f"[DataModule] Loaded {len(self.label_gene_ids)} label gene IDs from label_genes.txt")

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


# ─── Model Components ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block with increased dropout=0.4 (from parent's 0.3)."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.4):
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
    """Prediction head with bilinear interaction and pre-initialized output gene embeddings."""

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.4,
        n_residual_layers: int = 6,
        out_gene_emb_init: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.rank = rank

        self.input_norm = nn.LayerNorm(gnn_dim)
        self.oov_embedding = nn.Parameter(torch.zeros(gnn_dim))
        nn.init.normal_(self.oov_embedding, std=0.02)
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank)
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, rank))
        self.head_dropout = nn.Dropout(dropout)
        self._init_weights(out_gene_emb_init)

    def _init_weights(self, out_gene_emb_init: Optional[np.ndarray]):
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)

        if out_gene_emb_init is not None and out_gene_emb_init.shape[0] == self.n_genes_out:
            gnn_dim = out_gene_emb_init.shape[1]
            if gnn_dim != self.rank:
                proj_matrix = np.random.randn(gnn_dim, self.rank).astype(np.float32)
                proj_matrix /= np.sqrt(gnn_dim)
                projected = out_gene_emb_init @ proj_matrix  # [n_genes, rank]
            else:
                projected = out_gene_emb_init.copy()
            # Scale to std=0.02 to avoid dominating random init
            projected = projected / (projected.std() + 1e-8) * 0.02
            self.out_gene_emb.data.copy_(torch.from_numpy(projected))
        else:
            nn.init.normal_(self.out_gene_emb, std=0.02)

    def forward(self, gnn_emb: torch.Tensor, in_vocab: torch.Tensor) -> torch.Tensor:
        """
        gnn_emb: [B, 256] final embedding from backbone adapter
        in_vocab: [B] bool mask
        Returns: logits [B, 3, 6640]
        """
        # Replace OOV embeddings
        # Use expand + indexed assignment (same as parent) to avoid dtype/shape mismatch
        # under bf16-mixed autocast where gnn_emb may be bfloat16 but oov_embedding is float32
        oov_mask = ~in_vocab
        if oov_mask.any():
            B = gnn_emb.shape[0]
            oov_emb = self.oov_embedding.to(dtype=gnn_emb.dtype).unsqueeze(0).expand(B, -1)
            gnn_emb = gnn_emb.clone()
            gnn_emb[oov_mask] = oov_emb[oov_mask]

        x = self.input_norm(gnn_emb)
        x = self.proj_in(x)  # [B, 512]
        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)
        x = self.head_dropout(x)
        x = self.proj_bilinear(x)  # [B, 3*512]
        pert_proj = x.view(-1, self.n_classes, self.rank)  # [B, 3, 512]
        # Bilinear interaction: [B, 3, 512] x [6640, 512]^T -> [B, 3, 6640]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)
        return logits


class PartialBackboneAdapter(nn.Module):
    """Wraps trainable STRING_GNN layers mps.6, mps.7, post_mp.

    Takes full intermediate embedding matrix (N_nodes, 256) from frozen mps.0-5
    and applies the trainable layers using full graph structure.
    Returns final embeddings for all nodes.
    """

    def __init__(self, layer6, layer7, post_mp):
        super().__init__()
        self.layer6 = layer6
        self.layer7 = layer7
        self.post_mp = post_mp

    def forward(
        self,
        x: torch.Tensor,           # [N_nodes, 256] - intermediate after mps.0-5
        edge_index: torch.Tensor,  # [2, E]
        edge_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply trainable layers with full graph structure."""
        # GNNLayer: x = x + dropout(act(norm(conv(x, edge, w))))
        x = x + self.layer6.dropout(self.layer6.act(
            self.layer6.norm(self.layer6.conv(x, edge_index, edge_weight))
        ))
        x = x + self.layer7.dropout(self.layer7.act(
            self.layer7.norm(self.layer7.conv(x, edge_index, edge_weight))
        ))
        x = self.post_mp(x)
        return x  # [N_nodes, 256]


# ─── Focal Loss with Label Smoothing ──────────────────────────────────────────

def focal_loss_with_label_smoothing(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.10,
) -> torch.Tensor:
    """Focal loss with class weighting and label smoothing.

    Label smoothing epsilon=0.10 (increased from parent's 0.05) reduces
    overconfidence and sensitivity to noisy labels on the small 157-sample
    validation set.
    """
    B, C, G = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
    labels_flat = labels.reshape(-1)                        # [B*G]

    if label_smoothing > 0.0:
        # Smoothed soft-target loss
        one_hot = F.one_hot(labels_flat, num_classes=C).float()  # [B*G, C]
        smooth_targets = (1.0 - label_smoothing) * one_hot + label_smoothing / C

        log_probs = F.log_softmax(logits_flat, dim=1)  # [B*G, C]

        if class_weights is not None:
            # Weight soft targets by per-class importance
            weighted = smooth_targets * class_weights.unsqueeze(0)
            ce_loss = -(weighted * log_probs).sum(dim=1)  # [B*G]
        else:
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)

        # Focal weight based on argmax class probability
        with torch.no_grad():
            probs = torch.softmax(logits_flat, dim=1)
            pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
    else:
        ce_loss = F.cross_entropy(
            logits_flat, labels_flat,
            weight=class_weights,
            reduction="none",
        )
        with torch.no_grad():
            probs = F.softmax(logits_flat, dim=1)
            pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)

    focal_weight = (1.0 - pt).pow(gamma)
    return (focal_weight * ce_loss).mean()


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
    dist.all_gather(g_preds,  p)
    dist.all_gather(g_labels, l)

    real_preds  = torch.cat([g_preds[i][:all_sizes[i].item()].cpu()  for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][:all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """LightningModule for gene-perturbation DEG prediction.

    Architecture:
    - mps.0-5: frozen (precomputed embeddings stored in dataset as features)
    - mps.6, mps.7, post_mp: trainable (online graph forward at each training step)
    - 6-layer bilinear residual MLP head (rank=512, dropout=0.4)
    - MuonWithAuxAdam optimizer
    - Class-weighted focal loss with label smoothing epsilon=0.10
    - Checkpoint ensemble at test time for improved predictions
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_residual_layers: int = 6,
        dropout: float = 0.4,
        lr_muon: float = 0.005,
        lr_adamw: float = 5e-4,
        backbone_lr: float = 1e-5,
        weight_decay: float = 2e-3,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.10,
        use_class_weights: bool = True,
        warmup_steps: int = 100,
        total_steps: int = 2475,
        out_gene_emb_init: Optional[np.ndarray] = None,
        # Backbone data for setup
        frozen_embeddings_np: Optional[np.ndarray] = None,  # [N_nodes, 256]
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["out_gene_emb_init", "frozen_embeddings_np",
                                          "edge_index", "edge_weight"])
        self._out_gene_emb_init = out_gene_emb_init
        self._frozen_embeddings_np = frozen_embeddings_np
        self._edge_index = edge_index
        self._edge_weight = edge_weight

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams

        # Build bilinear prediction head
        self.head = GNNBilinearHead(
            gnn_dim=hp.gnn_dim,
            hidden_dim=hp.hidden_dim,
            rank=hp.rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
            out_gene_emb_init=self._out_gene_emb_init,
        )

        # Build partial backbone adapter (mps.6, mps.7, post_mp)
        # Load the full backbone model to extract these layers
        full_backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        self.backbone_adapter = PartialBackboneAdapter(
            layer6=full_backbone.mps[6],
            layer7=full_backbone.mps[7],
            post_mp=full_backbone.post_mp,
        )
        del full_backbone
        print(f"[Setup] Trainable backbone adapter: mps.6, mps.7, post_mp")
        backbone_params = sum(p.numel() for p in self.backbone_adapter.parameters())
        print(f"[Setup] Trainable backbone params: {backbone_params:,}")

        # Store frozen intermediate embeddings as a non-trainable buffer
        # This is the full [N_nodes, 256] matrix from mps.0-5
        if self._frozen_embeddings_np is not None:
            frozen_emb_tensor = torch.from_numpy(self._frozen_embeddings_np.astype(np.float32))
            self.register_buffer("frozen_node_embs", frozen_emb_tensor)
        else:
            print("[Setup] WARNING: No frozen embeddings provided")
            self.frozen_node_embs = None

        # Store graph structure for online backbone forward
        if self._edge_index is not None:
            self.register_buffer("edge_index_buf", self._edge_index)
        else:
            self.register_buffer("edge_index_buf", torch.zeros(2, 1, dtype=torch.long))
        if self._edge_weight is not None:
            self.register_buffer("edge_weight_buf", self._edge_weight)
        else:
            # Register as None buffer (supported in PyTorch >= 1.8)
            self.register_buffer("edge_weight_buf", None)

        # Class weights for focal loss
        if hp.use_class_weights:
            cw = torch.tensor(CLASS_WEIGHTS_LIST, dtype=torch.float32)
            self.register_buffer("class_weights", cw)
            print(f"[Setup] Using class weights [2.0, 0.5, 4.0]")
        else:
            self.class_weights = None

        # Cast all trainable parameters to float32 for stable optimization
        for p in self.head.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        for p in self.backbone_adapter.parameters():
            p.data = p.data.float()

        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        backbone_trainable = sum(p.numel() for p in self.backbone_adapter.parameters())
        print(f"[Setup] Head trainable: {head_trainable:,} | Backbone trainable: {backbone_trainable:,}")

    def _get_final_embeddings(
        self,
        batch_embeddings: torch.Tensor,   # [B, 256] - pre-frozen mps.0-5 embeddings
        node_indices: torch.Tensor,        # [B] - STRING node indices (-1 for OOV)
        in_vocab: torch.Tensor,            # [B] bool
    ) -> torch.Tensor:
        """Apply trainable backbone layers (mps.6, mps.7, post_mp) to get final embeddings.

        Strategy: Run the full graph forward through trainable layers once per batch.
        Then index the resulting node embeddings for batch genes.
        """
        if self.frozen_node_embs is None:
            return batch_embeddings

        # Run full graph through trainable layers: [N_nodes, 256] -> [N_nodes, 256]
        full_final_embs = self.backbone_adapter(
            self.frozen_node_embs.float(),
            self.edge_index_buf,
            self.edge_weight_buf,  # May be None, which GCNConv handles correctly
        )  # [N_nodes, 256]

        # Retrieve embeddings for the batch genes
        B = batch_embeddings.shape[0]
        device = batch_embeddings.device
        result = torch.zeros(B, batch_embeddings.shape[1], device=device, dtype=torch.float32)

        in_vocab_mask = in_vocab & (node_indices >= 0)
        if in_vocab_mask.any():
            valid_indices = node_indices[in_vocab_mask]
            # Cast to float32 to match result dtype (full_final_embs may be bf16 under autocast)
            result[in_vocab_mask] = full_final_embs[valid_indices].float()

        # OOV genes: use zeros (replaced by oov_embedding in the head's forward method)
        return result

    def forward(
        self,
        embedding: torch.Tensor,    # [B, 256] - frozen mps.0-5 embedding
        node_indices: torch.Tensor,  # [B] - STRING node indices
        in_vocab: torch.Tensor,      # [B] bool
    ) -> torch.Tensor:
        # Get final embeddings after trainable backbone layers
        final_emb = self._get_final_embeddings(embedding, node_indices, in_vocab)
        # Apply bilinear head
        logits = self.head(final_emb, in_vocab)  # [B, 3, 6640]
        return logits

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cw = None
        if self.hparams.use_class_weights and hasattr(self, "class_weights") and self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
        return focal_loss_with_label_smoothing(
            logits, labels,
            gamma=self.hparams.focal_gamma,
            class_weights=cw,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        logits = self(
            batch["embedding"].float(),
            batch["node_idx"],
            batch["in_vocab"],
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(
            batch["embedding"].float(),
            batch["node_idx"],
            batch["in_vocab"],
        )
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
        logits = self(
            batch["embedding"].float(),
            batch["node_idx"],
            batch["in_vocab"],
        )
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

            self.print(f"[Node2-1-1-1-2-1-1] Saved test predictions → {pred_path} ({len(seen_ids)} samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node2-1-1-1-2-1-1] Self-computed test F1 (single ckpt) = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Parameter groups:
        # 1. Muon for hidden weight matrices in ResidualBlocks
        # 2. AdamW (head lr) for embeddings, projections, norms in head
        # 3. AdamW (backbone lr, very small) for trainable backbone layers
        muon_params = []
        adamw_head_params = []
        adamw_backbone_params = []

        for name, param in self.head.named_parameters():
            if not param.requires_grad:
                continue
            is_hidden_matrix = (
                param.ndim >= 2
                and "res_blocks" in name
                and "weight" in name
                and "norm" not in name
            )
            if is_hidden_matrix:
                muon_params.append(param)
            else:
                adamw_head_params.append(param)

        for param in self.backbone_adapter.parameters():
            if param.requires_grad:
                adamw_backbone_params.append(param)

        n_muon = sum(p.numel() for p in muon_params)
        n_adamw_head = sum(p.numel() for p in adamw_head_params)
        n_adamw_backbone = sum(p.numel() for p in adamw_backbone_params)
        print(f"[Optimizer] Muon={n_muon:,} | AdamW head={n_adamw_head:,} | AdamW backbone={n_adamw_backbone:,}")

        try:
            from muon import MuonWithAuxAdam

            param_groups = [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=hp.lr_muon,
                    weight_decay=hp.weight_decay,
                    momentum=0.95,
                ),
                dict(
                    params=adamw_head_params,
                    use_muon=False,
                    lr=hp.lr_adamw,
                    betas=(0.9, 0.95),
                    eps=1e-10,
                    weight_decay=hp.weight_decay,
                ),
                dict(
                    params=adamw_backbone_params,
                    use_muon=False,
                    lr=hp.backbone_lr,
                    betas=(0.9, 0.95),
                    eps=1e-10,
                    weight_decay=hp.weight_decay * 0.1,  # Lighter backbone regularization
                ),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
            print(f"[Optimizer] MuonWithAuxAdam: Muon lr={hp.lr_muon}, "
                  f"AdamW head lr={hp.lr_adamw}, AdamW backbone lr={hp.backbone_lr}")

        except ImportError:
            print("[Optimizer] WARNING: MuonWithAuxAdam not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                [
                    {"params": muon_params,          "lr": hp.lr_adamw},
                    {"params": adamw_head_params,     "lr": hp.lr_adamw},
                    {"params": adamw_backbone_params, "lr": hp.backbone_lr},
                ],
                weight_decay=hp.weight_decay,
            )

        # Cosine annealing with linear warmup
        # Calibrated to 225-epoch window (extended from parent's 150-epoch target)
        # Longer window allows slower cosine decay, keeping useful LR longer
        def lr_lambda(step: int) -> float:
            if step < hp.warmup_steps:
                return float(step) / max(1, hp.warmup_steps)
            progress = (step - hp.warmup_steps) / max(1, hp.total_steps - hp.warmup_steps)
            progress = min(progress, 1.0)  # prevent schedule rebound
            return max(1e-7 / hp.lr_adamw, 0.5 * (1.0 + np.cos(np.pi * progress)))

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
        trainable_sd = {k: v for k, v in full_sd.items() if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)")
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Checkpoint Ensemble Inference ────────────────────────────────────────────

def run_ensemble_inference(
    lit: PerturbationLitModule,
    test_ds: PerturbationDataset,
    ckpt_paths: List[str],
    micro_batch_size: int,
    num_workers: int,
    out_dir: Path,
) -> None:
    """Run inference for each checkpoint on rank 0 and average predictions.

    This function:
    1. Loads each checkpoint's weights into the existing lit module (which already
       has all buffers initialized from DataModule)
    2. Runs inference on the test dataset
    3. Averages the softmax probabilities across all checkpoints
    4. Writes the averaged predictions to test_predictions.tsv

    Precondition: Called on rank 0 only (after distributed training is complete).
    """
    if not ckpt_paths:
        print("[Ensemble] No checkpoints provided, skipping ensemble")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create a fresh test DataLoader with consistent ordering
    test_loader = DataLoader(
        test_ds,
        batch_size=micro_batch_size * 2,
        shuffle=False,
        num_workers=min(num_workers, 4),
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    # Move the model to the inference device (single GPU)
    # The backbone adapter and head are already initialized with all buffers
    lit_model = lit
    lit_model.eval()
    lit_model = lit_model.to(device)

    # Move buffers to device
    if lit_model.frozen_node_embs is not None:
        lit_model.frozen_node_embs = lit_model.frozen_node_embs.to(device)
    lit_model.edge_index_buf = lit_model.edge_index_buf.to(device)
    if lit_model.edge_weight_buf is not None:
        lit_model.edge_weight_buf = lit_model.edge_weight_buf.to(device)

    all_ens_probs: List[torch.Tensor] = []
    ens_pert_ids: Optional[List[str]] = None
    ens_symbols: Optional[List[str]] = None

    print(f"[Ensemble] Running inference on {len(ckpt_paths)} checkpoints...")

    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"[Ensemble] Checkpoint {i+1}/{len(ckpt_paths)}: {Path(ckpt_path).name}")

        # Load checkpoint weights into the existing model
        # This updates trainable params while preserving buffers (frozen_node_embs, etc.)
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except Exception:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        # Load trainable parameters (strict=False handles partial state dicts)
        lit_model.load_state_dict(state_dict)  # uses custom load_state_dict (strict=False)
        lit_model.eval()

        probs_list: List[torch.Tensor] = []
        pert_ids_list: List[str] = []
        syms_list: List[str] = []

        with torch.no_grad():
            for batch in test_loader:
                emb = batch["embedding"].float().to(device)
                node_idx = batch["node_idx"].to(device)
                in_vocab = batch["in_vocab"].to(device)
                logits = lit_model(emb, node_idx, in_vocab)  # [B, 3, 6640]
                probs = torch.softmax(logits, dim=1).float().cpu()
                probs_list.append(probs)
                pert_ids_list.extend(batch["pert_id"])
                syms_list.extend(batch["symbol"])

        ckpt_probs = torch.cat(probs_list, dim=0)  # [N, 3, 6640]
        all_ens_probs.append(ckpt_probs)
        if ens_pert_ids is None:
            ens_pert_ids = pert_ids_list
            ens_symbols = syms_list

        print(f"[Ensemble] Checkpoint {i+1}: {ckpt_probs.shape[0]} predictions collected")

    if not all_ens_probs:
        print("[Ensemble] No predictions collected, skipping")
        return

    # Average predictions across checkpoints
    # Stack: [K, N, 3, 6640] → mean over K → [N, 3, 6640]
    avg_probs = torch.stack(all_ens_probs, dim=0).mean(dim=0).numpy()
    print(f"[Ensemble] Averaged {len(all_ens_probs)} checkpoint predictions, shape={avg_probs.shape}")

    # Write ensemble predictions (overwrites single-checkpoint prediction)
    pred_path = out_dir / "test_predictions.tsv"
    seen_ids: set = set()
    with open(pred_path, "w") as fh:
        fh.write("idx\tinput\tprediction\n")
        for pert_id, symbol, probs in zip(ens_pert_ids, ens_symbols, avg_probs):
            if pert_id not in seen_ids:
                seen_ids.add(pert_id)
                fh.write(f"{pert_id}\t{symbol}\t{json.dumps(probs.tolist())}\n")

    print(f"[Ensemble] Saved {len(seen_ids)} ensemble-averaged predictions → {pred_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-1-1-1-2-1-1 – Partial STRING_GNN + Deep Bilinear MLP + "
                    "Checkpoint Ensemble + Dropout=0.4 + Label Smoothing=0.10 + Extended Schedule"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--gnn-dim",            type=int,   default=256)
    p.add_argument("--hidden-dim",         type=int,   default=512)
    p.add_argument("--rank",               type=int,   default=512)
    p.add_argument("--n-residual-layers",  type=int,   default=6)
    p.add_argument("--dropout",            type=float, default=0.4,
                   help="Dropout (increased from parent 0.3; stronger regularization)")
    p.add_argument("--lr-muon",            type=float, default=0.005)
    p.add_argument("--lr-adamw",           type=float, default=5e-4)
    p.add_argument("--backbone-lr",        type=float, default=1e-5,
                   help="LR for trainable backbone layers (mps.6, mps.7, post_mp)")
    p.add_argument("--weight-decay",       type=float, default=2e-3)
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--label-smoothing",    type=float, default=0.10,
                   help="Label smoothing epsilon (increased from parent 0.05)")
    p.add_argument("--use-class-weights",  action="store_true", default=True)
    p.add_argument("--no-class-weights",   dest="use_class_weights", action="store_false")
    p.add_argument("--warmup-steps",       type=int,   default=100)
    p.add_argument("--micro-batch-size",   type=int,   default=16)
    p.add_argument("--global-batch-size",  type=int,   default=64)
    p.add_argument("--max-epochs",         type=int,   default=350)
    p.add_argument("--patience",           type=int,   default=100)
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--ensemble-top-k",     type=int,   default=5,
                   help="Number of top checkpoints to average at test time (0=disable)")
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
    return p.parse_args()


def build_out_gene_emb_init(
    frozen_embeddings: np.ndarray,
    node_name_to_idx: Dict[str, int],
    label_gene_ids: List[str],
    rank: int,
) -> Optional[np.ndarray]:
    """Build output gene embedding initializations from STRING_GNN intermediate embeddings.

    Uses the frozen mps.0-5 output embeddings to initialize output gene embeddings.
    These are the same embeddings that the trainable backbone layers will start from.
    """
    gnn_dim = frozen_embeddings.shape[1]
    out_init = np.random.randn(len(label_gene_ids), gnn_dim).astype(np.float32) * 0.02

    n_found = 0
    for i, gene_id in enumerate(label_gene_ids):
        if gene_id in node_name_to_idx:
            node_idx = node_name_to_idx[gene_id]
            out_init[i] = frozen_embeddings[node_idx]
            n_found += 1

    print(f"[Init] Output gene embeddings: {n_found}/{len(label_gene_ids)} "
          f"({100*n_found/len(label_gene_ids):.1f}%) initialized from STRING_GNN "
          f"(will be projected to rank={rank} in GNNBilinearHead)")
    return out_init


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataModule — computes frozen intermediate embeddings (mps.0-5 output)
    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    # Build output gene embedding initializations from frozen mps.0-5 embeddings
    out_gene_emb_init = None
    if hasattr(dm, "label_gene_ids") and len(dm.label_gene_ids) == N_GENES_OUT:
        out_gene_emb_init = build_out_gene_emb_init(
            dm.frozen_embeddings,
            dm.node_name_to_idx,
            dm.label_gene_ids,
            rank=args.rank,
        )
    else:
        print("[Main] WARNING: label_gene_ids not loaded or wrong length")

    # Estimate total training steps for LR schedule
    # Extended calibration: targeting ~225-epoch window (up from parent's 150 epochs)
    # This allows slower cosine decay, keeping useful LR longer before schedule expires
    # Parent's best epoch was at 52; cosine expired at ~75 — extending gives more room
    steps_per_epoch = max(1, len(dm.train_ds) // (args.micro_batch_size * n_gpus))
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)
    calibrated_total_steps = effective_steps_per_epoch * 225  # extended from 150
    total_steps = max(2000, calibrated_total_steps)  # min 2000 (up from parent's 1200)

    print(f"[Main] effective_steps_per_epoch={effective_steps_per_epoch}, "
          f"calibrated total_steps={total_steps} (targeting ~225 epochs)")

    # LightningModule
    lit = PerturbationLitModule(
        gnn_dim=args.gnn_dim,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        n_residual_layers=args.n_residual_layers,
        dropout=args.dropout,
        lr_muon=args.lr_muon,
        lr_adamw=args.lr_adamw,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        use_class_weights=args.use_class_weights,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        out_gene_emb_init=out_gene_emb_init,
        frozen_embeddings_np=dm.frozen_embeddings,
        edge_index=dm.edge_index,
        edge_weight=dm.edge_weight,
    )

    # Callbacks
    # Save top-5 checkpoints for checkpoint ensemble at test time
    # (parent saved only top-1; saving top-5 enables averaging across oscillation peaks)
    ensemble_top_k = args.ensemble_top_k if not args.fast_dev_run and args.debug_max_step is None else 0
    save_top_k = max(1, ensemble_top_k)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=save_top_k,  # Save top-k for ensemble
        save_last=True,
    )
    es_cb = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.patience,
        min_delta=1e-5,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    pb_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

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

    # ── Standard test with best single checkpoint ──────────────────────────
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    # ── Checkpoint Ensemble (rank 0 only, after distributed training) ──────
    # This is the primary improvement over the parent: averaging top-k checkpoints
    # smooths out the wide oscillations observed in Phase 2 of parent's training.
    # Expected: +0.002-0.005 F1 based on parent's oscillation pattern analysis.
    if (trainer.is_global_zero
            and ensemble_top_k > 1
            and not args.fast_dev_run
            and args.debug_max_step is None):
        try:
            # Collect top-k checkpoint paths sorted by val_f1 (best first)
            best_k_models = ckpt_cb.best_k_models  # dict {path: metric_value}
            if best_k_models:
                top_ckpts = sorted(
                    best_k_models.keys(),
                    key=lambda p: best_k_models[p],
                    reverse=True,
                )[:ensemble_top_k]

                print(f"[Ensemble] Top-{len(top_ckpts)} checkpoints for ensemble:")
                for rank_i, cp in enumerate(top_ckpts):
                    score = best_k_models[cp]
                    print(f"  [{rank_i+1}] val_f1={score:.4f} — {Path(cp).name}")

                # Run ensemble inference using the already-initialized module
                # (buffers like frozen_node_embs are already loaded from training)
                run_ensemble_inference(
                    lit=lit,
                    test_ds=dm.test_ds,
                    ckpt_paths=[str(cp) for cp in top_ckpts],
                    micro_batch_size=args.micro_batch_size,
                    num_workers=args.num_workers,
                    out_dir=out_dir,
                )
            else:
                print("[Ensemble] No best_k_models found, using single checkpoint prediction")
        except Exception as e:
            print(f"[Ensemble] WARNING: Ensemble inference failed: {e}")
            print("[Ensemble] Falling back to single-checkpoint prediction (already saved)")

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"Node 2-1-1-1-2-1-1 – Partial STRING_GNN Fine-Tuning + Deep Bilinear MLP "
            f"+ Checkpoint Ensemble + dropout=0.4 + label_smoothing=0.10 + extended schedule\n"
            f"Test results from trainer: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
