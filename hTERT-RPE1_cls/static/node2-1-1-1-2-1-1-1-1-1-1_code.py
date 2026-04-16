"""
Node 2-1-1-1-2-1-1-1-1-1-1 — Partial STRING_GNN + Deep Bilinear MLP
                             + SGDR(T0=20,Tmult=2) + FIXED Quality-Filtered Weighted SWA
                             + Dropout=0.45 + Label Smoothing=0.07 + Weight Decay=3e-3

Architecture (same as parent node2-1-1-1-2-1-1-1-1-1):
  - STRING_GNN backbone:
      * mps.0-5: frozen (precomputed embeddings through layer 5 stored in dataset)
      * mps.6, mps.7, post_mp: trainable (online graph forward with full graph,
        using precomputed mps.0-5 output as starting point, backbone_lr=1e-5 AdamW)
  - Output gene embedding initialization: STRING_GNN mps.5 embeddings (93.2% coverage)
  - 6-layer deep residual bilinear MLP head (rank=512, hidden=512, expand=4, dropout=0.45)
  - MuonWithAuxAdam optimizer:
      * Muon (lr=0.005) for hidden weight matrices in ResidualBlocks
      * AdamW (lr=5e-4) for head projections, embeddings, norms
      * AdamW (lr=1e-5, wd=3e-4) for trainable backbone layers (mps.6, mps.7, post_mp)
  - Class-weighted focal loss: gamma=2.0, weights=[2.0, 0.5, 4.0] for (down, neutral, up)
  - Label smoothing epsilon=0.07 (reduced from grandparent's 0.10 for better minority class recall)
  - SGDR (CosineAnnealingWarmRestarts): T_0=20, T_mult=2 for more restarts and richer pool
  - FIXED Quality-filtered weighted SWA: saves every 5 epochs, averages only top-K
    (top-15 by val_f1) with exponential weighting by val_f1 rank
    --- KEY FIX: uses regex to parse actual PL-generated checkpoint filenames ---
  - Patience=150, max_epochs=500 (reverted from parent's 200 to save compute)

Key differences from parent (node2-1-1-1-2-1-1-1-1-1, F1=0.5102):
  1. CRITICAL BUG FIX — Checkpoint filename parsing:
     Parent code expected: "periodic-EEEE-val_f1=X.XXXX.ckpt"
     PL actually generates: "periodic-epoch=EEEE-val_f1=val_f1=X.XXXX.ckpt"
     Fix: Use re.search(r'val_f1=(\d+\.\d+)\.ckpt$', filename) to extract val_f1 from actual names
     This unlocks the full quality-filtered SWA for 30+ qualifying periodic checkpoints

  2. SWA top_k: 12 -> 15 (use slightly larger pool since 30 checkpoints qualify)
     With 30 qualifying checkpoints, top-15 provides a richer ensemble than top-12
     while still excluding the lowest-quality configurations

  3. SWA weight temperature: 5.0 -> 3.0 (less aggressive exponential weighting)
     Temperature=5.0 concentrates >50% of weight on top-1 checkpoint (too close to single-ckpt)
     Temperature=3.0 gives a more balanced ensemble: top-3 checkpoints get ~55% of weight
     This should better capture the flat loss basin diversity

  4. Patience: 200 -> 150 (reverted to save compute; epochs 156-256 in parent added no benefit)

  5. swa_start_epoch: 15 -> 10 (slightly earlier start to collect more SGDR cycle 1 checkpoints)
     With T_0=20, cycle 1 runs epochs 0-20; starting at epoch 10 captures the late-cycle-1
     checkpoints which had val_f1 near 0.497 in the parent run

Rationale:
  - The parent's training dynamics were strong: best single checkpoint val_f1=0.5102,
    30 high-quality periodic checkpoints (val_f1 >= 0.497) across cycles 1-4.
  - The ONLY failure was the SWA checkpoint parsing bug. With 30 qualifying checkpoints
    and proper top-15 exponential-weighted SWA, we expect +0.004-0.007 F1 over baseline.
  - Temperature=3.0 creates a more balanced ensemble (not dominated by a single checkpoint)
    which should better generalize from the validation set's 157 samples to the test set.
  - top_k=15 (vs 12) makes better use of the rich pool of 30 qualifying checkpoints.
  - Expected F1: 0.5130-0.5155+ (surpassing tree-best of 0.5124)
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
    """Residual MLP block with dropout=0.45."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.45):
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
        dropout: float = 0.45,
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
    label_smoothing: float = 0.07,
) -> torch.Tensor:
    """Focal loss with class weighting and label smoothing (epsilon=0.07)."""
    B, C, G = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
    labels_flat = labels.reshape(-1)                        # [B*G]

    if label_smoothing > 0.0:
        # Smoothed soft-target loss
        one_hot = F.one_hot(labels_flat, num_classes=C).float()  # [B*G, C]
        smooth_targets = (1.0 - label_smoothing) * one_hot + label_smoothing / C

        log_probs = F.log_softmax(logits_flat, dim=1)  # [B*G, C]

        if class_weights is not None:
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


def parse_val_f1_from_checkpoint_path(ckpt_path: Path) -> float:
    """Parse val_f1 from checkpoint filename using regex.

    CRITICAL FIX: PyTorch Lightning generates filenames like:
        periodic-epoch=0009-val_f1=val_f1=0.4487.ckpt
    NOT the expected:
        periodic-0009-val_f1=0.4487.ckpt

    The `{val_f1:.4f}` format string in ModelCheckpoint filename
    causes PL to generate "val_f1=0.4487" because PL wraps each metric as
    "key=value". When the template string already contains "val_f1=", the
    result is "val_f1=val_f1=0.4487" (double prefix).

    This function uses regex to extract the last float number after the last
    "val_f1=" occurrence in the filename, which correctly handles all variants:
    - periodic-epoch=0009-val_f1=val_f1=0.4487.ckpt -> 0.4487
    - periodic-0009-val_f1=0.4487.ckpt -> 0.4487
    - best-epoch=0056-val_f1=0.5102.ckpt -> 0.5102
    - best-epoch=0056-val_f1=val_f1=0.5102.ckpt -> 0.5102
    """
    filename = ckpt_path.name
    # Match the last occurrence of a float after "val_f1=" in filename (before .ckpt)
    # This handles both single and double "val_f1=" prefix
    matches = re.findall(r'val_f1=(\d+\.\d+)', filename)
    if matches:
        # Take the last match (the actual float value, after any double-prefix)
        return float(matches[-1])

    # Fallback: try to find any float in the filename stem
    all_floats = re.findall(r'(\d+\.\d{3,4})', filename)
    if all_floats:
        # Return the last float that looks like a metric (0.x range)
        for f in reversed(all_floats):
            val = float(f)
            if 0.0 <= val <= 1.0:
                return val

    return 0.0  # Unknown, will be filtered by quality threshold


# ─── Quality-Filtered Weighted Multi-Checkpoint SWA Inference ─────────────────

def run_quality_filtered_swa_inference(
    checkpoint_info: List[Tuple[Path, float]],  # (ckpt_path, val_f1) pairs
    test_ds: "PerturbationDataset",
    lit_template: "PerturbationLitModule",
    micro_batch_size: int,
    num_workers: int,
    out_dir: Path,
    top_k: int = 15,
    val_f1_threshold: float = 0.497,
    swa_weight_temperature: float = 3.0,
) -> None:
    """Run inference using quality-filtered weighted multi-checkpoint SWA.

    FIXED over parent:
    1. Checkpoint val_f1 parsing now uses regex (parse_val_f1_from_checkpoint_path)
       to correctly handle actual PL-generated filenames
    2. top_k increased to 15 (from 12) to better utilize the rich 30-checkpoint pool
    3. swa_weight_temperature reduced to 3.0 (from 5.0) for more balanced ensemble

    Called on rank 0 only after distributed training completes.
    """
    if not checkpoint_info:
        print("[QualitySWA] No checkpoints found, skipping")
        return

    # Step 1: Apply quality filter
    qualified = [(p, f1) for p, f1 in checkpoint_info if f1 >= val_f1_threshold]
    if not qualified:
        # Fallback: use top-K regardless of threshold
        print(f"[QualitySWA] No checkpoints above threshold {val_f1_threshold}, using top-{top_k}")
        sorted_all = sorted(checkpoint_info, key=lambda x: x[1], reverse=True)
        qualified = sorted_all[:top_k]
    else:
        # Sort by val_f1 descending, keep top-K
        qualified = sorted(qualified, key=lambda x: x[1], reverse=True)[:top_k]

    print(f"[QualitySWA] Quality-filtered pool: {len(qualified)} checkpoints "
          f"(val_f1 range: [{qualified[-1][1]:.4f}, {qualified[0][1]:.4f}])")

    # Step 2: Compute exponential weights by val_f1 rank
    # rank 0 = best, rank N-1 = worst; temperature=3.0 (more balanced than parent's 5.0)
    n_ckpts = len(qualified)
    weights = np.array([np.exp(-rank * swa_weight_temperature / max(n_ckpts - 1, 1))
                        for rank in range(n_ckpts)], dtype=np.float64)
    weights = weights / weights.sum()
    print(f"[QualitySWA] Weights (sum=1.0): {[f'{w:.3f}' for w in weights]}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Step 3: Load all checkpoints and compute weighted average of parameters
    print(f"[QualitySWA] Loading and weighting {n_ckpts} checkpoints...")

    averaged_state: Optional[Dict[str, torch.Tensor]] = None
    non_float_state: Dict[str, torch.Tensor] = {}

    for rank_i, ((ckpt_path, val_f1), weight) in enumerate(zip(qualified, weights)):
        if not ckpt_path.exists():
            print(f"[QualitySWA] Skipping missing checkpoint: {ckpt_path}")
            continue
        try:
            ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            state = ckpt_data.get("state_dict", ckpt_data)

            if averaged_state is None:
                # Initialize with first checkpoint weighted by its weight
                averaged_state = {
                    k: v.float() * weight for k, v in state.items()
                    if isinstance(v, torch.Tensor) and torch.is_floating_point(v)
                }
                # Store non-float params separately (copy from first ckpt)
                non_float_state = {
                    k: v for k, v in state.items()
                    if not (isinstance(v, torch.Tensor) and torch.is_floating_point(v))
                }
            else:
                # Accumulate weighted parameters
                for k in averaged_state:
                    if k in state and isinstance(state[k], torch.Tensor):
                        averaged_state[k] = averaged_state[k] + state[k].float() * weight

            print(f"[QualitySWA] Loaded rank-{rank_i} ckpt "
                  f"(val_f1={val_f1:.4f}, weight={weight:.3f}): {ckpt_path.name}")

        except Exception as e:
            print(f"[QualitySWA] WARNING: Failed to load {ckpt_path}: {e}")
            continue

    if averaged_state is None:
        print("[QualitySWA] No checkpoints successfully loaded, skipping")
        return

    # Step 4: Merge averaged float params with non-float params and load into model
    merged_state = {**non_float_state, **averaged_state}

    # Load on CPU first, then move to device to avoid buffer device mismatch
    lit_template.cpu()
    lit_template.load_state_dict(merged_state)

    # Now move to device
    lit_template = lit_template.to(device)
    # Ensure buffers are on device after loading
    if hasattr(lit_template, "frozen_node_embs") and lit_template.frozen_node_embs is not None:
        lit_template.frozen_node_embs = lit_template.frozen_node_embs.to(device)
    if hasattr(lit_template, "edge_index_buf"):
        lit_template.edge_index_buf = lit_template.edge_index_buf.to(device)
    if hasattr(lit_template, "edge_weight_buf") and lit_template.edge_weight_buf is not None:
        lit_template.edge_weight_buf = lit_template.edge_weight_buf.to(device)

    lit_template.eval()

    # Step 5: Run inference
    test_loader = DataLoader(
        test_ds,
        batch_size=micro_batch_size * 2,
        shuffle=False,
        num_workers=min(num_workers, 4),
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    print(f"[QualitySWA] Running inference with {n_ckpts}-checkpoint quality-filtered weighted SWA...")

    probs_list: List[torch.Tensor] = []
    pert_ids_list: List[str] = []
    syms_list: List[str] = []

    with torch.no_grad():
        for batch in test_loader:
            emb = batch["embedding"].float().to(device)
            node_idx = batch["node_idx"].to(device)
            in_vocab = batch["in_vocab"].to(device)

            final_emb = lit_template._get_final_embeddings(emb, node_idx, in_vocab)
            logits = lit_template.head(final_emb, in_vocab)  # [B, 3, 6640]
            probs = torch.softmax(logits, dim=1).float().cpu()
            probs_list.append(probs)
            pert_ids_list.extend(batch["pert_id"])
            syms_list.extend(batch["symbol"])

    all_probs = torch.cat(probs_list, dim=0).numpy()  # [N, 3, 6640]
    print(f"[QualitySWA] Inference complete: {all_probs.shape[0]} samples from "
          f"{n_ckpts}-checkpoint quality-filtered weighted average")

    # Write predictions (overwrite single-checkpoint predictions)
    pred_path = out_dir / "test_predictions.tsv"
    seen_ids: set = set()
    with open(pred_path, "w") as fh:
        fh.write("idx\tinput\tprediction\n")
        for pert_id, symbol, probs in zip(pert_ids_list, syms_list, all_probs):
            if pert_id not in seen_ids:
                seen_ids.add(pert_id)
                fh.write(f"{pert_id}\t{symbol}\t{json.dumps(probs.tolist())}\n")

    print(f"[QualitySWA] Saved {len(seen_ids)} quality-filtered weighted SWA predictions -> {pred_path}")
    print(f"[QualitySWA] Top-{n_ckpts} quality-filtered SWA (temp={swa_weight_temperature}) complete")


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """LightningModule for gene-perturbation DEG prediction.

    Architecture:
    - mps.0-5: frozen (precomputed embeddings stored in dataset as features)
    - mps.6, mps.7, post_mp: trainable (online graph forward at each training step)
    - 6-layer bilinear residual MLP head (rank=512, dropout=0.45)
    - MuonWithAuxAdam optimizer
    - Class-weighted focal loss with label smoothing epsilon=0.07
    - SGDR (CosineAnnealingWarmRestarts) T_0=20 for more diverse checkpoints
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_residual_layers: int = 6,
        dropout: float = 0.45,
        lr_muon: float = 0.005,
        lr_adamw: float = 5e-4,
        backbone_lr: float = 1e-5,
        weight_decay: float = 3e-3,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.07,
        use_class_weights: bool = True,
        warmup_steps: int = 100,
        sgdr_t0: int = 20,
        sgdr_t_mult: int = 2,
        sgdr_eta_min: float = 1e-6,
        out_gene_emb_init: Optional[np.ndarray] = None,
        # Backbone data for setup
        frozen_embeddings_np: Optional[np.ndarray] = None,  # [N_nodes, 256]
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        steps_per_epoch: int = 22,      # For SGDR step conversion
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
        """Apply trainable backbone layers (mps.6, mps.7, post_mp) to get final embeddings."""
        if self.frozen_node_embs is None:
            return batch_embeddings

        # Run full graph through trainable layers: [N_nodes, 256] -> [N_nodes, 256]
        full_final_embs = self.backbone_adapter(
            self.frozen_node_embs.float(),
            self.edge_index_buf,
            self.edge_weight_buf,
        )  # [N_nodes, 256]

        # Retrieve embeddings for the batch genes
        B = batch_embeddings.shape[0]
        device = batch_embeddings.device
        result = torch.zeros(B, batch_embeddings.shape[1], device=device, dtype=torch.float32)

        in_vocab_mask = in_vocab & (node_indices >= 0)
        if in_vocab_mask.any():
            valid_indices = node_indices[in_vocab_mask]
            result[in_vocab_mask] = full_final_embs[valid_indices].float()

        # OOV genes: use zeros (replaced by oov_embedding in the head's forward method)
        return result

    def forward(
        self,
        embedding: torch.Tensor,    # [B, 256] - frozen mps.0-5 embedding
        node_indices: torch.Tensor,  # [B] - STRING node indices
        in_vocab: torch.Tensor,      # [B] bool
    ) -> torch.Tensor:
        final_emb = self._get_final_embeddings(embedding, node_indices, in_vocab)
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

            self.print(f"[Node2-1-1-1-2-1-1-1-1-1-1] Saved test predictions -> {pred_path} ({len(seen_ids)} samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node2-1-1-1-2-1-1-1-1-1-1] Self-computed test F1 (best ckpt) = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

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
                    weight_decay=hp.weight_decay * 0.1,  # 3e-4 (lighter backbone regularization)
                ),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
            print(f"[Optimizer] MuonWithAuxAdam: Muon lr={hp.lr_muon}, "
                  f"AdamW head lr={hp.lr_adamw}, AdamW backbone lr={hp.backbone_lr}, "
                  f"weight_decay={hp.weight_decay}")

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

        # SGDR (Cosine Annealing with Warm Restarts)
        # T_0=20 epochs × steps_per_epoch (shorter cycles for more diverse SWA pool)
        T_0_steps = hp.sgdr_t0 * hp.steps_per_epoch
        T_0_steps = max(T_0_steps, 50)  # Minimum T_0 for safety

        def lr_lambda_sgdr(step: int) -> float:
            if step < hp.warmup_steps:
                return float(step) / max(1, hp.warmup_steps)
            # SGDR after warmup
            t = step - hp.warmup_steps
            # Find which cycle we're in
            cycle_len = T_0_steps
            cycle = 0
            while t >= cycle_len:
                t -= cycle_len
                cycle += 1
                cycle_len = T_0_steps * (hp.sgdr_t_mult ** cycle)
            # Cosine within current cycle
            progress = t / max(1, cycle_len)
            progress = min(progress, 1.0)
            cos_val = 0.5 * (1.0 + np.cos(np.pi * progress))
            # Scale between eta_min and 1.0
            return hp.sgdr_eta_min / hp.lr_adamw + (1.0 - hp.sgdr_eta_min / hp.lr_adamw) * cos_val

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_sgdr)
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-1-1-1-2-1-1-1-1-1-1 – Partial STRING_GNN + Deep Bilinear MLP + "
                    "SGDR(T0=20) + FIXED Quality-Filtered Weighted SWA + Label Smoothing=0.07"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--gnn-dim",            type=int,   default=256)
    p.add_argument("--hidden-dim",         type=int,   default=512)
    p.add_argument("--rank",               type=int,   default=512)
    p.add_argument("--n-residual-layers",  type=int,   default=6)
    p.add_argument("--dropout",            type=float, default=0.45)
    p.add_argument("--lr-muon",            type=float, default=0.005)
    p.add_argument("--lr-adamw",           type=float, default=5e-4)
    p.add_argument("--backbone-lr",        type=float, default=1e-5,
                   help="LR for trainable backbone layers (mps.6, mps.7, post_mp)")
    p.add_argument("--weight-decay",       type=float, default=3e-3)
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--label-smoothing",    type=float, default=0.07,
                   help="Label smoothing (0.07: reduced from grandparent's 0.10)")
    p.add_argument("--use-class-weights",  action="store_true", default=True)
    p.add_argument("--no-class-weights",   dest="use_class_weights", action="store_false")
    p.add_argument("--warmup-steps",       type=int,   default=100)
    p.add_argument("--sgdr-t0",            type=int,   default=20,
                   help="SGDR first cycle length in epochs (T_0=20 shorter cycles)")
    p.add_argument("--sgdr-t-mult",        type=int,   default=2,
                   help="SGDR cycle multiplier (T_mult=2 doubles each cycle)")
    p.add_argument("--sgdr-eta-min",       type=float, default=1e-6,
                   help="SGDR minimum LR (absolute, not ratio)")
    p.add_argument("--swa-start-epoch",    type=int,   default=10,
                   help="Epoch to start saving periodic SWA checkpoints (earlier: 10 vs parent's 15)")
    p.add_argument("--swa-every-n-epochs", type=int,   default=5,
                   help="Save periodic checkpoint every N epochs (denser with T_0=20)")
    p.add_argument("--swa-top-k",          type=int,   default=15,
                   help="Quality-filtered SWA: use top-K checkpoints by val_f1 (15 vs parent's 12)")
    p.add_argument("--swa-val-f1-threshold", type=float, default=0.497,
                   help="Quality-filtered SWA: minimum val_f1 for checkpoint inclusion")
    p.add_argument("--swa-weight-temperature", type=float, default=3.0,
                   help="Temperature for exponential weighting by val_f1 rank (3.0 vs parent's 5.0)")
    p.add_argument("--use-swa",            action="store_true", default=True,
                   help="Enable quality-filtered weighted SWA for test time (default: True)")
    p.add_argument("--no-swa",             dest="use_swa", action="store_false")
    p.add_argument("--micro-batch-size",   type=int,   default=16)
    p.add_argument("--global-batch-size",  type=int,   default=64)
    p.add_argument("--max-epochs",         type=int,   default=500,
                   help="Max epochs for SGDR cycles")
    p.add_argument("--patience",           type=int,   default=150,
                   help="Early stopping patience (reduced from parent's 200; epochs 156-256 were wasteful)")
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
    return p.parse_args()


def build_out_gene_emb_init(
    frozen_embeddings: np.ndarray,
    node_name_to_idx: Dict[str, int],
    label_gene_ids: List[str],
    rank: int,
) -> Optional[np.ndarray]:
    """Build output gene embedding initializations from STRING_GNN intermediate embeddings."""
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

    # Estimate effective steps per epoch for SGDR calibration
    steps_per_epoch = max(1, len(dm.train_ds) // (args.micro_batch_size * n_gpus))
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)

    print(f"[Main] effective_steps_per_epoch={effective_steps_per_epoch}")
    print(f"[Main] SGDR: T_0={args.sgdr_t0} epochs, T_mult={args.sgdr_t_mult}")
    print(f"[Main] SGDR cycles: 0-{args.sgdr_t0}, {args.sgdr_t0}-{args.sgdr_t0*(1+args.sgdr_t_mult)}, ...")
    print(f"[Main] FIXED Quality SWA: top-{args.swa_top_k} by val_f1, threshold={args.swa_val_f1_threshold}, temp={args.swa_weight_temperature}")
    print(f"[Main] weight_decay={args.weight_decay}, dropout={args.dropout}, label_smooth={args.label_smoothing}")
    print(f"[Main] patience={args.patience} (reduced from parent's 200 to save compute)")

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
        sgdr_t0=args.sgdr_t0,
        sgdr_t_mult=args.sgdr_t_mult,
        sgdr_eta_min=args.sgdr_eta_min,
        out_gene_emb_init=out_gene_emb_init,
        frozen_embeddings_np=dm.frozen_embeddings,
        edge_index=dm.edge_index,
        edge_weight=dm.edge_weight,
        steps_per_epoch=effective_steps_per_epoch,
    )

    # ── Callbacks ──────────────────────────────────────────────────────────
    # Primary: Best val_f1 checkpoint (for single-checkpoint test prediction)
    ckpt_best = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=1,
        save_last=True,
    )

    # Secondary: Periodic checkpoints every swa_every_n_epochs starting from swa_start_epoch
    # NOTE: PL generates filename as "periodic-epoch=EEEE-val_f1=val_f1=X.XXXX.ckpt"
    # because {val_f1:.4f} in ModelCheckpoint filename is wrapped with "val_f1=" by PL
    # The parse_val_f1_from_checkpoint_path() function handles this correctly via regex
    ckpt_periodic = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints" / "periodic"),
        filename="periodic-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=-1,              # Save all (unlimited)
        every_n_epochs=args.swa_every_n_epochs,
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

    # Always use DDPStrategy so that torch.distributed is initialized even for 1 GPU.
    # This is required because the Muon optimizer calls dist.get_world_size() internally.
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
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[ckpt_best, ckpt_periodic, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    # ── Training ───────────────────────────────────────────────────────────
    trainer.fit(lit, datamodule=dm)

    # ── Standard test with best single checkpoint ──────────────────────────
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    # ── Quality-Filtered Weighted Multi-Checkpoint SWA Post-hoc Inference ─────
    # KEY FIX: Use parse_val_f1_from_checkpoint_path() with regex to handle
    # actual PL-generated filenames like "periodic-epoch=0009-val_f1=val_f1=0.4487.ckpt"
    if (trainer.is_global_zero
            and args.use_swa
            and not args.fast_dev_run
            and args.debug_max_step is None):
        try:
            # Collect all periodic checkpoints with val_f1 from filename
            periodic_ckpt_dir = out_dir / "checkpoints" / "periodic"
            all_periodic = sorted(periodic_ckpt_dir.glob("periodic-*.ckpt")) if periodic_ckpt_dir.exists() else []

            # FIXED: Parse val_f1 from checkpoint filename using regex
            # Handles PL's actual format: periodic-epoch=0009-val_f1=val_f1=0.4487.ckpt
            checkpoint_info: List[Tuple[Path, float]] = []
            for ckpt_p in all_periodic:
                val_f1 = parse_val_f1_from_checkpoint_path(ckpt_p)

                # Parse epoch number to filter by swa_start_epoch
                epoch_match = re.search(r'epoch=(\d+)', ckpt_p.name)
                if epoch_match:
                    epoch_num = int(epoch_match.group(1))
                else:
                    # Try stem split for legacy format
                    try:
                        parts = ckpt_p.stem.split("-")
                        epoch_num = int(parts[1]) if len(parts) > 1 else 0
                    except (IndexError, ValueError):
                        epoch_num = 0

                if epoch_num >= args.swa_start_epoch:
                    checkpoint_info.append((ckpt_p, val_f1))
                    print(f"[SWA] Parsed: {ckpt_p.name} -> epoch={epoch_num}, val_f1={val_f1:.4f}")

            # Also include best checkpoint
            best_ckpt_path = ckpt_best.best_model_path
            last_ckpt_path = ckpt_best.last_model_path

            # FIXED: Parse best val_f1 from best checkpoint filename using regex
            best_val_f1 = 0.0
            if best_ckpt_path:
                best_val_f1 = parse_val_f1_from_checkpoint_path(Path(best_ckpt_path))
                if Path(best_ckpt_path).exists():
                    periodic_paths = {str(p.resolve()) for p, _ in checkpoint_info}
                    if str(Path(best_ckpt_path).resolve()) not in periodic_paths:
                        checkpoint_info.append((Path(best_ckpt_path), best_val_f1))
                        print(f"[SWA] Added best ckpt: {Path(best_ckpt_path).name} -> val_f1={best_val_f1:.4f}")

            # Optionally include last checkpoint
            if last_ckpt_path and Path(last_ckpt_path).exists() and last_ckpt_path != best_ckpt_path:
                last_val_f1 = parse_val_f1_from_checkpoint_path(Path(last_ckpt_path))
                periodic_paths = {str(p.resolve()) for p, _ in checkpoint_info}
                if str(Path(last_ckpt_path).resolve()) not in periodic_paths:
                    checkpoint_info.append((Path(last_ckpt_path), last_val_f1))
                    print(f"[SWA] Added last ckpt: {Path(last_ckpt_path).name} -> val_f1={last_val_f1:.4f}")

            print(f"[SWA] Total checkpoint pool: {len(checkpoint_info)} checkpoints")
            if checkpoint_info:
                val_f1s = [f1 for _, f1 in checkpoint_info]
                qualifying = [f1 for f1 in val_f1s if f1 >= args.swa_val_f1_threshold]
                print(f"[SWA] val_f1 range: [{min(val_f1s):.4f}, {max(val_f1s):.4f}]")
                print(f"[SWA] Qualifying (>={args.swa_val_f1_threshold}): {len(qualifying)} checkpoints")

            if len(checkpoint_info) >= 1:
                run_quality_filtered_swa_inference(
                    checkpoint_info=checkpoint_info,
                    test_ds=dm.test_ds,
                    lit_template=lit,
                    micro_batch_size=args.micro_batch_size,
                    num_workers=args.num_workers,
                    out_dir=out_dir,
                    top_k=args.swa_top_k,
                    val_f1_threshold=args.swa_val_f1_threshold,
                    swa_weight_temperature=args.swa_weight_temperature,
                )
            else:
                print("[SWA] No checkpoints available for quality-filtered SWA")

        except Exception as e:
            print(f"[SWA] WARNING: Quality-filtered SWA inference failed: {e}")
            import traceback
            traceback.print_exc()
            print("[SWA] Falling back to single-checkpoint prediction (already saved)")

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"Node 2-1-1-1-2-1-1-1-1-1-1 – Partial STRING_GNN Fine-Tuning + Deep Bilinear MLP "
            f"+ SGDR(T0=20,Tmult=2) + FIXED Quality-Filtered Weighted SWA (top-{args.swa_top_k}, "
            f"threshold={args.swa_val_f1_threshold}, temp={args.swa_weight_temperature}) "
            f"+ dropout=0.45 + label_smoothing=0.07 + weight_decay=3e-3\n"
            f"Test results from trainer (best single checkpoint): {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py on SWA predictions)\n"
        )


if __name__ == "__main__":
    main()
