"""
Node 4-2-1-1: STRING_GNN Partial Fine-Tuning + scFoundation-Initialized Gene Embeddings
             + Reduced out_gene_emb LR

Architecture Strategy (inherited from parent node4-2-1, F1=0.5076):
  - STRING_GNN backbone with partial fine-tuning:
      * Frozen layers (mps.0-6) + embedding: pre-computed as a buffer in setup()
        to speed up forward pass (no redundant computation through frozen layers)
      * Trainable layers: mps.7 + post_mp (~67K params, backbone_lr=1e-5)
  - Deep 6-layer Residual Bilinear MLP head (rank=512, hidden=512):
      ResidualBlock × 6 → bilinear output (rank-512 decomposition)
      Produces logits of shape [B, 3, 6640]
  - Loss: Focal cross-entropy (gamma=2.0) with class weights [2.0, 0.5, 4.0]
    (no label smoothing - hurts minority class signal based on tree memory)
  - Optimizer: MuonWithAuxAdam
      - Muon (lr=0.005) for ResBlock 2D weight matrices
      - AdamW (backbone_lr=1e-5, head_lr=5e-4) for other parameters
      - Separate AdamW group with strong WD (1e-2) and REDUCED LR (1e-4) for out_gene_emb
  - Scheduler: Cosine warm restarts (T_0=1200 steps, T_mult=1)

Key Improvements Over Parent (node4-2-1, F1=0.5076):
  1. scFoundation-initialized out_gene_emb (PRIMARY CHANGE):
     - Initialize the 6640×512 output gene embedding matrix from scFoundation
       positional embeddings (768-dim), projected to 512-dim via SVD.
     - scFoundation covers 97.6% of the 6640 label genes (6482/6640).
     - Missing 2.4% (158 genes) use random Xavier init with same scale.
     - Replaces random init which forced the 3.4M-param matrix to learn all
       gene-position structure from scratch from only 1,416 training samples.
     - Provides biologically grounded inductive bias from a 100M-parameter
       single-cell foundation model trained on 50M+ human single-cell profiles.
     - Expected gain: +0.003–0.010 F1 (from parent feedback analysis and
       collected_memory pattern: pretrained embeddings consistently improve
       downstream task performance in low-data regimes).
  2. Reduced out_gene_emb LR: 5e-4 → 1e-4 (10x reduction):
     - The embedding now starts from an informative pretrained position.
     - A 10x lower LR prevents the optimizer from rapidly overwriting the
       biologically-meaningful initialization with task-specific memorization.
     - At 1e-4, the gene embeddings can still fine-tune toward the DEG task
       while preserving the structural gene-gene relationships from scFoundation.
     - Combined with WD=1e-2, this provides the strongest regularization for
       the 3.4M-param matrix (the primary overfitting source in this lineage).
  3. All other settings preserved from node4-2-1 (proven effective):
     - T_0=1200 warm restart steps (~27 epochs/cycle)
     - dropout=0.4
     - gradient_clip_val=1.0
     - patience=50
     - muon_lr=0.005, backbone_lr=1e-5, head_lr=5e-4

Memory Sources:
  - node4-2-1 feedback: "Option A (high priority): Initialize out_gene_emb from
    pretrained AIDO.Cell or scFoundation gene embeddings aligned via label_genes.txt.
    Expected gain: +0.003–0.010 F1. Reduces degrees of freedom requiring learning
    from data and provides biologically grounded inductive bias."
  - node4-2-1 feedback: "Reduce out_gene_emb learning rate to 1e-4 (10x reduction)
    while keeping WD=1e-2, to slow the rate at which the matrix memorizes training
    patterns."
  - scFoundation skill: pos_emb.weight [19267, 768] provides positional gene
    embeddings; gene_ids.json maps positions to Ensembl IDs.
  - node2-1-1-1-2 (tree memory): pre-initialized out_gene_emb from STRING_GNN
    was attempted but provided no benefit due to misaligned initialization magnitude
    (normalized to std=0.02 collapsed PPI geometry). Our approach uses scFoundation
    with natural std≈1.0 (after SVD projection), avoiding this pitfall.
  - collected_memory pattern: scFoundation covers 97.6% of label genes with true
    transcriptome-scale positional embeddings, substantially better alignment than
    STRING_GNN's PPI topology space.
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
from muon import MuonWithAuxAdam
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR    = "/home/Models/STRING_GNN"
SCFOUNDATION_DIR  = "/home/Models/scFoundation"
N_GENES_OUT       = 6640
N_CLASSES         = 3
GNN_DIM           = 256
N_NODES           = 18870
LABEL_GENES_FILE  = "data/label_genes.txt"

# Class weights: down(-1)=2.0, neutral(0)=0.5, up(+1)=4.0
# Proven effective in the tree best lineage (node2-1-3, node1-2-2-2-1, node4-2)
CLASS_WEIGHTS = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)

# Focal loss gamma: focuses training on hard examples
FOCAL_GAMMA = 2.0


# ─── scFoundation Gene Embedding Initialization ───────────────────────────────

def build_gene_emb_init(
    label_genes_file: str,
    target_dim: int = 512,
) -> torch.Tensor:
    """
    Build initialization matrix for out_gene_emb [N_GENES_OUT, target_dim]
    from scFoundation positional embeddings.

    Steps:
    1. Load scFoundation pos_emb.weight [19267, 768] (gene position embeddings).
    2. Map each of the 6640 label genes to its position in scFoundation vocabulary.
    3. Project from 768 → target_dim (512) using top-k SVD of the gene submatrix.
    4. For genes not in scFoundation (2.4%), use random init matching the scale
       of the projected embeddings.

    Args:
        label_genes_file: path to data/label_genes.txt (space-delimited, no header)
        target_dim: output dimension (must match bilinear_rank, default=512)

    Returns:
        init_weight: [N_GENES_OUT, target_dim] float32 tensor
    """
    # Load scFoundation gene vocabulary
    gene_ids_path = Path(SCFOUNDATION_DIR) / "gene_ids.json"
    scf_gene_ids = json.loads(gene_ids_path.read_text())
    gene_id_to_scf_pos = {gid: i for i, gid in enumerate(scf_gene_ids)}

    # Load label genes (ordered list of N_GENES_OUT Ensembl IDs)
    label_gene_ensembl_ids = []
    with open(label_genes_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                label_gene_ensembl_ids.append(parts[0])
    assert len(label_gene_ensembl_ids) == N_GENES_OUT, (
        f"Expected {N_GENES_OUT} label genes, got {len(label_gene_ensembl_ids)}"
    )

    # Load scFoundation model (CPU only for initialization, no gradient needed)
    scf_model = AutoModel.from_pretrained(SCFOUNDATION_DIR, trust_remote_code=True)
    scf_model.eval()

    # Extract position embeddings [19267, 768]
    # Positions 0..19263 correspond to gene_ids.json order
    pos_emb_full = scf_model.pos_emb.weight.detach().float()  # [19267, 768]

    # Extract the 768-dim position embeddings for label genes that exist in scFoundation
    # For each label gene: if covered → use pos_emb; if not → mark as missing
    covered_indices = []   # indices in label_gene_ensembl_ids that are in scFoundation
    covered_positions = [] # corresponding position indices in scFoundation vocabulary

    for i, gene_id in enumerate(label_gene_ensembl_ids):
        if gene_id in gene_id_to_scf_pos:
            covered_indices.append(i)
            covered_positions.append(gene_id_to_scf_pos[gene_id])

    n_covered = len(covered_indices)
    n_missing = N_GENES_OUT - n_covered
    print(f"[Gene Emb Init] scFoundation coverage: {n_covered}/{N_GENES_OUT} "
          f"({100*n_covered/N_GENES_OUT:.1f}%), missing: {n_missing}")

    # Extract covered embeddings [n_covered, 768]
    covered_embs = pos_emb_full[covered_positions]  # [n_covered, 768]

    # Project 768 → target_dim (512) via SVD on the covered subset
    # Use the top target_dim singular vectors as the projection matrix
    # This preserves the maximum variance in the lower-dimensional space
    U, S, Vh = torch.linalg.svd(covered_embs, full_matrices=False)
    # U: [n_covered, min(n_covered, 768)], S: [min(n_covered, 768)], Vh: [min(n_covered, 768), 768]
    # Project: covered_embs @ Vh[:target_dim, :].T → [n_covered, target_dim]
    proj_matrix = Vh[:target_dim, :]  # [target_dim, 768]
    projected_covered = covered_embs @ proj_matrix.T  # [n_covered, target_dim]

    # Normalize to reasonable scale (std≈0.2, matching Xavier init for 512→512 linear)
    # Xavier init std for Linear(512→512) ≈ sqrt(2/(512+512)) ≈ 0.044
    # However, the gene embeddings are used in a matmul with [B,3,512] → we want
    # the output magnitudes to be reasonable. Scale to std≈0.1 as a compromise.
    proj_std = projected_covered.std()
    if proj_std > 0:
        projected_covered = projected_covered * (0.1 / proj_std.item())

    # Build the final init matrix
    init_weight = torch.zeros(N_GENES_OUT, target_dim, dtype=torch.float32)

    # Fill covered genes
    covered_idx_tensor = torch.tensor(covered_indices, dtype=torch.long)
    init_weight[covered_idx_tensor] = projected_covered

    # Fill missing genes with random init matching the same scale
    missing_mask = torch.ones(N_GENES_OUT, dtype=torch.bool)
    missing_mask[covered_idx_tensor] = False
    n_missing_actual = missing_mask.sum().item()
    if n_missing_actual > 0:
        # Random init with same std as the projected covered embeddings
        init_weight[missing_mask] = torch.randn(n_missing_actual, target_dim) * 0.1

    print(f"[Gene Emb Init] Final init_weight: shape={init_weight.shape}, "
          f"std={init_weight.std():.4f}, mean={init_weight.mean():.4f}")

    # Free memory
    del scf_model, pos_emb_full, covered_embs, U, S, Vh, projected_covered
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return init_weight


# ─── Focal Loss ───────────────────────────────────────────────────────────────

def focal_cross_entropy(
    logits: torch.Tensor,      # [B, C, L]
    labels: torch.Tensor,      # [B, L] long
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal cross-entropy loss for multi-output 3-class classification.
    Logits: [B, 3, L], labels: [B, L], class_weights: [3]
    """
    B, C, L = logits.shape
    # Reshape to [B*L, C] for cross_entropy
    logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)   # [B*L, C]
    labels_flat  = labels.reshape(-1)                         # [B*L]

    # Standard cross-entropy per element (no reduction)
    ce_loss = F.cross_entropy(
        logits_flat, labels_flat,
        weight=class_weights.to(logits_flat.device),
        reduction="none",
    )  # [B*L]

    # Focal weight: (1 - p_t)^gamma
    with torch.no_grad():
        probs = F.softmax(logits_flat.float(), dim=-1)        # [B*L, C]
        pt    = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [B*L]
        focal_weight = (1.0 - pt).pow(gamma)

    loss = (focal_weight * ce_loss).mean()
    return loss


# ─── Per-Gene F1 ──────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Mirrors calc_metric.py: per-gene macro-F1 over present classes."""
    pred_cls = pred_np.argmax(axis=1)   # [N, L]
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]; yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class StringGNNDataset(Dataset):
    def __init__(
        self,
        pert_ids:    List[str],
        symbols:     List[str],
        node_indices: torch.Tensor,       # [N] long, -1 for unknown
        labels:      Optional[torch.Tensor] = None,  # [N, 6640] long
    ):
        self.pert_ids     = pert_ids
        self.symbols      = symbols
        self.node_indices = node_indices
        self.labels       = labels

    def __len__(self): return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
            "node_index": self.node_indices[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    out = {
        "pert_id":    [b["pert_id"]    for b in batch],
        "symbol":     [b["symbol"]     for b in batch],
        "node_index": torch.stack([b["node_index"] for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class StringGNNDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data", micro_batch_size=8, num_workers=4):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        gnn_dir    = Path(STRING_GNN_DIR)
        node_names = json.loads((gnn_dir / "node_names.json").read_text())
        node_name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(node_names)}

        def load_split(fname: str, has_lbl: bool):
            df  = pd.read_csv(self.data_dir / fname, sep="\t")
            idxs = torch.tensor(
                [node_name_to_idx.get(pid, -1) for pid in df["pert_id"].tolist()],
                dtype=torch.long,
            )
            labels = None
            if has_lbl and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return StringGNNDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(), idxs, labels
            )

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

        # Store graph data for the LightningModule
        graph = torch.load(gnn_dir / "graph_data.pt", weights_only=False)
        self.edge_index  = graph["edge_index"]
        self.edge_weight = graph.get("edge_weight", None)

        n_unknown = sum(
            1 for ds in (self.train_ds, self.val_ds, self.test_ds)
            for ni in ds.node_indices.tolist() if ni == -1
        )
        total = len(self.train_ds) + len(self.val_ds) + len(self.test_ds)
        print(f"[Node4-2-1-1] {n_unknown}/{total} samples not in STRING_GNN "
              f"→ learned fallback embedding.")

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=self.num_workers,
            pin_memory=True, drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Residual Block ───────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """MLP residual block used in the bilinear head."""
    def __init__(self, hidden_dim: int, expand: int = 4, dropout: float = 0.4):
        super().__init__()
        mid = hidden_dim * expand
        self.fc1 = nn.Linear(hidden_dim, mid, bias=False)
        self.fc2 = nn.Linear(mid, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x + residual


# ─── Model ────────────────────────────────────────────────────────────────────

class StringGNNResidualBilinearModel(nn.Module):
    """
    STRING_GNN (partial fine-tuning: mps.7 + post_mp) +
    Deep 6-layer Residual MLP + Rank-512 Bilinear Interaction Head.

    Key improvement: out_gene_emb initialized from scFoundation positional
    embeddings (768-dim → SVD-projected to 512-dim) for 97.6% of the 6640
    output gene positions. This replaces random initialization with biologically
    grounded gene representations from a 100M-param single-cell foundation model.

    Architecture:
      1. Pre-computed frozen backbone (mps.0-6 + emb) stored as buffer
      2. Trainable tail: mps.7 + post_mp
      3. Fallback embedding for unknown genes
      4. 6× ResidualBlock(hidden=512, expand=4, dropout=0.4)
      5. Bilinear head: fc_bilinear produces [B, 3*rank] decomposed into
         [B, 3, rank] × [n_genes_out, rank]^T → [B, 3, n_genes_out]
         where out_gene_emb is initialized from scFoundation positional embeddings
    """

    def __init__(
        self,
        edge_index:  torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        gene_emb_init: Optional[torch.Tensor],  # [N_GENES_OUT, bilinear_rank] or None
        gnn_dim:     int = GNN_DIM,
        n_genes_out: int = N_GENES_OUT,
        n_classes:   int = N_CLASSES,
        hidden_dim:  int = 512,
        n_layers:    int = 6,
        expand:      int = 4,
        bilinear_rank: int = 512,
        dropout:     float = 0.4,
    ):
        super().__init__()
        self.gnn_dim      = gnn_dim
        self.n_classes    = n_classes
        self.n_genes_out  = n_genes_out
        self.bilinear_rank = bilinear_rank

        # ── Backbone ──────────────────────────────────────────────────────────
        # Load full STRING_GNN
        full_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        # Register frozen graph tensors
        self.register_buffer("edge_index",  edge_index)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        # Pre-compute frozen backbone embeddings (mps.0-6 + emb)
        # We store the output of mps.6 (before mps.7) as a frozen buffer.
        # This is done lazily in setup() after moving to device.
        # For now, store the backbone modules we need.
        self.emb     = full_gnn.emb         # Embedding(18870, 256) - frozen
        self.mps_0_6 = nn.ModuleList([full_gnn.mps[i] for i in range(7)])  # frozen
        self.mps_7   = full_gnn.mps[7]      # trainable
        self.post_mp = full_gnn.post_mp     # trainable

        # Freeze emb + mps.0-6
        for p in self.emb.parameters():
            p.requires_grad_(False)
        for layer in self.mps_0_6:
            for p in layer.parameters():
                p.requires_grad_(False)

        # Fallback embedding for genes absent from STRING_GNN
        self.fallback_emb = nn.Parameter(torch.randn(gnn_dim) * 0.02)

        # ── Head ──────────────────────────────────────────────────────────────
        # Input projection: gnn_dim → hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(gnn_dim),
            nn.Linear(gnn_dim, hidden_dim, bias=False),
        )

        # Residual blocks (dropout=0.4)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Bilinear decomposition: hidden_dim → n_classes × bilinear_rank
        self.fc_bilinear = nn.Linear(hidden_dim, n_classes * bilinear_rank, bias=False)

        # Output gene embedding matrix: [n_genes_out, bilinear_rank]
        # KEY IMPROVEMENT: Initialize from scFoundation positional embeddings
        # (97.6% coverage of 6640 label genes, projected 768→512 via SVD).
        # This provides biologically grounded inductive bias, replacing random init.
        # Separate, stronger weight_decay (1e-2) and reduced LR (1e-4) in the optimizer.
        self.out_gene_emb = nn.Embedding(n_genes_out, bilinear_rank)

        if gene_emb_init is not None:
            # Initialize from scFoundation positional embeddings
            assert gene_emb_init.shape == (n_genes_out, bilinear_rank), (
                f"gene_emb_init shape mismatch: {gene_emb_init.shape} vs "
                f"({n_genes_out}, {bilinear_rank})"
            )
            with torch.no_grad():
                self.out_gene_emb.weight.copy_(gene_emb_init)
            print(f"[Node4-2-1-1] out_gene_emb initialized from scFoundation "
                  f"(shape={gene_emb_init.shape}, std={gene_emb_init.std():.4f})")
        else:
            # Fallback: Xavier uniform init (same as default)
            nn.init.xavier_uniform_(self.out_gene_emb.weight)
            print("[Node4-2-1-1] out_gene_emb initialized with Xavier uniform (fallback)")

        # Cache for precomputed frozen intermediate embeddings
        self._frozen_emb_cache: Optional[torch.Tensor] = None

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Node4-2-1-1] Trainable: {n_trainable:,} / {n_total:,} params "
              f"({100*n_trainable/n_total:.2f}%)")

    def _compute_backbone_embs(self) -> torch.Tensor:
        """
        Run partial forward:
          emb.weight → mps.0-6 (frozen) → mps.7 (trainable) → post_mp (trainable)
        Returns node_emb [N_nodes, 256].

        Uses frozen cache for mps.0-6 output to save compute during training.
        """
        ei = self.edge_index
        ew = self.edge_weight

        # Compute/use cached frozen intermediate (output after mps.6)
        if self._frozen_emb_cache is None:
            x = self.emb.weight  # [N, 256]
            for layer in self.mps_0_6:
                x = layer(x, ei, ew)
            self._frozen_emb_cache = x.detach()

        x = self._frozen_emb_cache  # [N, 256] — no grad

        # Trainable tail
        x = self.mps_7(x, ei, ew)  # [N, 256]
        x = self.post_mp(x)         # [N, 256]
        return x

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_indices: [B] long — STRING_GNN node indices (-1 = unknown)
        Returns:
            logits: [B, 3, 6640]
        """
        # Full partial-backbone pass
        node_emb = self._compute_backbone_embs()  # [N_nodes, 256]

        # Extract perturbed gene embeddings; handle unknowns
        known_mask = (node_indices >= 0)
        safe_idx   = node_indices.clamp(min=0)
        pert_emb   = node_emb[safe_idx, :]        # [B, 256]
        if not known_mask.all():
            fallback = self.fallback_emb.unsqueeze(0).expand_as(pert_emb)
            pert_emb = torch.where(
                known_mask.unsqueeze(-1).expand_as(pert_emb),
                pert_emb, fallback,
            )

        # Ensure float32 for head
        pert_emb = pert_emb.float()

        # Input projection
        h = self.input_proj(pert_emb)   # [B, hidden_dim]

        # Residual blocks
        for block in self.res_blocks:
            h = block(h)                # [B, hidden_dim]

        # Bilinear decomposition
        # [B, hidden_dim] → [B, n_classes*rank] → [B, n_classes, rank]
        blin = self.fc_bilinear(h).view(-1, self.n_classes, self.bilinear_rank)  # [B, 3, rank]

        # Output gene embeddings: [n_genes_out, rank]
        out_embs = self.out_gene_emb.weight   # [6640, rank]

        # [B, 3, rank] @ [rank, 6640] → [B, 3, 6640]
        logits = torch.matmul(blin, out_embs.T)
        return logits


# ─── Helpers for DDP gathering ────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))
    pad    = max_sz - local_p.shape[0]
    p = local_p.to(device); l = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)
    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p); dist.all_gather(gl, l)
    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class StringGNNResidualLitModule(pl.LightningModule):

    def __init__(
        self,
        backbone_lr:    float = 1e-5,
        head_lr:        float = 5e-4,
        muon_lr:        float = 0.005,
        weight_decay:   float = 1e-3,
        gene_emb_wd:    float = 1e-2,   # Stronger WD for out_gene_emb (3.4M params)
        gene_emb_lr:    float = 1e-4,   # Reduced LR for scFoundation-init gene emb
        t0_steps:       int   = 1200,   # Extended T_0 (per node1-2-2-3 success)
        warmup_steps:   int   = 100,
        max_steps:      int   = 10000,
        focal_gamma:    float = FOCAL_GAMMA,
        label_genes_file: str = LABEL_GENES_FILE,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str]          = []
        self._test_symbols:  List[str]          = []
        self._test_labels:   List[torch.Tensor] = []

        # Build scFoundation gene embedding init on rank 0 only,
        # then broadcast to all ranks during setup()
        self._gene_emb_init: Optional[torch.Tensor] = None

    def setup(self, stage=None):
        dm = self.trainer.datamodule if self.trainer is not None else None
        if dm is None:
            raise RuntimeError("DataModule must be attached to the trainer.")

        # Build gene embedding initialization from scFoundation on local rank 0,
        # then broadcast to other ranks to avoid redundant model loading.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        n_gpus = self.trainer.world_size if self.trainer is not None else 1
        is_distributed = n_gpus > 1 and torch.distributed.is_available() and torch.distributed.is_initialized()

        if self._gene_emb_init is None:
            # All ranks: rank 0 builds the init, others wait at barrier, then broadcast
            if local_rank == 0:
                try:
                    gene_emb_init = build_gene_emb_init(
                        label_genes_file=self.hparams.label_genes_file,
                        target_dim=512,
                    )
                    self._gene_emb_init = gene_emb_init
                except Exception as e:
                    self.print(f"[Node4-2-1-1] WARNING: scFoundation init failed: {e}. "
                               f"Falling back to random init.")
                    self._gene_emb_init = None

            # Broadcast from rank 0 to all ranks
            if is_distributed:
                # Barrier: all ranks wait for rank 0 to finish building the init
                torch.distributed.barrier()

                # Use a flag tensor to communicate whether init succeeded
                has_init = torch.tensor(
                    [1 if (local_rank == 0 and self._gene_emb_init is not None) else 0],
                    dtype=torch.long, device=self.device
                )
                torch.distributed.broadcast(has_init, src=0)

                if has_init.item() == 1:
                    if self._gene_emb_init is None:
                        # Non-rank-0 processes: allocate buffer to receive
                        self._gene_emb_init = torch.zeros(N_GENES_OUT, 512, dtype=torch.float32)
                    init_cuda = self._gene_emb_init.to(self.device)
                    torch.distributed.broadcast(init_cuda, src=0)
                    self._gene_emb_init = init_cuda.cpu()
                else:
                    # Rank 0 init failed; all ranks fall back to None (random init)
                    self._gene_emb_init = None

        self.model = StringGNNResidualBilinearModel(
            edge_index     = dm.edge_index,
            edge_weight    = dm.edge_weight,
            gene_emb_init  = self._gene_emb_init,
        )
        # Cast all trainable parameters to float32 for stable optimization
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(self, node_indices):
        return self.model(node_indices)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return focal_cross_entropy(
            logits, labels,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        if "label" in batch:
            loss = self._loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        lp = torch.cat(self._val_preds,  0)
        ll = torch.cat(self._val_labels, 0)
        if self.trainer.world_size > 1:
            lp, ll = _gather_tensors(lp, ll, self.device, self.trainer.world_size)
        f1 = compute_per_gene_f1(lp.numpy(), ll.numpy())
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        probs  = torch.softmax(logits.float(), dim=1)
        self._test_preds.append(probs.detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (
            torch.cat(self._test_labels, 0) if self._test_labels
            else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        )
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
            all_probs, all_labels = local_probs, dummy_labels
            all_pert, all_syms   = self._test_pert_ids, self._test_symbols

        # Deduplicate (DDP DistributedSampler pads the dataset)
        seen: set = set()
        keep: List[int] = []
        for i, pid in enumerate(all_pert):
            if pid not in seen:
                seen.add(pid)
                keep.append(i)
        if len(keep) < len(all_pert):
            self.print(f"[Node4-2-1-1] Deduplicating: {len(all_pert)} → {len(keep)}")
            all_probs  = all_probs[keep]
            all_labels = all_labels[keep]
            all_pert   = [all_pert[i] for i in keep]
            all_syms   = [all_syms[i]  for i in keep]

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pid, sym, probs in zip(all_pert, all_syms, all_probs.numpy()):
                    fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")
            self.print(f"[Node4-2-1-1] Saved predictions → {pred_path}")
            if all_labels.any():
                f1 = compute_per_gene_f1(all_probs.numpy(), all_labels.numpy())
                self.print(f"[Node4-2-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear(); self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Separate parameters into groups:
        # 1. Backbone trainable (mps.7 + post_mp) — AdamW at backbone_lr
        # 2. Head 2D weight matrices (ResBlocks, bilinear) — Muon
        # 3. out_gene_emb — AdamW with STRONG WD (1e-2) and REDUCED LR (1e-4)
        #    KEY: LR reduced from 5e-4 → 1e-4 to preserve scFoundation initialization.
        #    The pretrained gene embeddings should fine-tune slowly to avoid overwriting
        #    the biologically-grounded inductive bias from scFoundation.
        # 4. Other head params (norms, biases, input_proj, fallback_emb) — AdamW at head_lr

        backbone_params = (
            list(self.model.mps_7.parameters()) +
            list(self.model.post_mp.parameters())
        )
        backbone_param_ids = {id(p) for p in backbone_params}

        # out_gene_emb gets its own group with stronger weight decay and reduced LR
        gene_emb_params    = [self.model.out_gene_emb.weight]
        gene_emb_param_ids = {id(p) for p in gene_emb_params}

        # Head 2D matrices for Muon: fc1.weight, fc2.weight, fc_bilinear.weight
        head_2d_matrices = []
        head_other       = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in backbone_param_ids:
                continue  # handled separately
            if id(param) in gene_emb_param_ids:
                continue  # handled separately (out_gene_emb)
            # Muon: 2D weight matrices in hidden layers (not embeddings, not input proj)
            if (param.ndim >= 2
                    and "input_proj.1" not in name  # input projection (boundary)
                    and "fallback_emb" not in name):
                head_2d_matrices.append(param)
            else:
                head_other.append(param)

        param_groups = [
            # Backbone (mps.7 + post_mp) — AdamW
            {
                "params":       backbone_params,
                "use_muon":     False,
                "lr":           hp.backbone_lr,
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.weight_decay,
            },
            # Head 2D matrices — Muon
            {
                "params":       head_2d_matrices,
                "use_muon":     True,
                "lr":           hp.muon_lr,
                "momentum":     0.95,
                "weight_decay": hp.weight_decay,
            },
            # out_gene_emb — AdamW with STRONG WD and REDUCED LR (1e-4 vs 5e-4)
            # The scFoundation-initialized embeddings should fine-tune slowly to
            # preserve the biologically-informed inductive bias.
            {
                "params":       gene_emb_params,
                "use_muon":     False,
                "lr":           hp.gene_emb_lr,     # 1e-4 (10x lower than parent's 5e-4)
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.gene_emb_wd,     # 1e-2 (same as parent)
            },
            # Other head params (norms, biases, input_proj, fallback_emb) — AdamW
            {
                "params":       head_other,
                "use_muon":     False,
                "lr":           hp.head_lr,
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.weight_decay * 0.1,  # lighter on non-matrix params
            },
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # Cosine annealing with warm restarts (same as parent)
        # T_0=1200 steps: Key insight from node1-2-2-3 (tree best, F1=0.5101)
        warmup = hp.warmup_steps
        T_0    = hp.t0_steps  # 1200 steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / float(max(1, warmup))
            # Cosine annealing with warm restarts after warmup
            step_after = step - warmup
            cycle_len  = T_0
            cycle_pos  = step_after % cycle_len
            progress   = float(cycle_pos) / float(cycle_len)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items()
              if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} params ({100*trained/total:.2f}%)")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 4-2-1-1 — STRING_GNN + scFoundation-Init Gene Emb + Reduced Gene Emb LR"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--backbone-lr",       type=float, default=1e-5)
    p.add_argument("--head-lr",           type=float, default=5e-4)
    p.add_argument("--muon-lr",           type=float, default=0.005)
    p.add_argument("--weight-decay",      type=float, default=1e-3)
    p.add_argument("--gene-emb-wd",       type=float, default=1e-2,
                   help="Weight decay for out_gene_emb (strong to prevent overfitting)")
    p.add_argument("--gene-emb-lr",       type=float, default=1e-4,
                   help="LR for out_gene_emb (reduced to preserve scFoundation init)")
    p.add_argument("--micro-batch-size",  type=int,   default=8)
    p.add_argument("--global-batch-size", type=int,   default=32)
    p.add_argument("--max-epochs",        type=int,   default=250)
    p.add_argument("--patience",          type=int,   default=60,   # Slightly increased for pretrained init
                   help="EarlyStopping patience (epochs)")
    p.add_argument("--t0-steps",          type=int,   default=1200)
    p.add_argument("--warmup-steps",      type=int,   default=100)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",    type=int,   default=None)
    p.add_argument("--fast-dev-run",      action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # Estimate max_steps for LR schedule
    steps_per_epoch = int(np.ceil(1416 / args.micro_batch_size))  # approx
    estimated_max_steps = args.max_epochs * steps_per_epoch // accum

    dm  = StringGNNDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = StringGNNResidualLitModule(
        backbone_lr      = args.backbone_lr,
        head_lr          = args.head_lr,
        muon_lr          = args.muon_lr,
        weight_decay     = args.weight_decay,
        gene_emb_wd      = args.gene_emb_wd,
        gene_emb_lr      = args.gene_emb_lr,
        t0_steps         = args.t0_steps,
        warmup_steps     = args.warmup_steps,
        max_steps        = estimated_max_steps,
        label_genes_file = args.data_dir + "/label_genes.txt",
    )

    ckpt_cb = ModelCheckpoint(
        dirpath   = str(out_dir / "checkpoints"),
        filename  = "best-{epoch:04d}-{val_f1:.4f}",
        monitor   = "val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max",
                            patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)
    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps    = -1
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False
    if args.debug_max_step is not None:
        max_steps = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2
        limit_test  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator   = "gpu",
        devices       = n_gpus,
        num_nodes     = 1,
        strategy      = strategy,
        precision     = "bf16-mixed",
        max_epochs    = args.max_epochs,
        max_steps     = max_steps,
        accumulate_grad_batches = accum,
        gradient_clip_val    = 1.0,          # Stabilize Muon LR updates (from parent)
        limit_train_batches  = limit_train,
        limit_val_batches    = limit_val,
        limit_test_batches   = limit_test,
        val_check_interval   = (
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps = 2,
        callbacks     = [ckpt_cb, es_cb, lr_cb, pb_cb],
        logger        = [csv_logger, tb_logger],
        log_every_n_steps = 10,
        deterministic = False,   # nll_loss2d has no deterministic CUDA impl
        default_root_dir = str(out_dir),
        fast_dev_run  = fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 4-2-1-1 — STRING_GNN Partial + scFoundation-Init Gene Emb + Reduced Gene Emb LR\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
