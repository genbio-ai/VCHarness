"""
Node 1-2: Adaptive STRING_GNN — Partial Fine-Tuning + Post-GNN Additive Conditioning
          + Rank-512 Bilinear Head + Class-Weighted Focal Loss + Calibrated LR Schedule

Architecture:
  - STRING_GNN (partial fine-tuning: last 2 GCN layers + post_mp):
      * Enables task-specific adaptation while maintaining PPI prior
      * Frozen: embedding table (emb.weight) + first 6 GCN layers (mps.0-5)
      * Trainable: last 2 GCN layers (mps.6, mps.7) + output projection (post_mp)
      * ~198K trainable STRING_GNN params (3.6% of 5.43M total)
  - Post-GNN additive conditioning (low-rank pert_matrix):
      * A learnable low-rank per-node perturbation residual [N_nodes, rank_a] x [rank_a, gnn_dim]
      * Added AFTER the GNN forward pass (avoids batch-mixing contamination)
      * Low-rank factorization (rank_a=16): reduces params from 4.83M to 0.49M
      * Better sparse gradient convergence (each gene gets ~1 update/epoch)
  - Rank-512 deep residual bilinear MLP head:
      * 6 ResidualBlocks (hidden=512, expand=4, dropout=0.25)
      * Bilinear rank=512 (proven to outperform rank=256 in MCTS tree best)
  - Class-weighted focal loss (gamma=2.0, weights=[2.0, 0.5, 4.0]):
      * Down-regulated (8.1% of labels): weight=2.0
      * Neutral (88.9% of labels): weight=0.5
      * Up-regulated (3.0% of labels): weight=4.0
  - Calibrated cosine LR schedule:
      * total_steps aligned to ~2500 (~110 epoch horizon), not max_epochs * steps_per_epoch
      * Enables proper cosine decay before early stopping triggers
  - Patience=50 for extended secondary improvement phase

Key improvements over parent (node1-2-1, F1=0.4500):
  1. Fix cond_emb batch-mixing flaw -> Post-GNN clean per-sample pert_matrix
  2. Revert STRING_GNN output gene init -> Random normal (avoids positional misalignment)
  3. Rank-512 bilinear head (vs rank-256) -> +0.013 F1 in tree best
  4. Class-weighted focal loss [2.0, 0.5, 4.0] -> Improved minority class F1
  5. Low-rank pert_matrix factorization (rank=16) -> Fewer params, better sparse convergence
  6. Calibrated LR schedule (total_steps=2500) -> Proper cosine decay
  7. Patience=50 -> Allows secondary improvement phase
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


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset.

    Each sample stores the STRING_GNN node index for post-GNN conditioning
    and precomputed base embeddings (from frozen initial forward pass) as fallback.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gnn_embeddings: np.ndarray,       # [N_nodes, 256] - base embeddings (frozen)
        node_name_to_idx: Dict[str, int],
        embed_dim: int = 256,
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.has_labels = has_labels

        n_samples = len(df)
        base_embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)
        node_indices = []
        in_vocab = []

        for i, pert_id in enumerate(self.pert_ids):
            if pert_id in node_name_to_idx:
                node_idx = node_name_to_idx[pert_id]
                base_embeddings[i] = gnn_embeddings[node_idx]
                node_indices.append(node_idx)
                in_vocab.append(True)
            else:
                node_indices.append(0)  # placeholder; masked by in_vocab
                in_vocab.append(False)

        self.base_embeddings = torch.from_numpy(base_embeddings)  # [N, 256]
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
            "pert_id":        self.pert_ids[idx],
            "symbol":         self.symbols[idx],
            "base_embedding": self.base_embeddings[idx],   # [256]
            "node_idx":       self.node_indices[idx],       # scalar long
            "in_vocab":       self.in_vocab[idx],           # bool
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule using precomputed base STRING_GNN embeddings + node indices."""

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

        # Run STRING_GNN ONCE in frozen mode to get base embeddings (for dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[DataModule] Computing STRING_GNN base embeddings (frozen forward pass)...")
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", weights_only=False)

        model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        edge_index = graph["edge_index"].to(device)
        edge_weight = graph["edge_weight"].to(device) if graph.get("edge_weight") is not None else None

        with torch.no_grad():
            outputs = model(edge_index=edge_index, edge_weight=edge_weight)

        gnn_embeddings = outputs.last_hidden_state.float().cpu().numpy()  # [N_nodes, 256]
        node_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(node_names)}

        # Store for model setup
        self.gnn_embeddings = gnn_embeddings
        self.node_name_to_idx = node_name_to_idx
        self.n_nodes = len(node_names)

        del model
        torch.cuda.empty_cache()

        print(f"[DataModule] STRING_GNN base embeddings shape: {gnn_embeddings.shape}")

        # Load all splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        n_train_cov = sum(p in node_name_to_idx for p in dfs["train"]["pert_id"])
        print(f"[DataModule] Coverage: {n_train_cov}/{len(dfs['train'])} train genes in STRING_GNN")

        embed_dim = gnn_embeddings.shape[1]
        self.train_ds = PerturbationDataset(dfs["train"], gnn_embeddings, node_name_to_idx, embed_dim, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   gnn_embeddings, node_name_to_idx, embed_dim, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  gnn_embeddings, node_name_to_idx, embed_dim, True)

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


class AdaptiveStringGNN(nn.Module):
    """STRING_GNN with partial fine-tuning + low-rank post-GNN additive conditioning.

    Design:
      1. STRING_GNN runs one clean forward pass (no cond_emb, no batch mixing)
         with partial fine-tuning: mps.6, mps.7, post_mp trainable
      2. Per-sample post-GNN additive conditioning via low-rank pert_matrix:
         adapted_emb[i] = all_node_embs[node_idx[i]] + (pert_A @ pert_B)[node_idx[i]]
         where pert_A [N_nodes, rank_pert] and pert_B [rank_pert, gnn_dim] are learnable
      3. Low-rank factorization: rank_pert=16, params = 18870*16 + 16*256 = 305,936 (~0.31M)
         vs full pert_matrix: 18870*256 = 4,830,720 (~4.83M)
         Benefits: faster sparse convergence (each gene gets ~1 update/epoch), less overfitting

    Partial fine-tuning (as recommended by STRING_GNN skill for limited data):
      - Frozen: emb.weight + mps.0-5 (first 6 GCN layers)
      - Trainable: mps.6, mps.7 (last 2 GCN layers) + post_mp
      Total trainable STRING_GNN params: ~198K
    """

    def __init__(self, model_dir: Path, gnn_dim: int = 256, rank_pert: int = 16):
        super().__init__()
        self.model_dir = model_dir
        self.gnn_dim = gnn_dim
        self.rank_pert = rank_pert

        # Low-rank perturbation matrix factorization: pert_A [N_nodes, r] * pert_B [r, gnn_dim]
        # Initialized near zero to start close to pretrained PPI embeddings
        # Will be set properly once we know n_nodes (in setup_gnn)
        self.pert_A: Optional[nn.Parameter] = None
        self.pert_B: Optional[nn.Parameter] = None

        # OOV fallback embedding for genes not in STRING_GNN vocab
        self.oov_embedding = nn.Embedding(1, gnn_dim)
        nn.init.normal_(self.oov_embedding.weight, std=0.02)

        # GNN - initialized in setup_gnn()
        self.gnn: Optional[nn.Module] = None

    def setup_gnn(self, n_nodes: int):
        """Load and configure STRING_GNN with partial fine-tuning."""
        if self.gnn is not None:
            return

        graph = torch.load(self.model_dir / "graph_data.pt", weights_only=False)
        edge_index = graph["edge_index"]
        edge_weight = graph["edge_weight"] if graph.get("edge_weight") is not None else None

        self.register_buffer("edge_index", edge_index)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        # Load pretrained GNN
        self.gnn = AutoModel.from_pretrained(str(self.model_dir), trust_remote_code=True)
        self.n_nodes = self.gnn.emb.weight.shape[0]

        # Freeze all parameters first
        for param in self.gnn.parameters():
            param.requires_grad = False

        # Unfreeze last 2 GCN layers + output projection
        trainable_prefixes = ("mps.6.", "mps.7.", "post_mp.")
        n_trainable = 0
        n_total = sum(p.numel() for p in self.gnn.parameters())
        for name, param in self.gnn.named_parameters():
            if any(name.startswith(pref) for pref in trainable_prefixes):
                param.requires_grad = True
                n_trainable += param.numel()

        print(f"[AdaptiveStringGNN] Trainable STRING_GNN params: "
              f"{n_trainable}/{n_total} ({100*n_trainable/n_total:.1f}%) "
              f"[last 2 GCN layers + post_mp]")

        # Initialize low-rank perturbation matrix
        # pert_matrix[j] = pert_A[j] @ pert_B  for each node j
        self.pert_A = nn.Parameter(torch.zeros(self.n_nodes, self.rank_pert))
        self.pert_B = nn.Parameter(torch.zeros(self.rank_pert, self.gnn_dim))
        nn.init.normal_(self.pert_A, std=0.01)
        nn.init.normal_(self.pert_B, std=0.01)

        n_pert_params = self.pert_A.numel() + self.pert_B.numel()
        print(f"[AdaptiveStringGNN] Low-rank pert_matrix: rank={self.rank_pert}, "
              f"params={n_pert_params:,} (A:{self.pert_A.shape}, B:{self.pert_B.shape})")

    def forward(
        self,
        base_embedding: torch.Tensor,   # [B, 256] - precomputed frozen base embeddings
        node_idx: torch.Tensor,         # [B] long - STRING_GNN node index
        in_vocab: torch.Tensor,         # [B] bool
    ) -> torch.Tensor:
        """
        Computes perturbation-aware embeddings via clean post-GNN additive conditioning:
          1. Run STRING_GNN forward (clean, no cond_emb)
          2. Per-sample post-GNN adaptation:
             adapted_emb[i] = all_node_embs[node_idx[i]] + (pert_A[node_idx[i]] @ pert_B)
          3. For OOV genes, use the base embedding + learned OOV residual

        Returns:
            emb: [B, 256] perturbation-aware embeddings
        """
        B = node_idx.shape[0]
        device = node_idx.device

        # Step 1: Run STRING_GNN forward (clean, no cond_emb)
        edge_index = self.edge_index.to(device)
        edge_weight = self.edge_weight.to(device) if self.edge_weight is not None else None

        outputs = self.gnn(
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        all_node_embs = outputs.last_hidden_state.float()  # [N_nodes, 256]

        # Step 2: Extract embeddings + per-sample post-GNN additive conditioning
        # pert_delta[j] = pert_A[j] @ pert_B  (low-rank factorization)
        # We compute this only for the unique node indices in the batch for efficiency
        result = torch.zeros(B, self.gnn_dim, device=device, dtype=torch.float32)

        in_vocab_mask = in_vocab
        if in_vocab_mask.any():
            valid_idx = node_idx[in_vocab_mask]

            # GNN embedding for each in-vocab sample
            gnn_emb_valid = all_node_embs[valid_idx]  # [K, 256]

            # Low-rank perturbation: pert_A[valid_idx] @ pert_B -> [K, 256]
            pert_delta_valid = self.pert_A[valid_idx].float() @ self.pert_B.float()  # [K, 256]

            # Add perturbation-specific residual (zero-contamination: each sample independent)
            result[in_vocab_mask] = gnn_emb_valid + pert_delta_valid

        # Step 3: OOV genes use base embedding + OOV residual
        oov_mask = ~in_vocab_mask
        if oov_mask.any():
            oov_residual = self.oov_embedding(
                torch.zeros(oov_mask.sum(), dtype=torch.long, device=device)
            ).float()
            result[oov_mask] = base_embedding[oov_mask].to(device).float() + oov_residual

        return result  # [B, 256]


class GNNBilinearHead(nn.Module):
    """Prediction head with two-sided bilinear interaction.

    Left side: perturbation-aware GNN embedding -> Deep MLP -> [B, 3, rank]
    Right side: learnable output gene embeddings [n_genes_out, rank]
    Interaction: einsum("bcr,gr->bcg") -> logits [B, 3, n_genes_out]

    Key improvements:
    - rank=512 (vs rank=256 in parent) for better expressivity
    - dropout=0.25 (vs 0.2 in parent) for better regularization
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.25,
        n_residual_layers: int = 6,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.rank = rank

        # Input normalization
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

        # Output gene embeddings: [n_genes_out, rank]
        # Random normal initialization (std=0.02) - proven better than STRING_GNN init
        # (STRING_GNN positional alignment unknown -> random init is correct)
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, rank))

        # Head dropout
        self.head_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)
        nn.init.normal_(self.out_gene_emb, std=0.02)

    def forward(self, gnn_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gnn_emb: [B, gnn_dim] - perturbation-aware STRING_GNN embeddings
        Returns:
            logits: [B, 3, n_genes_out]
        """
        B = gnn_emb.shape[0]

        x = self.input_norm(gnn_emb)
        x = self.proj_in(x)   # [B, hidden_dim]

        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)   # [B, hidden_dim]

        x = self.head_dropout(x)
        pert_proj = self.proj_bilinear(x)                          # [B, n_classes * rank]
        pert_proj = pert_proj.view(B, self.n_classes, self.rank)   # [B, 3, rank]

        # Two-sided bilinear interaction
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, n_genes_out]
        return logits


# ─── Class-Weighted Focal Loss ────────────────────────────────────────────────

def class_weighted_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Class-weighted focal loss for multi-class classification.

    Combines:
    - Focal weighting: (1 - p_t)^gamma to down-weight easy examples
    - Class weighting: per-class weights to address imbalance

    Class weights [2.0, 0.5, 4.0] for [down(-1), neutral(0), up(+1)] classes:
    - Down (-1, 8.1% of labels): weight=2.0 (moderate up-weighting)
    - Neutral (0, 88.9% of labels): weight=0.5 (down-weighting dominant class)
    - Up (+1, 3.0% of labels): weight=4.0 (strong up-weighting of rare class)
    """
    B, C, G = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
    labels_flat = labels.reshape(-1)                        # [B*G]

    if class_weights is not None:
        ce_loss = F.cross_entropy(
            logits_flat, labels_flat,
            weight=class_weights.to(logits_flat.device),
            reduction="none"
        )
    else:
        ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction="none")

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
    """LightningModule for gene-perturbation DEG prediction (Node 1-2).

    Uses partially fine-tuned STRING_GNN with post-GNN low-rank additive conditioning,
    rank-512 bilinear head, and class-weighted focal loss.
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        rank_pert: int = 16,
        n_residual_layers: int = 6,
        dropout: float = 0.25,
        lr_backbone: float = 5e-5,
        lr_head: float = 5e-4,
        weight_decay: float = 1e-3,
        focal_gamma: float = 2.0,
        class_weight_down: float = 2.0,
        class_weight_neutral: float = 0.5,
        class_weight_up: float = 4.0,
        warmup_steps: int = 100,
        total_steps: int = 2500,
        n_nodes: int = 18870,
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

        # Build partially fine-tuned STRING_GNN backbone with low-rank pert conditioning
        self.backbone = AdaptiveStringGNN(
            model_dir=STRING_GNN_DIR,
            gnn_dim=hp.gnn_dim,
            rank_pert=hp.rank_pert,
        )
        self.backbone.setup_gnn(n_nodes=hp.n_nodes)

        # Build rank-512 prediction head with dropout=0.25
        self.head = GNNBilinearHead(
            gnn_dim=hp.gnn_dim,
            hidden_dim=hp.hidden_dim,
            rank=hp.rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
        )

        # out_gene_emb already randomly initialized in GNNBilinearHead (std=0.02)
        # No STRING_GNN initialization here (positional misalignment avoidance)

        # Register class weights as buffer (not parameter)
        class_weights = torch.tensor(
            [hp.class_weight_down, hp.class_weight_neutral, hp.class_weight_up],
            dtype=torch.float32
        )
        self.register_buffer("class_weights", class_weights)

        # Cast all trainable parameters to float32 for stable optimization
        for p in self.backbone.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        for p in self.head.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    def forward(
        self,
        base_embedding: torch.Tensor,
        node_idx: torch.Tensor,
        in_vocab: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.backbone(base_embedding, node_idx, in_vocab)  # [B, 256]
        logits = self.head(emb)                                    # [B, 3, 6640]
        return logits

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return class_weighted_focal_loss(
            logits, labels,
            gamma=self.hparams.focal_gamma,
            class_weights=self.class_weights,
        )

    def training_step(self, batch, batch_idx):
        logits = self(
            batch["base_embedding"].float(),
            batch["node_idx"],
            batch["in_vocab"],
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(
            batch["base_embedding"].float(),
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
            batch["base_embedding"].float(),
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

            self.print(f"[Node1-2] Saved test predictions -> {pred_path} ({len(seen_ids)} samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-2] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Three parameter groups:
        # 1. STRING_GNN backbone (last 2 layers + post_mp): lower LR to prevent forgetting
        # 2. low-rank pert_matrix (pert_A, pert_B): head LR for sparse updates
        # 3. Prediction head: higher LR for fast convergence

        backbone_params = [
            p for n, p in self.backbone.named_parameters()
            if p.requires_grad and ('gnn.' in n)
        ]
        pert_params = [
            p for n, p in self.backbone.named_parameters()
            if p.requires_grad and ('pert_' in n or 'oov_' in n)
        ]
        head_params = list(self.head.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": hp.lr_backbone, "weight_decay": hp.weight_decay},
                {"params": pert_params,     "lr": hp.lr_head,     "weight_decay": hp.weight_decay},
                {"params": head_params,     "lr": hp.lr_head,     "weight_decay": hp.weight_decay},
            ],
        )

        # Cosine annealing with linear warmup
        # total_steps calibrated to ~2500 (matching ~110-epoch training window)
        # This ensures cosine schedule decays significantly before early stopping
        def lr_lambda(step: int) -> float:
            if step < hp.warmup_steps:
                return float(step) / max(1, hp.warmup_steps)
            progress = (step - hp.warmup_steps) / max(1, hp.total_steps - hp.warmup_steps)
            # Clamp progress to [0, 1] to avoid unintended second LR cycle
            progress = min(progress, 1.0)
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
        description="Node 1-2 – Adaptive STRING_GNN + Low-Rank Pert + Rank-512 Bilinear + Class-Weighted Focal"
    )
    p.add_argument("--data-dir",              type=str,   default="data")
    p.add_argument("--gnn-dim",               type=int,   default=256)
    p.add_argument("--hidden-dim",            type=int,   default=512)
    p.add_argument("--rank",                  type=int,   default=512,
                   help="Bilinear rank (default=512, proven better than 256)")
    p.add_argument("--rank-pert",             type=int,   default=16,
                   help="Low-rank dimension for pert_matrix factorization")
    p.add_argument("--n-residual-layers",     type=int,   default=6)
    p.add_argument("--dropout",               type=float, default=0.25)
    p.add_argument("--lr-backbone",           type=float, default=5e-5,
                   help="LR for STRING_GNN partially fine-tuned layers")
    p.add_argument("--lr-head",               type=float, default=5e-4,
                   help="LR for bilinear prediction head + pert_matrix")
    p.add_argument("--weight-decay",          type=float, default=1e-3)
    p.add_argument("--focal-gamma",           type=float, default=2.0)
    p.add_argument("--class-weight-down",     type=float, default=2.0,
                   help="Focal loss weight for down-regulated class (-1 -> 0)")
    p.add_argument("--class-weight-neutral",  type=float, default=0.5,
                   help="Focal loss weight for neutral class (0 -> 1)")
    p.add_argument("--class-weight-up",       type=float, default=4.0,
                   help="Focal loss weight for up-regulated class (+1 -> 2)")
    p.add_argument("--warmup-steps",          type=int,   default=100)
    p.add_argument("--total-steps",           type=int,   default=2500,
                   help="Total cosine schedule steps (calibrated to ~110-epoch horizon)")
    p.add_argument("--micro-batch-size",      type=int,   default=16)
    p.add_argument("--global-batch-size",     type=int,   default=64)
    p.add_argument("--max-epochs",            type=int,   default=300)
    p.add_argument("--patience",              type=int,   default=50,
                   help="Early stopping patience (increased to 50 for secondary improvement phase)")
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

    # Gradient accumulation
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # LightningModule
    # Note: total_steps is fixed (not dynamic from max_epochs) to calibrate LR schedule
    lit = PerturbationLitModule(
        gnn_dim=args.gnn_dim,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        rank_pert=args.rank_pert,
        n_residual_layers=args.n_residual_layers,
        dropout=args.dropout,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        class_weight_down=args.class_weight_down,
        class_weight_neutral=args.class_weight_neutral,
        class_weight_up=args.class_weight_up,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        n_nodes=dm.n_nodes,
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
            f"Node 1-2 – Adaptive STRING_GNN + Low-Rank Pert + Rank-512 Bilinear + Class-Weighted Focal\n"
            f"Test results from trainer: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
