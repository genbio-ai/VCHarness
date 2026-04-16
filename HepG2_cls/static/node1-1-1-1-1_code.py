"""Node 1-1-1-1-1: STRING_GNN (partially fine-tuned) + ESM2-650M precomputed embeddings + Corrected Factorized MLP Head.

Architecture:
  - Input: perturbed gene identifier (ENSEMBL ID)
  - STRING_GNN partially fine-tuned: last 2 GNN layers (mps.6, mps.7) + post_mp unfrozen
    Per-gene lookup: use the partially-fine-tuned GNN embedding for the perturbed gene [256-dim]
  - ESM2 650M precomputed embeddings: frozen lookup into esm2_embeddings_t33_650M.pt [3840-dim]
    Projected to 256-dim via trainable Linear(3840->256) + LayerNorm
  - Feature fusion: concat [STRING_GNN 256-dim, ESM2-proj 256-dim] -> 512-dim input
  - Residual MLP trunk: 3 blocks (512-dim hidden, 1024-dim intermediate)
  - CORRECTED factorized output head: Linear(512->512) -> LN -> GELU -> Dropout -> Linear(512->19920)
    bottleneck_dim=512 (FIXED from parent's catastrophic 128-dim bottleneck)
  - Loss: Focal loss (gamma=2.0) with label smoothing (eps=0.05) and class weights
  - LR schedule: Cosine annealing (T_max=100, eta_min=1e-7) — prevents premature LR halving
  - Hyperparameters restored to proven settings: lr=2e-4, dropout=0.35, wd=5e-4

Key innovations vs parent node1-1-1-1 (F1=0.293):
  1. CRITICAL FIX: bottleneck_dim 128 -> 512 (prevents information chokepoint)
  2. Restore hidden_dim: 256 -> 512 (restores representation capacity)
  3. Restore n_blocks: 2 -> 3 (restores transformation depth)
  4. Add ESM2 650M precomputed embeddings as complementary features (frozen lookup,
     NO ESM2 inference overhead — just torch.load + index)
  5. Switch LR schedule: ReduceLROnPlateau -> cosine annealing T_max=100 (avoids
     premature LR halving that crippled parent at epoch 36)
  6. Restore lr: 1e-4 -> 2e-4 (appropriate for larger model)
  7. Restore dropout: 0.45 -> 0.35 (parent was over-regularized)
  8. Restore weight_decay: 1e-3 -> 5e-4 (appropriate for this model size)
  9. Keep label smoothing eps=0.05 (useful for noisy biological labels)
  10. Keep focal loss gamma=2.0 (proven effective for class imbalance)
  Total trainable params: ~14.0M (ESM2 proj ~1.0M + trunk ~3.9M + head ~10.4M + GNN ~0.4M)

Design rationale:
  - ESM2 650M embeddings are pre-stored as [18870, 3840] tensors in the STRING_GNN model dir.
    Using them requires only a tensor load + index — no ESM2 inference overhead.
  - Previous ESM2 attempt (node3-1) failed due to focal loss instability, NOT due to ESM2.
    This node uses proven weighted CE + focal loss (not pure focal) to avoid the problem.
  - The 3840-dim ESM2 features are much richer than the 480-dim from 35M model, capturing
    deep protein structural/functional information.
  - Concatenating STRING_GNN (PPI graph structure) + ESM2 (protein sequence) provides
    complementary biological signals: network topology + sequence-level protein properties.
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640          # number of response genes
N_CLASSES = 3           # {-1->0, 0->1, 1->2}
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
STRING_EMB_DIM = 256    # STRING_GNN output embedding dimension
ESM2_EMB_FILE = STRING_GNN_DIR / "esm2_embeddings_t33_650M.pt"  # precomputed [18870, 3840]
ESM2_RAW_DIM = 3840     # 650M ESM2 embedding dimension (mean-pooled from layer 33 × 3)
ESM2_PROJ_DIM = 256     # projected ESM2 dimension (same as STRING_GNN to allow equal fusion)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Dataset for gene perturbation -> differential expression prediction."""

    def __init__(
        self,
        df: pd.DataFrame,
        node_name2idx: Dict[str, int],
        n_nodes: int,
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        # Map each perturbation to STRING_GNN node index (-1 = not in STRING)
        gnn_indices = []
        for pid in self.pert_ids:
            idx = node_name2idx.get(pid, -1)
            gnn_indices.append(idx)
        self.gnn_indices = torch.tensor(gnn_indices, dtype=torch.long)

        if "label" in df.columns:
            labels = np.array([json.loads(x) for x in df["label"].tolist()], dtype=np.int64)
            # Shift {-1,0,1} -> {0,1,2}
            self.labels = torch.tensor(labels + 1, dtype=torch.long)  # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "gnn_idx": self.gnn_indices[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]  # [6640]
        return item


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class PerturbDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        micro_batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

        self.node_name2idx: Dict[str, int] = {}
        self.n_nodes: int = 0
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        # Build STRING_GNN node name to index mapping
        if not self.node_name2idx:
            node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
            self.node_name2idx = {name: i for i, name in enumerate(node_names)}
            self.n_nodes = len(node_names)

        self.train_ds = PerturbDataset(train_df, self.node_name2idx, self.n_nodes)
        self.val_ds = PerturbDataset(val_df, self.node_name2idx, self.n_nodes)
        self.test_ds = PerturbDataset(test_df, self.node_name2idx, self.n_nodes)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Focal Loss with Label Smoothing
# ---------------------------------------------------------------------------
def focal_loss_with_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
    smooth_eps: float = 0.05,
) -> torch.Tensor:
    """Multi-class focal loss with class weights and label smoothing.

    logits:  [N, C] float
    targets: [N]    long  values in {0,...,C-1}
    class_weights: [C] float
    gamma: focusing parameter (0 = standard CE)
    smooth_eps: label smoothing epsilon (0 = no smoothing)

    Label smoothing formula:
      L_smooth = -(1 - eps) * log p_y - (eps / C) * sum_k log p_k
    Combined with focal weight: (1 - p_y)^gamma * sample_weight * L_smooth
    """
    C = logits.size(1)
    log_softmax = F.log_softmax(logits, dim=-1)   # [N, C]
    softmax = log_softmax.exp()                    # [N, C]

    # Focal weight computed from the hard target probability
    log_pt = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
    pt = softmax.gather(1, targets.unsqueeze(1)).squeeze(1)          # [N]
    focal_weight = (1.0 - pt) ** gamma                                # [N]

    # Label-smoothed CE loss
    if smooth_eps > 0.0:
        smooth_loss = (
            -(1.0 - smooth_eps) * log_pt
            - (smooth_eps / C) * log_softmax.sum(dim=-1)
        )  # [N]
    else:
        smooth_loss = -log_pt  # [N] — standard CE

    sample_weight = class_weights[targets]  # [N]
    loss = focal_weight * sample_weight * smooth_loss
    return loss.mean()


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """MLP residual block with layer norm."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class StringGNNPlusESM2MLP(nn.Module):
    """Dual-source biological embedding MLP for perturbation response prediction.

    Combines:
    1. STRING_GNN PPI graph embeddings (256-dim, partially fine-tuned)
    2. Pre-computed ESM2 650M protein language model embeddings (3840-dim, frozen)
       projected to 256-dim via trainable Linear(3840->256) + LN
    Concatenated -> 512-dim input -> residual MLP trunk -> factorized output head

    Key fix vs parent (node1-1-1-1):
    - bottleneck_dim: 128 -> 512 (prevents critical information chokepoint)
    - hidden_dim: 256 -> 512 (restores representation capacity)
    - n_blocks: 2 -> 3 (restores transformation depth)
    - New: ESM2 650M precomputed features (frozen lookup, no inference cost)

    Parameter count:
    - ESM2 projection: Linear(3840->256) + LN ≈ 0.98M (trainable)
    - Input proj: Linear(512->512) + LN ≈ 0.26M
    - 3x ResidualBlock(512, 1024, drop): 3 × [Linear(512->1024) + LN + Linear(1024->512) + LN] ≈ 3.15M
    - Factorized head: LN(512) + Linear(512->512) + LN(512) + Linear(512->19920) ≈ 10.4M
    - Fallback emb: 256 params
    - Total MLP trainable: ~14.8M
    Plus: STRING_GNN unfrozen layers ~0.4M
    Grand total: ~15.2M trainable params
    """

    def __init__(
        self,
        n_nodes: int,
        gnn_emb_dim: int = STRING_EMB_DIM,
        esm2_raw_dim: int = ESM2_RAW_DIM,
        esm2_proj_dim: int = ESM2_PROJ_DIM,
        hidden_dim: int = 512,
        bottleneck_dim: int = 512,
        n_blocks: int = 3,
        n_genes: int = N_GENES,
        n_classes: int = N_CLASSES,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()

        # Fallback embedding for genes not in STRING_GNN
        self.fallback_emb = nn.Embedding(1, gnn_emb_dim)
        nn.init.normal_(self.fallback_emb.weight, std=0.01)

        # ESM2 projection: frozen 3840-dim -> trainable 256-dim projection
        # The frozen ESM2 embeddings provide rich protein sequence context;
        # the projection is trainable to adapt to the task
        self.esm2_proj = nn.Sequential(
            nn.Linear(esm2_raw_dim, esm2_proj_dim),
            nn.LayerNorm(esm2_proj_dim),
            nn.GELU(),
        )

        # Fallback ESM2 embedding for genes not in the ESM2 precomputed set
        self.esm2_fallback = nn.Embedding(1, esm2_proj_dim)
        nn.init.normal_(self.esm2_fallback.weight, std=0.01)

        # Input projection: concat [GNN 256-dim, ESM2-proj 256-dim] = 512-dim -> hidden_dim
        fused_dim = gnn_emb_dim + esm2_proj_dim  # 512
        self.input_proj = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Residual MLP blocks: 3 blocks, hidden_dim with 2x intermediate
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 2, dropout)
            for _ in range(n_blocks)
        ])

        # CORRECTED FACTORIZED output head: hidden_dim -> bottleneck_dim -> n_genes * n_classes
        # bottleneck_dim=512 is large enough to encode the 19,920-dim output space
        # (512 bottleneck dimensions / 19920 outputs = ~38.9 outputs per dimension,
        #  much more tractable than parent's 128-dim bottleneck with ~156 outputs per dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, n_genes * n_classes),
        )

        self.n_genes = n_genes
        self.n_classes = n_classes
        self.gnn_emb_dim = gnn_emb_dim
        self.esm2_proj_dim = esm2_proj_dim

    def forward(
        self,
        gnn_node_emb: torch.Tensor,   # [N_nodes, 256] - precomputed STRING_GNN output
        esm2_emb: torch.Tensor,        # [N_nodes, 3840] - precomputed ESM2 650M embeddings (frozen)
        gnn_idx: torch.Tensor,          # [B] - index into gnn_node_emb, or -1 for missing
    ) -> torch.Tensor:
        B = gnn_idx.size(0)
        device = gnn_idx.device

        # --- STRING_GNN embeddings ---
        gnn_emb_batch = torch.zeros(B, gnn_node_emb.size(1), device=device, dtype=gnn_node_emb.dtype)
        in_gnn_mask = gnn_idx >= 0
        not_in_gnn_mask = ~in_gnn_mask

        if in_gnn_mask.any():
            valid_idx = gnn_idx[in_gnn_mask]
            gnn_emb_batch[in_gnn_mask] = gnn_node_emb[valid_idx]

        if not_in_gnn_mask.any():
            fallback = self.fallback_emb(
                torch.zeros(not_in_gnn_mask.sum(), device=device, dtype=torch.long)
            )
            gnn_emb_batch[not_in_gnn_mask] = fallback.to(gnn_emb_batch.dtype)

        # --- ESM2 precomputed embeddings (frozen lookup + trainable projection) ---
        esm2_raw_batch = torch.zeros(B, esm2_emb.size(1), device=device, dtype=esm2_emb.dtype)

        if in_gnn_mask.any():
            valid_idx = gnn_idx[in_gnn_mask]
            esm2_raw_batch[in_gnn_mask] = esm2_emb[valid_idx]

        if not_in_gnn_mask.any():
            # Use zero vector for ESM2 fallback (genes not in STRING have no ESM2 embedding)
            # The ESM2 projection will map zero to a small near-zero output
            pass  # already zeros from initialization above

        # Project ESM2 from 3840 -> 256
        esm2_proj_batch = self.esm2_proj(esm2_raw_batch.float())  # [B, 256]

        # --- Feature fusion: concat STRING_GNN + ESM2 projected ---
        fused = torch.cat([gnn_emb_batch.float(), esm2_proj_batch], dim=-1)  # [B, 512]

        # --- Residual MLP trunk ---
        x = self.input_proj(fused)
        for block in self.blocks:
            x = block(x)

        # --- Corrected factorized head: 512 -> 512 -> 19920 ---
        logits = self.head(x)  # [B, n_genes * n_classes]

        return logits.view(B, self.n_classes, self.n_genes)  # [B, 3, 6640]


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int = 512,
        bottleneck_dim: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.35,
        lr: float = 2e-4,
        gnn_lr: float = 5e-5,
        weight_decay: float = 5e-4,
        focal_gamma: float = 2.0,
        label_smooth_eps: float = 0.05,
        cosine_t_max: int = 100,
        max_epochs: int = 200,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.gnn_lr = gnn_lr
        self.weight_decay = weight_decay
        self.focal_gamma = focal_gamma
        self.label_smooth_eps = label_smooth_eps
        self.cosine_t_max = cosine_t_max
        self.max_epochs = max_epochs

        self.model: Optional[StringGNNPlusESM2MLP] = None
        self.gnn_model = None  # STRING_GNN backbone (partially fine-tuned)

        # Buffers for validation and test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

        # Will hold the precomputed/updated GNN node embeddings [n_nodes, 256]
        self._gnn_emb_cache: Optional[torch.Tensor] = None

    def setup(self, stage: str = "fit") -> None:
        from transformers import AutoModel

        # Class weights: inversely proportional to frequency
        # class0 (down-reg) 4.77%, class1 (neutral) 92.82%, class2 (up-reg) 2.41%
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = 1.0 / freq
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        self.register_buffer("class_weights", class_weights)

        # Load STRING_GNN backbone — only once
        if self.gnn_model is None:
            self.gnn_model = AutoModel.from_pretrained(
                str(STRING_GNN_DIR), trust_remote_code=True
            )
            # Freeze all parameters first
            for param in self.gnn_model.parameters():
                param.requires_grad = False

            # Partially unfreeze: last 2 GNN layers (mps.6, mps.7) and post_mp
            # Per STRING_GNN skill: "freeze embedding table first and tune mps.6.*, mps.7.*, post_mp.*"
            layers_to_unfreeze = ["mps.6.", "mps.7.", "post_mp."]
            unfrozen_params = 0
            for name, param in self.gnn_model.named_parameters():
                if any(name.startswith(prefix) for prefix in layers_to_unfreeze):
                    param.requires_grad = True
                    unfrozen_params += param.numel()
            self.print(
                f"STRING_GNN loaded: {sum(p.numel() for p in self.gnn_model.parameters()):,} total params, "
                f"{unfrozen_params:,} unfrozen (last 2 layers + post_mp)"
            )
            # Cast unfrozen GNN params to float32 for stable optimization
            for param in self.gnn_model.parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        # Load precomputed ESM2 650M embeddings (frozen, just a lookup table)
        # Shape: [18870, 3840] — precomputed mean-pooled embeddings for all STRING proteins
        if not hasattr(self, "_esm2_emb_buf") or self._esm2_emb_buf is None:
            self.print(f"Loading precomputed ESM2 650M embeddings from {ESM2_EMB_FILE}...")
            esm2_emb = torch.load(str(ESM2_EMB_FILE), map_location="cpu", weights_only=True)
            self.print(f"ESM2 650M embeddings shape: {esm2_emb.shape}, dtype: {esm2_emb.dtype}")
            # Register as a buffer so it moves with the model to the right device automatically
            # persistent=False because it can be reloaded from disk; also excluded from checkpoints
            self.register_buffer("_esm2_emb_buf", esm2_emb.float(), persistent=False)

        # Initialize the MLP prediction model — only once
        if self.model is None:
            self.model = StringGNNPlusESM2MLP(
                n_nodes=self.n_nodes,
                gnn_emb_dim=STRING_EMB_DIM,
                esm2_raw_dim=ESM2_RAW_DIM,
                esm2_proj_dim=ESM2_PROJ_DIM,
                hidden_dim=self.hidden_dim,
                bottleneck_dim=self.bottleneck_dim,
                n_blocks=self.n_blocks,
                dropout=self.dropout,
            )
            # Cast trainable parameters to float32 for stable optimization
            for v in self.model.parameters():
                if v.requires_grad:
                    v.data = v.data.float()
            mlp_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            gnn_trainable = sum(p.numel() for p in self.gnn_model.parameters() if p.requires_grad)
            self.print(
                f"StringGNNPlusESM2MLP | hidden={self.hidden_dim} | bottleneck={self.bottleneck_dim} | "
                f"blocks={self.n_blocks} | dropout={self.dropout} | "
                f"MLP trainable={mlp_trainable:,} | GNN unfrozen={gnn_trainable:,}"
            )

    def _get_gnn_embeddings(self) -> torch.Tensor:
        """Get STRING_GNN node embeddings.

        When GNN is partially frozen and we're in eval mode (val/test), we use the cache.
        When in training mode, we always recompute since the unfrozen layers are being updated.
        """
        if self.training:
            # Recompute in training mode so gradients flow through unfrozen layers
            device = next(self.model.parameters()).device
            self.gnn_model = self.gnn_model.to(device)
            graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location=device, weights_only=False)
            edge_index = graph["edge_index"]
            edge_weight = graph.get("edge_weight", None)
            outputs = self.gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            return outputs.last_hidden_state.float()
        else:
            # In eval mode, cache the embeddings for efficiency
            if self._gnn_emb_cache is None:
                device = next(self.model.parameters()).device
                self.gnn_model = self.gnn_model.to(device)
                graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location=device, weights_only=False)
                edge_index = graph["edge_index"]
                edge_weight = graph.get("edge_weight", None)
                with torch.no_grad():
                    outputs = self.gnn_model(
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                    )
                self._gnn_emb_cache = outputs.last_hidden_state.float().detach()
            return self._gnn_emb_cache

    def _get_esm2_embeddings(self) -> torch.Tensor:
        """Get precomputed ESM2 650M embeddings (frozen, just lookup table).

        The _esm2_emb_buf is a registered buffer that Lightning automatically moves
        to the correct device during DDP and single-GPU training. No manual device
        synchronization is needed.
        """
        return self._esm2_emb_buf

    def on_train_epoch_start(self) -> None:
        # Invalidate the embedding cache at the start of each training epoch
        # so that val/test will recompute with the latest GNN weights
        self._gnn_emb_cache = None

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Focal loss with label smoothing and class weights.

        logits: [B, 3, 6640]
        labels: [B, 6640]  values in {0,1,2}
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        loss = focal_loss_with_label_smoothing(
            logits_flat,
            labels_flat,
            self.class_weights,
            gamma=self.focal_gamma,
            smooth_eps=self.label_smooth_eps,
        )
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        gnn_emb = self._get_gnn_embeddings()
        esm2_emb = self._get_esm2_embeddings()
        logits = self.model(gnn_emb, esm2_emb, batch["gnn_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        gnn_emb = self._get_gnn_embeddings()
        esm2_emb = self._get_esm2_embeddings()
        logits = self.model(gnn_emb, esm2_emb, batch["gnn_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds_local = torch.cat(self._val_preds, dim=0)   # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0) # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather from all DDP ranks
        all_preds = self.all_gather(preds_local)   # [ws, N_local, 3, 6640]
        all_labels = self.all_gather(labels_local)  # [ws, N_local, 6640]

        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
        all_labels = all_labels.view(-1, N_GENES)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()

        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        gnn_emb = self._get_gnn_embeddings()
        esm2_emb = self._get_esm2_embeddings()
        logits = self.model(gnn_emb, esm2_emb, batch["gnn_idx"])
        self._test_preds.append(logits.detach().cpu())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> Optional[Dict[str, Any]]:
        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        all_preds = self.all_gather(preds_local)
        ws = self.trainer.world_size
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        # Compute test F1 if labels are available (gathered across ranks)
        test_f1 = None
        if self._test_labels:
            labels_local = torch.cat(self._test_labels, dim=0)
            all_labels = self.all_gather(labels_local)
            all_labels = all_labels.view(-1, N_GENES).cpu().numpy()
            preds_np = all_preds.float().cpu().numpy()
            test_f1 = _compute_per_gene_f1(preds_np, all_labels)
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=True)
        self._test_labels.clear()

        all_pert_ids: List[str] = []
        all_symbols: List[str] = []
        if self.trainer.is_global_zero:
            _pert_ids_gathered: List[List[str]] = [[] for _ in range(ws)]
            _symbols_gathered: List[List[str]] = [[] for _ in range(ws)]
            torch.distributed.gather_object(self._test_pert_ids, _pert_ids_gathered, dst=0)
            torch.distributed.gather_object(self._test_symbols, _symbols_gathered, dst=0)
            for p_list, s_list in zip(_pert_ids_gathered, _symbols_gathered):
                all_pert_ids.extend(p_list)
                all_symbols.extend(s_list)
        else:
            torch.distributed.gather_object(self._test_pert_ids, dst=0)
            torch.distributed.gather_object(self._test_symbols, dst=0)

        self._test_pert_ids.clear()
        self._test_symbols.clear()

        if self.trainer.is_global_zero:
            if all_preds.size(0) == 0:
                self.print("Warning: No test predictions gathered. Skipping save.")
            else:
                preds_np = all_preds.float().cpu().numpy()

                # Deduplicate by pert_id (DistributedSampler may pad with replicated samples)
                seen = set()
                keep_mask = []
                for pid in all_pert_ids:
                    if pid not in seen:
                        seen.add(pid)
                        keep_mask.append(True)
                    else:
                        keep_mask.append(False)

                n_unique = sum(keep_mask)
                if n_unique < len(keep_mask):
                    self.print(
                        f"Deduplication: {n_unique}/{len(keep_mask)} unique samples after removing DistributedSampler padding."
                    )

                unique_preds = preds_np[keep_mask]
                unique_pert_ids = [p for p, k in zip(all_pert_ids, keep_mask) if k]
                unique_symbols = [s for s, k in zip(all_symbols, keep_mask) if k]

                _save_test_predictions(
                    pert_ids=unique_pert_ids,
                    symbols=unique_symbols,
                    preds=unique_preds,
                    out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
                )

        return {"test_f1": test_f1} if test_f1 is not None else {}

    def configure_optimizers(self):
        # Separate param groups: GNN unfrozen layers get lower LR
        gnn_params = [p for p in self.gnn_model.parameters() if p.requires_grad]
        mlp_params = list(self.model.parameters())

        param_groups = [
            {"params": mlp_params, "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": gnn_params, "lr": self.gnn_lr, "weight_decay": self.weight_decay},
        ]

        optimizer = torch.optim.AdamW(param_groups)

        # Cosine annealing: smoother LR decay that prevents premature halving
        # T_max=100: reaches eta_min at epoch 100, then cosine restarts.
        # Using T_max=max_epochs for a single-cycle decay to eta_min=1e-7.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cosine_t_max,
            eta_min=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters (skip frozen parts of STRING_GNN backbone)."""
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                # Skip the large ESM2 embedding buffer (it's frozen and not needed in checkpoint)
                if "_esm2_emb_buf" not in name:
                    trainable[key] = full_sd[key]
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} trainable params "
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute per-gene macro F1 averaged over genes (matches calc_metric.py logic).

    preds:  [N, 3, 6640] float -- class logits
    labels: [N, 6640]    int   -- class indices in {0,1,2}
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N, 6640]
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        f1_vals.append(float(per_class_f1[present].mean()))
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    """Save test predictions in required TSV format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
        pred_list = preds[i].tolist()  # [3][6640]
        rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-1-1-1-1: STRING_GNN (partial fine-tune) + ESM2-650M precomputed + Corrected Factorized MLP"
    )
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gnn-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--bottleneck-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smooth-eps", type=float, default=0.05)
    p.add_argument("--cosine-t-max", type=int, default=100)
    p.add_argument("--early-stop-patience", type=int, default=40)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--gradient-clip-val", type=float, default=1.0)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- DataModule ---
    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup("fit")

    # --- Model ---
    model = PerturbModule(
        n_nodes=datamodule.n_nodes,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        lr=args.lr,
        gnn_lr=args.gnn_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smooth_eps=args.label_smooth_eps,
        cosine_t_max=args.cosine_t_max,
        max_epochs=args.max_epochs,
    )

    # --- Trainer config ---
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if fast_dev_run:
        limit_train = None
        limit_val = None
        limit_test = None
        max_steps = -1
        val_check_interval = 1.0  # Always validate at end of epoch
    elif args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = args.debug_max_step
        max_steps = args.debug_max_step
        val_check_interval = 1.0  # Always validate at end of epoch
    else:
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.early_stop_patience,
        min_delta=1e-5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=2,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Save test score summary
    if test_results and "test_f1" in test_results[0]:
        test_f1 = test_results[0]["test_f1"]
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"test_f1: {test_f1:.6f}\n"
            f"hidden_dim: {args.hidden_dim}\n"
            f"bottleneck_dim: {args.bottleneck_dim}\n"
            f"n_blocks: {args.n_blocks}\n"
            f"dropout: {args.dropout}\n"
            f"lr: {args.lr}\n"
            f"gnn_lr: {args.gnn_lr}\n"
            f"weight_decay: {args.weight_decay}\n"
            f"label_smooth_eps: {args.label_smooth_eps}\n"
            f"cosine_t_max: {args.cosine_t_max}\n"
            f"esm2_model: 650M (precomputed, frozen lookup)\n"
        )
        print(f"Test F1: {test_f1:.6f}")


if __name__ == "__main__":
    main()
