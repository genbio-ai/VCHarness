"""Node 1-2-1-1-1: STRING_GNN (frozen) + ESM2-35M (frozen) Gated Additive Fusion
+ 3-Block Residual MLP + Weighted CE + Label Smoothing + Per-Gene Bias
+ ReduceLROnPlateau (patience=8) + Reduced Regularization (dropout=0.35, wd=5e-4).

Architecture:
  - Input: perturbed gene ENSEMBL ID
  - STRING_GNN frozen buffer: precomputed [18870, 256] PPI graph embeddings
  - ESM2-35M frozen buffer: precomputed [18870, 480] protein sequence embeddings
  - Gated additive fusion:
      proj_str = LayerNorm(256) -> Linear(256->256)
      proj_esm = LayerNorm(480) -> Linear(480->256)
      alpha = sigmoid(gate_logit)  [learnable scalar, 1 param, initialized 0.5]
      fused = alpha * proj_str + (1-alpha) * proj_esm  [256-dim]
      fused = LayerNorm(256) -> Dropout(0.10)
  - Fallback embedding: learnable 256-dim vector for genes not in STRING (~6%)
  - Input projection: Linear(256->512) + LayerNorm(512) + GELU
  - 3 residual MLP blocks (512 -> 1024 -> 512) with LayerNorm + GELU + Dropout(0.35)
  - Output head: LayerNorm(512) + Linear(512 -> 6640*3)  [FLAT -- no factorization]
  - Per-gene bias: learnable [6640*3] additive bias term (19920 params)
  - Loss: Weighted cross-entropy + label smoothing=0.05
  - LR schedule: ReduceLROnPlateau (patience=8, factor=0.5, min_delta=1e-4, mode='max')
  - Regularization: dropout=0.35, weight_decay=5e-4

Key improvements over parent node1-2-1-1 (additive fusion, cosine annealing, F1=0.461):
  1. Reduce regularization: dropout=0.40 -> 0.35, weight_decay=1e-3 -> 5e-4
     - Parent's high regularization traded off fit capacity: train_loss=0.215 at best vs
       node1-1-1's 0.012 (same arch + STRING-only = F1=0.474)
     - Recovering fit capacity should allow the model to better exploit ESM2-35M signal
  2. Switch cosine annealing -> ReduceLROnPlateau (patience=8, factor=0.5)
     - node1-1-1 (best in tree) used ReduceLROnPlateau with two critical halvings at
       epochs 52-53 (3e-4->1.5e-4) and epoch 61 (->7.5e-5) that rescued val/F1 plateau
     - Cosine annealing in parent converged to suboptimal point (decaying slowly)
     - patience=8 is less aggressive than node1-3's patience=5 (which fired too early)
  3. Add learnable scalar gate (1 param) for STRING/ESM2 balance:
     - alpha = sigmoid(gate_logit), initialized at gate_logit=0.0 (alpha=0.5)
     - fused = alpha * proj_str + (1-alpha) * proj_esm
     - Allows the model to self-optimize the STRING vs ESM2 contribution
     - If ESM2 is unhelpful, alpha -> 1.0 (use mostly STRING)
     - If ESM2 is helpful, alpha settles below 1.0 (balanced mixture)
     - Only 1 additional parameter (minimal overfitting risk)
  4. Increase early stopping patience: 30 -> 35
     - ReduceLROnPlateau needs time for multiple LR reductions; patience=35 gives room
       for at least 2-3 LR halvings while waiting for recovery
  5. Keep FLAT output head (512->19920):
     - Factorized head (512->256->19920) failed 3 independent times in tree:
       node1-1-2 (F1=0.436), node3-2-1-1 (F1=0.358), node1-3-1 (F1=0.430)
     - Flat head is essential for output capacity on this 19920-dim classification

Distinct from parent nodes:
  - node1-2-1-1 (parent, F1=0.461): cosine annealing + warmup, dropout=0.40, wd=1e-3,
    simple additive fusion (no gate) — this node uses adaptive LR + lower regularization + gate
  - node1-1-1 (tree best, F1=0.474): STRING-only, same LR schedule/regularization as this
    node but no ESM2 — this node adds the learnable-gated ESM2 fusion on top
  - node1-3 (F1=0.463): same additive fusion + ReduceLROnPlateau but patience=5 (too
    aggressive) — this node uses patience=8
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
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
ESM2_35M_DIM = 480      # ESM2-35M embedding dimension (precomputed)
FUSED_DIM = 256         # shared projection dimension for gated additive fusion


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Dataset for gene perturbation -> differential expression prediction."""

    def __init__(
        self,
        df: pd.DataFrame,
        node_name2idx: Dict[str, int],  # ENSEMBL ID -> STRING_GNN node index (-1 if not found)
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        # Map each perturbation to STRING_GNN/ESM2 node index (-1 = not in STRING)
        gnn_indices = []
        for pid in self.pert_ids:
            idx = node_name2idx.get(pid, -1)
            gnn_indices.append(idx)
        self.gnn_indices = torch.tensor(gnn_indices, dtype=torch.long)  # [N]

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
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        # Load STRING_GNN node names to build ENSEMBL->node_index map
        if not self.node_name2idx:
            node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
            self.node_name2idx = {name: i for i, name in enumerate(node_names)}

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df, self.node_name2idx)
        self.val_ds = PerturbDataset(val_df, self.node_name2idx)
        self.test_ds = PerturbDataset(test_df, self.node_name2idx)

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
# Model components
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """MLP residual block with LayerNorm (stable across small batch sizes)."""
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.35) -> None:
        super().__init__()
        inner_dim = dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GatedAdditiveFusionModule(nn.Module):
    """Gated additive fusion of STRING_GNN and ESM2-35M embeddings.

    Improvement over parent's simple additive fusion (node1-2-1-1):
    - Adds a learnable scalar gate alpha = sigmoid(gate_logit), initialized at 0.5
    - fused = alpha * proj_str + (1-alpha) * proj_esm  (weighted combination)
    - Allows the model to self-optimize the STRING vs ESM2 balance:
      * If ESM2 provides no useful signal -> alpha -> 1.0 (effectively STRING-only)
      * If ESM2 is complementary -> alpha settles below 1.0 (balanced mixture)
    - Adds only 1 trainable parameter (vs 188K in fusion projections), minimal overfitting risk

    Architecture:
      STRING_GNN (256-dim): LayerNorm(256) -> Linear(256->256)  ==> proj_str [B, 256]
      ESM2-35M   (480-dim): LayerNorm(480) -> Linear(480->256)  ==> proj_esm [B, 256]
      alpha = sigmoid(gate_logit)  [learnable scalar, 0 to 1]
      fused = alpha * proj_str + (1-alpha) * proj_esm  [B, 256]
      fused = LayerNorm(256) -> Dropout(0.10)  [B, 256]
    """
    def __init__(
        self,
        string_dim: int = STRING_EMB_DIM,   # 256
        esm2_dim: int = ESM2_35M_DIM,       # 480
        out_dim: int = FUSED_DIM,            # 256
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        # STRING projection (256 -> 256): refine the GNN embedding
        self.string_proj = nn.Sequential(
            nn.LayerNorm(string_dim),
            nn.Linear(string_dim, out_dim),
        )
        # ESM2-35M projection (480 -> 256): compress to shared dim
        self.esm2_proj = nn.Sequential(
            nn.LayerNorm(esm2_dim),
            nn.Linear(esm2_dim, out_dim),
        )
        # Learnable gate: alpha = sigmoid(gate_logit), initialized at 0.0 -> alpha=0.5
        # alpha controls how much STRING vs ESM2 contributes to the fused representation
        self.gate_logit = nn.Parameter(torch.zeros(1))

        # Light fusion normalization
        self.fusion_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        string_emb: torch.Tensor,   # [B, 256]
        esm2_emb: torch.Tensor,     # [B, 480]
    ) -> torch.Tensor:
        proj_str = self.string_proj(string_emb)     # [B, 256]
        proj_esm = self.esm2_proj(esm2_emb)         # [B, 256]
        alpha = torch.sigmoid(self.gate_logit)       # scalar in (0, 1)
        fused = alpha * proj_str + (1.0 - alpha) * proj_esm  # [B, 256] weighted sum
        return self.dropout(self.fusion_norm(fused))  # [B, 256]


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        n_string_nodes: int,
        hidden_dim: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.35,
        lr: float = 3e-4,
        weight_decay: float = 5e-4,
        label_smoothing: float = 0.05,
        lr_patience: int = 8,
        lr_factor: float = 0.5,
        lr_min: float = 1e-6,
        max_epochs: int = 200,
        fusion_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_string_nodes = n_string_nodes
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.max_epochs = max_epochs
        self.fusion_dropout = fusion_dropout

        self.n_genes = N_GENES
        self.n_classes = N_CLASSES
        self.n_output = N_GENES * N_CLASSES  # 19920

        # Fallback learnable embedding for genes not in STRING graph (fused 256-dim)
        self.fallback_emb = nn.Parameter(torch.zeros(FUSED_DIM))

        # Gated additive fusion module: STRING(256) + ESM2-35M(480) -> 256-dim
        # Key improvement: learnable scalar gate alpha controls STRING vs ESM2 balance
        self.fusion = GatedAdditiveFusionModule(
            string_dim=STRING_EMB_DIM,
            esm2_dim=ESM2_35M_DIM,
            out_dim=FUSED_DIM,
            dropout=fusion_dropout,
        )

        # Input projection: 256-dim fused -> hidden_dim (512)
        self.input_proj = nn.Sequential(
            nn.Linear(FUSED_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Residual MLP blocks (3 blocks, proven optimal in tree)
        # dropout=0.35 matches node1-1-1 (F1=0.474), reduced from parent's 0.40
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expansion=2, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Output head: hidden_dim -> n_genes * n_classes (FLAT, no factorization)
        # Factorized head (512->256->19920) failed 3x in tree; flat head is essential.
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, self.n_output)

        # Per-gene bias: learnable additive bias capturing gene-specific baseline expression
        # Proven beneficial in node1-1-1 (+0.002 F1). 19,920 parameters.
        self.gene_bias = nn.Parameter(torch.zeros(self.n_output))

        # Buffers for frozen embeddings (populated in setup())
        # STRING_GNN PPI graph embeddings [n_string_nodes, 256]
        self.register_buffer("string_emb", torch.zeros(n_string_nodes, STRING_EMB_DIM))
        # ESM2-35M precomputed protein sequence embeddings [n_string_nodes, 480]
        self.register_buffer("esm2_emb", torch.zeros(n_string_nodes, ESM2_35M_DIM))

        # Validation / test accumulation buffers
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Class weights: inversely proportional to class frequency
        # Approx: class0 (neutral) ~92.82%, class1 (down) ~4.77%, class2 (up) ~2.41%
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq)
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        self.register_buffer("class_weights", class_weights)

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # --- Load STRING_GNN frozen embeddings ---
        if local_rank == 0:
            self.print("Loading STRING_GNN model for PPI graph embeddings...")
        from transformers import AutoModel
        string_model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        string_model.eval()

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", weights_only=False)
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        string_model = string_model.to(device)
        edge_index = graph["edge_index"].to(device)
        edge_weight = graph.get("edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        with torch.no_grad():
            outputs = string_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            string_emb = outputs.last_hidden_state.cpu().float()  # [18870, 256]

        del string_model
        torch.cuda.empty_cache()

        # Copy STRING embeddings to buffer (rank 0 only loaded these)
        self.string_emb.copy_(string_emb)
        if local_rank == 0:
            self.print(f"STRING_GNN embeddings loaded: shape={string_emb.shape}")

        # Barrier: all ranks must wait for rank 0 to finish STRING loading
        # before proceeding to ESM2 loading and parameter casting
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # --- Load precomputed ESM2-35M embeddings (frozen buffer) ---
        # Available at STRING_GNN_DIR per data/EXTERNAL_DATA.md (approved skill path)
        if local_rank == 0:
            self.print("Loading precomputed ESM2-35M protein sequence embeddings (frozen)...")
        esm2_path = STRING_GNN_DIR / "esm2_embeddings_35M.pt"
        esm2_raw = torch.load(esm2_path, weights_only=True).float()  # [18870, 480]
        assert esm2_raw.shape == (self.n_string_nodes, ESM2_35M_DIM), \
            f"Expected ESM2-35M shape [{self.n_string_nodes}, {ESM2_35M_DIM}], got {esm2_raw.shape}"

        # Copy ESM2-35M embeddings to frozen buffer
        self.esm2_emb.copy_(esm2_raw)
        if local_rank == 0:
            self.print(f"ESM2-35M embeddings loaded: shape={esm2_raw.shape}")

        # Ensure trainable parameters are float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        self.print(
            f"StringGNN+ESM2-35M-GatedFusion-MLP | fused_dim={FUSED_DIM} | hidden={self.hidden_dim} | "
            f"blocks={self.n_blocks} | dropout={self.dropout} | fusion_params={fusion_params:,} | "
            f"gate_logit={self.fusion.gate_logit.item():.4f} (alpha={torch.sigmoid(self.fusion.gate_logit).item():.3f}) | "
            f"trainable={trainable_params:,}/{total_params:,} params"
        )

    def forward(self, gnn_idx: torch.Tensor) -> torch.Tensor:
        """
        gnn_idx: [B] int tensor, -1 for genes not in STRING
        Returns: [B, 3, 6640] logits
        """
        B = gnn_idx.size(0)
        in_string_mask = gnn_idx >= 0

        # Gather STRING embeddings [B, 256]
        str_emb = torch.zeros(B, STRING_EMB_DIM, device=gnn_idx.device, dtype=self.string_emb.dtype)
        # Gather ESM2-35M embeddings [B, 480]
        esm2 = torch.zeros(B, ESM2_35M_DIM, device=gnn_idx.device, dtype=self.esm2_emb.dtype)

        if in_string_mask.any():
            valid_indices = gnn_idx[in_string_mask]
            str_emb[in_string_mask] = self.string_emb[valid_indices]
            esm2[in_string_mask] = self.esm2_emb[valid_indices]

        # Gated additive fusion: alpha * proj_str + (1-alpha) * proj_esm -> 256-dim
        fused = self.fusion(str_emb.float(), esm2.float())  # [B, 256]

        # For genes not in STRING, use learnable fallback embedding [256-dim]
        if (~in_string_mask).any():
            B_missing = (~in_string_mask).sum().item()
            fused[~in_string_mask] = self.fallback_emb.unsqueeze(0).expand(B_missing, -1).float()

        # Project 256-dim fused -> hidden_dim=512, then process through MLP blocks
        x = self.input_proj(fused)      # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)

        # Output projection + per-gene bias (FLAT head, no factorization)
        logits = self.output_linear(self.output_norm(x))  # [B, n_genes*n_classes]
        logits = logits + self.gene_bias                  # broadcast bias over batch

        return logits.view(B, self.n_classes, self.n_genes)  # [B, 3, 6640]

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy with label smoothing.

        logits: [B, 3, 6640]
        labels: [B, 6640]  values in {0,1,2}

        Uses standard weighted CE + label smoothing (proven better than focal loss in tree).
        Label smoothing=0.05 prevents overconfident predictions on the majority neutral class.
        """
        # Flatten: [B, 3, 6640] -> [B*6640, 3]
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)  # [B*6640]

        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["gnn_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # Log gate alpha to monitor if model prefers STRING vs ESM2
        alpha = torch.sigmoid(self.fusion.gate_logit).item()
        self.log("train/gate_alpha", alpha, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["gnn_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds_local = torch.cat(self._val_preds, dim=0)    # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0)  # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather across DDP ranks for accurate global F1
        all_preds = self.all_gather(preds_local)
        all_labels = self.all_gather(labels_local)
        ws = self.trainer.world_size
        if ws > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
            all_labels = all_labels.view(-1, N_GENES)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        # Squeeze world_size=1 leading dim
        if preds_np.ndim == 4 and preds_np.shape[0] == 1:
            preds_np = preds_np[0]
            labels_np = labels_np[0]

        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["gnn_idx"])  # [B, 3, 6640]
        self._test_preds.append(logits.detach().cpu())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        labels_local = torch.cat(self._test_labels, dim=0) if self._test_labels else None
        self._test_preds.clear()
        self._test_labels.clear()

        # Gather across all DDP ranks
        all_preds = self.all_gather(preds_local)  # [ws, N_local, 3, 6640] or [N_local, 3, 6640]
        ws = self.trainer.world_size
        if ws > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
        else:
            if all_preds.ndim == 4:
                all_preds = all_preds.squeeze(0)

        # Compute test F1 if labels are available (gathered on all ranks)
        if labels_local is not None:
            all_labels = self.all_gather(labels_local)
            if ws > 1:
                all_labels = all_labels.view(-1, N_GENES)
            else:
                if all_labels.ndim == 2 and all_labels.shape[0] == 1:
                    all_labels = all_labels.squeeze(0)
            preds_np = all_preds.float().cpu().numpy()
            labels_np = all_labels.cpu().numpy()
            if preds_np.ndim == 4 and preds_np.shape[0] == 1:
                preds_np = preds_np[0]
                labels_np = labels_np[0]
            f1 = _compute_per_gene_f1(preds_np, labels_np)
            self.log("test/f1", f1, prog_bar=True, sync_dist=True)

        # Gather string metadata from all ranks for predictions
        all_pert_ids: List[str] = []
        all_symbols: List[str] = []
        if self.trainer.is_global_zero:
            _pert_gathered: List[List[str]] = [[] for _ in range(ws)]
            _sym_gathered: List[List[str]] = [[] for _ in range(ws)]
            torch.distributed.gather_object(self._test_pert_ids, _pert_gathered, dst=0)
            torch.distributed.gather_object(self._test_symbols, _sym_gathered, dst=0)
            for p_list, s_list in zip(_pert_gathered, _sym_gathered):
                all_pert_ids.extend(p_list)
                all_symbols.extend(s_list)
        else:
            torch.distributed.gather_object(self._test_pert_ids, dst=0)
            torch.distributed.gather_object(self._test_symbols, dst=0)

        if self.trainer.is_global_zero:
            preds_np = all_preds.float().cpu().numpy()  # [N, 3, 6640]
            # De-duplicate by pert_id (in case of DDP padding)
            seen = set()
            dedup_pert_ids, dedup_symbols, dedup_preds = [], [], []
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_pert_ids.append(pid)
                    dedup_symbols.append(all_symbols[i])
                    dedup_preds.append(preds_np[i])
            dedup_preds_np = np.stack(dedup_preds, axis=0) if dedup_preds else preds_np

            _save_test_predictions(
                pert_ids=dedup_pert_ids,
                symbols=dedup_symbols,
                preds=dedup_preds_np,
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

        # Clear metadata lists on ALL ranks to prevent memory leaks in multi-test scenarios
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        # Only optimize trainable parameters (frozen buffers are excluded)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # ReduceLROnPlateau: adaptive LR reduction based on validation F1.
        # This mirrors node1-1-1's successful strategy (F1=0.474) where two LR halvings
        # at epochs 52-53 (3e-4->1.5e-4) and 61 (->7.5e-5) rescued val/F1 from a plateau.
        # patience=8 is less aggressive than node1-3's patience=5 (which fired too early
        # on 141-sample noisy validation), but still responsive to genuine plateaus.
        # min_delta=1e-4 prevents spurious triggers from noise.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.lr_min,
            threshold=1e-4,
            threshold_mode="abs",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/f1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_sd = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable_sd[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                trainable_sd[key] = full_sd[key]

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute per-gene macro F1 averaged over genes (matches calc_metric.py logic exactly).

    preds:  [N, 3, 6640] float  — class logits (argmax selects predicted class)
    labels: [N, 6640]    int    — class indices in {0,1,2}
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
    """Save test predictions in required TSV format (idx, input, prediction)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
        pred_list = preds[i].tolist()  # shape [3][6640] -> nested lists
        rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-2-1-1-1: STRING_GNN + ESM2-35M Gated Additive Fusion + 3-Block MLP + "
                    "Weighted CE + Per-Gene Bias + ReduceLROnPlateau + Reduced Regularization"
    )
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--fusion-dropout", type=float, default=0.10)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lr-patience", type=int, default=8)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--early-stop-patience", type=int, default=35)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load STRING_GNN node count
    node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    n_string_nodes = len(node_names)

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
        n_string_nodes=n_string_nodes,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        max_epochs=args.max_epochs,
        fusion_dropout=args.fusion_dropout,
    )

    # --- Trainer config ---
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        max_steps = -1

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
        enable_version_counter=False,
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
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=300)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
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

    # Save score
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results -> {score_path}")


if __name__ == "__main__":
    main()
