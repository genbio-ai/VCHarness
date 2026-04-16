"""Node 1-2-1-1-1-1-1: STRING_GNN (partial fine-tuning) + PostLN MLP
+ Muon+AdamW optimizer + WCE + Manifold Mixup + CosineAnnealingWarmRestarts

Architecture:
  - Input: perturbed gene ENSEMBL ID
  - STRING_GNN partially fine-tuned (mps.6, mps.7, post_mp), gnn_lr=5e-5
  - No ESM2 fusion (gate stagnation at ~0.5 across all ESM2 nodes; STRING-only is cleaner)
  - Fallback embedding: learnable 256-dim for genes not in STRING (~6%)
  - Input projection: Linear(256->512) + LayerNorm(512) + GELU
  - 3 PostLN residual MLP blocks (512 -> 1024 -> 512) with GELU + Dropout(0.35)
  - Output head: Dropout(0.15) + Linear(512 -> 6640*3)  [FLAT, no factorization]
  - NO per-gene bias (confirmed negligible across best nodes in tree)
  - Loss: Weighted Cross Entropy (WCE) + Manifold Mixup (alpha=0.2, prob=0.65)
  - Optimizer: Muon (lr=0.01) for MLP hidden weights, AdamW (lr=3e-4) for head/norms/biases
    + separate AdamW group for unfrozen GNN at gnn_lr=5e-5
  - LR schedule: CosineAnnealingWarmRestarts (T_0=80, T_mult=2)
  - max_epochs=400, early_stop_patience=80

Key improvements over parent node1-2-1-1-1-1 (AdamW+focal+ESM2-gated, F1=0.4844):
  1. Muon optimizer for MLP hidden weights: ALL tree-best nodes (F1>=0.4966) use Muon.
     Evidence: node3-3-1-2 (F1=0.4966), node3-3-1-2-1-1-1 (F1=0.5243), etc.
     CRITICAL: Must use WCE, NOT focal loss. Muon + focal loss is catastrophically
     incompatible (node1-1-3-1: F1=0.191).
  2. Weighted Cross Entropy (WCE) instead of focal loss: Required pairing with Muon.
     WCE is proven in all Muon-based high-performing nodes.
  3. Remove ESM2 gated fusion: Gate alpha stagnated at ~0.5 throughout 212 epochs in
     parent — model never learned to favor STRING or ESM2, just blended 50/50.
     STRING-only PostLN recipe (node1-1-1-2-1, F1=0.4912) outperforms every ESM2-fusion
     variant in this lineage. Removing ESM2 avoids the gate optimization dead-end.
  4. Keep partial GNN fine-tuning (mps.6, mps.7, post_mp at gnn_lr=5e-5): Proven
     +0.002 marginal F1 gain from task-specific GNN adaptation.
  5. Increase weight_decay: 5e-4 -> 8e-4. Matches all best Muon nodes.
  6. Increase mixup_prob: 0.5 -> 0.65. More aggressive augmentation for 1,273-sample dataset.
  7. Increase early_stop_patience: 60 -> 80. CosineWR restarts at epochs 80/240 need
     full recovery time. Patience=80 allows one full cycle for recovery.
  8. Keep PostLN, head_dropout=0.15, flat output head, CosineWarmRestarts (all proven).
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
N_CLASSES = 3           # {-1->0 (down), 0->1 (neutral), 1->2 (up)}
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
STRING_EMB_DIM = 256    # STRING_GNN output embedding dimension


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Dataset for gene perturbation -> differential expression prediction."""

    def __init__(
        self,
        df: pd.DataFrame,
        node_name2idx: Dict[str, int],
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        # Map each perturbation to STRING_GNN node index (-1 = not in STRING)
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
# Loss function: Weighted Cross Entropy (WCE)
# ---------------------------------------------------------------------------
def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted multi-class cross entropy loss.

    logits:  [N, C] float
    targets: [N]    long  values in {0,...,C-1}
    class_weights: [C] float

    WCE is used (not focal loss) because Muon optimizer is catastrophically
    incompatible with focal loss (node1-1-3-1: F1=0.191). All high-performing
    Muon nodes use WCE: node3-3-1-2-1-1-1 (F1=0.5243), node3-3-1-2-1 (F1=0.5170).
    """
    return F.cross_entropy(logits.float(), targets, weight=class_weights.float())


def wce_mixup(
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """WCE loss for Manifold Mixup (convex combination of two losses).

    h_mix = lam * h_a + (1-lam) * h_b[perm]
    L_mix = lam * wce(h_mix, a) + (1-lam) * wce(h_mix, b)
    """
    loss_a = weighted_cross_entropy(logits, targets_a, class_weights)
    loss_b = weighted_cross_entropy(logits, targets_b, class_weights)
    return lam * loss_a + (1 - lam) * loss_b


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class PostLNResidualBlock(nn.Module):
    """Post-LayerNorm residual MLP block.

    PostLN applies LayerNorm AFTER the sublayer and residual addition:
      out = LayerNorm(x + sublayer(x))

    For shallow networks (3 blocks), PostLN provides stronger gradient flow
    through the skip connection vs PreLN (where the skip path bypasses normalization).
    Evidence: node1-1-1-2-1 (PostLN, F1=0.4912) vs node1-1-1-2 (PreLN, F1=0.4711).
    """
    def __init__(self, dim: int, inner_dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PostLN: apply sublayer, add residual, then normalize
        return self.norm(x + self.net(x))


class StringOnlyPostLNMLP(nn.Module):
    """STRING_GNN + PostLN MLP with head dropout.

    Architecture:
    1. STRING_GNN lookup (with fallback for ~6% missing genes)
    2. Input projection (256 -> 512)
    3. 3x PostLN residual MLP blocks (512 -> 1024 -> 512)
    4. Output head: Dropout(head_dropout) + Linear(512 -> n_genes * n_classes) FLAT

    No ESM2 or other fusion — STRING-only is cleaner and gate stagnation in
    the parent confirmed ESM2 provides no complementary signal in this lineage.
    """

    def __init__(
        self,
        gnn_emb_dim: int = STRING_EMB_DIM,
        hidden_dim: int = 512,
        n_blocks: int = 3,
        n_genes: int = N_GENES,
        n_classes: int = N_CLASSES,
        dropout: float = 0.35,
        head_dropout: float = 0.15,
    ) -> None:
        super().__init__()

        # Fallback embedding for genes not in STRING_GNN (~6%)
        self.fallback_emb = nn.Embedding(1, gnn_emb_dim)
        nn.init.normal_(self.fallback_emb.weight, std=0.01)

        # Input projection: gnn_emb_dim (256) -> hidden_dim (512)
        self.input_proj = nn.Sequential(
            nn.Linear(gnn_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # PostLN residual MLP blocks
        self.blocks = nn.ModuleList([
            PostLNResidualBlock(hidden_dim, hidden_dim * 2, dropout)
            for _ in range(n_blocks)
        ])

        # Output head with head_dropout before final linear
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, n_genes * n_classes),
        )

        self.n_genes = n_genes
        self.n_classes = n_classes

    def _get_string_embedding(
        self,
        gnn_node_emb: torch.Tensor,   # [N_nodes, 256] - STRING_GNN output
        gnn_idx: torch.Tensor,         # [B] - index into node embeddings, or -1 if missing
    ) -> torch.Tensor:
        """Gather STRING embedding for batch, applying fallback for missing genes."""
        B = gnn_idx.size(0)
        in_string_mask = gnn_idx >= 0

        # Build STRING embedding tensor for the batch
        str_emb = torch.zeros(B, STRING_EMB_DIM, device=gnn_idx.device, dtype=gnn_node_emb.dtype)

        if in_string_mask.any():
            valid_indices = gnn_idx[in_string_mask]
            str_emb[in_string_mask] = gnn_node_emb[valid_indices]

        # For genes not in STRING, replace with learnable fallback
        if (~in_string_mask).any():
            n_missing = int((~in_string_mask).sum().item())
            fallback_idx = torch.zeros(n_missing, dtype=torch.long, device=gnn_idx.device)
            str_emb[~in_string_mask] = self.fallback_emb(fallback_idx).float()

        return str_emb  # [B, 256]

    def forward(
        self,
        gnn_node_emb: torch.Tensor,
        gnn_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Standard forward pass (no Mixup)."""
        emb = self._get_string_embedding(gnn_node_emb, gnn_idx).float()
        x = self.input_proj(emb)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)
        return logits.view(gnn_idx.size(0), self.n_classes, self.n_genes)  # [B, 3, 6640]

    def forward_with_mixup(
        self,
        gnn_node_emb: torch.Tensor,
        gnn_idx: torch.Tensor,
        mixup_lam: float,
        mixup_perm: torch.Tensor,
        mixup_block_idx: int,
    ) -> torch.Tensor:
        """Forward pass with Manifold Mixup applied at the specified block index.

        Manifold Mixup interpolates hidden representations at a random PostLN block:
          h_mix = lam * h_a + (1-lam) * h_b[perm]
        Creates biologically diverse training examples in hidden space,
        preventing memorization on the small 1,273-sample training set.
        Evidence: +0.019 to +0.032 F1 across independent tree nodes.
        """
        emb = self._get_string_embedding(gnn_node_emb, gnn_idx).float()
        x = self.input_proj(emb)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == mixup_block_idx:
                # Apply Manifold Mixup AFTER this PostLN block
                x = mixup_lam * x + (1 - mixup_lam) * x[mixup_perm]

        logits = self.head(x)
        return logits.view(gnn_idx.size(0), self.n_classes, self.n_genes)  # [B, 3, 6640]


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
        head_dropout: float = 0.15,
        muon_lr: float = 0.01,
        adamw_lr: float = 3e-4,
        gnn_lr: float = 5e-5,
        weight_decay: float = 8e-4,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.65,
        t0: int = 80,
        t_mult: int = 2,
        max_epochs: int = 400,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_string_nodes = n_string_nodes
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.muon_lr = muon_lr
        self.adamw_lr = adamw_lr
        self.gnn_lr = gnn_lr
        self.weight_decay = weight_decay
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.t0 = t0
        self.t_mult = t_mult
        self.max_epochs = max_epochs

        # The main model and GNN backbone (initialized in setup)
        self.model: Optional[StringOnlyPostLNMLP] = None
        self.gnn_model = None  # STRING_GNN partially fine-tuned

        # GNN embedding cache for eval/test mode
        self._gnn_emb_cache: Optional[torch.Tensor] = None

        # Accumulation buffers for val/test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        from transformers import AutoModel

        # Class weights for WCE:
        # After +1 shift: class0=down-reg(4.77%), class1=neutral(92.82%), class2=up-reg(2.41%)
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = 1.0 / freq
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        self.register_buffer("class_weights", class_weights)

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # --- Load STRING_GNN backbone (partially fine-tuned) ---
        if self.gnn_model is None:
            self.gnn_model = AutoModel.from_pretrained(
                str(STRING_GNN_DIR), trust_remote_code=True
            )
            # Freeze all parameters first
            for param in self.gnn_model.parameters():
                param.requires_grad = False

            # Partially unfreeze: last 2 GNN layers + post_mp projection
            # Per STRING_GNN skill: tune mps.6.*, mps.7.*, post_mp.*
            layers_to_unfreeze = ["mps.6.", "mps.7.", "post_mp."]
            unfrozen_params = 0
            for name, param in self.gnn_model.named_parameters():
                if any(name.startswith(prefix) for prefix in layers_to_unfreeze):
                    param.requires_grad = True
                    unfrozen_params += param.numel()

            self.print(
                f"STRING_GNN loaded: {sum(p.numel() for p in self.gnn_model.parameters()):,} total, "
                f"{unfrozen_params:,} unfrozen (mps.6, mps.7, post_mp) at gnn_lr={self.gnn_lr}"
            )
            # Cast trainable GNN params to float32 for stable optimization
            for param in self.gnn_model.parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        # Barrier: all ranks wait for GNN load completion
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # --- Initialize StringOnlyPostLNMLP model ---
        if self.model is None:
            self.model = StringOnlyPostLNMLP(
                gnn_emb_dim=STRING_EMB_DIM,
                hidden_dim=self.hidden_dim,
                n_blocks=self.n_blocks,
                dropout=self.dropout,
                head_dropout=self.head_dropout,
            )
            # Cast trainable MLP params to float32 for stable optimization
            for v in self.model.parameters():
                if v.requires_grad:
                    v.data = v.data.float()

            mlp_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print(
                f"StringOnlyPostLNMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
                f"dropout={self.dropout} | head_dropout={self.head_dropout} | "
                f"MLP trainable={mlp_trainable:,}"
            )

    def _get_gnn_embeddings(self) -> torch.Tensor:
        """Get STRING_GNN node embeddings.

        During training: recompute so gradients flow through unfrozen GNN layers.
        During eval/test: use cached embeddings for efficiency.
        """
        if self.training:
            # Recompute each training step to allow gradient flow through unfrozen layers
            device = next(self.model.parameters()).device
            self.gnn_model = self.gnn_model.to(device)
            graph = torch.load(
                STRING_GNN_DIR / "graph_data.pt",
                map_location=device,
                weights_only=False,
            )
            edge_index = graph["edge_index"]
            edge_weight = graph.get("edge_weight", None)
            outputs = self.gnn_model(edge_index=edge_index, edge_weight=edge_weight)
            return outputs.last_hidden_state.float()  # [N_nodes, 256]
        else:
            # Cache embeddings in eval mode for efficiency
            if self._gnn_emb_cache is None:
                device = next(self.model.parameters()).device
                self.gnn_model = self.gnn_model.to(device)
                graph = torch.load(
                    STRING_GNN_DIR / "graph_data.pt",
                    map_location=device,
                    weights_only=False,
                )
                edge_index = graph["edge_index"]
                edge_weight = graph.get("edge_weight", None)
                with torch.no_grad():
                    outputs = self.gnn_model(edge_index=edge_index, edge_weight=edge_weight)
                self._gnn_emb_cache = outputs.last_hidden_state.float().detach()
            return self._gnn_emb_cache

    def on_train_epoch_start(self) -> None:
        # Invalidate embedding cache at epoch start so val/test recomputes with updated GNN weights
        self._gnn_emb_cache = None

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """WCE loss (no label smoothing, no focal modulation).

        logits: [B, 3, 6640]
        labels: [B, 6640]  values in {0,1,2}

        WCE is REQUIRED with Muon optimizer. Using focal loss with Muon caused
        catastrophic failure in node1-1-3-1 (F1=0.191). All Muon-based high-performing
        nodes (F1>=0.4966) use WCE.
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return weighted_cross_entropy(logits_flat, labels_flat, self.class_weights)

    def _compute_loss_mixup(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """WCE loss for Manifold Mixup.

        logits:   [B, 3, 6640]
        labels_a: [B, 6640] original labels
        labels_b: [B, 6640] permuted labels for Mixup
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        la_flat = labels_a.reshape(-1)
        lb_flat = labels_b.reshape(-1)
        return wce_mixup(logits_flat, la_flat, lb_flat, lam, self.class_weights)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        gnn_emb = self._get_gnn_embeddings()   # [N_nodes, 256]
        labels = batch["label"]                 # [B, 6640]
        B = labels.size(0)

        # Manifold Mixup: randomly decide whether to apply on this batch
        use_mixup = (
            self.training
            and self.mixup_prob > 0
            and B > 1
            and torch.rand(1).item() < self.mixup_prob
        )

        if use_mixup:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            perm = torch.randperm(B, device=labels.device)
            mixup_block_idx = int(torch.randint(0, self.n_blocks, (1,)).item())

            logits = self.model.forward_with_mixup(
                gnn_emb, batch["gnn_idx"], lam, perm, mixup_block_idx
            )
            labels_b = labels[perm]
            loss = self._compute_loss_mixup(logits, labels, labels_b, lam)
        else:
            logits = self.model(gnn_emb, batch["gnn_idx"])
            loss = self._compute_loss(logits, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        gnn_emb = self._get_gnn_embeddings()
        logits = self.model(gnn_emb, batch["gnn_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds_local = torch.cat(self._val_preds, dim=0)   # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0)  # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather across DDP ranks for accurate global F1
        all_preds = self.all_gather(preds_local)
        all_labels = self.all_gather(labels_local)
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
        all_labels = all_labels.view(-1, N_GENES)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()

        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        gnn_emb = self._get_gnn_embeddings()
        logits = self.model(gnn_emb, batch["gnn_idx"])  # [B, 3, 6640]
        self._test_preds.append(logits.detach().cpu())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> Optional[Dict[str, Any]]:
        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        ws = self.trainer.world_size
        all_preds = self.all_gather(preds_local)
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

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
            preds_np = all_preds.float().cpu().numpy()

            # Deduplicate by pert_id (DistributedSampler may pad with replicated samples)
            seen: set = set()
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
                    f"Deduplication: {n_unique}/{len(keep_mask)} unique after DDP padding removal."
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
        """Muon+AdamW dual optimizer configuration.

        Muon is applied to MLP hidden weight matrices (ndim >= 2, excluding head and input_proj).
        AdamW is applied to:
          - Normalization layers, biases, scalar params (ndim < 2)
          - Input projection (Linear + LayerNorm) — first layer excluded per Muon skill
          - Output head — last layer excluded per Muon skill
          - Fallback embedding
        GNN unfrozen params get a separate AdamW group at gnn_lr.

        Reference: MuonWithAuxAdam from muon-optimizer-skill.
        CRITICAL: Use WCE loss (not focal) with Muon. Muon+focal causes catastrophic
        failure (node1-1-3-1: F1=0.191).
        """
        from muon import MuonWithAuxAdam

        # Identify MLP hidden weight matrices (ndim >= 2) that are NOT:
        # - In input_proj (first layer)
        # - In head (output layer)
        # - In fallback_emb (embedding)
        # These are the weight matrices in self.model.blocks (3x PostLN residual blocks)
        hidden_weight_params = []
        adamw_mlp_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "blocks." in name and param.ndim >= 2:
                    # Hidden weight matrices in PostLN residual blocks -> Muon
                    hidden_weight_params.append(param)
                else:
                    # Everything else (input_proj, head, fallback_emb, norms, biases) -> AdamW
                    adamw_mlp_params.append(param)

        # GNN unfrozen parameters -> separate AdamW group
        gnn_params = [p for p in self.gnn_model.parameters() if p.requires_grad]

        param_groups = [
            # Muon group for MLP hidden weight matrices
            dict(
                params=hidden_weight_params,
                use_muon=True,
                lr=self.muon_lr,
                weight_decay=self.weight_decay,
                momentum=0.95,
            ),
            # AdamW group for MLP non-hidden params (head, proj, norms, biases)
            dict(
                params=adamw_mlp_params,
                use_muon=False,
                lr=self.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=self.weight_decay,
            ),
            # AdamW group for unfrozen GNN params
            dict(
                params=gnn_params,
                use_muon=False,
                lr=self.gnn_lr,
                betas=(0.9, 0.95),
                weight_decay=self.weight_decay,
            ),
        ]

        self.print(
            f"Optimizer: MuonWithAuxAdam | "
            f"Muon hidden weights: {sum(p.numel() for p in hidden_weight_params):,} | "
            f"AdamW MLP other: {sum(p.numel() for p in adamw_mlp_params):,} | "
            f"AdamW GNN unfrozen: {sum(p.numel() for p in gnn_params):,}"
        )

        optimizer = MuonWithAuxAdam(param_groups)

        # CosineAnnealingWarmRestarts: warm restarts escape local optima.
        # T_0=80 -> first restart at epoch 80
        # T_mult=2 -> cycles: 80, 160, 320 epochs
        # With max_epochs=400, we complete 2 full cycles + beginning of 3rd.
        # Evidence: node3-3-1-2-1-1-1 (F1=0.5243) peaked at epoch 137 in cycle 2.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.t0,
            T_mult=self.t_mult,
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
        description=(
            "Node1-2-1-1-1-1-1: STRING_GNN (partial fine-tune) + PostLN MLP "
            "+ Muon+AdamW + WCE + Manifold Mixup + CosineWarmRestarts"
        )
    )
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=400)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--gnn-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=8e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--head-dropout", type=float, default=0.15)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.65)
    p.add_argument("--t0", type=int, default=80)
    p.add_argument("--t-mult", type=int, default=2)
    p.add_argument("--early-stop-patience", type=int, default=80)
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
        head_dropout=args.head_dropout,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        gnn_lr=args.gnn_lr,
        weight_decay=args.weight_decay,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        t0=args.t0,
        t_mult=args.t_mult,
        max_epochs=args.max_epochs,
    )

    # --- Trainer configuration ---
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
        filename="best-{epoch:03d}-{val_f1:.4f}",
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
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=3600)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
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

    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
