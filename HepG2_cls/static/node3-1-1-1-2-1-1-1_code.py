"""Node improvement: STRING_GNN (frozen) + Muon optimizer + hidden_dim=384 + WCE loss +
CosineAnnealingWarmRestarts + Manifold Mixup + output head dropout.

Architecture Overview:
  - Precomputed frozen STRING_GNN embeddings (256-dim): proven PPI graph signal
  - 3-block Residual MLP head with hidden_dim=384: proven capacity from node1-3-2 (F1=0.4756)
  - Per-gene learnable bias (19920 params): captures baseline DEG tendencies
  - Unfactorized output head Linear(384->19920): maintains full expressive capacity
  - Output head dropout (p=0.15): node1-3-2-2-1 (F1=0.4777) proved this regularizes the 7.65M head
  - Muon optimizer (lr=0.01) for hidden block weight matrices
  - AdamW (lr=3e-4) for embeddings, norms, biases, output head
  - CosineAnnealingWarmRestarts (T_0=80, T_mult=2): enables escaping local optima
  - Manifold Mixup (alpha=0.2, prob=0.5): effective regularization for 1,273 samples
  - Weighted cross-entropy loss (NOT focal): proven compatible with Muon optimizer
  - Gradient clipping (max_norm=1.0): stabilizes Muon training

Key Improvements over Parent (node3-1-1-1-2-1-1, F1=0.4724):
  1. REPLACE focal loss with weighted cross-entropy:
     - Parent used focal loss (gamma=2.0) which is INCOMPATIBLE with Muon (node1-1-3-1: catastrophic failure)
     - node1-3-2 (F1=0.4756, the inspiration node) used WCE with Muon
     - node1-3-3 (F1=0.4950), node1-3-2-2-1-1-1-1-1-1 (F1=0.4968) all use WCE with Muon
     - This is the single most impactful fix
  2. SWITCH from ReduceLROnPlateau to CosineAnnealingWarmRestarts (T_0=80, T_mult=2):
     - Parent training was truncated at epoch 200 with model still improving
     - CosineWarmRestarts enables escaping local optima through LR restarts
     - node1-3-3 (F1=0.4950) used T_0=80, T_mult=2, 500 epochs -- same recipe
  3. EXTEND max_epochs to 500:
     - Parent was cut off at 200 epochs mid-trajectory
     - node1-3-3 completed 500 epochs and achieved F1=0.4950
  4. ADD Manifold Mixup (alpha=0.2, prob=0.5):
     - node1-3-3 (F1=0.4950), node1-3-2-2-1-1-1-1-1-1 (F1=0.4968) both used Mixup
     - Doubles effective training set through embedding interpolation on 1,273 samples
  5. ADD output head dropout (p=0.15):
     - node1-3-2-2-1 (F1=0.4777) showed +0.0021 gain from isolating this as the only change
     - Provides targeted regularization of the dominant overfitting source (7.65M-param head)
  6. REDUCE trunk dropout from 0.40 to 0.30:
     - With Manifold Mixup providing stochastic regularization, trunk dropout can be reduced
     - node1-3-3 (F1=0.4950) used dropout=0.30 (lower than parent's 0.40)
     - Allows model to fit training data more effectively while Mixup prevents overfitting
  7. REDUCE Muon lr from 0.02 to 0.01:
     - node1-3-3 (F1=0.4950) and all top STRING-only nodes use Muon lr=0.01
     - 0.01 provides more stable convergence; 0.02 can cause instability with CosineWarmRestarts
  8. REDUCE weight_decay from 0.01 to 8e-4:
     - Matching proven recipe from node1-3-3 (F1=0.4950) and node1-3-2-2-1-1-1-1-1-1 (F1=0.4968)
     - With WCE + Mixup + head_dropout, heavy weight_decay is redundant
  9. INCREASE early_stop patience from 30 to 80:
     - CosineWarmRestarts requires longer patience to complete cycles
     - node1-3-3 used patience similar to the epoch window of a full cycle
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
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES = 6640
N_CLASSES = 3
STRING_EMB_DIM = 256   # STRING_GNN output dim
FEATURE_DIM = STRING_EMB_DIM  # 256


# ---------------------------------------------------------------------------
# Weighted Cross-Entropy Loss
# ---------------------------------------------------------------------------
def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted cross-entropy loss (no focal modulation).

    logits: [N, C] (raw unnormalized scores)
    targets: [N] (class indices in {0,1,2})
    weight: [C] class weights

    WCE is proven compatible with Muon optimizer:
    - node1-3-2 (F1=0.4756): Muon + WCE
    - node1-3-3 (F1=0.4950): Muon + WCE + Mixup
    - node1-3-2-2-1-1-1-1-1-1 (F1=0.4968): Muon + WCE + Mixup

    Focal loss (gamma=2.0) is INCOMPATIBLE with Muon:
    - node1-1-3-1: Muon + focal => catastrophic failure (F1=0.1910)
    - node3-1-1-1-2-1-1 (parent): Muon + focal => only F1=0.4724
    """
    return F.cross_entropy(logits, targets, weight=weight.to(logits.dtype))


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block.

    Architecture: LayerNorm -> FC1 -> GELU -> Dropout -> FC2 -> Dropout -> residual add
    hidden_dim=384: proven in node1-3-2 (F1=0.4756) and node1-3-3 (F1=0.4950).
    dropout=0.30: lower than parent's 0.40 since Manifold Mixup provides regularization.
    """

    def __init__(
        self,
        dim: int,
        expand: int = 2,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expand)
        self.fc2 = nn.Linear(dim * expand, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


# ---------------------------------------------------------------------------
# Prediction Head
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """3-block residual MLP with unfactorized output + output head dropout + per-gene bias.

    Key design choices:
    - hidden_dim=384: node1-3-2 (F1=0.4756) proved this beats 512 (overfitting)
    - 3 blocks: node1-1-1 (F1=0.474) and node1-3-2 (F1=0.4756) both used 3 blocks
    - Unfactorized Linear(384->19920): maintains expressive output head
    - Output head dropout (p=0.15): node1-3-2-2-1 (F1=0.4777) proved +0.0021 improvement
    - Per-gene bias: captures baseline DEG tendencies per response gene
    """

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = 384,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        dropout: float = 0.30,
        head_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes

        # Input projection: FEATURE_DIM -> hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual MLP blocks (3 blocks)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=2, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Output head with dropout: targeted regularization of dominant overfitting source
        # head_dropout=0.15 proven by node1-3-2-2-1 (F1=0.4777) as optimal
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head_dropout = nn.Dropout(head_dropout)
        self.out_proj = nn.Linear(hidden_dim, n_genes * N_CLASSES)

        # Per-gene bias: captures baseline DEG tendencies per response gene
        # Parameters: 6640 * 3 = 19,920 (negligible but informative)
        self.per_gene_bias = nn.Parameter(torch.zeros(n_genes * N_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)               # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)                      # [B, hidden_dim]
        x = self.head_norm(x)                # [B, hidden_dim]
        x = self.head_dropout(x)             # [B, hidden_dim] -- regularize output head
        out = self.out_proj(x)               # [B, N_GENES * N_CLASSES]
        out = out + self.per_gene_bias.to(out.dtype)  # add per-gene bias
        return out.view(-1, N_CLASSES, self.n_genes)  # [B, 3, N_GENES]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Maps each perturbed gene to its precomputed STRING_GNN feature vector."""

    def __init__(
        self,
        df: pd.DataFrame,
        gene_features: torch.Tensor,
        ensg_to_idx: Dict[str, int],
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_features = gene_features       # [N_NODES, FEATURE_DIM] CPU float32
        self.ensg_to_idx = ensg_to_idx

        if "label" in df.columns:
            labels = np.array(
                [json.loads(x) for x in df["label"].tolist()], dtype=np.int64
            )
            # Shift labels: {-1, 0, 1} -> {0, 1, 2}
            self.labels: Optional[torch.Tensor] = torch.tensor(
                labels + 1, dtype=torch.long
            )
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pert_id = self.pert_ids[idx]
        gnn_idx = self.ensg_to_idx.get(pert_id, -1)

        if gnn_idx >= 0:
            feat = self.gene_features[gnn_idx]   # [FEATURE_DIM]
        else:
            # Fallback: zero vector for genes not in STRING graph (~7% of data)
            feat = torch.zeros(self.gene_features.shape[1])

        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": pert_id,
            "symbol": self.symbols[idx],
            "features": feat,
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
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
        micro_batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

        self.gene_features: Optional[torch.Tensor] = None
        self.ensg_to_idx: Optional[Dict[str, int]] = None
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        # Precompute STRING_GNN features (run once per process)
        if self.gene_features is None:
            self._precompute_features()

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df, self.gene_features, self.ensg_to_idx)
        self.val_ds = PerturbDataset(val_df, self.gene_features, self.ensg_to_idx)
        self.test_ds = PerturbDataset(test_df, self.gene_features, self.ensg_to_idx)

    def _precompute_features(self) -> None:
        """Run STRING_GNN forward once to get frozen PPI topology embeddings [N, 256].

        Using FROZEN STRING_GNN is the proven strategy in this lineage:
        - node1-3-2 (F1=0.4756): frozen STRING_GNN
        - node1-3-3 (F1=0.4950): frozen STRING_GNN
        - Partial GNN fine-tuning consistently hurt in the node3 lineage

        DDP-safe: all ranks precompute independently (deterministic, identical results).
        """
        import torch.distributed as dist

        model_dir = Path(STRING_GNN_DIR)
        is_dist = dist.is_available() and dist.is_initialized()

        if is_dist:
            dist.barrier()

        # Build node index map
        node_names: List[str] = json.loads(
            (model_dir / "node_names.json").read_text()
        )
        self.ensg_to_idx = {name: i for i, name in enumerate(node_names)}

        device = torch.device("cuda")
        print("Loading STRING_GNN for precomputing topology embeddings...", flush=True)
        gnn = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device)
        gnn.eval()

        graph = torch.load(model_dir / "graph_data.pt", map_location=device)
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)

        with torch.no_grad():
            out = gnn(
                edge_index=edge_index,
                edge_weight=edge_weight,
                output_hidden_states=False,
            )
            self.gene_features = out.last_hidden_state.float().cpu()  # [N, 256]

        del gnn, graph, out
        torch.cuda.empty_cache()

        print(
            f"Precomputed gene features: {self.gene_features.shape} "
            f"(STRING_GNN frozen PPI topology)",
            flush=True,
        )

        if is_dist:
            dist.barrier()

    def _make_loader(self, ds: PerturbDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = 384,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        lr: float = 3e-4,
        muon_lr: float = 0.01,
        weight_decay: float = 8e-4,
        dropout: float = 0.30,
        head_dropout: float = 0.15,
        t0: int = 80,
        t_mult: int = 2,
        lr_eta_min: float = 1e-5,
        grad_clip_val: float = 1.0,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.head: Optional[PerturbHead] = None

        # Accumulation buffers for validation and test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []

        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        self.head = PerturbHead(
            in_dim=self.hparams.in_dim,
            hidden_dim=self.hparams.hidden_dim,
            n_genes=self.hparams.n_genes,
            n_blocks=self.hparams.n_blocks,
            dropout=self.hparams.dropout,
            head_dropout=self.hparams.head_dropout,
        )

        # Cast to float32 for stable optimization
        for p in self.head.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Class weights: shifted labels {0:down, 1:neutral, 2:up}
        # Frequencies from DATA_ABSTRACT: down=4.77%, neutral=92.82%, up=2.41%
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = 1.0 / freq
        class_weights = class_weights / class_weights.mean()
        self.register_buffer("class_weights", class_weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"STRING_GNN (frozen) + 3-block MLP (h={self.hparams.hidden_dim}) + "
            f"Muon optimizer + WCE + Manifold Mixup + Head Dropout + CosineWarmRestarts | "
            f"trainable={trainable:,}/{total:,} | "
            f"dropout={self.hparams.dropout}, head_dropout={self.hparams.head_dropout}, "
            f"wd={self.hparams.weight_decay}, muon_lr={self.hparams.muon_lr}, "
            f"mixup_prob={self.hparams.mixup_prob}"
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy loss (compatible with Muon optimizer).

        CRITICAL: WCE (not focal loss) is used here because:
        1. Muon + focal loss is INCOMPATIBLE (node1-1-3-1: catastrophic failure F1=0.19)
        2. node1-3-2 (F1=0.4756) used Muon + WCE -> best STRING-only result before node1-3-3
        3. node1-3-3 (F1=0.4950) used Muon + WCE + Mixup -> confirmed superiority
        """
        # logits: [B, 3, N_GENES], labels: [B, N_GENES] in {0,1,2}
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return weighted_cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
        )

    def _manifold_mixup(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Manifold Mixup augmentation in the input embedding space.

        Proven effective for small datasets (1,273 samples):
        - node1-3-3 (F1=0.4950): Mixup prob=0.5, alpha=0.2
        - node1-3-2-2-1-1-1-1-1-1 (F1=0.4968): Mixup prob=0.5, alpha=0.2

        Returns mixed features and mixed labels (as soft float labels for WCE).
        """
        if not self.training or torch.rand(1).item() > self.hparams.mixup_prob:
            return feats, labels, None, None  # no mixup

        batch_size = feats.size(0)
        lam = torch.distributions.Beta(
            self.hparams.mixup_alpha, self.hparams.mixup_alpha
        ).sample().item()

        # Random permutation for mixing partner
        perm = torch.randperm(batch_size, device=feats.device)
        feats_b = feats[perm]
        labels_b = labels[perm]

        # Mix features in input embedding space (manifold mixup at input level)
        mixed_feats = lam * feats + (1.0 - lam) * feats_b

        return mixed_feats, labels, labels_b, lam

    def _compute_mixed_loss(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: Optional[torch.Tensor],
        lam: Optional[float],
    ) -> torch.Tensor:
        """Compute loss for mixed (or unmixed) samples."""
        if labels_b is None:
            # No mixup applied
            return self._compute_loss(logits, labels_a)

        # Mixup loss: lambda * loss_a + (1 - lambda) * loss_b
        loss_a = self._compute_loss(logits, labels_a)
        loss_b = self._compute_loss(logits, labels_b)
        return lam * loss_a + (1.0 - lam) * loss_b

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        feats = batch["features"].to(self.device).float()
        labels = batch["label"]

        # Apply Manifold Mixup augmentation
        mixed_feats, labels_a, labels_b, lam = self._manifold_mixup(feats, labels)

        logits = self.head(mixed_feats)
        loss = self._compute_mixed_loss(logits, labels_a, labels_b, lam)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        feats = batch["features"].to(self.device).float()
        logits = self.head(feats)
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        """Compute val/f1 correctly across all GPUs.

        CORRECT multi-GPU implementation: gather all predictions across ranks before
        computing F1. This computes val/f1 on all 141 validation samples (not ~18 per GPU).
        sync_dist=False since metric is already globally synchronized via all_gather.
        """
        import torch.distributed as dist

        if not self._val_preds:
            return

        preds_local = torch.cat(self._val_preds, dim=0)   # [local_N, 3, N_GENES]
        labels_local = torch.cat(self._val_labels, dim=0) # [local_N, N_GENES]

        self._val_preds.clear()
        self._val_labels.clear()

        # Gather from all ranks to get the full validation set
        all_preds = self.all_gather(preds_local)   # [world_size, local_N, 3, N_GENES]
        all_labels = self.all_gather(labels_local) # [world_size, local_N, N_GENES]

        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)  # [total_N, 3, N_GENES]
        all_labels = all_labels.view(-1, N_GENES)            # [total_N, N_GENES]

        # Compute per-gene macro F1 on all validation samples
        f1 = _compute_per_gene_f1(
            all_preds.float().cpu().numpy(),
            all_labels.cpu().numpy(),
        )
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        feats = batch["features"].to(self.device).float()
        logits = self.head(feats)
        self._test_preds.append(logits.detach().cpu().float())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)    # [local_N, 3, N_GENES]
        labels_local = (
            torch.cat(self._test_labels, dim=0) if self._test_labels else None
        )
        self._test_preds.clear()
        self._test_labels.clear()

        # Gather predictions from all ranks
        all_preds = self.all_gather(preds_local)  # [world_size, local_N, 3, N_GENES]
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)  # [total_N, 3, N_GENES]

        # Gather labels (if available)
        if labels_local is not None:
            all_labels = self.all_gather(labels_local)  # [world_size, local_N, N_GENES]
            all_labels = all_labels.view(-1, N_GENES)   # [total_N, N_GENES]

        # Gather string metadata
        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        gathered_pert_ids = [local_pert_ids]
        gathered_symbols = [local_symbols]
        if world_size > 1:
            obj_pert = [None] * world_size
            obj_sym = [None] * world_size
            dist.all_gather_object(obj_pert, local_pert_ids)
            dist.all_gather_object(obj_sym, local_symbols)
            gathered_pert_ids = obj_pert
            gathered_symbols = obj_sym

        if self.trainer.is_global_zero:
            all_pert_ids = [p for rank_list in gathered_pert_ids for p in rank_list]
            all_symbols = [s for rank_list in gathered_symbols for s in rank_list]
            all_preds_np = all_preds.float().cpu().numpy()

            # Deduplicate by pert_id (handles DDP padding)
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(all_preds_np[i])

            dedup_preds_np = np.stack(dedup_preds, axis=0)

            # Compute test F1 if labels are available
            if labels_local is not None:
                all_labels_np = all_labels.cpu().numpy()
                # Build dedup labels in same order as dedup_ids
                pid_to_label: Dict[str, np.ndarray] = {}
                all_pert_list = [p for rank_list in gathered_pert_ids for p in rank_list]
                for i, pid in enumerate(all_pert_list):
                    if pid not in pid_to_label:
                        pid_to_label[pid] = all_labels_np[i]
                dedup_labels = np.stack([pid_to_label[pid] for pid in dedup_ids], axis=0)

                test_f1 = _compute_per_gene_f1(dedup_preds_np, dedup_labels)
                self.log("test/f1", test_f1, prog_bar=True, sync_dist=False)

            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=dedup_preds_np,
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    def configure_optimizers(self):
        """Muon optimizer for hidden block weight matrices + AdamW for everything else.

        This design follows the proven recipe from node1-3-3 (F1=0.4950):
        - Muon lr=0.01 for hidden block weight matrices (safer than 0.02 with CosineWR)
        - AdamW lr=3e-4 for input_proj, output head, norms, biases
        - CosineAnnealingWarmRestarts (T_0=80, T_mult=2) for escaping local optima
        """
        from muon import MuonWithAuxAdam

        # Separate parameters by role
        # Muon: hidden weight matrices in residual blocks (fc1, fc2 weights)
        muon_params = []
        adamw_params = []

        for name, param in self.head.named_parameters():
            if not param.requires_grad:
                continue
            # Apply Muon to 2D weight matrices in residual blocks only
            # Excludes: input_proj, out_proj, per_gene_bias, LayerNorm params, head_norm
            if (param.ndim >= 2
                    and "blocks" in name
                    and "norm" not in name
                    and "bias" not in name):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        param_groups = [
            # Muon group for hidden block weight matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=self.hparams.muon_lr,
                weight_decay=self.hparams.weight_decay,
                momentum=0.95,
            ),
            # AdamW group for all other parameters
            dict(
                params=adamw_params,
                use_muon=False,
                lr=self.hparams.lr,
                betas=(0.9, 0.95),
                weight_decay=self.hparams.weight_decay,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # CosineAnnealingWarmRestarts: enables escaping local optima through LR restarts.
        # T_0=80, T_mult=2 proven by node1-3-3 (F1=0.4950) and node1-3-2-2-1-1-1-1-1-1 (F1=0.4968).
        # The warm restarts at epochs 80, 240, etc. produce successive F1 peaks.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.t0,
            T_mult=self.hparams.t_mult,
            eta_min=self.hparams.lr_eta_min,
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
        """Save only trainable parameters + buffers."""
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        result = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    result[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                result[key] = full_sd[key]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving checkpoint: {trainable}/{total} params "
            f"({100 * trainable / total:.2f}%)"
        )
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-F1 averaged over all 6,640 response genes.

    Exactly matches data/calc_metric.py logic:
    - argmax over class dim -> hard predictions
    - per-gene F1 averaged over present classes only
    - final score = mean over all genes
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)     # [N, N_GENES]
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        if present.any():
            f1_vals.append(float(per_class_f1[present].mean()))
        else:
            f1_vals.append(0.0)
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert len(pert_ids) == len(preds), (
        f"Length mismatch: {len(pert_ids)} ids vs {len(preds)} predictions"
    )
    rows = []
    for i in range(len(pert_ids)):
        rows.append({
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions -> {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STRING_GNN (frozen) + 3-block MLP (h=384) + Muon + WCE + Manifold Mixup + CosineWarmRestarts"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=500,
                   help="Extended to 500: node1-3-3 (F1=0.4950) used 500 epochs; parent was truncated at 200")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="AdamW learning rate for non-hidden-matrix params")
    p.add_argument("--muon-lr", type=float, default=0.01,
                   help="Muon LR for hidden block weight matrices; 0.01 proven by node1-3-3 (F1=0.4950)")
    p.add_argument("--weight-decay", type=float, default=8e-4,
                   help="Weight decay; 8e-4 from node1-3-3 proven recipe (was 0.01 in parent)")
    p.add_argument("--hidden-dim", type=int, default=384,
                   help="Hidden dim; 384 proven by node1-3-2 (F1=0.4756)")
    p.add_argument("--n-blocks", type=int, default=3,
                   help="Number of residual blocks; 3 matches proven configs")
    p.add_argument("--dropout", type=float, default=0.30,
                   help="Trunk dropout; reduced to 0.30 (Mixup provides stochastic regularization)")
    p.add_argument("--head-dropout", type=float, default=0.15,
                   help="Output head dropout; p=0.15 proven by node1-3-2-2-1 (+0.0021 F1)")
    p.add_argument("--t0", type=int, default=80,
                   help="CosineWarmRestarts T_0; 80 proven by node1-3-3 (F1=0.4950)")
    p.add_argument("--t-mult", type=int, default=2,
                   help="CosineWarmRestarts T_mult; 2 proven by node1-3-3 (F1=0.4950)")
    p.add_argument("--lr-eta-min", type=float, default=1e-5,
                   help="CosineWarmRestarts minimum LR")
    p.add_argument("--grad-clip-val", type=float, default=1.0,
                   help="Gradient clipping max norm; 1.0 stabilizes Muon training")
    p.add_argument("--mixup-alpha", type=float, default=0.2,
                   help="Manifold Mixup Beta distribution alpha; 0.2 proven by node1-3-3")
    p.add_argument("--mixup-prob", type=float, default=0.5,
                   help="Manifold Mixup probability; 0.5 proven by node1-3-3 (F1=0.4950)")
    p.add_argument("--early-stop-patience", type=int, default=80,
                   help="Early stopping patience; 80 to allow warm restart cycles")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None,
                   help="Limit steps for quick debugging")
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    model = PerturbModule(
        in_dim=FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        n_genes=N_GENES,
        n_blocks=args.n_blocks,
        lr=args.lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        t0=args.t0,
        t_mult=args.t_mult,
        lr_eta_min=args.lr_eta_min,
        grad_clip_val=args.grad_clip_val,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
    )

    fast_dev_run = args.fast_dev_run
    debug_max_step = args.debug_max_step
    if debug_max_step is not None:
        limit_train = limit_val = limit_test = debug_max_step
        max_steps = debug_max_step
        val_check_interval = 1.0
        num_sanity_val_steps = 0
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval if not fast_dev_run else 1.0
        num_sanity_val_steps = 2

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
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(
            find_unused_parameters=True, timeout=timedelta(seconds=120)
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=args.grad_clip_val,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, datamodule=datamodule)

    # Use best checkpoint for final test evaluation
    if fast_dev_run or debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        result = test_results[0]
        primary_metric = result.get("test/f1", result.get("test/f1_score", float("nan")))
        score_path.write_text(str(float(primary_metric)))
        print(f"Test results -> {score_path} (f1_score={primary_metric})", flush=True)


if __name__ == "__main__":
    main()
