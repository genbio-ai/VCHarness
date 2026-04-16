"""Node 1-2-2-2: STRING_GNN Frozen Embeddings + 3-Block Pre-Norm MLP (h=384)
               + Muon Optimizer + FOCAL LOSS (gamma=2.0) + CosineAnnealingWarmRestarts (T_0=40)
               + Manifold Mixup + Selective Regularization (wd_mlp=0.01, wd_bias=0.05)
               + Per-Gene Bias (wd=0.05) + Top-3 Checkpoint Ensemble
================================================================
Parent  : node1-2-2 (test F1=0.4433)
          STRING_GNN + 3-block h=384 + Muon + FocalLoss(gamma=2.0) + CosineWR(T_0=40)
          + Manifold Mixup (prob=0.5) + head_dropout=0.15 + wd=0.06 + class_bias (3 params)
          [Over-regularized: all 8 simultaneous changes, val_f1 plateau at 0.44]

Sibling : node1-2-2-1 (test F1=0.4582)
          + Restored per-gene bias (19920 params, wd=0.10)
          + Reduced MLP wd from 0.06 to 0.03
          + Reduced Mixup prob from 0.5 to 0.25
          [Successful recovery +0.0149 F1; healthy 2.56x train/val ratio]

Key changes in this node vs sibling (node1-2-2-1):
-------------------------------------------------
1. MLP WEIGHT DECAY: 0.03 -> 0.01  [REGULARIZATION LOOSENING - PRIMARY CHANGE]
   Sibling feedback (Direction 1, High Priority): reduce MLP wd from 0.03 toward 0.01-0.02.
   Reference nodes achieving F1=0.495-0.497 (node1-3-2-2-1-1-1-1-1-1-1) used wd~8e-4.
   The sibling's 2.56x train/val ratio indicates room to allow more fitting.
   wd=0.01 still provides meaningful L2 regularization while allowing sharper discriminative features.
   The Muon optimizer's self-normalizing updates already provide implicit regularization.

2. GENE BIAS WEIGHT DECAY: 0.10 -> 0.05  [BIOLOGICAL CALIBRATION IMPROVEMENT]
   Sibling feedback (Direction 3, Low Risk): reduce gene_bias wd from 0.10 to 0.05.
   At wd=0.10, the per-gene bias is so heavily penalized that it may not capture even
   genuine population-level gene response tendencies. At wd=0.05, it is still 5x stronger
   than MLP wd=0.01, maintaining the "selective regularization" principle (bias penalized more
   than MLP weights) while allowing stronger biological calibration signals.
   Expected impact: +0.5-2pp F1 from stronger per-gene calibration.

3. TOP-3 CHECKPOINT ENSEMBLE  [VARIANCE REDUCTION]
   Multiple reference nodes demonstrate ensemble benefit (top-5: +0.0066 F1 in node3-3-1-2-1-1-1).
   With 141 validation samples, val_f1 has ±0.001-0.003 noise; averaging top-3 checkpoints
   reduces prediction variance. Low risk, potential +0.001-0.005 F1 at minimal cost.

4. PATIENCE: 20 -> 25  [SLIGHTLY EXTENDED TRAINING]
   With lower wd (0.01), the model may converge more slowly.
   Extending patience from 20 to 25 gives 5 more epochs to find the optimal checkpoint
   in the CosineWR second cycle (epochs 40-120). Conservative change.

Preserved from sibling (node1-2-2-1):
-----------------------------------------------------------
- STRING_GNN frozen PPI graph embeddings (256-dim) -- ESM2 NOT used (all zeros)
- Per-gene additive bias [N_GENES, N_CLASSES] (19,920 params) -- proven biological signal
- Pre-norm residual block structure (3 blocks, h=384, inner=768) -- proven optimal
- Trunk dropout=0.30 -- proven optimal
- Head dropout=0.15 -- proven optimal from tree history
- Manifold Mixup (prob=0.25, alpha=0.2) -- proven balanced
- Muon+AdamW dual optimizer (Muon LR=0.01, AdamW LR=3e-4) -- proven, better than AdamW-only
- Three-group optimizer (Muon, AdamW, gene_bias) -- selective regularization framework
- Focal loss (gamma=2.0) + class weights [0.0477, 0.9282, 0.0241]
- CosineAnnealingWarmRestarts (T_0=40, T_mult=2, eta_min=1e-7)
- Gradient clip=2.0
- Max epochs=300
- Seed=0 for reproducibility
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
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES = 6640        # number of response genes per perturbation
N_CLASSES = 3         # down (-1->0), neutral (0->1), up (1->2)
GNN_DIM = 256         # STRING_GNN output embedding dimension
HIDDEN_DIM = 384      # MLP hidden dimension -- proven optimal for STRING-only
INNER_DIM = 768       # MLP inner (expansion) dimension (2x hidden per PreLN block)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Each sample is one gene perturbation experiment in HepG2 cells."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        if "label" in df.columns:
            # Labels in {-1,0,1} -> shift to {0,1,2}
            labels = np.array(
                [json.loads(x) for x in df["label"].tolist()], dtype=np.int64
            )
            self.labels: Optional[torch.Tensor] = torch.tensor(
                labels + 1, dtype=torch.long
            )
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
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

        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df)
        self.val_ds = PerturbDataset(val_df)
        self.test_ds = PerturbDataset(test_df)

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
# Model Components
# ---------------------------------------------------------------------------
class PreNormResBlock(nn.Module):
    """Pre-LayerNorm residual block (proven stable in node1-3-2 lineage).

    Architecture:
        output = x + LN(x) -> Linear(dim->inner) -> GELU -> Dropout
                               -> Linear(inner->dim) -> Dropout
    """

    def __init__(self, dim: int, inner_dim: int, dropout: float = 0.30) -> None:
        super().__init__()
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


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        inner_dim: int = INNER_DIM,
        n_blocks: int = 3,
        dropout: float = 0.30,            # Trunk dropout (proven optimal)
        head_dropout: float = 0.15,       # Proven optimal (node1-3-2-2-1 ceiling)
        muon_lr: float = 0.01,            # Proven stable Muon LR
        adamw_lr: float = 3e-4,           # AdamW LR for non-block params
        mlp_weight_decay: float = 0.01,   # REDUCED: 0.03->0.01 for MLP weights
        bias_weight_decay: float = 0.05,  # REDUCED: 0.10->0.05 for gene_bias
        label_smoothing: float = 0.0,     # Essential for focal loss
        focal_gamma: float = 2.0,         # Proven optimal
        # CosineAnnealingWarmRestarts params
        cosine_t0: int = 40,              # Proven: more frequent warm restarts
        cosine_t_mult: int = 2,           # Cycle length multiplier
        cosine_eta_min: float = 1e-7,     # Minimum LR after annealing
        grad_clip_norm: float = 2.0,      # Proven for Muon orthogonalized updates
        # Manifold Mixup params
        mixup_alpha: float = 0.2,         # Beta distribution alpha for mixing coefficient
        mixup_prob: float = 0.25,         # Proven optimal (not 0.5 which over-regularizes)
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Populated in setup()
        self.input_proj: Optional[nn.Sequential] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None  # Per-gene calibration [N_GENES, N_CLASSES]

        # STRING_GNN gene-ID -> embedding-row index
        self.gnn_id_to_idx: Dict[str, int] = {}

        # Metric accumulators
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """Build model and precompute frozen STRING_GNN node embeddings."""
        from transformers import AutoModel

        self.print("Loading STRING_GNN and computing frozen node embeddings ...")
        gnn_model = AutoModel.from_pretrained(
            STRING_GNN_DIR, trust_remote_code=True
        )
        gnn_model.eval()
        gnn_model = gnn_model.to(self.device)

        graph = torch.load(
            Path(STRING_GNN_DIR) / "graph_data.pt",
            map_location=self.device,
        )
        edge_index = graph["edge_index"].to(self.device)
        edge_weight = graph.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        with torch.no_grad():
            gnn_out = gnn_model(edge_index=edge_index, edge_weight=edge_weight)

        # Register as a non-trainable float32 buffer [18870, 256]
        all_emb = gnn_out.last_hidden_state.detach().float()
        self.register_buffer("gnn_embeddings", all_emb)

        # Free GNN model memory
        del gnn_model, gnn_out, graph, edge_index, edge_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.print(f"STRING_GNN embeddings shape: {all_emb.shape}")

        # Build ENSG-ID -> row-index mapping
        node_names: List[str] = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.gnn_id_to_idx = {name: i for i, name in enumerate(node_names)}
        n_covered = len(self.gnn_id_to_idx)
        self.print(f"STRING_GNN covers {n_covered} Ensembl gene IDs")

        # ---- MLP architecture ----
        hp = self.hparams
        self.input_proj = nn.Sequential(
            nn.LayerNorm(GNN_DIM),
            nn.Linear(GNN_DIM, hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )
        self.blocks = nn.ModuleList(
            [
                PreNormResBlock(hp.hidden_dim, hp.inner_dim, hp.dropout)
                for _ in range(hp.n_blocks)
            ]
        )
        # Output head: head_dropout=0.15 (proven optimal)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )

        # Per-gene additive bias: [N_GENES, N_CLASSES] (19,920 parameters)
        # RESTORED from sibling: encodes population-level gene response tendencies
        # Governed by dedicated weight decay (bias_weight_decay=0.05, REDUCED from 0.10)
        # This is 5x stronger than MLP wd=0.01, maintaining selective regularization
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # ---- Class weights (CORRECT ordering after +1 label shift) ----
        # class 0 = down-regulated  (4.77%)  -> high weight
        # class 1 = neutral         (92.82%) -> low weight
        # class 2 = up-regulated    (2.41%)  -> highest weight
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq) / (1.0 / freq).mean()
        self.register_buffer("class_weights", class_weights)

        # Cast trainable params to float32 for stable optimization
        for k, v in self.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Architecture: STRING_GNN({GNN_DIM}) -> Proj -> "
            f"{hp.n_blocks}xPreNormResBlock({hp.hidden_dim},{hp.inner_dim}) "
            f"-> HeadDropout({hp.head_dropout}) -> Linear({hp.hidden_dim},{N_GENES}x{N_CLASSES}) + gene_bias[{N_GENES},{N_CLASSES}]"
        )
        self.print(f"Trainable params: {trainable:,} / {total:,}")
        self.print(f"FOCAL LOSS: gamma={hp.focal_gamma}")
        self.print(
            f"LR SCHEDULE: CosineAnnealingWarmRestarts "
            f"T_0={hp.cosine_t0}, T_mult={hp.cosine_t_mult}, eta_min={hp.cosine_eta_min}"
        )
        self.print(f"MLP WEIGHT DECAY: {hp.mlp_weight_decay} (REDUCED from sibling 0.03 to 0.01)")
        self.print(f"GENE BIAS WEIGHT DECAY: {hp.bias_weight_decay} (REDUCED from sibling 0.10 to 0.05)")
        self.print(f"HEAD DROPOUT: {hp.head_dropout}")
        self.print(f"MANIFOLD MIXUP: alpha={hp.mixup_alpha}, prob={hp.mixup_prob}")
        self.print(f"PER-GENE BIAS: [N_GENES={N_GENES}, N_CLASSES={N_CLASSES}] (19,920 params with dedicated wd=0.05)")

    # ------------------------------------------------------------------
    def _get_gene_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Batch lookup of frozen STRING_GNN embeddings for ENSG IDs.

        Genes absent from STRING_GNN (~7% of samples) receive a zero vector.
        """
        emb_list: List[torch.Tensor] = []
        for pid in pert_ids:
            row = self.gnn_id_to_idx.get(pid)
            if row is not None:
                emb_list.append(self.gnn_embeddings[row])
            else:
                emb_list.append(
                    torch.zeros(GNN_DIM, device=self.device, dtype=torch.float32)
                )
        return torch.stack(emb_list, dim=0)  # [B, 256]

    def _forward_from_emb(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from STRING embedding [B, 256] to logits [B, 3, N_GENES]."""
        x = self.input_proj(x)     # [B, 384]
        for block in self.blocks:
            x = block(x)           # [B, 384]
        logits = self.output_head(x)                    # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)    # [B, 3, 6640]
        # Per-gene bias: gene_bias [N_GENES, N_CLASSES] -> transpose -> [N_CLASSES, N_GENES]
        # -> unsqueeze(0) -> [1, N_CLASSES, N_GENES] broadcast over batch
        logits = logits + self.gene_bias.t().unsqueeze(0)
        return logits

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits of shape [B, N_CLASSES, N_GENES]."""
        x = self._get_gene_emb(pert_ids)  # [B, 256]
        return self._forward_from_emb(x)

    def _manifold_mixup(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Apply Manifold Mixup in the hidden space (after input projection).

        Interpolates pairs of hidden representations and their labels.
        Returns (mixed_x, labels_a, labels_b, lam) for mixed loss calculation.

        x:      [B, hidden_dim] -- hidden representation (after input_proj)
        labels: [B, N_GENES]   -- integer class labels in {0, 1, 2}
        """
        hp = self.hparams
        if self.training and np.random.random() < hp.mixup_prob:
            lam = np.random.beta(hp.mixup_alpha, hp.mixup_alpha)
            # Clamp lambda to a safe range to avoid degenerate mixes
            lam = max(lam, 1 - lam)  # Always take the larger weight
            batch_size = x.size(0)
            index = torch.randperm(batch_size, device=x.device)
            mixed_x = lam * x + (1 - lam) * x[index]
            return mixed_x, labels, labels[index], lam
        else:
            return x, labels, labels, 1.0

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        labels_b: Optional[torch.Tensor] = None,
        lam: float = 1.0,
    ) -> torch.Tensor:
        """Focal loss on [B, N_CLASSES, N_GENES] logits.

        Supports Manifold Mixup via optional labels_b and lam:
        loss = lam * focal(logits, labels_a) + (1-lam) * focal(logits, labels_b)
        """
        hp = self.hparams

        def _focal_loss_single(lgts, lbls):
            logits_flat = lgts.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
            labels_flat = lbls.reshape(-1)

            if hp.focal_gamma == 0.0:
                return F.cross_entropy(
                    logits_flat,
                    labels_flat,
                    weight=self.class_weights,
                    label_smoothing=hp.label_smoothing,
                )

            # Focal loss: FL(p_t) = -(1 - p_t)^gamma * w_c * log(p_t)
            ce_per_sample = F.cross_entropy(
                logits_flat,
                labels_flat,
                weight=self.class_weights,
                reduction="none",
                label_smoothing=hp.label_smoothing,
            )
            with torch.no_grad():
                probs = F.softmax(logits_flat, dim=-1)
                pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - pt.clamp(min=1e-8)) ** hp.focal_gamma
            return (focal_weight * ce_per_sample).mean()

        loss_a = _focal_loss_single(logits, labels)

        if labels_b is not None and lam < 1.0:
            loss_b = _focal_loss_single(logits, labels_b)
            return lam * loss_a + (1.0 - lam) * loss_b

        return loss_a

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pert_ids = batch["pert_id"]
        labels = batch["label"]

        # Get STRING_GNN embedding
        x = self._get_gene_emb(pert_ids)  # [B, 256]

        # Apply input projection to get hidden representation
        x = self.input_proj(x)  # [B, hidden_dim]

        # Apply Manifold Mixup in the hidden (MLP input) space
        x, labels_a, labels_b, lam = self._manifold_mixup(x, labels)

        # Continue forward pass from hidden representation
        for block in self.blocks:
            x = block(x)                                # [B, hidden_dim]
        logits = self.output_head(x)                    # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)    # [B, 3, 6640]
        # Per-gene bias: [N_GENES, N_CLASSES] -> [N_CLASSES, N_GENES] -> [1, N_CLASSES, N_GENES]
        logits = logits + self.gene_bias.t().unsqueeze(0)

        # Compute mixup loss
        loss = self._compute_loss(logits, labels_a, labels_b, lam)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        preds_local = torch.cat(self._val_preds, dim=0)    # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0)  # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather predictions and labels from all ranks before computing F1.
        # Each rank only sees a subset of the validation data in DDP, so we must
        # gather ALL predictions globally to compute the correct per-gene F1.
        is_dist = dist.is_available() and dist.is_initialized()
        if is_dist:
            world_size = dist.get_world_size()
            # Gather prediction tensors across all ranks
            gathered_preds = self.all_gather(preds_local)   # [world_size, N_local, 3, 6640]
            gathered_labels = self.all_gather(labels_local) # [world_size, N_local, 6640]
            all_preds = gathered_preds.view(-1, N_CLASSES, N_GENES)   # [N_total, 3, 6640]
            all_labels = gathered_labels.view(-1, N_GENES)            # [N_total, 6640]
            # Compute F1 on the full validation set. all_gather gives identical
            # data to all ranks, so computed F1 is identical on every rank.
            f1 = _compute_per_gene_f1(
                all_preds.cpu().numpy(), all_labels.cpu().numpy()
            )
            # sync_dist=True averages identical values across ranks -- result unchanged.
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        else:
            f1 = _compute_per_gene_f1(preds_local.numpy(), labels_local.numpy())
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        self._test_preds.append(logits.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        # self.all_gather is Lightning's DDP-safe wrapper -- called on all ranks.
        gathered = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640]
        all_preds = gathered.view(-1, N_CLASSES, N_GENES)  # [N_total, 3, 6640]

        # Gather string metadata via dist.all_gather_object (torch collective).
        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        gathered_pert_ids_list: List[List[str]] = [local_pert_ids]
        gathered_symbols_list: List[List[str]] = [local_symbols]
        if is_dist:
            obj_pids = [None] * world_size
            obj_syms = [None] * world_size
            dist.all_gather_object(obj_pids, local_pert_ids)
            dist.all_gather_object(obj_syms, local_symbols)
            gathered_pert_ids_list = obj_pids
            gathered_symbols_list = obj_syms

        if self.trainer.is_global_zero:
            all_pert_ids = [pid for lst in gathered_pert_ids_list for pid in lst]
            all_symbols = [sym for lst in gathered_symbols_list for sym in lst]

            # De-duplicate (DDP may replicate samples across ranks)
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            preds_np = all_preds.cpu().numpy()  # [N_total, 3, 6640]
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(preds_np[i])

            # Store for saving in main()
            self._current_test_preds = np.stack(dedup_preds, axis=0)
            self._current_test_ids = dedup_ids
            self._current_test_syms = dedup_syms

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        from muon import MuonWithAuxAdam

        hp = self.hparams

        # THREE parameter groups with selective regularization:
        # 1. Muon: 2D weight matrices in the hidden residual blocks (ndim>=2)
        #    - wd=mlp_weight_decay (0.01): MLP weights can develop sharper discriminative features
        # 2. AdamW: all other params except gene_bias (norms, biases, input_proj, output_head)
        #    - wd=mlp_weight_decay (0.01): same moderate regularization as MLP
        # 3. AdamW (gene_bias): per-gene calibration bias specifically
        #    - wd=bias_weight_decay (0.05): 5x stronger L2 to prevent pure memorization
        #    - Still allows legitimate biological tendencies (wd=0.05 vs 0.01 = 5x penalty ratio)

        muon_params = [
            p for name, p in self.blocks.named_parameters()
            if p.ndim >= 2 and p.requires_grad
        ]
        muon_param_ids = set(id(p) for p in muon_params)

        # Separate gene_bias from all other AdamW params
        gene_bias_params = [self.gene_bias]
        gene_bias_ids = set(id(p) for p in gene_bias_params)

        # All other trainable params go to AdamW (not Muon, not gene_bias)
        adamw_params = [
            p for p in self.parameters()
            if p.requires_grad
            and id(p) not in muon_param_ids
            and id(p) not in gene_bias_ids
        ]

        param_groups = [
            # Muon group for hidden block weight matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.mlp_weight_decay,  # 0.01: looser L2 for MLP weights
                momentum=0.95,
            ),
            # AdamW group for projections, norms, head (excl. gene_bias)
            dict(
                params=adamw_params,
                use_muon=False,
                lr=hp.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.mlp_weight_decay,  # 0.01: looser L2 for non-block params
            ),
            # AdamW group ONLY for gene_bias with dedicated high L2 penalty
            dict(
                params=gene_bias_params,
                use_muon=False,
                lr=hp.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.bias_weight_decay,  # 0.05: stronger L2 to prevent memorization
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # CosineAnnealingWarmRestarts: T_0=40 for more frequent restarts
        # Restarts at epochs 40, 120, 280 (T_mult=2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.cosine_t0,
            T_mult=hp.cosine_t_mult,
            eta_min=hp.cosine_eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers: save only trainable params + buffers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_state_dict:
                    trainable_state_dict[key] = full_state_dict[key]

        for name, buffer in self.named_buffers():
            key = prefix + name
            if key in full_state_dict:
                trainable_state_dict[key] = full_state_dict[key]

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )

        return trainable_state_dict

    def load_state_dict(self, state_dict, strict=True):
        full_state_keys = set(super().state_dict().keys())
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        buffer_keys = {
            name for name, _ in self.named_buffers() if name in full_state_keys
        }
        expected_keys = trainable_keys | buffer_keys

        missing_keys = [k for k in expected_keys if k not in state_dict]
        unexpected_keys = [k for k in state_dict if k not in expected_keys]

        if missing_keys:
            self.print(f"Warning: Missing checkpoint keys: {missing_keys[:5]}...")
        if unexpected_keys:
            self.print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

        loaded_trainable = len([k for k in state_dict if k in trainable_keys])
        loaded_buffers = len([k for k in state_dict if k in buffer_keys])
        self.print(
            f"Loading checkpoint: {loaded_trainable} trainable parameters and "
            f"{loaded_buffers} buffers"
        )

        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-averaged F1 exactly matching calc_metric.py.

    preds  : [N_samples, 3, N_genes]  -- logits / class scores
    labels : [N_samples, N_genes]     -- integer class labels in {0, 1, 2}
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N_samples, N_genes]
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(
            yt, yh, labels=[0, 1, 2], average=None, zero_division=0
        )
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        f1_vals.append(float(per_class_f1[present].mean()))
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    """Save test predictions in the TSV format required by calc_metric.py."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert len(pert_ids) == len(preds), (
        f"Length mismatch: {len(pert_ids)} pert_ids vs {len(preds)} pred rows"
    )
    rows = [
        {
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),  # [3, 6640] as JSON
        }
        for i in range(len(pert_ids))
    ]
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions -> {out_path}")


def _ensemble_test_predictions(
    checkpoint_dir: Path,
    datamodule: PerturbDataModule,
    trainer: pl.Trainer,
    model: PerturbModule,
    save_top_k: int = 3,
) -> Optional[np.ndarray]:
    """Load top-K checkpoints by val_f1 and average their predictions.

    Returns averaged predictions array of shape [N_test, 3, N_GENES],
    or None if fewer than 2 valid checkpoints are found.
    """
    import glob
    import re

    # Find all checkpoint files with val_f1 in name
    ckpt_pattern = str(checkpoint_dir / "best-*.ckpt")
    ckpt_files = glob.glob(ckpt_pattern)

    if not ckpt_files:
        print("No checkpoint files found for ensemble.")
        return None

    # Parse val_f1 from filename: best-{epoch:03d}-{val_f1:.4f}.ckpt
    def parse_f1(path: str) -> float:
        basename = Path(path).stem  # e.g., "best-094-0.4587"
        parts = basename.split("-")
        try:
            return float(parts[-1])
        except (ValueError, IndexError):
            return 0.0

    # Sort by val_f1 descending and take top-K
    ckpt_files_sorted = sorted(ckpt_files, key=parse_f1, reverse=True)
    top_k_ckpts = ckpt_files_sorted[:save_top_k]

    if len(top_k_ckpts) < 2:
        print(f"Only {len(top_k_ckpts)} checkpoint(s) found, skipping ensemble.")
        return None

    print(f"\n=== TOP-{len(top_k_ckpts)} CHECKPOINT ENSEMBLE ===")
    for i, ckpt_path in enumerate(top_k_ckpts):
        print(f"  Checkpoint {i+1}: {Path(ckpt_path).name} (val_f1={parse_f1(ckpt_path):.4f})")

    # Collect predictions from each checkpoint
    all_preds_list = []
    for ckpt_path in top_k_ckpts:
        try:
            trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
            if trainer.is_global_zero and hasattr(model, '_current_test_preds'):
                all_preds_list.append(model._current_test_preds.copy())
                # Save order/ids from first checkpoint only
                if len(all_preds_list) == 1:
                    ensemble_ids = model._current_test_ids[:]
                    ensemble_syms = model._current_test_syms[:]
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {ckpt_path}: {e}")
            continue

    if len(all_preds_list) < 2:
        print("Insufficient valid checkpoints for ensemble.")
        return None

    if trainer.is_global_zero:
        # Average predictions across checkpoints
        ensemble_preds = np.mean(all_preds_list, axis=0)  # [N_test, 3, N_GENES]
        print(f"Ensemble averaged {len(all_preds_list)} checkpoints.")
        return ensemble_preds, ensemble_ids, ensemble_syms

    return None, None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Node1-2-2-2: STRING_GNN + 3-Block MLP (h=384) + Muon(lr=0.01) + "
            "FocalLoss(gamma=2.0) + CosineWR(T_0=40) + ManifoldMixup(prob=0.25) + "
            "Selective Regularization (wd_mlp=0.01, wd_bias=0.05) + Top-3 Ensemble"
        )
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--mlp-weight-decay", type=float, default=0.01,
                   help="Weight decay for MLP (Muon + AdamW non-bias params): REDUCED 0.03->0.01")
    p.add_argument("--bias-weight-decay", type=float, default=0.05,
                   help="Weight decay for per-gene bias: REDUCED 0.10->0.05")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--head-dropout", type=float, default=0.15)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--inner-dim", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--cosine-t0", type=int, default=40)
    p.add_argument("--cosine-t-mult", type=int, default=2)
    p.add_argument("--cosine-eta-min", type=float, default=1e-7)
    p.add_argument("--grad-clip-norm", type=float, default=2.0)
    p.add_argument("--early-stop-patience", type=int, default=25,
                   help="Slightly extended from sibling 20 to 25 for lower-wd convergence")
    p.add_argument("--save-top-k", type=int, default=3,
                   help="Number of top-K checkpoints to save and ensemble during test")
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.25,
                   help="Manifold Mixup probability: proven balanced (vs 0.5 which over-regularizes)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--no-ensemble", action="store_true",
                   help="Disable top-K ensemble (use single best checkpoint)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DataModule
    # ------------------------------------------------------------------
    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = PerturbModule(
        hidden_dim=args.hidden_dim,
        inner_dim=args.inner_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        mlp_weight_decay=args.mlp_weight_decay,
        bias_weight_decay=args.bias_weight_decay,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_t_mult,
        cosine_eta_min=args.cosine_eta_min,
        grad_clip_norm=args.grad_clip_norm,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
    )

    # ------------------------------------------------------------------
    # Trainer configuration
    # ------------------------------------------------------------------
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = limit_val = limit_test = args.debug_max_step
        max_steps = args.debug_max_step
        val_check_interval = 1.0
        num_sanity_val_steps = 0
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval
        num_sanity_val_steps = 2

    checkpoint_dir = output_dir / "checkpoints"

    # Save top-K checkpoints for ensemble; use best single checkpoint for test if ensemble disabled.
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    # patience=25: slightly extended from sibling's 20 to allow lower-wd convergence
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stop_patience,
        min_delta=1e-5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    callbacks = [checkpoint_cb, early_stop_cb, lr_monitor, progress_bar]

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            timeout=timedelta(seconds=120),
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval if (
            args.debug_max_step is None and not fast_dev_run
        ) else 1.0,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        gradient_clip_val=args.grad_clip_norm,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(model, datamodule=datamodule)

    # ------------------------------------------------------------------
    # Test: Use top-K ensemble OR best single checkpoint
    # ------------------------------------------------------------------
    out_path = output_dir / "test_predictions.tsv"

    if args.fast_dev_run or args.debug_max_step is not None:
        print("\n=== DEBUG MODE: Single checkpoint test ===")
        trainer.test(model, datamodule=datamodule)
        if trainer.is_global_zero and hasattr(model, '_current_test_preds'):
            _save_test_predictions(
                pert_ids=model._current_test_ids,
                symbols=model._current_test_syms,
                preds=model._current_test_preds,
                out_path=out_path,
            )

    elif args.no_ensemble:
        print("\n=== PRODUCTION MODE: Single best checkpoint test ===")
        trainer.test(model, datamodule=datamodule, ckpt_path='best')
        if trainer.is_global_zero and hasattr(model, '_current_test_preds'):
            _save_test_predictions(
                pert_ids=model._current_test_ids,
                symbols=model._current_test_syms,
                preds=model._current_test_preds,
                out_path=out_path,
            )

    else:
        print(f"\n=== PRODUCTION MODE: Top-{args.save_top_k} Checkpoint Ensemble ===")

        # First run with best checkpoint (as fallback)
        print("Step 1: Running best single checkpoint for predictions (fallback)")
        trainer.test(model, datamodule=datamodule, ckpt_path='best')

        best_single_preds = None
        best_single_ids = None
        best_single_syms = None
        if trainer.is_global_zero and hasattr(model, '_current_test_preds'):
            best_single_preds = model._current_test_preds.copy()
            best_single_ids = model._current_test_ids[:]
            best_single_syms = model._current_test_syms[:]

        # Try ensemble
        print(f"\nStep 2: Running Top-{args.save_top_k} ensemble")
        try:
            result = _ensemble_test_predictions(
                checkpoint_dir=checkpoint_dir,
                datamodule=datamodule,
                trainer=trainer,
                model=model,
                save_top_k=args.save_top_k,
            )

            if trainer.is_global_zero:
                if result is not None and result[0] is not None:
                    ensemble_preds, ensemble_ids, ensemble_syms = result
                    _save_test_predictions(
                        pert_ids=ensemble_ids,
                        symbols=ensemble_syms,
                        preds=ensemble_preds,
                        out_path=out_path,
                    )
                    print(f"Ensemble predictions saved -> {out_path}")
                else:
                    # Fall back to single best checkpoint
                    print("Falling back to single best checkpoint predictions.")
                    if best_single_preds is not None:
                        _save_test_predictions(
                            pert_ids=best_single_ids,
                            symbols=best_single_syms,
                            preds=best_single_preds,
                            out_path=out_path,
                        )
        except Exception as e:
            print(f"Warning: Ensemble failed with error: {e}")
            print("Falling back to single best checkpoint predictions.")
            if trainer.is_global_zero and best_single_preds is not None:
                _save_test_predictions(
                    pert_ids=best_single_ids,
                    symbols=best_single_syms,
                    preds=best_single_preds,
                    out_path=out_path,
                )

    if trainer.is_global_zero:
        print(f"\nTest predictions saved -> {out_path}")


if __name__ == "__main__":
    main()
