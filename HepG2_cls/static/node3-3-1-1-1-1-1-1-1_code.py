"""Node 3-3-1-1-1-1-1-1-1: STRING_GNN Frozen Embeddings + 3-Block Pre-Norm MLP (h=384)
               + Muon Optimizer + FOCAL LOSS (gamma=2.5) — NO MIXUP
================================================================
Parent  : node3-3-1-1-1-1-1-1  (STRING_GNN+3-block+h=384+Muon+FocalLoss(γ=2.5)+Mixup(α=0.2), test F1=0.4498)
Grandparent: node3-3-1-1-1-1-1  (STRING_GNN+3-block+h=384+Muon+FocalLoss(γ=2.0), test F1=0.4851)

Core Motivation: The parent's Mixup+γ=2.5 combination caused a significant regression from
the grandparent's 0.4851 → 0.4498. The feedback explicitly recommends testing γ=2.5 WITHOUT
Mixup to isolate whether higher gamma alone (without Mixup interference) can match or exceed
the grandparent's test F1=0.4851.

Changes in this node vs parent (node3-3-1-1-1-1-1-1)
-------------------------------------------------
1. REMOVE Mixup augmentation (α=0.0)  [PRIMARY FIX]
   The parent's γ=2.5 + Mixup(α=0.2) + weight_decay=0.015 combination caused:
   - Test F1 REGRESSION: 0.4498 vs grandparent's 0.4851 (-0.0353)
   - Early training termination at epoch 95 (vs grandparent's epoch 188)
   - Disrupted focal loss hard-example mining (Mixup's soft interpolation conflicts
     with focal loss's emphasis on hard minority-class examples)
   The feedback's primary recommendation was to isolate the γ=2.5 effect by removing Mixup.
   With Mixup removed, focal loss can fully concentrate on hard up/down-regulated gene
   predictions without interference from averaged label signals.

2. REVERT weight_decay: 0.015 → 0.01  [COMPLEMENTARY FIX]
   The parent used weight_decay=0.015 as "complementary to Mixup" — but Mixup is now removed.
   Weight decay of 0.015 without Mixup's regularization contribution may over-constrain the
   model, preventing convergence to the best solution that the grandparent (wd=0.01) found.
   Reverting to the grandparent's proven weight_decay=0.01 restores the optimization landscape.

3. INCREASE early_stop_patience: 35 → 50  [KEY ENABLER]
   The grandparent's best performance came at epoch 188 — well into low-LR training phases.
   The parent's training terminated prematurely at epoch 95 due to Mixup destabilization.
   With Mixup removed, the model should follow the grandparent's trajectory of steady
   improvement through multiple LR halvings. Increasing patience from 35 to 50 allows:
   - More time in low-LR refinement phases (grandparent needed epochs 160–223)
   - Recovery from post-RLROP-halving temporary dips
   - Sufficient exploration before declaring a plateau
   This is critical: the grandparent would have been cut off at ~epoch 223 with patience=35,
   but continued to improve. With patience=50 and the same RLROP dynamics, we give the model
   more room to refine at the lower learning rates.

4. KEEP focal gamma=2.5  [UNCHANGED FROM PARENT]
   The key hypothesis: γ=2.5 alone (without Mixup interference) should provide better
   minority-class focus than γ=2.0, potentially achieving test F1 > grandparent's 0.4851.
   Rationale:
   - γ=2.5 more aggressively suppresses easy neutral gene gradients (p_t≈0.95):
     (0.05)^2.5 ≈ 0.00056 vs (0.05)^2.0 ≈ 0.0025 (4.5× stronger suppression)
   - γ=2.5 maintains strong gradients for hard minority-class genes (p_t≈0.2):
     (0.8)^2.5 ≈ 0.573 vs (0.8)^2.0 ≈ 0.64 (10% reduction, acceptable trade-off)
   - The grandparent achieved excellent val-test alignment (gap=0.0003) with γ=2.0;
     γ=2.5 should preserve this property while improving minority-class precision.

Preserved confirmed working fixes from grandparent (node3-3-1-1-1-1-1)
-----------------------------------------------------------
- Focal loss (γ=2.5, same as parent — this is the isolated test)
- RLROP threshold=1e-5 (prevents over-aggressive halvings)
- RLROP patience=12 (grandparent value)
- head_dropout=0.05 (proven optimal)
- max_epochs=500 (extended training budget)
- val_f1 metric name (no slash) prevents Lightning subdirectory creation
- auto_insert_metric_name=False prevents malformed checkpoint filenames
- Muon LR=0.01 (proven stable)
- AdamW LR=3e-4

Architecture (UNCHANGED from parent/grandparent)
--------------------------------------------------------------------
- STRING_GNN frozen PPI graph embeddings (256-dim)
- Per-gene additive bias (19,920 learnable parameters)
- Pre-norm residual block structure (3 blocks, h=384, inner=768)
- Trunk dropout=0.30
- Label smoothing=0.0
- Gradient clip=2.0
- RLROP factor=0.5
- Correct class-weight order: [0.0477, 0.9282, 0.0241] (down/neutral/up)
- Single best checkpoint inference (ensemble was zero-benefit in ancestor)
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import re
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
N_CLASSES = 3         # down (-1→0), neutral (0→1), up (1→2)
GNN_DIM = 256         # STRING_GNN output embedding dimension
HIDDEN_DIM = 384      # MLP hidden dimension — proven optimal (node1-3-2: F1=0.4756)
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
            # Labels in {-1,0,1} → shift to {0,1,2}
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
        output = x + LN(x) → Linear(dim→inner) → GELU → Dropout
                               → Linear(inner→dim) → Dropout
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
        dropout: float = 0.30,          # Trunk dropout
        head_dropout: float = 0.05,     # Proven optimal (0.10 was ineffective)
        muon_lr: float = 0.01,          # Proven stable Muon LR (from node3-3-1-1)
        adamw_lr: float = 3e-4,         # AdamW LR for non-block params
        weight_decay: float = 0.01,     # REVERTED from parent's 0.015 (Mixup now removed)
        label_smoothing: float = 0.0,   # Removed: eliminates training loss floor
        focal_gamma: float = 2.5,       # KEPT from parent: higher minority-class focus
        rlrop_factor: float = 0.5,
        rlrop_patience: int = 12,       # Proven value (grandparent)
        rlrop_threshold: float = 1e-5,  # Proven value (grandparent)
        min_lr: float = 1e-7,
        grad_clip_norm: float = 2.0,    # Proven for Muon orthogonalized updates
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Populated in setup()
        self.input_proj: Optional[nn.Sequential] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None

        # STRING_GNN gene-ID → embedding-row index
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

        self.print("Loading STRING_GNN and computing frozen node embeddings …")
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

        # Build ENSG-ID → row-index mapping
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
        # Output head: head_dropout=0.05 (proven optimal)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )

        # Per-gene additive bias: one offset per (gene × class) pair
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # ---- Class weights (CORRECT ordering after +1 label shift) ----
        # class 0 = down-regulated  (4.77%)  → high weight
        # class 1 = neutral         (92.82%) → low weight
        # class 2 = up-regulated    (2.41%)  → highest weight
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
            f"Architecture: STRING_GNN({GNN_DIM}) → Proj → "
            f"{hp.n_blocks}×PreNormResBlock({hp.hidden_dim},{hp.inner_dim}) "
            f"→ HeadDropout({hp.head_dropout}) → Linear({hp.hidden_dim},{N_GENES}×{N_CLASSES}) + gene_bias"
        )
        self.print(f"Trainable params: {trainable:,} / {total:,}")
        self.print(f"FOCAL LOSS: gamma={hp.focal_gamma} (KEPT from parent, NO Mixup)")
        self.print(f"WEIGHT DECAY: {hp.weight_decay} (REVERTED to grandparent value)")
        self.print("MIXUP: REMOVED (was causing regression in parent)")

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

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits of shape [B, N_CLASSES, N_GENES]."""
        x = self._get_gene_emb(pert_ids)              # [B, 256]
        x = self.input_proj(x)                         # [B, 384]
        for block in self.blocks:
            x = block(x)                               # [B, 384]
        logits = self.output_head(x)                   # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)   # [B, 3, 6640]
        # gene_bias: [N_GENES, N_CLASSES].T → [N_CLASSES, N_GENES] → [1, 3, N_GENES]
        logits = logits + self.gene_bias.T.unsqueeze(0)
        return logits

    def _compute_focal_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss on [B, N_CLASSES, N_GENES] logits.

        FL(p_t) = -(1 - p_t)^gamma * w_c * log(p_t)

        gamma=2.5:
        - Easy samples (neutral, p_t≈0.95): factor=(0.05)^2.5≈0.00056 → near-zero gradient
        - Hard samples (up/down, p_t≈0.2): factor=(0.8)^2.5≈0.573 → strong gradient
        Combined with class_weights for global imbalance reweighting.
        """
        hp = self.hparams
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)

        if hp.focal_gamma == 0.0:
            # Degenerate case: plain weighted CE
            return F.cross_entropy(
                logits_flat,
                labels_flat,
                weight=self.class_weights,
                label_smoothing=hp.label_smoothing,
            )

        # Focal loss: FL(p_t) = -(1 - p_t)^gamma * w_c * log(p_t)
        # Step 1: per-sample weighted CE loss (no reduction)
        ce_per_sample = F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=hp.label_smoothing,
        )

        # Step 2: probability of the true class (p_t)
        with torch.no_grad():
            probs = F.softmax(logits_flat, dim=-1)
            pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)

        # Step 3: focal modulation factor (1 - p_t)^gamma
        focal_weight = (1.0 - pt.clamp(min=1e-8)) ** hp.focal_gamma

        # Step 4: weighted focal loss
        return (focal_weight * ce_per_sample).mean()

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pert_ids = batch["pert_id"]
        labels = batch["label"]

        logits = self(pert_ids)
        loss = self._compute_focal_loss(logits, labels)

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        loss = self._compute_focal_loss(logits, batch["label"])
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

        is_dist = dist.is_available() and dist.is_initialized()

        if is_dist and self.trainer.world_size > 1:
            world_size = dist.get_world_size()
            preds_np_local = preds_local.numpy()
            labels_np_local = labels_local.numpy()

            obj_preds = [None] * world_size
            obj_labels = [None] * world_size
            dist.all_gather_object(obj_preds, preds_np_local)
            dist.all_gather_object(obj_labels, labels_np_local)

            preds_np = np.concatenate(obj_preds, axis=0)
            labels_np = np.concatenate(obj_labels, axis=0)
            f1 = _compute_per_gene_f1(preds_np, labels_np)
            # Log as "val_f1" (NO SLASH) to avoid Lightning creating subdirectories
            # in checkpoint filenames. This is the confirmed root-cause fix.
            self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        else:
            preds_np = preds_local.numpy()
            labels_np = labels_local.numpy()
            f1 = _compute_per_gene_f1(preds_np, labels_np)
            self.log("val_f1", f1, prog_bar=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        self._test_preds.append(logits.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        # self.all_gather always prepends world_size dim
        gathered = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640]
        all_preds = gathered.view(-1, N_CLASSES, N_GENES)  # [N_total, 3, 6640]

        # Gather string metadata
        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        gathered_pert_ids_flat: List[List[str]] = [local_pert_ids]
        gathered_symbols_flat: List[List[str]] = [local_symbols]
        if world_size > 1 and is_dist:
            obj_pids = [None] * world_size
            obj_syms = [None] * world_size
            dist.all_gather_object(obj_pids, local_pert_ids)
            dist.all_gather_object(obj_syms, local_symbols)
            gathered_pert_ids_flat = obj_pids
            gathered_symbols_flat = obj_syms

        if self.trainer.is_global_zero:
            all_pert_ids = [pid for lst in gathered_pert_ids_flat for pid in lst]
            all_symbols = [sym for lst in gathered_symbols_flat for sym in lst]

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

            # Store current checkpoint's predictions
            self._current_test_preds_for_ensemble = np.stack(dedup_preds, axis=0)
            self._current_test_ids = dedup_ids
            self._current_test_syms = dedup_syms

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        from muon import MuonWithAuxAdam

        hp = self.hparams

        # Separate parameters for Muon vs AdamW:
        # Muon: 2D weight matrices in the hidden residual blocks
        # AdamW: all other parameters (norms, biases, input_proj, output_head, gene_bias)
        muon_params = [
            p for name, p in self.blocks.named_parameters()
            if p.ndim >= 2 and p.requires_grad
        ]
        # All other trainable params go to AdamW
        muon_param_ids = set(id(p) for p in muon_params)
        adamw_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in muon_param_ids
        ]

        param_groups = [
            # Muon group for hidden block weight matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            # AdamW group for embeddings, norms, biases, head
            dict(
                params=adamw_params,
                use_muon=False,
                lr=hp.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.weight_decay,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # ReduceLROnPlateau: monitors val_f1 (mode='max'); halves LR at plateaus.
        # threshold=1e-5 (proven grandparent value) — prevents over-aggressive halvings.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=hp.rlrop_factor,
            patience=hp.rlrop_patience,
            min_lr=hp.min_lr,
            threshold=hp.rlrop_threshold,
            threshold_mode="abs",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",  # Matches logged metric name (no slash)
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers (save full state for reliability)
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # Use the standard nn.Module.state_dict() without custom filtering.
        # This is critical for checkpoint loading to work correctly.
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        n_tensors = len(sd)
        total_elems = sum(v.numel() for v in sd.values())
        print(f"Saving checkpoint: {n_tensors} tensors ({total_elems:,} elements)")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-averaged F1 exactly matching calc_metric.py.

    preds  : [N_samples, 3, N_genes]  — logits / class scores
    labels : [N_samples, N_genes]     — integer class labels in {0, 1, 2}
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
    print(f"Saved {len(rows)} test predictions → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node3-3-1-1-1-1-1-1-1: STRING_GNN + 3-Block MLP (h=384) + Muon(lr=0.01) + "
                    "FocalLoss(gamma=2.5) + NO Mixup + WeightDecay=0.01 + Extended Patience=50"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)  # REVERTED from parent's 0.015
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--focal-gamma", type=float, default=2.5,    # KEPT from parent
                   help="Focal loss gamma parameter (0.0 = plain weighted CE)")
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--head-dropout", type=float, default=0.05)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--inner-dim", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--rlrop-factor", type=float, default=0.5)
    p.add_argument("--rlrop-patience", type=int, default=12)
    p.add_argument("--rlrop-threshold", type=float, default=1e-5)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--grad-clip-norm", type=float, default=2.0)
    p.add_argument("--early-stop-patience", type=int, default=50)  # INCREASED from 35
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--val_check_interval", type=float, default=1.0)
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
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        rlrop_factor=args.rlrop_factor,
        rlrop_patience=args.rlrop_patience,
        rlrop_threshold=args.rlrop_threshold,
        min_lr=args.min_lr,
        grad_clip_norm=args.grad_clip_norm,
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

    # Save top-3 checkpoints. Using single best checkpoint for inference
    # (ensemble was zero-benefit in grandparent's 5-ckpt test).
    # CONFIRMED FIX: monitor "val_f1" (no slash) → flat filenames
    # CONFIRMED FIX: auto_insert_metric_name=False prevents Lightning 2.5+ from
    # prepending the metric name to the filename.
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",   # Matches logged metric name
        mode="max",
        patience=args.early_stop_patience,  # INCREASED to 50 (was 35)
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
            find_unused_parameters=False,
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
    # Test: Single best checkpoint
    # ------------------------------------------------------------------
    if args.fast_dev_run or args.debug_max_step is not None:
        # Debug/fast mode: single checkpoint test without ckpt loading
        print("\n=== DEBUG MODE: Single checkpoint test ===")
        trainer.test(model, datamodule=datamodule)
        if hasattr(model, '_current_test_preds_for_ensemble') and trainer.is_global_zero:
            _save_test_predictions(
                pert_ids=model._current_test_ids,
                symbols=model._current_test_syms,
                preds=model._current_test_preds_for_ensemble,
                out_path=output_dir / "test_predictions.tsv",
            )
        test_results = [{"note": "debug_mode_single_ckpt"}]
    else:
        # Production mode: use single best checkpoint
        # The grandparent's 5-checkpoint ensemble showed zero benefit (F1 spread=0.00026)
        # — all checkpoints converged to essentially the same solution.
        # Using single best checkpoint is cleaner and faster.
        print("\n=== PRODUCTION MODE: Single best checkpoint test ===")
        checkpoint_dir = output_dir / "checkpoints"
        ckpt_files = sorted(checkpoint_dir.glob("best-*.ckpt"))

        if ckpt_files:
            # Sort by val_f1 (embedded in filename) to get the best
            def extract_f1(path):
                m = re.search(r'best-[^-]+-([0-9]+\.[0-9]{4})\.ckpt', str(path))
                if m:
                    return float(m.group(1))
                m2 = re.search(r'val_f1=([0-9]+\.[0-9]+)', str(path))
                if m2:
                    return float(m2.group(1))
                return 0.0

            scored = [(f, extract_f1(f)) for f in ckpt_files]
            scored.sort(key=lambda x: x[1], reverse=True)
            best_ckpt = scored[0][0]
            print(f"Best checkpoint: {best_ckpt.name} (val_f1={scored[0][1]:.4f})")

            trainer.test(model, datamodule=datamodule, ckpt_path=str(best_ckpt))
        else:
            print("WARNING: No checkpoint files found. Using best checkpoint via Lightning.")
            trainer.test(model, datamodule=datamodule, ckpt_path='best')

        if hasattr(model, '_current_test_preds_for_ensemble') and trainer.is_global_zero:
            _save_test_predictions(
                pert_ids=model._current_test_ids,
                symbols=model._current_test_syms,
                preds=model._current_test_preds_for_ensemble,
                out_path=output_dir / "test_predictions.tsv",
            )
        test_results = [{"mode": "single_best_checkpoint"}]

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results saved → {score_path}")


if __name__ == "__main__":
    main()
