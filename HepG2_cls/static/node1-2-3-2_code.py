"""Node 1-2-3-2: Extended Training (max_epochs=600) + SWA + Learnable Unknown Gene Embedding
==============================================================================================
Parent  : node1-2-3 (STRING_GNN + Targeted L2 gene bias wd=0.10, test F1=0.4572)
Sibling : node1-2-3-1 (gene_bias wd=0.05 + No Mixup + max_epochs=300, test F1=0.4844)

Key insight from sibling (node1-2-3-1) feedback (HIGHEST PRIORITY):
  1. Training terminated at max_epochs=300 with val_f1 STILL IMPROVING (not converged)
     - Model was only 20 epochs into the 4th CosineWR cycle (T=320, epochs 280-599)
     - Best checkpoint at epoch 295 (val_f1=0.5384), still improving at epoch 299
     - Extrapolated: completing the full 4th cycle → val_f1=0.545-0.555 → test F1 ≥ 0.490-0.500
     - PRIORITY 1: "Extend max_epochs to 500-600; this is the single highest-impact change"
  2. Structural val-test gap (~0.054) persists regardless of regularization
     - PRIORITY 2: "Add Stochastic Weight Averaging (SWA, swa_epoch_start=400-450)"
  3. ~7% of training perturbation genes lack STRING_GNN embeddings (receive zero vectors)
     - PRIORITY 4: "Learnable 'unknown gene' embedding initialized to mean of known embeddings"

Changes from sibling (node1-2-3-1):
------------------------------------
1. MAX EPOCHS: 300 → 600   [PRIMARY: complete the full 4th CosineWR cycle (280-599)]
   - 4th cycle: T=320, epochs 280-599
   - Sibling confirmed val_f1=0.5384 at epoch 295 (just 20 epochs in) and still improving
   - CosineWR schedule: cycles 0-39, 40-119, 120-279, 280-599 ALL fit within max_epochs=600

2. PATIENCE: 30 → 50       [accommodate full 4th cycle exploration]
   - Sibling's patience=30 never fired in 300 epochs (model still improving)
   - With max_epochs=600, patience=50 allows model to explore the declining LR phase
   - Best checkpoint expected at epochs 450-560 → early stop at 500-610 (or max_epochs=600)

3. MANUAL SWA: swa_start_epoch=450, swa_every_n_epochs=10
   [Stochastic Weight Averaging over the late training phase to address the
    structural val-test gap (~0.054). SWA smooths the loss landscape and often
    improves generalization across distribution shifts.]
   - Starts at epoch 450: within 4th CosineWR cycle when LR ≈ 0.0045 (settling phase)
   - Collects snapshots every 10 epochs: 450, 460, 470, ..., 540+ = 10-15 snapshots
   - At train_end: averages collected snapshots → uses SWA model for test evaluation
   - If < 2 snapshots collected: falls back to best checkpoint (robust to early stopping)

4. LEARNABLE UNKNOWN GENE EMBEDDING   [replaces zero vectors for ~7% genes]
   - ~7% of training perturbation genes are absent from STRING_GNN
   - Prior nodes gave these genes zero-vector embeddings → no informative signal
   - This node: nn.Parameter(mean of known STRING_GNN embeddings), initialized biologically
   - Falls into AdamW trunk group (ndim=1, not in Muon group or gene_bias group)
   - Expected impact: improved discrimination for the ~89 unknown-gene perturbations

Preserved from sibling (node1-2-3-1) — proven improvements over parent (node1-2-3):
--------------------------------------------------------------------
- STRING_GNN-only embeddings (no ESM2; confirmed harmful in 5+ tree experiments)
- gene_bias wd=0.05 (REDUCED from parent's 0.10; better biological calibration signal)
  Evidence: Sibling confirmed +0.027 test F1 improvement from this + Mixup removal
- No Manifold Mixup (prob=0.0; removal raised val F1 ceiling from 0.5107 → 0.5384)
- Focal gamma=2.0 (proven optimal across all tree-best nodes, F1≥0.49)
- head_dropout=0.15 (proven optimal; node1-3-2-2-1 and node1-1-1-2-1-1 confirmed)
- trunk wd=0.03 (moderate; above parent's 0.02, below sibling-of-parent's excessive 0.06)
- CosineAnnealingWarmRestarts T_0=40, T_mult=2 (confirmed beneficial)
- MuonWithAuxAdam (Muon for 2D blocks, AdamW for head/biases/gene_bias)
- 3-block PreNormResBlock MLP (hidden=384, inner=768, dropout=0.30)
- Focal loss class weights [0.0477, 0.9282, 0.0241]
- find_unused_parameters=True (DebuggerAgent-confirmed fix from sibling)

Architecture:
    STRING_GNN frozen embeddings [18870, 256]
      -> LookupEmbedding [B, 256]
         (known gene: frozen STRING_GNN[row], unknown gene: learnable unknown_gene_emb)
      -> LN + Linear(256->384) + GELU + Dropout(0.30)  [input_proj]
      -> 3 x PreNormResBlock(384, 768, dropout=0.30)
      -> LN + Dropout(0.15) + Linear(384 -> 6640x3)    [output_head]
      -> reshape [B, 3, 6640]
      -> + gene_bias[6640, 3].T [1, 3, 6640]  [per-gene bias, wd=0.05]
    Output: logits [B, 3, 6640]

Loss: Focal loss (gamma=2.0) with class weights [0.0477, 0.9282, 0.0241]
Optimizer: MuonWithAuxAdam (3 param groups: Muon blocks / AdamW trunk+unknown_emb / AdamW gene_bias)
Scheduler: CosineAnnealingWarmRestarts(T_0=40, T_mult=2, eta_min=1e-7)
SWA: Manual weight averaging from epoch 450 onwards (every 10 epochs)
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
HIDDEN_DIM = 384      # MLP hidden dimension -- proven optimal
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
# Manual SWA Callback
# ---------------------------------------------------------------------------
class ManualSWACallback(pl.Callback):
    """Manual Stochastic Weight Averaging over the late training phase.

    Starting from swa_start_epoch, collects CPU snapshots of all trainable
    parameters every swa_every_n_epochs epochs. At training end, if >= 2
    snapshots were collected, averages them and loads the averaged weights back
    into the model for test evaluation.

    This targets the persistent structural val-test gap (~0.054) by producing
    a smoother, flatter loss-landscape solution that generalizes better across
    the distribution shift between the 141 validation and ~140 test perturbations.

    DDP compatibility: All ranks run identical code and collect the same
    snapshots (since DDP synchronizes parameters during training). All ranks
    apply the same averaging independently, resulting in a consistent SWA model.

    Test logic in main():
        if swa_callback.n_swa_snapshots >= 2:
            trainer.test(model, ...)           # use SWA-averaged model
        else:
            trainer.test(model, ..., ckpt_path='best')  # fallback
    """

    def __init__(self, swa_start_epoch: int = 450, swa_every_n_epochs: int = 10):
        super().__init__()
        self.swa_start_epoch = swa_start_epoch
        self.swa_every_n_epochs = swa_every_n_epochs
        self._snapshots: List[Dict[str, torch.Tensor]] = []
        # Flag set to True once SWA averaging is successfully applied in on_train_end()
        self.swa_applied: bool = False

    @property
    def n_swa_snapshots(self) -> int:
        return len(self._snapshots)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch = trainer.current_epoch
        if (
            epoch >= self.swa_start_epoch
            and (epoch - self.swa_start_epoch) % self.swa_every_n_epochs == 0
        ):
            # Collect a CPU float32 copy of all trainable parameters
            snapshot = {
                name: param.detach().cpu().float().clone()
                for name, param in pl_module.named_parameters()
                if param.requires_grad
            }
            self._snapshots.append(snapshot)
            pl_module.print(
                f"[SWA] Snapshot {len(self._snapshots)} collected at epoch {epoch} "
                f"(target: epoch {self.swa_start_epoch}+, every {self.swa_every_n_epochs} epochs)"
            )

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        n = len(self._snapshots)
        if n < 2:
            pl_module.print(
                f"[SWA] Only {n} snapshot(s) collected (need >= 2). "
                f"Skipping weight averaging — will use best checkpoint for testing."
            )
            return

        pl_module.print(
            f"[SWA] Averaging {n} parameter snapshots "
            f"(epochs {self.swa_start_epoch} to {trainer.current_epoch}, "
            f"step={self.swa_every_n_epochs})..."
        )

        # Compute element-wise mean across all snapshots
        averaged: Dict[str, torch.Tensor] = {}
        for key in self._snapshots[0].keys():
            stacked = torch.stack([sd[key] for sd in self._snapshots], dim=0)
            averaged[key] = stacked.mean(dim=0)

        # Load averaged weights back into the model
        n_loaded = 0
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in averaged:
                param.data.copy_(averaged[name].to(param.device))
                n_loaded += 1

        pl_module.print(
            f"[SWA] Weight averaging complete: {n_loaded} parameter tensors averaged "
            f"over {n} snapshots. Model is now the SWA-averaged version."
        )
        self._snapshots.clear()
        # Mark SWA as successfully applied so main() can use the averaged model for testing
        self.swa_applied = True


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        inner_dim: int = INNER_DIM,
        n_blocks: int = 3,
        dropout: float = 0.30,             # Trunk dropout (proven optimal)
        head_dropout: float = 0.15,        # Proven optimal across tree history
        muon_lr: float = 0.01,             # Proven stable Muon LR
        adamw_lr: float = 3e-4,            # AdamW LR for non-block params
        weight_decay: float = 0.03,        # Moderate wd for MLP trunk
        gene_bias_wd: float = 0.05,        # REDUCED from parent 0.10 -> better calibration
        label_smoothing: float = 0.0,      # Essential for focal loss
        focal_gamma: float = 2.0,          # Proven optimal (tree-best value)
        # CosineAnnealingWarmRestarts params
        cosine_t0: int = 40,               # Confirmed beneficial
        cosine_t_mult: int = 2,            # Cycle length multiplier
        cosine_eta_min: float = 1e-7,      # Minimum LR after annealing
        grad_clip_norm: float = 2.0,       # Proven for Muon orthogonalized updates
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Populated in setup()
        self.input_proj: Optional[nn.Sequential] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None
        self.unknown_gene_emb: Optional[nn.Parameter] = None  # NEW: learnable fallback embedding

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
        # Output head with head_dropout=0.15 (proven optimal)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )

        # Per-gene additive bias: one offset per (gene x class) pair
        # DEDICATED parameter group with wd=0.05 (REDUCED from 0.10 for better
        # biological calibration of housekeeping vs stress-response gene tendencies)
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # Learnable unknown gene embedding for genes absent from STRING_GNN (~7%)
        # Initialized to the MEAN of all known STRING_GNN embeddings — provides a
        # biologically meaningful "average gene" starting point, much better than zeros.
        # Falls into AdamW trunk group (ndim=1; not Muon-eligible, not gene_bias).
        mean_emb = self.gnn_embeddings.mean(dim=0).float()  # [GNN_DIM]
        self.unknown_gene_emb = nn.Parameter(mean_emb.clone())

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
            f"Architecture: STRING_GNN({GNN_DIM}) -> [LookupEmb+LearnableUnknown({GNN_DIM})] "
            f"-> Proj -> {hp.n_blocks}xPreNormResBlock({hp.hidden_dim},{hp.inner_dim}) "
            f"-> HeadDropout({hp.head_dropout}) -> Linear({hp.hidden_dim},{N_GENES}x{N_CLASSES}) "
            f"+ gene_bias(wd={hp.gene_bias_wd})"
        )
        self.print(f"Trainable params: {trainable:,} / {total:,}")
        self.print(f"FOCAL LOSS: gamma={hp.focal_gamma}")
        self.print(
            f"LR SCHEDULE: CosineAnnealingWarmRestarts "
            f"T_0={hp.cosine_t0}, T_mult={hp.cosine_t_mult}, eta_min={hp.cosine_eta_min}"
        )
        self.print(
            f"CosineWR Cycles: 0-39, 40-119, 120-279, 280-599 "
            f"(ALL COMPLETED within max_epochs=600)"
        )
        self.print(f"WEIGHT DECAY: trunk={hp.weight_decay}, gene_bias={hp.gene_bias_wd} (REDUCED from 0.10)")
        self.print(f"HEAD DROPOUT: {hp.head_dropout} (proven optimal)")
        self.print("NO MANIFOLD MIXUP (removed; confirmed raises val F1 ceiling)")
        self.print("LEARNABLE UNKNOWN GENE EMBEDDING: mean-initialized (replaces zero vectors)")

    # ------------------------------------------------------------------
    def _get_gene_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Batch lookup of STRING_GNN embeddings for ENSG IDs.

        Known genes: frozen STRING_GNN embedding (buffer, no gradient)
        Unknown genes (~7%): learnable unknown_gene_emb parameter (trainable)
        """
        emb_list: List[torch.Tensor] = []
        for pid in pert_ids:
            row = self.gnn_id_to_idx.get(pid)
            if row is not None:
                emb_list.append(self.gnn_embeddings[row])
            else:
                # Learnable fallback: initialized to mean of known STRING_GNN embeddings
                # Gradient flows through unknown_gene_emb for these samples
                emb_list.append(self.unknown_gene_emb)
        return torch.stack(emb_list, dim=0)  # [B, 256]

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits of shape [B, N_CLASSES, N_GENES] (inference mode)."""
        x = self._get_gene_emb(pert_ids)              # [B, 256]
        x = self.input_proj(x)                         # [B, 384]
        for block in self.blocks:
            x = block(x)                               # [B, 384]
        logits = self.output_head(x)                   # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)   # [B, 3, 6640]
        # gene_bias: [N_GENES, N_CLASSES].T -> [N_CLASSES, N_GENES] -> [1, 3, N_GENES]
        logits = logits + self.gene_bias.T.unsqueeze(0)
        return logits

    def _compute_focal_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss on [B, N_CLASSES, N_GENES] logits.

        Focal loss (gamma=2.0): FL(p_t) = -(1 - p_t)^gamma * w_c * log(p_t)
        gamma=2.0 is used by ALL tree-best nodes (F1>=0.497).
        """
        hp = self.hparams
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)

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

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pert_ids = batch["pert_id"]
        labels = batch["label"]
        # No Manifold Mixup (removed; confirmed raises val F1 ceiling from 0.5107 to 0.5384)
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

        # Compute F1 on each rank from its local slice, then average across ranks.
        is_dist = dist.is_available() and dist.is_initialized()
        if is_dist:
            f1_local = _compute_per_gene_f1(
                preds_local.numpy(), labels_local.numpy()
            )
            f1_tensor = torch.tensor(f1_local, device=self.device)
            dist.all_reduce(f1_tensor, op=dist.ReduceOp.SUM)
            f1_tensor /= dist.get_world_size()
            self.log("val_f1", f1_tensor.item(), prog_bar=True, sync_dist=True)
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

        # Lightning DDP-safe all_gather
        gathered = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640]
        all_preds = gathered.view(-1, N_CLASSES, N_GENES)  # [N_total, 3, 6640]

        # Gather string metadata via dist.all_gather_object
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

        # ---- Parameter Group Separation ----
        # Group 1 — Muon: 2D weight matrices in residual blocks
        # Group 2 — AdamW (trunk): head, norms, biases, input_proj, unknown_gene_emb
        #   NOTE: unknown_gene_emb has ndim=1, so it's NOT in Muon group; auto-routes here
        # Group 3 — AdamW (gene_bias): dedicated group with wd=0.05

        muon_params = [
            p for name, p in self.blocks.named_parameters()
            if p.ndim >= 2 and p.requires_grad
        ]
        muon_param_ids = set(id(p) for p in muon_params)
        gene_bias_ids = {id(self.gene_bias)}

        # All params not in Muon group and not gene_bias go to AdamW trunk group
        # This includes: input_proj, output_head, unknown_gene_emb (ndim=1)
        adamw_trunk_params = [
            p for p in self.parameters()
            if p.requires_grad
            and id(p) not in muon_param_ids
            and id(p) not in gene_bias_ids
        ]

        param_groups = [
            # Group 1: Muon for hidden block weight matrices (2D)
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,   # wd=0.03 (moderate for trunk)
                momentum=0.95,
            ),
            # Group 2: AdamW for head, norms, biases, input_proj, unknown_gene_emb
            dict(
                params=adamw_trunk_params,
                use_muon=False,
                lr=hp.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.weight_decay,   # wd=0.03 (moderate)
            ),
            # Group 3: AdamW for gene_bias with calibrated wd=0.05
            # REDUCED from parent's 0.10 to allow better biological calibration signal
            dict(
                params=[self.gene_bias],
                use_muon=False,
                lr=hp.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.gene_bias_wd,   # wd=0.05 (1.67x trunk, calibrated)
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # CosineAnnealingWarmRestarts: ALL 4 cycles complete within max_epochs=600
        # Cycle schedule: 0-39 (T=40), 40-119 (T=80), 120-279 (T=160), 280-599 (T=320)
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
    # Checkpoint helpers (save only trainable params + buffers)
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Node1-2-3-2: Extended Training (max_epochs=600) + SWA + Learnable Unknown Gene Emb\n"
            "  Primary: Complete 4th CosineWR cycle (280-599) that sibling (node1-2-3-1) missed\n"
            "  Secondary: SWA (swa_start=450) to address structural val-test gap (~0.054)\n"
            "  Tertiary: Learnable unknown gene embedding replaces zero vectors for ~7% genes"
        )
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=600,
                   help="EXTENDED to 600 to complete the 4th CosineWR cycle (280-599)")
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.03,
                   help="MLP trunk weight decay (moderate, proven optimal)")
    p.add_argument("--gene-bias-wd", type=float, default=0.05,
                   help="REDUCED from parent 0.10 -> better biological calibration (sibling confirmed)")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma (proven optimal: used by all tree-best nodes F1>=0.497)")
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--head-dropout", type=float, default=0.15,
                   help="Head dropout (proven optimal from tree history)")
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--inner-dim", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--cosine-t0", type=int, default=40,
                   help="CosineWR T_0 (confirmed beneficial from sibling lineage)")
    p.add_argument("--cosine-t-mult", type=int, default=2)
    p.add_argument("--cosine-eta-min", type=float, default=1e-7)
    p.add_argument("--grad-clip-norm", type=float, default=2.0)
    p.add_argument("--early-stop-patience", type=int, default=50,
                   help="EXTENDED to 50 to allow full 4th cycle exploration without premature stopping")
    p.add_argument("--save-top-k", type=int, default=3,
                   help="Number of best checkpoints to save (used as fallback if SWA not triggered)")
    p.add_argument("--swa-start-epoch", type=int, default=450,
                   help="Epoch to start collecting SWA snapshots (4th cycle settling phase)")
    p.add_argument("--swa-every-n-epochs", type=int, default=10,
                   help="Interval between SWA snapshots")
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
        gene_bias_wd=args.gene_bias_wd,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_t_mult,
        cosine_eta_min=args.cosine_eta_min,
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

    # Checkpoint: save top-3 by val_f1 for fallback in case SWA not triggered
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    # Patience=50: allows full 4th CosineWR cycle (280-599) to be explored
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stop_patience,
        min_delta=1e-5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # SWA callback: collect snapshots from epoch 450 onwards, average at train end
    swa_callback = ManualSWACallback(
        swa_start_epoch=args.swa_start_epoch,
        swa_every_n_epochs=args.swa_every_n_epochs,
    )

    callbacks = [checkpoint_cb, early_stop_cb, lr_monitor, progress_bar, swa_callback]

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(
            find_unused_parameters=True,   # Required for unknown_gene_emb not in every batch
            timeout=timedelta(seconds=120),
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,        # 600 to complete the full 4th CosineWR cycle
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
    # Test: Use SWA-averaged model if available, else best checkpoint
    #
    # SWA averaging happens in swa_callback.on_train_end() BEFORE this point.
    # If >= 2 snapshots were collected, the model's weights are now the SWA
    # average — use the current model for testing (do NOT load 'best' checkpoint,
    # which would overwrite the SWA-averaged weights).
    # If < 2 snapshots (e.g., training ended too early), fall back to the
    # ModelCheckpoint's best checkpoint.
    # ------------------------------------------------------------------
    if args.fast_dev_run or args.debug_max_step is not None:
        print("\n=== DEBUG MODE: Testing without loading checkpoint ===")
        trainer.test(model, datamodule=datamodule)
    elif swa_callback.swa_applied:
        # SWA was applied in on_train_end(); model weights are already the SWA average.
        # Do NOT load 'best' checkpoint — that would overwrite the averaged weights.
        print(
            f"\n=== PRODUCTION MODE: Testing with SWA-averaged model ==="
        )
        trainer.test(model, datamodule=datamodule)
    else:
        print(
            f"\n=== PRODUCTION MODE: Testing with best checkpoint "
            f"(SWA not applied — fewer than 2 snapshots collected) ==="
        )
        trainer.test(model, datamodule=datamodule, ckpt_path='best')

    # Save test predictions
    if trainer.is_global_zero:
        if hasattr(model, '_current_test_preds'):
            out_path = output_dir / "test_predictions.tsv"
            _save_test_predictions(
                pert_ids=model._current_test_ids,
                symbols=model._current_test_syms,
                preds=model._current_test_preds,
                out_path=out_path,
            )
            print(f"Test predictions saved -> {out_path}")
        else:
            print("WARNING: No test predictions to save.")

        # Save test results placeholder
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            json.dumps({"status": "test_complete_awaiting_eval"}, indent=2)
        )
        print(f"Test results placeholder saved -> {score_path}")


if __name__ == "__main__":
    main()
