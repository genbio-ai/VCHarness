"""Node 1-2-3-1: STRING_GNN + Calibrated Gene Bias + Reduced Over-Regularization
================================================================
Parent  : node1-2-3 (STRING_GNN + targeted L2 gene bias wd=0.10 + Mixup prob=0.25)
          test F1=0.4572, best val_f1=0.5107 at epoch 94
          Early stopped at epoch 114, missed second CosineWR restart at epoch 120

Key insights from node1-2-3 feedback (Priority 1, 2, 3):
  Priority 1: Reduce gene_bias wd from 0.10 to 0.05
    - wd=0.10 too aggressive for biological calibration signal (housekeeping gene tendencies)
    - 0.05 still provides 1.67x higher L2 than trunk (wd=0.03) for targeted memorization control
  Priority 2: Extend patience from 20 to 30 to capture second warm restart
    - First restart at epoch 40 provided +0.012 val_f1
    - Second restart at epoch 120 was missed by only 6 epochs (early stop at epoch 114)
    - patience=30 guarantees epoch 124+ is reached after any best_epoch<=94
  Priority 3: Remove Manifold Mixup (prob=0.0) to reduce collective over-regularization
    - Parent had 5 simultaneous regularizers: gamma=2.0 + wd=0.03 + gene_bias wd=0.10 +
      head_dropout=0.15 + mixup prob=0.25 → lowered val F1 ceiling to 0.5107
    - Remaining 4-mechanism stack (gamma=2.0 + wd=0.03 + gene_bias wd=0.05 +
      head_dropout=0.15) provides sufficient regularization
    - node4-1-1-1-1 (F1=0.5072, tree-best STRING-only): No Mixup
    - node1-3-2-2-1-1-1-1-1-1-1-1 (F1=0.4968): WCE without Mixup

Additional change:
  Max epochs 250 → 300 to accommodate second warm restart + patience window.

Changes in this node vs parent (node1-2-3):
-------------------------------------------------
1. GENE BIAS WEIGHT DECAY: 0.10 → 0.05  [PRIMARY BIOLOGICAL CALIBRATION FIX]
   Evidence:
   - node1-2-3 feedback: "gene_bias wd=0.10 may be slightly too aggressive"
   - gene_bias encodes biological constants (housekeeping vs stress-response tendencies)
   - 0.05 still penalizes memorization with 1.67x more L2 than trunk (wd=0.03)
   - Expected: +0.01-0.02 test F1 from recovering calibration signal

2. EARLY STOP PATIENCE: 20 → 30  [CAPTURE SECOND WARM RESTART]
   Evidence:
   - node1-2-3 training: best at epoch 94, stopped at epoch 114 (20 after best)
   - Second CosineWR restart at epoch 120, missed by 6 epochs
   - First restart at epoch 40 gave +0.012 val_f1
   - patience=30: training extends to at least epoch 124, second restart captured

3. MANIFOLD MIXUP: Removed (prob=0.25 → 0.0)  [REDUCE COLLECTIVE OVER-REGULARIZATION]
   Evidence:
   - node1-2-3 feedback: "collective over-regularization from 5 simultaneous mechanisms"
   - node4-1-1-1-1 (F1=0.5072): STRING-only, NO Mixup, achieves tree-best STRING-only F1
   - node1-3-2-2-1-1-1-1-1-1-1-1 (F1=0.4968): WCE without Mixup
   - Remaining 4 regularizers (gamma=2.0 + wd=0.03 + gene_bias wd=0.05 +
     head_dropout=0.15) is sufficient without Mixup
   - Expected: raise val F1 ceiling from 0.5107 to >0.520

4. MAX EPOCHS: 250 → 300  [EXTENDED BUDGET FOR 2ND WARM RESTART]
   Evidence:
   - CosineWR cycles: 0-39, 40-119, 120-279 (T_0=40, T_mult=2)
   - With best expected at epoch 100-160 and patience=30, early stop at ~130-190
   - 300 epochs provides sufficient budget for three complete cycles

Preserved from parent (node1-2-3):
-----------------------------------------------------------
- STRING_GNN frozen PPI graph embeddings (256-dim) — ESM2 NOT used (confirmed harmful)
- Per-gene additive bias (19,920 params) — PRESERVED with calibrated high wd=0.05
- Pre-norm residual block structure (3 blocks, h=384, inner=768)
- Trunk dropout=0.30 — proven optimal
- Head dropout=0.15 — proven optimal (tree-confirmed sweet spot)
- Muon+AdamW dual optimizer (Muon LR=0.01, AdamW LR=3e-4) — proven best
- Focal loss (gamma=2.0) — primary anti-overfitting fix from node1-2-3
- Trunk weight decay=0.03 — moderate, appropriate
- CosineAnnealingWarmRestarts T_0=40, T_mult=2 — confirmed beneficial
- Gradient clip=2.0
- Class-weight order: [0.0477, 0.9282, 0.0241] (down/neutral/up)
- val_f1 metric name (no slash) + auto_insert_metric_name=False
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
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        inner_dim: int = INNER_DIM,
        n_blocks: int = 3,
        dropout: float = 0.30,            # Trunk dropout (proven optimal)
        head_dropout: float = 0.15,       # Proven optimal (tree-confirmed sweet spot)
        muon_lr: float = 0.01,            # Proven stable Muon LR
        adamw_lr: float = 3e-4,           # AdamW LR for non-block params
        weight_decay: float = 0.03,       # Moderate wd for MLP trunk
        gene_bias_wd: float = 0.05,       # REDUCED: 0.10->0.05 for better biological calibration
        label_smoothing: float = 0.0,     # Essential for low focal train loss
        focal_gamma: float = 2.0,         # Proven optimal (tree-best value)
        # CosineAnnealingWarmRestarts params
        cosine_t0: int = 40,              # Shortened T_0 (confirmed beneficial)
        cosine_t_mult: int = 2,           # Cycle length multiplier
        cosine_eta_min: float = 1e-7,     # Minimum LR after annealing
        grad_clip_norm: float = 2.0,      # Proven for Muon orthogonalized updates
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Populated in setup()
        self.input_proj: Optional[nn.Sequential] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None

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
        # RETAINED with CALIBRATED weight decay (wd=0.05) in optimizer
        # This preserves biological calibration signal while penalizing memorization
        # Reduced from parent's 0.10 to allow better expression of baseline gene tendencies
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
            f"-> HeadDropout({hp.head_dropout}) -> Linear({hp.hidden_dim},{N_GENES}x{N_CLASSES}) + gene_bias"
        )
        self.print(f"Trainable params: {trainable:,} / {total:,}")
        self.print(f"FOCAL LOSS: gamma={hp.focal_gamma} (proven optimal)")
        self.print(
            f"LR SCHEDULE: CosineAnnealingWarmRestarts "
            f"T_0={hp.cosine_t0}, T_mult={hp.cosine_t_mult}, eta_min={hp.cosine_eta_min}"
        )
        self.print(f"WEIGHT DECAY: trunk={hp.weight_decay}, gene_bias={hp.gene_bias_wd} "
                   f"(REDUCED from parent's 0.10 for better biological calibration)")
        self.print(f"HEAD DROPOUT: {hp.head_dropout} (proven optimal, tree-confirmed sweet spot)")
        self.print("MANIFOLD MIXUP: REMOVED (reduces collective over-regularization, "
                   "4 remaining regularizers are sufficient)")

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

        Focal loss (gamma=2.0) -- proven optimal configuration (tree-best nodes).

        With gamma=2.0:
        - Easy samples (p_t=0.95): factor = (0.05)^2.0 = 0.0025x base weight
        - Hard samples (p_t=0.20): factor = (0.80)^2.0 = 0.640x base weight
        - Misclassified minority (p_t=0.10): factor = (0.90)^2.0 = 0.810x base weight

        This configuration provides healthier 2-5x train/val loss ratios
        vs gamma=2.5's extreme 13x overfitting in node1-2.
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

        # No Manifold Mixup: removed to reduce collective over-regularization
        # The remaining 4 regularizers (gamma=2.0 + wd=0.03 + gene_bias wd=0.05 +
        # head_dropout=0.15) provide sufficient regularization for this task.
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
        # Muon: 2D weight matrices in the hidden residual blocks
        # AdamW (trunk): all other params EXCEPT gene_bias
        # AdamW (gene_bias): DEDICATED group with calibrated wd=0.05
        #
        # Key design: gene_bias gets 1.67x higher weight decay than MLP trunk
        # This balances biological calibration signal with memorization control.
        # Reduced from parent's 0.10 to allow better expression of baseline gene tendencies.
        muon_params = [
            p for name, p in self.blocks.named_parameters()
            if p.ndim >= 2 and p.requires_grad
        ]
        muon_param_ids = set(id(p) for p in muon_params)
        gene_bias_ids = {id(self.gene_bias)}

        # All params not in Muon group and not gene_bias go to AdamW trunk group
        adamw_trunk_params = [
            p for p in self.parameters()
            if p.requires_grad
            and id(p) not in muon_param_ids
            and id(p) not in gene_bias_ids
        ]

        param_groups = [
            # Group 1: Muon for hidden block weight matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,  # wd=0.03 (moderate)
                momentum=0.95,
            ),
            # Group 2: AdamW for head, norms, biases (except gene_bias), input_proj
            dict(
                params=adamw_trunk_params,
                use_muon=False,
                lr=hp.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.weight_decay,  # wd=0.03 (moderate)
            ),
            # Group 3: AdamW for gene_bias with CALIBRATED wd=0.05
            # This is the IMPROVED design: targeted L2 on memorization-prone
            # per-gene bias (19,920 params) while allowing biological calibration.
            # Reduced from parent's wd=0.10 to wd=0.05:
            # - Still 1.67x higher than trunk wd=0.03 for targeted memorization control
            # - Low enough to preserve biological baseline expression tendencies
            dict(
                params=[self.gene_bias],
                use_muon=False,
                lr=hp.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.gene_bias_wd,  # wd=0.05 (CALIBRATED, down from parent's 0.10)
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # CosineAnnealingWarmRestarts with T_0=40.
        # First restart at epoch 40: confirmed +0.012 val_f1 in parent node.
        # Second restart at epoch 120: missed by 6 epochs in parent; NOW CAPTURED
        # via extended patience=30.
        # Cycles: 0-39, 40-119, 120-279 (T_mult=2)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Node1-2-3-1: STRING_GNN + Calibrated Gene Bias + Reduced Over-Regularization\n"
            "  gamma=2.0, head_dropout=0.15, gene_bias_wd=0.05 (REDUCED from 0.10),\n"
            "  NO Manifold Mixup, CosineWR T_0=40, patience=30 (EXTENDED)\n"
            "\n"
            "Key improvements over parent (node1-2-3):\n"
            "  1. gene_bias_wd: 0.10 -> 0.05 (recover biological calibration signal)\n"
            "  2. patience: 20 -> 30 (capture second warm restart at epoch 120)\n"
            "  3. Manifold Mixup removed (reduce collective over-regularization)\n"
            "  4. max_epochs: 250 -> 300 (extended budget for 2nd warm restart + patience)"
        )
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=300,
                   help="Extended budget: T_0=40 provides 3 cycles in 280 epochs; "
                        "with patience=30 and best at epoch 100-160, early stop at ~130-190")
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.03,
                   help="Moderate wd for MLP trunk (unchanged from parent)")
    p.add_argument("--gene-bias-wd", type=float, default=0.05,
                   help="REDUCED wd for gene_bias: 0.10->0.05 to allow better biological "
                        "calibration while still providing 1.67x higher L2 than trunk")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma (proven optimal, tree-best value)")
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--head-dropout", type=float, default=0.15,
                   help="Head dropout (proven optimal, tree-confirmed sweet spot)")
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--inner-dim", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--cosine-t0", type=int, default=40,
                   help="CosineAnnealingWR T_0: confirmed beneficial in node1-2-2")
    p.add_argument("--cosine-t-mult", type=int, default=2)
    p.add_argument("--cosine-eta-min", type=float, default=1e-7)
    p.add_argument("--grad-clip-norm", type=float, default=2.0)
    p.add_argument("--early-stop-patience", type=int, default=30,
                   help="EXTENDED patience: 20->30 to capture second CosineWR restart "
                        "at epoch 120 (parent missed by only 6 epochs)")
    p.add_argument("--save-top-k", type=int, default=3,
                   help="Number of best checkpoints to save")
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

    # Save top-3 checkpoints; use best single checkpoint for test.
    # CONFIRMED FIX: monitor "val_f1" (no slash) -> flat filenames
    # CONFIRMED FIX: auto_insert_metric_name=False prevents metric name prefix duplication
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    # Patience=30: EXTENDED to capture second CosineWR warm restart at epoch 120
    # (parent with patience=20 missed the second restart by only 6 epochs)
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
    # Test: Use best single checkpoint
    # (Ensemble confirmed to provide zero/negative benefit per tree history)
    # ------------------------------------------------------------------
    if args.fast_dev_run or args.debug_max_step is not None:
        print("\n=== DEBUG MODE: Testing without loading checkpoint ===")
        trainer.test(model, datamodule=datamodule)
    else:
        print("\n=== PRODUCTION MODE: Testing with best checkpoint ===")
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
        score_path.write_text(json.dumps({"status": "test_complete_awaiting_eval"}, indent=2))
        print(f"Test results placeholder saved -> {score_path}")


if __name__ == "__main__":
    main()
