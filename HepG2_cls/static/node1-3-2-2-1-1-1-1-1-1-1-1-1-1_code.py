r"""Node 1-3-2-2-1-1-1-1-1-1-1-1-1-1: Focal Loss Calibration + Patience Reduction + DDP Fix

PARENT: node1-3-2-2-1-1-1-1-1-1-1-1-1 (test F1=0.4940, focal loss gamma=2.0 alpha=0.30)

ROOT CAUSE ANALYSIS (from parent feedback):
  The parent's focal loss (gamma=2.0, alpha=0.30) converged faster (epoch 266 vs WCE epoch 469)
  but landed at a slightly lower peak (val_f1=0.4935 vs WCE val_f1=0.4999).

  Three calibration issues identified:
  1. class_weight_alpha=0.30 was too conservative when combined with focal_gamma=2.0.
     Class weighting and focal loss are INDEPENDENT mechanisms:
     - focal_gamma handles hard examples via (1-pt)^gamma down-weighting
     - class_weight_alpha provides minority class emphasis via frequency-based weights
     Reducing alpha from 0.35->0.30 over-corrected; minority classes were under-weighted.

  2. patience=160 was sized for WCE's late convergence (epoch 469).
     Focal loss peaked at epoch 266; 160 more epochs of training wasted compute.
     patience=100 aligns with focal loss's faster convergence dynamics.

  3. focal_gamma=2.0 is aggressive; (1-pt)^2 for many neutral examples is non-trivial
     (the neutral class has 92.8% frequency but not all are trivially easy).
     gamma=1.5 produces milder modulation and may find a better local minimum.

  4. DDP test F1 was computed via all_reduce(SUM)/ws (average of per-rank F1s),
     which is INCORRECT because F1 is a non-linear metric. Using all_gather to
     collect all predictions before computing F1 gives the correct result.

CHANGES IN THIS NODE:
  1. class_weight_alpha: 0.30 -> 0.40 (primary fix: stronger minority class emphasis)
  2. focal_gamma: 2.0 -> 1.5 (secondary fix: milder modulation for better local minimum)
  3. patience: 160 -> 100 (aligned with focal loss's faster convergence)
  4. max_epochs: 500 -> 400 (sufficient: T_0=80, peak ~280-300, patience=100 -> stops ~400)
  5. Fix DDP test F1: all_gather instead of all_reduce for correct metric computation
  6. All other hyperparams preserved: muon_lr=0.01, adamw_lr=3e-4, weight_decay=8e-4,
     mixup_prob=0.5, mixup_alpha=0.2, head_dropout=0.18, trunk_dropout=0.28,
     cosine_t0=80, cosine_t_mult=2, cosine_eta_min=5e-7, hidden_dim=384, n_blocks=3

Architecture (same as parent, unchanged):
  STRING_GNN -> frozen 256-dim embeddings
  -> Manifold Mixup (prob=0.5, alpha=0.2) at embedding level
  -> Linear(256->384) + LayerNorm + GELU
  -> 3x ResidualBlock(384, dropout=0.28)
  -> LayerNorm + Dropout(0.18) + Linear(384->6640*3) + per-gene bias
  -> reshape [B, 3, 6640]

Loss: Focal loss (gamma=1.5) with class weights (alpha=0.40)
Optimizer: Muon+AdamW dual optimizer
Scheduler: CosineAnnealingWarmRestarts(T_0=80, T_mult=2, eta_min=5e-7)
Test: Single best checkpoint ONLY (ensemble disabled)

Tree context:
  node1-3-2-2-1-1-1-1-1-1-1-1-1 (parent, F1=0.4940) | focal gamma=2.0/alpha=0.30
  node1-3-2-2-1-1-1-1-1-1 (grandparent+1, F1=0.4968) | WCE, warm restarts
  node4-1-1-1-1-1 (tree-best, F1=0.5175) | ESM2+STRING, focal loss
  This node targets F1 > 0.4968 (beating grandparent) via calibrated focal loss
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
N_GENES = 6640
N_CLASSES = 3
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Gene-perturbation -> differential-expression dataset."""

    def __init__(self, df: pd.DataFrame, gene2str_idx: Dict[str, int]) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        # Map ENSEMBL pert_id -> STRING-node-index; -1 = not in STRING graph
        self.str_indices = torch.tensor(
            [gene2str_idx.get(pid, -1) for pid in self.pert_ids], dtype=torch.long
        )
        if "label" in df.columns:
            labels = np.array([json.loads(x) for x in df["label"]], dtype=np.int64)
            self.labels = torch.tensor(labels + 1, dtype=torch.long)  # {-1,0,1} -> {0,1,2}
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "str_idx": self.str_indices[idx],
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
        micro_batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.gene2str_idx: Dict[str, int] = {}
        self.train_ds = self.val_ds = self.test_ds = None

    def setup(self, stage: str = "fit") -> None:
        # Build ENSEMBL-ID -> STRING-node-index mapping once
        if not self.gene2str_idx:
            node_names: List[str] = json.loads(
                (STRING_GNN_DIR / "node_names.json").read_text()
            )
            self.gene2str_idx = {ensg: i for i, ensg in enumerate(node_names)}

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df, self.gene2str_idx)
        self.val_ds = PerturbDataset(val_df, self.gene2str_idx)
        self.test_ds = PerturbDataset(test_df, self.gene2str_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Pre-LayerNorm residual MLP block (hidden_dim -> hidden_dim*2 -> hidden_dim)."""

    def __init__(self, dim: int, dropout: float = 0.28) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(self.norm(x)))


class PerturbMLP(nn.Module):
    """STRING-only MLP for gene perturbation response prediction.

    Architecture (per sample):
      (1) STRING_GNN embedding lookup [256-dim, frozen buffer]
         (fallback learnable 256-dim for ~6% genes not in STRING)
      (2) Input projection: Linear(256->hidden_dim) + LN + GELU
      (3) n_blocks x ResidualBlock(hidden_dim, trunk_dropout=0.28)
      (4) LN(hidden_dim) -> Dropout(head_dropout=0.18) -> Linear(hidden_dim -> 6640*3) + per-gene-bias
      (5) reshape -> [B, 3, 6640]

    Methods for Manifold Mixup:
      - _lookup_embedding(): separate embedding lookup for Mixup
      - forward_from_emb(): forward from pre-computed embeddings
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.28,        # trunk dropout
        head_dropout: float = 0.18,   # head dropout (proven grandparent value)
    ) -> None:
        super().__init__()
        # Learnable fallback embedding for genes not in STRING graph
        self.fallback_emb = nn.Parameter(torch.zeros(256))
        nn.init.normal_(self.fallback_emb, std=0.02)

        # Input projection: 256 -> hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Residual MLP blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        # Flat output head + per-gene additive bias.
        # head_dropout=0.18 - proven grandparent value.
        # Flat head confirmed superior over factorized heads across 4+ nodes.
        # Per-gene bias confirmed helpful.
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, N_GENES * N_CLASSES),
        )
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES * N_CLASSES))

    def _lookup_embedding(self, str_idx: torch.Tensor, string_embs: torch.Tensor) -> torch.Tensor:
        """Lookup STRING embeddings with fallback for missing genes. Returns [B, 256]."""
        valid_mask = str_idx >= 0
        safe_idx = str_idx.clamp(min=0)
        emb = string_embs[safe_idx].to(self.fallback_emb)
        if not valid_mask.all():
            fallback = self.fallback_emb.unsqueeze(0).expand(int((~valid_mask).sum()), -1)
            emb = emb.clone()
            emb[~valid_mask] = fallback
        return emb

    def forward_from_emb(self, emb: torch.Tensor) -> torch.Tensor:
        """Forward from pre-computed embeddings (for Mixup at embedding level). Returns [B, 3, 6640]."""
        x = self.input_proj(emb)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x) + self.gene_bias.to(x)
        return logits.view(-1, N_CLASSES, N_GENES)

    def forward(
        self,
        str_idx: torch.Tensor,       # [B]  STRING node indices, -1 = not in graph
        string_embs: torch.Tensor,   # [18870, 256] frozen buffer
    ) -> torch.Tensor:
        emb = self._lookup_embedding(str_idx, string_embs)
        return self.forward_from_emb(emb)


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.28,            # trunk dropout
        head_dropout: float = 0.18,       # proven grandparent value
        lr: float = 3e-4,
        muon_lr: float = 0.01,            # proven stable
        weight_decay: float = 8e-4,       # proven grandparent value
        focal_gamma: float = 1.5,         # CALIBRATED: 2.0->1.5 (milder modulation)
        class_weight_alpha: float = 0.40, # CALIBRATED: 0.30->0.40 (stronger minority emphasis)
        cosine_t0: int = 80,              # T_0 for CosineAnnealingWarmRestarts
        cosine_t_mult: int = 2,           # T_mult for warm restarts
        cosine_eta_min: float = 5e-7,     # Minimum LR
        max_epochs: int = 400,            # Reduced: 500->400 (aligned with focal convergence)
        grad_clip_norm: float = 1.0,
        use_muon: bool = True,
        mixup_alpha: float = 0.2,         # Manifold Mixup alpha
        mixup_prob: float = 0.5,          # Proven grandparent value
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.lr = lr
        self.muon_lr = muon_lr
        self.weight_decay = weight_decay
        self.focal_gamma = focal_gamma
        self.class_weight_alpha = class_weight_alpha
        self.cosine_t0 = cosine_t0
        self.cosine_t_mult = cosine_t_mult
        self.cosine_eta_min = cosine_eta_min
        self.max_epochs = max_epochs
        self.grad_clip_norm = grad_clip_norm
        self.use_muon = use_muon
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob

        # Build model in __init__ so configure_optimizers sees all parameters
        self.model = PerturbMLP(
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout,
            head_dropout=head_dropout,
        )

        # Accumulation buffers for epoch-level metrics
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def _load_string_embeddings(self, skip_barrier: bool = False) -> None:
        """Load STRING_GNN embeddings and register as frozen buffer."""
        if hasattr(self, "string_embs") and self.string_embs is not None:
            return  # already loaded

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            from transformers import AutoModel as _AM
            _AM.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        if not skip_barrier and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        from transformers import AutoModel
        gnn = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        gnn.eval()
        graph = torch.load(
            STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False
        )
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)
        with torch.no_grad():
            gnn_out = gnn(edge_index=edge_index, edge_weight=edge_weight)
        string_embs = gnn_out.last_hidden_state.detach().float().cpu()  # [18870, 256]
        del gnn, gnn_out
        self.register_buffer("string_embs", string_embs)

    def setup(self, stage: str = "fit") -> None:
        # Calibrated class weights: w_i = (1/freq_i)^alpha
        # alpha=0.40: stronger minority emphasis (CALIBRATED from parent's 0.30)
        # Rationale: focal loss handles hard examples via (1-pt)^gamma independently;
        # class weighting provides minority emphasis regardless of prediction confidence.
        # alpha=0.30 was too conservative; alpha=0.40 recovers proper minority class focus.
        alpha = self.class_weight_alpha
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq).pow(alpha)
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        self.register_buffer("class_weights", class_weights)

        if hasattr(self, "string_embs") and self.string_embs is not None:
            return  # already loaded

        # ---- Load STRING_GNN node embeddings ----
        self._load_string_embeddings()

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        if hasattr(self, "trainer") and self.trainer is not None:
            self.print(
                f"Node1-3-2-2-1-1-1-1-1-1-1-1-1-1 PerturbMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
                f"trunk_dropout={self.dropout} | head_dropout={self.head_dropout} | "
                f"class_weight_alpha={self.class_weight_alpha} | focal_gamma={self.focal_gamma} | "
                f"muon_lr={self.muon_lr} | cosine_t0={self.cosine_t0} | cosine_t_mult={self.cosine_t_mult} | "
                f"cosine_eta_min={self.cosine_eta_min} | use_muon={self.use_muon} | "
                f"max_epochs={self.max_epochs} | mixup_alpha={self.mixup_alpha} | "
                f"mixup_prob={self.mixup_prob} | weight_decay={self.weight_decay} | "
                f"trainable={n_trainable:,}/{n_total:,}"
            )

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss with class weights.

        logits: [B, 3, 6640]
        labels: [B, 6640]  - values in {0, 1, 2}
        gamma=1.5: CALIBRATED milder modulation (vs parent's 2.0)
        - Less aggressive down-weighting of 'easy' neutral examples
        - Produces more stable optimization landscape
        - Combined with stronger alpha=0.40 for proper minority class emphasis
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                                # [B*6640]

        # Compute cross-entropy per element (no reduction)
        ce_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            reduction="none",
        )

        # Compute p_t for focal weighting
        with torch.no_grad():
            probs = F.softmax(logits_flat, dim=-1)
            p_t = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - p_t) ** self.focal_gamma

        return (focal_weight * ce_loss).mean()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        str_idx = batch["str_idx"]
        labels = batch["label"]

        if (self.mixup_alpha > 0
                and self.training
                and np.random.random() < self.mixup_prob):
            B = str_idx.size(0)
            # Sample mixing coefficient, enforce lambda >= 0.5 for stability
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            lam = max(lam, 1.0 - lam)

            # Random permutation for Mixup pair selection
            perm = torch.randperm(B, device=str_idx.device)

            # Get STRING embeddings
            emb = self.model._lookup_embedding(str_idx, self.string_embs)  # [B, 256]

            # Mixed embedding (Manifold Mixup at STRING embedding level)
            mixed_emb = lam * emb + (1.0 - lam) * emb[perm]

            # Forward with mixed embedding
            logits = self.model.forward_from_emb(mixed_emb)  # [B, 3, 6640]

            # Mixed focal loss
            labels_b = labels[perm]
            loss = lam * self._compute_loss(logits, labels) + (1.0 - lam) * self._compute_loss(logits, labels_b)
        else:
            logits = self.model(str_idx, self.string_embs)
            loss = self._compute_loss(logits, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["str_idx"], self.string_embs)
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
        all_preds = self.all_gather(preds_local)   # [world_size, N_local, 3, 6640]
        all_labels = self.all_gather(labels_local) # [world_size, N_local, 6640]
        ws = self.trainer.world_size
        if ws > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
            all_labels = all_labels.view(-1, N_GENES)
        else:
            # With ws=1 all_gather prepends a size-1 dim
            if all_preds.dim() == 4:
                all_preds = all_preds.squeeze(0)
            if all_labels.dim() == 3:
                all_labels = all_labels.squeeze(0)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        f1 = _compute_per_gene_f1(preds_np, labels_np)

        # Log as "val_f1" (underscore) to fix checkpoint filename substitution.
        # Slash notation "val/f1" causes Lightning to substitute 0.0000 in filenames.
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["str_idx"], self.string_embs)
        self._test_preds.append(logits.detach().cpu())
        self._test_labels.append(batch.get("label", torch.zeros(logits.size(0), N_GENES, dtype=torch.long)).detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> Dict[str, float]:
        """Save test predictions and compute test F1.

        FIXED (vs parent): Use all_gather to collect predictions from ALL ranks before
        computing F1. The parent's all_reduce(SUM)/ws approach was INCORRECT because
        F1 is a non-linear metric - average(F1_rank0, F1_rank1) != F1(all_predictions).

        This fix ensures the reported test_f1 in metrics.csv matches calc_metric.py.
        """
        if not self._test_preds:
            return {}

        preds_local = torch.cat(self._test_preds, dim=0)    # [N_local, 3, 6640]
        labels_local = torch.cat(self._test_labels, dim=0)  # [N_local, 6640]
        self._test_preds.clear()
        self._test_labels.clear()

        # Capture metadata before clearing
        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        n_local = len(local_pert_ids)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        # Save per-rank NPZ (DDP: each rank saves its local subset)
        preds_np_out = preds_local.float().cpu().numpy()
        out_path = Path(__file__).parent / "run" / f"test_predictions_rank{self.global_rank}.npz"
        np.savez_compressed(out_path, preds=preds_np_out, pert_ids=local_pert_ids, symbols=local_symbols)

        # FIX: Use all_gather to collect global predictions for accurate F1 computation
        # (parent's all_reduce(SUM)/ws was incorrect for non-linear metrics like F1)
        ws = self.trainer.world_size
        if ws > 1:
            # Gather predictions and labels from all ranks
            all_preds_gathered = self.all_gather(preds_local)    # [ws, N_local, 3, 6640]
            all_labels_gathered = self.all_gather(labels_local)  # [ws, N_local, 6640]
            all_preds_global = all_preds_gathered.view(-1, N_CLASSES, N_GENES)
            all_labels_global = all_labels_gathered.view(-1, N_GENES)
        else:
            all_preds_global = preds_local
            all_labels_global = labels_local
            if all_preds_global.dim() == 4:
                all_preds_global = all_preds_global.squeeze(0)
            if all_labels_global.dim() == 3:
                all_labels_global = all_labels_global.squeeze(0)

        preds_np = all_preds_global.float().cpu().numpy()
        labels_np = all_labels_global.cpu().numpy()
        avg_f1 = _compute_per_gene_f1(preds_np, labels_np)

        self.print(f"[Rank {self.global_rank}] Saved {n_local} local predictions -> {out_path}, global_f1={avg_f1:.4f}")

        # Write test_score.txt directly on rank 0
        if self.trainer.is_global_zero:
            score_path = Path(__file__).parent / "test_score.txt"
            score_path.write_text(json.dumps({"test_f1": avg_f1}, indent=2))
            self.print(f"Test results -> {score_path}")

        self.log("test_f1", avg_f1, prog_bar=False, logger=True, sync_dist=True)
        return {"test_f1": avg_f1}

    def configure_optimizers(self):
        """Configure optimizer: Muon for hidden MLP weight matrices, AdamW for everything else.

        LR schedule: CosineAnnealingWarmRestarts (T_0=80, T_mult=2, eta_min=5e-7).
        """
        if self.use_muon:
            try:
                from muon import MuonWithAuxAdam
                muon_available = True
            except ImportError:
                self.print("Warning: muon not installed, falling back to AdamW")
                muon_available = False
        else:
            muon_available = False

        if muon_available and self.use_muon:
            # Identify hidden MLP weight matrices (Linear weights in residual blocks only)
            hidden_weight_names = set()
            for name, param in self.model.named_parameters():
                if (param.ndim >= 2
                        and "blocks." in name
                        and ".weight" in name
                        and "norm" not in name):
                    hidden_weight_names.add(name)

            hidden_weights = []
            other_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                # Strip prefixes for model params:
                #   - LightningModule wraps PerturbMLP as self.model -> "model.xxx"
                #   - DDP wraps LightningModule -> "model.module.xxx"
                # hidden_weight_names are collected via self.model.named_parameters() (no prefix).
                model_name = name[len("model.module."):] if name.startswith("model.module.") else (
                    name[len("model."):] if name.startswith("model.") else name
                )
                if model_name in hidden_weight_names:
                    hidden_weights.append(param)
                else:
                    other_params.append(param)

            self.print(
                f"Muon params: {sum(p.numel() for p in hidden_weights):,} | "
                f"AdamW params: {sum(p.numel() for p in other_params):,}"
            )

            param_groups = [
                dict(
                    params=hidden_weights,
                    use_muon=True,
                    lr=self.muon_lr,       # 0.01 (proven stable)
                    weight_decay=self.weight_decay,
                    momentum=0.95,
                ),
                dict(
                    params=other_params,
                    use_muon=False,
                    lr=self.lr,
                    betas=(0.9, 0.95),
                    weight_decay=self.weight_decay,
                ),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            # Fallback: pure AdamW for all parameters
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        # CosineAnnealingWarmRestarts: periodic restarts to escape local optima
        # T_0=80: first cycle length = 80 epochs (restart at epoch 80)
        # T_mult=2: each subsequent cycle doubles in length
        #   -> cycle 2 = 160 epochs (80-240), cycle 3 = 320 epochs (240-560)
        # With max_epochs=400: covers epochs 240-400 (160 of cycle 3's 320 epochs)
        # With focal loss peaking around epoch 280-300, patience=100 stops at ~380-400
        # eta_min=5e-7: minimum LR per cycle
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cosine_t0,
            T_mult=self.cosine_t_mult,
            eta_min=self.cosine_eta_min,
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
    # Checkpoint: save only trainable params + small essential buffers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save trainable parameters and essential buffers, excluding large frozen buffers."""
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        saved: Dict[str, Any] = {}

        # Keys to always exclude (large frozen buffers)
        exclude_keys = {"string_embs"}
        # Keys that are trainable params or essential small buffers
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        essential_buffers = {
            name for name, buf in self.named_buffers()
            if name not in exclude_keys
        }
        keep_keys = trainable_keys | essential_buffers

        for key, val in full_sd.items():
            rel_key = key[len(prefix):] if key.startswith(prefix) else key
            if rel_key in keep_keys or any(k in rel_key for k in keep_keys):
                saved[key] = val

        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_saved = sum(v.numel() for v in saved.values())
        pct = f"{100*n_trainable/n_total:.1f}%" if n_total > 0 else "N/A"
        print(
            f"Saving checkpoint: {n_trainable:,}/{n_total:,} trainable params ({pct}), "
            f"{n_saved:,} total saved"
        )
        return saved

    def load_state_dict(self, state_dict, strict=True):
        # strict=False: string_embs is not in checkpoint but was populated by setup()
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-F1 averaged over all genes - matches calc_metric.py logic.

    preds:  [N, 3, 6640] float - class logits
    labels: [N, 6640]    int   - class indices in {0,1,2}
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N, 6640]
    n_genes = labels.shape[1]
    f1_vals: List[float] = []
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
    """Save test predictions in required TSV format (idx / input / prediction)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
        rows.append({
            "idx": pid,
            "input": sym,
            "prediction": json.dumps(preds[i].tolist()),  # [3][6640] list
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions -> {out_path}")


def _assemble_test_predictions_from_npz(out_path: Path) -> None:
    """Assemble per-rank NPZ prediction files into a single TSV.

    Each DDP rank saves its local predictions as NPZ during on_test_epoch_end.
    This function reads all rank NPZ files, deduplicates by pert_id, and
    writes the final test_predictions.tsv.
    """
    run_dir = out_path.parent

    # Collect all NPZ files
    npz_files = sorted(run_dir.glob("test_predictions_rank*.npz"))
    if not npz_files:
        print(f"[Assembly] No per-rank NPZ files found in {run_dir}")
        return

    all_pert_ids: List[str] = []
    all_symbols: List[str] = []
    all_preds: List[np.ndarray] = []

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        preds = data["preds"]  # [N_local, 3, 6640]
        pert_ids = data["pert_ids"].tolist()
        symbols = data["symbols"].tolist()
        all_preds.append(preds)
        all_pert_ids.extend(pert_ids)
        all_symbols.extend(symbols)

    # Concatenate predictions from all ranks
    combined_preds = np.concatenate(all_preds, axis=0)  # [N_total, 3, 6640]

    # De-duplicate by pert_id (keep first occurrence)
    seen: set = set()
    dedup_pert = []
    dedup_sym = []
    dedup_idx = []
    for i, pid in enumerate(all_pert_ids):
        if pid and pid not in seen:
            seen.add(pid)
            dedup_pert.append(pid)
            dedup_sym.append(all_symbols[i])
            dedup_idx.append(i)

    dedup_preds = combined_preds[dedup_idx]

    _save_test_predictions(
        pert_ids=dedup_pert,
        symbols=dedup_sym,
        preds=dedup_preds,
        out_path=out_path,
    )

    # Clean up per-rank NPZ files
    for npz_path in npz_files:
        npz_path.unlink()
    print(f"[Assembly] Cleaned up {len(npz_files)} per-rank NPZ files")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-3-2-2-1-1-1-1-1-1-1-1-1-1: STRING-only + Flat Head + hidden=384 + "
                    "Muon LR=0.01 + CosineWarmRestarts + Manifold Mixup + "
                    "CALIBRATED FOCAL LOSS (gamma=1.5, alpha=0.40) + "
                    "SINGLE BEST CHECKPOINT + FIXED DDP TEST F1"
    )
    p.add_argument("--micro-batch-size",      type=int,   default=32)
    p.add_argument("--global-batch-size",     type=int,   default=256)
    p.add_argument("--max-epochs",            type=int,   default=400,
                   help="Reduced 500->400: aligned with focal loss convergence speed")
    p.add_argument("--lr",                    type=float, default=3e-4)
    p.add_argument("--muon-lr",               type=float, default=0.01,
                   help="Muon optimizer LR (proven stable)")
    p.add_argument("--weight-decay",          type=float, default=8e-4,
                   help="Grandparent proven value: prevents premature convergence")
    p.add_argument("--hidden-dim",            type=int,   default=384)
    p.add_argument("--n-blocks",              type=int,   default=3)
    p.add_argument("--dropout",               type=float, default=0.28,
                   help="Trunk dropout")
    p.add_argument("--head-dropout",          type=float, default=0.18,
                   help="Grandparent proven value")
    p.add_argument("--focal-gamma",           type=float, default=1.5,
                   help="CALIBRATED: 2.0->1.5 (milder modulation for better local minimum)")
    p.add_argument("--class-weight-alpha",    type=float, default=0.40,
                   help="CALIBRATED: 0.30->0.40 (stronger minority class emphasis for focal loss)")
    p.add_argument("--cosine-t0",             type=int,   default=80,
                   help="T_0 for CosineAnnealingWarmRestarts (first cycle length)")
    p.add_argument("--cosine-t-mult",         type=int,   default=2,
                   help="T_mult for CosineAnnealingWarmRestarts")
    p.add_argument("--cosine-eta-min",        type=float, default=5e-7,
                   help="CosineAnnealingWarmRestarts minimum LR per cycle")
    p.add_argument("--early-stop-patience",   type=int,   default=100,
                   help="REDUCED 160->100: aligned with focal loss's faster convergence dynamics")
    p.add_argument("--grad-clip-norm",        type=float, default=1.0)
    p.add_argument("--no-muon",               action="store_true",
                   help="Disable Muon optimizer, fall back to AdamW for all params")
    p.add_argument("--save-top-k",            type=int,   default=3,
                   help="Save top-3 checkpoints by val_f1 (single-best evaluation)")
    p.add_argument("--mixup-alpha",           type=float, default=0.2,
                   help="Manifold Mixup Beta distribution parameter")
    p.add_argument("--mixup-prob",            type=float, default=0.5,
                   help="Grandparent proven value: balanced augmentation")
    p.add_argument("--num-workers",           type=int,   default=4)
    p.add_argument("--val-check-interval",    type=float, default=1.0)
    p.add_argument("--debug_max_step",        type=int,   default=None)
    p.add_argument("--fast_dev_run",          action="store_true")
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

    # --- DataModule ---
    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup("fit")

    # --- LightningModule ---
    model = PerturbModule(
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        lr=args.lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        class_weight_alpha=args.class_weight_alpha,
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_t_mult,
        cosine_eta_min=args.cosine_eta_min,
        max_epochs=args.max_epochs,
        grad_clip_norm=args.grad_clip_norm,
        use_muon=not args.no_muon,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
    )

    # --- Trainer configuration ---
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = 1.0
        max_steps = args.debug_max_step
    else:
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        max_steps = -1

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
    )
    # patience=100: REDUCED from parent's 160 (aligned with focal loss's faster convergence)
    # Focal loss peaked at epoch 266 in parent; patience=100 stops at ~366 (vs 426 in parent)
    # Saves ~60 epochs of wasted compute past the peak
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
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
        strategy=DDPStrategy(
            find_unused_parameters=True,  # Must be True: registered buffers (e.g., class_weights)
                                           # are used in _compute_loss but not traced in forward graph
            timeout=timedelta(seconds=120),
        ),
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
        gradient_clip_val=args.grad_clip_norm,
    )

    # --- Fit ---
    trainer.fit(model, datamodule=datamodule)

    # --- Test ---
    # SINGLE BEST CHECKPOINT only (inherited from parent, confirmed better than ensemble)
    if args.fast_dev_run or args.debug_max_step is not None:
        # Debug mode: standard Lightning test (all ranks, no ensemble)
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        # Production mode: SINGLE BEST CHECKPOINT only
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Assemble per-rank NPZ files into final TSV on rank 0
    if trainer.is_global_zero:
        _assemble_test_predictions_from_npz(output_dir / "test_predictions.tsv")

    # --- Save test score (rank 0 only) ---
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results -> {score_path}")


if __name__ == "__main__":
    main()
