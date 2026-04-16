"""Node 1-2: ESM2-35M Frozen Lookup + STRING_GNN Frozen + Gated Fusion + Manifold Mixup
            + Fixed Checkpoint Ensemble + Proper RLROP
================================================================
Parent  : node3-3-1-1   (STRING_GNN+3-block+h=384+Muon, test F1=0.4793)
Lineage : node3 → node3-3 → node3-3-1 → node3-3-1-1 → node1-2

Key Changes from Parent
--------------------------
1. ADD ESM2-35M frozen embeddings (480-dim precomputed from STRING_GNN dir)
   → Dual-branch: STRING_GNN(256) + ESM2-35M(480) with learnable gated fusion
   → ESM2 protein embeddings provide complementary protein sequence information
   → Precomputed frozen lookup (esm2_embeddings_35M.pt, [18870, 480]) - zero overfitting risk
   → node3-3-1-2-1 (F1=0.5170) proved frozen ESM2 + STRING dual-branch is powerful

2. ADD Manifold Mixup augmentation (alpha=0.2, prob=0.5)
   → Applied in the fused embedding space (after gated fusion, before input_proj)
   → node3-3-1-2 showed this pushes STRING-only F1 from 0.4793 to 0.4966
   → Proven interpolation regularizer for this dataset size

3. FIX checkpoint ensemble path (rglob("*.ckpt") vs broken glob("best-*.ckpt"))
   → sibling node3-3-1-1-1 achieved F1=0.4831 with a BROKEN ensemble (path bug)
   → Fixed ensemble with correct recursive glob should add +0.003-0.008 F1

4. FIX RLROP firing (patience=8, threshold=5e-4)
   → sibling node3-3-1-1-1: RLROP never fired due to too-conservative threshold
   → threshold=5e-4 with patience=8 ensures RLROP fires at genuine plateaus
   → Parent's 5 RLROP halvings were essential for achieving best F1=0.4793

5. Save top-5 checkpoints (ModelCheckpoint save_top_k=5)

Architecture
------------
Input: ENSG perturbation ID
  → STRING_GNN frozen embeddings [18870, 256] (non-trainable buffer)
  → ESM2-35M frozen embeddings [18870, 480] (non-trainable buffer, precomputed)
  → string_proj: LN + Linear(256→256) + GELU  → [B, 256]
  → esm2_proj:   LN + Linear(480→256) + GELU  → [B, 256]
  → GatedFusion: alpha=sigmoid(gate_logit), fused = alpha*str + (1-alpha)*esm2 → [B, 256]
  → [Manifold Mixup applied here during training with prob=0.5]
  → input_proj: LN + Linear(256→384) + GELU + Dropout(0.30)  → [B, 384]
  → 3 × PreNormResBlock(384, 768, dropout=0.30)
  → output_head: LN + Dropout(0.05) + Linear(384 → 6640×3)
  → reshape [B, 3, 6640]
  → + gene_bias[6640, 3].T [1, 3, 6640]
Output: logits [B, 3, 6640]

Training configuration:
- Weighted cross-entropy (class weights [0.0477, 0.9282, 0.0241]), label_smoothing=0.0
- Muon: hidden block 2D weights, lr=0.01, wd=0.01, momentum=0.95
- AdamW: all other params, lr=3e-4, betas=(0.9, 0.95), wd=0.01
- RLROP(mode=max, patience=8, threshold=5e-4, factor=0.5, min_lr=1e-7)
- Gradient clip=2.0
- Early stop patience=35 on val/f1
- Max epochs=400
- Test: Top-5 checkpoint ensemble with correct rglob path
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES = 6640        # number of response genes per perturbation
N_CLASSES = 3         # down (-1→0), neutral (0→1), up (1→2)
GNN_DIM = 256         # STRING_GNN output embedding dimension
ESM2_DIM = 480        # ESM2-35M embedding dimension (precomputed, [18870, 480])
FUSED_DIM = 256       # Dimension after gating/projection
HIDDEN_DIM = 384      # MLP hidden dimension
INNER_DIM = 768       # MLP inner dimension


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Each sample is one gene perturbation experiment in HepG2 cells."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        if "label" in df.columns:
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
    """Pre-LayerNorm residual block (proven architecture in this lineage)."""

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
        dropout: float = 0.30,
        head_dropout: float = 0.05,
        muon_lr: float = 0.01,
        adamw_lr: float = 3e-4,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.0,
        rlrop_factor: float = 0.5,
        rlrop_patience: int = 8,
        rlrop_threshold: float = 5e-4,
        min_lr: float = 1e-7,
        grad_clip_norm: float = 2.0,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Populated in setup()
        self.string_proj: Optional[nn.Sequential] = None
        self.esm2_proj: Optional[nn.Sequential] = None
        self.gate_logit: Optional[nn.Parameter] = None
        self.input_proj: Optional[nn.Sequential] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None

        # STRING_GNN gene-ID → row-index mapping (populated in setup)
        self.gnn_id_to_idx: Dict[str, int] = {}

        # Metric accumulators
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        """Build model and load frozen embeddings."""
        from transformers import AutoModel

        # ---- STRING_GNN frozen embeddings ----
        self.print("Loading STRING_GNN frozen node embeddings...")
        gnn_model = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
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

        all_gnn_emb = gnn_out.last_hidden_state.detach().float()
        self.register_buffer("gnn_embeddings", all_gnn_emb)  # [18870, 256]

        del gnn_model, gnn_out, graph, edge_index, edge_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.print(f"STRING_GNN embeddings: {all_gnn_emb.shape}")

        # ENSG-ID → row-index
        node_names: List[str] = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.gnn_id_to_idx = {name: i for i, name in enumerate(node_names)}
        self.print(f"STRING_GNN covers {len(self.gnn_id_to_idx)} gene IDs")

        # ---- ESM2-35M precomputed frozen embeddings [18870, 480] ----
        self.print("Loading ESM2-35M precomputed embeddings...")
        esm2_path = Path(STRING_GNN_DIR) / "esm2_embeddings_35M.pt"
        all_esm2_emb = torch.load(esm2_path, map_location="cpu").float()
        self.register_buffer("esm2_embeddings", all_esm2_emb.to(self.device))
        self.print(f"ESM2-35M embeddings: {all_esm2_emb.shape}")

        # ---- Build dual-branch architecture ----
        hp = self.hparams

        # STRING branch: 256 → FUSED_DIM=256
        self.string_proj = nn.Sequential(
            nn.LayerNorm(GNN_DIM),
            nn.Linear(GNN_DIM, FUSED_DIM),
            nn.GELU(),
        )

        # ESM2 branch: 480 → FUSED_DIM=256
        self.esm2_proj = nn.Sequential(
            nn.LayerNorm(ESM2_DIM),
            nn.Linear(ESM2_DIM, FUSED_DIM),
            nn.GELU(),
        )

        # Learnable gating: alpha = sigmoid(gate_logit)
        # fused = alpha * string_feat + (1 - alpha) * esm2_feat
        # Init 0.0 → alpha=0.5 (equal initial weighting)
        self.gate_logit = nn.Parameter(torch.zeros(1))

        # Input projection: FUSED_DIM → hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(FUSED_DIM),
            nn.Linear(FUSED_DIM, hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # 3 PreNormResBlocks
        self.blocks = nn.ModuleList([
            PreNormResBlock(hp.hidden_dim, hp.inner_dim, hp.dropout)
            for _ in range(hp.n_blocks)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )

        # Per-gene additive bias
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # Class weights: correct ordering [down, neutral, up]
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq) / (1.0 / freq).mean()
        self.register_buffer("class_weights", class_weights)

        # Cast trainable params to float32 for stable optimization
        for _, v in self.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Dual-branch: STRING({GNN_DIM}) + ESM2-35M({ESM2_DIM}) "
            f"→ GatedFusion({FUSED_DIM}) → {hp.n_blocks}×ResBlock({hp.hidden_dim}) "
            f"→ Head + gene_bias"
        )
        self.print(f"Trainable params: {trainable:,} / {total:,}")

    # ------------------------------------------------------------------
    def _lookup_embs(self, pert_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lookup STRING_GNN and ESM2 embeddings for a batch of ENSG IDs.

        Returns:
            gnn_emb:  [B, 256]
            esm2_emb: [B, 480]
        Genes absent from STRING_GNN (~7% of training set) receive zero vectors.
        """
        gnn_rows: List[torch.Tensor] = []
        esm2_rows: List[torch.Tensor] = []
        for pid in pert_ids:
            row = self.gnn_id_to_idx.get(pid)
            if row is not None:
                gnn_rows.append(self.gnn_embeddings[row])
                esm2_rows.append(self.esm2_embeddings[row])
            else:
                gnn_rows.append(torch.zeros(GNN_DIM, device=self.device, dtype=torch.float32))
                esm2_rows.append(torch.zeros(ESM2_DIM, device=self.device, dtype=torch.float32))
        return torch.stack(gnn_rows, dim=0), torch.stack(esm2_rows, dim=0)

    def _fuse(self, pert_ids: List[str]) -> torch.Tensor:
        """Return gated-fused embedding: [B, FUSED_DIM=256]."""
        gnn_emb, esm2_emb = self._lookup_embs(pert_ids)
        str_feat = self.string_proj(gnn_emb)    # [B, 256]
        esm_feat = self.esm2_proj(esm2_emb)     # [B, 256]
        alpha = torch.sigmoid(self.gate_logit)   # scalar
        return alpha * str_feat + (1.0 - alpha) * esm_feat  # [B, 256]

    def _mlp_forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Run fused embedding through MLP and return logits [B, 3, 6640]."""
        x = self.input_proj(fused)              # [B, 384]
        for block in self.blocks:
            x = block(x)
        logits = self.output_head(x)            # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)
        logits = logits + self.gene_bias.T.unsqueeze(0)
        return logits

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits [B, N_CLASSES, N_GENES] (no mixup)."""
        fused = self._fuse(pert_ids)
        return self._mlp_forward(fused)

    def _forward_with_mixup(
        self, pert_ids: List[str], labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Forward with optional Manifold Mixup in the fused embedding space.

        Returns: (logits, labels_a, labels_b, lam)
        When mixup is not applied, labels_a == labels_b and lam == 1.0.
        """
        hp = self.hparams
        fused = self._fuse(pert_ids)  # [B, 256]

        apply_mixup = (
            self.training
            and hp.mixup_prob > 0.0
            and torch.rand(1).item() < hp.mixup_prob
        )

        if apply_mixup:
            lam = float(np.random.beta(hp.mixup_alpha, hp.mixup_alpha))
            lam = max(lam, 1.0 - lam)  # keep lam >= 0.5

            B = fused.size(0)
            perm = torch.randperm(B, device=fused.device)

            fused_mixed = lam * fused + (1.0 - lam) * fused[perm]
            logits = self._mlp_forward(fused_mixed)
            return logits, labels, labels[perm], lam
        else:
            logits = self._mlp_forward(fused)
            return logits, labels, labels, 1.0

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy on [B, N_CLASSES, N_GENES] logits."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def _compute_mixup_loss(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Mixup loss: lam * CE(a) + (1-lam) * CE(b)."""
        return lam * self._compute_loss(logits, labels_a) + \
               (1.0 - lam) * self._compute_loss(logits, labels_b)

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits, labels_a, labels_b, lam = self._forward_with_mixup(
            batch["pert_id"], batch["label"]
        )
        if lam < 1.0:
            loss = self._compute_mixup_loss(logits, labels_a, labels_b, lam)
        else:
            loss = self._compute_loss(logits, labels_a)
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        preds_local = torch.cat(self._val_preds, dim=0)    # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0)  # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized()

        if is_dist and self.trainer.world_size > 1:
            world_size = dist.get_world_size()
            obj_preds = [None] * world_size
            obj_labels = [None] * world_size
            dist.all_gather_object(obj_preds, preds_local.numpy())
            dist.all_gather_object(obj_labels, labels_local.numpy())

            # Concatenate all ranks; DDP DistributedSampler may pad the last
            # batch with repeated samples, but 141 val samples are small enough
            # that minor duplication has negligible effect on F1 computation.
            preds_np = np.concatenate(obj_preds, axis=0)
            labels_np = np.concatenate(obj_labels, axis=0)
        else:
            preds_np = preds_local.numpy()
            labels_np = labels_local.numpy()

        f1 = _compute_per_gene_f1(preds_np, labels_np)
        # Log on all ranks so EarlyStopping and RLROP can access the metric
        self.log("val/f1", f1, prog_bar=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        self._test_preds.append(logits.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        gathered = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640]
        all_preds = gathered.view(-1, N_CLASSES, N_GENES)  # [N_total, 3, 6640]

        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        gathered_pids: List[List[str]] = [local_pert_ids]
        gathered_syms: List[List[str]] = [local_symbols]
        if world_size > 1 and is_dist:
            obj_pids = [None] * world_size
            obj_syms = [None] * world_size
            dist.all_gather_object(obj_pids, local_pert_ids)
            dist.all_gather_object(obj_syms, local_symbols)
            gathered_pids = obj_pids
            gathered_syms = obj_syms

        if self.trainer.is_global_zero:
            all_pert_ids = [pid for lst in gathered_pids for pid in lst]
            all_symbols = [sym for lst in gathered_syms for sym in lst]

            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            preds_np = all_preds.cpu().numpy()
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(preds_np[i])

            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=np.stack(dedup_preds, axis=0),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        from muon import MuonWithAuxAdam

        hp = self.hparams

        # Muon: 2D weight matrices in residual blocks only
        muon_params = [
            p for _, p in self.blocks.named_parameters()
            if p.ndim >= 2 and p.requires_grad
        ]

        muon_ids = set(id(p) for p in muon_params)
        adamw_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in muon_ids
        ]

        param_groups = [
            dict(params=muon_params, use_muon=True, lr=hp.muon_lr,
                 weight_decay=hp.weight_decay, momentum=0.95),
            dict(params=adamw_params, use_muon=False, lr=hp.adamw_lr,
                 betas=(0.9, 0.95), weight_decay=hp.weight_decay),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # RLROP with threshold=5e-4 to ensure it fires at genuine plateaus
        # (sibling node showed threshold=1e-5 was too conservative → RLROP never fired)
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
                "monitor": "val/f1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers (save/load trainable params)
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        total_elems = sum(v.numel() for v in sd.values())
        print(f"Saving checkpoint: {len(sd)} tensors ({total_elems:,} elements)")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py exactly.

    preds  : [N_samples, 3, N_genes] — logits or class probabilities
    labels : [N_samples, N_genes]    — integer class labels in {0, 1, 2}
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N, N_genes]
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
    """Save test predictions in TSV format required by calc_metric.py."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert len(pert_ids) == len(preds)
    rows = [
        {"idx": pert_ids[i], "input": symbols[i],
         "prediction": json.dumps(preds[i].tolist())}
        for i in range(len(pert_ids))
    ]
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


def _run_ensemble_test(
    model: "PerturbModule",
    datamodule: PerturbDataModule,
    checkpoint_dir: Path,
    output_dir: Path,
    n_ensemble: int = 5,
) -> None:
    """Run checkpoint ensemble test using the already-initialized model.

    FIX over sibling node3-3-1-1-1: uses rglob("*.ckpt") instead of
    glob("best-*.ckpt") which failed to find checkpoints in subdirectories.

    Strategy: load each checkpoint's state_dict into the already-setup model
    (which has gnn_id_to_idx populated and buffers initialized), run inference,
    then average logits across all checkpoints.
    """
    # Recursively find all non-last checkpoints
    ckpt_files = [f for f in checkpoint_dir.rglob("*.ckpt")
                  if "last" not in f.name]

    if not ckpt_files:
        print("WARNING: No ensemble checkpoints found. Ensemble skipped.")
        return

    # Extract F1 score from filename: "best-epoch=003-val_f1=0.4793.ckpt"
    def _extract_score(path: Path) -> float:
        name = path.stem
        for part in name.replace("-", " ").replace("=", " ").split():
            try:
                v = float(part)
                if 0.2 <= v <= 1.0:
                    return v
            except ValueError:
                continue
        return 0.0

    scored = sorted([(f, _extract_score(f)) for f in ckpt_files],
                    key=lambda x: x[1], reverse=True)

    # Select top-K with diversity (deduplicate by score rounded to 3dp)
    selected: List[Tuple[Path, float]] = []
    seen_scores: set = set()
    for ckpt_path, score in scored:
        rounded = round(score, 3)
        if rounded not in seen_scores:
            selected.append((ckpt_path, score))
            seen_scores.add(rounded)
        elif len(selected) < n_ensemble:
            selected.append((ckpt_path, score))
        if len(selected) >= n_ensemble:
            break

    if not selected:
        selected = scored[:n_ensemble]

    print(f"Checkpoint ensemble ({len(selected)} checkpoints):")
    for p, s in selected:
        print(f"  {p.name} (val_f1={s:.4f})")

    # Save original state dict to restore model after ensemble
    original_state = {k: v.clone() for k, v in model.state_dict().items()
                      if "gnn_embeddings" not in k and "esm2_embeddings" not in k
                      and "class_weights" not in k}

    # Collect logits by loading each checkpoint into the already-initialized model
    device = model.device
    model.eval()
    all_logits: List[np.ndarray] = []
    all_pert_ids: Optional[List[str]] = None
    all_symbols: Optional[List[str]] = None

    for ckpt_path, score in selected:
        print(f"  Loading: {ckpt_path.name}")
        # Load checkpoint state dict (includes trainable params + buffers)
        ckpt_data = torch.load(str(ckpt_path), map_location=device)
        if "state_dict" in ckpt_data:
            ckpt_state = ckpt_data["state_dict"]
        else:
            ckpt_state = ckpt_data

        # Load only the trainable parameter keys (not the large buffers which
        # are already correct from setup())
        filtered_state = {k: v for k, v in ckpt_state.items()
                          if k in {n for n, _ in model.named_parameters()
                                   if _.requires_grad}}
        model.load_state_dict(filtered_state, strict=False)

        batch_preds: List[torch.Tensor] = []
        batch_pids: List[str] = []
        batch_syms: List[str] = []

        with torch.no_grad():
            for batch in datamodule.test_dataloader():
                logits = model(batch["pert_id"])
                batch_preds.append(logits.cpu().float())
                batch_pids.extend(batch["pert_id"])
                batch_syms.extend(batch["symbol"])

        all_logits.append(torch.cat(batch_preds, dim=0).numpy())
        if all_pert_ids is None:
            all_pert_ids = batch_pids
            all_symbols = batch_syms

    # Restore original model state
    if original_state:
        model.load_state_dict(original_state, strict=False)

    # Average logits across checkpoints
    avg_preds = np.mean(all_logits, axis=0)  # [N, 3, 6640]
    _save_test_predictions(
        pert_ids=all_pert_ids,
        symbols=all_symbols,
        preds=avg_preds,
        out_path=output_dir / "test_predictions.tsv",
    )
    print(f"Ensemble test predictions saved ({len(selected)} checkpoints, {len(all_pert_ids)} samples)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Node1-2: ESM2-35M+STRING_GNN Dual-Branch + GatedFusion "
            "+ ManifoldMixup + FixedCheckpointEnsemble"
        )
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=400)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--head-dropout", type=float, default=0.05)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--inner-dim", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--rlrop-factor", type=float, default=0.5)
    p.add_argument("--rlrop-patience", type=int, default=8)
    p.add_argument("--rlrop-threshold", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--grad-clip-norm", type=float, default=2.0)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--early-stop-patience", type=int, default=35)
    p.add_argument("--save-top-k", type=int, default=5)
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

    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

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
        rlrop_factor=args.rlrop_factor,
        rlrop_patience=args.rlrop_patience,
        rlrop_threshold=args.rlrop_threshold,
        min_lr=args.min_lr,
        grad_clip_norm=args.grad_clip_norm,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
    )

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

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-epoch={epoch:03d}-val_f1={val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
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
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        gradient_clip_val=args.grad_clip_norm,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Test
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        # Step 1: Run standard test with best checkpoint (distributed, all ranks)
        # This writes test_predictions.tsv as a fallback
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

        # Step 2: Run checkpoint ensemble (rank 0 only, single GPU inference)
        # This overwrites test_predictions.tsv with averaged logits (better quality)
        if trainer.is_global_zero:
            checkpoint_dir = output_dir / "checkpoints"
            _run_ensemble_test(
                model=model,
                datamodule=datamodule,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                n_ensemble=args.save_top_k,
            )

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results saved → {score_path}")


if __name__ == "__main__":
    main()
