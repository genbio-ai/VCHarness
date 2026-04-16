"""Node 3-2-1-1-1-1: Frozen STRING_GNN Multi-Scale (768-dim) + 3-Block Pre-Norm
             Residual MLP + Flat Output Head (512->19920) + RLROP(patience=8)
=============================================================================
Parent: node3-2-1-1-1 (STRING_GNN partial fine-tuning last 2 layers + 3-block +
        flat head + RLROP(patience=15), test F1=0.3960)

ROOT CAUSE FIX (from node3-2-1-1-1 feedback):

  1. ABANDON PARTIAL GNN FINE-TUNING: The parent's partial GNN fine-tuning is
     conclusively counterproductive on 1,273 samples. Every partial fine-tuning
     attempt in the tree achieves F1~0.39-0.40 vs frozen STRING achieving 0.474+.
     Revert to fully FROZEN STRING_GNN with precomputed embeddings for training
     stability and eliminate GNN forward pass overhead during training.

  2. MULTI-SCALE STRING FEATURES (768-dim): Instead of using only the final
     hidden state (layer 8, 256-dim), concatenate hidden states from layers 4,
     6, and 8 to form a 768-dim multi-scale embedding. Different layers capture:
     - Layer 4: Local PPI neighborhood structure (direct protein interactions)
     - Layer 6: Broader network motifs (pathway-level context)
     - Layer 8: Global graph topology (complex-level relationships)
     Node1-1-2 tried multi-scale (768-dim) with a FACTORIZED HEAD (bottleneck=256)
     and achieved F1=0.436. Combining multi-scale with the PROVEN FLAT HEAD
     (512->19920) has NEVER been tested and should push beyond 0.474.

  3. RLROP(patience=8): Match node1-1-1's proven recipe exactly. patience=8
     worked with only 2 halvings in node1-1-1 (F1=0.474) because frozen STRING
     produces stable training dynamics. The parent used patience=15 as a
     workaround for noisy GNN fine-tuning gradients — no longer needed.

Architecture:
  - Input : Frozen STRING_GNN multi-scale 768-dim PPI embeddings
            (precomputed once: layers 4+6+8 concatenated, all GNN frozen)
  - Projection : LayerNorm(768) -> Linear(768->512) -> GELU -> Dropout(0.35)
  - Trunk  : 3x pre-norm residual blocks  512 -> 1024 -> 512, Dropout(0.35)
  - Head   : LayerNorm(512) -> Linear(512 -> 6640*3) [FLAT, no bottleneck]
  - Bias   : per-gene additive bias (19,920 extra learnable parameters)

Training:
  - Loss      : weighted cross-entropy + label smoothing=0.05
                class weights: [down~3.4x, neutral~0.17x, up~6.7x] CORRECT order
  - Optimizer : AdamW (single group, all MLP params): lr=3e-4, wd=5e-4
  - Schedule  : ReduceLROnPlateau(patience=8, factor=0.5, threshold=1e-3)
                Matching node1-1-1's proven recipe exactly
  - Epochs    : 300, early-stop patience=30

Key differences from parent node3-2-1-1-1:
  - GNN FROZEN: Precomputed multi-scale embeddings, no GNN gradient updates
    Proven to be 0.06-0.08 F1 points better than partial fine-tuning on 1,273 samples
  - MULTI-SCALE FEATURES: 768-dim (layers 4+6+8) vs parent's 256-dim (layer 8 only)
    Richer biological signal from multi-resolution PPI topology representation
  - RLROP patience=8: Matches node1-1-1's proven recipe (2 halvings to F1=0.474)
    parent's patience=15 was a workaround for GNN fine-tuning instability

Expected outcome: >= 0.450 (abandoning partial GNN fine-tuning + richer multi-scale
                   features, targeting node1-1-1's proven 0.474 level)
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
GNN_DIM = 256         # STRING_GNN output embedding dimension per layer
MULTI_SCALE_DIM = 768 # multi-scale embedding: 3 GNN layers concatenated (3*256)
HIDDEN_DIM = 512      # MLP hidden dimension
INNER_DIM = 1024      # MLP inner (expansion) dimension

# Multi-scale layer indices (indices into hidden_states tuple)
# STRING_GNN has 8 GNN layers (0-indexed: 0-7). hidden_states tuple has 9 entries:
#   hidden_states[0] = initial embedding (before any GNN layer)
#   hidden_states[k] = after GNN layer k-1 (for k=1..8)
# We sample from 3 evenly-spaced points:
#   hidden_states[5] = after GNN layer 4 (mid-low)
#   hidden_states[7] = after GNN layer 6 (mid-high)
#   hidden_states[8] = after GNN layer 7 = last_hidden_state (final output)
MULTISCALE_HIDDEN_INDICES = [5, 7, 8]  # after GNN layers 4, 6, 7 (0-indexed)


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
# Model components
# ---------------------------------------------------------------------------
class PreNormResBlock(nn.Module):
    """Pre-LayerNorm residual block: LN -> Linear -> GELU -> Dropout -> Linear -> Dropout -> add."""

    def __init__(self, dim: int, inner_dim: int, dropout: float = 0.35) -> None:
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
        dropout: float = 0.35,
        lr: float = 3e-4,
        weight_decay: float = 5e-4,
        label_smoothing: float = 0.05,
        rlrop_patience: int = 8,
        rlrop_factor: float = 0.5,
        rlrop_threshold: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # These are populated in setup()
        self.input_proj: Optional[nn.Sequential] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None

        # STRING_GNN ID -> index mapping (populated in setup)
        self.gnn_id_to_idx: Dict[str, int] = {}

        # Accumulators for metrics
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """Build model with frozen STRING_GNN multi-scale (768-dim) precomputed embeddings.

        Multi-scale strategy: precompute embeddings from layers 4, 6, and 8 (768-dim total)
        at setup time. All STRING_GNN parameters are frozen — no GNN fine-tuning.
        This is the key change from the parent node which used partial GNN fine-tuning
        (conclusively counterproductive on 1,273 samples).
        """
        from transformers import AutoModel

        self.print("Loading STRING_GNN for frozen multi-scale embedding extraction ...")
        gnn_model = AutoModel.from_pretrained(
            STRING_GNN_DIR, trust_remote_code=True
        )

        # Freeze ALL GNN parameters — no fine-tuning
        for param in gnn_model.parameters():
            param.requires_grad = False

        gnn_total = sum(p.numel() for p in gnn_model.parameters())
        self.print(
            f"STRING_GNN: ALL {gnn_total:,} params FROZEN — "
            f"precomputing multi-scale embeddings (layers 4+6+8)"
        )

        # Move to device for embedding computation
        gnn_model = gnn_model.to(self.device)
        gnn_model = gnn_model.float()

        # Load graph data for embedding computation
        graph = torch.load(
            Path(STRING_GNN_DIR) / "graph_data.pt",
            map_location=self.device,
        )
        edge_index = graph["edge_index"].to(self.device)
        edge_weight = graph.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        # Precompute multi-scale embeddings: concatenate outputs from GNN layers 4, 6, 7
        # hidden_states tuple: index 0 = initial embedding, 1..8 = after GNN layers 0..7
        # We want: after layer 4 = index 5, after layer 6 = index 7, after layer 7 = index 8 (=last)
        self.print("Precomputing multi-scale STRING_GNN embeddings (GNN layers 4, 6, 7 => 768-dim) ...")
        with torch.no_grad():
            gnn_out = gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
                output_hidden_states=True,
            )

        # Verify hidden_states length (should be 9: initial + 8 GNN layers)
        n_hidden = len(gnn_out.hidden_states)
        self.print(f"STRING_GNN returned {n_hidden} hidden states (expected 9)")
        assert n_hidden >= 9, f"Expected >= 9 hidden states, got {n_hidden}"

        # Concatenate 3 scales: [N_nodes, 256] x3 -> [N_nodes, 768]
        layer4_emb = gnn_out.hidden_states[MULTISCALE_HIDDEN_INDICES[0]]   # after GNN layer 4 (index 5)
        layer6_emb = gnn_out.hidden_states[MULTISCALE_HIDDEN_INDICES[1]]   # after GNN layer 6 (index 7)
        layer7_emb = gnn_out.hidden_states[MULTISCALE_HIDDEN_INDICES[2]]   # after GNN layer 7 = last_hidden_state (index 8)

        # Verify shapes
        assert layer4_emb.shape == layer6_emb.shape == layer7_emb.shape, (
            f"Shape mismatch: {layer4_emb.shape}, {layer6_emb.shape}, {layer7_emb.shape}"
        )
        assert layer4_emb.shape[-1] == GNN_DIM, (
            f"Expected GNN_DIM={GNN_DIM}, got {layer4_emb.shape[-1]}"
        )

        multi_scale_emb = torch.cat([layer4_emb, layer6_emb, layer7_emb], dim=-1)  # [N_nodes, 768]
        assert multi_scale_emb.shape[-1] == MULTI_SCALE_DIM, (
            f"Expected MULTI_SCALE_DIM={MULTI_SCALE_DIM}, got {multi_scale_emb.shape[-1]}"
        )

        self.print(
            f"Multi-scale embeddings precomputed: shape={list(multi_scale_emb.shape)}, "
            f"dtype={multi_scale_emb.dtype}"
        )

        # Register as non-trainable buffer (moved to device automatically)
        self.register_buffer("_multi_scale_emb", multi_scale_emb.cpu())

        # Build ENSG-ID -> row-index mapping for the embedding table
        node_names: List[str] = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.gnn_id_to_idx = {name: i for i, name in enumerate(node_names)}
        self.print(
            f"STRING_GNN covers {len(self.gnn_id_to_idx)} Ensembl gene IDs; "
            f"multi-scale 768-dim feature buffer stored on CPU"
        )

        # Clean up GNN model and graph from GPU memory (no longer needed)
        del gnn_model, graph, edge_index, edge_weight, gnn_out
        del layer4_emb, layer6_emb, layer7_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- MLP architecture ----
        hp = self.hparams

        # Input projection: 768 -> 512 (multi-scale -> hidden_dim)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(MULTI_SCALE_DIM),
            nn.Linear(MULTI_SCALE_DIM, hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        self.blocks = nn.ModuleList(
            [
                PreNormResBlock(hp.hidden_dim, hp.inner_dim, hp.dropout)
                for _ in range(hp.n_blocks)
            ]
        )

        # FLAT output head: LayerNorm -> Linear(512 -> N_GENES*N_CLASSES)
        # This is the proven architecture from node1-1-1 (F1=0.474).
        # Multi-scale features (768-dim) provide richer input but the flat head
        # architecture remains identical to the proven recipe.
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )

        # Per-gene additive bias: one offset per (gene x class) pair
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # ---- Class weights ----
        # After label shift (-1->0, 0->1, 1->2):
        #   class 0 = down-regulated  (4.77%)  -> high weight
        #   class 1 = neutral         (92.82%) -> low weight
        #   class 2 = up-regulated    (2.41%)  -> highest weight
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq) / (1.0 / freq).mean()
        self.register_buffer("class_weights", class_weights)

        # Cast trainable params to float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.output_head.parameters())
        proj_params = sum(p.numel() for p in self.input_proj.parameters())
        self.print(f"Trainable params: {trainable:,} / {total:,}")
        self.print(
            f"  Input projection (768->512): {proj_params:,} params"
        )
        self.print(
            f"  Output head (flat 512->19920): {head_params:,} params "
            f"({100*head_params/trainable:.1f}% of trainable)"
        )

    # ------------------------------------------------------------------
    def _get_gene_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Look up multi-scale STRING_GNN embeddings (768-dim) for a batch of ENSG IDs.

        Since the GNN is fully frozen with precomputed embeddings, this is a simple
        lookup into the _multi_scale_emb buffer. Genes absent from STRING_GNN (~7%)
        receive a zero vector of dimension MULTI_SCALE_DIM.
        """
        emb_list: List[torch.Tensor] = []
        for pid in pert_ids:
            idx = self.gnn_id_to_idx.get(pid)
            if idx is not None:
                # Direct lookup from buffer (float32)
                emb_list.append(
                    self._multi_scale_emb[idx].to(self.device, dtype=torch.float32)
                )
            else:
                emb_list.append(
                    torch.zeros(MULTI_SCALE_DIM, device=self.device, dtype=torch.float32)
                )
        return torch.stack(emb_list, dim=0)  # [B, 768]

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits of shape [B, N_CLASSES, N_GENES]."""
        x = self._get_gene_emb(pert_ids)           # [B, 768]
        x = self.input_proj(x)                      # [B, 512]
        for block in self.blocks:
            x = block(x)                            # [B, 512]
        logits = self.output_head(x)                # [B, N_GENES*N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)  # [B, 3, 6640]
        # gene_bias: [N_GENES, N_CLASSES] -> .T -> [N_CLASSES, N_GENES] -> [1,3,6640]
        logits = logits + self.gene_bias.T.unsqueeze(0)
        return logits

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted CE + label smoothing on [B, N_CLASSES, N_GENES] logits."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["pert_id"])
        loss = self._compute_loss(logits, batch["label"])
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

        import torch.distributed as dist

        is_dist = dist.is_available() and dist.is_initialized()

        if is_dist and self.trainer.world_size > 1:
            world_size = dist.get_world_size()
            preds_np_local = preds_local.numpy()
            labels_np_local = labels_local.numpy()

            # Gather variable-length arrays from all ranks using all_gather_object
            obj_preds = [None] * world_size
            obj_labels = [None] * world_size
            dist.all_gather_object(obj_preds, preds_np_local)
            dist.all_gather_object(obj_labels, labels_np_local)

            # All ranks compute F1 on the global dataset.
            # Logging WITHOUT rank_zero_only so EarlyStopping can access the
            # metric on ALL ranks (avoids RuntimeError on non-zero ranks).
            preds_np = np.concatenate(obj_preds, axis=0)
            labels_np = np.concatenate(obj_labels, axis=0)
            f1 = _compute_per_gene_f1(preds_np, labels_np)
            self.log("val/f1", f1, prog_bar=True, sync_dist=True)
        else:
            preds_np = preds_local.numpy()
            labels_np = labels_local.numpy()
            f1 = _compute_per_gene_f1(preds_np, labels_np)
            self.log("val/f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        self._test_preds.append(logits.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        # Gather tensor predictions from all ranks.
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

            # De-duplicate
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            preds_np = all_preds.cpu().numpy()  # [N_total, 3, 6640]
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
        hp = self.hparams

        # Single parameter group — all MLP params (no GNN fine-tuning)
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )

        # ReduceLROnPlateau with patience=8 matching node1-1-1's proven recipe
        # (only 2 halvings needed when frozen STRING produces stable training dynamics)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=hp.rlrop_patience,
            factor=hp.rlrop_factor,
            threshold=hp.rlrop_threshold,
            min_lr=1e-6,
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
    # Checkpoint helpers (save only trainable params + buffers)
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
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
            f"Saving checkpoint: {trainable:,}/{total:,} params "
            f"({100 * trainable / total:.2f}%)"
        )
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute the per-gene macro-averaged F1 as defined in calc_metric.py.

    preds  : [N_samples, 3, N_genes]  -- logits / probabilities
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
        description="Node3-2-1-1-1-1: Frozen STRING_GNN Multi-Scale 768-dim + 3-Block MLP + Flat Head + RLROP"
    )
    p.add_argument("--micro_batch_size", type=int, default=64)
    p.add_argument("--global_batch_size", type=int, default=512)
    p.add_argument("--max_epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--inner_dim", type=int, default=1024)
    p.add_argument("--n_blocks", type=int, default=3)
    # ReduceLROnPlateau parameters — matching node1-1-1's proven recipe
    p.add_argument("--rlrop_patience", type=int, default=8)
    p.add_argument("--rlrop_factor", type=float, default=0.5)
    p.add_argument("--rlrop_threshold", type=float, default=1e-3)
    p.add_argument("--early_stop_patience", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=4)
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
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        rlrop_patience=args.rlrop_patience,
        rlrop_factor=args.rlrop_factor,
        rlrop_threshold=args.rlrop_threshold,
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
            find_unused_parameters=False,  # All params participate; precomputed embeddings as buffers
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
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(model, datamodule=datamodule)

    # ------------------------------------------------------------------
    # Test (use best checkpoint in production; raw model in debug mode)
    # ------------------------------------------------------------------
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results saved -> {score_path}")


if __name__ == "__main__":
    main()
