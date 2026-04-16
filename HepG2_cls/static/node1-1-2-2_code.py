"""Node 1-1-2-2: STRING_GNN Final-Layer + PreLN MLP (hidden=384) + Head Dropout + Muon+AdamW

Architecture:
  - Input: perturbed gene identifier (ENSEMBL ID)
  - STRING_GNN pretrained backbone (frozen): final layer ONLY (256-dim last_hidden_state)
  - Input projection: Linear(256->384) + LayerNorm + GELU
  - 3x PreLN ResidualBlock (384-dim hidden, 768-dim intermediate), dropout=0.35
  - Output head: Dropout(0.15) + LayerNorm(384) + Linear(384->19920)
  - Per-gene learnable bias [19920] for baseline calibration
  - Loss: weighted cross-entropy + label_smoothing=0.05 (pure, no focal)
  - Optimizer: MuonWithAuxAdam
      * Muon (lr=0.02, wd=8e-4) for all 2D weight matrices in trunk
        (input_proj Linear + residual block Linears)
      * AdamW (lr=3e-4, wd=5e-4) for biases, norms, head, per_gene_bias, fallback_emb
  - LR schedule: ReduceLROnPlateau (patience=8, factor=0.5, mode=max) on val/f1

Key improvements vs parent (node1-1-2, F1=0.4357):
  1. REVERT multi-scale (768-dim concat layers 4+6+8) -> final layer only (256-dim)
     Double bottleneck (768->512 + 512->256 factorized) caused severe underfitting.
  2. REVERT factorized head (512->256->19920) -> flat head (384->19920)
     Factorized head has consistently failed across 3+ independent experiments in the tree.
  3. REVERT hybrid focal+WCE loss -> pure WCE + label_smoothing=0.05
     Hybrid loss kept train/loss=0.514 (42x above node1-1-1's 0.012) confirming underfitting.
  4. INTRODUCE Muon+AdamW dual optimizer
     node1-3-2 (F1=0.4756) first broke through 0.474 ceiling using Muon+AdamW+hidden=384.
  5. REDUCE hidden_dim 512->384
     384 provides better generalization than 512 on 1,273 samples (17.6x worse train fit
     but better test F1). node1-1-3 (hidden=640, 4 blocks) confirmed wider=worse: F1=0.430.
  6. ADD head_dropout=0.15 before final Linear
     node1-3-2-2-1 (F1=0.4777) showed targeted output-head dropout consistently improves
     vs no head dropout (F1=0.4756), providing regularization exactly where it's needed most.
  7. SWITCH PostLN -> PreLN residual blocks
     PreLN provides better gradient flow; validated across node1-3 lineage with Muon.

Key improvements vs sibling (node1-1-2-1, F1=0.4288):
  1. REPLACE cosine T_max=200 (warmup+single-cycle) -> ReduceLROnPlateau
     Cosine caused train/loss=0.941 at best checkpoint (underfitting). RLROP is proven
     reactive: fires only when model truly plateaus on validation signal.
  2. ADD Muon+AdamW optimizer (not AdamW-only)
  3. REDUCE hidden_dim 512->384
  4. ADD head_dropout=0.15
  5. PreLN blocks (not PostLN)
  6. REMOVE 5-epoch warmup (warmup caused 30% val/F1 collapse in sibling, confirmed harmful)

Memory provenance:
  - node1-3-2 (F1=0.4756): Muon+AdamW+hidden=384+RLROP first exceeded 0.474 ceiling
  - node1-3-2-2-1 (F1=0.4777): head_dropout=0.15 pushed further above 0.476 ceiling
  - node1-1-1 (F1=0.474): proven ReduceLROnPlateau baseline for STRING-only
  - node1-1-2 (parent, F1=0.4357): multi-scale+factorized+hybrid = underfitting cascade
  - node1-1-2-1 (sibling, F1=0.4288): cosine T_max=200 = aggressive LR decay failure
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640        # number of response genes
N_CLASSES = 3         # {-1->0, 0->1, 1->2}
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
STRING_EMB_DIM = 256  # STRING_GNN final-layer output embedding dimension


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Dataset for gene perturbation -> differential expression prediction."""

    def __init__(
        self,
        df: pd.DataFrame,
        node_name2idx: Dict[str, int],
        n_nodes: int,
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        gnn_indices = [node_name2idx.get(pid, -1) for pid in self.pert_ids]
        self.gnn_indices = torch.tensor(gnn_indices, dtype=torch.long)

        if "label" in df.columns:
            labels = np.array([json.loads(x) for x in df["label"].tolist()], dtype=np.int64)
            self.labels = torch.tensor(labels + 1, dtype=torch.long)  # {-1,0,1} -> {0,1,2}
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
        self.n_nodes: int = 0
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        if not self.node_name2idx:
            node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
            self.node_name2idx = {name: i for i, name in enumerate(node_names)}
            self.n_nodes = len(node_names)

        self.train_ds = PerturbDataset(train_df, self.node_name2idx, self.n_nodes)
        self.val_ds = PerturbDataset(val_df, self.node_name2idx, self.n_nodes)
        self.test_ds = PerturbDataset(test_df, self.node_name2idx, self.n_nodes)

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
class PreLNResidualBlock(nn.Module):
    """Pre-LayerNorm residual block (pre-normalization variant).

    Applies LayerNorm BEFORE the transformation for better gradient flow.
    Structure: x + FF(LN(x))  where FF = Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.pre_norm(x))


class StringGNNMLP(nn.Module):
    """STRING_GNN final-layer embedding -> PreLN MLP -> flat output head.

    Uses only the STRING_GNN final layer (256-dim), avoiding multi-scale
    concatenation which dilutes the discriminative signal from the final layer.
    The flat output head (384->19920) with head dropout provides targeted
    regularization of the dominant parameter source.
    """

    def __init__(
        self,
        input_dim: int = STRING_EMB_DIM,   # 256
        hidden_dim: int = 384,
        n_blocks: int = 3,
        n_genes: int = N_GENES,
        n_classes: int = N_CLASSES,
        dropout: float = 0.35,
        head_dropout: float = 0.15,
    ) -> None:
        super().__init__()

        # Fallback embedding for genes not in STRING_GNN (OOV)
        self.fallback_emb = nn.Embedding(1, input_dim)
        nn.init.normal_(self.fallback_emb.weight, std=0.01)

        # Input projection: 256 -> 384
        # Note: input_proj.0.weight (ndim=2, not "head", not "fallback_emb") -> Muon
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # PreLN residual blocks
        # Note: blocks[i].net.0.weight and blocks[i].net.3.weight (ndim=2) -> Muon
        self.blocks = nn.ModuleList([
            PreLNResidualBlock(hidden_dim, hidden_dim * 2, dropout)
            for _ in range(n_blocks)
        ])

        # Output head: Dropout(0.15) -> LayerNorm -> Linear(384->19920)
        # Note: head.2.weight contains "head" -> AdamW (output classifier)
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_genes * n_classes),
        )

        # Per-gene learnable bias: 1D parameter -> AdamW
        # Initialized to zero (no effect at start); learns gene-specific baselines
        self.per_gene_bias = nn.Parameter(
            torch.zeros(n_genes * n_classes), requires_grad=True
        )

        self.n_genes = n_genes
        self.n_classes = n_classes

    def forward(
        self,
        gnn_emb: torch.Tensor,  # [N_nodes, 256] - precomputed STRING_GNN final-layer output
        gnn_idx: torch.Tensor,  # [B] - index into gnn_emb, or -1 for OOV
    ) -> torch.Tensor:
        B = gnn_idx.size(0)
        device = gnn_idx.device

        # Build per-sample embeddings
        emb = torch.zeros(B, gnn_emb.size(1), device=device, dtype=gnn_emb.dtype)

        in_mask = gnn_idx >= 0
        not_in_mask = ~in_mask

        if in_mask.any():
            emb[in_mask] = gnn_emb[gnn_idx[in_mask]]

        if not_in_mask.any():
            fallback = self.fallback_emb(
                torch.zeros(not_in_mask.sum(), device=device, dtype=torch.long)
            )
            emb[not_in_mask] = fallback.to(emb.dtype)

        # Forward through MLP
        x = self.input_proj(emb)
        for block in self.blocks:
            x = block(x)

        logits = self.head(x)  # [B, n_genes * n_classes]
        logits = logits + self.per_gene_bias.to(logits.dtype)

        return logits.view(B, self.n_classes, self.n_genes)  # [B, 3, 6640]


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.35,
        head_dropout: float = 0.15,
        lr_muon: float = 0.02,
        lr_adamw: float = 3e-4,
        weight_decay_muon: float = 8e-4,
        weight_decay_adamw: float = 5e-4,
        label_smoothing: float = 0.05,
        lr_patience: int = 8,
        lr_factor: float = 0.5,
        max_epochs: int = 200,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.lr_muon = lr_muon
        self.lr_adamw = lr_adamw
        self.weight_decay_muon = weight_decay_muon
        self.weight_decay_adamw = weight_decay_adamw
        self.label_smoothing = label_smoothing
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.max_epochs = max_epochs

        self.model: Optional[StringGNNMLP] = None
        self.gnn_model = None       # STRING_GNN backbone (frozen)

        # Cached final-layer STRING_GNN embeddings [n_nodes, 256]
        self._gnn_emb_cache: Optional[torch.Tensor] = None

        # Buffers for validation and test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        from transformers import AutoModel

        # Class weights: inversely proportional to class frequency
        # After label shift: original -1 -> class0 (down, 4.77%), 0 -> class1 (neutral, 92.82%), 1 -> class2 (up, 2.41%)
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = 1.0 / freq
        class_weights = class_weights / class_weights.sum() * N_CLASSES  # normalize to mean=1
        self.register_buffer("class_weights", class_weights)

        # Load STRING_GNN backbone (frozen) — once per setup
        if self.gnn_model is None:
            self.gnn_model = AutoModel.from_pretrained(
                str(STRING_GNN_DIR), trust_remote_code=True
            )
            self.gnn_model.eval()
            for param in self.gnn_model.parameters():
                param.requires_grad = False
            self.print(
                f"STRING_GNN loaded (frozen): "
                f"{sum(p.numel() for p in self.gnn_model.parameters()):,} params"
            )

        # Initialize the MLP model — once per setup
        if self.model is None:
            self.model = StringGNNMLP(
                input_dim=STRING_EMB_DIM,
                hidden_dim=self.hidden_dim,
                n_blocks=self.n_blocks,
                dropout=self.dropout,
                head_dropout=self.head_dropout,
            )
            # Cast trainable parameters to float32 for stable optimization
            for v in self.model.parameters():
                if v.requires_grad:
                    v.data = v.data.float()

            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print(
                f"StringGNNMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
                f"head_dropout={self.head_dropout} | trainable_params={total_params:,}"
            )

    def _get_gnn_embeddings(self) -> torch.Tensor:
        """Get cached STRING_GNN final-layer node embeddings.

        Uses only last_hidden_state (final layer, 256-dim).
        Computed once per device and cached for efficiency.
        """
        if self._gnn_emb_cache is not None:
            return self._gnn_emb_cache

        device = next(self.model.parameters()).device
        self.gnn_model = self.gnn_model.to(device)

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location=device)
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)

        with torch.no_grad():
            outputs = self.gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

        # Final-layer embeddings only: [N_nodes, 256]
        gnn_emb = outputs.last_hidden_state.float().detach()
        self._gnn_emb_cache = gnn_emb
        self.print(f"STRING_GNN final-layer cache computed: shape={gnn_emb.shape}")
        return self._gnn_emb_cache

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Pure weighted cross-entropy with label smoothing.

        logits: [B, 3, 6640]
        labels: [B, 6640]  values in {0,1,2}
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        gnn_emb = self._get_gnn_embeddings()
        logits = self.model(gnn_emb, batch["gnn_idx"])
        loss = self._compute_loss(logits, batch["label"])
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
        preds_local = torch.cat(self._val_preds, dim=0)    # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0)  # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather from all DDP ranks
        all_preds = self.all_gather(preds_local)    # [ws, N_local, 3, 6640]
        all_labels = self.all_gather(labels_local)  # [ws, N_local, 6640]

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

    def on_test_epoch_end(self) -> Dict[str, float]:
        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        # Gather predictions from all DDP ranks
        all_preds = self.all_gather(preds_local)  # [ws, N_local, 3, 6640]
        ws = self.trainer.world_size
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        # Gather pert_ids and symbols from all ranks
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

        # Compute local F1 per rank and aggregate
        test_f1 = 0.0
        if self._test_labels:
            labels_local = torch.cat(self._test_labels, dim=0)
            local_f1 = _compute_per_gene_f1(
                preds_local.float().cpu().numpy(),
                labels_local.cpu().numpy()
            )
            # Average F1 across ranks for a stable estimate
            test_f1 = self.trainer.strategy.reduce(
                torch.tensor(local_f1, device=self.device), reduce_op="mean"
            )
            test_f1 = float(test_f1)
        self._test_labels.clear()

        # Save predictions and score on rank 0
        if self.trainer.is_global_zero:
            if all_preds.size(0) == 0:
                self.print("Warning: No test predictions gathered. Skipping save.")
            else:
                preds_np = all_preds.float().cpu().numpy()

                # Deduplicate by pert_id (DistributedSampler padding removal)
                seen = set()
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
                        f"Deduplication: {n_unique}/{len(keep_mask)} unique samples "
                        f"after removing DistributedSampler padding."
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

                # Save test score
                score_path = Path(__file__).parent / "test_score.txt"
                score_path.write_text(f'{{"test_f1": {test_f1:.6f}}}\n')
                self.print(f"Test F1: {test_f1:.6f} -> {score_path}")

        return {"test_f1": test_f1}

    def configure_optimizers(self):
        """Configure MuonWithAuxAdam + ReduceLROnPlateau.

        Muon is applied to all 2D weight matrices in the trunk (input_proj + residual blocks),
        which benefits most from the orthogonal gradient updates.
        AdamW handles the output head, norms, biases, per_gene_bias, and fallback_emb.
        ReduceLROnPlateau reduces ALL param groups' LRs when val/f1 plateaus.
        """
        from muon import MuonWithAuxAdam

        # Separate parameters:
        # Muon: 2D+ weight matrices in trunk (input_proj, blocks) - NOT head, NOT embedding
        # AdamW: 1D params (norms, biases), head Linear weight, fallback_emb, per_gene_bias
        muon_params = []
        adamw_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_trunk_weight = (
                param.ndim >= 2
                and "head" not in name
                and "fallback_emb" not in name
            )
            if is_trunk_weight:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        n_muon = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_params)
        self.print(
            f"Optimizer split: Muon={n_muon:,} params (2D trunk), "
            f"AdamW={n_adamw:,} params (1D/head/emb)"
        )

        param_groups = [
            dict(
                params=muon_params,
                use_muon=True,
                lr=self.lr_muon,
                weight_decay=self.weight_decay_muon,
                momentum=0.95,
            ),
            dict(
                params=adamw_params,
                use_muon=False,
                lr=self.lr_adamw,
                betas=(0.9, 0.95),
                weight_decay=self.weight_decay_adamw,
            ),
        ]
        optimizer = MuonWithAuxAdam(param_groups)

        # ReduceLROnPlateau: reactive LR reduction when val/f1 plateaus
        # Reduces ALL param groups by factor. Proven superior to cosine for this dataset.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",         # maximize val/f1
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=1e-6,
            threshold=1e-4,
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
        """Save only trainable parameters + buffers (skip frozen STRING_GNN backbone)."""
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
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

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        bufs = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable}/{total} trainable params "
            f"({100 * trainable / total:.2f}%), plus {bufs} buffer values"
        )
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable parameters from a partial checkpoint (strict=False)."""
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute per-gene macro F1 averaged over genes (matches calc_metric.py logic).

    preds:  [N, 3, 6640] float -- class logits
    labels: [N, 6640]    int   -- class indices in {0,1,2}
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
    """Save test predictions in required TSV format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
        pred_list = preds[i].tolist()  # [3][6640]
        rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-1-2-2: STRING_GNN Final-Layer + PreLN MLP + Muon+AdamW"
    )
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr-muon", type=float, default=0.02)
    p.add_argument("--lr-adamw", type=float, default=3e-4)
    p.add_argument("--weight-decay-muon", type=float, default=8e-4)
    p.add_argument("--weight-decay-adamw", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--head-dropout", type=float, default=0.15)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lr-patience", type=int, default=8)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--early-stop-patience", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


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

    # --- Model ---
    model = PerturbModule(
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        lr_muon=args.lr_muon,
        lr_adamw=args.lr_adamw,
        weight_decay_muon=args.weight_decay_muon,
        weight_decay_adamw=args.weight_decay_adamw,
        label_smoothing=args.label_smoothing,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        max_epochs=args.max_epochs,
    )

    # --- Trainer config ---
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if fast_dev_run:
        limit_train = None
        limit_val = None
        limit_test = None
        max_steps = -1
    elif args.debug_max_step is not None:
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
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=1.0 if (args.debug_max_step is None and not fast_dev_run) else 1.0,
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
        trainer.test(model, datamodule=datamodule)
    else:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
