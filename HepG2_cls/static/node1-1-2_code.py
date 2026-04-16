"""Node 1-2: Multi-Scale STRING_GNN + Factorized Head + Hybrid Loss.

Architecture:
  - Input: perturbed gene identifier (ENSEMBL ID)
  - STRING_GNN pretrained backbone (frozen): run once to extract multi-scale embeddings
    from intermediate layers (hidden_states[4], hidden_states[6], hidden_states[8])
  - Multi-scale compression: concat 3x256=768-dim -> Linear(768->512) + LayerNorm + GELU
  - Deep residual MLP: 3 blocks, 512-dim hidden, 1024-dim intermediate, dropout=0.35
  - Factorized output head: Linear(512->256) -> GELU -> Linear(256->19920)
    (50% fewer output head parameters vs flat 512->19920: 5.2M vs 10.2M)
  - Per-gene learnable bias [19920] for baseline differential expression calibration
  - Loss: hybrid focal (gamma=1.5) + weighted CE with label_smoothing=0.05 (50/50 mix)
  - LR schedule: ReduceLROnPlateau (patience=8, factor=0.5, min_lr=1e-6)

Key innovations vs node1-1 (parent) and node1-1-1 (sibling):
  1. Multi-scale STRING_GNN features (concat layers 4,6,8) instead of final layer only
     -> richer 768-dim biological representation capturing different levels of PPI topology
  2. Factorized output head with bottleneck=256 reduces output params from 10.2M to 5.2M
     -> directly addresses the dominant overfitting source identified in sibling's feedback
  3. Hybrid loss: focal (gamma=1.5) + weighted CE with label_smoothing=0.05
     -> better calibration than pure focal loss, label smoothing reduces overconfident minority class predictions
  4. Aggressive weight decay (1e-3 vs parent's 5e-4)
     -> combined with factorized head, provides stronger regularization for 1,273 samples
  5. Fully frozen STRING_GNN (no partial fine-tuning) with multi-scale precomputed caching
     -> more stable than partial fine-tuning while extracting more information
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640          # number of response genes
N_CLASSES = 3           # {-1->0, 0->1, 1->2}
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
STRING_EMB_DIM = 256    # STRING_GNN output embedding dimension per layer
# Multi-scale: concat layers 4, 6, 8 -> 3 * 256 = 768
MULTI_SCALE_DIM = 3 * STRING_EMB_DIM  # 768

# Hidden state indices to extract for multi-scale features
# hidden_states[0] = initial embedding, [1..8] = after each GNN layer
# We use layers 4, 6, 8 for low/mid/high-level PPI features
MULTI_SCALE_LAYER_INDICES = [4, 6, 8]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Dataset for gene perturbation -> differential expression prediction."""

    def __init__(
        self,
        df: pd.DataFrame,
        node_name2idx: Dict[str, int],  # ENSEMBL ID -> STRING_GNN node index (-1 if not found)
        n_nodes: int,  # total number of STRING_GNN nodes
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
            "gnn_idx": self.gnn_indices[idx],  # STRING_GNN node index or -1
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

        # Build STRING_GNN node name to index mapping
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
# Hybrid Loss
# ---------------------------------------------------------------------------
def focal_loss_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 1.5,
) -> torch.Tensor:
    """Multi-class focal loss with class weights.

    logits:  [N, C] float
    targets: [N]    long  values in {0,...,C-1}
    class_weights: [C] float
    gamma: focusing parameter (0 = standard CE)
    """
    log_softmax = F.log_softmax(logits, dim=-1)   # [N, C]
    softmax = log_softmax.exp()                     # [N, C]

    log_pt = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
    pt = softmax.gather(1, targets.unsqueeze(1)).squeeze(1)           # [N]

    focal_weight = (1 - pt) ** gamma                # [N]
    sample_weight = class_weights[targets]          # [N]

    loss = -focal_weight * sample_weight * log_pt  # [N]
    return loss.mean()


def hybrid_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    focal_gamma: float = 1.5,
    label_smoothing: float = 0.05,
    focal_weight: float = 0.5,
) -> torch.Tensor:
    """Hybrid loss: weighted combination of focal loss and weighted CE with label smoothing.

    logits:  [N, C] float
    targets: [N]    long  values in {0,...,C-1}
    class_weights: [C] float
    """
    # Focal loss component
    fl = focal_loss_fn(logits, targets, class_weights, gamma=focal_gamma)

    # Weighted CE with label smoothing
    # F.cross_entropy supports label_smoothing natively in PyTorch >= 1.10
    ce = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="mean",
    )

    return focal_weight * fl + (1.0 - focal_weight) * ce


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """MLP residual block with layer norm."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class MultiScaleStringGNNMLP(nn.Module):
    """Multi-scale PPI-graph-informed MLP for perturbation response prediction.

    Extracts STRING_GNN intermediate layer embeddings (layers 4, 6, 8),
    concatenates to form a 768-dim multi-scale representation, and maps through
    a residual MLP with a factorized output head to predict 6640 x 3 class logits.
    """

    def __init__(
        self,
        n_nodes: int,
        multi_scale_input_dim: int = MULTI_SCALE_DIM,  # 3 * 256 = 768
        hidden_dim: int = 512,
        n_blocks: int = 3,
        n_genes: int = N_GENES,
        n_classes: int = N_CLASSES,
        dropout: float = 0.35,
        output_bottleneck: int = 256,  # factorized output head bottleneck
    ) -> None:
        super().__init__()

        # Fallback embedding for genes not in STRING_GNN (used for all scale levels)
        # Single 768-dim fallback to match multi-scale dimension
        self.fallback_emb = nn.Embedding(1, multi_scale_input_dim)
        nn.init.normal_(self.fallback_emb.weight, std=0.01)

        # Multi-scale compression: 768 -> hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(multi_scale_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 2, dropout)
            for _ in range(n_blocks)
        ])

        # Factorized output head: hidden_dim -> output_bottleneck -> n_genes * n_classes
        # This reduces params from (512 * 19920) = 10.2M to
        # (512 * 256 + 256 * 19920) = 0.13M + 5.1M = 5.24M
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_bottleneck),
            nn.GELU(),
            nn.Linear(output_bottleneck, n_genes * n_classes),
        )

        # Per-gene learnable bias: captures baseline differential expression per gene
        # Initialized to zero (no-op at start)
        self.per_gene_bias = nn.Parameter(
            torch.zeros(n_genes * n_classes), requires_grad=True
        )

        self.n_genes = n_genes
        self.n_classes = n_classes

    def forward(
        self,
        gnn_multi_emb: torch.Tensor,  # [N_nodes, 768] - precomputed multi-scale STRING_GNN output
        gnn_idx: torch.Tensor,         # [B] - index into gnn_multi_emb, or -1 for missing
    ) -> torch.Tensor:
        B = gnn_idx.size(0)
        device = gnn_idx.device

        # Build per-sample embeddings
        emb = torch.zeros(B, gnn_multi_emb.size(1), device=device, dtype=gnn_multi_emb.dtype)

        in_gnn_mask = gnn_idx >= 0
        not_in_gnn_mask = ~in_gnn_mask

        if in_gnn_mask.any():
            valid_idx = gnn_idx[in_gnn_mask]
            emb[in_gnn_mask] = gnn_multi_emb[valid_idx]

        if not_in_gnn_mask.any():
            fallback = self.fallback_emb(
                torch.zeros(not_in_gnn_mask.sum(), device=device, dtype=torch.long)
            )
            emb[not_in_gnn_mask] = fallback.to(emb.dtype)

        # Forward through MLP
        x = self.input_proj(emb)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)  # [B, n_genes * n_classes]

        # Add per-gene bias (broadcast over batch)
        logits = logits + self.per_gene_bias.to(logits.dtype)

        return logits.view(B, self.n_classes, self.n_genes)  # [B, 3, 6640]


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.35,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        focal_gamma: float = 1.5,
        label_smoothing: float = 0.05,
        focal_weight: float = 0.5,
        lr_patience: int = 8,
        lr_factor: float = 0.5,
        output_bottleneck: int = 256,
        max_epochs: int = 150,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.focal_weight = focal_weight
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.output_bottleneck = output_bottleneck
        self.max_epochs = max_epochs

        self.model: Optional[MultiScaleStringGNNMLP] = None
        self.gnn_model = None  # STRING_GNN backbone (frozen)

        # Buffers for validation and test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

        # Cached multi-scale STRING_GNN embeddings [n_nodes, 768]
        self._gnn_multi_emb_cache: Optional[torch.Tensor] = None

    def setup(self, stage: str = "fit") -> None:
        from transformers import AutoModel

        # Class weights: inversely proportional to frequency
        # After label shift: original -1 -> class0 (down), 0 -> class1 (neutral), 1 -> class2 (up)
        # Frequencies: class0=4.77%, class1=92.82%, class2=2.41%
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = 1.0 / freq
        class_weights = class_weights / class_weights.sum() * N_CLASSES  # normalize to mean=1
        self.register_buffer("class_weights", class_weights)

        # Load STRING_GNN backbone (frozen) — only once
        if self.gnn_model is None:
            self.gnn_model = AutoModel.from_pretrained(
                str(STRING_GNN_DIR), trust_remote_code=True
            )
            self.gnn_model.eval()
            for param in self.gnn_model.parameters():
                param.requires_grad = False
            self.print(
                f"STRING_GNN loaded (frozen): {sum(p.numel() for p in self.gnn_model.parameters()):,} params"
            )

        # Initialize the MLP prediction model — only once
        if self.model is None:
            self.model = MultiScaleStringGNNMLP(
                n_nodes=self.n_nodes,
                multi_scale_input_dim=MULTI_SCALE_DIM,
                hidden_dim=self.hidden_dim,
                n_blocks=self.n_blocks,
                dropout=self.dropout,
                output_bottleneck=self.output_bottleneck,
            )
            # Cast trainable parameters to float32 for stable optimization
            for v in self.model.parameters():
                if v.requires_grad:
                    v.data = v.data.float()
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print(
                f"MultiScaleStringGNNMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
                f"bottleneck={self.output_bottleneck} | trainable_params={total_params:,}"
            )

    def _get_multi_scale_gnn_embeddings(self) -> torch.Tensor:
        """Get cached multi-scale STRING_GNN node embeddings.

        Extracts intermediate hidden states from layers 4, 6, 8 and concatenates
        to form [n_nodes, 768] multi-scale embeddings. Computed once and cached.
        """
        if self._gnn_multi_emb_cache is not None:
            return self._gnn_multi_emb_cache

        device = next(self.model.parameters()).device

        # Move GNN model to same device
        self.gnn_model = self.gnn_model.to(device)

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location=device)
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)

        with torch.no_grad():
            outputs = self.gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
                output_hidden_states=True,
            )

        # Extract multi-scale features from layers 4, 6, 8
        # outputs.hidden_states is a tuple of length 9: [initial, layer1, ..., layer8]
        hidden_states = outputs.hidden_states  # tuple of [n_nodes, 256] tensors
        scale_embs = []
        for layer_idx in MULTI_SCALE_LAYER_INDICES:
            scale_emb = hidden_states[layer_idx].float().detach()  # [n_nodes, 256]
            scale_embs.append(scale_emb)

        # Concatenate along feature dimension: [n_nodes, 768]
        multi_scale_emb = torch.cat(scale_embs, dim=-1)
        self._gnn_multi_emb_cache = multi_scale_emb
        self.print(
            f"Multi-scale STRING_GNN cache computed: layers={MULTI_SCALE_LAYER_INDICES}, "
            f"shape={multi_scale_emb.shape}"
        )
        return self._gnn_multi_emb_cache

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Hybrid focal + weighted CE loss with label smoothing.

        logits: [B, 3, 6640]
        labels: [B, 6640]  values in {0,1,2}
        """
        # logits: [B, 3, 6640] -> [B*6640, 3]
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)  # [B*6640]
        loss = hybrid_loss(
            logits_flat,
            labels_flat,
            self.class_weights,
            focal_gamma=self.focal_gamma,
            label_smoothing=self.label_smoothing,
            focal_weight=self.focal_weight,
        )
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        gnn_emb = self._get_multi_scale_gnn_embeddings()
        logits = self.model(gnn_emb, batch["gnn_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        gnn_emb = self._get_multi_scale_gnn_embeddings()
        logits = self.model(gnn_emb, batch["gnn_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds_local = torch.cat(self._val_preds, dim=0)   # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0) # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather from all DDP ranks
        all_preds = self.all_gather(preds_local)  # [ws, N_local, 3, 6640]
        all_labels = self.all_gather(labels_local)  # [ws, N_local, 6640]

        # Reshape to flat tensors
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)  # [ws*N_local, 3, 6640]
        all_labels = all_labels.view(-1, N_GENES)            # [ws*N_local, 6640]

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()

        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        gnn_emb = self._get_multi_scale_gnn_embeddings()
        logits = self.model(gnn_emb, batch["gnn_idx"])  # [B, 3, 6640]
        self._test_preds.append(logits.detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        # Gather predictions from all DDP ranks
        all_preds = self.all_gather(preds_local)  # [ws, N_local, 3, 6640]
        ws = self.trainer.world_size
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)  # [ws*N_local, 3, 6640]

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

        if self.trainer.is_global_zero:
            if all_preds.size(0) == 0:
                self.print("Warning: No test predictions gathered. Skipping save.")
            else:
                preds_np = all_preds.float().cpu().numpy()  # [N, 3, 6640]

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

    def configure_optimizers(self):
        # Only optimize the MLP head (GNN backbone is frozen)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # ReduceLROnPlateau: reactive LR reduction when val/f1 plateaus
        # This was shown to be more effective than cosine annealing (parent)
        # because it triggers when the model actually stops improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # maximize val/f1
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
        """Save only trainable parameters (skip frozen STRING_GNN backbone)."""
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                trainable[key] = full_sd[key]
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} trainable params "
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
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
        description="Node1-2: Multi-Scale STRING_GNN + Factorized Head + Hybrid Loss"
    )
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--focal-weight", type=float, default=0.5)
    p.add_argument("--output-bottleneck", type=int, default=256)
    p.add_argument("--lr-patience", type=int, default=8)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--early-stop-patience", type=int, default=30)
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
        n_nodes=datamodule.n_nodes,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        focal_weight=args.focal_weight,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        output_bottleneck=args.output_bottleneck,
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
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Save score
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results -> {score_path}")


if __name__ == "__main__":
    main()
