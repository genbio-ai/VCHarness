"""Node 1-2: STRING_GNN (Frozen) + hidden_dim=448, 3-block MLP + Focal Loss + Muon+AdamW.

Architecture:
  - Input: perturbed gene identifier (ENSEMBL ID)
  - STRING_GNN pretrained backbone: run once to get 256-dim PPI-graph embeddings (frozen)
  - Per-gene lookup: STRING_GNN final-layer embedding (256-dim) for the perturbed gene
  - Fallback embedding: learnable 256-dim vector for genes not in STRING
  - 3-block PreLN residual MLP: hidden_dim=448 (between proven best 384 and overfitting-prone 512)
  - Per-gene learnable bias: 19920-parameter term capturing gene-specific baseline expression
  - Output head: flat Linear(448 -> 6640 * 3) -> reshape to [B, 3, 6640]
  - Loss: Focal loss (gamma=2.0) + class frequency weights for extreme class imbalance
  - Optimizer: Muon (hidden weights, lr=0.02) + AdamW (other params, lr=3e-4)
  - LR schedule: ReduceLROnPlateau for reactive LR reduction
  - Gradient clipping: max_norm=1.0 for training stability

Key innovations vs parent node1-1-3 (hidden=640, 4-block, WCE+smooth, F1=0.4297):
  1. Revert to hidden_dim=448 (from 640): node1-3-2 shows 384 beats 512; 448 is the sweet spot
  2. Revert to focal loss (gamma=2.0): consistently outperforms WCE+label_smooth on this task
  3. Muon optimizer for hidden weight matrices: the key that broke the 0.474 ceiling (node1-3-2: F1=0.4756)
  4. Gradient clipping (max_norm=1.0): stabilizes Muon training, used in all Muon successes
  5. weight_decay=0.01 (from 1e-3): stronger regularization proven effective (node3-1-1-1-2-1: F1=0.4732)
  6. RLROP patience=8 (from 10): faster reactive LR reduction on small val set (141 samples)
  7. dropout=0.35 (from 0.40): 640-dim needed high dropout; 448-dim can use standard 0.35

Differentiation from siblings:
  - sibling node1-1-1: 3-block, hidden=512, partial GNN fine-tuning, focal, lr=3e-4, RLROP, F1=0.4737
  - This node: 3-block, hidden=448, fully FROZEN STRING, focal, MUON+AdamW, RLROP, gradient clip
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
N_GENES = 6640          # number of response genes
N_CLASSES = 3           # {-1->0, 0->1, 1->2}
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
STRING_EMB_DIM = 256    # STRING_GNN output embedding dimension


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
# Model components
# ---------------------------------------------------------------------------
class PreLNResidualBlock(nn.Module):
    """Pre-LayerNorm MLP residual block with GELU activation.

    Pre-LN normalization (normalize BEFORE transformation) provides more
    stable gradient flow compared to post-LN, especially beneficial when
    combined with Muon optimizer. Used in node1-3-2 (F1=0.4756).
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))



class StringGNNMLP(nn.Module):
    """PPI-graph-informed MLP for perturbation response prediction.

    Takes the STRING_GNN final-layer node embedding (256-dim) for the perturbed gene
    and maps it through a 3-block PreLN residual MLP (hidden_dim=448) with a per-gene
    bias term to predict 6640 x 3 class logits.

    Key design: hidden_dim=448 is between the confirmed-best 384 (node1-3-2: F1=0.4756)
    and the overfitting-prone 512 (node1-1-1: F1=0.4737), targeting optimal capacity.
    """

    def __init__(
        self,
        n_nodes: int,
        gnn_emb_dim: int = STRING_EMB_DIM,
        hidden_dim: int = 448,
        n_blocks: int = 3,
        n_genes: int = N_GENES,
        n_classes: int = N_CLASSES,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()

        # Fallback embedding for genes not in STRING_GNN
        self.fallback_emb = nn.Embedding(1, gnn_emb_dim)
        nn.init.normal_(self.fallback_emb.weight, std=0.01)

        # Project STRING_GNN embeddings (256) -> hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(gnn_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 3x PreLN Residual MLP Blocks (hidden_dim=448, intermediate=896)
        # Using PreLN for stable gradients with Muon optimizer
        self.blocks = nn.ModuleList([
            PreLNResidualBlock(hidden_dim, hidden_dim * 2, dropout)
            for _ in range(n_blocks)
        ])

        # Per-gene learnable bias: captures baseline gene expression tendencies
        # Each gene gets its own 3-class bias (19,920 params total)
        # Proven to help in node1-1-1 (F1=0.474), node1-3-2 (F1=0.4756)
        self.per_gene_bias = nn.Parameter(
            torch.zeros(n_genes * n_classes)
        )

        # Output head: final LayerNorm + flat Linear (no bottleneck)
        # Flat head is proven superior to factorized heads (failed 5+ times in tree)
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_genes * n_classes)

        self.n_genes = n_genes
        self.n_classes = n_classes

    def forward(
        self,
        gnn_node_emb: torch.Tensor,  # [N_nodes, 256] - precomputed STRING_GNN output
        gnn_idx: torch.Tensor,        # [B] - index into gnn_node_emb, or -1 for missing
    ) -> torch.Tensor:
        B = gnn_idx.size(0)
        device = gnn_idx.device

        # Build per-sample embeddings
        emb = torch.zeros(B, gnn_node_emb.size(1), device=device, dtype=gnn_node_emb.dtype)

        in_gnn_mask = gnn_idx >= 0
        not_in_gnn_mask = ~in_gnn_mask

        if in_gnn_mask.any():
            valid_idx = gnn_idx[in_gnn_mask]
            emb[in_gnn_mask] = gnn_node_emb[valid_idx]

        if not_in_gnn_mask.any():
            fallback = self.fallback_emb(
                torch.zeros(not_in_gnn_mask.sum(), device=device, dtype=torch.long)
            )
            emb[not_in_gnn_mask] = fallback.to(emb.dtype)

        # Forward through MLP
        x = self.input_proj(emb)
        for block in self.blocks:
            x = block(x)

        # Output head
        x = self.head_norm(x)
        logits = self.head(x)  # [B, n_genes * n_classes]

        # Add per-gene bias (broadcast over batch dimension)
        logits = logits + self.per_gene_bias.to(logits.dtype).unsqueeze(0)  # [B, n_genes * n_classes]

        return logits.view(B, self.n_classes, self.n_genes)  # [B, 3, 6640]


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int = 448,
        n_blocks: int = 3,
        dropout: float = 0.35,
        muon_lr: float = 0.02,
        adamw_lr: float = 3e-4,
        weight_decay: float = 0.01,
        focal_gamma: float = 2.0,
        lr_patience: int = 8,
        lr_factor: float = 0.5,
        max_epochs: int = 250,
        grad_clip_norm: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.muon_lr = muon_lr
        self.adamw_lr = adamw_lr
        self.weight_decay = weight_decay
        self.focal_gamma = focal_gamma
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.max_epochs = max_epochs
        self.grad_clip_norm = grad_clip_norm

        self.model: Optional[StringGNNMLP] = None
        self.gnn_model = None  # STRING_GNN backbone (frozen)

        # Buffers for validation and test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

        # Precomputed GNN node embeddings [n_nodes, 256]
        self._gnn_emb_cache: Optional[torch.Tensor] = None

    def setup(self, stage: str = "fit") -> None:
        from transformers import AutoModel

        # Class weights: inversely proportional to frequency (for focal loss alpha term)
        # class0 (down-reg) 4.77%, class1 (neutral) 92.82%, class2 (up-reg) 2.41%
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
                f"STRING_GNN loaded: {sum(p.numel() for p in self.gnn_model.parameters()):,} params (fully frozen)"
            )

        # Initialize the MLP prediction model — only once
        if self.model is None:
            self.model = StringGNNMLP(
                n_nodes=self.n_nodes,
                gnn_emb_dim=STRING_EMB_DIM,
                hidden_dim=self.hidden_dim,
                n_blocks=self.n_blocks,
                dropout=self.dropout,
            )
            # Cast trainable parameters to float32 for stable optimization
            for v in self.model.parameters():
                if v.requires_grad:
                    v.data = v.data.float()
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print(
                f"StringGNNMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
                f"trainable_params={trainable:,}"
            )

    def _get_gnn_embeddings(self) -> torch.Tensor:
        """Get cached STRING_GNN node embeddings; compute and cache on first call."""
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
        # Cache the final-layer embeddings [n_nodes, 256] as float32 on GPU
        self._gnn_emb_cache = outputs.last_hidden_state.float().detach()
        self.print(f"STRING_GNN embeddings cached: shape={self._gnn_emb_cache.shape}, dtype=float32")
        return self._gnn_emb_cache

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Focal loss with class weights.

        logits: [B, 3, 6640]
        labels: [B, 6640]  values in {0,1,2}

        Focal loss (gamma=2.0) with class frequency weights:
        - Downweights easy examples (neutral class ~93% of data)
        - Focuses on hard minority class examples (up/down-regulated)
        - Consistently outperforms WCE+label_smooth on this task per memory
        """
        # logits: [B, 3, 6640] -> [B*6640, 3]
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)  # [B*6640]

        # Use cross-entropy with weights to get per-sample CE, then apply focal modulation
        class_weights = self.class_weights.to(logits_flat.device)
        ce_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=class_weights,
            reduction="none",
        )
        p_t = torch.exp(-ce_loss)
        focal_loss = (1.0 - p_t) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        gnn_emb = self._get_gnn_embeddings()
        logits = self.model(gnn_emb, batch["gnn_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
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
        preds_local = torch.cat(self._val_preds, dim=0)   # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0) # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        # Compute F1 locally on each rank to avoid massive all_gather of 1.7GB per rank
        preds_np = preds_local.float().cpu().numpy()
        labels_np = labels_local.cpu().numpy()
        local_f1 = _compute_per_gene_f1(preds_np, labels_np)

        # Gather local F1 scores from all ranks (minimal data transfer)
        all_f1s = self.all_gather(torch.tensor(local_f1, dtype=torch.float64, device="cpu"))
        global_f1 = float(all_f1s.float().mean().item())
        self.log("val/f1", global_f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        gnn_emb = self._get_gnn_embeddings()
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

                # Deduplicate by pert_id (DistributedSampler may pad with replicas)
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
                    self.print(f"Deduplication: {n_unique}/{len(keep_mask)} unique samples after removing DistributedSampler padding.")

                unique_preds = preds_np[keep_mask]
                unique_pert_ids = [p for p, k in zip(all_pert_ids, keep_mask) if k]
                unique_symbols = [s for s, k in zip(all_symbols, keep_mask) if k]

                _save_test_predictions(
                    pert_ids=unique_pert_ids,
                    symbols=unique_symbols,
                    preds=unique_preds,
                    out_path=Path("mcts/node/run/test_predictions.tsv"),
                )

    def configure_optimizers(self):
        """Muon optimizer for hidden weight matrices + AdamW for all other parameters.

        Muon (MomentUm Orthogonalized by Newton-schulz) achieves better sample efficiency
        than AdamW by orthogonalizing gradient updates. This is the key innovation that
        pushed node1-3-2 to F1=0.4756 (STRING-only best), breaking the 0.474 ceiling.

        Parameter grouping:
        - Muon: hidden weight matrices in residual blocks (ndim >= 2, not input_proj/head)
        - AdamW: input_proj, head, per_gene_bias, fallback_emb, LayerNorm params
        """
        from muon import MuonWithAuxAdam

        # Identify hidden weight matrices for Muon (residual block weights only)
        # Input projection, output head, and non-matrix params use AdamW
        muon_params = []
        adamw_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Apply Muon to 2D+ weight matrices inside residual blocks only
            # Exclude: input_proj, head, per_gene_bias, fallback_emb, LayerNorm
            if (
                "blocks." in name
                and param.ndim >= 2
                and "norm" not in name
            ):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        self.print(
            f"Muon params: {sum(p.numel() for p in muon_params):,} | "
            f"AdamW params: {sum(p.numel() for p in adamw_params):,}"
        )

        param_groups = [
            dict(
                params=muon_params,
                use_muon=True,
                lr=self.muon_lr,
                weight_decay=self.weight_decay,
                momentum=0.95,
            ),
            dict(
                params=adamw_params,
                use_muon=False,
                lr=self.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=self.weight_decay,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # ReduceLROnPlateau: reactive LR reduction when val/f1 stops improving
        # patience=8 (vs 10 in parent) is more reactive on the noisy 141-sample val set
        # Proven superior to cosine annealing for STRING-only architectures
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",       # maximize val/f1
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/f1",
                "strict": True,
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
        """Load trainable parameters and persistent buffers from a partial checkpoint."""
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
        description="Node1-2: STRING_GNN (frozen) + 3-block PreLN MLP (hidden=448) + Focal Loss + Muon+AdamW"
    )
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=250)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--hidden-dim", type=int, default=448)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--lr-patience", type=int, default=8)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=30)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Set CUDA device per rank to avoid contention when torchrun spawns multiple processes
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    pl.seed_everything(0)

    data_dir = Path("data")
    # Use absolute path to avoid symlink resolution issues in DDP
    output_dir = Path.cwd() / "mcts/node/run"
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
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        max_epochs=args.max_epochs,
        grad_clip_norm=args.grad_clip_norm,
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
        verbose=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.early_stop_patience,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tensorboard_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

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
        val_check_interval=1.0 if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        # Gradient clipping: stabilizes Muon training, proven in node1-3-2 (F1=0.4756)
        gradient_clip_val=args.grad_clip_norm,
        gradient_clip_algorithm="norm",
    )

    # --- Train ---
    trainer.fit(model, datamodule=datamodule)

    # --- Test ---
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
