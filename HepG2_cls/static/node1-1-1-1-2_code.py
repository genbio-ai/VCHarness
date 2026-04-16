"""Node 1-1-1-1-2: STRING_GNN (partially fine-tuned) + Restored Flat MLP Head +
CosineAnnealingLR (T_max=200, no restart) + STRING-only features.

This node directly addresses the two key failure modes identified in the sibling
(node1-1-1-1-1, F1=0.462):
1. Cosine restart at T_max=100 disrupted convergence (val F1 plateaued after epoch 68)
2. ESM2-650M features provided marginal/negative value (0.462 < 0.474 STRING-only)

Architecture:
  - Input: perturbed gene identifier (ENSEMBL ID)
  - STRING_GNN partially fine-tuned: last 2 GNN layers (mps.6, mps.7) + post_mp (~400K unfrozen)
  - Per-gene lookup: use the partially-fine-tuned GNN embedding for the perturbed gene
  - Fallback embedding: learnable 256-dim vector for genes not in STRING
  - 3x Residual MLP blocks: 512-dim hidden, 1024-dim intermediate
  - FLAT output head: LayerNorm(512) -> Linear(512->19920) — NO factorized bottleneck
  - Loss: Focal loss (gamma=2.0) with class weights and label smoothing (eps=0.05)
  - Optimizer: AdamW (lr=3e-4 for MLP, gnn_lr=5e-5 for GNN, weight_decay=5e-4)
  - LR schedule: CosineAnnealingLR (T_max=200, eta_min=1e-7) — single full-length cycle
  - Gradient clipping: 1.0
  - max_epochs=200, early_stop_patience=40

Key differences from sibling node1-1-1-1-1:
  1. NO ESM2 features: STRING-only (ESM2 consistently degrades performance across all lineages)
  2. T_max=200 instead of 100: avoids cosine restart that disrupted sibling's convergence
  3. Flat output head (512->19920): removes extra parameters added by factorized (512->512->19920)
  4. lr=3e-4 instead of 2e-4: same as proven node1-1-1 (F1=0.474)
  5. No ESM2 projection layer: simpler architecture, fewer parameters to overfit

Key differences from parent node1-1-1-1:
  1. Bottleneck fixed: Flat head instead of 256->128->19920 (critical fix for underfitting)
  2. hidden_dim=512 (restored from 256): sufficient capacity for 19920-dim output
  3. n_blocks=3 (restored from 2): same depth as proven node1-1-1
  4. dropout=0.35 (reduced from 0.45): appropriate for restored model capacity
  5. lr=3e-4 (increased from 1e-4): appropriate for larger model
  6. weight_decay=5e-4 (reduced from 1e-3): restore proven setting
  7. CosineAnnealingLR T_max=200 instead of ReduceLROnPlateau (avoids premature halving)

Memory sources:
  - node1-1-1-1-1 feedback: primary prescription — fix T_max=200, revert STRING-only, lr=3e-4
  - node1-1-1 (F1=0.474): STRING_GNN partial fine-tune + 3-block 512-dim + flat head = proven ceiling
  - node1-3-3 (F1=0.495): STRING-only with warm restarts/Mixup reaches 0.495 → STRING-only is viable
  - collected_memory ESM2 pattern: every ESM2-augmented node underperforms STRING-only baseline
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
        self.gnn_indices = torch.tensor(gnn_indices, dtype=torch.long)

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
# Focal Loss with Label Smoothing
# ---------------------------------------------------------------------------
def focal_loss_with_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
    smooth_eps: float = 0.05,
) -> torch.Tensor:
    """Multi-class focal loss with class weights and label smoothing.

    logits:  [N, C] float
    targets: [N]    long  values in {0,...,C-1}
    class_weights: [C] float
    gamma: focusing parameter (0 = standard CE)
    smooth_eps: label smoothing epsilon (0 = no smoothing)
    """
    C = logits.size(1)
    log_softmax = F.log_softmax(logits, dim=-1)   # [N, C]
    softmax = log_softmax.exp()                    # [N, C]

    # Focal weight computed from the hard target probability
    log_pt = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
    pt = softmax.gather(1, targets.unsqueeze(1)).squeeze(1)          # [N]
    focal_weight = (1.0 - pt) ** gamma                                # [N]

    # Label-smoothed CE loss
    if smooth_eps > 0.0:
        smooth_loss = (
            -(1.0 - smooth_eps) * log_pt
            - (smooth_eps / C) * log_softmax.sum(dim=-1)
        )  # [N]
    else:
        smooth_loss = -log_pt  # [N] — standard CE

    sample_weight = class_weights[targets]  # [N]
    loss = focal_weight * sample_weight * smooth_loss
    return loss.mean()


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """MLP residual block with layer norm."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
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


class StringGNNFlatHeadMLP(nn.Module):
    """PPI-graph-informed MLP with flat output head for perturbation response prediction.

    Key difference from node1-1-1-1 (parent): Uses FLAT output head (512->19920)
    instead of the catastrophic 256->128->19920 factorized bottleneck that caused F1=0.293.

    Key difference from node1-1-1-1-1 (sibling): STRING-only (no ESM2), which is
    consistently shown to be better across all tree nodes.

    Architecture matches proven node1-1-1 (F1=0.474):
    - 3 residual blocks, 512-dim hidden, 1024-dim intermediate
    - Flat output head: Linear(512->19920)
    - STRING_GNN partial fine-tuning (mps.6, mps.7, post_mp)

    Parameter count:
    - Input proj: Linear(256->512) + LN ≈ 131K
    - 3x ResidualBlock(512, 1024, drop): 3 × [Linear(512->1024) + LN + Linear(1024->512) + LN] ≈ 3.15M
    - Flat head: LN(512) + Linear(512->19920) ≈ 10.2M
    - Fallback emb: 256 params
    - STRING_GNN unfrozen: ~400K
    - Total trainable: ~13.9M (matches node1-1-1's proven capacity)
    """

    def __init__(
        self,
        n_nodes: int,
        gnn_emb_dim: int = STRING_EMB_DIM,
        hidden_dim: int = 512,
        n_blocks: int = 3,
        n_genes: int = N_GENES,
        n_classes: int = N_CLASSES,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()

        # Fallback embedding for genes not in STRING_GNN
        self.fallback_emb = nn.Embedding(1, gnn_emb_dim)
        nn.init.normal_(self.fallback_emb.weight, std=0.01)

        # Project STRING_GNN embeddings (256) -> hidden_dim (512)
        self.input_proj = nn.Sequential(
            nn.Linear(gnn_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Residual MLP blocks: 3 blocks, 512-dim hidden with 1024-dim intermediate
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 2, dropout)
            for _ in range(n_blocks)
        ])

        # FLAT output head: hidden_dim -> n_genes * n_classes (no bottleneck)
        # This matches the proven architecture of node1-1-1 (F1=0.474)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_genes * n_classes),
        )

        self.n_genes = n_genes
        self.n_classes = n_classes

    def forward(
        self,
        gnn_node_emb: torch.Tensor,  # [N_nodes, 256] - precomputed STRING_GNN output
        gnn_idx: torch.Tensor,        # [B] - index into gnn_node_emb, or -1 for missing
    ) -> torch.Tensor:
        B = gnn_idx.size(0)
        device = gnn_idx.device

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

        # Forward through restored MLP
        x = self.input_proj(emb)
        for block in self.blocks:
            x = block(x)

        # Flat head: 512 -> 19920
        logits = self.head(x)  # [B, n_genes * n_classes]

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
        gnn_lr: float = 5e-5,
        weight_decay: float = 5e-4,
        focal_gamma: float = 2.0,
        label_smooth_eps: float = 0.05,
        t_max: int = 200,
        eta_min: float = 1e-7,
        max_epochs: int = 200,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.gnn_lr = gnn_lr
        self.weight_decay = weight_decay
        self.focal_gamma = focal_gamma
        self.label_smooth_eps = label_smooth_eps
        self.t_max = t_max
        self.eta_min = eta_min
        self.max_epochs = max_epochs

        self.model: Optional[StringGNNFlatHeadMLP] = None
        self.gnn_model = None  # STRING_GNN backbone (partially fine-tuned)

        # Buffers for validation and test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

        # Will hold the precomputed/updated GNN node embeddings [n_nodes, 256]
        self._gnn_emb_cache: Optional[torch.Tensor] = None

    def setup(self, stage: str = "fit") -> None:
        from transformers import AutoModel

        # Class weights: inversely proportional to frequency
        # class0 (down-reg) 4.77%, class1 (neutral) 92.82%, class2 (up-reg) 2.41%
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = 1.0 / freq
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        self.register_buffer("class_weights", class_weights)

        # Load STRING_GNN backbone — only once
        if self.gnn_model is None:
            self.gnn_model = AutoModel.from_pretrained(
                str(STRING_GNN_DIR), trust_remote_code=True
            )
            # Freeze all parameters first
            for param in self.gnn_model.parameters():
                param.requires_grad = False

            # Partially unfreeze: last 2 GNN layers (mps.6, mps.7) and post_mp
            # Same as proven node1-1-1 (F1=0.474) configuration
            layers_to_unfreeze = ["mps.6.", "mps.7.", "post_mp."]
            unfrozen_params = 0
            for name, param in self.gnn_model.named_parameters():
                if any(name.startswith(prefix) for prefix in layers_to_unfreeze):
                    param.requires_grad = True
                    unfrozen_params += param.numel()
            self.print(
                f"STRING_GNN loaded: {sum(p.numel() for p in self.gnn_model.parameters()):,} total params, "
                f"{unfrozen_params:,} unfrozen (last 2 layers + post_mp)"
            )
            # Cast unfrozen GNN params to float32 for stable optimization
            for param in self.gnn_model.parameters():
                if param.requires_grad:
                    param.data = param.data.float()

        # Initialize the MLP prediction model — only once
        if self.model is None:
            self.model = StringGNNFlatHeadMLP(
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
            mlp_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print(
                f"StringGNNFlatHeadMLP | hidden={self.hidden_dim} | "
                f"blocks={self.n_blocks} | dropout={self.dropout} | "
                f"MLP trainable params={mlp_trainable:,}"
            )

    def _get_gnn_embeddings(self) -> torch.Tensor:
        """Get STRING_GNN node embeddings."""
        if self.training:
            # Recompute in training mode so gradients flow through unfrozen layers
            device = next(self.model.parameters()).device
            self.gnn_model = self.gnn_model.to(device)
            graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location=device, weights_only=False)
            edge_index = graph["edge_index"]
            edge_weight = graph.get("edge_weight", None)
            outputs = self.gnn_model(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            return outputs.last_hidden_state.float()
        else:
            # In eval mode, cache the embeddings for efficiency
            if self._gnn_emb_cache is None:
                device = next(self.model.parameters()).device
                self.gnn_model = self.gnn_model.to(device)
                graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location=device, weights_only=False)
                edge_index = graph["edge_index"]
                edge_weight = graph.get("edge_weight", None)
                with torch.no_grad():
                    outputs = self.gnn_model(
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                    )
                self._gnn_emb_cache = outputs.last_hidden_state.float().detach()
            return self._gnn_emb_cache

    def on_train_epoch_start(self) -> None:
        # Invalidate the embedding cache at the start of each training epoch
        # so that val/test will recompute with the latest GNN weights
        self._gnn_emb_cache = None

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Focal loss with label smoothing and class weights."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        loss = focal_loss_with_label_smoothing(
            logits_flat,
            labels_flat,
            self.class_weights,
            gamma=self.focal_gamma,
            smooth_eps=self.label_smooth_eps,
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
        preds_local = torch.cat(self._val_preds, dim=0)   # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0) # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather from all DDP ranks
        all_preds = self.all_gather(preds_local)   # [ws, N_local, 3, 6640]
        all_labels = self.all_gather(labels_local)  # [ws, N_local, 6640]

        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
        all_labels = all_labels.view(-1, N_GENES)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()

        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        gnn_emb = self._get_gnn_embeddings()
        logits = self.model(gnn_emb, batch["gnn_idx"])
        self._test_preds.append(logits.detach().cpu())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> Optional[Dict[str, Any]]:
        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        all_preds = self.all_gather(preds_local)
        ws = self.trainer.world_size
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        # Compute test F1 if labels are available (gathered across ranks)
        test_f1 = None
        if self._test_labels:
            labels_local = torch.cat(self._test_labels, dim=0)
            all_labels = self.all_gather(labels_local)
            all_labels = all_labels.view(-1, N_GENES).cpu().numpy()
            preds_np = all_preds.float().cpu().numpy()
            test_f1 = _compute_per_gene_f1(preds_np, all_labels)
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=True)
        self._test_labels.clear()

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
                preds_np = all_preds.float().cpu().numpy()

                # Deduplicate by pert_id (DistributedSampler may pad with replicated samples)
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
                        f"Deduplication: {n_unique}/{len(keep_mask)} unique samples after removing DistributedSampler padding."
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

        return {"test_f1": test_f1} if test_f1 is not None else {}

    def configure_optimizers(self):
        # Separate param groups: GNN unfrozen layers get lower LR
        gnn_params = [p for p in self.gnn_model.parameters() if p.requires_grad]
        mlp_params = list(self.model.parameters())

        param_groups = [
            {"params": mlp_params, "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": gnn_params, "lr": self.gnn_lr, "weight_decay": self.weight_decay},
        ]

        optimizer = torch.optim.AdamW(param_groups)

        # CosineAnnealingLR with T_max=max_epochs: single full-length cosine decay
        # This avoids the cosine restart at T_max=100 that disrupted sibling node1-1-1-1-1
        # The model decays from lr to eta_min over the full training duration
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.t_max,
            eta_min=self.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters (skip frozen parts of STRING_GNN backbone)."""
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
        description="Node1-1-1-1-2: STRING_GNN partial fine-tune + Flat Head + CosineAnnealingLR T_max=200"
    )
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gnn-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smooth-eps", type=float, default=0.05)
    p.add_argument("--t-max", type=int, default=200)
    p.add_argument("--eta-min", type=float, default=1e-7)
    p.add_argument("--early-stop-patience", type=int, default=40)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
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
        gnn_lr=args.gnn_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smooth_eps=args.label_smooth_eps,
        t_max=args.t_max,
        eta_min=args.eta_min,
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
        val_check_interval=1.0,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Save test score summary
    if test_results and "test_f1" in test_results[0]:
        test_f1 = test_results[0]["test_f1"]
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"test_f1: {test_f1:.6f}\n"
            f"hidden_dim: {args.hidden_dim}\n"
            f"n_blocks: {args.n_blocks}\n"
            f"dropout: {args.dropout}\n"
            f"lr: {args.lr}\n"
            f"gnn_lr: {args.gnn_lr}\n"
            f"weight_decay: {args.weight_decay}\n"
            f"focal_gamma: {args.focal_gamma}\n"
            f"label_smooth_eps: {args.label_smooth_eps}\n"
            f"t_max: {args.t_max}\n"
        )
        print(f"Test F1: {test_f1:.6f}")


if __name__ == "__main__":
    main()
