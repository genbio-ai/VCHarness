"""Node node1-3-3: STRING_GNN Partial Fine-Tuning + 3-block MLP (hidden=384) + Muon Optimizer.

Architecture Overview:
  - Partial fine-tuning of STRING_GNN backbone (last 2 MP layers + post_mp at gnn_lr=5e-5)
  - 3-block Residual MLP head with hidden_dim=384 (reduced from 512 to combat overfitting)
  - Flat unfactorized output head: Linear(384->19920) + per-gene additive bias
  - Muon optimizer (lr=0.02) for hidden weight matrices + AdamW (lr=3e-4) for others
  - ReduceLROnPlateau (patience=20, threshold=5e-4): matches proven node1-1-1 config
  - Gradient clipping (max_norm=1.0): shown effective in node1-3-2

Key Differences from Parent (node1-3, F1=0.381):
  1. ENABLE partial GNN fine-tuning (last 2 layers + post_mp at gnn_lr=5e-5)
     - node1-3 used frozen GNN -> train_loss=0.936 at best epoch
     - node1-1-1 used partial fine-tuning -> train_loss=0.012 = 78x better fit
     - This is the DOMINANT bottleneck: +0.093 F1 gap attributable to GNN fine-tuning
  2. REDUCE hidden_dim 512->384
     - node1-3-2 (hidden=384, Muon) = F1=0.4756 (new tree best)
     - All hidden=512 nodes show severe overfitting (13.5M params for 1,273 samples)
     - 384-dim reduces output head from 10.2M to 7.5M params (-26% in dominant component)
  3. SWITCH to Muon optimizer for hidden weight matrices
     - node1-3-2 showed stable monotonic convergence with Muon+AdamW
     - Muon orthogonalizes momentum updates, improving sample efficiency on small datasets
  4. INCREASE RLROP patience/threshold: patience=15,threshold=1e-4 -> patience=20,threshold=5e-4
     - Matches node1-1-1's proven calibration (F1=0.474)
     - node1-3's threshold=1e-4 was too strict for the noisy 141-sample val metric
  5. REMOVE SWA: confirmed ineffective in parent (val-test gap unchanged at 0.047)
  6. ADD gradient clipping (max_norm=1.0): used successfully in node1-3-2

Target: Combine node1-1-1's partial GNN fine-tuning (key to F1=0.474) with node1-3-2's
hidden_dim=384 + Muon optimizer (key to F1=0.4756), aiming for F1 > 0.48.
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
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES = 6640
N_CLASSES = 3
STRING_EMB_DIM = 256   # STRING_GNN output dim


# ---------------------------------------------------------------------------
# Residual Block (PreLN)
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, dim: int, expand: int = 2, dropout: float = 0.35) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expand)
        self.fc2 = nn.Linear(dim * expand, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


# ---------------------------------------------------------------------------
# Prediction Head with Per-Gene Bias
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """3-block residual MLP + per-gene bias: [B, STRING_EMB_DIM] -> [B, 3, N_GENES].

    Key design choices:
    - hidden_dim=384 (not 512): node1-3-2 proved 384 reduces overfitting on 1,273 samples
      while maintaining strong generalization (F1=0.4756 vs 0.474 with hidden_dim=512)
    - 3 blocks (not 5): node1-1-1 (3 blocks, F1=0.474) vs node1-2 (5 blocks, F1=0.385)
    - Unfactorized output head Linear(384->19920): factorized bottlenecks consistently fail
    - Per-gene bias: proven +0.002 in node1-1-1 (3-block architecture)
    """

    def __init__(
        self,
        in_dim: int = STRING_EMB_DIM,
        hidden_dim: int = 384,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=2, dropout=dropout)
            for _ in range(n_blocks)
        ])
        # Unfactorized output head: proven to outperform factorized across 5+ nodes
        # 384->19920 = 7.5M params (vs 512->19920 = 10.2M); reduced capacity improves generalization
        self.out_proj = nn.Linear(hidden_dim, n_genes * N_CLASSES)

        # Per-gene additive bias: 6640 x 3 = 19,920 learned scalars
        # Learns the "baseline" class distribution for each gene
        # Proven +0.002 F1 in node1-1-1 (3-block architecture)
        self.gene_bias = nn.Parameter(torch.zeros(N_CLASSES, n_genes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)               # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)
        out = self.out_proj(x)               # [B, N_GENES * 3]
        out = out.view(-1, N_CLASSES, self.n_genes)  # [B, 3, N_GENES]
        # Add gene-specific bias: [3, N_GENES] broadcast to [B, 3, N_GENES]
        out = out + self.gene_bias.unsqueeze(0)
        return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Maps each perturbed gene to its STRING_GNN index (not precomputed features)."""

    def __init__(
        self,
        df: pd.DataFrame,
        ensg_to_idx: Dict[str, int],
        n_string_nodes: int,
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.ensg_to_idx = ensg_to_idx
        self.n_string_nodes = n_string_nodes

        if "label" in df.columns:
            labels = np.array(
                [json.loads(x) for x in df["label"].tolist()], dtype=np.int64
            )
            self.labels: Optional[torch.Tensor] = torch.tensor(
                labels + 1, dtype=torch.long
            )  # {-1, 0, 1} -> {0, 1, 2}
        else:
            self.labels = None

        # Pre-compute gnn_idx for each sample (integer index, or -1 for unknown)
        self.gnn_indices: List[int] = [
            self.ensg_to_idx.get(pid, -1) for pid in self.pert_ids
        ]

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "gnn_node_idx": self.gnn_indices[idx],  # integer: valid index or -1
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

        self.ensg_to_idx: Optional[Dict[str, int]] = None
        self.n_string_nodes: int = 0
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        # Build node index map from STRING_GNN node names
        # Guard against multiple calls (Lightning calls setup() for each stage)
        if self.ensg_to_idx is None:
            model_dir = Path(STRING_GNN_DIR)
            node_names: List[str] = json.loads(
                (model_dir / "node_names.json").read_text()
            )
            self.ensg_to_idx = {name: i for i, name in enumerate(node_names)}
            self.n_string_nodes = len(node_names)

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df, self.ensg_to_idx, self.n_string_nodes)
        self.val_ds = PerturbDataset(val_df, self.ensg_to_idx, self.n_string_nodes)
        self.test_ds = PerturbDataset(test_df, self.ensg_to_idx, self.n_string_nodes)

    def _make_loader(self, ds: PerturbDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    """STRING_GNN (partial fine-tuning) + 3-block MLP (hidden=384) + Muon optimizer.

    Design motivation:
    - Partial GNN fine-tuning: node1-3 frozen GNN -> train_loss=0.936, F1=0.381
                              node1-1-1 partial GNN -> train_loss=0.012, F1=0.474
                              The 78x better training fit confirms GNN adaptation is critical
    - hidden_dim=384: node1-3-2 (384, Muon) = F1=0.4756 vs all 512 nodes <= F1=0.474
                      512-dim nodes are overparameterized for 1,273 training samples
    - Muon for hidden weights: stable convergence without erratic AdamW fluctuations
    - RLROP patience=20, threshold=5e-4: matches node1-1-1's proven calibration
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        lr: float = 3e-4,
        gnn_lr: float = 5e-5,
        muon_lr: float = 0.02,
        weight_decay: float = 0.001,
        dropout: float = 0.35,
        label_smoothing: float = 0.05,
        rlrop_patience: int = 20,
        rlrop_factor: float = 0.5,
        rlrop_threshold: float = 5e-4,
        rlrop_min_lr: float = 1e-6,
        grad_clip_val: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.gnn: Optional[nn.Module] = None
        self.head: Optional[PerturbHead] = None

        # Graph tensors (regular attrs, not buffers, to avoid name conflict with init)
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_weight: Optional[torch.Tensor] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Guard against multiple calls (Lightning calls setup() for each stage: fit, validate, test)
        if self.gnn is not None:
            return

        model_dir = Path(STRING_GNN_DIR)

        # Load STRING_GNN model
        print("Loading STRING_GNN for partial fine-tuning...", flush=True)
        self.gnn = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        self.gnn.train()

        # Partial fine-tuning: freeze all except last 2 MP layers and post_mp
        # This replicates node1-1-1's proven configuration (F1=0.474)
        # Frozen layers: emb, mps.0-5 (early topology layers)
        # Fine-tuned layers: mps.6, mps.7, post_mp (top-level task-specific adaptation)
        for name, param in self.gnn.named_parameters():
            if any(name.startswith(prefix) for prefix in ["mps.6", "mps.7", "post_mp"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Initialize prediction head
        self.head = PerturbHead(
            in_dim=STRING_EMB_DIM,
            hidden_dim=self.hparams.hidden_dim,
            n_genes=self.hparams.n_genes,
            n_blocks=self.hparams.n_blocks,
            dropout=self.hparams.dropout,
        )

        # Cast trainable parameters to float32 for stable optimization
        for p in self.gnn.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        for p in self.head.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Load graph tensors as regular attributes (stored on CPU; Lightning moves them to GPU automatically)
        graph = torch.load(model_dir / "graph_data.pt", map_location="cpu")
        self._edge_index = graph["edge_index"].long()
        edge_weight = graph.get("edge_weight", None)
        self._edge_weight = edge_weight.float() if edge_weight is not None else None

        # Class weights: inversely proportional to class frequency, normalized to mean=1
        # Frequencies from DATA_ABSTRACT: down=4.77%, neutral=92.82%, up=2.41%
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = 1.0 / freq
        class_weights = class_weights / class_weights.mean()
        self.register_buffer("class_weights", class_weights)

        gnn_trainable = sum(p.numel() for p in self.gnn.parameters() if p.requires_grad)
        gnn_total = sum(p.numel() for p in self.gnn.parameters())
        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        head_total = sum(p.numel() for p in self.head.parameters())
        self.print(
            f"STRING_GNN (partial fine-tune): {gnn_trainable:,}/{gnn_total:,} trainable params | "
            f"Head (hidden={self.hparams.hidden_dim}): {head_trainable:,}/{head_total:,} | "
            f"Total trainable: {gnn_trainable + head_trainable:,}"
        )

    def _get_gnn_embeddings(self) -> torch.Tensor:
        """Run STRING_GNN forward pass to get node embeddings [N, 256]."""
        # Move graph tensors to the same device as the GNN
        edge_index = self._edge_index.to(device=self.gnn.device)
        edge_weight = self._edge_weight.to(device=self.gnn.device) if self._edge_weight is not None else None
        out = self.gnn(
            edge_index=edge_index,
            edge_weight=edge_weight,
            output_hidden_states=False,
        )
        return out.last_hidden_state.float()  # [N_NODES, 256]

    def _lookup_features(self, gnn_node_idx: torch.Tensor, all_embeddings: torch.Tensor) -> torch.Tensor:
        """Look up features from precomputed embeddings tensor.

        Args:
            gnn_node_idx: [B] tensor of GNN node indices (-1 for unknown genes)
            all_embeddings: [N_NODES, 256] current GNN embeddings

        Returns:
            features: [B, 256] feature vectors (zero vector for unknown genes)
        """
        B = gnn_node_idx.shape[0]
        # Use valid indices; fallback to zero vector for unknowns (-1)
        valid_mask = gnn_node_idx >= 0
        features = torch.zeros(B, STRING_EMB_DIM, dtype=all_embeddings.dtype, device=all_embeddings.device)
        if valid_mask.any():
            valid_idx = gnn_node_idx[valid_mask]
            features[valid_mask] = all_embeddings[valid_idx]
        return features

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard weighted cross-entropy with label smoothing.

        No focal loss (caused catastrophic collapse in node3-1).
        Weighted CE + label smoothing is the proven recipe (node1-1: 0.472, node1-1-1: 0.474).
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Run STRING_GNN forward to get current (partially fine-tuned) embeddings
        all_embeddings = self._get_gnn_embeddings()  # [N_NODES, 256]
        feats = self._lookup_features(batch["gnn_node_idx"], all_embeddings)  # [B, 256]
        logits = self.head(feats)
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        # Run GNN forward for validation
        all_embeddings = self._get_gnn_embeddings()
        feats = self._lookup_features(batch["gnn_node_idx"], all_embeddings)
        logits = self.head(feats)
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
        preds = torch.cat(self._val_preds, dim=0).numpy()    # [local_N, 3, N_GENES]
        labels = torch.cat(self._val_labels, dim=0).numpy()  # [local_N, N_GENES]
        self._val_preds.clear()
        self._val_labels.clear()

        f1 = _compute_per_gene_f1(preds, labels)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        all_embeddings = self._get_gnn_embeddings()
        feats = self._lookup_features(batch["gnn_node_idx"], all_embeddings)
        logits = self.head(feats)
        self._test_preds.append(logits.detach().cpu().float())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)   # [local_N, 3, N_GENES]
        labels_local = (
            torch.cat(self._test_labels, dim=0) if self._test_labels else None
        )
        self._test_preds.clear()
        self._test_labels.clear()

        # Gather predictions from all ranks
        all_preds = self.all_gather(preds_local)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        # Gather labels and compute test F1
        if labels_local is not None:
            all_labels = self.all_gather(labels_local)
            all_labels = all_labels.view(-1, N_GENES)
            test_f1 = _compute_per_gene_f1(
                all_preds.float().cpu().numpy(),
                all_labels.cpu().numpy(),
            )
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=True)

        # Gather string metadata
        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        gathered_pert_ids = [local_pert_ids]
        gathered_symbols = [local_symbols]
        if world_size > 1:
            obj_pert = [None] * world_size
            obj_sym = [None] * world_size
            dist.all_gather_object(obj_pert, local_pert_ids)
            dist.all_gather_object(obj_sym, local_symbols)
            gathered_pert_ids = obj_pert
            gathered_symbols = obj_sym

        if self.trainer.is_global_zero:
            all_pert_ids = [p for rank_list in gathered_pert_ids for p in rank_list]
            all_symbols = [s for rank_list in gathered_symbols for s in rank_list]
            all_preds_np = all_preds.float().cpu().numpy()

            # Deduplicate by pert_id (handles DDP padding)
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(all_preds_np[i])

            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=np.stack(dedup_preds, axis=0),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    def configure_optimizers(self):
        """Muon optimizer for hidden weight matrices + AdamW for others.

        Optimizer strategy:
        - Muon (lr=0.02): hidden 2D weight matrices in residual blocks
          * Applies orthogonalization to momentum updates for better sample efficiency
          * node1-3-2 proved this approach with stable monotonic convergence (F1=0.4756)
          * DO NOT apply Muon to: LayerNorm, biases, embeddings, output heads
        - AdamW (lr=3e-4): all other parameters (norms, biases, GNN fine-tune layers,
                            output head, per-gene bias)
        - GNN fine-tune layers use gnn_lr=5e-5 (10x lower than head lr=3e-4)
          * Matches node1-1-1's proven GNN fine-tuning configuration (F1=0.474)
          * Lower LR prevents catastrophic forgetting of PPI topology priors

        ReduceLROnPlateau:
        - patience=20, threshold=5e-4: matches node1-1-1's proven configuration
        - node1-3 used threshold=1e-4 (too strict for 141-sample noisy val metric)
        - node1-3 used patience=15 (too aggressive; RLROP fired too late in parent)
        """
        try:
            from muon import MuonWithAuxAdam
            muon_available = True
        except ImportError:
            muon_available = False
            print("WARNING: muon package not available, falling back to AdamW", flush=True)

        if muon_available:
            # Separate parameters by type for Muon vs AdamW
            # Muon: hidden 2D weight matrices in residual blocks (fc1, fc2 in ResidualBlock)
            # AdamW: everything else (norms, biases, GNN layers, out_proj, gene_bias)
            muon_params = []
            gnn_params = []  # GNN fine-tune layers: lower LR
            adamw_params = []  # All other non-GNN trainable params

            # Head block matrices for Muon
            muon_param_ids = set()
            for name, param in self.head.blocks.named_parameters():
                if param.requires_grad and param.ndim >= 2 and "norm" not in name:
                    # fc1 and fc2 weight matrices in ResidualBlock
                    muon_params.append(param)
                    muon_param_ids.add(id(param))

            # GNN fine-tune layers: AdamW with lower LR
            gnn_param_ids = set()
            for name, param in self.gnn.named_parameters():
                if param.requires_grad:
                    gnn_params.append(param)
                    gnn_param_ids.add(id(param))

            # Remaining head params: input_proj, out_proj, gene_bias + norms/biases in blocks
            for name, param in self.head.named_parameters():
                if param.requires_grad and id(param) not in muon_param_ids:
                    adamw_params.append(param)

            param_groups = [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=self.hparams.muon_lr,
                    weight_decay=self.hparams.weight_decay,
                    momentum=0.95,
                ),
                dict(
                    params=gnn_params,
                    use_muon=False,
                    lr=self.hparams.gnn_lr,
                    betas=(0.9, 0.999),
                    weight_decay=self.hparams.weight_decay,
                ),
                dict(
                    params=adamw_params,
                    use_muon=False,
                    lr=self.hparams.lr,
                    betas=(0.9, 0.999),
                    weight_decay=self.hparams.weight_decay,
                ),
            ]

            # Filter out empty param groups
            param_groups = [pg for pg in param_groups if len(pg["params"]) > 0]
            optimizer = MuonWithAuxAdam(param_groups)

        else:
            # Fallback to AdamW with two param groups (GNN vs head)
            gnn_params = [p for p in self.gnn.parameters() if p.requires_grad]
            head_params = [p for p in self.head.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW([
                {"params": gnn_params, "lr": self.hparams.gnn_lr,
                 "weight_decay": self.hparams.weight_decay},
                {"params": head_params, "lr": self.hparams.lr,
                 "weight_decay": self.hparams.weight_decay},
            ])

        # ReduceLROnPlateau: proven in node1-1-1 (F1=0.474)
        # patience=20, threshold=5e-4 matches node1-1-1's calibration exactly
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.rlrop_factor,
            patience=self.hparams.rlrop_patience,
            threshold=self.hparams.rlrop_threshold,
            threshold_mode="abs",
            min_lr=self.hparams.rlrop_min_lr,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/f1",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
        if self.head is None:
            return {}
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
        pct = f"{100 * trainable / total:.2f}%" if total > 0 else "N/A"
        self.print(
            f"Saving checkpoint: {trainable}/{total} params ({pct})"
        )
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-F1 averaged over all response genes.

    Matches data/calc_metric.py logic exactly:
    - argmax over class dim to get hard predictions
    - per-gene F1 averaged over present classes only
    - final F1 = mean over all genes
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)     # [N, N_GENES]
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        if present.any():
            f1_vals.append(float(per_class_f1[present].mean()))
        else:
            f1_vals.append(0.0)
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert len(pert_ids) == len(preds), (
        f"Length mismatch: {len(pert_ids)} ids vs {len(preds)} predictions"
    )
    rows = []
    for i in range(len(pert_ids)):
        rows.append({
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions -> {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STRING_GNN (partial fine-tuning) + 3-block MLP (hidden=384) + Muon"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gnn-lr", type=float, default=5e-5)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--rlrop-patience", type=int, default=20)
    p.add_argument("--rlrop-factor", type=float, default=0.5)
    p.add_argument("--rlrop-threshold", type=float, default=5e-4)
    p.add_argument("--rlrop-min-lr", type=float, default=1e-6)
    p.add_argument("--grad-clip-val", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=40)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    model = PerturbModule(
        hidden_dim=args.hidden_dim,
        n_genes=N_GENES,
        n_blocks=args.n_blocks,
        lr=args.lr,
        gnn_lr=args.gnn_lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        rlrop_patience=args.rlrop_patience,
        rlrop_factor=args.rlrop_factor,
        rlrop_threshold=args.rlrop_threshold,
        rlrop_min_lr=args.rlrop_min_lr,
        grad_clip_val=args.grad_clip_val,
    )

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = limit_val = limit_test = args.debug_max_step
        max_steps = args.debug_max_step
        val_check_interval = 1.0
        num_sanity_val_steps = 0
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval if not fast_dev_run else 1.0
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

    # Gradient clipping: used successfully in node1-3-2 (F1=0.4756)
    # Stabilizes training when GNN gradients and head gradients have different scales
    gradient_clip_val = args.grad_clip_val if (
        args.debug_max_step is None and not fast_dev_run
    ) else None

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(
            timeout=timedelta(seconds=120)
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    # Use best checkpoint for final test evaluation
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        result = test_results[0]
        primary_metric = result.get("test/f1", result.get("test/f1_score", float("nan")))
        score_path.write_text(str(float(primary_metric)))
        print(f"Test results -> {score_path} (f1_score={primary_metric})", flush=True)


if __name__ == "__main__":
    main()
