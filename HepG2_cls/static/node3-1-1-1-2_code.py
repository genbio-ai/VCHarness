"""Node improvement: STRING_GNN (frozen) + Unfactorized Head + DropPath + T_max=200 cosine.

Architecture Overview:
  - Precomputed frozen STRING_GNN embeddings (256-dim): proven PPI graph signal (node1-1 F1=0.472)
  - 5-block Residual MLP head with DropPath (stochastic depth): progressive rates 0→0.1
  - Unfactorized output head Linear(512→19920): proven in node1-1 (F1=0.472) and node1-1-1 (F1=0.474)
  - Single-cycle cosine annealing T_max=200: avoids LR restart instability
  - Standard weighted cross-entropy + label smoothing

Key Improvements over Parent (node3-1-1-1, F1=0.390):
  1. REMOVE factorized bottleneck (512→256→19920) → unfactorized Linear(512→19920)
     - Parent's 256-dim bottleneck forced 4:1 compression before 77× expansion to 19920
     - node1-1 (F1=0.472) and node1-1-1 (F1=0.474) used unfactorized head successfully
  2. CHANGE T_max=50 (causes LR restarts at ep50, ep100) → T_max=200 (single-cycle, no restart)
     - Parent's T_max=50 caused val/f1 oscillation 0.351→0.438→0.391 due to LR restarts
     - Single-cycle cosine monotonically decays LR over full training window
  3. ADD DropPath (stochastic depth) regularization — progressive rates 0.0→0.1 across 5 blocks
     - Randomly drops entire residual block outputs during training
     - Forces model to learn more robust representations from the 1,273-sample dataset
     - Addresses the val-loss plateau issue seen in parent/sibling lineage
     - UNTRIED in any node in the MCTS tree (novel contribution)
  4. FIX multi-GPU val/f1 computation: gather predictions across all ranks before computing F1
     - Parent and sibling compute val/f1 on per-GPU subset (~18 samples each with 8 GPUs)
     - Per-GPU F1 on 18 samples is noisy → unreliable checkpoint selection → val-test gap
     - Fix: all_gather + deduplicate by pert_id → correct F1 on all 141 samples

Key Distinction vs Sibling (node3-1-1-1-1, F1=0.392):
  - Sibling used partial STRING_GNN fine-tuning (trainable mps.6/mps.7/post_mp): counterproductive
    on 1,273 samples, caused 15.8M trainable params and destabilized PPI embeddings
  - This node uses FROZEN STRING_GNN (precomputed embeddings, same as node1-1 which got 0.472)
  - Sibling had no DropPath → this node adds stochastic depth regularization
  - Sibling T_max=200 + GNN unfreeze = 0.392; this node T_max=200 + frozen GNN + DropPath ≠ same
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
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES = 6640
N_CLASSES = 3
STRING_EMB_DIM = 256   # STRING_GNN output dim
FEATURE_DIM = STRING_EMB_DIM  # 256


# ---------------------------------------------------------------------------
# DropPath (Stochastic Depth) — new regularizer, untried in any prior node
# ---------------------------------------------------------------------------
def drop_path_func(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """Per-sample stochastic depth: randomly zero entire paths in the residual stream.

    Standard DropPath implementation as used in DeiT, Swin Transformer, etc.
    At test time (training=False), returns x unchanged (full model is used).
    At train time, each sample in the batch independently drops the path with
    probability drop_prob; the surviving paths are rescaled by 1/keep_prob to
    maintain expected magnitude.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # Shape: (batch_size, 1, 1, ...) — broadcast over all non-batch dims
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
    random_tensor.floor_()  # binarize: 1 with prob=keep_prob, 0 with prob=drop_prob
    return x / keep_prob * random_tensor  # scale surviving paths to maintain expectation


class DropPath(nn.Module):
    """Stochastic depth: drop an entire residual path per sample during training."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path_func(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ---------------------------------------------------------------------------
# Residual Block with DropPath
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Pre-norm residual block with stochastic depth.

    Architecture: LayerNorm → FC1 → GELU → Dropout → FC2 → Dropout → DropPath + residual
    DropPath randomly zeroes the entire transformation path per sample during training,
    forcing the network to learn more robust representations by not relying on any
    single block always being active.
    """

    def __init__(
        self,
        dim: int,
        expand: int = 2,
        dropout: float = 0.35,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expand)
        self.fc2 = nn.Linear(dim * expand, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        # DropPath: each block gets its own rate (progressive schedule)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        # DropPath applied to the transformed path (not the residual)
        return residual + self.drop_path(x)


# ---------------------------------------------------------------------------
# Prediction Head (Unfactorized Output — proven in node1-1/node1-1-1)
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """5-block residual MLP with unfactorized output head and stochastic depth.

    Key design choices:
    - Unfactorized Linear(512→19920): proven in node1-1 (F1=0.472) and node1-1-1 (F1=0.474).
      The parent's factorized bottleneck (512→256→19920) was too narrow (4:1 compression
      before 77× expansion), causing the primary regression to F1=0.390.
    - Progressive DropPath: rates increase linearly from 0 to drop_path_rate across blocks,
      following standard practice (DeiT, Swin). Deeper blocks have higher drop rates, acting
      as stronger regularization on later features.
    """

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = 512,
        n_genes: int = N_GENES,
        n_blocks: int = 5,
        dropout: float = 0.35,
        drop_path_rate: float = 0.10,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes

        # Input projection: FEATURE_DIM → hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Progressive drop path rates: 0.0 → drop_path_rate across n_blocks
        # Earlier blocks get lower rates (preserve low-level features)
        # Later blocks get higher rates (force robustness in deeper representations)
        dpr = [
            float(x) for x in torch.linspace(0.0, drop_path_rate, n_blocks)
        ]

        # 5 Residual MLP blocks with stochastic depth
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=2, dropout=dropout, drop_path_rate=dpr[i])
            for i in range(n_blocks)
        ])

        # Unfactorized output projection: hidden_dim → n_genes × N_CLASSES
        # This is the proven design from node1-1 (F1=0.472) and node1-1-1 (F1=0.474).
        # Parameters: 512 × 19920 + 19920 ≈ 10.2M
        self.out_proj = nn.Linear(hidden_dim, n_genes * N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)               # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)                      # [B, hidden_dim]
        out = self.out_proj(x)               # [B, N_GENES * N_CLASSES]
        return out.view(-1, N_CLASSES, self.n_genes)  # [B, 3, N_GENES]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Maps each perturbed gene to its precomputed STRING_GNN feature vector."""

    def __init__(
        self,
        df: pd.DataFrame,
        gene_features: torch.Tensor,
        ensg_to_idx: Dict[str, int],
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_features = gene_features       # [N_NODES, FEATURE_DIM] CPU float32
        self.ensg_to_idx = ensg_to_idx

        if "label" in df.columns:
            labels = np.array(
                [json.loads(x) for x in df["label"].tolist()], dtype=np.int64
            )
            # Shift labels: {-1, 0, 1} → {0, 1, 2}
            self.labels: Optional[torch.Tensor] = torch.tensor(
                labels + 1, dtype=torch.long
            )
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pert_id = self.pert_ids[idx]
        gnn_idx = self.ensg_to_idx.get(pert_id, -1)

        if gnn_idx >= 0:
            feat = self.gene_features[gnn_idx]   # [FEATURE_DIM]
        else:
            # Fallback: zero vector for genes not in STRING graph (~7% of data)
            feat = torch.zeros(self.gene_features.shape[1])

        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": pert_id,
            "symbol": self.symbols[idx],
            "features": feat,
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

        self.gene_features: Optional[torch.Tensor] = None
        self.ensg_to_idx: Optional[Dict[str, int]] = None
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        # Precompute STRING_GNN features (run once per process)
        if self.gene_features is None:
            self._precompute_features()

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df, self.gene_features, self.ensg_to_idx)
        self.val_ds = PerturbDataset(val_df, self.gene_features, self.ensg_to_idx)
        self.test_ds = PerturbDataset(test_df, self.gene_features, self.ensg_to_idx)

    def _precompute_features(self) -> None:
        """Run STRING_GNN forward once to get frozen PPI topology embeddings [N, 256].

        Using frozen STRING_GNN is the proven strategy from node1-1 (F1=0.472).
        node3-1-1-1-1 (sibling) showed partial GNN fine-tuning is counterproductive
        on 1,273 samples — it destabilized PPI embeddings and caused F1=0.392.

        DDP-safe: all ranks precompute independently (deterministic, identical results).
        """
        import torch.distributed as dist

        model_dir = Path(STRING_GNN_DIR)
        is_dist = dist.is_available() and dist.is_initialized()
        if is_dist:
            dist.barrier()

        # Build node index map
        node_names: List[str] = json.loads(
            (model_dir / "node_names.json").read_text()
        )
        self.ensg_to_idx = {name: i for i, name in enumerate(node_names)}

        device = torch.device("cuda")
        print("Loading STRING_GNN for precomputing topology embeddings...", flush=True)
        gnn = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device)
        gnn.eval()

        graph = torch.load(model_dir / "graph_data.pt", map_location=device)
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)

        with torch.no_grad():
            out = gnn(
                edge_index=edge_index,
                edge_weight=edge_weight,
                output_hidden_states=False,
            )
            self.gene_features = out.last_hidden_state.float().cpu()  # [N, 256]

        del gnn, graph, out
        torch.cuda.empty_cache()

        print(
            f"Precomputed gene features: {self.gene_features.shape} "
            f"(STRING_GNN frozen PPI topology)",
            flush=True,
        )

        if is_dist:
            dist.barrier()

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
    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = 512,
        n_genes: int = N_GENES,
        n_blocks: int = 5,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        dropout: float = 0.35,
        label_smoothing: float = 0.05,
        drop_path_rate: float = 0.10,
        t_max: int = 200,
        eta_min: float = 1e-7,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.head: Optional[PerturbHead] = None

        # Accumulation buffers for validation and test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_pert_ids: List[str] = []

        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        self.head = PerturbHead(
            in_dim=self.hparams.in_dim,
            hidden_dim=self.hparams.hidden_dim,
            n_genes=self.hparams.n_genes,
            n_blocks=self.hparams.n_blocks,
            dropout=self.hparams.dropout,
            drop_path_rate=self.hparams.drop_path_rate,
        )

        # Cast to float32 for stable optimization
        for p in self.head.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Class weights: shifted labels {0:down, 1:neutral, 2:up}
        # Frequencies from DATA_ABSTRACT: down=4.77%, neutral=92.82%, up=2.41%
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = 1.0 / freq
        class_weights = class_weights / class_weights.mean()
        self.register_buffer("class_weights", class_weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"STRING_GNN-only (frozen) + Unfactorized Head + DropPath | "
            f"trainable={trainable:,}/{total:,} | "
            f"in={self.hparams.in_dim}, hidden={self.hparams.hidden_dim}, "
            f"blocks={self.hparams.n_blocks}, dropout={self.hparams.dropout}, "
            f"drop_path={self.hparams.drop_path_rate}, T_max={self.hparams.t_max}"
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy with label smoothing.

        Proven in node1-1 (F1=0.472). Focal loss confirmed harmful in node3-1 (F1=0.157).
        """
        # logits: [B, 3, N_GENES], labels: [B, N_GENES] in {0,1,2}
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        feats = batch["features"].to(self.device).float()
        logits = self.head(feats)
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        feats = batch["features"].to(self.device).float()
        logits = self.head(feats)
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())
        self._val_pert_ids.extend(batch["pert_id"])

    def on_validation_epoch_end(self) -> None:
        """Compute val/f1 correctly across all GPUs.

        FIX vs parent/sibling: gather all predictions across ranks before computing F1.
        Parent/sibling compute val/f1 per-GPU (only ~18 samples per GPU with 8 GPUs),
        which is noisy and causes unreliable checkpoint selection. This fix computes
        val/f1 on all 141 validation samples after gathering across all ranks.
        """
        import torch.distributed as dist

        if not self._val_preds:
            return

        preds_local = torch.cat(self._val_preds, dim=0)   # [local_N, 3, N_GENES]
        labels_local = torch.cat(self._val_labels, dim=0) # [local_N, N_GENES]
        pert_ids_local = list(self._val_pert_ids)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_pert_ids.clear()

        # Gather from all ranks to get the full validation set
        all_preds = self.all_gather(preds_local)   # [world_size, local_N, 3, N_GENES]
        all_labels = self.all_gather(labels_local) # [world_size, local_N, N_GENES]

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)  # [total_N, 3, N_GENES]
        all_labels = all_labels.view(-1, N_GENES)            # [total_N, N_GENES]

        # Gather pert_ids for deduplication (handles DDP padding)
        gathered_pert_ids = [pert_ids_local]
        if world_size > 1:
            obj = [None] * world_size
            dist.all_gather_object(obj, pert_ids_local)
            gathered_pert_ids = obj

        all_pert_ids = [p for rank_list in gathered_pert_ids for p in rank_list]

        # Deduplicate: DDP with drop_last=False may pad the dataset, creating duplicates
        seen: set = set()
        dedup_indices: List[int] = []
        for i, pid in enumerate(all_pert_ids):
            if pid not in seen:
                seen.add(pid)
                dedup_indices.append(i)

        dedup_preds = all_preds[dedup_indices]   # [n_unique, 3, N_GENES]
        dedup_labels = all_labels[dedup_indices] # [n_unique, N_GENES]

        # Compute per-gene macro F1 on all deduped validation samples
        f1 = _compute_per_gene_f1(
            dedup_preds.float().cpu().numpy(),
            dedup_labels.cpu().numpy(),
        )
        # Log without sync_dist=False (already global after all_gather + dedup on rank 0).
        # Lightning warns about sync_dist=True for epoch-level logs in DDP, but this
        # metric is pre-gathered so no additional distributed reduction is needed.
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        feats = batch["features"].to(self.device).float()
        logits = self.head(feats)
        self._test_preds.append(logits.detach().cpu().float())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)    # [local_N, 3, N_GENES]
        labels_local = (
            torch.cat(self._test_labels, dim=0) if self._test_labels else None
        )
        self._test_preds.clear()
        self._test_labels.clear()

        # Gather predictions from all ranks
        all_preds = self.all_gather(preds_local)  # [world_size, local_N, 3, N_GENES]
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)  # [total_N, 3, N_GENES]

        # Gather labels (if available)
        if labels_local is not None:
            all_labels = self.all_gather(labels_local)  # [world_size, local_N, N_GENES]
            all_labels = all_labels.view(-1, N_GENES)   # [total_N, N_GENES]

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

            dedup_preds_np = np.stack(dedup_preds, axis=0)

            # Compute test F1 if labels are available
            if labels_local is not None:
                all_labels_np = all_labels.cpu().numpy()
                # Build dedup labels in same order as dedup_ids
                pid_to_label: Dict[str, np.ndarray] = {}
                all_pert_list = [p for rank_list in gathered_pert_ids for p in rank_list]
                for i, pid in enumerate(all_pert_list):
                    if pid not in pid_to_label:
                        pid_to_label[pid] = all_labels_np[i]
                dedup_labels = np.stack([pid_to_label[pid] for pid in dedup_ids], axis=0)

                test_f1 = _compute_per_gene_f1(dedup_preds_np, dedup_labels)
                # sync_dist=False: this runs inside is_global_zero, no DDP sync needed.
                self.log("test/f1", test_f1, prog_bar=True, sync_dist=False)

            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=dedup_preds_np,
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Single-cycle CosineAnnealingLR with T_max=max_epochs=200.
        # This avoids the LR restart instability from T_max=50 in the parent,
        # which caused val/f1 oscillation (0.351→0.438→0.391) due to LR resetting
        # to 3e-4 at epochs 50, 100, 150. With T_max=200=max_epochs, the LR
        # decays monotonically over the full training window — no restarts.
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.t_max,
            eta_min=self.hparams.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": cosine_scheduler, "interval": "epoch"},
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters + buffers."""
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
            f"Saving checkpoint: {trainable}/{total} params "
            f"({100 * trainable / total:.2f}%)"
        )
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-F1 averaged over all 6,640 response genes.

    Exactly matches data/calc_metric.py logic:
    - argmax over class dim → hard predictions
    - per-gene F1 averaged over present classes only
    - final score = mean over all genes
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
    print(f"Saved {len(rows)} test predictions → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STRING_GNN (frozen) + Unfactorized Head + DropPath + Single-Cycle Cosine"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--drop-path-rate", type=float, default=0.10,
                   help="Maximum DropPath rate (progressive: 0 → drop_path_rate across blocks)")
    p.add_argument("--t-max", type=int, default=200,
                   help="CosineAnnealingLR T_max; set equal to max-epochs for single cycle")
    p.add_argument("--eta-min", type=float, default=1e-7)
    p.add_argument("--early-stop-patience", type=int, default=40)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None,
                   help="Limit steps for quick debugging")
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--val-check-interval", type=float, default=1.0)
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
        in_dim=FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        n_genes=N_GENES,
        n_blocks=args.n_blocks,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        drop_path_rate=args.drop_path_rate,
        t_max=args.t_max,
        eta_min=args.eta_min,
    )

    fast_dev_run = args.fast_dev_run
    debug_max_step = args.debug_max_step
    if debug_max_step is not None:
        limit_train = limit_val = limit_test = debug_max_step
        max_steps = debug_max_step
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

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(
            find_unused_parameters=False, timeout=timedelta(seconds=120)
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
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    # Use best checkpoint for final test evaluation
    if fast_dev_run or debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        result = test_results[0]
        primary_metric = result.get("test/f1", result.get("test/f1_score", float("nan")))
        score_path.write_text(str(float(primary_metric)))
        print(f"Test results → {score_path} (f1_score={primary_metric})", flush=True)


if __name__ == "__main__":
    main()
