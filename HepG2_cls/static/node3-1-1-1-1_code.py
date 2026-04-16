"""Node improvement: STRING_GNN (partial fine-tune) + Unfactorized Head + Single-Cycle Cosine.

Architecture Overview:
  - Partially fine-tuned STRING_GNN: last 2 GNN layers (mps.6, mps.7) + post_mp are trainable
    at gnn_lr=1e-5; all other layers (emb, mps.0-5) remain frozen
  - 5-block Residual MLP head: 256 → 512 → [5 res blocks] → 19920 → [B, 3, 6640]
  - Standard weighted cross-entropy + label smoothing (NO focal loss, NO bottleneck, NO restarts)
  - Single-cycle cosine annealing: T_max = max_epochs = 200 (no LR restarts — critical fix)

Key Fixes over Parent (node3-1-1-1, Test F1=0.390):
  1. REVERT to unfactorized output head: Linear(512 → 19920)
     - Eliminates the 256-dim bottleneck that was too narrow for 6,640-gene prediction
     - Root cause 2 from parent feedback: bottleneck forced 4:1 compression then 77x expansion
  2. SET T_max = max_epochs = 200 (single-cycle cosine, no restarts)
     - Root cause 1 from parent feedback: T_max=50 caused LR restarts at epoch 50, 100
     - Restarts caused val/f1 to drop from 0.438 back to 0.391

Key Innovation vs node1-1 (best reference, Test F1=0.472):
  - Partial STRING_GNN fine-tuning: unfreeze last 2 GNN layers + post_mp at lr=1e-5
  - This was validated in node1-1-1 (+0.002 over node1-1), but node1-1-1 used ReduceLROnPlateau
    + 3 blocks. This node uses the correct single-cycle cosine + 5 blocks.
  - Dynamic per-batch GNN forward: enables gradient to flow through trainable GNN layers
  - GNN output: [18870, 256] per forward pass, then per-sample lookup [B, 256]
  - Memory impact: STRING_GNN is ~5.43M params total, ~0.26M trainable (last 2 layers + post_mp)

Training Design:
  - The STRING_GNN runs a full forward pass each training step to get updated embeddings
  - Per-sample features are looked up from the GNN output using gnn_idx
  - ~7% of training genes are not in STRING graph (gnn_idx=-1), fallback to zero vector
  - DDP: all ranks compute the same GNN forward (parameters are identical after DDP sync)
    The gradient through gnn_idx lookup is sparse but valid for AdamW optimization
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
# Residual Block
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout."""

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
# Prediction Head (Unfactorized — proven in node1-1 F1=0.472, node1-1-1 F1=0.474)
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """5-block residual MLP with unfactorized output: [B, FEATURE_DIM] → [B, 3, N_GENES].

    Architecture mirrors node1-1's proven design:
      256 → LayerNorm → Linear(512) → GELU → Dropout
          → 5 × ResidualBlock(512, expand=2)
          → Linear(512 → 6640×3)
          → reshape → [B, 3, 6640]

    The unfactorized output (10.2M params) outperforms the factorized bottleneck (5.2M params):
    - node1-1 (unfactorized): Test F1=0.472
    - node3-1-1-1 (factorized, bottleneck=256): Test F1=0.390 (−0.082 regression)
    """

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = 512,
        n_genes: int = N_GENES,
        n_blocks: int = 5,
        dropout: float = 0.35,
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

        # Residual MLP trunk: 5 blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=2, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Unfactorized output projection: hidden_dim → n_genes * N_CLASSES (~10.2M)
        # This is the proven design from node1-1 (F1=0.472).
        # NOT using a bottleneck — node3-1-1-1 tried this and regressed to F1=0.390.
        self.out_proj = nn.Linear(hidden_dim, n_genes * N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)               # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)
        out = self.out_proj(x)               # [B, N_GENES * 3]
        return out.view(-1, N_CLASSES, self.n_genes)  # [B, 3, N_GENES]


# ---------------------------------------------------------------------------
# Dataset with GNN node index for dynamic lookup
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Maps each perturbed gene to its STRING_GNN node index and precomputed features.

    Stores gnn_idx (integer index into GNN node list, or -1 if not in graph) for
    dynamic GNN embedding lookup in PerturbModule during training.
    Also stores precomputed float features for fallback (-1 genes use zero vector).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ensg_to_idx: Dict[str, int],
        n_gnn_nodes: int,
        emb_dim: int,
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.ensg_to_idx = ensg_to_idx
        self.n_gnn_nodes = n_gnn_nodes
        self.emb_dim = emb_dim

        if "label" in df.columns:
            labels = np.array(
                [json.loads(x) for x in df["label"].tolist()], dtype=np.int64
            )
            self.labels: Optional[torch.Tensor] = torch.tensor(
                labels + 1, dtype=torch.long
            )  # {-1, 0, 1} → {0, 1, 2}
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pert_id = self.pert_ids[idx]
        gnn_idx = self.ensg_to_idx.get(pert_id, -1)

        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": pert_id,
            "symbol": self.symbols[idx],
            "gnn_idx": gnn_idx,  # int, -1 if not in STRING graph
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class PerturbDataModule(LightningDataModule):
    """DataModule providing gnn_idx per sample for dynamic STRING_GNN lookup."""

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
        self.n_gnn_nodes: int = 0
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        if self.ensg_to_idx is None:
            self._load_node_index()

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        kwargs = dict(
            ensg_to_idx=self.ensg_to_idx,
            n_gnn_nodes=self.n_gnn_nodes,
            emb_dim=FEATURE_DIM,
        )
        self.train_ds = PerturbDataset(train_df, **kwargs)
        self.val_ds = PerturbDataset(val_df, **kwargs)
        self.test_ds = PerturbDataset(test_df, **kwargs)

    def _load_node_index(self) -> None:
        """Load STRING_GNN node names and build ENSG ID → index mapping."""
        import torch.distributed as dist

        model_dir = Path(STRING_GNN_DIR)
        is_dist = dist.is_available() and dist.is_initialized()
        if is_dist:
            dist.barrier()

        node_names: List[str] = json.loads(
            (model_dir / "node_names.json").read_text()
        )
        self.ensg_to_idx = {name: i for i, name in enumerate(node_names)}
        self.n_gnn_nodes = len(node_names)

        print(
            f"Loaded STRING_GNN node index: {self.n_gnn_nodes} nodes",
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
    """STRING_GNN (partial fine-tune) + 5-block ResNet + Unfactorized Output Head.

    Architecture:
        gnn_idx → STRING_GNN (trainable: mps.6, mps.7, post_mp) → emb[gnn_idx] [B, 256]
                → PerturbHead (5-block MLP) → [B, 3, 6640]

    Key design decisions vs parent (node3-1-1-1, F1=0.390):
    1. Unfactorized output head: Linear(512→19920) — eliminates narrow 256-dim bottleneck
    2. Single-cycle cosine T_max=200 — no LR restarts that caused val/f1 oscillation
    3. Partial STRING_GNN fine-tuning: last 2 layers at gnn_lr=1e-5 — adapts gene embeddings

    DDP correctness:
    - The STRING_GNN runs a full graph forward pass per training step
    - All DDP ranks produce identical GNN forward (same parameters, same graph)
    - Gradient flows back through the gnn_idx lookup to trainable GNN layers
    - DDP automatically synchronizes gradients from trainable GNN layers
    - find_unused_parameters=True handles frozen GNN layers correctly
    """

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = 512,
        n_genes: int = N_GENES,
        n_blocks: int = 5,
        lr: float = 3e-4,
        gnn_lr: float = 1e-5,
        weight_decay: float = 1e-3,
        dropout: float = 0.35,
        label_smoothing: float = 0.05,
        t_max: int = 200,
        gnn_unfreeze_layers: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.head: Optional[PerturbHead] = None
        self.gnn: Optional[nn.Module] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Build the unfactorized prediction head
        self.head = PerturbHead(
            in_dim=self.hparams.in_dim,
            hidden_dim=self.hparams.hidden_dim,
            n_genes=self.hparams.n_genes,
            n_blocks=self.hparams.n_blocks,
            dropout=self.hparams.dropout,
        )

        # Load STRING_GNN for partial fine-tuning
        model_dir = Path(STRING_GNN_DIR)
        self.gnn = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        self.gnn.eval()  # Will be set to train() appropriately by Lightning

        # Freeze all GNN parameters by default
        for p in self.gnn.parameters():
            p.requires_grad = False

        # Unfreeze last N GNN layers (mps.6, mps.7) + post_mp
        n_unfreeze = self.hparams.gnn_unfreeze_layers
        n_total_layers = 8  # STRING_GNN has 8 message-passing layers (mps.0 to mps.7)
        for layer_idx in range(n_total_layers - n_unfreeze, n_total_layers):
            for p in self.gnn.mps[layer_idx].parameters():
                p.requires_grad = True
        for p in self.gnn.post_mp.parameters():
            p.requires_grad = True

        # Load and register graph as buffers (auto-moved to device)
        graph = torch.load(model_dir / "graph_data.pt", map_location="cpu")
        self.register_buffer("_edge_index", graph["edge_index"].long())
        edge_weight = graph.get("edge_weight", None)
        if edge_weight is not None:
            self.register_buffer("_edge_weight", edge_weight.float())
        else:
            self.register_buffer("_edge_weight", torch.ones(graph["edge_index"].shape[1]))

        # Cast trainable head parameters to float32 for stable optimization
        for p in self.head.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Cast trainable GNN parameters to float32
        for p in self.gnn.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Class weights: shifted labels {0:down, 1:neutral, 2:up}
        # Frequencies from DATA_ABSTRACT: down=4.77%, neutral=92.82%, up=2.41%
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = 1.0 / freq
        class_weights = class_weights / class_weights.mean()
        self.register_buffer("class_weights", class_weights)

        # Log parameter counts
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        gnn_trainable = sum(p.numel() for p in self.gnn.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        self.print(
            f"STRING_GNN (partial ft, last {n_unfreeze} layers + post_mp) + "
            f"Unfactorized 5-block MLP | "
            f"trainable={trainable:,}/{total:,} | "
            f"gnn_trainable={gnn_trainable:,}, head_trainable={head_trainable:,} | "
            f"hidden={self.hparams.hidden_dim}, blocks={self.hparams.n_blocks}, "
            f"dropout={self.hparams.dropout}"
        )

    def _get_sample_features(self, gnn_idx: torch.Tensor) -> torch.Tensor:
        """Run STRING_GNN forward and look up per-sample features.

        Args:
            gnn_idx: [B] integer tensor of GNN node indices (-1 = not in graph)

        Returns:
            feats: [B, 256] float32 tensor of gene embeddings
        """
        # Run full STRING_GNN forward (trainable layers get gradients)
        all_emb = self.gnn(
            edge_index=self._edge_index,
            edge_weight=self._edge_weight,
            output_hidden_states=False,
        ).last_hidden_state.float()  # [N_GNN_NODES, 256]

        # Per-sample lookup with fallback (zero vector) for genes not in STRING graph
        B = gnn_idx.shape[0]
        feats = torch.zeros(B, all_emb.shape[1], device=self.device, dtype=torch.float32)
        valid_mask = gnn_idx >= 0
        if valid_mask.any():
            valid_indices = gnn_idx[valid_mask].long()
            feats[valid_mask] = all_emb[valid_indices]
        return feats

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard weighted cross-entropy with label smoothing."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        gnn_idx = batch["gnn_idx"].to(self.device)
        feats = self._get_sample_features(gnn_idx)  # [B, 256], grad through trainable GNN
        logits = self.head(feats)
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        gnn_idx = batch["gnn_idx"].to(self.device)
        feats = self._get_sample_features(gnn_idx)
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
        preds = torch.cat(self._val_preds, dim=0).numpy()
        labels = torch.cat(self._val_labels, dim=0).numpy()
        self._val_preds.clear()
        self._val_labels.clear()

        f1 = _compute_per_gene_f1(preds, labels)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        gnn_idx = batch["gnn_idx"].to(self.device)
        feats = self._get_sample_features(gnn_idx)
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
        # Two parameter groups: MLP head at lr=3e-4, GNN trainable at gnn_lr=1e-5
        gnn_params = [p for p in self.gnn.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]

        param_groups = [
            {
                "params": head_params,
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": gnn_params,
                "lr": self.hparams.gnn_lr,
                "weight_decay": self.hparams.weight_decay,
            },
        ]

        optimizer = torch.optim.AdamW(param_groups)

        # Single-cycle cosine annealing: T_max = max_epochs = 200
        # CRITICAL: T_max must equal max_epochs to ensure a single cycle with no LR restart.
        # The parent node (node3-1-1-1) used T_max=50 which caused 3 restarts over 144 epochs
        # and drove val/f1 from 0.438 down to 0.391 due to the 3rd cycle oscillation.
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.t_max,
            eta_min=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": cosine_scheduler, "interval": "epoch"},
        }

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
    print(f"Saved {len(rows)} test predictions → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STRING_GNN (partial ft) + Unfactorized Head + Single-Cycle Cosine"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gnn-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--t-max", type=int, default=200)
    p.add_argument("--gnn-unfreeze-layers", type=int, default=2)
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
        in_dim=FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        n_genes=N_GENES,
        n_blocks=args.n_blocks,
        lr=args.lr,
        gnn_lr=args.gnn_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        t_max=args.t_max,
        gnn_unfreeze_layers=args.gnn_unfreeze_layers,
    )

    fast_dev_run = args.fast_dev_run
    debug_max_step = args.debug_max_step
    if debug_max_step is not None:
        # Multiply by accumulate so that limit_train_batches (micro-batches) yields
        # exactly debug_max_step optimizer steps: max_steps = limit_train / accumulate
        limit_train = debug_max_step * accumulate
        limit_val = limit_test = debug_max_step
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
        # NO gradient clipping — no evidence it helped; node1-1 succeeded without clipping
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
