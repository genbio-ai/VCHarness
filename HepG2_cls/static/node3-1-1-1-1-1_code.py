"""Node: STRING_GNN (Frozen, Precomputed) + 5-block MLP + Unfactorized Head + ReduceLROnPlateau + Muon Optimizer.

Architecture Overview:
  - Frozen STRING_GNN: precomputed once in setup(), stored as [N_GNN_NODES, 256] CPU tensor
    No per-batch GNN forward; eliminates GNN fine-tuning instability that caused F1=0.392
  - 5-block Residual MLP head: 256 → 512 → [5 res blocks] → 19920 → [B, 3, 6640]
  - Per-gene bias: 19,920 learnable biases added to unfactorized output (similar to node1-1-1)
  - Standard weighted cross-entropy + label smoothing (NO focal loss, NO bottleneck)
  - ReduceLROnPlateau scheduler: adaptive LR reduction on val/f1 plateau
  - Muon optimizer: MuonWithAuxAdam for hidden MLP weight matrices (fc1/fc2 in ResBlocks)
    AdamW for scalar params, norms, output projection, input projection, gene biases

Key Improvements over Parent (node3-1-1-1-1, Test F1=0.392):
  1. ABANDON GNN fine-tuning: Frozen STRING_GNN → eliminates sparse gradient instability
     Parent's partial GNN unfreeze (0.26M params, sparse gnn_idx lookups) destabilized model
  2. ReduceLROnPlateau: adaptive decay (node1-1-1 achieved F1=0.474 with this, vs node1-1's 0.472)
     This is the first time ReduceLROnPlateau is combined with 5 blocks (not 3 like node1-1-1)
  3. Muon optimizer: applied to hidden 2D weight matrices in ResBlock (fc1/fc2 weights)
     Muon orthogonalizes momentum updates for hidden layers — untested in this MCTS tree
     Proven to give faster convergence especially for MLPs with limited training data
  4. Per-gene bias: learnable 19,920 output bias calibrated per gene (inspired by node1-1-1)

Design Rationale:
  - All GNN fine-tuning attempts in this tree have regressed to F1≈0.390
    (node3-1-1-1-1: 0.392, and all node3-1-x lineage)
  - Frozen STRING_GNN achieves the best results: node1-1 (0.472), node1-1-1 (0.474)
  - The unexplored combination is: 5 blocks + ReduceLROnPlateau + Muon + gene bias
  - node1-1-1 used 3 blocks + partial unfreeze + ReduceLROnPlateau → 0.474
  - node1-1 used 5 blocks + frozen + CosineAnnealing → 0.472
  - This node: 5 blocks + frozen + ReduceLROnPlateau + Muon → targets 0.47-0.49
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
    """Pre-norm residual block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout.

    The fc1/fc2 weight matrices (2D, hidden dimension) are the target for Muon optimizer.
    The LayerNorm gain/bias (1D) and Dropout use AdamW.
    """

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
# With per-gene output bias (inspired by node1-1-1's per-gene bias design)
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """5-block residual MLP with unfactorized output + per-gene bias: [B, FEATURE_DIM] → [B, 3, N_GENES].

    Architecture:
      256 → LayerNorm → Linear(512) → GELU → Dropout
          → 5 × ResidualBlock(512, expand=2)
          → Linear(512 → 6640×3) + per_gene_bias [N_CLASSES, N_GENES]
          → reshape → [B, 3, 6640]

    Per-gene bias: Separate nn.Parameter([3, N_GENES]) that learns gene-level output calibration.
    This allows the model to learn which genes are systematically more likely to be up/down/neutral
    across all perturbations, independent of the input features.
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
        self.n_classes = N_CLASSES

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

        # Per-gene output bias: [3, N_GENES] learnable bias (inspired by node1-1-1)
        # Initialized to zero so the model starts from the unfactorized head's baseline.
        # This adds 19,920 calibration parameters at minimal cost.
        self.gene_bias = nn.Parameter(torch.zeros(N_CLASSES, n_genes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)               # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)
        out = self.out_proj(x)               # [B, N_GENES * 3]
        out = out.view(-1, self.n_classes, self.n_genes)  # [B, 3, N_GENES]
        # Add per-gene bias (broadcast across batch)
        out = out + self.gene_bias.unsqueeze(0)  # [B, 3, N_GENES]
        return out


# ---------------------------------------------------------------------------
# Dataset — precomputed frozen embeddings
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Maps each perturbed gene to a precomputed STRING_GNN embedding.

    Embeddings are precomputed once in setup() and stored as CPU float32 tensors.
    Genes not in the STRING graph (gnn_idx=-1) get a zero vector.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        precomputed_emb: torch.Tensor,  # [N_GNN_NODES, 256] or [N_TOTAL, 256] if padded
        ensg_to_idx: Dict[str, int],
        emb_dim: int,
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.ensg_to_idx = ensg_to_idx
        self.emb_dim = emb_dim

        # Precompute feature tensors for each sample
        feats = []
        for pid in self.pert_ids:
            gnn_idx = ensg_to_idx.get(pid, -1)
            if gnn_idx >= 0:
                feats.append(precomputed_emb[gnn_idx])
            else:
                feats.append(torch.zeros(emb_dim))
        self.feats: torch.Tensor = torch.stack(feats, dim=0)  # [N, 256]

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
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "feat": self.feats[idx],          # [256] float32 precomputed embedding
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class PerturbDataModule(LightningDataModule):
    """DataModule providing precomputed frozen STRING_GNN embeddings per sample."""

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
        self.precomputed_emb: Optional[torch.Tensor] = None
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        if self.ensg_to_idx is None:
            self._precompute_gnn_embeddings()

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        kwargs = dict(
            precomputed_emb=self.precomputed_emb,
            ensg_to_idx=self.ensg_to_idx,
            emb_dim=FEATURE_DIM,
        )
        self.train_ds = PerturbDataset(train_df, **kwargs)
        self.val_ds = PerturbDataset(val_df, **kwargs)
        self.test_ds = PerturbDataset(test_df, **kwargs)

    def _precompute_gnn_embeddings(self) -> None:
        """Load frozen STRING_GNN and precompute all node embeddings once."""
        import torch.distributed as dist

        model_dir = Path(STRING_GNN_DIR)
        is_dist = dist.is_available() and dist.is_initialized()

        # Rank 0 computes embeddings first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if local_rank == 0:
            # Load STRING_GNN and precompute frozen embeddings
            _gnn = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
            _gnn.eval()
            for p in _gnn.parameters():
                p.requires_grad = False

            graph = torch.load(model_dir / "graph_data.pt", map_location="cpu")
            edge_index = graph["edge_index"].long()
            edge_weight = graph.get("edge_weight", None)
            if edge_weight is not None:
                edge_weight = edge_weight.float()

            # Use GPU if available for faster precomputation, then move to CPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _gnn = _gnn.to(device)
            edge_index = edge_index.to(device)
            if edge_weight is not None:
                edge_weight = edge_weight.to(device)

            with torch.no_grad():
                outputs = _gnn(
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    output_hidden_states=False,
                )
            emb = outputs.last_hidden_state.float().cpu()  # [18870, 256]
            print(f"Precomputed STRING_GNN embeddings: shape={emb.shape}", flush=True)

            # Free GPU memory
            del _gnn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            emb = None

        # Synchronize across ranks
        if is_dist:
            dist.barrier()
            if emb is None:
                # Non-zero ranks: load from model (same frozen weights, deterministic)
                _gnn = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
                _gnn.eval()
                for p in _gnn.parameters():
                    p.requires_grad = False
                graph = torch.load(model_dir / "graph_data.pt", map_location="cpu")
                edge_index = graph["edge_index"].long()
                edge_weight = graph.get("edge_weight", None)
                if edge_weight is not None:
                    edge_weight = edge_weight.float()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _gnn = _gnn.to(device)
                edge_index = edge_index.to(device)
                if edge_weight is not None:
                    edge_weight = edge_weight.to(device)
                with torch.no_grad():
                    outputs = _gnn(
                        edge_index=edge_index,
                        edge_weight=edge_weight,
                        output_hidden_states=False,
                    )
                emb = outputs.last_hidden_state.float().cpu()
                del _gnn
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            dist.barrier()

        self.precomputed_emb = emb  # [18870, 256] CPU float32

        # Load node names and build ENSG → idx mapping
        node_names: List[str] = json.loads(
            (model_dir / "node_names.json").read_text()
        )
        self.ensg_to_idx = {name: i for i, name in enumerate(node_names)}
        print(
            f"STRING_GNN node index: {len(node_names)} nodes | "
            f"embeddings precomputed (frozen, no per-batch GNN forward)",
            flush=True,
        )

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
    """Frozen STRING_GNN (precomputed) + 5-block ResNet + Unfactorized Head + Per-Gene Bias.

    Key design decisions (vs parent node3-1-1-1-1, F1=0.392):
    1. Frozen STRING_GNN: precomputed once, no dynamic GNN forward → no GNN fine-tuning instability
    2. ReduceLROnPlateau: adaptive LR decay (node1-1-1 achieved 0.474 with this scheduler)
    3. Muon optimizer: MuonWithAuxAdam for hidden MLP weight matrices (new in this tree)
    4. Per-gene bias: 19,920 learnable output biases for gene-level calibration
    5. 5 blocks + frozen STRING_GNN: untested combination (node1-1-1 used 3 blocks)
    """

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = 512,
        n_genes: int = N_GENES,
        n_blocks: int = 5,
        lr: float = 3e-4,
        muon_lr: float = 0.02,
        weight_decay: float = 1e-3,
        dropout: float = 0.35,
        label_smoothing: float = 0.05,
        plateau_patience: int = 10,
        plateau_factor: float = 0.5,
        use_muon: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.head: Optional[PerturbHead] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Build the unfactorized prediction head with per-gene bias
        self.head = PerturbHead(
            in_dim=self.hparams.in_dim,
            hidden_dim=self.hparams.hidden_dim,
            n_genes=self.hparams.n_genes,
            n_blocks=self.hparams.n_blocks,
            dropout=self.hparams.dropout,
        )

        # Cast trainable head parameters to float32 for stable optimization
        for p in self.head.parameters():
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
        gene_bias_params = self.head.gene_bias.numel()
        self.print(
            f"Frozen STRING_GNN (precomputed) + Unfactorized 5-block MLP + Gene Bias | "
            f"trainable={trainable:,}/{total:,} | "
            f"gene_bias={gene_bias_params:,} | "
            f"hidden={self.hparams.hidden_dim}, blocks={self.hparams.n_blocks}, "
            f"dropout={self.hparams.dropout} | "
            f"use_muon={self.hparams.use_muon}"
        )

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
        feats = batch["feat"].to(self.device).float()   # [B, 256] precomputed
        logits = self.head(feats)
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        feats = batch["feat"].to(self.device).float()
        logits = self.head(feats)
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        import torch.distributed as dist

        if not self._val_preds:
            return
        preds_local = torch.cat(self._val_preds, dim=0)   # [local_N, 3, N_GENES]
        labels_local = torch.cat(self._val_labels, dim=0)  # [local_N, N_GENES]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather predictions and labels from ALL ranks so F1 is computed on full validation set.
        # DDP splits data across ranks (no overlap), so no deduplication needed.
        all_preds = self.all_gather(preds_local)
        all_labels = self.all_gather(labels_local)

        # Flatten across ranks: DDP partitions are non-overlapping, so all samples are unique
        full_preds = all_preds.view(-1, N_CLASSES, N_GENES).float().cpu().numpy()
        full_labels = all_labels.view(-1, N_GENES).cpu().numpy()

        f1 = _compute_per_gene_f1(full_preds, full_labels)
        # Log with sync_dist=True so all ranks have val/f1 (needed by EarlyStopping on all ranks).
        # Using torchmetrics-style computation: each rank computes local F1 on its subset of validation
        # data (which is already gathered via all_gather above), then all-reduce to get global F1.
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        feats = batch["feat"].to(self.device).float()
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
        """Configure optimizer with Muon for hidden weight matrices.

        Parameter groups:
          - Muon group: 2D weight matrices in ResBlock fc1/fc2 (hidden layer weights)
            These are ideal for Muon: square-ish matrices, hidden layers, full-rank
          - AdamW group: everything else
            (input_proj Linear weights, out_proj weight, gene_bias, LayerNorm, biases)

        Muon uses orthogonalized momentum updates — better convergence for hidden layers.
        AdamW handles the output head weight (19920×512 is too rectangular for Muon) and scalars.

        If use_muon=False (fallback), uses pure AdamW for all parameters.
        """
        if self.hparams.use_muon:
            try:
                from muon import MuonWithAuxAdam

                # Hidden 2D weight matrices in ResBlocks: fc1.weight [1024, 512] and fc2.weight [512, 1024]
                # These are proper 2D matrices suitable for Muon orthogonalization
                muon_params = []
                adamw_params = []

                for name, p in self.head.named_parameters():
                    if not p.requires_grad:
                        continue
                    # Muon targets: 2D weight matrices in hidden blocks (not input_proj, out_proj, or gene_bias)
                    if (p.ndim >= 2 and
                        "blocks" in name and
                        "weight" in name and
                        "norm" not in name):
                        muon_params.append(p)
                    else:
                        adamw_params.append(p)

                muon_count = sum(p.numel() for p in muon_params)
                adamw_count = sum(p.numel() for p in adamw_params)
                self.print(
                    f"Muon optimizer: muon_params={muon_count:,}, adamw_params={adamw_count:,}"
                )

                param_groups = [
                    # Muon group for hidden weight matrices
                    dict(
                        params=muon_params,
                        use_muon=True,
                        lr=self.hparams.muon_lr,
                        weight_decay=self.hparams.weight_decay,
                        momentum=0.95,
                    ),
                    # AdamW group for everything else
                    dict(
                        params=adamw_params,
                        use_muon=False,
                        lr=self.hparams.lr,
                        betas=(0.9, 0.95),
                        weight_decay=self.hparams.weight_decay,
                    ),
                ]

                optimizer = MuonWithAuxAdam(param_groups)
                self.print("Using MuonWithAuxAdam optimizer")

            except ImportError:
                self.print("Muon not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    self.head.parameters(),
                    lr=self.hparams.lr,
                    weight_decay=self.hparams.weight_decay,
                )
        else:
            optimizer = torch.optim.AdamW(
                self.head.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

        # ReduceLROnPlateau: adaptive LR decay on val/f1 plateau
        # node1-1-1 used ReduceLROnPlateau and achieved F1=0.474 (vs node1-1's 0.472 with cosine)
        # patience=10: reduce LR after 10 epochs without val/f1 improvement (threshold=1e-4)
        # factor=0.5: halve the learning rate on plateau
        # min_lr=1e-7: minimum LR floor

        # Custom wrapper that reduces LR for ALL param groups uniformly.
        # Standard ReduceLROnPlateau only updates optimizer.param_groups[0]['lr'],
        # which would leave the AdamW group at its original LR when using MuonWithAuxAdam.
        class ReduceLROnPlateauAllGroups(torch.optim.lr_scheduler.ReduceLROnPlateau):
            def step(self, metrics: float, *args, **kwargs) -> None:
                # Call parent's step which updates self._last_lr = [param_groups[0]['lr']]
                super().step(metrics, *args, **kwargs)
                # Propagate the reduced LR to ALL param groups so AdamW and Muon groups
                # receive the same proportional LR reduction (muon_lr and adamw_lr both scale).
                new_lr = self._last_lr[0]
                for group in self.optimizer.param_groups:
                    group["lr"] = max(new_lr, 1e-7)

        scheduler = ReduceLROnPlateauAllGroups(
            optimizer,
            mode="max",        # maximize val/f1
            factor=self.hparams.plateau_factor,
            patience=self.hparams.plateau_patience,
            threshold=1e-4,
            min_lr=1e-7,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/f1",    # Monitor val/f1 for plateau detection
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and buffers (excludes frozen STRING_GNN)."""
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
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable}/{total} params "
            f"({100 * trainable / total:.2f}%), plus {total_buffers} buffer values"
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
        description="Frozen STRING_GNN + 5-block MLP + Unfactorized + Gene Bias + Muon + ReduceLROnPlateau"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--plateau-patience", type=int, default=10)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--use-muon", action="store_true", default=True)
    p.add_argument("--no-muon", action="store_false", dest="use_muon")
    p.add_argument("--early-stop-patience", type=int, default=50)
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
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        use_muon=args.use_muon,
    )

    fast_dev_run = args.fast_dev_run
    debug_max_step = args.debug_max_step
    if debug_max_step is not None:
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
