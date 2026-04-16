"""Node: Frozen STRING_GNN + Frozen ESM2-35M Dual-Source Fusion + 3-block MLP + Cosine Annealing.

Architecture Overview:
  - Dual biological feature fusion:
    * Frozen STRING_GNN (256-dim PPI topology embedding) — precomputed once
    * Frozen ESM2-35M (480-dim protein sequence embedding) — precomputed once
  - ESM2 projection: Linear(480 → 256) + GELU — gentle compression (1.875:1 ratio, vs
    1280→256 = 5:1 ratio used in node1-1-1-1-1 which discarded too much information)
  - Concatenation: [STRING 256 | ESM2 256] → 512-dim fused representation
  - 3-block Residual MLP head (hidden_dim=512, expand=2, dropout=0.35)
  - Per-gene output bias: [3, 6640] (from node1-1-1 proven design)
  - Unfactorized output projection: 512 → 19920

Training Strategy:
  - Weighted cross-entropy with label smoothing=0.05 (proven better than focal loss)
  - Pure AdamW (lr=3e-4, wd=1e-3) — standard, proven across all top nodes
  - CosineAnnealingLR with T_max=150, eta_min=1e-7 (single cycle — no restart!)
    * T_max=150 ensures single full decay cycle without restart
    * Distinct from node1-1-1-1-1's T_max=100 (caused restart at epoch 100, hurting F1)
  - Early stopping patience=35

Critical Improvement over Parent (node3-1-1-1-1-1-1, Test F1=0.422):
  1. ESM2-35M protein sequence features: Adds complementary biological information
     orthogonal to STRING_GNN's PPI topology. The 480→256 projection is a gentle
     1.875:1 compression, retaining far more protein semantic information than the
     1280→256 (5:1) projection in node1-1-1-1-1 that achieved only F1=0.462.
  2. 3 blocks instead of 5: Parent's 5-block + ReduceLROnPlateau trained 270 epochs
     in overfitting regime. Node1-1-1's 3-block achieved 0.474 (best in tree).
  3. Cosine annealing instead of ReduceLROnPlateau: Parent's ReduceLROnPlateau with
     patience=20 was too passive (4 reductions, still overfitting to epoch 274).
     Cosine T_max=150 terminates before severe overfitting saturates.
  4. Richer feature space: 512-dim concatenated representation vs 256-dim string-only,
     enabling the 3-block MLP to exploit both PPI topology and protein sequence context.

References:
  - node1-1-1 (F1=0.474): 3-block + STRING_GNN + ReduceLROnPlateau + per-gene bias [best node]
  - node1-1 (F1=0.472): 5-block + STRING_GNN + cosine annealing
  - node1-1-1-1-1 (F1=0.462): dual ESM2-650M + STRING_GNN + cosine T_max=100 (too short)
  - node1-2 (F1=0.455): 3-block + STRING_GNN + focal loss (confirmed: weighted CE >> focal)
  - node3-1-1-1-1-1-1 (parent, F1=0.422): 5-block + ReduceLROnPlateau (overfitting)
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
ESM2_EMBEDDINGS_PATH = "/home/Models/STRING_GNN/esm2_embeddings_35M.pt"
N_GENES = 6640
N_CLASSES = 3
STRING_EMB_DIM = 256    # STRING_GNN output dim
ESM2_RAW_DIM = 480      # ESM2-35M output dim (precomputed [18870, 480])
ESM2_PROJ_DIM = 256     # Projected ESM2 dim (concatenated with STRING to give 512)
FUSED_DIM = STRING_EMB_DIM + ESM2_PROJ_DIM  # 512


# ---------------------------------------------------------------------------
# ESM2 Projection: 480 → 256 (gentle 1.875:1 compression)
# ---------------------------------------------------------------------------
class ESM2Projector(nn.Module):
    """Projects 480-dim ESM2-35M embeddings down to 256-dim.

    Uses a simple Linear+GELU with LayerNorm for stable training.
    The 1.875:1 compression ratio retains far more protein semantic
    information than the 5:1 ratio (1280→256) tried in node1-1-1-1-1.
    """

    def __init__(self, in_dim: int = ESM2_RAW_DIM, out_dim: int = ESM2_PROJ_DIM) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


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
# Prediction Head (Dual-source fusion + 3-block MLP + per-gene bias)
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """Dual-source MLP: [STRING 256 | ESM2-proj 256] → 512 → [3-block MLP] → [B, 3, N_GENES].

    Architecture:
      512 (concat) → LayerNorm → Linear(512) → GELU → Dropout
          → 3 × ResidualBlock(512, expand=2)
          → Linear(512 → N_GENES×3) + per_gene_bias [3, N_GENES]
          → [B, 3, N_GENES]

    The ESM2 projection is separate (not part of this module) and is applied
    before concatenation in the LightningModule forward pass.
    """

    def __init__(
        self,
        in_dim: int = FUSED_DIM,    # 512 (fused)
        hidden_dim: int = 512,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.n_classes = N_CLASSES

        # Input projection: FUSED_DIM (512) → hidden_dim (512)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual MLP trunk: n_blocks blocks (3 proven in node1-1-1)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=2, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Unfactorized output projection: hidden_dim → n_genes * N_CLASSES (~10.2M)
        self.out_proj = nn.Linear(hidden_dim, n_genes * N_CLASSES)

        # Per-gene output bias: [3, N_GENES] learnable bias (inspired by node1-1-1)
        # Initialized to zero for stable start from unfactorized head's baseline.
        self.gene_bias = nn.Parameter(torch.zeros(N_CLASSES, n_genes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)                               # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)
        out = self.out_proj(x)                               # [B, N_GENES * 3]
        out = out.view(-1, self.n_classes, self.n_genes)     # [B, 3, N_GENES]
        out = out + self.gene_bias.unsqueeze(0)              # [B, 3, N_GENES]
        return out


# ---------------------------------------------------------------------------
# Dataset — precomputed frozen embeddings (STRING + ESM2)
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Maps each perturbed gene to precomputed STRING_GNN + ESM2-35M embeddings.

    Both embedding tensors are precomputed once in setup() and stored as
    CPU float32 tensors. Genes not in the STRING graph (gnn_idx=-1) get
    zero vectors for both STRING and ESM2 features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        string_emb: torch.Tensor,    # [N_GNN_NODES, 256]
        esm2_emb: torch.Tensor,      # [N_GNN_NODES, 480]
        ensg_to_idx: Dict[str, int],
        string_dim: int,
        esm2_dim: int,
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        # Precompute feature tensors for each sample
        string_feats = []
        esm2_feats = []
        for pid in self.pert_ids:
            gnn_idx = ensg_to_idx.get(pid, -1)
            if gnn_idx >= 0:
                string_feats.append(string_emb[gnn_idx])
                esm2_feats.append(esm2_emb[gnn_idx])
            else:
                string_feats.append(torch.zeros(string_dim))
                esm2_feats.append(torch.zeros(esm2_dim))
        self.string_feats: torch.Tensor = torch.stack(string_feats, dim=0)  # [N, 256]
        self.esm2_feats: torch.Tensor = torch.stack(esm2_feats, dim=0)      # [N, 480]

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
            "string_feat": self.string_feats[idx],   # [256] float32
            "esm2_feat": self.esm2_feats[idx],       # [480] float32
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class PerturbDataModule(LightningDataModule):
    """DataModule providing precomputed frozen STRING_GNN + ESM2-35M embeddings."""

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
        self._train_batch_size = micro_batch_size  # Used for train/val dataloaders
        self.num_workers = num_workers

        self.ensg_to_idx: Optional[Dict[str, int]] = None
        self.string_emb: Optional[torch.Tensor] = None
        self.esm2_emb: Optional[torch.Tensor] = None
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        if self.ensg_to_idx is None:
            self._precompute_embeddings()

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        kwargs = dict(
            string_emb=self.string_emb,
            esm2_emb=self.esm2_emb,
            ensg_to_idx=self.ensg_to_idx,
            string_dim=STRING_EMB_DIM,
            esm2_dim=ESM2_RAW_DIM,
        )
        self.train_ds = PerturbDataset(train_df, **kwargs)
        self.val_ds = PerturbDataset(val_df, **kwargs)
        self.test_ds = PerturbDataset(test_df, **kwargs)

    def _precompute_embeddings(self) -> None:
        """Precompute frozen STRING_GNN embeddings and load ESM2-35M precomputed embeddings."""
        import torch.distributed as dist

        model_dir = Path(STRING_GNN_DIR)
        is_dist = dist.is_available() and dist.is_initialized()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if local_rank == 0:
            # ---- STRING_GNN: run frozen forward pass ----
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
            string_emb = outputs.last_hidden_state.float().cpu()  # [18870, 256]
            print(f"Precomputed STRING_GNN embeddings: shape={string_emb.shape}", flush=True)

            del _gnn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ---- ESM2-35M: load precomputed embeddings ----
            esm2_emb = torch.load(ESM2_EMBEDDINGS_PATH, map_location="cpu").float()  # [18870, 480]
            print(f"Loaded ESM2-35M embeddings: shape={esm2_emb.shape}", flush=True)
        else:
            string_emb = None
            esm2_emb = None

        # Synchronize across ranks
        if is_dist:
            dist.barrier()
            if string_emb is None:
                # Non-rank-0 ranks: recompute STRING_GNN from the same frozen weights
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
                string_emb = outputs.last_hidden_state.float().cpu()
                del _gnn
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                esm2_emb = torch.load(ESM2_EMBEDDINGS_PATH, map_location="cpu").float()
            dist.barrier()

        self.string_emb = string_emb    # [18870, 256] CPU float32
        self.esm2_emb = esm2_emb        # [18870, 480] CPU float32

        # Load node names and build ENSG → idx mapping
        node_names: List[str] = json.loads(
            (model_dir / "node_names.json").read_text()
        )
        self.ensg_to_idx = {name: i for i, name in enumerate(node_names)}
        print(
            f"Embeddings ready: {len(node_names)} nodes | "
            f"STRING_GNN: {string_emb.shape} | ESM2-35M: {esm2_emb.shape}",
            flush=True,
        )

    def _make_loader(self, ds: PerturbDataset, shuffle: bool, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self._train_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        # Use batch_size=1 and drop_last=True for test to ensure all 141
        # samples are processed correctly in DDP without padding/dropping issues.
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    """Dual-source (STRING_GNN + ESM2-35M) frozen embeddings + 3-block MLP + Per-Gene Bias.

    Key innovations over parent (node3-1-1-1-1-1-1, F1=0.422):
    1. ESM2-35M protein sequence features concatenated with STRING_GNN PPI topology features.
       Provides complementary biological signal: PPI topology + protein sequence context.
       ESM2-35M 480-dim → project 256-dim (gentle 1.875:1 ratio vs node1-1-1-1-1's 5:1 ratio).
    2. 3 residual blocks (not 5): Aligns with node1-1-1 (F1=0.474), avoids overfitting.
    3. Cosine annealing T_max=150 (not ReduceLROnPlateau): Terminates before severe overfitting.
       T_max=150 ensures single full decay cycle without restart (not T_max=100 as node1-1-1-1-1).
    4. Per-gene bias [3, 6640]: Proven in node1-1-1 for gene-level output calibration.
    5. Weighted CE + label smoothing: Confirmed better than focal loss (node1-2: F1=0.455 with
       focal loss vs node1-1-1: F1=0.474 with weighted CE, same architecture otherwise).
    """

    def __init__(
        self,
        string_dim: int = STRING_EMB_DIM,
        esm2_raw_dim: int = ESM2_RAW_DIM,
        esm2_proj_dim: int = ESM2_PROJ_DIM,
        hidden_dim: int = 512,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        dropout: float = 0.35,
        label_smoothing: float = 0.05,
        t_max: int = 150,
        eta_min: float = 1e-7,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.esm2_proj: Optional[ESM2Projector] = None
        self.head: Optional[PerturbHead] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        fused_dim = self.hparams.string_dim + self.hparams.esm2_proj_dim  # 512

        # ESM2 projection: 480 → 256 (gentle compression)
        self.esm2_proj = ESM2Projector(
            in_dim=self.hparams.esm2_raw_dim,
            out_dim=self.hparams.esm2_proj_dim,
        )

        # 3-block prediction head
        self.head = PerturbHead(
            in_dim=fused_dim,
            hidden_dim=self.hparams.hidden_dim,
            n_genes=self.hparams.n_genes,
            n_blocks=self.hparams.n_blocks,
            dropout=self.hparams.dropout,
        )

        # Cast trainable parameters to float32 for stable optimization
        for p in self.esm2_proj.parameters():
            if p.requires_grad:
                p.data = p.data.float()
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
        esm2_proj_params = sum(p.numel() for p in self.esm2_proj.parameters())
        self.print(
            f"Dual-Source (STRING 256 + ESM2-35M→256) + 3-block MLP + Gene Bias | "
            f"trainable={trainable:,}/{total:,} | "
            f"esm2_proj={esm2_proj_params:,} | "
            f"hidden={self.hparams.hidden_dim}, blocks={self.hparams.n_blocks}, "
            f"dropout={self.hparams.dropout} | "
            f"AdamW(lr={self.hparams.lr}, wd={self.hparams.weight_decay}) | "
            f"CosineAnnealingLR(T_max={self.hparams.t_max}, eta_min={self.hparams.eta_min})"
        )

    def _forward(self, string_feat: torch.Tensor, esm2_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass: concatenate STRING + projected ESM2 → 3-block MLP → logits."""
        # Project ESM2: [B, 480] → [B, 256]
        esm2_projected = self.esm2_proj(esm2_feat.float())   # [B, 256]
        # Concatenate: [B, 256] + [B, 256] → [B, 512]
        fused = torch.cat([string_feat.float(), esm2_projected], dim=-1)  # [B, 512]
        # Predict: [B, 512] → [B, 3, N_GENES]
        return self.head(fused)

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
        logits = self._forward(
            batch["string_feat"].to(self.device),
            batch["esm2_feat"].to(self.device),
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward(
            batch["string_feat"].to(self.device),
            batch["esm2_feat"].to(self.device),
        )
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
        preds_local = torch.cat(self._val_preds, dim=0)    # [local_N, 3, N_GENES]
        labels_local = torch.cat(self._val_labels, dim=0)  # [local_N, N_GENES]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather predictions and labels from ALL ranks so F1 is computed on full validation set
        all_preds = self.all_gather(preds_local)
        all_labels = self.all_gather(labels_local)

        # Flatten across ranks: DDP partitions are non-overlapping
        full_preds = all_preds.view(-1, N_CLASSES, N_GENES).float().cpu().numpy()
        full_labels = all_labels.view(-1, N_GENES).cpu().numpy()

        f1 = _compute_per_gene_f1(full_preds, full_labels)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward(
            batch["string_feat"].to(self.device),
            batch["esm2_feat"].to(self.device),
        )
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

            # Deduplicate by pert_id. When micro_batch_size=1 and drop_last=True,
            # all_preds_np has the same length as all_pert_ids (no padding). We track
            # the prediction index (i) for each unique pert_id to ensure correct pairing.
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            used_pred_indices: List[int] = []  # Track which prediction indices are used
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(all_preds_np[i])
                    used_pred_indices.append(i)

            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=np.stack(dedup_preds, axis=0),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine annealing (single cycle).

        Key differences from parent (node3-1-1-1-1-1-1):
        - CosineAnnealingLR T_max=150 instead of ReduceLROnPlateau:
          The parent's ReduceLROnPlateau with patience=20 was too passive —
          4 LR reductions still allowed 270 epochs of overfitting.
          Cosine annealing terminates training at T_max=150 naturally,
          well before the overfitting saturation regime.
        - T_max=150 (not 100): node1-1-1-1-1 used T_max=100 which caused a
          cosine restart at epoch 100 that disrupted the converged model (val/F1
          dropped from 0.466 to oscillating). T_max=150 ensures the LR only
          reaches eta_min once and early stopping ends training gracefully.
        """
        # Optimize only ESM2 projector + head (STRING_GNN is precomputed/frozen)
        params = list(self.esm2_proj.parameters()) + list(self.head.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.t_max,
            eta_min=self.hparams.eta_min,
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
        """Save only trainable parameters and buffers."""
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
        description="Dual-Source STRING_GNN + ESM2-35M Fusion + 3-block MLP + Cosine Annealing"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--t-max", type=int, default=150)
    p.add_argument("--eta-min", type=float, default=1e-7)
    p.add_argument("--early-stop-patience", type=int, default=35)
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

    # Estimate total training batches per epoch: ceil(1273 / (micro_batch_size * n_gpus))
    # Cap limit_train at this to avoid exceeding available batches in debug mode
    total_train_batches = (1273 + args.micro_batch_size * n_gpus - 1) // (args.micro_batch_size * n_gpus)

    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    model = PerturbModule(
        string_dim=STRING_EMB_DIM,
        esm2_raw_dim=ESM2_RAW_DIM,
        esm2_proj_dim=ESM2_PROJ_DIM,
        hidden_dim=args.hidden_dim,
        n_genes=N_GENES,
        n_blocks=args.n_blocks,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        t_max=args.t_max,
        eta_min=args.eta_min,
    )

    fast_dev_run = args.fast_dev_run
    debug_max_step = args.debug_max_step
    if debug_max_step is not None:
        # Cap at total available batches per epoch to avoid overshooting
        limit_train = min(debug_max_step * accumulate, total_train_batches)
        # For val/test: use full dataset for comprehensive evaluation during debug
        limit_val = 1.0
        limit_test = 1.0
        max_steps = debug_max_step
        val_check_interval = 1.0
        num_sanity_val_steps = 2  # Always run sanity checks for early error detection
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
            find_unused_parameters=True, timeout=timedelta(seconds=120)
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
        score_path = output_dir / "test_score.txt"
        test_f1 = test_results[0].get("test/f1", float("nan"))
        score_path.write_text(f"{test_f1}\n")
        print(f"Test F1 = {test_f1:.6f} → {score_path}", flush=True)


if __name__ == "__main__":
    main()
