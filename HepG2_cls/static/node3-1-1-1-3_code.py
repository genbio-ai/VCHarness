"""Node: node3-1-1-1-3 — STRING_GNN (Frozen) + 3-block PreLN MLP + Muon + RLROP + head_dropout

Architecture Overview:
  - Precomputed frozen STRING_GNN embeddings (256-dim): proven PPI graph signal
  - 3-block PreLN Residual MLP (hidden_dim=384): proven optimal in node1-3-2 (F1=0.4756)
  - Muon optimizer for hidden weight matrices + AdamW for other params
  - ReduceLROnPlateau scheduler: proven to reach F1=0.474 in node1-1-1 and F1=0.473+ in recovery nodes
  - Unfactorized Linear(384→19920) output head: proven superior to bottleneck variants
  - Head dropout (p=0.15): proven innovation in node1-3-2-2-1 (STRING-only tree best 0.4777)
  - Per-gene bias: innovation from node1-1-1
  - Correct multi-GPU val/f1 via all_gather + dedup (critical for reliable checkpointing)
  - Weighted cross-entropy + label smoothing (NO focal loss — interferes with Muon)
  - Gradient clipping max_norm=1.0 (proven with Muon in node1-3-2)

Key Improvements vs Parent (node3-1-1-1, F1=0.390):
  1. Fix T_max=50 cosine restart → use ReduceLROnPlateau (adaptive, proven in node1-1-1 F1=0.474)
  2. Remove factorized bottleneck (dim=256) → use unfactorized Linear(384→19920)
  3. Reduce from 5 blocks→3 blocks, hidden_dim 512→384: optimal capacity for 1,273 samples
  4. Add Muon optimizer: +0.002 over AdamW in node1-3-2 vs AdamW-only baselines
  5. Add head dropout (p=0.15): first STRING-only F1 improvement beyond 0.474 ceiling
  6. Fix multi-GPU val/f1 gathering: eliminates noisy per-GPU F1, improves checkpointing
  7. Keep frozen STRING_GNN: universally proven superior to GNN fine-tuning in node3 lineage
  8. Per-gene bias: innovation from node1-1-1 (F1=0.474)

Differentiation from Siblings:
  - sibling_node_1 (node3-1-1-1-1, F1=0.392): partial GNN fine-tuning — counterproductive
  - sibling_node_2 (node3-1-1-1-2, F1=0.383): DropPath + T_max=200 — val/loss stuck
  - THIS NODE: Muon + RLROP + 3 blocks + hidden=384 + head_dropout (proven recipe)
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


# ---------------------------------------------------------------------------
# PreLN Residual Block
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """PreLN residual block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout + residual."""

    def __init__(self, dim: int, expand: int = 2, dropout: float = 0.30) -> None:
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
# Prediction Head
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """3-block PreLN residual MLP with head dropout: [B, FEATURE_DIM] → [B, 3, N_GENES].

    Based on node1-3-2 proven recipe (F1=0.4756) + node1-3-2-2-1 enhancement (F1=0.4777):
    - hidden_dim=384 (optimal for 1,273 samples vs 512 which overfits — node1-3-2 confirmed)
    - 3 residual blocks (proven: node1-1-1 F1=0.474, node1-3-2 F1=0.4756; vs 5 blocks = worse)
    - Unfactorized Linear(384→19920) output head (proven: 3 independent failures of bottleneck=256)
    - Head dropout p=0.15 before final projection (node1-3-2-2-1 proved +0.002 improvement)
    - Per-gene bias (node1-1-1 innovation: 19,920 additional learned per-gene calibration values)
    """

    def __init__(
        self,
        in_dim: int = STRING_EMB_DIM,
        hidden_dim: int = 384,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        dropout: float = 0.30,
        head_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes

        # Input projection: in_dim → hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # PreLN Residual MLP trunk: n_blocks blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=2, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Output head with targeted dropout regularization
        # Head dropout p=0.15 is placed BEFORE the final projection, targeting the dominant
        # overfitting source (the ~7.65M-param output head) without slowing trunk convergence.
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.head_dropout = nn.Dropout(head_dropout)
        self.out_proj = nn.Linear(hidden_dim, n_genes * N_CLASSES)

        # Per-gene bias (from node1-1-1): allows each gene to have its own baseline
        # activation, compensating for gene-specific class imbalance patterns.
        self.gene_bias = nn.Parameter(torch.zeros(n_genes * N_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)               # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        x = self.head_dropout(x)             # Head dropout before final projection
        out = self.out_proj(x) + self.gene_bias  # [B, N_GENES * 3] + per-gene bias
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

        Frozen STRING_GNN (precomputed once) proven universally superior to GNN fine-tuning:
        - node1-1 (5-block, frozen) F1=0.472
        - node1-1-1 (3-block, frozen) F1=0.474
        - node3-1-1-1-1 (5-block, partial GNN fine-tune) F1=0.392 — regression of 0.080
        All GNN fine-tuning attempts in node3 lineage failed to reach F1=0.392.
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

        print("Loading STRING_GNN for precomputing frozen topology embeddings...", flush=True)
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
            f"(STRING_GNN topology only, frozen)",
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
        in_dim: int = STRING_EMB_DIM,
        hidden_dim: int = 384,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        lr: float = 3e-4,
        muon_lr: float = 0.01,
        weight_decay: float = 8e-4,
        dropout: float = 0.30,
        head_dropout: float = 0.15,
        label_smoothing: float = 0.05,
        rlrop_patience: int = 8,
        rlrop_factor: float = 0.5,
        rlrop_min_lr: float = 1e-6,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.head: Optional[PerturbHead] = None

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
            head_dropout=self.hparams.head_dropout,
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
            f"STRING_GNN (Frozen) + {self.hparams.n_blocks}-block Muon+AdamW+RLROP | "
            f"trainable={trainable:,}/{total:,} | "
            f"in={self.hparams.in_dim}, hidden={self.hparams.hidden_dim}, "
            f"dropout={self.hparams.dropout}, head_dropout={self.hparams.head_dropout}"
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy with label smoothing.

        NO focal loss: node1-1-3-1 proved Muon + focal loss = catastrophic collapse (F1=0.191).
        node1-3-2 proved Muon + WCE = F1=0.4756. WCE is the correct combination.
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
        if not self._val_preds:
            return

        import torch.distributed as dist

        preds_local = torch.cat(self._val_preds, dim=0)    # [local_N, 3, N_GENES]
        labels_local = torch.cat(self._val_labels, dim=0)  # [local_N, N_GENES]
        local_pert_ids = list(self._val_pert_ids)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_pert_ids.clear()

        # Gather from ALL ranks for CORRECT global val/f1 computation.
        # With 8 GPUs and 141 val samples, each GPU has only ~18 samples.
        # Per-GPU F1 on 18 samples is extremely noisy: many genes will have all-neutral
        # labels in each subset, yielding inconsistent F1 values that mislead checkpointing.
        # Gathering all 141 samples before computing F1 gives the true validation metric.
        # This fix was identified in sibling node3-1-1-1-2's documentation and is essential.
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        all_preds = self.all_gather(preds_local)    # [world_size, local_N, 3, N_GENES]
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
        all_labels = self.all_gather(labels_local)  # [world_size, local_N, N_GENES]
        all_labels = all_labels.view(-1, N_GENES)

        # Gather pert_ids for deduplication (DDP pads last batch across ranks)
        gathered_pert_ids = [local_pert_ids]
        if world_size > 1:
            obj_pert = [None] * world_size
            dist.all_gather_object(obj_pert, local_pert_ids)
            gathered_pert_ids = obj_pert

        all_pert_ids_flat = [p for rank_list in gathered_pert_ids for p in rank_list]
        all_preds_np = all_preds.float().cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()

        # Deduplicate by pert_id (handles DDP padding of last batch)
        seen: set = set()
        dedup_preds, dedup_labels = [], []
        for i, pid in enumerate(all_pert_ids_flat):
            if pid not in seen:
                seen.add(pid)
                dedup_preds.append(all_preds_np[i])
                dedup_labels.append(all_labels_np[i])

        if dedup_preds:
            f1 = _compute_per_gene_f1(
                np.stack(dedup_preds, axis=0),
                np.stack(dedup_labels, axis=0),
            )
            # Log with sync_dist=True so all ranks see the same value for RLROP and EarlyStopping
            self.log("val/f1", f1, prog_bar=True, sync_dist=True)

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

        preds_local = torch.cat(self._test_preds, dim=0)   # [local_N, 3, N_GENES]
        labels_local = (
            torch.cat(self._test_labels, dim=0) if self._test_labels else None
        )
        self._test_preds.clear()
        self._test_labels.clear()

        # Gather predictions from all ranks
        all_preds = self.all_gather(preds_local)  # [world_size, local_N, 3, N_GENES]
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)  # [total_N, 3, N_GENES]

        # Gather labels (for test F1 logging)
        if labels_local is not None:
            all_labels = self.all_gather(labels_local)  # [world_size, local_N, N_GENES]
            all_labels = all_labels.view(-1, N_GENES)   # [total_N, N_GENES]
        else:
            all_labels = None

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
            dedup_indices = []
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(all_preds_np[i])
                    dedup_indices.append(i)

            dedup_preds_np = np.stack(dedup_preds, axis=0)

            # Compute test F1 if labels available
            if all_labels is not None:
                all_labels_np = all_labels.cpu().numpy()
                dedup_labels_np = all_labels_np[np.array(dedup_indices)]
                test_f1 = _compute_per_gene_f1(dedup_preds_np, dedup_labels_np)
                self.log("test/f1", test_f1, prog_bar=True, rank_zero_only=True)

            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=dedup_preds_np,
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    def configure_optimizers(self):
        """Muon + AdamW dual optimizer with ReduceLROnPlateau.

        Recipe from node1-3-2 (F1=0.4756 — new STRING-only tree best at the time):
        - Muon (lr=0.01) for hidden fc1/fc2 weight matrices in residual blocks
        - AdamW (lr=3e-4) for all other params (norms, biases, input_proj, output head)
        - ReduceLROnPlateau(patience=8, factor=0.5, mode=max) on val/f1

        Muon is NOT applied to:
        - input_proj: first projection layer (convention)
        - out_proj, gene_bias: output/last layer
        - LayerNorm parameters
        - Biases (1D)
        Note: The skill says output heads should not use Muon, and first layers should not.
        """
        try:
            from muon import MuonWithAuxAdam
        except ImportError:
            raise ImportError(
                "Muon optimizer not found. Install with: "
                "pip install git+https://github.com/KellerJordan/Muon"
            )

        # Separate parameters: Muon for hidden 2D weight matrices in residual blocks
        muon_params = []
        adamw_params = []

        for name, param in self.head.named_parameters():
            if not param.requires_grad:
                continue
            # Apply Muon only to 2D matrices in hidden residual blocks (fc1, fc2)
            # NOT to input_proj (first layer), NOT to output head, NOT to norms/biases
            is_muon_eligible = (
                param.ndim >= 2
                and "blocks." in name
                and (".fc1.weight" in name or ".fc2.weight" in name)
            )
            if is_muon_eligible:
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
                lr=self.hparams.muon_lr,        # 0.01 (proven in node1-3-2)
                weight_decay=self.hparams.weight_decay,
                momentum=0.95,
            ),
            dict(
                params=adamw_params,
                use_muon=False,
                lr=self.hparams.lr,              # 3e-4
                betas=(0.9, 0.95),
                weight_decay=self.hparams.weight_decay,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # ReduceLROnPlateau: monitors val/f1 (max), halves LR after patience=8 epochs
        # without improvement (threshold=5e-4). Proven to rescue nodes from plateaus:
        # - node1-1-1: ReduceLROnPlateau triggered twice → F1=0.474
        # - node1-3-2: ReduceLROnPlateau patience=8 → F1=0.4756
        # - node1-3-2-2-1: 3 halvings → F1=0.4777 (STRING-only tree best)
        # - node3-1-1-1-2-1: ReduceLROnPlateau recovered from parent's 0.383 to 0.473
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.rlrop_factor,    # 0.5
            patience=self.hparams.rlrop_patience, # 8
            min_lr=self.hparams.rlrop_min_lr,     # 1e-6
            threshold=5e-4,   # Minimum improvement to count as progress
            verbose=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/f1",
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
            },
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
        description="STRING_GNN (Frozen) + 3-block PreLN MLP + Muon + RLROP + head_dropout"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=8e-4)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--head-dropout", type=float, default=0.15)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--rlrop-patience", type=int, default=8)
    p.add_argument("--rlrop-factor", type=float, default=0.5)
    p.add_argument("--rlrop-min-lr", type=float, default=1e-6)
    p.add_argument("--grad-clip-val", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=25)
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
        in_dim=STRING_EMB_DIM,
        hidden_dim=args.hidden_dim,
        n_genes=N_GENES,
        n_blocks=args.n_blocks,
        lr=args.lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        label_smoothing=args.label_smoothing,
        rlrop_patience=args.rlrop_patience,
        rlrop_factor=args.rlrop_factor,
        rlrop_min_lr=args.rlrop_min_lr,
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
        # Gradient clipping (from node1-3-2 proven recipe for Muon)
        gradient_clip_val=args.grad_clip_val,
        gradient_clip_algorithm="norm",
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
