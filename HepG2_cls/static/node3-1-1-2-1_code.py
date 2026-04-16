"""Node node1-3: STRING_GNN + 3-block MLP + Per-gene Bias + ReduceLROnPlateau + SWA.

Architecture Overview:
  - Precomputed frozen STRING_GNN embeddings (256-dim): encodes PPI graph topology,
    the proven signal from node1-1 (F1=0.472) and node1-1-1 (F1=0.474)
  - 3-block Residual MLP head: 256 -> 512 -> [3 res blocks] -> 6640x3 -> [B, 3, 6640]
  - Per-gene additive bias (6640x3 = 19,920 parameters): proven +0.002 in node1-1-1
  - Standard weighted cross-entropy + label smoothing (NO focal loss, NO warmup)
  - ReduceLROnPlateau (patience=15, factor=0.5, min_lr=1e-6): proven in node1-1-1 (F1=0.474)
  - Stochastic Weight Averaging (SWA) tail: stabilizes checkpoint quality on noisy 141-sample val set

Key Improvements over Parent (node1-2, STRING_GNN + 5-block + cosine T_max=200, F1=0.385):
  1. REDUCE from 5 to 3 residual blocks (primary fix: 5 blocks severely overfit on 1,273 samples)
  2. SWITCH to ReduceLROnPlateau (patience=15, factor=0.5): proven in node1-1-1 vs cosine T_max=200
  3. ADD Stochastic Weight Averaging (SWA, swa_epoch_start=0.8): novel addition to stabilize
     checkpoint selection on the noisy 141-sample validation set (oscillates +-0.03 per epoch)
  4. INCREASE early stopping patience to 60 (141-sample val oscillation requires >40 patience)
  5. REDUCE max epochs from 200 to 150: RLROP converges faster than cosine annealing
  6. Keep per-gene bias (proven +0.002 F1 in node1-1-1 with 3-block architecture)
  7. Keep unfactorized output head Linear(512->19920): factorized bottleneck proven to fail

Distinction from Parent (node1-2, F1=0.385):
  - 3 blocks (not 5): node1-1-1 (3 blocks, F1=0.474) vs node1-2 (5 blocks, F1=0.385) = +0.089 delta
  - RLROP (not cosine T_max=200): node1-1-1 (RLROP, F1=0.474) vs node1-1-2-1 (cosine, F1=0.429)
  - SWA tail: novel; addresses the val-test gap observed in all STRING-based nodes
  - Early stopping patience=60 (not 40): accommodates +-0.03 val/f1 oscillation on 141 samples

Expected Performance:
  - Target: match or exceed node1-1-1 (F1=0.474) by replicating its proven recipe
  - Expected range: 0.46-0.48 by combining node1-1-1 hyperparameters with SWA stabilization
  - SWA upside: if SWA reduces val-test gap by 0.02-0.03, could push toward 0.49+
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
    StochasticWeightAveraging,
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
# Residual Block
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

    The per-gene bias (19,920 parameters) provides additive gene-specific calibration
    learned from training data. This was used in node1-1-1 (F1=0.474) and contributed
    a small but consistent +0.002 improvement over node1-1 (F1=0.472).

    3 blocks (not 5): node1-1-1 (3 blocks, RLROP) achieves F1=0.474; node1-2 (5 blocks,
    cosine T_max=200) achieves F1=0.385. The 2 extra blocks cost 0.089 F1 on 1,273 samples.
    """

    def __init__(
        self,
        in_dim: int = STRING_EMB_DIM,
        hidden_dim: int = 512,
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
        # Unfactorized output head (512 -> 19920): proven to outperform factorized bottleneck
        # Factorized heads (bottleneck 128-384) have conclusively failed across 5+ nodes
        # The unfactorized head was key to node1-1 (F1=0.472) and node1-1-1 (F1=0.474)
        self.out_proj = nn.Linear(hidden_dim, n_genes * N_CLASSES)

        # Per-gene additive bias: 6640 x 3 = 19,920 learned scalars
        # Proven in node1-1-1 which showed +0.002 improvement over node1-1
        # These biases learn the "baseline" class distribution for each gene
        # regardless of which gene is perturbed
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
    """Maps each perturbed gene to its precomputed STRING_GNN feature vector."""

    def __init__(
        self,
        df: pd.DataFrame,
        gene_features: torch.Tensor,
        ensg_to_idx: Dict[str, int],
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_features = gene_features       # [N_NODES, STRING_EMB_DIM] CPU float32
        self.ensg_to_idx = ensg_to_idx

        if "label" in df.columns:
            labels = np.array(
                [json.loads(x) for x in df["label"].tolist()], dtype=np.int64
            )
            self.labels: Optional[torch.Tensor] = torch.tensor(
                labels + 1, dtype=torch.long
            )  # {-1, 0, 1} -> {0, 1, 2}
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pert_id = self.pert_ids[idx]
        gnn_idx = self.ensg_to_idx.get(pert_id, -1)

        if gnn_idx >= 0:
            feat = self.gene_features[gnn_idx]   # [STRING_EMB_DIM]
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

        STRING-only approach: ESM2 fusion consistently degrades performance across all fusion
        nodes in the tree (node1-2: 0.455, node1-3: 0.463, node1-1-1-1-1: 0.462) vs
        STRING-only baseline (node1-1-1: 0.474). The 256-dim PPI topology is the proven signal.

        In DDP (multi-GPU), each rank independently loads and computes its own embeddings.
        No inter-rank communication is needed since the gene features are identical across
        all ranks (precomputed from the frozen STRING graph).
        """
        model_dir = Path(STRING_GNN_DIR)

        # Build node index map
        node_names: List[str] = json.loads(
            (model_dir / "node_names.json").read_text()
        )
        self.ensg_to_idx = {name: i for i, name in enumerate(node_names)}

        # Use GPU for STRING_GNN forward pass if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading STRING_GNN for precomputing topology embeddings...", flush=True)
        gnn = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device)
        gnn.eval()

        graph = torch.load(model_dir / "graph_data.pt", map_location=device)
        edge_index = graph["edge_index"]            # [2, E]
        edge_weight = graph.get("edge_weight", None)  # [E] or None

        with torch.no_grad():
            out = gnn(
                edge_index=edge_index,
                edge_weight=edge_weight,
                output_hidden_states=False,
            )
            self.gene_features = out.last_hidden_state.float().cpu()  # [N, 256]

        del gnn, graph, out
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Barrier after computation to ensure all ranks complete before proceeding
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        print(
            f"Precomputed gene features: {self.gene_features.shape} "
            f"(STRING_GNN topology only)",
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
    def __init__(
        self,
        in_dim: int = STRING_EMB_DIM,
        hidden_dim: int = 512,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        lr: float = 3e-4,
        weight_decay: float = 0.001,
        dropout: float = 0.35,
        label_smoothing: float = 0.05,
        rlrop_patience: int = 15,
        rlrop_factor: float = 0.5,
        rlrop_min_lr: float = 1e-6,
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
        self.head = PerturbHead(
            in_dim=self.hparams.in_dim,
            hidden_dim=self.hparams.hidden_dim,
            n_genes=self.hparams.n_genes,
            n_blocks=self.hparams.n_blocks,
            dropout=self.hparams.dropout,
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
            f"STRING_GNN + {self.hparams.n_blocks}-block MLP + per-gene bias | "
            f"trainable={trainable:,}/{total:,} | "
            f"in_dim={self.hparams.in_dim}, hidden={self.hparams.hidden_dim}, "
            f"blocks={self.hparams.n_blocks}, dropout={self.hparams.dropout}"
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard weighted cross-entropy with label smoothing.

        No focal loss (caused catastrophic collapse in node3-1 and sibling-like nodes).
        No warmup (caused 28% val/f1 drop in grandparent node3-1-1).
        Weighted CE + label smoothing is the proven recipe from node1-1(F1=0.472)
        and node1-1-1 (F1=0.474).
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
        all_preds = self.all_gather(preds_local)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        # Reshape to [total_samples, 3, N_GENES]
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        # Gather labels
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
        """AdamW optimizer with ReduceLROnPlateau scheduler.

        ReduceLROnPlateau (patience=15, factor=0.5) is the proven scheduler from node1-1-1
        which achieved F1=0.474 — the current tree ceiling. It reacts to validation plateaus
        by halving the LR, which is more effective than a fixed cosine schedule for small
        datasets (1,273 samples) where the validation F1 is noisy.

        node1-1-2-1 (cosine T_max=200, F1=0.429) and node1-2 (cosine T_max=200, F1=0.385)
        both confirm that the fixed cosine schedule is inferior to RLROP on this task.

        Patience=15 (vs node1-1-1's patience=20): Slightly more aggressive LR reduction
        to compensate for the 141-sample val oscillation while keeping 3-block capacity stable.
        This is close to node3-1-2-1 (patience=15, F1=0.416 with factorized head) but here
        we use the proven unfactorized head, so we expect much better fit capacity.
        """
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # ReduceLROnPlateau: proven in node1-1-1 (F1=0.474)
        # Monitors val/f1 and halves LR when plateaued for patience=15 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.rlrop_factor,
            patience=self.hparams.rlrop_patience,
            threshold=1e-4,
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
        # Guard: if head is not yet initialized (e.g., SWA callback deepcopies the model
        # without re-running setup()), return empty dict to avoid ZeroDivisionError.
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
        description="STRING_GNN + 3-block MLP + per-gene bias + ReduceLROnPlateau + SWA"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--rlrop-patience", type=int, default=15)
    p.add_argument("--rlrop-factor", type=float, default=0.5)
    p.add_argument("--rlrop-min-lr", type=float, default=1e-6)
    p.add_argument("--early-stop-patience", type=int, default=60)
    p.add_argument("--swa-lrs", type=float, default=1e-5)
    p.add_argument("--swa-epoch-start", type=float, default=0.8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--disable-swa", action="store_true",
                   help="Disable Stochastic Weight Averaging (for ablation)")
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
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        rlrop_patience=args.rlrop_patience,
        rlrop_factor=args.rlrop_factor,
        rlrop_min_lr=args.rlrop_min_lr,
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

    # Stochastic Weight Averaging: stabilizes predictions on the noisy 141-sample val set
    # SWA averages model weights over the last (1-swa_epoch_start) fraction of training,
    # creating a flatter loss minimum that generalizes better to the test set.
    # This addresses the persistent val-test gap observed across all STRING-based nodes
    # (e.g., node1-2: best_val=0.432, test=0.385 = 0.047 gap; node1-1-1: similar gap).
    # SWA is disabled in debug/fast_dev_run modes.
    use_swa = not args.disable_swa and not fast_dev_run and args.debug_max_step is None
    swa_cb = StochasticWeightAveraging(
        swa_lrs=args.swa_lrs,
        swa_epoch_start=args.swa_epoch_start,
        annealing_epochs=5,
        annealing_strategy="cos",
    ) if use_swa else None

    callbacks = [checkpoint_cb, early_stop_cb, lr_monitor, progress_bar]
    if swa_cb is not None:
        callbacks.append(swa_cb)

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
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        # No gradient clipping: node1-1 and node1-1-1 succeeded without it;
        # no evidence of benefit on this task
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
