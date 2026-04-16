"""Node improvement over parent (node3-1): STRING_GNN-only with Weighted CE + LR Warmup.

Architecture Overview:
  - Precomputed frozen STRING_GNN embeddings (256-dim): encodes PPI graph topology,
    gene-gene interaction network structure (the proven signal from node1-1)
  - 4-block Residual MLP head: 256 → 512 → [4 res blocks] → 6640×3 → [B, 3, 6640]
  - Standard weighted cross-entropy + label smoothing (NO focal loss)
  - Linear warmup (5 epochs) + cosine annealing LR

Key Improvements over Parent (node3-1, STRING_GNN+ESM2 + focal loss, F1=0.157):
  1. DROP ESM2 embeddings — proved to be noise-adding in node3-1 feedback
  2. REPLACE focal loss with standard weighted CE (focal loss caused 66% val/f1 crash)
  3. ADD 5-epoch linear warmup to prevent early training instability
  4. REDUCE learning rate to 2e-4 (down from 5e-4 which was too aggressive)
  5. INCREASE dropout to 0.40 for stronger regularization on 1273 training samples
  6. 4 residual blocks (between parent's 3 and node1-1's 5) — explores middle ground
  7. Gradient clipping (max_norm=1.0) for additional stability
  8. Weight decay 0.015 for moderate L2 regularization

Distinction from node1-1 (STRING_GNN-only, 5-block, F1=0.472):
  - 4 blocks vs. 5 (reduced capacity to prevent overfitting)
  - LR=2e-4 with 5-epoch warmup vs. LR=3e-4 (smoother training start)
  - Dropout=0.40 vs. unknown (likely 0.35), more regularization
  - Gradient clipping and LR warmup absent in node1-1
  - T_max=80 for faster cosine annealing vs. node1-1's setup
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
STRING_EMB_DIM = 256   # STRING_GNN output dim (only signal used)
FEATURE_DIM = STRING_EMB_DIM  # 256 (STRING_GNN only, ESM2 dropped)


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, dim: int, expand: int = 2, dropout: float = 0.40) -> None:
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
    """4-block residual MLP: [B, FEATURE_DIM] → [B, 3, N_GENES]."""

    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = 512,
        n_genes: int = N_GENES,
        n_blocks: int = 4,
        dropout: float = 0.40,
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
        self.out_proj = nn.Linear(hidden_dim, n_genes * N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)               # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)
        out = self.out_proj(x)               # [B, N_GENES * 3]
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

        ESM2 embeddings are intentionally excluded — node3-1 feedback showed that
        ESM2 (480-dim) diluted the STRING_GNN signal, contributed noise from
        non-coding genes, and the 736-dim combined vector destabilized training.
        STRING_GNN-only (256-dim) is the proven signal source from node1-1 (F1=0.472).
        """
        model_dir = Path(STRING_GNN_DIR)

        # Sync barrier so all DDP processes start loading together
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

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

        print(
            f"Precomputed gene features: {self.gene_features.shape} "
            f"(STRING_GNN topology only, ESM2 excluded)",
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
        in_dim: int = FEATURE_DIM,
        hidden_dim: int = 512,
        n_genes: int = N_GENES,
        n_blocks: int = 4,
        lr: float = 2e-4,
        weight_decay: float = 0.015,
        dropout: float = 0.40,
        label_smoothing: float = 0.05,
        t_max: int = 80,
        warmup_epochs: int = 5,
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
            f"STRING_GNN-only Head | "
            f"trainable={trainable:,}/{total:,} | "
            f"in_dim={self.hparams.in_dim}, hidden={self.hparams.hidden_dim}, "
            f"blocks={self.hparams.n_blocks}, dropout={self.hparams.dropout}"
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard weighted cross-entropy with label smoothing.

        Focal loss (used in node3-1) caused catastrophic training collapse:
        val/f1 crashed 66% at epoch 1 due to the (1-pt)^gamma term aggressively
        down-weighting the majority neutral class (92.8%), destroying gradient
        signal for per-gene F1 which requires correctly classifying neutral genes.

        This simple weighted CE + label smoothing is what node1-1 used to achieve
        F1=0.472 without any instability.
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
        # all_preds shape: [world_size, local_N, 3, N_GENES]
        all_preds = self.all_gather(preds_local)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        # total_samples = world_size * local_N (number of test samples, NOT including class dim)
        total_samples = all_preds.shape[0] * all_preds.shape[1]
        # Reshape to [total_samples, 3, N_GENES] — N_CLASSES=3 is the class dim, NOT batch
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        # Gather labels using the same shape logic as predictions for consistency
        if labels_local is not None:
            # labels_local shape: [local_N, N_GENES]; all_labels: [world_size, local_N, N_GENES]
            all_labels = self.all_gather(labels_local)
            all_labels = all_labels.view(-1, N_GENES)  # [total_samples, N_GENES]

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
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Linear warmup for warmup_epochs, then cosine annealing
        # Warmup prevents the epoch-1 crash pattern observed in node3-1
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.t_max,
            eta_min=1e-7,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
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
        description="STRING_GNN-only + Weighted CE + LR Warmup for HepG2 DEG Prediction"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.015)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.40)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--t-max", type=int, default=80)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--early-stop-patience", type=int, default=30)
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
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        t_max=args.t_max,
        warmup_epochs=args.warmup_epochs,
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
        gradient_clip_val=1.0,          # Clip gradients to prevent instability
        gradient_clip_algorithm="norm",  # L2 norm clipping
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
        print(f"Test results → {score_path} (f1_score={primary_metric})", flush=True)


if __name__ == "__main__":
    main()
