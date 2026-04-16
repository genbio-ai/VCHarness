#!/usr/bin/env python3
"""
Node 3-1: STRING GNN + ESM2 Conditioning – PPI Graph-Based Perturbation Encoding
=================================================================================
Core insight: A gene's position in the protein-protein interaction (PPI) network
determines how its knockout propagates through the transcriptional regulatory cascade.
This is fundamentally more relevant to DEG prediction than protein sequences (which
encode molecular function, not regulatory network position).

Architecture:
  - STRING GNN backbone (5.43M params, full fine-tuning):
      * Pretrained on human STRING v12 PPI graph (18,870 nodes, threshold=900)
      * Input: edge_index + edge_weight from the human PPI graph
      * Optional: per-node ESM2 conditioning (additive before message passing)
  - ESM2-35M pre-computed embeddings [18870, 480] projected to [18870, 256]
      * Adds protein sequence context to graph topology
  - STRING GNN forward: edge_index, edge_weight, cond_emb → [18870, 256]
  - For each perturbation sample:
      * Extract perturbed gene's embedding → [B, 256] (fallback for unknown genes)
      * Learned per-output-gene query embeddings → [6640, 256]
      * Concatenate: [B, 6640, 512]
      * MLP head: 512 → 512 → 3 → [B, 3, 6640]

Key improvements over node3 (protein sequence approach):
  1. PPI graph topology > protein sequence for regulatory cascade prediction
  2. Network neighborhood captures regulatory position directly
  3. ESM2 conditioning adds protein function context to graph structure
  4. Full fine-tuning of compact GNN (5.43M vs 16B) is feasible with 1500 samples

Training:
  - Focal loss (γ=2.0) + class weights for severe class imbalance (95% class 0)
  - AdamW: backbone lr=1e-4, other params lr=3e-4
  - CosineAnnealingLR
  - Gradient clipping (norm=1.0)
  - Early stopping: patience=25 on val_f1
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
    EarlyStopping, LearningRateMonitor, ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
N_GENES_OUT = 6_640
N_CLASSES = 3

# Class weights: inverse frequency from val stats
# down ~3.56%, unchanged ~94.82%, up ~1.63%
CLASS_WEIGHTS = torch.tensor([1.0 / 0.0356, 1.0 / 0.9482, 1.0 / 0.0163], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Multi-class focal loss to down-weight easy examples (dominant class 0)."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [N, C], targets: [N]
        # Step 1: Standard CE with class weights and label smoothing
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        # Step 2: Focal weighting
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class PerturbationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_test: bool = False):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.is_test = is_test
        self.indices: List[int] = list(range(len(df)))

        if not is_test:
            raw_labels = [json.loads(x) for x in df["label"].tolist()]
            # Map {-1,0,1} → {0,1,2}
            self.labels = np.array(raw_labels, dtype=np.int8) + 1
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "pert_ids": [b["pert_id"] for b in batch],
        "symbols": [b["symbol"] for b in batch],
    }
    if "label" in batch[0]:
        result["label"] = torch.stack([b["label"] for b in batch])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, micro_batch_size: int = 8, num_workers: int = 0):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.train_ds: Optional[PerturbationDataset] = None
        self.val_ds: Optional[PerturbationDataset] = None
        self.test_ds: Optional[PerturbationDataset] = None
        self.test_pert_ids: List[str] = []
        self.test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            train_df = pd.read_csv(self.data_dir / "train.tsv", sep="\t")
            val_df = pd.read_csv(self.data_dir / "val.tsv", sep="\t")
            self.train_ds = PerturbationDataset(train_df)
            self.val_ds = PerturbationDataset(val_df)
        if stage in ("test", None):
            test_df = pd.read_csv(self.data_dir / "test.tsv", sep="\t")
            self.test_ds = PerturbationDataset(test_df, is_test=True)
            self.test_pert_ids = test_df["pert_id"].tolist()
            self.test_symbols = test_df["symbol"].tolist()

    def _loader(self, ds: PerturbationDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class StringGNNDEGModel(nn.Module):
    """
    STRING GNN backbone (with ESM2 conditioning) + per-gene MLP head for DEG prediction.

    Forward pass (all on GPU):
      1. Compute ESM2 conditioning: cond_proj(esm2_embs) → [18870, 256]
      2. STRING GNN: (edge_index, edge_weight, cond_emb) → [18870, 256]
      3. Extract perturbed gene embedding → [B, 256]
      4. Gene query embeddings [6640, 256]
      5. Concat + MLP head → [B, 3, 6640]
    """

    GNN_DIM = 256
    ESM2_DIM = 480  # ESM2-35M embedding dimension

    def __init__(
        self,
        gene_to_node: Dict[str, int],
        head_hidden: int = 512,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.gene_to_node = gene_to_node

        # STRING GNN backbone (full fine-tuning, ~5.43M params)
        self.gnn = AutoModel.from_pretrained(
            str(STRING_GNN_DIR), trust_remote_code=True
        )
        # Cast all GNN params to float32 for stable training
        for p in self.gnn.parameters():
            p.data = p.data.float()
            p.requires_grad = True

        # Projection of ESM2 embeddings to GNN dimension (conditioning)
        # ESM2 captures protein function; projected to 256 for additive conditioning
        self.cond_proj = nn.Linear(self.ESM2_DIM, self.GNN_DIM, bias=False)
        nn.init.trunc_normal_(self.cond_proj.weight, std=0.01)

        # Fallback embedding for perturbed genes not in STRING GNN
        self.unknown_emb = nn.Parameter(torch.zeros(self.GNN_DIM))
        nn.init.normal_(self.unknown_emb, std=0.02)

        # Learned per-output-gene query embeddings: captures each output gene's context
        self.gene_query = nn.Embedding(N_GENES_OUT, self.GNN_DIM)
        nn.init.normal_(self.gene_query.weight, std=0.02)

        # MLP head: (pert_emb || gene_query) → 3 classes per gene
        in_dim = self.GNN_DIM * 2  # 512
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, N_CLASSES),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Register graph tensors and ESM2 embeddings as buffers
        # These are non-trainable and will be moved to the correct device automatically
        graph_data = torch.load(
            str(STRING_GNN_DIR / "graph_data.pt"), map_location="cpu"
        )
        self.register_buffer("edge_index", graph_data["edge_index"].long())
        edge_w = graph_data.get("edge_weight")
        if edge_w is not None:
            self.register_buffer("edge_weight", edge_w.float())
        else:
            self.register_buffer("edge_weight", torch.ones(graph_data["edge_index"].shape[1]))

        # ESM2-35M pre-computed embeddings [18870, 480] – frozen
        esm2_embs = torch.load(
            str(STRING_GNN_DIR / "esm2_embeddings_35M.pt"), map_location="cpu"
        ).float()
        self.register_buffer("esm2_embs", esm2_embs)

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """
        Args:
            pert_ids: list of B ENSG gene IDs (e.g., "ENSG00000001084")
        Returns:
            logits: [B, 3, 6640]
        """
        B = len(pert_ids)
        device = self.edge_index.device

        # 1. Compute ESM2 conditioning for all graph nodes: [18870, 256]
        #    cond_emb is the same for all samples in the batch (static w.r.t. input)
        cond_emb = self.cond_proj(self.esm2_embs)  # [18870, 256]

        # 2. STRING GNN forward with ESM2 conditioning → [18870, 256]
        gnn_out = self.gnn(
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            cond_emb=cond_emb,
        ).last_hidden_state.float()  # [18870, 256]

        # 3. Extract perturbed gene embeddings
        pert_embs = []
        for pid in pert_ids:
            base = pid.split(".")[0]  # strip version suffix (e.g., "ENSG00000001084.7" → "ENSG00000001084")
            node_idx = self.gene_to_node.get(base, -1)
            if node_idx >= 0:
                pert_embs.append(gnn_out[node_idx])
            else:
                # Gene not in STRING GNN (6.5% of training data)
                pert_embs.append(self.unknown_emb.float())
        pert_embs = torch.stack(pert_embs, dim=0)  # [B, 256]

        # 4. Gene query embeddings (learned, output-side)
        gene_idx = torch.arange(N_GENES_OUT, device=device)
        gene_q = self.gene_query(gene_idx).float()  # [6640, 256]

        # 5. Broadcast and concatenate: [B, 6640, 512]
        p_exp = pert_embs.unsqueeze(1).expand(B, N_GENES_OUT, self.GNN_DIM)  # [B, 6640, 256]
        g_exp = gene_q.unsqueeze(0).expand(B, N_GENES_OUT, self.GNN_DIM)     # [B, 6640, 256]
        combined = torch.cat([p_exp, g_exp], dim=-1)  # [B, 6640, 512]

        # 6. MLP head: [B*6640, 512] → [B*6640, 3] → [B, 6640, 3]
        logits = self.head(combined.reshape(B * N_GENES_OUT, -1))  # [B*6640, 3]
        logits = logits.view(B, N_GENES_OUT, N_CLASSES)             # [B, 6640, 3]

        return logits.permute(0, 2, 1)  # [B, 3, 6640]


# ──────────────────────────────────────────────────────────────────────────────
# Metric helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_deg_f1(y_pred: np.ndarray, y_true_remapped: np.ndarray) -> float:
    """Compute per-gene macro-averaged F1 score (matches calc_metric.py logic)."""
    n_genes = y_true_remapped.shape[1]
    f1_vals: List[float] = []
    y_hat = y_pred.argmax(axis=1)  # [n_samples, n_genes]
    for g in range(n_genes):
        yt = y_true_remapped[:, g]
        yp = y_hat[:, g]
        present = np.array([(yt == c).any() for c in range(3)])
        pf1 = sk_f1_score(yt, yp, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pf1[present].mean()))
    return float(np.mean(f1_vals))


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────
class DEGLightningModule(LightningModule):
    def __init__(
        self,
        head_hidden: int = 512,
        dropout: float = 0.15,
        lr: float = 1e-4,
        lr_head_multiplier: float = 3.0,
        weight_decay: float = 1e-4,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        max_epochs: int = 200,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: Optional[StringGNNDEGModel] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_indices: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_indices: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            # Build ENSG → node index mapping
            node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
            gene_to_node = {g: i for i, g in enumerate(node_names)}
            self.print(f"[setup] STRING GNN node vocab: {len(gene_to_node)} genes")

            self.model = StringGNNDEGModel(
                gene_to_node=gene_to_node,
                head_hidden=self.hparams.head_hidden,
                dropout=self.hparams.dropout,
            )

            # Loss function
            self.loss_fn = FocalLoss(
                gamma=self.hparams.focal_gamma,
                weight=CLASS_WEIGHTS,
                label_smoothing=self.hparams.label_smoothing,
            )

        # Populate test metadata
        if stage in ("test", None):
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
                dm = self.trainer.datamodule
                if hasattr(dm, "test_pert_ids") and dm.test_pert_ids:
                    self._test_pert_ids = dm.test_pert_ids
                    self._test_symbols = dm.test_symbols

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.model(batch["pert_ids"])  # [B, 3, 6640]

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, 3, 6640], labels: [B, 6640]
        B, C, G = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C).float()  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                               # [B*6640]
        return self.loss_fn(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()  # [B, 3, 6640]
        self._val_preds.append(probs)
        self._val_labels.append(batch["label"].cpu())
        self._val_indices.append(batch["idx"].cpu())

    def on_validation_epoch_end(self) -> None:
        # Gather from all ranks
        lp = torch.cat(self._val_preds, 0) if self._val_preds else torch.zeros(0, N_CLASSES, N_GENES_OUT)
        ll = torch.cat(self._val_labels, 0) if self._val_labels else torch.zeros(0, N_GENES_OUT, dtype=torch.long)
        li = torch.cat(self._val_indices, 0) if self._val_indices else torch.zeros(0, dtype=torch.long)

        ap = self.all_gather(lp)   # [world_size, B, 3, 6640] or [B, 3, 6640]
        al = self.all_gather(ll)
        ai = self.all_gather(li)

        self._val_preds.clear()
        self._val_labels.clear()
        self._val_indices.clear()

        # De-duplicate by sample index and compute metric
        preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
        labels = al.view(-1, N_GENES_OUT).cpu().numpy()
        idxs = ai.view(-1).cpu().numpy()
        _, uniq = np.unique(idxs, return_index=True)
        f1_val = compute_deg_f1(preds[uniq], labels[uniq])
        self.log("val_f1", f1_val, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch)
        probs = F.softmax(logits.detach().float(), dim=1).cpu()
        self._test_preds.append(probs)
        self._test_indices.append(batch["idx"].cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        lp = torch.cat(self._test_preds, 0)
        li = torch.cat(self._test_indices, 0)
        ap = self.all_gather(lp)
        ai = self.all_gather(li)
        self._test_preds.clear()
        self._test_indices.clear()

        if self.trainer.is_global_zero:
            preds = ap.view(-1, N_CLASSES, N_GENES_OUT).cpu().numpy()
            idxs = ai.view(-1).cpu().numpy()
            _, uniq = np.unique(idxs, return_index=True)
            preds = preds[uniq]
            idxs = idxs[uniq]
            order = np.argsort(idxs)
            preds = preds[order]
            idxs = idxs[order]

            output_dir = Path(__file__).parent / "run"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "idx": self._test_pert_ids[i],
                    "input": self._test_symbols[i],
                    "prediction": json.dumps(preds[r].tolist()),
                }
                for r, i in enumerate(idxs)
            ]
            pred_path = output_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(pred_path, sep="\t", index=False)
            self.print(f"Test predictions saved → {pred_path}")

    def configure_optimizers(self):
        # Separate learning rates: backbone GNN (lower) vs. other params (higher)
        gnn_params = list(self.model.gnn.parameters())
        other_params = [p for n, p in self.model.named_parameters()
                        if not n.startswith("gnn.") and p.requires_grad]

        opt = torch.optim.AdamW([
            {"params": gnn_params, "lr": self.hparams.lr},
            {"params": other_params, "lr": self.hparams.lr * self.hparams.lr_head_multiplier},
        ], weight_decay=self.hparams.weight_decay)

        # Cosine annealing LR
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_epochs, eta_min=1e-7
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and persistent buffers."""
        full_state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_state_dict:
                    trainable_state_dict[key] = full_state_dict[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_state_dict:
                trainable_state_dict[key] = full_state_dict[key]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving checkpoint: {trainable}/{total} params "
            f"({100 * trainable / total:.2f}%)"
        )
        return trainable_state_dict

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Node 3-1: STRING GNN DEG predictor")
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "data"),
    )
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-head-multiplier", type=float, default=3.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--head-hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--early-stopping-patience", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(0)
    args = parse_args()

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() or 1))
    accumulate_grad = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    fast_dev_run = 1 if args.fast_dev_run else False
    max_steps = args.debug_max_step if args.debug_max_step is not None else -1
    limit = args.debug_max_step if args.debug_max_step is not None else 1.0

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="node3-1-{epoch:03d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=args.early_stopping_patience,
        verbose=True,
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
            find_unused_parameters=True,
            timeout=timedelta(seconds=300),
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate_grad,
        limit_train_batches=limit,
        limit_val_batches=limit,
        limit_test_batches=limit,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    datamodule = DEGDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    model_module = DEGLightningModule(
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        lr=args.lr,
        lr_head_multiplier=args.lr_head_multiplier,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
    )

    trainer.fit(model_module, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model_module, datamodule=datamodule)
    else:
        test_results = trainer.test(
            model_module, datamodule=datamodule, ckpt_path="best"
        )

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        primary_val = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None
            else float("nan")
        )
        score_path.write_text(
            f"# Node 3-1 Test Evaluation Results\n"
            f"# Primary metric: f1_score (macro-averaged per-gene F1)\n"
            f"# Model: STRING GNN + ESM2 conditioning\n"
            f"f1_score: {primary_val:.6f}\n"
            f"val_f1_best: {primary_val:.6f}\n"
        )
        print(f"Test score saved → {score_path}")


if __name__ == "__main__":
    main()
