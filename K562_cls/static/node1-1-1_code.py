"""Node 1-2 – Frozen STRING_GNN + Focal Loss + Deep Residual Head.

Key improvements over node1-1 (parent, test F1=0.4527):
1. Frozen STRING_GNN backbone with pre-computed embeddings — eliminates the costly
   per-step full GNN forward pass (~76ms/step), enabling longer training with same budget.
2. Focal loss (gamma=2.0) with per-class alpha weights — replaces weighted CE + label
   smoothing to better focus on hard minority-class (up/down DEG) examples.
3. Deep residual MLP head: LayerNorm → Linear(256→512) → GELU → Dropout → 2×ResBlock(512)
   → LayerNorm → bilinear. Two-stage residual blocks improve representation expressiveness.
4. Bilinear dim increased 256 → 512 for richer gene-perturbation interaction space.
5. Per-gene-class bias: learnable [3, 6640] bias captures baseline DEG tendency per gene
   (some genes are almost always neutral; others frequently up/down-regulated).
6. Scaled bilinear dot product: divide by sqrt(bilinear_dim) for numerical stability.
7. Linear warmup (20 epochs, start_factor=0.1) + CosineAnnealingWarmRestarts (T_0=100,
   T_mult=2) — properly implements warmup that was declared but never used in node1-1.
8. Dropout reduced 0.4 → 0.2: frozen backbone provides stable input features, reducing
   need for heavy dropout in the downstream head.
9. Extended training: max_epochs=300, early_stopping patience=20 (from 15).
10. Head-only optimization: only head params receive gradients (backbone frozen).
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3

# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

STRING_DIM = 256  # STRING_GNN hidden dimension


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights for focal loss alpha."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def load_string_gnn_mapping() -> Dict[str, int]:
    """Load STRING_GNN node_names.json → Ensembl-ID to node-index mapping."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    return {name: idx for idx, name in enumerate(node_names)}


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Multi-class focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Args:
        logits:  [N, C]  unnormalized class scores
        targets: [N]     integer class labels in {0, ..., C-1}
        alpha:   [C]     per-class weight tensor (e.g., sqrt-inverse-freq)
        gamma:   float   focusing parameter (0 = standard CE, 2 = standard focal)
    Returns:
        Scalar loss.
    """
    # Log-softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)                        # [N, C]
    # Log-probability of the true class
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)    # [N]
    pt = torch.exp(log_pt)                                            # [N]
    # Per-sample class weight (normalize so mean weight = 1)
    alpha_t = alpha[targets]                                          # [N]
    alpha_t = alpha_t / (alpha.sum() + 1e-9) * len(alpha)
    # Focal modulation
    loss = -alpha_t * (1.0 - pt) ** gamma * log_pt
    return loss.mean()


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0, 1, 2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)           # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)           # [N, G]
        is_pred = (y_hat == c)             # [N, G]
        present = is_true.any(dim=0)       # [G]

        tp = (is_pred & is_true).float().sum(0)
        fp = (is_pred & ~is_true).float().sum(0)
        fn = (~is_pred & is_true).float().sum(0)

        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(
            prec + rec > 0,
            2 * prec * rec / (prec + rec + 1e-8),
            torch.zeros_like(prec),
        )
        f1_per_gene += f1_c * present.float()
        n_present   += present.float()

    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    """K562 DEG prediction dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_map: Dict[str, int],
    ) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()

        # STRING_GNN node index for each sample (-1 means not in STRING)
        self.string_node_indices = torch.tensor(
            [string_map.get(p, -1) for p in self.pert_ids], dtype=torch.long
        )

        has_label = "label" in df.columns and df["label"].notna().all()
        if has_label:
            self.labels = [
                torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
                for row in df["label"].tolist()
            ]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx":        idx,
            "pert_id":           self.pert_ids[idx],
            "symbol":            self.symbols[idx],
            "string_node_idx":   self.string_node_indices[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]  # [G] in {0, 1, 2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Use the sample's ORIGINAL position in the dataset as the global index.
    # This is the row number in the original TSV file (0 to 153 for test set).
    # This is independent of Lightning DDP sharding — no formula needed.
    out: Dict[str, Any] = {
        "sample_idx": torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":         [b["pert_id"]  for b in batch],
        "symbol":          [b["symbol"]   for b in batch],
        "string_node_idx": torch.stack([b["string_node_idx"]   for b in batch]),
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.string_map: Optional[Dict[str, int]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.string_map is None:
            self.string_map = load_string_gnn_mapping()

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df, self.string_map)
        self.val_ds   = DEGDataset(val_df,   self.string_map)
        self.test_ds  = DEGDataset(test_df,  self.string_map)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        sampler = SequentialSampler(self.test_ds)
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
            sampler=sampler,
        )


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Pre-norm residual feed-forward block."""

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.fc1   = nn.Linear(dim, dim * expansion)
        self.act   = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(dim * expansion, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.drop2(self.fc2(self.drop1(self.act(self.fc1(h)))))
        return x + h


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FrozenStringGNNResidualModel(pl.LightningModule):
    """Frozen STRING_GNN with deep residual head and focal loss.

    Architecture:
        1. STRING_GNN run ONCE at setup() to pre-compute node embeddings [18870, 256]
           → stored as a fixed buffer; no GNN gradients during training.
        2. Lookup pre-computed embedding by pert_id string_node_idx → [B, 256]
           (learnable fallback for ~6.4% unknown pert_ids)
        3. Deep residual projection:
             LayerNorm(256) → Linear(256→bilinear_dim) → GELU → Dropout
             → ResBlock(bilinear_dim) × 2 → LayerNorm(bilinear_dim)
        4. Scaled bilinear output + gene bias:
             logits[b,c,g] = (h[b] · gene_class_emb[c,g]) / sqrt(D) + gene_bias[c,g]
        5. Focal loss (gamma=2.0) with sqrt-inverse-freq alpha weights
    """

    def __init__(
        self,
        bilinear_dim: int = 512,
        dropout: float = 0.2,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_epochs: int = 20,
        T_0: int = 100,
        T_mult: int = 2,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Model layers initialized in setup()

    def setup(self, stage: Optional[str] = None) -> None:
        # Guard against repeated setup calls
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        hp = self.hparams

        # ----------------------------------------------------------------
        # 1. Pre-compute STRING_GNN node embeddings (backbone stays frozen)
        # Model is at a local path — all ranks load independently (no download).
        # The barrier has been removed to avoid potential DDP deadlocks.
        # ----------------------------------------------------------------
        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph["edge_index"].long()
        edge_weight = graph["edge_weight"].float()

        # One-time forward pass on CPU — produces the fixed lookup table
        with torch.no_grad():
            gnn_out  = backbone(edge_index=edge_index, edge_weight=edge_weight)
            node_emb = gnn_out.last_hidden_state.float().detach()  # [18870, 256]

        # Register as buffer → Lightning moves it to GPU automatically
        self.register_buffer("node_embeddings", node_emb)

        # Release backbone memory (not needed again)
        del backbone, graph, edge_index, edge_weight, gnn_out

        # ----------------------------------------------------------------
        # 2. Learnable fallback for unknown pert_ids (~6.4% of training data)
        # ----------------------------------------------------------------
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 3. Deep residual head: STRING_DIM → bilinear_dim
        # ----------------------------------------------------------------
        self.proj = nn.Sequential(
            nn.LayerNorm(STRING_DIM),
            nn.Linear(STRING_DIM, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )
        self.res1     = ResBlock(hp.bilinear_dim, expansion=2, dropout=hp.dropout)
        self.res2     = ResBlock(hp.bilinear_dim, expansion=2, dropout=hp.dropout)
        self.out_norm = nn.LayerNorm(hp.bilinear_dim)

        # ----------------------------------------------------------------
        # 4. Bilinear gene-class embedding + per-gene bias
        # ----------------------------------------------------------------
        # gene_class_emb: interaction matrix for pert_emb × gene_class embedding
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )
        # gene_bias: learnable per-gene-class prior (captures baseline DEG tendencies)
        self.gene_bias = nn.Parameter(torch.zeros(N_CLASSES, N_GENES))

        # Class weights for focal loss alpha
        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        for _, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators for val/test (cleared each epoch)
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _get_pert_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Lookup pre-computed embeddings; use learnable fallback for unknowns.

        Args:
            string_node_idx: [B] long tensor, -1 for pert_ids not in STRING.
        Returns:
            [B, STRING_DIM] float32 perturbation embeddings.
        """
        B = string_node_idx.shape[0]
        emb = torch.zeros(
            B, STRING_DIM,
            dtype=self.node_embeddings.dtype,
            device=self.node_embeddings.device,
        )
        known   = string_node_idx >= 0
        unknown = ~known

        if known.any():
            emb[known] = self.node_embeddings[string_node_idx[known]]

        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(unknown.sum(), dtype=torch.long, device=self.node_embeddings.device)
            ).to(self.node_embeddings.dtype)
            emb[unknown] = fb

        return emb.float()

    def forward(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        pert_emb = self._get_pert_embeddings(string_node_idx)  # [B, STRING_DIM]
        h = self.proj(pert_emb)                                 # [B, bilinear_dim]
        h = self.res1(h)                                        # [B, bilinear_dim]
        h = self.res2(h)                                        # [B, bilinear_dim]
        h = self.out_norm(h)                                    # [B, bilinear_dim]

        # Scaled bilinear interaction + per-gene bias
        scale  = h.shape[-1] ** -0.5
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb) * scale  # [B, 3, G]
        logits = logits + self.gene_bias.unsqueeze(0)                          # [B, 3, G]
        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return focal_loss(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, 3]
            targets.reshape(-1),                       # [B*G]
            alpha=self.class_weights,
            gamma=self.hparams.focal_gamma,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["string_node_idx"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits, dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, dim=0)   # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  dim=0)   # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   dim=0)   # [N_local]
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        # Gather across all DDP ranks
        all_preds = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)    # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)     # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        # De-duplicate (DDP padding may introduce repeated samples)
        order  = torch.argsort(idx_flat)
        s_idx  = idx_flat[order]
        s_pred = preds_flat[order]
        s_tgt  = tgts_flat[order]
        mask   = torch.cat([
            torch.tensor([True], device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        preds_dedup = s_pred[mask]
        tgts_dedup  = s_tgt[mask]

        f1 = compute_per_gene_f1(preds_dedup, tgts_dedup)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        # Concatenate local predictions and sample indices.
        # sample_idx contains the ORIGINAL row positions in the test TSV (0 to 153).
        local_preds = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)  # [N_local]

        # Gather predictions and indices from ALL ranks via all_gather.
        # Each rank has a unique subset of the test set (DistributedSampler sharding).
        all_preds = self.all_gather(local_preds)  # [W, N_local, 3, G]
        all_idx   = self.all_gather(local_idx)    # [W, N_local]

        # Only rank 0 writes the final predictions file
        if self.trainer.is_global_zero:
            # Reshape gathered data to [W * N_local, 3, G] and [W * N_local]
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            # Build global prediction map: index → prediction (first occurrence wins)
            pred_map: Dict[int, torch.Tensor] = {}
            for i in range(len(idx_flat)):
                gid = int(idx_flat[i].item())
                if gid not in pred_map:
                    pred_map[gid] = preds_flat[i]

            # Iterate through test file in original row order (0 to 153)
            test_df = pd.read_csv(TEST_TSV, sep="\t")
            rows = []
            for i in range(len(test_df)):
                if i not in pred_map:
                    continue  # skip missing samples
                pid = test_df.iloc[i]["pert_id"]
                sym = test_df.iloc[i]["symbol"]
                pred = pred_map[i].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-2] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ------------------------------------------------------------------
    # Checkpoint helpers — save only trainable params + buffers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full:
                    trainable[key] = full[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full:
                trainable[key] = full[key]
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {train}/{total} params ({100 * train / total:.1f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: head-only AdamW + linear warmup + CosineAnnealingWarmRestarts
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        # Only head parameters — backbone is frozen
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=hp.lr, weight_decay=hp.weight_decay)

        # Phase 1: linear warmup from 0.1×lr to lr over warmup_epochs epochs
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        # Phase 2: cosine annealing with warm restarts (T_0=100, T_mult=2)
        # Restart schedule: epochs 0→100, 100→300, 300→700, …
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=hp.T_0,
            T_mult=hp.T_mult,
            eta_min=1e-6,
        )
        # Sequential: warmup first, then CAWR
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[hp.warmup_epochs],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(description="Node1-2 – Frozen STRING_GNN Residual + Focal Loss")
    parser.add_argument("--micro-batch-size",  type=int,   default=32)
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument("--max-epochs",        type=int,   default=300)
    parser.add_argument("--lr",                type=float, default=3e-4)
    parser.add_argument("--weight-decay",      type=float, default=1e-2)
    parser.add_argument("--bilinear-dim",      type=int,   default=512)
    parser.add_argument("--dropout",           type=float, default=0.2)
    parser.add_argument("--warmup-epochs",     type=int,   default=20)
    parser.add_argument("--t-0",               type=int,   default=100,
                        dest="t_0")
    parser.add_argument("--t-mult",            type=int,   default=2,
                        dest="t_mult")
    parser.add_argument("--focal-gamma",       type=float, default=2.0,
                        dest="focal_gamma")
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--debug-max-step",    type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",      action="store_true",
                        dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Limit / debug logic
    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        lim_test  = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        lim_test  = 1.0
        max_steps = -1

    # val_check_interval: validate once per epoch (1.0).
    # In debug mode (lim_train=int), the entire training run is one epoch,
    # so validation fires exactly once at the end of the limited steps.
    val_check_interval = 1.0

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    # Model
    model = FrozenStringGNNResidualModel(
        bilinear_dim  = args.bilinear_dim,
        dropout       = args.dropout,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        warmup_epochs = args.warmup_epochs,
        T_0           = args.t_0,
        T_mult        = args.t_mult,
        focal_gamma   = args.focal_gamma,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
    )
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=20, min_delta=1e-4)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy
    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = n_gpus,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        max_steps               = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches     = lim_train,
        limit_val_batches       = lim_val,
        limit_test_batches      = lim_test,
        val_check_interval      = 1.0,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    # Save test score
    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
