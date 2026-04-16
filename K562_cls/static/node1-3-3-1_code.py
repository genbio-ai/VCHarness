"""Node 1-3-3-1: STRING_GNN-Only K=16 Neighborhood Attention + GenePriorBias (Clean Reboot).

Strategy: Abandon the AIDO.Cell-10M fusion (confirmed dead-end in parent node1-3-3, F1=0.4647)
and return to the proven STRING_GNN-only architecture from node1-1-1-1-2-1 (F1=0.4913, best
STRING-only in tree). Apply precise hyperparameter improvements from parent feedback:
  - Remove AIDO.Cell-10M entirely (3+ independent experiments confirm fusion failure)
  - Use proven optimal hyperparameters: lr=3e-4, wd=4e-2, dropout=0.40, T_max=200, patience=25
  - GenePriorBias warmup=40 (parent's warmup=20 caused disruptive val/f1 spike at epoch 20)
  - Single unified learning rate (discriminative LR caused inferior convergence in node1-3-2)

Memory connections:
- node1-1-1-1-2-1 (F1=0.4913): direct architecture template — K=16 + GenePriorBias warmup=40
- parent node1-3-3 feedback: explicit recommendation to abandon AIDO.Cell-10M fusion
- node2-1-1-1-1-1 (F1=0.5128): T_max=200 inspired; patience=25 to catch late-epoch spikes
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3
# Remapped class frequencies (after -1->0, 0->1, 1->2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights; neutral class stays ~ 1."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  - softmax probabilities
        targets: [N, G]    long   - class labels in {0, 1, 2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)       # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)      # [N, G]
        is_pred = (y_hat == c)        # [N, G]
        present = is_true.any(dim=0)  # [G]

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
# Pre-computation utilities (STRING_GNN)
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_string_gnn_embeddings() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load STRING_GNN and compute all node embeddings. Returns (emb[N,256], pert_id->idx)."""
    model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
    model.eval()
    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
    node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())

    edge_index = graph["edge_index"]
    ew = graph.get("edge_weight", None)

    outputs = model(edge_index=edge_index, edge_weight=ew)
    emb = outputs.last_hidden_state.float().cpu()  # [18870, 256]

    pert_to_idx = {name: i for i, name in enumerate(node_names)}
    del model
    return emb, pert_to_idx


@torch.no_grad()
def precompute_neighborhood(
    emb: torch.Tensor,
    K: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute top-K neighbor indices and normalized edge weights.

    Returns:
        neighbor_indices [N, K] long   - STRING_GNN node indices (-1 = padding)
        neighbor_weights [N, K] float  - softmax-normalized STRING confidence weights
    """
    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
    edge_index = graph["edge_index"]
    ew = graph.get("edge_weight", None)

    N = emb.shape[0]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    weights = ew.tolist() if ew is not None else [1.0] * len(src)

    adj: Dict[int, List[Tuple[int, float]]] = {}
    for s, d, w in zip(src, dst, weights):
        adj.setdefault(s, []).append((d, w))

    neighbor_indices = torch.full((N, K), -1, dtype=torch.long)
    neighbor_weights = torch.zeros(N, K, dtype=torch.float32)

    for node in range(N):
        nbrs = adj.get(node, [])
        if not nbrs:
            continue
        nbrs_sorted = sorted(nbrs, key=lambda x: -x[1])[:K]
        for j, (nb_idx, nb_w) in enumerate(nbrs_sorted):
            neighbor_indices[node, j] = nb_idx
            neighbor_weights[node, j] = nb_w

    # Softmax-normalize valid neighbor weights per node
    mask = neighbor_indices >= 0   # [N, K]
    raw  = neighbor_weights.clone()
    raw[~mask] = -1e9
    norm_w = torch.softmax(raw, dim=-1)
    norm_w[~mask] = 0.0

    return neighbor_indices, norm_w


# ---------------------------------------------------------------------------
# Neighborhood Attention Aggregator (proven node1-1-1-1-2-1 design, K=16, attn_dim=64)
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Aggregate top-K PPI neighbors for a center gene using learned attention.

    Identical to the proven node1-1-1-1-2-1 implementation (K=16, attn_dim=64):
    - Pairwise attention scoring over (center, neighbor) embeddings
    - STRING confidence as an additive prior on attention scores
    - Gated residual connection: center + gate * aggregated
    """

    def __init__(self, emb_dim: int = 256, attn_dim: int = 64) -> None:
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False),
        )
        self.gate_proj = nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        center_emb: torch.Tensor,       # [B, D]
        neighbor_emb: torch.Tensor,     # [B, K, D]
        neighbor_weights: torch.Tensor, # [B, K] pre-normalized edge weights
        valid_mask: torch.Tensor,       # [B, K] bool
    ) -> torch.Tensor:
        """Returns aggregated representation [B, D]."""
        B, K, D = neighbor_emb.shape
        center_exp = center_emb.unsqueeze(1).expand(-1, K, -1)    # [B, K, D]
        pair = torch.cat([center_exp, neighbor_emb], dim=-1)       # [B, K, 2D]
        attn_scores = self.attn_proj(pair).squeeze(-1)             # [B, K]

        # Combine learned scores with STRING confidence as prior
        attn_scores = attn_scores + neighbor_weights
        attn_scores = attn_scores.masked_fill(~valid_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)          # [B, K]
        attn_weights = attn_weights * valid_mask.float()

        aggregated = (attn_weights.unsqueeze(-1) * neighbor_emb).sum(dim=1)  # [B, D]
        gate = torch.sigmoid(self.gate_proj(center_emb))                      # [B, D]
        return center_emb + gate * aggregated                                  # [B, D]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    """K562 DEG prediction dataset."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        self.sample_indices = list(range(len(df)))
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
            "sample_idx": idx,
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]   # [G] in {0, 1, 2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx": torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":    [b["pert_id"]  for b in batch],
        "symbol":     [b["symbol"]   for b in batch],
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
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df)
        self.val_ds   = DEGDataset(val_df)
        self.test_ds  = DEGDataset(test_df)

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
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Lightning Model
# ---------------------------------------------------------------------------
class StringGNNDEGModel(pl.LightningModule):
    """Frozen STRING_GNN (K=16 Neighborhood Attention) + GenePriorBias + Bilinear head.

    This is the proven node1-1-1-1-2-1 architecture (F1=0.4913, best STRING-only)
    with improved hyperparameters guided by parent feedback:
      - T_max=200 (was 150 in node1-1-1-1-2-1, 120 in parent)
      - dropout=0.40, weight_decay=4e-2 (exact proven values)
      - GenePriorBias warmup=40 (was 20 in parent, caused disruptive epoch-20 spike)
      - patience=25, max_epochs=300 (allows late convergence)

    Architecture:
      pert_id -> STRING_GNN (frozen cached [18870,256])
             -> NeighborhoodAttn (K=16, attn_dim=64) -> [B, 256]
             -> LayerNorm(256)
             -> Bilinear: h[B,256] . gene_class_emb[3,6640,256] -> logits[B,3,6640]
             -> GenePriorBias [3, 6640] (active after epoch gene_prior_warmup=40)
    """

    def __init__(
        self,
        bilinear_dim: int = 256,
        attn_dim: int = 64,
        K: int = 16,
        dropout: float = 0.40,
        lr: float = 3e-4,
        weight_decay: float = 4e-2,
        warmup_epochs: int = 10,
        t_max: int = 200,
        eta_min: float = 5e-6,
        label_smoothing: float = 0.05,
        gene_prior_warmup: int = 40,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Collect all pert_ids across all splits for consistent index mapping ----
        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        all_pert_ids = (
            train_df["pert_id"].tolist() +
            val_df["pert_id"].tolist() +
            test_df["pert_id"].tolist()
        )
        unique_sorted = sorted(set(all_pert_ids))
        self.pert_to_pos = {pid: i for i, pid in enumerate(unique_sorted)}

        # ---- STRING_GNN: precompute embeddings + neighborhood ----
        self.print("Precomputing STRING_GNN embeddings...")
        string_emb, pert_to_gnn_idx = precompute_string_gnn_embeddings()

        # Register as buffer so Lightning moves to GPU automatically
        self.register_buffer("node_embeddings", string_emb)   # [18870, 256]

        gnn_idx_tensor = torch.tensor(
            [pert_to_gnn_idx.get(pid, -1) for pid in unique_sorted], dtype=torch.long
        )
        self.register_buffer("pert_gnn_idx", gnn_idx_tensor)   # [M]

        self.print(f"Precomputing PPI neighborhood tables (K={hp.K})...")
        nb_indices, nb_weights = precompute_neighborhood(string_emb, K=hp.K)
        self.register_buffer("neighbor_indices", nb_indices)   # [18870, K]
        self.register_buffer("neighbor_weights", nb_weights)   # [18870, K]

        # Fallback embedding for pert_ids not in STRING_GNN
        self.fallback_emb = nn.Parameter(torch.zeros(1, 256))

        # ---- Trainable modules ----
        self.neighborhood_attn = NeighborhoodAttentionAggregator(
            emb_dim=256, attn_dim=hp.attn_dim
        )

        # LayerNorm on GNN output for stable optimization
        self.layer_norm = nn.LayerNorm(256)

        # Bilinear gene-class head: logits[b,c,g] = h[b] . gene_class_emb[c,g]
        # Small init to avoid saturated softmax at start
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # Dropout applied to GNN representation before bilinear head
        self.dropout = nn.Dropout(hp.dropout)

        # GenePriorBias [N_CLASSES, N_GENES]: per-gene per-class additive bias
        # Initialized from 0.3 * log(class_freq) for class-frequency prior
        # Use persistent=True buffer for the active flag to survive checkpoint loading
        class_freq = torch.tensor(CLASS_FREQ, dtype=torch.float32)
        log_freq   = torch.log(class_freq + 1e-9)                          # [3]
        bias_init  = 0.3 * log_freq.unsqueeze(1).expand(-1, N_GENES)       # [3, 6640]
        self.gene_prior_bias = nn.Parameter(bias_init.clone())              # [3, N_GENES]
        # Persistent buffer to track whether bias is active (survives checkpoint loading)
        self.register_buffer("bias_active", torch.tensor(False))

        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor] = []
        self._val_tgts:  List[torch.Tensor] = []
        self._val_idx:   List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    # ---- Embedding helper ----

    def _get_neighborhood_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Return [B, 256] neighborhood-attention-aggregated STRING_GNN embeddings."""
        pos = torch.tensor(
            [self.pert_to_pos[pid] for pid in pert_ids], dtype=torch.long, device=self.device
        )
        gnn_node_idx = self.pert_gnn_idx[pos]         # [B]
        valid_center = gnn_node_idx >= 0
        safe_center_idx = gnn_node_idx.clamp(min=0)
        center_emb_raw = self.node_embeddings[safe_center_idx]  # [B, 256]

        # Use fallback embedding for pert_ids not in STRING_GNN
        fallback = self.fallback_emb.expand(center_emb_raw.shape[0], -1).to(center_emb_raw.dtype)
        center_emb = torch.where(valid_center.unsqueeze(-1), center_emb_raw, fallback).float()

        K = self.hparams.K
        nb_idx = self.neighbor_indices[safe_center_idx]   # [B, K]
        nb_wts = self.neighbor_weights[safe_center_idx]   # [B, K]
        valid_mask = nb_idx >= 0                           # [B, K] bool

        safe_nb_idx = nb_idx.clamp(min=0)
        nb_emb = self.node_embeddings[safe_nb_idx].float()  # [B, K, 256]
        nb_emb = nb_emb * valid_mask.unsqueeze(-1).float()  # zero-out invalid

        return self.neighborhood_attn(center_emb, nb_emb, nb_wts, valid_mask)   # [B, 256]

    # ---- Epoch start hooks for bias activation ----

    def on_train_epoch_start(self) -> None:
        """Activate GenePriorBias after warmup epochs."""
        if self.current_epoch >= self.hparams.gene_prior_warmup:
            if not self.bias_active.item():
                self.print(f"[Epoch {self.current_epoch}] Activating GenePriorBias.")
            self.bias_active.fill_(True)

    def on_test_epoch_start(self) -> None:
        """Ensure bias is active at test time (invariant to checkpoint loading)."""
        self.bias_active.fill_(True)

    def on_validation_epoch_start(self) -> None:
        """Activate bias during validation if past warmup."""
        if self.current_epoch >= self.hparams.gene_prior_warmup:
            self.bias_active.fill_(True)

    # ---- Forward ----

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        # Get STRING_GNN neighborhood-aggregated embeddings
        h = self._get_neighborhood_emb(pert_ids)   # [B, 256] float32

        # LayerNorm + Dropout for regularization
        h = self.layer_norm(h)
        h = self.dropout(h)

        # Bilinear gene-class head: captures perturbation-gene co-regulation
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)  # [B, 3, G]

        # GenePriorBias: active only after warmup (via persistent buffer, safe across checkpoint loading)
        if self.bias_active.item():
            logits = logits + self.gene_prior_bias.unsqueeze(0)  # [B, 3, G] + [1, 3, G]

        return logits

    # ---- Loss ----

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        loss = F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),   # [B*G, 3]
            targets.reshape(-1),                        # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )
        return loss

    # ---- Training/Validation/Test steps ----

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["pert_id"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["pert_id"])
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
        local_preds = torch.cat(self._val_preds, dim=0)    # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  dim=0)    # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   dim=0)    # [N_local]
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        # Gather from all GPUs
        all_preds = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)    # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)     # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        # De-duplicate across GPUs (DDP may duplicate boundary samples)
        order  = torch.argsort(idx_flat)
        s_idx  = idx_flat[order]
        s_pred = preds_flat[order]
        s_tgt  = tgts_flat[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device),
                            s_idx[1:] != s_idx[:-1]])
        preds_dedup = s_pred[mask]
        tgts_dedup  = s_tgt[mask]

        f1 = compute_per_gene_f1(preds_dedup, tgts_dedup)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)   # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)   # [N_local]
        all_preds   = self.all_gather(local_preds)          # [W, N_local, 3, G]
        all_idx     = self.all_gather(local_idx)            # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            order   = torch.argsort(idx_flat)
            s_idx   = idx_flat[order]
            s_pred  = preds_flat[order]
            mask    = torch.cat([torch.ones(1, dtype=torch.bool, device=s_idx.device),
                                 s_idx[1:] != s_idx[:-1]])
            preds_dedup = s_pred[mask]             # [N_test, 3, G]
            unique_sid  = s_idx[mask].tolist()     # [N_test]

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                           for i in range(len(test_df))}

            rows = []
            for dedup_pos, sid in enumerate(unique_sid):
                pid, sym = idx_to_meta[int(sid)]
                pred_list = preds_dedup[dedup_pos].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"[Node1-3-3-1] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ---- Checkpoint helpers ----

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters + persistent buffers."""
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
        bufs  = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Checkpoint: {train}/{total} params ({100 * train / total:.2f}%), "
            f"plus {bufs} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer ----

    def configure_optimizers(self):
        hp = self.hparams

        # Single unified learning rate (discriminative LR caused inferior convergence
        # in node1-3-2; proven single lr=3e-4 achieves F1=0.4913 in node1-1-1-1-2-1)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )

        warmup_sch = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=hp.warmup_epochs,
        )
        cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=hp.t_max, eta_min=hp.eta_min,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_sch, cosine_sch], milestones=[hp.warmup_epochs],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-3-3-1: STRING_GNN K=16 Neighborhood Attention + GenePriorBias"
    )
    # Batch size
    parser.add_argument("--micro-batch-size",  type=int,   default=32,
                        help="Per-GPU micro batch size")
    parser.add_argument("--global-batch-size", type=int,   default=256,
                        help="Global batch size (must be multiple of micro_batch_size*8)")
    # Training schedule
    parser.add_argument("--max-epochs",        type=int,   default=300)
    parser.add_argument("--warmup-epochs",     type=int,   default=10)
    parser.add_argument("--t-max",             type=int,   default=200)
    parser.add_argument("--eta-min",           type=float, default=5e-6)
    # Learning rate
    parser.add_argument("--lr",                type=float, default=3e-4,
                        help="Single unified learning rate for all trainable params")
    parser.add_argument("--weight-decay",      type=float, default=4e-2)
    # Architecture
    parser.add_argument("--bilinear-dim",      type=int,   default=256)
    parser.add_argument("--attn-dim",          type=int,   default=64)
    parser.add_argument("--K",                 type=int,   default=16)
    parser.add_argument("--dropout",           type=float, default=0.40)
    parser.add_argument("--label-smoothing",   type=float, default=0.05)
    parser.add_argument("--gene-prior-warmup", type=int,   default=40,
                        help="Number of epochs before activating GenePriorBias")
    # EarlyStopping
    parser.add_argument("--patience",          type=int,   default=25)
    parser.add_argument("--min-delta",         type=float, default=1e-4)
    # Training utilities
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--debug-max-step",    type=int,   default=None,
                        help="Limit train/val/test batches for debugging")
    parser.add_argument("--fast-dev-run",      action="store_true")
    args = parser.parse_args()

    # ---- Output directories ----
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Trainer configuration ----
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(1, n_gpus)

    accumulate_grad_batches = max(
        1, args.global_batch_size // (args.micro_batch_size * n_gpus)
    )

    # Batch limits for debugging
    if args.debug_max_step is not None:
        limit_train_batches = args.debug_max_step
        limit_val_batches   = args.debug_max_step
        limit_test_batches  = args.debug_max_step
        max_epochs_for_trainer = 1
    else:
        limit_train_batches = 1.0
        limit_val_batches   = 1.0
        limit_test_batches  = 1.0
        max_epochs_for_trainer = args.max_epochs

    fast_dev_run = args.fast_dev_run

    # ---- Callbacks ----
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=False,
    )
    early_stop_callback = EarlyStopping(
        monitor="val/f1",
        patience=args.patience,
        mode="max",
        min_delta=args.min_delta,
        verbose=True,
    )
    lr_monitor    = LearningRateMonitor(logging_interval="epoch")
    progress_bar  = TQDMProgressBar(refresh_rate=20)

    # ---- Loggers ----
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tensorboard_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # ---- Trainer ----
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=max_epochs_for_trainer,
        accumulate_grad_batches=accumulate_grad_batches,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, progress_bar],
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    # ---- Model and DataModule ----
    model = StringGNNDEGModel(
        bilinear_dim=args.bilinear_dim,
        attn_dim=args.attn_dim,
        K=args.K,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        t_max=args.t_max,
        eta_min=args.eta_min,
        label_smoothing=args.label_smoothing,
        gene_prior_warmup=args.gene_prior_warmup,
    )

    datamodule = DEGDataModule(
        batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # ---- Train ----
    trainer.fit(model, datamodule=datamodule)

    # ---- Test (use best checkpoint unless debugging) ----
    if fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print(f"Test results: {test_results}")

    # ---- Save test score ----
    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1-3-3-1] Test results saved to {score_path}")


if __name__ == "__main__":
    main()
