"""Node 1-2 – Frozen STRING_GNN + K=16 PPI Neighborhood Attention + CosineAnnealingWarmRestarts.

Distinct from sibling node1-1-1-3-1 (which kept GenePriorBias and used T_max=150):
this node removes GenePriorBias entirely (proven net-negative in STRING_GNN lineage)
and replaces the single cosine schedule with CosineAnnealingWarmRestarts for periodic
local-minima escapes (proven effective in node4-3-2-1-1-1 which achieved F1=0.4921).

Key changes from parent (node1-1-1-3, F1=0.4610):
1. REMOVED GenePriorBias — the disruptive epoch-50 activation caused a sharp val loss
   spike (+0.417) and prevented the model from reaching node1-1-1-1-1's F1=0.4846.
   node1-1-1-1-1 (K=16 attention, NO GenePriorBias) is the best STRING_GNN-only node.
2. CosineAnnealingWarmRestarts(T_0=30, T_mult=2) replaces single CosineAnnealingLR.
   Cycles: 30→60→120→240 epochs. Each restart boosts LR back to peak (3e-4), allowing
   escape from local minima. Proven in node4-3-2-1-1-1: "warm restarts successfully
   escape local minima, val F1 monotonically improved across 7 cycles" (0.466→0.492).
3. REMOVED sample-level importance weighting — confirmed counterproductive per parent
   feedback ("adds complexity without proven benefit in this lineage").
4. Reduced label_smoothing 0.05 → 0.02 — sharper minority-class gradient signal.
5. Extended training: max_epochs=400, patience=30 — warm restarts need longer budget.
6. Reduced weight_decay 3e-2 → 2e-2 — matches proven node1-1-1-1 config (F1=0.4746).

Architecture:
    Pre-computed node_embeddings [18870, 256] (frozen STRING_GNN, computed once)
    → for each sample:
         center_emb = node_embeddings[pert_idx]           # [B, 256]
         neighbor_emb = node_embeddings[top_K_neighbors]  # [B, K, 256]
         attn_scores = MLP(concat(center, neighbor)) + edge_weight  # [B, K]
         aggregated = softmax(attn_scores) @ neighbor_emb            # [B, 256]
         h_agg = center + gate * aggregated               # [B, 256] (gated fusion)
    → head: LN(256) → Linear(256→256) → GELU → Dropout(0.35)
    → bilinear: logits[b,c,g] = h[b] · gene_class_emb[c,g]  # [B, 3, G]
    → Loss: weighted CE + label_smoothing=0.02
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
    """Sqrt-inverse-frequency weights; neutral class stays ~1."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def load_string_gnn_mapping() -> Dict[str, int]:
    """Load STRING_GNN node_names.json → Ensembl-ID to node-index mapping."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    return {name: idx for idx, name in enumerate(node_names)}


def precompute_neighborhood(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    n_nodes: int,
    K: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute top-K PPI neighbors for each node by edge confidence.

    Returns:
        neighbor_indices: [n_nodes, K] long — top-K neighbor node indices
                          (padded with -1 if fewer than K neighbors exist)
        neighbor_weights: [n_nodes, K] float — STRING confidence scores
    """
    src = edge_index[0]
    dst = edge_index[1]
    wgt = edge_weight

    # Sort by weight descending, then stable-sort by src
    sort_by_weight = torch.argsort(wgt, descending=True)
    src_sorted = src[sort_by_weight]
    dst_sorted = dst[sort_by_weight]
    wgt_sorted = wgt[sort_by_weight]

    sort_by_src = torch.argsort(src_sorted, stable=True)
    src_final = src_sorted[sort_by_src]
    dst_final = dst_sorted[sort_by_src]
    wgt_final = wgt_sorted[sort_by_src]

    counts = torch.bincount(src_final, minlength=n_nodes)

    neighbor_indices = torch.full((n_nodes, K), -1, dtype=torch.long)
    neighbor_weights = torch.zeros(n_nodes, K, dtype=torch.float32)

    start = 0
    for node_i in range(n_nodes):
        c = int(counts[node_i].item())
        if c == 0:
            start += c
            continue
        n_k = min(K, c)
        neighbor_indices[node_i, :n_k] = dst_final[start:start + n_k]
        neighbor_weights[node_i, :n_k] = wgt_final[start:start + n_k]
        start += c

    return neighbor_indices, neighbor_weights


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0, 1, 2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)  # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0)

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
            "sample_idx":      idx,
            "pert_id":         self.pert_ids[idx],
            "symbol":          self.symbols[idx],
            "string_node_idx": self.string_node_indices[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]   # [G] in {0, 1, 2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx":      torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":         [b["pert_id"]  for b in batch],
        "symbol":          [b["symbol"]   for b in batch],
        "string_node_idx": torch.stack([b["string_node_idx"] for b in batch]),
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
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Neighborhood Attention Module
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Center-context gated attention over top-K PPI neighbors.

    Proven innovation from node1-1-1-1-1 (F1=0.4846, +0.010 over node1-1-1-1).

    Architecture:
        attn_proj: [center(256) + neighbor(256)] → attn_dim(64) → score(1)
        attention = softmax(edge_weight + attn_proj_score)   # [B, K]
        aggregated = attention @ neighbor_emb                # [B, 256]
        gate = sigmoid(gate_proj(center_emb))                # [B, 256]
        output = center_emb + gate * aggregated              # [B, 256]
    """

    def __init__(self, embed_dim: int = 256, attn_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim  = attn_dim

        self.attn_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, 1),
        )
        self.gate_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        center_emb: torch.Tensor,         # [B, D]
        neighbor_emb: torch.Tensor,        # [B, K, D]
        neighbor_weights: torch.Tensor,    # [B, K]
        neighbor_mask: torch.Tensor,       # [B, K] bool: True = valid
    ) -> torch.Tensor:
        B, K, D = neighbor_emb.shape
        center_expanded = center_emb.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]
        pair_features = torch.cat([center_expanded, neighbor_emb], dim=-1)  # [B, K, 2D]
        attn_scores = self.attn_proj(pair_features).squeeze(-1)       # [B, K]
        attn_scores = attn_scores + neighbor_weights
        attn_scores = attn_scores.masked_fill(~neighbor_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)              # [B, K]
        aggregated = torch.bmm(attn_weights.unsqueeze(1), neighbor_emb).squeeze(1)  # [B, D]
        gate = torch.sigmoid(self.gate_proj(center_emb))
        return center_emb + gate * aggregated


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FrozenGNNNeighborhoodModel(pl.LightningModule):
    """Frozen STRING_GNN + K=16 PPI Neighborhood Attention + CosineAnnealingWarmRestarts.

    Architecture combines:
    1. Frozen STRING_GNN pre-computed embeddings (buffer, no GNN gradient)
    2. K=16 PPI neighborhood attention aggregation (proven +0.010 F1)
    3. Simple 2-layer MLP head (bilinear_dim=256) — proven optimal for 1,388 samples
    4. Bilinear gene-class embedding interaction
    5. CosineAnnealingWarmRestarts — periodic LR restarts escape local minima
       (GenePriorBias deliberately omitted: proven net-negative in STRING_GNN lineage)
    """

    def __init__(
        self,
        bilinear_dim: int = 256,
        K: int = 16,
        attn_dim: int = 64,
        dropout: float = 0.35,
        lr: float = 3e-4,
        weight_decay: float = 2e-2,
        warmup_epochs: int = 20,
        t_zero: int = 30,
        t_mult: int = 2,
        label_smoothing: float = 0.02,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        hp = self.hparams

        # ----------------------------------------------------------------
        # 1. Pre-compute STRING_GNN node embeddings (backbone stays frozen)
        # ----------------------------------------------------------------
        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph["edge_index"].long()
        edge_weight = graph["edge_weight"].float()

        with torch.no_grad():
            gnn_out  = backbone(edge_index=edge_index, edge_weight=edge_weight)
            node_emb = gnn_out.last_hidden_state.float().detach()  # [18870, 256]

        self.register_buffer("node_embeddings", node_emb)
        n_nodes = node_emb.shape[0]

        # ----------------------------------------------------------------
        # 2. Pre-compute top-K neighbors
        # ----------------------------------------------------------------
        self.print(f"Pre-computing top-{hp.K} PPI neighbors for {n_nodes} nodes...")
        nbr_idx, nbr_wgt = precompute_neighborhood(
            edge_index, edge_weight, n_nodes, K=hp.K
        )
        self.register_buffer("neighbor_indices", nbr_idx)  # [n_nodes, K]
        self.register_buffer("neighbor_weights", nbr_wgt)  # [n_nodes, K]

        del backbone, graph, edge_index, edge_weight, gnn_out

        # ----------------------------------------------------------------
        # 3. Learnable fallback for unknown pert_ids
        # ----------------------------------------------------------------
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 4. Neighborhood Attention (proven config: K=16, attn_dim=64)
        # ----------------------------------------------------------------
        self.neighborhood_attn = NeighborhoodAttentionAggregator(
            embed_dim=STRING_DIM,
            attn_dim=hp.attn_dim,
        )

        # ----------------------------------------------------------------
        # 5. Simple 2-layer MLP head (proven optimal from node1-1-1-1)
        #    LayerNorm(256) → Linear(256→bilinear_dim) → GELU → Dropout
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(STRING_DIM),
            nn.Linear(STRING_DIM, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # ----------------------------------------------------------------
        # 6. Bilinear gene-class embedding [3, G, bilinear_dim]
        # ----------------------------------------------------------------
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # Class weights for weighted CE
        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        for _, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_meta:  List[Dict]         = []

    # ------------------------------------------------------------------
    # Embedding lookup with PPI neighborhood aggregation
    # ------------------------------------------------------------------
    def _get_pert_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Lookup pre-computed embeddings with PPI neighborhood aggregation.

        Args:
            string_node_idx: [B] long tensor, -1 for pert_ids not in STRING.
        Returns:
            [B, STRING_DIM] aggregated perturbation embeddings.
        """
        B = string_node_idx.shape[0]
        dev = self.node_embeddings.device
        K   = self.neighbor_indices.shape[1]

        # Base center embeddings (fallback for unknowns)
        center_emb = torch.zeros(B, STRING_DIM, dtype=torch.float32, device=dev)
        known   = string_node_idx >= 0
        unknown = ~known

        if known.any():
            center_emb[known] = self.node_embeddings[string_node_idx[known]]
        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(unknown.sum(), dtype=torch.long, device=dev)
            ).float()
            center_emb[unknown] = fb

        # Neighborhood aggregation (only for known nodes)
        output_emb = center_emb.clone()

        if known.any():
            known_idx = string_node_idx[known]  # [B_known]

            # Get neighbor indices/weights for known nodes
            nbr_idx = self.neighbor_indices[known_idx]   # [B_known, K]
            nbr_wgt = self.neighbor_weights[known_idx]   # [B_known, K]
            nbr_msk = nbr_idx >= 0                       # [B_known, K] valid mask

            # For padding (-1 indices), substitute 0 before lookup to avoid index error
            nbr_idx_safe = nbr_idx.clamp(min=0)
            nbr_emb = self.node_embeddings[nbr_idx_safe]  # [B_known, K, D]

            # Zero out padded neighbor embeddings
            nbr_emb = nbr_emb * nbr_msk.unsqueeze(-1).float()

            aggregated = self.neighborhood_attn(
                center_emb[known], nbr_emb, nbr_wgt, nbr_msk
            )  # [B_known, D]
            output_emb[known] = aggregated

        return output_emb

    def forward(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        pert_emb = self._get_pert_embeddings(string_node_idx)  # [B, STRING_DIM]
        h = self.head(pert_emb)                                 # [B, bilinear_dim]
        # Bilinear interaction: h[b] · gene_class_emb[c,g] = logits[b,c,g]
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)  # [B, 3, G]
        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)   # [B*G, 3]
        targets_flat = targets.reshape(-1)                        # [B*G]

        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )
        return loss

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["string_node_idx"])
        loss = self._loss(logits, batch["labels"])
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

        all_preds = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)    # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)     # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        # De-duplicate
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
        meta = [
            {"sample_idx": int(i.item()), "pert_id": p, "symbol": s}
            for i, p, s in zip(
                batch["sample_idx"], batch["pert_id"], batch["symbol"]
            )
        ]
        self._test_meta.extend(meta)
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds  = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx_t  = torch.tensor(
            [m["sample_idx"] for m in self._test_meta], dtype=torch.long,
            device=local_preds.device,
        )

        all_preds = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_idx   = self.all_gather(local_idx_t)   # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            # Build map: original row index → prediction
            pred_map: Dict[int, torch.Tensor] = {}
            for i in range(len(idx_flat)):
                gid = int(idx_flat[i].item())
                if gid not in pred_map:
                    pred_map[gid] = preds_flat[i]

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            rows = []
            for i in range(len(test_df)):
                if i not in pred_map:
                    continue
                pid  = test_df.iloc[i]["pert_id"]
                sym  = test_df.iloc[i]["symbol"]
                pred = pred_map[i].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            self.print(f"[Node1-2] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_meta.clear()

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
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=hp.lr, weight_decay=hp.weight_decay)

        # Phase 1: linear warmup from 0.1×lr → lr over warmup_epochs epochs
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        # Phase 2: CosineAnnealingWarmRestarts — periodic LR restarts escape local minima
        # T_0=30, T_mult=2 → cycles: 30→60→120→240 epochs
        # Each restart boosts LR back to peak (3e-4), allowing continued exploration.
        # Proven effective in node4-3-2-1-1-1 (F1=0.4921, scFoundation lineage record).
        cawr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=hp.t_zero,
            T_mult=hp.t_mult,
            eta_min=5e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_sched, cawr_sched],
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

    parser = argparse.ArgumentParser(
        description="Node1-2 – Frozen STRING_GNN + K=16 Neighborhood Attention + CosineAnnealingWarmRestarts"
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=32)
    parser.add_argument("--global-batch-size",   type=int,   default=256)
    parser.add_argument("--max-epochs",          type=int,   default=400)
    parser.add_argument("--lr",                  type=float, default=3e-4)
    parser.add_argument("--weight-decay",        type=float, default=2e-2)
    parser.add_argument("--bilinear-dim",        type=int,   default=256)
    parser.add_argument("--dropout",             type=float, default=0.35)
    parser.add_argument("--k",                   type=int,   default=16,
                        dest="k")
    parser.add_argument("--attn-dim",            type=int,   default=64,
                        dest="attn_dim")
    parser.add_argument("--warmup-epochs",       type=int,   default=20)
    parser.add_argument("--t-zero",              type=int,   default=30,
                        dest="t_zero",
                        help="CosineAnnealingWarmRestarts T_0: length of first restart cycle")
    parser.add_argument("--t-mult",              type=int,   default=2,
                        dest="t_mult",
                        help="CosineAnnealingWarmRestarts T_mult: cycle length multiplier")
    parser.add_argument("--label-smoothing",     type=float, default=0.02,
                        dest="label_smoothing")
    parser.add_argument("--val-check-interval",  type=float, default=1.0,
                        dest="val_check_interval",
                        help="How often within one epoch to check validation set (float=fraction, int=batches)")
    parser.add_argument("--num-workers",         type=int,   default=4)
    parser.add_argument("--debug-max-step",      type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",        action="store_true",
                        dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    model = FrozenGNNNeighborhoodModel(
        bilinear_dim    = args.bilinear_dim,
        K               = args.k,
        attn_dim        = args.attn_dim,
        dropout         = args.dropout,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        t_zero          = args.t_zero,
        t_mult          = args.t_mult,
        label_smoothing = args.label_smoothing,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
    )
    # patience=30: allows waiting through restart events (which may temporarily
    # dip val F1) before declaring convergence — critical for CAWR optimization.
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=30, min_delta=1e-4)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Always use DDPStrategy for multi-GPU consistency; works correctly with 1 GPU too
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

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
        val_check_interval      = args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
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

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
