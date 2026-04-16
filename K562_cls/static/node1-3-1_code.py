"""Node 1-3-1: Enhanced STRING_GNN + scFoundation Fusion (larger capacity, no Mixup, discriminative LR).

Strategy: Fix the three primary bottlenecks of node1-3 (F1=0.4669):
  1. Increase fusion_out_dim 256 → 512 to eliminate the aggressive 1024→256 bottleneck
  2. Remove Mixup (alpha=0.0) — blurred perturbation-specific signals in bilinear architecture
  3. Add residual connection from STRING_GNN → fusion output, guaranteeing GNN signal flows even if
     scFoundation provides no value
  4. LayerNorm each modality before fusion to normalize the scale mismatch (256-dim vs 768-dim)
  5. Discriminative LR: fusion module (neighborhood_attn + fusion + fusion_proj) at 3e-4,
     bilinear head (gene_class_emb) at 1e-4 — smaller head capacity gets gentler optimization
  6. Reduce T_max: 150 → 100 to match observed convergence epoch 80 in parent
  7. Increase patience: 10 → 15 for more exploration near the optimum

Memory connections:
- node1-3 (F1=0.4669, parent): scFoundation fusion hurt vs STRING_GNN-only; bottleneck + Mixup identified
- node1-2 (F1=0.4769): frozen STRING_GNN + neighborhood attn K=16 — direct predecessor
- node1-1-1-1-1 (F1=0.4846, best): discriminative LR strategy proven key
- node4-2 (F1=0.4801): scFoundation fusion CAN work when properly configured (6 FT layers + no Mixup)
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
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3
# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

STRING_GNN_DIR    = Path("/home/Models/STRING_GNN")
SCFOUNDATION_DIR  = Path("/home/Models/scFoundation")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights; neutral class stays ≈ 1."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0,1,2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)          # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)          # [N, G]
        is_pred = (y_hat == c)            # [N, G]
        present = is_true.any(dim=0)      # [G]

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
# Pre-computation utilities
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_string_gnn_embeddings() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load STRING_GNN and compute all node embeddings. Returns (emb[N,256], pert_id→idx)."""
    import json as _json
    from transformers import AutoModel as _AM

    model = _AM.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
    model.eval()
    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
    node_names = _json.loads((STRING_GNN_DIR / "node_names.json").read_text())

    edge_index = graph["edge_index"]
    ew = graph.get("edge_weight", None)

    outputs = model(edge_index=edge_index, edge_weight=ew)
    emb = outputs.last_hidden_state.float().cpu()   # [18870, 256]

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
        neighbor_indices [N, K] long — STRING_GNN node indices of top-K neighbors (-1 = padding)
        neighbor_weights [N, K] float — normalized STRING confidence weights
    """
    N = emb.shape[0]
    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
    edge_index = graph["edge_index"]   # [2, E]
    ew = graph.get("edge_weight", None)

    # Build adjacency list per node
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

    # Normalize weights per node (softmax over valid neighbors)
    mask = neighbor_indices >= 0    # [N, K]
    raw = neighbor_weights.clone()
    raw[~mask] = -1e9
    norm_w = torch.softmax(raw, dim=-1)   # [N, K]
    norm_w[~mask] = 0.0

    return neighbor_indices, norm_w


@torch.no_grad()
def precompute_scfoundation_embeddings(
    pert_ids: List[str],
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load scFoundation and compute embeddings for a list of pert_ids.

    Each perturbation is encoded as a single-gene expression profile:
    {gene_ids: [pert_id], expression: [1.0]}

    This gives nnz=1 → sequence length = 3 → output [B, 3, 768].
    We mean-pool all tokens for a compact embedding.

    Returns:
        emb_matrix [M, 768] float — embeddings for each unique pert_id
        pert_id_to_idx dict — maps pert_id → row in emb_matrix
    """
    from transformers import AutoModel as _AM, AutoTokenizer as _AT

    model = _AM.from_pretrained(str(SCFOUNDATION_DIR), trust_remote_code=True)
    tokenizer = _AT.from_pretrained(str(SCFOUNDATION_DIR), trust_remote_code=True)
    model.eval()

    unique_ids = sorted(set(pert_ids))
    pert_id_to_idx = {pid: i for i, pid in enumerate(unique_ids)}

    batch_size = 64
    embeddings = []

    for start in range(0, len(unique_ids), batch_size):
        batch_pids = unique_ids[start:start + batch_size]
        # Encode each perturbation as single-gene expression
        expr_dicts = [
            {"gene_ids": [pid], "expression": [1.0]}
            for pid in batch_pids
        ]
        tokenized = tokenizer(expr_dicts, return_tensors="pt")
        # input_ids: [B, 19264] float32
        input_ids = tokenized["input_ids"]

        outputs = model(input_ids=input_ids, output_hidden_states=False)
        # last_hidden_state: [B, nnz+2, 768] = [B, 3, 768] for nnz=1
        h = outputs.last_hidden_state.float()  # [B, 3, 768]
        # Mean pool all tokens for a compact representation
        pooled = h.mean(dim=1)  # [B, 768]
        embeddings.append(pooled.cpu())

    emb_matrix = torch.cat(embeddings, dim=0)  # [M, 768]
    del model, tokenizer
    return emb_matrix, pert_id_to_idx


# ---------------------------------------------------------------------------
# Neighborhood Attention Aggregator (identical to node1-2 proven design)
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Aggregate top-K PPI neighbors for a center gene using learned attention."""

    def __init__(self, emb_dim: int = 256, attn_dim: int = 64) -> None:
        super().__init__()
        # Attention score: concat(center, neighbor) → scalar score
        self.attn_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False),
        )
        # Gate: how much neighbor context to add to center
        self.gate_proj = nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        center_emb: torch.Tensor,       # [B, D]
        neighbor_emb: torch.Tensor,     # [B, K, D]
        neighbor_weights: torch.Tensor, # [B, K]  pre-normalized edge weights
        valid_mask: torch.Tensor,       # [B, K]  bool, True = valid neighbor
    ) -> torch.Tensor:
        """Returns aggregated representation [B, D]."""
        B, K, D = neighbor_emb.shape
        center_exp = center_emb.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]
        pair = torch.cat([center_exp, neighbor_emb], dim=-1)     # [B, K, 2D]
        attn_scores = self.attn_proj(pair).squeeze(-1)           # [B, K]

        # Combine learned scores with STRING confidence as prior
        attn_scores = attn_scores + neighbor_weights

        # Mask invalid neighbors
        attn_scores = attn_scores.masked_fill(~valid_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)        # [B, K]
        attn_weights = attn_weights * valid_mask.float()         # zero-out invalid

        # Weighted aggregation
        aggregated = (attn_weights.unsqueeze(-1) * neighbor_emb).sum(dim=1)  # [B, D]

        # Gated residual: center + gate * aggregated
        gate = torch.sigmoid(self.gate_proj(center_emb))         # [B, D]
        return center_emb + gate * aggregated                     # [B, D]


# ---------------------------------------------------------------------------
# Enhanced Gated Fusion Module (KEY IMPROVEMENT over node1-3)
# ---------------------------------------------------------------------------
class EnhancedGatedFusion(nn.Module):
    """Enhanced fusion: STRING_GNN [256] + scFoundation [768] → [out_dim=512].

    Improvements over node1-3's GatedFusion:
    1. LayerNorm on EACH modality before fusion (normalizes scale mismatch 256 vs 768)
    2. Larger out_dim=512 (up from 256) eliminates the 1024→256 bottleneck
    3. Residual connection from GNN embedding → output (guarantees GNN signal preserved)
       If scFoundation provides no useful signal, the gate closes and output ≈ residual_proj(gnn)
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        scf_dim: int = 768,
        out_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Per-modality normalization before fusion
        self.gnn_norm = nn.LayerNorm(gnn_dim)
        self.scf_norm = nn.LayerNorm(scf_dim)

        in_dim = gnn_dim + scf_dim  # 1024
        self.proj = nn.Linear(in_dim, out_dim * 2)  # gate half + value half
        self.gate_norm = nn.LayerNorm(out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

        # Residual: project GNN directly to out_dim
        # If scFoundation is uninformative, output ≈ residual_proj(gnn_norm(gnn_emb))
        self.residual_proj = nn.Linear(gnn_dim, out_dim)
        self.residual_norm = nn.LayerNorm(out_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, gnn_emb: torch.Tensor, scf_emb: torch.Tensor) -> torch.Tensor:
        """Args: gnn_emb [B, 256], scf_emb [B, 768]. Returns [B, out_dim]."""
        gnn_n = self.gnn_norm(gnn_emb)    # [B, 256] — normalize scale
        scf_n = self.scf_norm(scf_emb)    # [B, 768] — normalize scale

        x = torch.cat([gnn_n, scf_n], dim=-1)  # [B, 1024]
        proj_out = self.proj(x)                  # [B, out_dim*2]
        half = proj_out.shape[-1] // 2
        gate_logit, value = proj_out[:, :half], proj_out[:, half:]

        gate = torch.sigmoid(self.gate_norm(gate_logit))   # [B, out_dim]
        fused = gate * self.out_norm(value)                # [B, out_dim]

        # Residual ensures STRING_GNN signal always flows regardless of scFoundation gate
        residual = self.residual_norm(self.residual_proj(gnn_n))  # [B, out_dim]

        out = fused + residual  # [B, out_dim]
        return self.dropout(out)


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
            item["labels"] = self.labels[idx]   # [G] in {0,1,2}
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
class FusionDEGModelV2(pl.LightningModule):
    """Frozen STRING_GNN (neighborhood attention) + Frozen scFoundation → EnhancedGatedFusion → Bilinear head.

    Key improvements over node1-3 (F1=0.4669):
    - EnhancedGatedFusion: LayerNorm per modality + residual from GNN + larger out_dim=512
    - No Mixup: bilinear architecture needs clean perturbation signals
    - Discriminative LR: fusion module at 3e-4, bilinear head at 1e-4
    - T_max=100 matches observed convergence (~epoch 80 in node1-3)
    - patience=15 allows more exploration near optimum
    """

    def __init__(
        self,
        bilinear_dim: int = 256,
        attn_dim: int = 64,
        K: int = 16,
        fusion_out_dim: int = 512,
        fusion_dropout: float = 0.3,
        dropout: float = 0.35,
        lr: float = 3e-4,
        head_lr: float = 1e-4,
        weight_decay: float = 3e-2,
        warmup_epochs: int = 20,
        t_max: int = 100,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # ---- Collect all pert_ids across splits ----
        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        all_pert_ids = (
            train_df["pert_id"].tolist() +
            val_df["pert_id"].tolist() +
            test_df["pert_id"].tolist()
        )

        # ---- STRING_GNN: precompute embeddings + neighborhood ----
        self.print("Precomputing STRING_GNN embeddings...")
        string_emb, pert_to_gnn_idx = precompute_string_gnn_embeddings()
        # string_emb: [18870, 256], pert_to_gnn_idx: {pert_id → node_idx}

        # Register as buffer so Lightning moves to GPU automatically
        self.register_buffer("node_embeddings", string_emb)    # [18870, 256]

        # Build pert_id → STRING_GNN node index mapping
        unique_sorted = sorted(set(all_pert_ids))
        self.pert_to_pos = {pid: i for i, pid in enumerate(unique_sorted)}

        gnn_idx_tensor = torch.tensor(
            [pert_to_gnn_idx.get(pid, -1) for pid in unique_sorted], dtype=torch.long
        )
        self.register_buffer("pert_gnn_idx", gnn_idx_tensor)   # [M]

        # Precompute neighborhood tables
        self.print("Precomputing PPI neighborhood tables (K={})...".format(hp.K))
        nb_indices, nb_weights = precompute_neighborhood(string_emb, K=hp.K)
        self.register_buffer("neighbor_indices", nb_indices)   # [18870, K]
        self.register_buffer("neighbor_weights", nb_weights)   # [18870, K]

        # Fallback embedding for pert_ids not in STRING
        self.fallback_emb = nn.Parameter(torch.zeros(1, 256))

        # ---- scFoundation: precompute embeddings (with disk cache for DDP) ----
        scf_cache_path = Path(__file__).parent / "run" / "scf_embeddings_cache.pt"
        # Rank 0 creates the dir and computes embeddings if cache missing
        if local_rank == 0:
            scf_cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not scf_cache_path.exists():
                self.print("Precomputing scFoundation embeddings (first time)...")
                scf_emb, pert_to_scf_idx = precompute_scfoundation_embeddings(all_pert_ids)
                torch.save({"emb": scf_emb, "map": pert_to_scf_idx}, str(scf_cache_path))
            else:
                self.print("Loading cached scFoundation embeddings...")

        # All other ranks wait for rank 0 to finish
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        scf_data = torch.load(str(scf_cache_path), map_location="cpu", weights_only=False)
        scf_emb = scf_data["emb"]            # [M_unique, 768]
        pert_to_scf_idx = scf_data["map"]    # {pert_id → row_in_scf_emb}

        # Replace NaN embeddings with mean of valid embeddings
        nan_mask = ~torch.isfinite(scf_emb).all(dim=1)
        if nan_mask.any():
            finite_mean = scf_emb[~nan_mask].mean(dim=0)
            scf_emb[nan_mask] = finite_mean
            self.print(f"[Setup] Replaced {nan_mask.sum().item()} NaN scFoundation embeddings with mean.")

        # Build lookup tensor aligned with unique_sorted ordering
        scf_lookup = torch.zeros(len(unique_sorted), 768, dtype=torch.float32)
        for pid, pos in self.pert_to_pos.items():
            scf_row = pert_to_scf_idx.get(pid, None)
            if scf_row is not None:
                scf_lookup[pos] = scf_emb[scf_row]

        # Final check: replace any remaining NaN in lookup with zeros
        if not torch.isfinite(scf_lookup).all():
            nan_in_lookup = ~torch.isfinite(scf_lookup)
            scf_lookup[nan_in_lookup] = 0.0
            self.print(f"[Setup] Replaced {nan_in_lookup.any(dim=1).sum().item()} NaN rows in scf_lookup with zeros.")

        self.register_buffer("scf_embeddings", scf_lookup)    # [M, 768]

        # ---- Trainable modules ----
        self.neighborhood_attn = NeighborhoodAttentionAggregator(
            emb_dim=256, attn_dim=hp.attn_dim
        )
        # IMPROVED: EnhancedGatedFusion with residual + per-modality LN + larger out_dim
        self.fusion = EnhancedGatedFusion(
            gnn_dim=256,
            scf_dim=768,
            out_dim=hp.fusion_out_dim,    # 512 (was 256 in node1-3)
            dropout=hp.fusion_dropout,
        )

        # Projection from fusion output (512) → bilinear_dim (256)
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(hp.fusion_out_dim),
            nn.Linear(hp.fusion_out_dim, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # Bilinear gene-class embedding: logits[b,c,g] = h[b] · gene_class_emb[c,g]
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        self.register_buffer("class_weights", get_class_weights())

        # Cast trainable parameters to float32 for stable optimization
        for k, v in self.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    def _get_gnn_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Return [B, 256] STRING_GNN embeddings for a batch of pert_ids."""
        pos = torch.tensor(
            [self.pert_to_pos[pid] for pid in pert_ids], dtype=torch.long, device=self.device
        )
        gnn_node_idx = self.pert_gnn_idx[pos]   # [B] — STRING_GNN node indices
        valid = gnn_node_idx >= 0               # [B] bool
        safe_idx = gnn_node_idx.clamp(min=0)
        emb = self.node_embeddings[safe_idx]    # [B, 256]
        fallback = self.fallback_emb.expand(emb.shape[0], -1).to(emb.dtype)
        emb = torch.where(valid.unsqueeze(-1), emb, fallback)
        return emb.float()  # [B, 256]

    def _get_neighborhood_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Return [B, 256] neighborhood-attention-aggregated embeddings."""
        pos = torch.tensor(
            [self.pert_to_pos[pid] for pid in pert_ids], dtype=torch.long, device=self.device
        )
        gnn_node_idx = self.pert_gnn_idx[pos]   # [B]
        valid_center = gnn_node_idx >= 0
        safe_center_idx = gnn_node_idx.clamp(min=0)
        center_emb_raw = self.node_embeddings[safe_center_idx]  # [B, 256]
        # Apply fallback for missing centers using torch.where
        fallback = self.fallback_emb.expand(center_emb_raw.shape[0], -1).to(center_emb_raw.dtype)
        center_emb = torch.where(valid_center.unsqueeze(-1), center_emb_raw, fallback).float()

        K = self.hparams.K

        nb_idx = self.neighbor_indices[safe_center_idx]   # [B, K]
        nb_wts = self.neighbor_weights[safe_center_idx]   # [B, K]
        valid_mask = nb_idx >= 0                           # [B, K] bool

        # Safe lookup: clamp -1 → 0, zero out invalid after lookup
        safe_nb_idx = nb_idx.clamp(min=0)                 # [B, K]
        nb_emb = self.node_embeddings[safe_nb_idx].float()  # [B, K, 256]
        nb_emb = nb_emb * valid_mask.unsqueeze(-1).float()

        aggregated = self.neighborhood_attn(
            center_emb, nb_emb, nb_wts, valid_mask
        )   # [B, 256]
        return aggregated

    def _get_scf_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Return [B, 768] scFoundation embeddings."""
        pos = torch.tensor(
            [self.pert_to_pos[pid] for pid in pert_ids], dtype=torch.long, device=self.device
        )
        return self.scf_embeddings[pos]   # [B, 768]

    # ---- Forward ----
    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        gnn_emb = self._get_neighborhood_emb(pert_ids)    # [B, 256] float32
        scf_emb = self._get_scf_emb(pert_ids).float()     # [B, 768] float32

        # EnhancedGatedFusion: per-modality LN + residual + larger capacity
        fused = self.fusion(gnn_emb, scf_emb)              # [B, fusion_out_dim=512]
        h = self.fusion_proj(fused)                        # [B, bilinear_dim=256]
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)   # [B, 3, G]
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

    # ---- Steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        # No Mixup — bilinear architecture relies on clean perturbation signals
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

        all_preds = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)    # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)     # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        order  = torch.argsort(idx_flat)
        s_idx  = idx_flat[order]
        s_pred = preds_flat[order]
        s_tgt  = tgts_flat[order]
        mask   = torch.cat([torch.ones(1, dtype=torch.bool, device=s_idx.device),
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
        local_preds = torch.cat(self._test_preds, dim=0)    # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)    # [N_local]
        all_preds   = self.all_gather(local_preds)           # [W, N_local, 3, G]
        all_idx     = self.all_gather(local_idx)             # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            order = torch.argsort(idx_flat)
            s_idx = idx_flat[order]
            s_pred = preds_flat[order]
            mask = torch.cat([torch.ones(1, dtype=torch.bool, device=s_idx.device),
                              s_idx[1:] != s_idx[:-1]])
            preds_dedup = s_pred[mask]     # [N_test, 3, G]
            unique_sid  = s_idx[mask].tolist()

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                           for i in range(len(test_df))}

            rows = []
            for sid in unique_sid:
                pid, sym = idx_to_meta[int(sid)]
                dedup_pos = (s_idx == sid).nonzero(as_tuple=True)[0][0].item()
                pred_list = preds_dedup[dedup_pos].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-3-1] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_idx.clear()

    # ---- Checkpoint helpers ----
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
        bufs  = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Checkpoint: {train}/{total} params ({100*train/total:.2f}%), "
            f"plus {bufs} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer (DISCRIMINATIVE LR) ----
    def configure_optimizers(self):
        hp = self.hparams

        # Identify parameter groups by name
        # Group 1 (fusion module): neighborhood_attn, fusion, fusion_proj — higher lr
        # Group 2 (bilinear head): gene_class_emb, fallback_emb — lower lr
        fusion_params = []
        head_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(name.startswith(prefix) for prefix in
                   ["neighborhood_attn.", "fusion.", "fusion_proj."]):
                fusion_params.append(p)
            else:
                # gene_class_emb, fallback_emb
                head_params.append(p)

        param_groups = [
            {"params": fusion_params, "lr": hp.lr,      "name": "fusion"},
            {"params": head_params,   "lr": hp.head_lr, "name": "head"},
        ]

        opt = torch.optim.AdamW(param_groups, weight_decay=hp.weight_decay)

        # Linear warmup for warmup_epochs, then CosineAnnealingLR for t_max epochs
        warmup_sch = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=hp.t_max,
            eta_min=5e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_sch, cosine_sch],
            milestones=[hp.warmup_epochs],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-3-1: Enhanced STRING_GNN+scFoundation Fusion (larger capacity, no Mixup, discriminative LR)"
    )
    parser.add_argument("--micro-batch-size",  type=int,   default=32)
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument("--max-epochs",        type=int,   default=220)
    parser.add_argument("--lr",                type=float, default=3e-4,
                        help="LR for fusion module (neighborhood_attn, fusion, fusion_proj)")
    parser.add_argument("--head-lr",           type=float, default=1e-4,
                        help="LR for bilinear head (gene_class_emb); lower than fusion LR")
    parser.add_argument("--weight-decay",      type=float, default=3e-2)
    parser.add_argument("--bilinear-dim",      type=int,   default=256)
    parser.add_argument("--attn-dim",          type=int,   default=64)
    parser.add_argument("--k-neighbors",       type=int,   default=16)
    parser.add_argument("--fusion-out-dim",    type=int,   default=512,
                        help="Output dim of EnhancedGatedFusion; 512 (was 256 in node1-3)")
    parser.add_argument("--fusion-dropout",    type=float, default=0.3)
    parser.add_argument("--dropout",           type=float, default=0.35)
    parser.add_argument("--warmup-epochs",     type=int,   default=20)
    parser.add_argument("--t-max",             type=int,   default=100,
                        help="Cosine T_max; 100 matches observed convergence epoch ~80 in parent")
    parser.add_argument("--label-smoothing",   type=float, default=0.05)
    parser.add_argument("--patience",          type=int,   default=15,
                        help="EarlyStopping patience; 15 to allow more exploration near optimum")
    parser.add_argument("--val-check-interval", type=float, default=1.0,
                        help="Run validation every N epochs (fraction supported); default 1.0")
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

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        lim_test  = 1.0   # Process all test data even in debug mode (test set is small: 154 samples)
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        lim_test  = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule + Model
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    model = FusionDEGModelV2(
        bilinear_dim     = args.bilinear_dim,
        attn_dim         = args.attn_dim,
        K                = args.k_neighbors,
        fusion_out_dim   = args.fusion_out_dim,
        fusion_dropout   = args.fusion_dropout,
        dropout          = args.dropout,
        lr               = args.lr,
        head_lr          = args.head_lr,
        weight_decay     = args.weight_decay,
        warmup_epochs    = args.warmup_epochs,
        t_max            = args.t_max,
        label_smoothing  = args.label_smoothing,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    early_stop_callback = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.patience,
        min_delta=1e-4,
        verbose=True,
    )
    lr_monitor   = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=10)

    csv_logger         = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tensorboard_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        limit_train_batches=lim_train,
        limit_val_batches=lim_val,
        limit_test_batches=lim_test,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, progress_bar],
        logger=[csv_logger, tensorboard_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=dm)
    else:
        test_results = trainer.test(model, datamodule=dm, ckpt_path="best")

    # Save test score
    if test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        with open(score_path, "w") as f:
            f.write(str(test_results))
        print(f"[Node1-3-1] Test results: {test_results}")


if __name__ == "__main__":
    main()
