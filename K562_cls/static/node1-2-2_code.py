"""Node 1-2-2 – Frozen STRING_GNN + Frozen scFoundation Dual-Fusion + PPI Neighborhood Attention.

This node implements the critical recommendation from node1-2-1's feedback:
"Fusion architecture is mandatory to break the STRING_GNN-only ceiling (~0.485).
 Combine frozen STRING_GNN embeddings with frozen scFoundation embeddings via
 concatenation or gated fusion."

The entire node1-2 lineage (node1-2: F1=0.4769, node1-2-1: F1=0.4748) confirms that
STRING_GNN-only approaches cap at ~0.485 regardless of training strategy (frozen, DLR).
The missing ingredient is perturbation-specific transcriptional context that scFoundation
provides through its 50M+ human single-cell training profiles.

Architecture:
─────────────────────────────────────────────────────────────────────────────
  Input: pert_id (Ensembl gene ID), string_node_idx (int)
         │                             │
  ┌──────┴──────┐            ┌────────┴────────┐
  │ scFoundation│            │   STRING_GNN     │
  │  (frozen)   │            │   (frozen)       │
  │             │            │                  │
  │ Single-gene │            │ Pre-computed     │
  │ encoding:   │            │ node embeddings  │
  │ pert_id →   │            │ + PPI Neighbor-  │
  │ input 1.0   │            │ hood Attention   │
  │ → pool →    │            │ (K=16, d=64)     │
  │ [768]       │            │ → [256]          │
  └──────┬──────┘            └────────┬─────────┘
         │                            │
         └──────────┬─────────────────┘
                    │ concat → [1024]
                    │
              LayerNorm(1024)
              Linear(1024 → 512)
              GELU
              Dropout(0.35)
              Linear(512 → 256)
              GELU
              Dropout(0.35)
                    │
                  h [256]
                    │
           Bilinear: logits[b,c,g] = h[b] · gene_class_emb[c,g]
                    │
             logits [B, 3, 6640]
                    │
        Weighted CE + label smoothing ε=0.05
─────────────────────────────────────────────────────────────────────────────

Distinct from parent (node1-2):
  - Adds frozen scFoundation single-gene perturbation embeddings [768]
  - 2-layer fusion head instead of flat 1-layer head

Distinct from sibling (node1-2-1):
  - Uses DUAL-SOURCE embeddings (STRING + scFoundation) vs. single STRING source
  - Frozen backbone (no DLR) — proven superior on this small dataset

Memory connections:
  - node1-2 (parent, F1=0.4769): proven frozen STRING+attn K=16 config → retained
  - node1-2-1 (sibling, F1=0.4748): DLR failed → frozen backbone confirmed
  - node1-2-1 feedback: "Fusion with scFoundation is mandatory to break ceiling" → implemented
  - node1-1-1-1-1 (best, F1=0.4846): K=16, attn_dim=64, lr=3e-4, wd=3e-2 → retained
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
N_GENES   = 6640
N_CLASSES = 3

# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

# STRING_GNN: try env-var first, then absolute path, then project-relative fallback
STRING_GNN_DIR = Path(os.environ.get(
    "STRING_GNN_DIR",
    os.environ.get("STRING_GNN_PATH", "/home/Models/STRING_GNN")
))
if not STRING_GNN_DIR.exists():
    STRING_GNN_DIR = (
        Path("/home/Models/STRING_GNN")
    )

# scFoundation: absolute path as documented in scfoundation-model-skill
SCF_MODEL_DIR = Path(os.environ.get("SCF_MODEL_DIR", "/home/Models/scFoundation"))

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

STRING_DIM = 256   # STRING_GNN output dimension
SCF_DIM    = 768   # scFoundation pooled cell embedding dimension


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

    Uses sort-based approach: sort edges by (src, -weight), then slice top-K.

    Returns:
        neighbor_indices: [n_nodes, K] long — top-K neighbor node indices
                          (-1 for padding if fewer than K neighbors)
        neighbor_weights: [n_nodes, K] float — STRING confidence scores
    """
    src = edge_index[0]
    dst = edge_index[1]
    wgt = edge_weight

    # Sort by weight descending, then stable-sort by src ascending
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
        preds:   [N, 3, G] float – softmax probabilities
        targets: [N, G]    long  – class labels in {0, 1, 2}
    Returns:
        Scalar float: mean F1 over all G genes.
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
            item["labels"] = self.labels[idx]
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
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Neighborhood Attention Module (proven design from node1-1-1-1-1)
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Center-context gated attention over top-K PPI neighbors.

    Architecture:
        attn_proj: [center(256) + neighbor(256)] → attn_dim(64) → score(1)
        attention = softmax(edge_weight + attn_proj_score)  [B, K]
        aggregated = attention @ neighbor_emb               [B, 256]
        gate = sigmoid(gate_proj(center_emb))               [B, 256]
        output = center_emb + gate * aggregated             [B, 256]
    """

    def __init__(self, embed_dim: int = 256, attn_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim  = attn_dim

        self.attn_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, 1),
        )
        self.gate_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        center_emb:       torch.Tensor,  # [B, D]
        neighbor_emb:     torch.Tensor,  # [B, K, D]
        neighbor_weights: torch.Tensor,  # [B, K]
        neighbor_mask:    torch.Tensor,  # [B, K] bool
    ) -> torch.Tensor:
        B, K, D = neighbor_emb.shape

        center_expanded = center_emb.unsqueeze(1).expand(-1, K, -1)       # [B, K, D]
        pair_features   = torch.cat([center_expanded, neighbor_emb], dim=-1)  # [B, K, 2D]

        attn_scores = self.attn_proj(pair_features).squeeze(-1)  # [B, K]
        attn_scores = attn_scores + neighbor_weights              # [B, K]
        attn_scores = attn_scores.masked_fill(~neighbor_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)         # [B, K]
        attn_weights = self.attn_dropout(attn_weights)

        aggregated = torch.bmm(attn_weights.unsqueeze(1), neighbor_emb).squeeze(1)  # [B, D]
        gate   = torch.sigmoid(self.gate_proj(center_emb))   # [B, D]
        output = center_emb + gate * aggregated               # [B, D]
        return output


# ---------------------------------------------------------------------------
# Fusion Model: Frozen STRING_GNN + Frozen scFoundation + Bilinear Head
# ---------------------------------------------------------------------------
class FusionSCFStringModel(pl.LightningModule):
    """Frozen STRING_GNN + Frozen scFoundation Fusion + Bilinear DEG Prediction Head.

    Architecture:
    1. STRING_GNN runs ONCE at setup() to pre-compute node embeddings [18870, 256]
       → PPI neighborhood attention (K=16, attn_dim=64) → [B, 256]
    2. scFoundation runs ONCE at setup() for all unique pert_ids
       → Single-gene perturbation encoding (expression=1.0 for pert gene) → pool → [B, 768]
    3. Fusion head:
       concat([STRING_emb, SCF_emb]) → LN(1024) → 512 → GELU → Dropout → 256 → GELU → Dropout
    4. Bilinear: logits[b,c,g] = h[b] · gene_class_emb[c,g]  [B, 3, G]
    5. Weighted CE + label smoothing ε=0.05
    """

    def __init__(
        self,
        bilinear_dim:    int   = 256,
        K:               int   = 16,
        attn_dim:        int   = 64,
        fusion_dim:      int   = 512,
        dropout:         float = 0.35,
        lr:              float = 3e-4,
        weight_decay:    float = 3e-2,
        warmup_epochs:   int   = 20,
        t_max:           int   = 150,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Register gene_class_emb in __init__ so it's included in the optimizer
        # (optimizer is configured before setup() runs).
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, bilinear_dim) * 0.02
        )

    # ------------------------------------------------------------------
    # Setup (called once per rank before training)
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        # Sequential DDP setup: rank 0 computes pre-computed buffers first,
        # then all ranks wait and load identical buffers.
        if getattr(self, "_setup_done", False):
            return
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        use_barrier = (
            world_size > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

        hp = self.hparams

        # ----------------------------------------------------------------
        # 1. Pre-compute STRING_GNN node embeddings (frozen backbone)
        # ----------------------------------------------------------------
        print("[Setup] Loading STRING_GNN backbone for one-time embedding pre-computation...")
        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        graph       = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph["edge_index"].long()
        edge_weight = graph["edge_weight"].float()

        with torch.no_grad():
            gnn_out  = backbone(edge_index=edge_index, edge_weight=edge_weight)
            node_emb = gnn_out.last_hidden_state.float().detach()  # [18870, 256]

        self.register_buffer("node_embeddings", node_emb)
        n_nodes = node_emb.shape[0]

        print(f"[Setup] Pre-computing top-{hp.K} PPI neighbors for {n_nodes} nodes...")
        nbr_idx, nbr_wgt = precompute_neighborhood(edge_index, edge_weight, n_nodes, K=hp.K)
        self.register_buffer("neighbor_indices", nbr_idx)
        self.register_buffer("neighbor_weights", nbr_wgt)
        del backbone, graph, edge_index, edge_weight, gnn_out
        print("[Setup] STRING_GNN done.")

        # ----------------------------------------------------------------
        # 2. Pre-compute scFoundation perturbation embeddings (frozen)
        # ----------------------------------------------------------------
        self._precompute_scf_embeddings()

        # ----------------------------------------------------------------
        # 3. Fallback embeddings for genes not in STRING or scFoundation
        # ----------------------------------------------------------------
        self.fallback_emb      = nn.Embedding(1, STRING_DIM)
        self.scf_fallback_emb  = nn.Embedding(1, SCF_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)
        nn.init.normal_(self.scf_fallback_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 4. Neighborhood Attention (K=16, attn_dim=64 — proven best)
        # ----------------------------------------------------------------
        self.neighborhood_attn = NeighborhoodAttentionAggregator(
            embed_dim=STRING_DIM,
            attn_dim=hp.attn_dim,
            dropout=0.0,
        )

        # ----------------------------------------------------------------
        # 5. Fusion head: [STRING_DIM + SCF_DIM] → bilinear_dim
        #    2-layer MLP to combine two different embedding spaces
        # ----------------------------------------------------------------
        in_dim = STRING_DIM + SCF_DIM  # 256 + 768 = 1024
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hp.fusion_dim),  # 1024 → 512
            nn.GELU(),
            nn.Dropout(hp.dropout),
            nn.Linear(hp.fusion_dim, hp.bilinear_dim),  # 512 → 256
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # ----------------------------------------------------------------
        # 6. Bilinear gene-class embedding  [3, 6640, 256] — registered in __init__
        # ----------------------------------------------------------------

        # Class weights for weighted CE
        self.register_buffer("class_weights", get_class_weights())

        # Barrier: ensure all ranks sync after setup so STRING_GNN model
        # on each rank is freed before training starts (avoids GPU OOM).
        if use_barrier:
            torch.distributed.barrier()
        self._setup_done = True

        # Cast all trainable parameters to float32 for stable optimization
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Setup] Parameters: {total_params} total, {trainable_params} trainable")
        for name, p in self.named_parameters():
            if p.numel() > 1000:
                print(f"  {name}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}, dtype={p.dtype}")
        for _, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators for val/test
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []
        self._test_pids:  List[str]          = []

    # ------------------------------------------------------------------
    # scFoundation pre-computation
    # ------------------------------------------------------------------
    def _precompute_scf_embeddings(self) -> None:
        """Pre-compute scFoundation embeddings for all unique perturbation genes.

        Strategy:
        - For each unique pert_id (Ensembl gene ID), tokenize it as a single-gene
          expression input (expression=1.0) and run through frozen scFoundation.
        - Pool the dynamic-length output (mean over nnz+2 positions) → [768]
        - Store as a fixed buffer: scf_embeddings [N_unique_perts, 768]
        - Also store as a non-buffer dict: scf_pert_to_idx for lookup in forward()
        """
        print("[Setup] Pre-computing scFoundation embeddings for all perturbation genes...")

        # Determine compute device
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        # Load scFoundation
        scf_model     = AutoModel.from_pretrained(str(SCF_MODEL_DIR), trust_remote_code=True)
        scf_tokenizer = AutoTokenizer.from_pretrained(str(SCF_MODEL_DIR), trust_remote_code=True)
        scf_model.eval().to(device)
        for p in scf_model.parameters():
            p.requires_grad = False

        # Collect all unique pert_ids and symbols from all data splits
        all_data: Dict[str, str] = {}  # pert_id → symbol
        for tsv in [TRAIN_TSV, VAL_TSV, TEST_TSV]:
            df = pd.read_csv(tsv, sep="\t")
            for pid, sym in zip(df["pert_id"].tolist(), df["symbol"].tolist()):
                if pid not in all_data:
                    all_data[pid] = sym

        all_perts_ordered = list(all_data.keys())
        all_syms_ordered  = [all_data[pid] for pid in all_perts_ordered]
        scf_pert_to_idx   = {pid: i for i, pid in enumerate(all_perts_ordered)}
        N_perts = len(all_perts_ordered)

        print(f"[Setup] Tokenizing {N_perts} unique pert_ids for scFoundation...")

        # Tokenize each pert_id: try Ensembl ID → symbol → zeros fallback
        # IMPORTANT: scFoundation produces NaN on all-zero inputs (confirmed bug with
        # softmax-based expression discretization). We must check tokenization BEFORE
        # running scFoundation and use zero vectors as DIRECT embeddings (not model inputs).
        input_ids_list: List[torch.Tensor] = []  # each [1, 19264]
        valid_mask_list: List[bool] = []  # True if scFoundation forward should be run
        n_found_by_ensg  = 0
        n_found_by_sym   = 0
        n_not_found      = 0

        for pid, sym in zip(all_perts_ordered, all_syms_ordered):
            # Attempt 1: Ensembl gene ID
            inp = None
            try:
                inp_candidate = scf_tokenizer(
                    {"gene_ids": [pid], "expression": [1.0]},
                    return_tensors="pt",
                )
                if (inp_candidate["input_ids"] > 0).any():
                    inp = inp_candidate
                    n_found_by_ensg += 1
            except Exception:
                pass

            # Attempt 2: Gene symbol (flat dict format: {symbol: expression})
            if inp is None:
                try:
                    inp_candidate = scf_tokenizer({sym: 1.0}, return_tensors="pt")
                    if (inp_candidate["input_ids"] > 0).any():
                        inp = inp_candidate
                        n_found_by_sym += 1
                except Exception:
                    pass

            # Attempt 3: Gene name format
            if inp is None:
                try:
                    inp_candidate = scf_tokenizer(
                        {"gene_names": [sym], "expression": [1.0]},
                        return_tensors="pt",
                    )
                    if (inp_candidate["input_ids"] > 0).any():
                        inp = inp_candidate
                        n_found_by_sym += 1
                except Exception:
                    pass

            if inp is None:
                # Completely unknown: NO scFoundation embedding available
                # Store zero tensor as placeholder; will use fallback_emb in forward()
                inp = {"input_ids": torch.zeros(1, 19264, dtype=torch.float32)}
                n_not_found += 1
                valid_mask_list.append(False)
            else:
                valid_mask_list.append(True)

            input_ids_list.append(inp["input_ids"])  # [1, 19264]

        print(
            f"[Setup] scFoundation vocab coverage: "
            f"{n_found_by_ensg} by ENSG, {n_found_by_sym} by symbol, "
            f"{n_not_found} not found ({100*n_not_found/N_perts:.1f}%)"
        )

        # Stack all inputs: [N, 19264]
        all_input_ids = torch.cat(input_ids_list, dim=0)  # [N, 19264]
        valid_mask = torch.tensor(valid_mask_list, dtype=torch.bool)  # [N]

        # Initialize all embeddings to zero; only fill valid positions
        scf_emb_tensor = torch.zeros(N_perts, 768, dtype=torch.float32)

        # Run scFoundation in batches → pool → [N, 768]
        BATCH_SIZE = 64
        valid_indices = valid_mask.nonzero(as_tuple=True)[0].tolist()
        print(f"[Setup] Running scFoundation forward for {len(valid_indices)}/{N_perts} valid pert_ids (batch={BATCH_SIZE})...")
        with torch.no_grad():
            for i in range(0, len(valid_indices), BATCH_SIZE):
                batch_local_idx = valid_indices[i:i + BATCH_SIZE]
                batch_ids = all_input_ids[batch_local_idx].to(device)  # [B, 19264]
                out = scf_model(input_ids=batch_ids)
                cell_emb = out.last_hidden_state.mean(dim=1).float().cpu()  # [B, 768]
                if torch.isnan(cell_emb).any():
                    nan_count = torch.isnan(cell_emb).sum().item()
                    print(f"[Setup WARNING] NaN in batch {i//BATCH_SIZE}! count={nan_count}, "
                          f"batch_ids sample={batch_ids[0, :8].tolist()}")
                # Write directly to pre-allocated tensor
                scf_emb_tensor[batch_local_idx] = cell_emb

        n_nan = torch.isnan(scf_emb_tensor).sum().item()
        n_valid_nan = torch.isnan(scf_emb_tensor[valid_mask]).sum().item()
        print(f"[Setup] scf_emb_tensor shape: {scf_emb_tensor.shape}")
        if n_valid_nan > 0:
            print(f"[Setup WARNING] NaN in VALID positions: {n_valid_nan}/{valid_mask.sum().item()*768}")
        if n_nan > 0:
            print(f"[Setup WARNING] Total NaN in scf_emb_tensor (includes not-found): {n_nan}/{N_perts*768}")
            print(f"  Filling NaN positions with zeros.")
            scf_emb_tensor = torch.nan_to_num(scf_emb_tensor, nan=0.0)
        else:
            print(f"[Setup] scf_emb_tensor stats: mean={scf_emb_tensor.mean():.4f}, "
                  f"std={scf_emb_tensor.std():.4f}, min={scf_emb_tensor.min():.4f}, max={scf_emb_tensor.max():.4f}")

        # Register as buffer (PL moves to correct device automatically)
        self.register_buffer("scf_embeddings", scf_emb_tensor)

        # Store lookup dict as non-buffer attribute (plain Python dict, not saved to ckpt)
        self.scf_pert_to_idx: Dict[str, int] = scf_pert_to_idx

        # Cleanup scFoundation to free GPU memory
        del scf_model, scf_tokenizer, all_input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Setup] scFoundation done and memory freed.")

    # ------------------------------------------------------------------
    # Embedding lookups
    # ------------------------------------------------------------------
    def _get_string_embedding(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """STRING_GNN neighborhood-aggregated embedding lookup.

        For known pert_ids: apply PPI neighborhood attention.
        For unknown pert_ids (-1): use learnable fallback embedding.

        Returns [B, STRING_DIM] float32.
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
            known_idx = string_node_idx[known]  # [K_known]

            center   = self.node_embeddings[known_idx]         # [K_known, 256]
            nbr_idx  = self.neighbor_indices[known_idx]        # [K_known, K]
            nbr_wgt  = self.neighbor_weights[known_idx]        # [K_known, K]
            nbr_mask = nbr_idx >= 0                            # [K_known, K]

            nbr_idx_clamped = nbr_idx.clamp(min=0)
            n_known     = int(known.sum().item())
            K_neighbors = nbr_idx.shape[1]
            flat_nbr_idx = nbr_idx_clamped.view(-1)
            flat_nbr_emb = self.node_embeddings[flat_nbr_idx]
            neighbor_emb = flat_nbr_emb.view(n_known, K_neighbors, STRING_DIM)
            neighbor_emb = neighbor_emb * nbr_mask.unsqueeze(-1).float()

            aggregated = self.neighborhood_attn(
                center_emb       = center.float(),
                neighbor_emb     = neighbor_emb.float(),
                neighbor_weights = nbr_wgt.float(),
                neighbor_mask    = nbr_mask,
            )  # [K_known, 256]
            emb[known] = aggregated

        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(unknown.sum(), dtype=torch.long, device=emb.device)
            ).to(emb.dtype)
            emb[unknown] = fb

        return emb.float()

    def _get_scf_embedding(self, pert_ids: List[str]) -> torch.Tensor:
        """Lookup pre-computed scFoundation perturbation embeddings.

        Returns [B, SCF_DIM] float32.
        """
        indices:      List[int]  = []
        use_fallback: List[bool] = []

        for pid in pert_ids:
            idx = self.scf_pert_to_idx.get(pid, -1)
            if idx >= 0:
                indices.append(idx)
                use_fallback.append(False)
            else:
                indices.append(0)  # placeholder index
                use_fallback.append(True)

        idx_tensor = torch.tensor(
            indices, dtype=torch.long, device=self.scf_embeddings.device
        )
        emb = self.scf_embeddings[idx_tensor].float()  # [B, 768]

        fallback_mask = torch.tensor(use_fallback, device=emb.device)
        if fallback_mask.any():
            fb = self.scf_fallback_emb.weight[0].to(emb.dtype)
            emb = emb.clone()
            emb[fallback_mask] = fb

        return emb

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        string_node_idx: torch.Tensor,
        pert_ids:        List[str],
    ) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        # All embeddings are float32 from pre-computation.
        string_emb = self._get_string_embedding(string_node_idx)  # [B, 256]
        scf_emb    = self._get_scf_embedding(pert_ids)           # [B, 768]

        combined = torch.cat([string_emb, scf_emb], dim=-1)  # [B, 1024]

        # Fusion head in float32 to prevent bf16 numerical instability
        h = self.fusion_head(combined.float())

        # Bilinear: logits[b,c,g] = h[b] · gene_class_emb[c,g]
        # gene_class_emb is float32 (initialized in __init__, cast in setup)
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb.float())  # [B, 3, G]

        return logits

    # ------------------------------------------------------------------
    # Loss: weighted CE + label smoothing (proven best by lineage)
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure float32 for loss computation to prevent bf16 overflow in the
        # large bilinear matmul (3*6640*256) that can cause NaN gradients.
        logits_f = logits.float()
        targets_l = targets.long()
        B, C, G = logits_f.shape
        return F.cross_entropy(
            logits_f.permute(0, 2, 1).reshape(-1, C),
            targets_l.reshape(-1),
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["string_node_idx"], batch["pert_id"])
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"], batch["pert_id"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits.float(), dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        local_preds = torch.cat(self._val_preds, dim=0)  # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  dim=0)  # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   dim=0)  # [N_local]
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        # Gather across all DDP ranks
        all_preds = self.all_gather(local_preds)  # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)   # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)    # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        # De-duplicate (DDP padding)
        order  = torch.argsort(idx_flat)
        s_idx  = idx_flat[order]
        s_pred = preds_flat[order]
        s_tgt  = tgts_flat[order]
        mask   = torch.cat([
            torch.ones(1, dtype=torch.bool, device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        preds_dedup = s_pred[mask]
        tgts_dedup  = s_tgt[mask]

        f1 = compute_per_gene_f1(preds_dedup, tgts_dedup)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"], batch["pert_id"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        self._test_pids.extend(batch["pert_id"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)  # [N_local]

        all_preds = self.all_gather(local_preds)  # [W, N_local, 3, G]
        all_idx   = self.all_gather(local_idx)    # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            # De-duplicate
            order  = torch.argsort(idx_flat)
            s_idx  = idx_flat[order]
            s_pred = preds_flat[order]
            mask   = torch.cat([
                torch.ones(1, dtype=torch.bool, device=s_idx.device),
                s_idx[1:] != s_idx[:-1],
            ])
            preds_dedup = s_pred[mask]
            unique_sid  = s_idx[mask].tolist()

            test_df     = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {
                i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                for i in range(len(test_df))
            }

            rows = []
            dedup_counter = 0
            for sid in unique_sid:
                sid_i = int(sid)
                if sid_i in idx_to_meta:
                    pid, sym = idx_to_meta[sid_i]
                    pred = preds_dedup[dedup_counter].float().cpu().numpy().tolist()
                    rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})
                dedup_counter += 1

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            print(f"[Node1-2-2] Saved {len(rows)} test predictions to {out_path}")

        self._test_preds.clear()
        self._test_idx.clear()
        self._test_pids.clear()

    # ------------------------------------------------------------------
    # Checkpoint helpers — save only trainable params + small buffers
    # Large pre-computed buffers (node_embeddings, scf_embeddings) are
    # re-computed in setup() and excluded to keep checkpoints small.
    # ------------------------------------------------------------------
    _SKIP_BUFFERS = frozenset({
        "node_embeddings",
        "neighbor_indices",
        "neighbor_weights",
        "scf_embeddings",
    })

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        saved = {}

        # Include all trainable parameters
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full:
                    saved[key] = full[key]

        # Include small buffers only (skip large pre-computed backbone outputs)
        for name, _buf in self.named_buffers():
            short = name.rsplit(".", 1)[-1] if "." in name else name
            if short in self._SKIP_BUFFERS:
                continue
            key = prefix + name
            if key in full:
                saved[key] = full[key]

        total  = sum(p.numel() for p in self.parameters())
        train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_bufs = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Checkpoint: {train}/{total} trainable params "
            f"({100 * train / total:.1f}%), plus {n_bufs} buffer values (excl. large pre-computed)"
        )
        return saved

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: AdamW + linear warmup + CosineAnnealingLR
    # (Proven recipe from node1-1-1-1-1: F1=0.4846)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        # Only head parameters — both backbones are frozen
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=hp.lr, weight_decay=hp.weight_decay)

        # Phase 1: linear warmup from 0.1×lr to lr over warmup_epochs
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=hp.warmup_epochs,
        )
        # Phase 2: cosine annealing (monotonic decay, proven best by lineage)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=hp.t_max, eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[hp.warmup_epochs],
        )
        return {
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-2-2 – Frozen STRING_GNN + Frozen scFoundation Fusion"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=32)
    parser.add_argument("--global-batch-size",  type=int,   default=256)
    parser.add_argument("--max-epochs",         type=int,   default=220)
    parser.add_argument("--lr",                 type=float, default=3e-4)
    parser.add_argument("--weight-decay",       type=float, default=3e-2)
    parser.add_argument("--bilinear-dim",       type=int,   default=256)
    parser.add_argument("--K",                  type=int,   default=16,   dest="K")
    parser.add_argument("--attn-dim",           type=int,   default=64,   dest="attn_dim")
    parser.add_argument("--fusion-dim",         type=int,   default=512,  dest="fusion_dim")
    parser.add_argument("--dropout",            type=float, default=0.35)
    parser.add_argument("--label-smoothing",    type=float, default=0.05, dest="label_smoothing")
    parser.add_argument("--warmup-epochs",      type=int,   default=20,   dest="warmup_epochs")
    parser.add_argument("--t-max",              type=int,   default=150,  dest="t_max")
    parser.add_argument("--patience",           type=int,   default=8)
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--val-check-interval", type=float, default=1.0,  dest="val_check_interval")
    parser.add_argument("--debug-max-step",     type=int,   default=None, dest="debug_max_step")
    parser.add_argument("--fast-dev-run",       action="store_true",      dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean checkpoint directory (rank 0 only)
    import shutil as _shutil
    ckpt_dir = output_dir / "checkpoints"
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        if ckpt_dir.exists():
            _shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Debug / fast-dev-run limits
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

    # DataModule
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    # Model
    model = FusionSCFStringModel(
        bilinear_dim    = args.bilinear_dim,
        K               = args.K,
        attn_dim        = args.attn_dim,
        fusion_dim      = args.fusion_dim,
        dropout         = args.dropout,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        t_max           = args.t_max,
        label_smoothing = args.label_smoothing,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
        save_on_train_epoch_end = False,
    )
    es_cb = EarlyStopping(
        monitor   = "val/f1",
        mode      = "max",
        patience  = args.patience,
        min_delta = 1e-4,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy: find_unused_parameters=True because fallback_emb / scf_fallback_emb
    # may not be used in every batch if all pert_ids are in the vocab.
    use_ddp = n_gpus > 1 and not fast_dev_run
    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if use_ddp else "auto"
    )
    devices_for_trainer = 1 if (fast_dev_run and n_gpus > 1) else n_gpus

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = devices_for_trainer,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        max_steps               = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches     = lim_train,
        limit_val_batches       = lim_val,
        limit_test_batches      = lim_test,
        val_check_interval      = (
            1.0 if (args.debug_max_step is not None or fast_dev_run)
            else args.val_check_interval
        ),
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

    # Save test score summary
    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1-2-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
