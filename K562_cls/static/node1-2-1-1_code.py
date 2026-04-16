"""Node 1-2 (second child) — Frozen STRING_GNN + Frozen scFoundation Late Fusion.

This node implements the mandatory fusion direction identified across the entire MCTS tree:
combining the proven frozen STRING_GNN PPI topology embeddings (node1-2, F1=0.4769)
with frozen scFoundation perturbation-aware cell embeddings to break the confirmed
~0.485 STRING_GNN-only ceiling.

Key design decisions (all memory-driven):
1. FROZEN STRING_GNN backbone with PPI neighborhood attention (K=16, attn_dim=64)
   - Proven best config: node1-1-1-1-1 (F1=0.4846), node1-2 (F1=0.4769)
   - Frozen is consistently better than DLR or full fine-tune on this small dataset
2. FROZEN scFoundation (100M) cell encoder providing perturbation identity signal
   - Input: 1-hot gene expression (perturbed gene=1.0, all others=0.0)
   - Output: mean-pooled cell embedding [B, 768]
   - Precomputed at setup() time for all samples (both backbones frozen -> embeddings never change)
   - Coverage: 1681/1696 perturbations in scFoundation vocabulary (>99%)
3. Gated fusion of STRING [256] + scFoundation [768] = [1024] concatenation
   - Simple gated fusion: gate_proj maps concat_emb -> gate, output = gate * string_emb + (1-gate) * scf_emb_proj
   - Followed by the proven MLP head: LN -> Linear -> GELU -> Dropout
4. Proven bilinear gene-class embedding head [3, 6640, 256]
5. Weighted CE + label smoothing (epsilon=0.05) — proven by entire node1 lineage
6. AdamW with cosine annealing (warmup=10, T_max=150, eta_min=5e-6) — no backbone LR needed
7. weight_decay=3e-2, dropout=0.35 — proven regularization for this task

Architecture:
    STRING_GNN (frozen)
    → node embeddings [18870, 256]
    → for each sample:
         center_emb = node_emb[pert_idx]                    # [B, 256]
         + neighborhood attention (K=16, attn_dim=64)       # -> [B, 256]
    scFoundation (frozen, precomputed)
    → scf_emb [B, 768]

    Gated fusion:
         fused = concat([string_emb, scf_emb])              # [B, 1024]
         gate = sigmoid(gate_proj(fused))                   # [B, fusion_dim]
         output = gate * string_proj(string_emb) + (1-gate) * scf_proj(scf_emb)
                                                            # [B, fusion_dim=256]

    MLP head: LN(256) → Linear(256→bilinear_dim) → GELU → Dropout
    Bilinear: logits[b,c,g] = h[b] · gene_class_emb[c,g]   # [B, 3, G]

Inspired by:
  - node1-2 (F1=0.4769): proven frozen STRING_GNN + neighborhood attention base
  - node1-1-1-1-1 (F1=0.4846): best STRING_GNN config (frozen K=16 attn_dim=64)
  - node1-2-1 feedback: "Fusion architecture is MANDATORY to break the ~0.485 ceiling"
  - node4 (F1=0.4585) / node4-1 (F1=0.4629): scFoundation+STRING_GNN fusion validated
  - node4-1-1-1 feedback: "Replace focal loss with weighted CE, freeze GNN" -> more comparable
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

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

# Remapped class frequencies (after -1->0, 0->1, 1->2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
SCF_DIR = Path("/home/Models/scFoundation")
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

STRING_DIM = 256   # STRING_GNN hidden dimension
SCF_DIM    = 768   # scFoundation hidden dimension


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights; neutral class stays ~1."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def load_string_gnn_mapping() -> Dict[str, int]:
    """Load STRING_GNN node_names.json -> Ensembl-ID to node-index mapping."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    return {name: idx for idx, name in enumerate(node_names)}


def load_scf_mapping() -> Dict[str, int]:
    """Load scFoundation gene_ids.json -> Ensembl-ID to vocab-index mapping."""
    gene_ids: List[str] = json.loads((SCF_DIR / "gene_ids.json").read_text())
    return {gid: idx for idx, gid in enumerate(gene_ids)}


def precompute_neighborhood(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    n_nodes: int,
    K: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute top-K PPI neighbors for each node by edge confidence."""
    src = edge_index[0]
    dst = edge_index[1]
    wgt = edge_weight

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


def precompute_scf_embeddings(
    pert_ids: List[str],
    scf_model: nn.Module,
    tokenizer,
    scf_id_map: Dict[str, int],
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Precompute frozen scFoundation embeddings for all perturbations.

    For each perturbation gene, creates a 1-hot expression vector (perturbed
    gene = 1.0, all others = 0.0) and runs scFoundation to get a perturbation-
    specific embedding. This effectively queries scFoundation about what
    transcriptional state is associated with this gene's activation.

    For genes not in scFoundation vocabulary, uses a zero vector as fallback.

    Returns:
        [N, SCF_DIM] float32 tensor of precomputed embeddings
    """
    scf_model.eval()
    all_embs = []

    for batch_start in range(0, len(pert_ids), batch_size):
        batch_ids = pert_ids[batch_start:batch_start + batch_size]
        batch_inputs = []
        batch_fallback_mask = []

        for pid in batch_ids:
            if pid in scf_id_map:
                batch_inputs.append({'gene_ids': [pid], 'expression': [1.0]})
                batch_fallback_mask.append(False)
            else:
                # Not in vocabulary — use a dummy input, will be replaced by fallback
                batch_inputs.append({'gene_ids': [next(iter(scf_id_map))], 'expression': [1.0]})
                batch_fallback_mask.append(True)

        # Batch tokenize
        inputs = tokenizer(batch_inputs, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = scf_model(**inputs)

        # Mean pool over dynamic sequence: [B, nnz+2, 768] -> [B, 768]
        emb = out.last_hidden_state.float().mean(dim=1)  # [B, 768]

        # Zero out fallback positions
        for i, is_fallback in enumerate(batch_fallback_mask):
            if is_fallback:
                emb[i] = 0.0

        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0)  # [N, 768]


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic."""
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
    """K562 DEG prediction dataset with precomputed scFoundation embeddings."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_map: Dict[str, int],
        scf_embeddings: torch.Tensor,  # [N, 768] precomputed
    ) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()

        # STRING_GNN node index for each sample (-1 means not in STRING)
        self.string_node_indices = torch.tensor(
            [string_map.get(p, -1) for p in self.pert_ids], dtype=torch.long
        )

        # Pre-computed scFoundation embeddings [N, 768]
        self.scf_embeddings = scf_embeddings

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
            "scf_emb":         self.scf_embeddings[idx],  # [768]
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]  # [G] in {0, 1, 2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx":      torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":         [b["pert_id"]  for b in batch],
        "symbol":          [b["symbol"]   for b in batch],
        "string_node_idx": torch.stack([b["string_node_idx"] for b in batch]),
        "scf_emb":         torch.stack([b["scf_emb"] for b in batch]),  # [B, 768]
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
        self.scf_id_map: Optional[Dict[str, int]] = None
        self._train_scf_emb: Optional[torch.Tensor] = None
        self._val_scf_emb:   Optional[torch.Tensor] = None
        self._test_scf_emb:  Optional[torch.Tensor] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.string_map is None:
            self.string_map = load_string_gnn_mapping()
        if self.scf_id_map is None:
            self.scf_id_map = load_scf_mapping()

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        # Precompute scFoundation embeddings for all splits (rank-safe: each rank does same work)
        if self._train_scf_emb is None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))

            # Rank 0 downloads/loads tokenizer first, others wait
            if local_rank == 0:
                _ = AutoTokenizer.from_pretrained(str(SCF_DIR), trust_remote_code=True)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

            scf_tokenizer = AutoTokenizer.from_pretrained(str(SCF_DIR), trust_remote_code=True)
            scf_model = AutoModel.from_pretrained(str(SCF_DIR), trust_remote_code=True)
            scf_model.eval()

            # Move to GPU for fast precomputation
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            scf_model = scf_model.to(device)

            print("[DataModule] Precomputing scFoundation embeddings for all splits...")
            self._train_scf_emb = precompute_scf_embeddings(
                train_df["pert_id"].tolist(), scf_model, scf_tokenizer, self.scf_id_map,
                batch_size=64, device=device
            )
            self._val_scf_emb = precompute_scf_embeddings(
                val_df["pert_id"].tolist(), scf_model, scf_tokenizer, self.scf_id_map,
                batch_size=64, device=device
            )
            self._test_scf_emb = precompute_scf_embeddings(
                test_df["pert_id"].tolist(), scf_model, scf_tokenizer, self.scf_id_map,
                batch_size=64, device=device
            )
            print(f"[DataModule] Precomputed: train={self._train_scf_emb.shape}, "
                  f"val={self._val_scf_emb.shape}, test={self._test_scf_emb.shape}")

            # Free scFoundation model from GPU (no longer needed)
            del scf_model
            torch.cuda.empty_cache()

        self.train_ds = DEGDataset(train_df, self.string_map, self._train_scf_emb)
        self.val_ds   = DEGDataset(val_df,   self.string_map, self._val_scf_emb)
        self.test_ds  = DEGDataset(test_df,  self.string_map, self._test_scf_emb)

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
# Neighborhood Attention Module (proven config from lineage)
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Center-context gated attention over top-K PPI neighbors.

    K=16, attn_dim=64 — proven best configuration from node1-1-1-1-1 (F1=0.4846)
    and confirmed in node1-2 (F1=0.4769).
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
        center_emb: torch.Tensor,
        neighbor_emb: torch.Tensor,
        neighbor_weights: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, K, D = neighbor_emb.shape

        center_expanded = center_emb.unsqueeze(1).expand(-1, K, -1)
        pair_features = torch.cat([center_expanded, neighbor_emb], dim=-1)
        attn_scores = self.attn_proj(pair_features).squeeze(-1)
        attn_scores = attn_scores + neighbor_weights
        attn_scores = attn_scores.masked_fill(~neighbor_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        aggregated = torch.bmm(attn_weights.unsqueeze(1), neighbor_emb).squeeze(1)
        gate = torch.sigmoid(self.gate_proj(center_emb))
        output = center_emb + gate * aggregated

        return output


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FrozenFusionModel(pl.LightningModule):
    """Frozen STRING_GNN + Frozen scFoundation Late Fusion.

    Architecture breakdown:
    1. STRING_GNN (frozen, precomputed at setup)
       → [B, 256] via PPI neighborhood attention (K=16, attn_dim=64)
    2. scFoundation cell encoder (frozen, precomputed in DataModule)
       → [B, 768] via mean pooling of dynamic sequence
    3. Gated fusion: concat([string, scf]) → gate-weighted combination → [B, fusion_dim]
    4. MLP head: LN → Linear → GELU → Dropout → [B, bilinear_dim]
    5. Bilinear: h · gene_class_emb → [B, 3, G]

    Fusion design:
    - concat_emb = [string_emb (256) | scf_emb (768)] = [1024]
    - string_proj: Linear(256, fusion_dim)
    - scf_proj:    Linear(768, fusion_dim)
    - gate_proj:   Linear(1024, fusion_dim)  → sigmoid → gate
    - fused = gate * string_proj(string_emb) + (1-gate) * scf_proj(scf_emb)
    This allows the model to learn which modality is more informative per sample.
    """

    def __init__(
        self,
        bilinear_dim: int = 256,
        K: int = 16,
        attn_dim: int = 64,
        fusion_dim: int = 256,
        dropout: float = 0.35,
        head_lr: float = 3e-4,
        weight_decay: float = 3e-2,
        warmup_epochs: int = 10,
        t_max: int = 150,
        eta_min: float = 5e-6,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        hp = self.hparams

        # ----------------------------------------------------------------
        # 1. Load STRING_GNN backbone — FROZEN
        # Pretrain on PPI graph topology; frozen for stable neighborhood attention
        # ----------------------------------------------------------------
        self.backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph["edge_index"].long()
        edge_weight = graph["edge_weight"].float()

        n_nodes = getattr(self.backbone.config, 'num_nodes', 18870)

        # Pre-compute top-K PPI neighbors
        print(f"Pre-computing top-{hp.K} PPI neighbors for {n_nodes} nodes...")
        nbr_idx, nbr_wgt = precompute_neighborhood(edge_index, edge_weight, n_nodes, K=hp.K)

        self.register_buffer("neighbor_indices", nbr_idx)
        self.register_buffer("neighbor_weights", nbr_wgt)
        self.register_buffer("_edge_index",  edge_index)
        self.register_buffer("_edge_weight", edge_weight)

        # Precompute frozen STRING_GNN embeddings (backbone is frozen, so embeddings are static)
        print("Precomputing frozen STRING_GNN node embeddings...")
        with torch.no_grad():
            gnn_out = self.backbone(
                edge_index=self._edge_index,
                edge_weight=self._edge_weight,
            )
            static_node_emb = gnn_out.last_hidden_state.float().cpu()  # [18870, 256]
        self.register_buffer("_static_node_emb", static_node_emb)

        # ----------------------------------------------------------------
        # 2. Learnable fallback for unknown pert_ids
        # ----------------------------------------------------------------
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 3. Neighborhood Attention Aggregator (proven K=16, attn_dim=64)
        # ----------------------------------------------------------------
        self.neighborhood_attn = NeighborhoodAttentionAggregator(
            embed_dim=STRING_DIM,
            attn_dim=hp.attn_dim,
            dropout=0.0,
        )

        # ----------------------------------------------------------------
        # 4. Gated fusion: STRING [256] + scFoundation [768] -> [fusion_dim]
        # gate_proj maps concatenated [1024] -> fusion_dim for element-wise gating
        # ----------------------------------------------------------------
        self.string_proj = nn.Linear(STRING_DIM, hp.fusion_dim, bias=False)
        self.scf_proj    = nn.Linear(SCF_DIM,    hp.fusion_dim, bias=False)
        self.gate_proj   = nn.Linear(STRING_DIM + SCF_DIM, hp.fusion_dim)

        # ----------------------------------------------------------------
        # 5. MLP projection head (proven flat design)
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(hp.fusion_dim),
            nn.Linear(hp.fusion_dim, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # ----------------------------------------------------------------
        # 6. Bilinear gene-class embedding
        # ----------------------------------------------------------------
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # Class weights for weighted CE
        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        for name, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # STRING embedding lookup with PPI neighborhood aggregation
    # Uses STATIC precomputed backbone embeddings (frozen)
    # ------------------------------------------------------------------
    def _get_string_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Look up frozen STRING_GNN embeddings with neighborhood attention.

        Uses precomputed static node embeddings (backbone is frozen).
        Applies neighborhood attention aggregation (K=16, attn_dim=64).

        Returns: [B, STRING_DIM=256] float32
        """
        B = string_node_idx.shape[0]
        node_emb = self._static_node_emb  # [18870, 256] on correct device

        emb = torch.zeros(B, STRING_DIM, dtype=node_emb.dtype, device=node_emb.device)
        known   = string_node_idx >= 0
        unknown = ~known

        if known.any():
            known_idx = string_node_idx[known]
            center = node_emb[known_idx]  # [K_known, 256]

            nbr_idx = self.neighbor_indices[known_idx]  # [K_known, K]
            nbr_wgt = self.neighbor_weights[known_idx]  # [K_known, K]
            nbr_mask = nbr_idx >= 0

            nbr_idx_clamped = nbr_idx.clamp(min=0)
            n_known = int(known.sum().item())
            K_neighbors = nbr_idx.shape[1]

            flat_nbr_emb = node_emb[nbr_idx_clamped.view(-1)]
            neighbor_emb = flat_nbr_emb.view(n_known, K_neighbors, STRING_DIM)
            neighbor_emb = neighbor_emb * nbr_mask.unsqueeze(-1).float()

            aggregated = self.neighborhood_attn(
                center_emb       = center.float(),
                neighbor_emb     = neighbor_emb.float(),
                neighbor_weights = nbr_wgt.float(),
                neighbor_mask    = nbr_mask,
            )
            emb[known] = aggregated

        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(unknown.sum(), dtype=torch.long, device=node_emb.device)
            ).to(node_emb.dtype)
            emb[unknown] = fb

        return emb.float()

    def forward(
        self,
        string_node_idx: torch.Tensor,
        scf_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits [B, 3, G].

        Args:
            string_node_idx: [B] long — STRING node indices (-1 for unknowns)
            scf_emb:         [B, 768] float — precomputed scFoundation embeddings
        """
        # STRING branch: frozen embeddings + neighborhood attention -> [B, 256]
        string_emb = self._get_string_embeddings(string_node_idx)  # [B, 256]

        # scFoundation branch: precomputed, already [B, 768]
        scf_emb = scf_emb.float()  # Ensure float32

        # Gated fusion
        concat_emb = torch.cat([string_emb, scf_emb], dim=-1)  # [B, 1024]
        gate = torch.sigmoid(self.gate_proj(concat_emb))         # [B, fusion_dim]
        fused = gate * self.string_proj(string_emb) + (1.0 - gate) * self.scf_proj(scf_emb)
        # fused: [B, fusion_dim=256]

        # MLP head
        h = self.head(fused)  # [B, bilinear_dim]

        # Bilinear interaction
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)  # [B, 3, G]
        return logits

    # ------------------------------------------------------------------
    # Loss: weighted CE + label smoothing (proven by entire lineage)
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["string_node_idx"], batch["scf_emb"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"], batch["scf_emb"])
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
        local_preds = torch.cat(self._val_preds, dim=0)
        local_tgts  = torch.cat(self._val_tgts,  dim=0)
        local_idx   = torch.cat(self._val_idx,   dim=0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds)
        all_tgts  = self.all_gather(local_tgts)
        all_idx   = self.all_gather(local_idx)

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

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
        logits = self(batch["string_node_idx"], batch["scf_emb"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = torch.cat(self._test_preds, dim=0)
        local_idx   = torch.cat(self._test_idx,   dim=0)

        all_preds = self.all_gather(local_preds)
        all_idx   = self.all_gather(local_idx)

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            order  = torch.argsort(idx_flat)
            s_idx  = idx_flat[order]
            s_pred = preds_flat[order]
            mask   = torch.cat([
                torch.ones(1, dtype=torch.bool, device=s_idx.device),
                s_idx[1:] != s_idx[:-1],
            ])
            preds_dedup = s_pred[mask]
            unique_sid  = s_idx[mask].tolist()

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                           for i in range(len(test_df))}

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
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[FrozenFusion] Saved {len(rows)} test predictions.")

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
        bufs  = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Checkpoint: {train}/{total} params ({100 * train / total:.1f}%), "
            f"plus {bufs} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: AdamW with warmup + cosine (no backbone LR needed — frozen)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        # All trainable parameters share the same head_lr
        # (backbone is frozen -> no backbone param group needed)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=hp.head_lr,
            weight_decay=hp.weight_decay,
        )

        # Warmup + cosine annealing (proven schedule from node1-2)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=hp.t_max,
            eta_min=hp.eta_min,
        )
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

    parser = argparse.ArgumentParser(
        description="Node1-2 (child 2) — Frozen STRING_GNN + Frozen scFoundation Late Fusion"
    )
    parser.add_argument("--micro-batch-size",  type=int,   default=32)
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument("--max-epochs",        type=int,   default=250)
    parser.add_argument("--head-lr",           type=float, default=3e-4, dest="head_lr")
    parser.add_argument("--weight-decay",      type=float, default=3e-2)
    parser.add_argument("--bilinear-dim",      type=int,   default=256)
    parser.add_argument("--fusion-dim",        type=int,   default=256, dest="fusion_dim")
    parser.add_argument("--K",                 type=int,   default=16,  dest="K")
    parser.add_argument("--attn-dim",          type=int,   default=64,  dest="attn_dim")
    parser.add_argument("--dropout",           type=float, default=0.35)
    parser.add_argument("--label-smoothing",   type=float, default=0.05, dest="label_smoothing")
    parser.add_argument("--warmup-epochs",     type=int,   default=10,  dest="warmup_epochs")
    parser.add_argument("--t-max",             type=int,   default=150, dest="t_max")
    parser.add_argument("--eta-min",           type=float, default=5e-6, dest="eta_min")
    parser.add_argument("--patience",          type=int,   default=10)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--debug-max-step",    type=int,   default=None, dest="debug_max_step")
    parser.add_argument("--fast-dev-run",      action="store_true", dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        lim_test  = 1.0  # Always run full test to save ALL 154 predictions
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
    model = FrozenFusionModel(
        bilinear_dim    = args.bilinear_dim,
        K               = args.K,
        attn_dim        = args.attn_dim,
        fusion_dim      = args.fusion_dim,
        dropout         = args.dropout,
        head_lr         = args.head_lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        t_max           = args.t_max,
        eta_min         = args.eta_min,
        label_smoothing = args.label_smoothing,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
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

    # Strategy — find_unused_parameters=True: required by Lightning DDP when model has
    # parameters not used in every forward pass (e.g., fallback_emb for unknown gene IDs)
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
    print(f"[FrozenFusion] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
