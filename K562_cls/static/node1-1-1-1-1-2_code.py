"""Node 1-1-1-1-1-2 – STRING_GNN K=16 Neighborhood Attention + AIDO.Cell-100M (LoRA) Fusion.

Improves on parent node1-1-1-1-1 (test F1=0.4846, best STRING-GNN-only) by:
1. Late-fusion with AIDO.Cell-100M (PRIMARY INNOVATION):
   - AIDO.Cell-100M with LoRA r=8 encodes perturbation as synthetic expression vector
     (perturbed gene = 1.0, all other genes = -1.0 / missing)
   - Provides perturbation-specific transcriptome context that STRING_GNN lacks
   - Summary token [B, 640] from position 19264 captures global perturbation signature
   - Directly addresses parent's "static PPI topology without perturbation-specific signal"
2. RETAIN parent's proven K=16 neighborhood attention (attn_dim=64, single-head gating)
   - Proven superior: K=32 and attn_dim=32 both regressed (node1-1-1-1-1-1: -0.0103)
3. Fusion head: concat([STRING [256], AIDO.Cell [640]]) → [B, 896]
   → Linear(896→256) → LN → GELU → Dropout(0.5) → Linear(256→19920)
4. Hyperparameters tuned for AIDO.Cell:
   - lr=1e-4 (LoRA requires lower LR than pure linear head)
   - warmup_epochs=10, T_max=200, max_epochs=300
   - patience=15 (AIDO.Cell exhibits late spikes, proven in node2-1-1-1-1-1)
   - weight_decay=2e-2 (proven for AIDO.Cell fusion)
   - dropout=0.5 (proven for AIDO.Cell fusion, stronger head regularization)

Memory sources:
- node2-1-1-1-1-1 (F1=0.5128, best in tree): AIDO.Cell + STRING_GNN K=16 2-head = proven approach
- node1-1-1-1-1 (parent F1=0.4846): K=16 attn_dim=64 single-head proven superior
- node1-1-1-1-1-1 (sibling F1=0.4743): K=32/attn_dim=32 confirmed to regress → keep K=16+64
- parent feedback: "fusion with scFoundation" = mandatory next step for breaking ~0.485 ceiling

Architecture:
    Stream 1 — STRING_GNN (frozen, pre-computed buffer [18870, 256]):
        pert_id → node index → center_emb [B, 256]
        + top-K=16 neighbors → NeighborhoodAttention(attn_dim=64) → [B, 256]

    Stream 2 — AIDO.Cell-100M (LoRA r=8, α=16):
        Pre-computed input_ids [N, 19264] float32 (pert gene=1.0, rest=-1.0)
        → AIDO.Cell-100M (bf16) → last_hidden_state [B, 19266, 640]
        → summary_token [:, 19264, :] → [B, 640]

    Fusion:
        concat([string_emb, aido_emb]) → [B, 896]
        → Linear(896→256) → LayerNorm → GELU → Dropout(0.5)
        → Linear(256→19920) → reshape → [B, 3, 6640]

    Loss: weighted CE + label_smoothing=0.05
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
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3

# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

STRING_GNN_DIR   = Path("/home/Models/STRING_GNN")
AIDO_CELL_DIR    = Path("/home/Models/AIDO.Cell-100M")
DATA_ROOT        = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV        = DATA_ROOT / "train.tsv"
VAL_TSV          = DATA_ROOT / "val.tsv"
TEST_TSV         = DATA_ROOT / "test.tsv"

STRING_DIM  = 256    # STRING_GNN hidden dimension
AIDO_DIM    = 640    # AIDO.Cell-100M hidden dimension
ATTN_DIM    = 64     # Neighborhood attention projection dimension (proven from parent)
TOPK        = 16     # Top-K PPI neighbors (proven optimal — K=32 regressed F1 by -0.0103)
FUSED_DIM   = STRING_DIM + AIDO_DIM   # 896 for fusion head


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights for weighted cross-entropy."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def load_string_gnn_mapping() -> Dict[str, int]:
    """Load STRING_GNN node_names.json → Ensembl-ID to node-index mapping."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    return {name: idx for idx, name in enumerate(node_names)}


def compute_topk_neighbors(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    k: int,
) -> tuple:
    """Precompute top-K neighbors per node by edge weight.

    Args:
        edge_index: [2, E] long — source and destination indices
        edge_weight: [E] float — STRING combined_score weights
        num_nodes: total number of nodes in the graph
        k: number of top neighbors to keep per node

    Returns:
        topk_neighbors: [num_nodes, k] long — top-K neighbor indices
        topk_weights:   [num_nodes, k] float — corresponding softmaxed edge weights
                        (for nodes with fewer than K neighbors, padded with self-loops)
    """
    src = edge_index[0]
    dst = edge_index[1]
    weights = edge_weight

    # Initialize: default to self-loops
    topk_neighbors_np = torch.zeros(num_nodes, k, dtype=torch.long)
    topk_weights_raw  = torch.zeros(num_nodes, k, dtype=torch.float)

    for i in range(num_nodes):
        topk_neighbors_np[i] = i  # default: self-loop

    # Sort by source node for efficient grouping
    sort_idx = torch.argsort(src)
    src_sorted = src[sort_idx]
    dst_sorted = dst[sort_idx]
    wt_sorted  = weights[sort_idx]

    unique_srcs, counts = torch.unique_consecutive(src_sorted, return_counts=True)
    offset = 0
    for i, (node_id, cnt) in enumerate(zip(unique_srcs.tolist(), counts.tolist())):
        nb_dst = dst_sorted[offset:offset + cnt]
        nb_wt  = wt_sorted[offset:offset + cnt]

        actual_k = min(k, cnt)
        topk_vals, topk_idx = torch.topk(nb_wt, actual_k)

        top_dst = nb_dst[topk_idx]
        top_wt  = topk_vals

        topk_neighbors_np[node_id, :actual_k] = top_dst
        topk_weights_raw[node_id, :actual_k]   = top_wt

        offset += cnt

    # Softmax-normalize weights per node
    topk_weights_soft = torch.softmax(topk_weights_raw, dim=1)

    return topk_neighbors_np, topk_weights_soft


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0, 1, 2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)            # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)            # [N, G]
        is_pred = (y_hat == c)              # [N, G]
        present = is_true.any(dim=0)        # [G]

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
    """K562 DEG prediction dataset with pre-computed AIDO.Cell tokenized inputs."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_map: Dict[str, int],
        tokenizer,
    ) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()

        # STRING_GNN node index for each sample (-1 means not in STRING)
        self.string_node_indices = torch.tensor(
            [string_map.get(p, -1) for p in self.pert_ids], dtype=torch.long
        )

        # Pre-compute AIDO.Cell tokenized inputs for efficiency
        # Each sample: only the perturbed gene = 1.0, all others = -1.0 (missing)
        # This creates a "virtual single-gene perturbation" expression profile
        print(f"[DEGDataset] Pre-computing AIDO.Cell tokenized inputs for {len(self.pert_ids)} samples...")
        input_ids_list = []
        for pert_id in self.pert_ids:
            # Tokenize with only the perturbed gene at expression=1.0
            # Tokenizer fills missing genes with -1.0 automatically
            tok = tokenizer(
                {"gene_ids": [pert_id], "expression": [1.0]},
                return_tensors="pt",
            )
            # AIDO.Cell tokenizer returns 1D tensor [19264], not 2D [1, 19264]
            input_ids_list.append(tok["input_ids"])  # [19264] float32 (1D)
        self.cell_input_ids = torch.stack(input_ids_list)  # [N, 19264] float32
        print(f"[DEGDataset] Done. cell_input_ids shape: {self.cell_input_ids.shape}")

        # Labels (if available)
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
            "cell_input_ids":    self.cell_input_ids[idx],   # [19264] float32
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
        "cell_input_ids":  torch.stack([b["cell_input_ids"] for b in batch]),  # [B, 19264]
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.string_map: Optional[Dict[str, int]] = None
        self.tokenizer = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.string_map is None:
            self.string_map = load_string_gnn_mapping()

        # Load AIDO.Cell tokenizer (rank-0 first, then barrier)
        if self.tokenizer is None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank == 0:
                AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            self.tokenizer = AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df, self.string_map, self.tokenizer)
        self.val_ds   = DEGDataset(val_df,   self.string_map, self.tokenizer)
        self.test_ds  = DEGDataset(test_df,  self.string_map, self.tokenizer)

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
# Neighborhood Attention Module (identical to parent node1-1-1-1-1)
# ---------------------------------------------------------------------------
class NeighborhoodAttention(nn.Module):
    """Lightweight PPI neighborhood attention (proven in parent, attn_dim=64).

    Architecture:
        q = W_q(center)            [B, attn_dim]
        k = W_k(neigh_embs)        [B, K, attn_dim]
        attn = softmax(q @ k^T / sqrt(d)) * ppi_weights   [B, 1, K]
        attn = attn / sum(attn)    (renormalize after PPI weighting)
        context = attn @ neigh_embs  [B, 1, 256] → squeeze → [B, 256]
        gate = sigmoid(W_gate([center, context]))   [B, 256]
        output = gate * center + (1 - gate) * context    [B, 256]
    """

    def __init__(self, emb_dim: int = 256, attn_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn_dim = attn_dim
        self.scale    = attn_dim ** -0.5

        self.W_q    = nn.Linear(emb_dim, attn_dim, bias=False)
        self.W_k    = nn.Linear(emb_dim, attn_dim, bias=False)
        self.W_gate = nn.Linear(emb_dim * 2, emb_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize gate bias toward identity (center preferred initially)
        nn.init.zeros_(self.W_gate.bias)
        nn.init.xavier_uniform_(self.W_gate.weight)

    def forward(
        self,
        center: torch.Tensor,          # [B, D]
        neigh_embs: torch.Tensor,      # [B, K, D]
        ppi_weights: torch.Tensor,     # [B, K]  — softmaxed STRING confidence
    ) -> torch.Tensor:
        """Returns [B, D] context-aware perturbation embedding."""
        B, K, D = neigh_embs.shape

        # Attention scores: center queries over neighbor keys
        q = self.W_q(center)           # [B, attn_dim]
        k = self.W_k(neigh_embs.reshape(B * K, D)).reshape(B, K, self.attn_dim)  # [B, K, attn_dim]

        # Scaled dot-product attention
        attn = (q.unsqueeze(1) @ k.transpose(1, 2)) * self.scale  # [B, 1, K]
        attn = attn.squeeze(1)                                       # [B, K]

        # Incorporate PPI edge confidence as prior (multiplicative)
        attn = F.softmax(attn, dim=-1)      # [B, K]
        attn = attn * ppi_weights           # [B, K]
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)  # renormalize

        # Dropout on attention weights for regularization
        attn = self.dropout(attn)           # [B, K]

        # Context vector: attention-weighted sum of neighbor embeddings
        context = (attn.unsqueeze(1) @ neigh_embs).squeeze(1)  # [B, D]

        # Learnable gating: blend center and neighborhood context
        gate = torch.sigmoid(self.W_gate(
            torch.cat([center, context], dim=-1)  # [B, 2D]
        ))  # [B, D]
        output = gate * center + (1.0 - gate) * context  # [B, D]

        return output


# ---------------------------------------------------------------------------
# Fusion Model
# ---------------------------------------------------------------------------
class FusionModel(pl.LightningModule):
    """STRING_GNN Neighborhood Attention + AIDO.Cell-100M (LoRA) Late Fusion.

    Architecture:
        Stream 1 — STRING_GNN (frozen):
            Pre-computed node_embeddings [18870, 256] (frozen)
            K=16 PPI neighborhood attention (attn_dim=64) → pert_emb [B, 256]
        Stream 2 — AIDO.Cell-100M (LoRA r=8):
            cell_input_ids [B, 19264] float32 (perturbed gene=1.0, rest=-1.0)
            → last_hidden_state [B, 19266, 640]
            → summary_token = lhs[:, 19264, :] → [B, 640]
        Fusion:
            concat([pert_emb, summary_token]) → [B, 896]
            → Linear(896→256) → LN → GELU → Dropout(0.5)
            → Linear(256→3*6640) → reshape [B, 3, 6640]
    """

    def __init__(
        self,
        head_hidden:    int   = 256,
        dropout:        float = 0.50,
        attn_dim:       int   = 64,
        topk:           int   = 16,
        lora_r:         int   = 8,
        lora_alpha:     int   = 16,
        lr:             float = 1e-4,
        weight_decay:   float = 2e-2,
        warmup_epochs:  int   = 10,
        T_max:          int   = 200,
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
        # 1. Pre-compute STRING_GNN node embeddings (frozen backbone)
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

        # ----------------------------------------------------------------
        # 2. Pre-compute top-K=16 PPI neighbors per node (from parent, proven)
        # ----------------------------------------------------------------
        num_nodes = node_emb.shape[0]
        print(f"[Node1-1-1-1-1-2] Pre-computing top-{hp.topk} PPI neighbors for {num_nodes} nodes...")
        topk_nb, topk_wt = compute_topk_neighbors(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            k=hp.topk,
        )
        self.register_buffer("topk_neighbors", topk_nb)  # [N, K] long
        self.register_buffer("topk_weights",   topk_wt)  # [N, K] float

        del backbone, graph, edge_index, edge_weight, gnn_out
        print("[Node1-1-1-1-1-2] STRING_GNN buffers ready.")

        # ----------------------------------------------------------------
        # 3. Learnable fallback embedding for pert_ids not in STRING graph
        # ----------------------------------------------------------------
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 4. PPI Neighborhood Attention Module (K=16, attn_dim=64 — proven from parent)
        #    K=16 optimal: K=32 caused -0.0103 regression; attn_dim=64 > 32
        # ----------------------------------------------------------------
        self.neighborhood_attn = NeighborhoodAttention(
            emb_dim  = STRING_DIM,
            attn_dim = hp.attn_dim,    # 64 (proven; halving to 32 regressed F1)
            dropout  = 0.1,
        )

        # ----------------------------------------------------------------
        # 5. AIDO.Cell-100M with LoRA r=8 (proven in node2-1-1-1-1-1, F1=0.5128)
        # ----------------------------------------------------------------
        print("[Node1-1-1-1-1-2] Loading AIDO.Cell-100M with LoRA...")
        aido_model = AutoModel.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        aido_model = aido_model.to(torch.bfloat16)
        aido_model.config.use_cache = False

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,            # 8 (proven — r=16 caused overfitting in node2)
            lora_alpha=hp.lora_alpha,  # 16
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
            # flash_self shares weight tensors with self, LoRA applied to both
        )
        self.aido_model = get_peft_model(aido_model, lora_cfg)
        self.aido_model.config.use_cache = False
        # Gradient checkpointing reduces activation memory substantially for 100M
        self.aido_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        n_lora_params = sum(p.numel() for p in self.aido_model.parameters() if p.requires_grad)
        print(f"[Node1-1-1-1-1-2] AIDO.Cell-100M LoRA trainable: {n_lora_params:,} params")

        # ----------------------------------------------------------------
        # 6. Fusion head: Linear(896→256) → LN → GELU → Dropout(0.5) → Linear(256→19920)
        #    Head architecture proven in node2-1-1-1-1-1 (F1=0.5128)
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(FUSED_DIM, hp.head_hidden),    # 896 → 256
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.dropout),                   # 0.5 (proven for AIDO.Cell fusion)
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),  # 256 → 19920
        )

        # Class weights for weighted CE
        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        for name, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators for val/test
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []

    # ------------------------------------------------------------------
    # Stream 1: STRING_GNN Neighborhood Context (from parent, proven K=16)
    # ------------------------------------------------------------------
    def _get_string_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Get K=16 neighborhood-aggregated STRING_GNN embedding.

        Identical logic to parent node1-1-1-1-1 (K=16, attn_dim=64 proven superior).
        """
        B = string_node_idx.shape[0]
        K = self.hparams.topk
        device = self.node_embeddings.device

        known   = string_node_idx >= 0
        unknown = ~known

        out_emb = torch.zeros(B, STRING_DIM, dtype=torch.float32, device=device)

        if known.any():
            known_idx  = string_node_idx[known]                      # [B_k]
            center_emb = self.node_embeddings[known_idx].float()     # [B_k, 256]

            nb_idx = self.topk_neighbors[known_idx]                  # [B_k, K]
            nb_wt  = self.topk_weights[known_idx].float()            # [B_k, K]

            nb_idx_flat = nb_idx.reshape(-1)                         # [B_k * K]
            nb_emb_flat = self.node_embeddings[nb_idx_flat].float()  # [B_k * K, 256]
            neigh_embs  = nb_emb_flat.reshape(known_idx.shape[0], K, STRING_DIM)  # [B_k, K, 256]

            context_emb = self.neighborhood_attn(
                center     = center_emb,
                neigh_embs = neigh_embs,
                ppi_weights = nb_wt,
            )  # [B_k, 256]

            out_emb[known] = context_emb

        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(unknown.sum(), dtype=torch.long, device=device)
            ).float()
            out_emb[unknown] = fb

        return out_emb  # [B, 256]

    # ------------------------------------------------------------------
    # Stream 2: AIDO.Cell-100M Summary Token
    # ------------------------------------------------------------------
    def _get_aido_embeddings(self, cell_input_ids: torch.Tensor) -> torch.Tensor:
        """Get AIDO.Cell-100M perturbation embedding via summary token.

        Args:
            cell_input_ids: [B, 19264] float32 (pre-computed tokenized inputs)
                            perturbed gene = 1.0, all others = -1.0
        Returns:
            [B, 640] float32 perturbation embedding from summary token position 19264
        """
        # AIDO.Cell processes float32 input_ids
        # attention_mask is always overridden to all-ones inside the model
        attention_mask = torch.ones(
            cell_input_ids.shape[0], cell_input_ids.shape[1],
            dtype=torch.long, device=cell_input_ids.device
        )

        # Forward through AIDO.Cell (bf16 internally, cast back to float32)
        outputs = self.aido_model(
            input_ids=cell_input_ids,
            attention_mask=attention_mask,
        )
        # last_hidden_state: [B, 19266, 640]
        # Position 19264 is the first summary token (global context)
        # This captures the transcriptome-level signature for the perturbation
        summary_emb = outputs.last_hidden_state[:, 19264, :].float()  # [B, 640]
        return summary_emb

    def forward(self, string_node_idx: torch.Tensor, cell_input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        # Stream 1: STRING_GNN neighborhood context
        string_emb = self._get_string_embeddings(string_node_idx)   # [B, 256]

        # Stream 2: AIDO.Cell perturbation embedding
        aido_emb   = self._get_aido_embeddings(cell_input_ids)      # [B, 640]

        # Late fusion: concatenate both streams
        fused = torch.cat([string_emb, aido_emb], dim=-1)           # [B, 896]

        # Classification head → [B, 3*G] → [B, 3, G]
        logits = self.head(fused).reshape(-1, N_CLASSES, N_GENES)   # [B, 3, 6640]
        return logits

    # ------------------------------------------------------------------
    # Loss: weighted CE + label smoothing (identical to parent)
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy + mild label smoothing."""
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, 3]
            targets.reshape(-1),                       # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["string_node_idx"], batch["cell_input_ids"])
        loss   = self._loss(logits, batch["labels"])
        # sync_dist=True ensures DDP multi-GPU: all ranks log the same averaged loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False,
                  sync_dist=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"], batch["cell_input_ids"])
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
        logits = self(batch["string_node_idx"], batch["cell_input_ids"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        self._test_pert_ids.extend(batch["pert_id"])
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
                pid = test_df.iloc[i]["pert_id"]
                sym = test_df.iloc[i]["symbol"]
                pred = pred_map[i].float().cpu().numpy().tolist()  # [3, 6640]
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-1-1-1-1-2] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()
        self._test_pert_ids.clear()

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
    # Optimizer: AdamW + linear warmup + CosineAnnealingLR
    # Tuned for AIDO.Cell LoRA (lower lr, longer schedule)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=hp.lr, weight_decay=hp.weight_decay)

        # Phase 1: linear warmup from 0.1×lr to lr over warmup_epochs
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        # Phase 2: CosineAnnealingLR (T_max=200, proven for AIDO.Cell fusion)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=hp.T_max,
            eta_min=1e-6,
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
        description="Node1-1-1-1-1-2 – STRING_GNN K=16 Neighborhood + AIDO.Cell-100M Fusion"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=16)
    parser.add_argument("--global-batch-size",  type=int,   default=128)
    parser.add_argument("--max-epochs",         type=int,   default=300)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--weight-decay",       type=float, default=2e-2)
    parser.add_argument("--head-hidden",        type=int,   default=256,
                        dest="head_hidden")
    parser.add_argument("--dropout",            type=float, default=0.50)
    parser.add_argument("--attn-dim",           type=int,   default=64,
                        dest="attn_dim")
    parser.add_argument("--topk",               type=int,   default=16)
    parser.add_argument("--lora-r",             type=int,   default=8,
                        dest="lora_r")
    parser.add_argument("--lora-alpha",         type=int,   default=16,
                        dest="lora_alpha")
    parser.add_argument("--warmup-epochs",      type=int,   default=10)
    parser.add_argument("--t-max",              type=int,   default=200,
                        dest="t_max")
    parser.add_argument("--label-smoothing",    type=float, default=0.05,
                        dest="label_smoothing")
    parser.add_argument("--patience",           type=int,   default=15)
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--debug-max-step",     type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",       action="store_true",
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

    val_check_interval = int(lim_train) if isinstance(lim_train, int) else 1.0

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule — pre-computes AIDO.Cell tokenized inputs at setup time
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    # Model
    model = FusionModel(
        head_hidden     = args.head_hidden,
        dropout         = args.dropout,
        attn_dim        = args.attn_dim,
        topk            = args.topk,
        lora_r          = args.lora_r,
        lora_alpha      = args.lora_alpha,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        T_max           = args.t_max,
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
        patience  = args.patience,   # 15: AIDO.Cell shows late improvement spikes
        min_delta = 1e-3,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy: use DDP for multi-GPU; single device for fast_dev_run (avoids AMP deadlock)
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
        val_check_interval      = val_check_interval,
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
    print(f"[Node1-1-1-1-1-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
