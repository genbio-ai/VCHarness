"""Node 1-3-2-2: AIDO.Cell-100M LoRA + Frozen STRING_GNN (NeighborhoodAttention K=16) + Concatenation.

Strategy: Fork from the proven best-in-tree architecture (node2-1-1-1-1-1, F1=0.5128) which
uses AIDO.Cell-100M LoRA + STRING_GNN NeighborhoodAttention K=16 + simple concatenation.
The parent lineage (node1-3-2-2 from node1-3-2) is confirmed to be a dead-end at the frozen
STRING-only ceiling (~0.485). The sibling node1-3-2-1 stays on the STRING-only path with
GenePriorBias bug fix (F1=0.4625), which confirms the lineage is suboptimal.

Key design rationale:
1. AIDO.Cell-100M LoRA (r=8): Perturbation-specific transcriptomic context from the
   best available foundation model. The LoRA keeps memory manageable at 3.4 GiB/GPU
   while providing fine-tuning flexibility. This is the primary architecture driver.
2. Frozen STRING_GNN + NeighborhoodAttention (K=16): PPI context supplements the
   cell-level transcriptomic signal with protein network topology.
3. Simple concatenation (640+256→256): Proven in node2-1-1-1-1-1 to work better
   than GatedFusion for LoRA-adapted AIDO.Cell embeddings.
4. No GenePriorBias: node2-1-1-1-1-1 (the best in tree) does NOT use GenePriorBias.
   node4-2-2-2 (AIDO.Cell + GenePriorBias) showed it causes stagnation at epoch 20 by
   locking in suboptimal bias values before the backbone converges.
5. Bilinear gene-class head: logits[b,c,g] = h[b] · gene_class_emb[c,g], the proven
   biologically motivated design throughout the tree.

Memory connections:
- node2-1-1-1-1-1 (F1=0.5128, best in tree): AIDO.Cell LoRA + STRING_GNN K=16.
  Primary source of architecture design. No GenePriorBias. Simple concatenation.
- node2-1-1-1 (F1=0.5059): AIDO.Cell LoRA (r=8) + STRING_GNN K=16, simple concat.
  Proven lr=1e-4, weight_decay=1e-2, head_dropout=0.3, label_smoothing=0.05.
- node4-2-2-2 (F1=0.4936): AIDO.Cell LoRA + STRING_GNN + concat + GenePriorBias.
  GenePriorBias with warmup=20 caused stagnation at epoch 20 → confirmed to skip.
- node1-3-2-1 (F1=0.4625, sibling): STRING-only + GenePriorBias (bug-fixed).
  Confirms STRING-only ceiling; discriminative LR parent contamination still hurts.
- node1-3-2 (F1=0.3821, parent): GenePriorBias test-inference bug confirmed catastrophic.

Hyperparameter choices (aligned with node2 proven recipe):
- lr=1e-4 for AIDO.Cell LoRA and head (conservative for transformer fine-tuning)
- lr_string=3e-4 for STRING_GNN head (higher LR for the small PPI attention module)
- weight_decay=1e-2 (node2-1-1-1-1-1 proven; lighter than STRING-only 3e-2)
- head_dropout=0.3 (node2-1-1-1 proven)
- label_smoothing=0.05 (proven throughout tree)
- T_max=150, eta_min=5e-6 (generous cosine schedule)
- patience=15, min_delta=5e-4 (avoid premature stopping)
- gradient_checkpointing=True (required for AIDO.Cell-100M memory)
- micro_batch_size=4 (AIDO.Cell-100M requires small batches per GPU)
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3
# Class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

AIDO_CELL_DIR = Path("/home/Models/AIDO.Cell-100M")
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

# AIDO.Cell hidden size for 100M model
AIDO_HIDDEN = 640
STRING_EMB_DIM = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights; neutral class stays close to 1."""
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
    y_hat = preds.argmax(dim=1)  # [N, G]
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
        neighbor_indices [N, K] long — STRING_GNN node indices of top-K neighbors (-1=padding)
        neighbor_weights [N, K] float — normalized STRING confidence weights
    """
    N = emb.shape[0]
    graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False)
    edge_index = graph["edge_index"]  # [2, E]
    ew = graph.get("edge_weight", None)

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
    mask = neighbor_indices >= 0  # [N, K]
    raw = neighbor_weights.clone()
    raw[~mask] = -1e9
    norm_w = torch.softmax(raw, dim=-1)  # [N, K]
    norm_w[~mask] = 0.0

    return neighbor_indices, norm_w


# ---------------------------------------------------------------------------
# Neighborhood Attention Aggregator (proven K=16, attn_dim=64)
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
        gate = torch.sigmoid(self.gate_proj(center_emb))  # [B, D]
        return center_emb + gate * aggregated              # [B, D]


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
            item["labels"] = self.labels[idx]  # [G] in {0,1,2}
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
    def __init__(self, batch_size: int = 4, num_workers: int = 4) -> None:
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
# Lightning Model: AIDO.Cell-100M LoRA + STRING_GNN Neighborhood Attention
# ---------------------------------------------------------------------------
class AIDOCellStringFusion(pl.LightningModule):
    """AIDO.Cell-100M LoRA + Frozen STRING_GNN Neighborhood Attention + Bilinear Head.

    Architecture:
        pert_id
            |
        AIDO.Cell-100M (LoRA r=8, 0.55M trainable params) → summary token [B, 640]
            |
        STRING_GNN (FROZEN, pre-computed buffer [18870, 256])
        NeighborhoodAttentionAggregator (K=16, attn_dim=64) → [B, 256]
            |
        Concatenation: [B, 640+256=896] → proj [B, 256]
            |
        LayerNorm(256) → Dropout
            |
        Bilinear: h[B,256] dot gene_class_emb[3,6640,256] → logits[B,3,6640]
            |
        Output: [B, 3, 6640] → softmax probabilities

    Key design choices:
    - AIDO.Cell mean-pool over all 19264 gene positions → [B, 640] cell embedding
    - Frozen STRING_GNN embeddings + NeighborhoodAttention: lightweight PPI context
    - Simple concatenation+projection (not GatedFusion): proven in node2-1-1-1-1-1
    - No GenePriorBias: node2 lineage best (F1=0.5128) did not use it
    - Bilinear head: biologically motivated factorization (node1/node2 lineages)
    """

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bilinear_dim: int = 256,
        fusion_hidden: int = 256,
        attn_dim: int = 64,
        K: int = 16,
        dropout: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_epochs: int = 10,
        t_max: int = 150,
        eta_min: float = 5e-6,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        # ----- AIDO.Cell-100M with LoRA -----
        # Rank 0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(AIDO_CELL_DIR), trust_remote_code=True
        )

        aido_model = AutoModel.from_pretrained(
            str(AIDO_CELL_DIR), trust_remote_code=True,
            dtype=torch.bfloat16,  # bf16 ensures FlashAttention is used
            # (model code: use_flash = ln_outputs.dtype in {fp16, bf16};
            #  LayerNorm output is float32 under autocast, but is cast back to
            #  query.weight.dtype before the attention; loading in bf16 makes that
            #  cast go to bf16 → FlashAttention fires → avoids 50+ GiB OOM)
        )
        aido_model.config.use_cache = False
        # NOTE: Do NOT call gradient_checkpointing_enable() on the base model BEFORE
        # get_peft_model().  PEFT detects the GC flag and calls
        # enable_input_require_grads() → get_input_embeddings(), which raises
        # NotImplementedError in AIDO.Cell.  We enable GC on the PEFT model AFTER
        # wrapping, using a no-op patch to bypass the incompatible transformers hook.

        # Apply LoRA to Q/K/V projections
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=["query", "key", "value"],
        )
        self.aido_cell = get_peft_model(aido_model, lora_cfg)

        # Enable gradient checkpointing AFTER LoRA.  Patch enable_input_require_grads
        # to a no-op so the transformers GC path does not call get_input_embeddings()
        # (unimplemented in AIDO.Cell).  With use_reentrant=False, PyTorch's checkpoint
        # correctly propagates gradients to LoRA params without this hook.
        self.aido_cell.base_model.model.enable_input_require_grads = lambda: None
        self.aido_cell.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA params to float32 for stable optimization
        for name, param in self.aido_cell.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ----- STRING_GNN: precompute embeddings -----
        # Collect all pert_ids for index mapping
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

        self.print("Precomputing STRING_GNN embeddings...")
        string_emb, pert_to_gnn_idx = precompute_string_gnn_embeddings()
        self.register_buffer("node_embeddings", string_emb)  # [18870, 256]

        gnn_idx_tensor = torch.tensor(
            [pert_to_gnn_idx.get(pid, -1) for pid in unique_sorted], dtype=torch.long
        )
        self.register_buffer("pert_gnn_idx", gnn_idx_tensor)  # [M]

        self.print(f"Precomputing PPI neighborhood tables (K={hp.K})...")
        nb_indices, nb_weights = precompute_neighborhood(string_emb, K=hp.K)
        self.register_buffer("neighbor_indices", nb_indices)  # [18870, K]
        self.register_buffer("neighbor_weights", nb_weights)  # [18870, K]

        # Fallback embedding for pert_ids not in STRING
        self.fallback_emb = nn.Parameter(torch.zeros(1, STRING_EMB_DIM))

        # ----- Trainable modules -----
        self.neighborhood_attn = NeighborhoodAttentionAggregator(
            emb_dim=STRING_EMB_DIM, attn_dim=hp.attn_dim
        )

        # Fusion: AIDO (640) + STRING_GNN (256) → bilinear_dim (256)
        self.fusion_proj = nn.Sequential(
            nn.Linear(AIDO_HIDDEN + STRING_EMB_DIM, hp.fusion_hidden),
            nn.GELU(),
        )

        # Normalization before bilinear projection
        self.proj_norm = nn.LayerNorm(hp.fusion_hidden)
        self.proj_dropout = nn.Dropout(hp.dropout)

        # Bilinear gene-class embedding: logits[b,c,g] = h[b] · gene_class_emb[c,g]
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        self.register_buffer("class_weights", get_class_weights())

        # Cast trainable parameters to float32 for stable optimization
        for name, v in self.named_parameters():
            if v.requires_grad and "aido_cell" not in name:
                v.data = v.data.float()

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor] = []
        self._val_tgts:  List[torch.Tensor] = []
        self._val_idx:   List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    def _get_aido_embedding(self, pert_ids: List[str]) -> torch.Tensor:
        """Get AIDO.Cell cell embedding by tokenizing pert_ids as gene IDs.

        For DEG prediction, the input is the perturbed gene ID. We create
        a pseudo-cell expression vector where only the perturbed gene has
        expression value 1.0 and all others are missing (-1.0, treated as
        0-count by AIDO.Cell internally). This provides perturbation-specific
        context from the foundation model.

        Returns: [B, AIDO_HIDDEN] float32
        """
        # Tokenize: each pert_id treated as the single expressed gene
        # AIDO.Cell fills missing genes with -1.0 by default
        cell_inputs = []
        for pid in pert_ids:
            cell_inputs.append({
                "gene_ids": [pid],
                "expression": [1.0],
            })

        tokenized = self.tokenizer(cell_inputs, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)           # [B, 19264] float32
        attention_mask = tokenized["attention_mask"].to(self.device) # [B, 19264] int64

        # Forward through AIDO.Cell
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            outputs = self.aido_cell(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        # Mean pool over gene positions (exclude 2 summary tokens)
        # last_hidden_state: [B, 19266, 640]
        cell_emb = outputs.last_hidden_state[:, :19264, :].mean(dim=1)  # [B, 640]
        return cell_emb.float()

    def _get_string_neighborhood_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Return [B, 256] neighborhood-attention-aggregated STRING embeddings."""
        pos = torch.tensor(
            [self.pert_to_pos[pid] for pid in pert_ids], dtype=torch.long, device=self.device
        )
        gnn_node_idx = self.pert_gnn_idx[pos]   # [B]
        valid_center = gnn_node_idx >= 0
        safe_center_idx = gnn_node_idx.clamp(min=0)
        center_emb_raw = self.node_embeddings[safe_center_idx]  # [B, 256]
        fallback = self.fallback_emb.expand(center_emb_raw.shape[0], -1).to(center_emb_raw.dtype)
        center_emb = torch.where(valid_center.unsqueeze(-1), center_emb_raw, fallback).float()

        K = self.hparams.K
        nb_idx = self.neighbor_indices[safe_center_idx]   # [B, K]
        nb_wts = self.neighbor_weights[safe_center_idx]   # [B, K]
        valid_mask = nb_idx >= 0                           # [B, K] bool

        safe_nb_idx = nb_idx.clamp(min=0)                # [B, K]
        nb_emb = self.node_embeddings[safe_nb_idx].float()  # [B, K, 256]
        nb_emb = nb_emb * valid_mask.unsqueeze(-1).float()

        aggregated = self.neighborhood_attn(
            center_emb, nb_emb, nb_wts, valid_mask
        )  # [B, 256]
        return aggregated

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        # Get AIDO.Cell embedding: [B, 640]
        aido_emb = self._get_aido_embedding(pert_ids)

        # Get STRING_GNN neighborhood embedding: [B, 256]
        string_emb = self._get_string_neighborhood_emb(pert_ids)

        # Concatenation + projection: [B, 896] → [B, 256]
        fused = self.fusion_proj(torch.cat([aido_emb, string_emb], dim=-1))  # [B, 256]

        # Normalization + dropout
        h = self.proj_norm(fused)
        h = self.proj_dropout(h)

        # Bilinear head: [B, 256] × [3, 6640, 256] → [B, 3, 6640]
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)
        return logits

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        loss = F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, 3]
            targets.reshape(-1),                       # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )
        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["pert_id"])
        loss = self._loss(logits, batch["labels"])
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
        local_preds = torch.cat(self._val_preds, dim=0)  # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  dim=0)  # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   dim=0)  # [N_local]
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds)  # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)   # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)    # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

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
        local_preds = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)  # [N_local]
        all_preds   = self.all_gather(local_preds)          # [W, N_local, 3, G]
        all_idx     = self.all_gather(local_idx)            # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            order  = torch.argsort(idx_flat)
            s_idx  = idx_flat[order]
            s_pred = preds_flat[order]
            mask   = torch.cat([torch.ones(1, dtype=torch.bool, device=s_idx.device),
                                s_idx[1:] != s_idx[:-1]])
            preds_dedup = s_pred[mask]     # [N_test, 3, G]
            unique_sid  = s_idx[mask].tolist()

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                           for i in range(len(test_df))}

            rows = []
            for sid in unique_sid:
                pid, sym = idx_to_meta[int(sid)]
                dedup_pos = (s_idx[mask] == sid).nonzero(as_tuple=True)[0][0].item()
                pred_list = preds_dedup[dedup_pos].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-3-2-2] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_idx.clear()

    # ---- Checkpoint helpers (save only trainable params) ----
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

    # ---- Optimizer ----
    def configure_optimizers(self):
        hp = self.hparams

        # All trainable parameters at the same learning rate
        # (AIDO.Cell LoRA + fusion head + neighborhood attn + bilinear head)
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=hp.lr,
            weight_decay=hp.weight_decay,
        )

        # Linear warmup then cosine annealing
        warmup_sch = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=hp.t_max,
            eta_min=hp.eta_min,
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
        description="Node1-3-2-2: AIDO.Cell-100M LoRA + STRING_GNN Neighborhood Attention"
    )
    parser.add_argument("--micro-batch-size",  type=int,   default=4)
    parser.add_argument("--global-batch-size", type=int,   default=64)
    parser.add_argument("--max-epochs",        type=int,   default=200)
    parser.add_argument("--lora-r",            type=int,   default=8)
    parser.add_argument("--lora-alpha",        type=int,   default=16)
    parser.add_argument("--lora-dropout",      type=float, default=0.05)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--weight-decay",      type=float, default=1e-2)
    parser.add_argument("--dropout",           type=float, default=0.3)
    parser.add_argument("--bilinear-dim",      type=int,   default=256)
    parser.add_argument("--fusion-hidden",     type=int,   default=256)
    parser.add_argument("--attn-dim",          type=int,   default=64)
    parser.add_argument("--k-neighbors",       type=int,   default=16)
    parser.add_argument("--warmup-epochs",     type=int,   default=10)
    parser.add_argument("--t-max",             type=int,   default=150)
    parser.add_argument("--eta-min",           type=float, default=5e-6)
    parser.add_argument("--label-smoothing",   type=float, default=0.05)
    parser.add_argument("--patience",          type=int,   default=15)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--val-check-interval",type=float, default=1.0,
                        dest="val_check_interval")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0,
                        dest="gradient_clip_val")
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
        lim_test  = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        lim_test  = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    if args.debug_max_step is not None or fast_dev_run:
        val_check_interval = 1.0
    else:
        val_check_interval = args.val_check_interval

    # DataModule + Model
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    # Do NOT call dm.setup() manually — the trainer handles setup() internally

    model = AIDOCellStringFusion(
        lora_r          = args.lora_r,
        lora_alpha      = args.lora_alpha,
        lora_dropout    = args.lora_dropout,
        bilinear_dim    = args.bilinear_dim,
        fusion_hidden   = args.fusion_hidden,
        attn_dim        = args.attn_dim,
        K               = args.k_neighbors,
        dropout         = args.dropout,
        lr              = args.lr,
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
        auto_insert_metric_name = False,
    )
    es_cb = EarlyStopping(
        monitor="val/f1", mode="max", patience=args.patience, min_delta=5e-4
    )
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
        val_check_interval      = val_check_interval,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = args.gradient_clip_val,
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
    print(f"[Node1-3-2-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
