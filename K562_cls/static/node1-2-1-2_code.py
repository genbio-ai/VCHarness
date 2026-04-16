"""Node 1-2-1-2: AIDO.Cell-100M (LoRA r=8) + Frozen STRING_GNN (K=16 Neighborhood Attention)
                + Extended Training (300 epochs).

This node implements the best-performing architecture in the entire MCTS tree
(node2-1-1-1, F1=0.5059) with the single most impactful improvement: extended training.
node2-1-1-1 only ran 63 epochs without triggering early stopping — val/f1 was still
climbing monotonically from 0.21→0.51 at termination with no signs of plateau.
Extending to 300 epochs with patience=15 is expected to allow full convergence.

Architecture (proven from node2-1-1-1, test F1=0.5059 — best in MCTS tree):
  AIDO.Cell-100M (LoRA r=8, α=16 on Q/K/V) → summary token [B, 640]
      | 1-hot single-gene perturbation input: pert_id=1.0, all others=-1.0 (missing)
      | Uses summary token extraction (mean of appended positions 19264-19265)
  STRING_GNN (frozen, precomputed static embeddings)
      → K=16 neighborhood attention aggregation → [B, 256]
      | Frozen backbone avoids noisy gradient updates on small dataset (1388 samples)
      | K=16 + attn_dim=64: proven best config for PPI neighborhood aggregation
  Concatenation fusion: [640 + 256] = 896-dim
  2-layer MLP head: LayerNorm(896) → Linear(896→256) → GELU → Dropout(0.4) → Linear(256→19920)
      → reshape to [B, 3, 6640] for per-gene 3-class logits
  Loss: Weighted cross-entropy (sqrt-inverse-freq) + label smoothing ε=0.05

Key improvements over node2-1-1-1 (F1=0.5059):
  1. Extended training: max_epochs=300, patience=15 (vs 63 epochs, no stopping, in node2-1-1-1)
  2. Lower eta_min: 5e-7 (vs 1e-6) for deeper late-stage cosine exploration
  3. Slightly reduced head dropout: 0.4 (vs 0.5) — per node2-1-1-1 feedback recommendation
  4. Gradient checkpointing enabled for AIDO.Cell memory efficiency

Distinct from sibling node1-2-1-1 (frozen STRING_GNN + frozen scFoundation + gated fusion):
  - Uses AIDO.Cell-100M with live LoRA fine-tuning (not frozen scFoundation, 768-dim)
  - Simple concatenation fusion (not gated modality weighting)
  - 2-layer MLP head (not bilinear gene-class embedding)
  - Different perturbation encoding: AIDO.Cell with 1-hot input gives perturbation-specific
    transcriptional context from 50M human single-cell pretraining

Memory connections:
  - node2-1-1-1 (F1=0.5059): exact architecture replicated; training extended
  - node2-1-1-1 feedback: "#1 recommendation: Extend training to 300 epochs (free lunch)"
  - node1-2-1 feedback: "DLR failed — frozen backbone is strictly better for STRING_GNN"
  - node1-2-1-1 doc: Sibling is exploring scFoundation path; this node explores AIDO.Cell path
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
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES    = 6640
N_CLASSES  = 3
AIDO_DIM   = 640   # AIDO.Cell-100M hidden dimension
STRING_DIM = 256   # STRING_GNN hidden dimension
FUSION_DIM = AIDO_DIM + STRING_DIM  # 896

# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

AIDO_CELL_DIR = Path("/home/Models/AIDO.Cell-100M")
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


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
        neighbor_weights: [n_nodes, K] float — corresponding STRING edge confidence scores
    """
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


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic exactly.

    For each gene, F1 is computed only over the classes that are actually present
    in the true labels for that gene (zero_division=0 for absent classes).
    The final metric is the mean of per-gene F1 scores.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0, 1, 2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)  # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)            # [N, G]
        is_pred = (y_hat == c)              # [N, G]
        present = is_true.any(dim=0)        # [G] — True only if class c exists in true labels

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
        # Zero out F1 for absent classes (zero_division=0) — only contribute to present classes
        f1_c = f1_c * present.float()
        f1_per_gene += f1_c

    # Per-gene: count how many of the 3 classes are actually present in true labels for each gene
    # Result has shape [G] — one count per gene (matching f1_per_gene shape)
    n_present = sum((targets == c).any(dim=0).float() for c in range(3))  # [G]
    f1_per_gene = f1_per_gene / n_present.clamp(min=1)  # [G] / [G] → [G]
    return f1_per_gene.mean().item()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    """K562 DEG prediction dataset with pre-tokenized AIDO.Cell inputs."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_map: Dict[str, int],
        aido_input_ids: torch.Tensor,   # [N, 19264] float32
        aido_attn_mask: torch.Tensor,   # [N, 19264] int64
    ) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()

        # STRING_GNN node index for each sample (-1 means not in STRING)
        self.string_node_indices = torch.tensor(
            [string_map.get(p, -1) for p in self.pert_ids], dtype=torch.long
        )

        # Pre-tokenized AIDO.Cell inputs (1-hot single-gene representation)
        self.aido_input_ids = aido_input_ids  # [N, 19264] float32
        self.aido_attn_mask = aido_attn_mask  # [N, 19264] int64

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
            "aido_input_ids":  self.aido_input_ids[idx],   # [19264] float32
            "aido_attn_mask":  self.aido_attn_mask[idx],   # [19264] int64
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
        "aido_input_ids":  torch.stack([b["aido_input_ids"]  for b in batch]),  # [B, 19264]
        "aido_attn_mask":  torch.stack([b["aido_attn_mask"]  for b in batch]),  # [B, 19264]
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
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self._is_setup    = False

    def setup(self, stage: Optional[str] = None) -> None:
        if self._is_setup:
            return
        self._is_setup = True

        # ------------------------------------------------------------------
        # Load STRING_GNN gene-to-index mapping
        # ------------------------------------------------------------------
        string_map = load_string_gnn_mapping()

        # ------------------------------------------------------------------
        # Load AIDO.Cell tokenizer (local model, no download needed)
        # Rank 0 loads first for safety; all ranks then load
        # ------------------------------------------------------------------
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        tokenizer = AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)

        # ------------------------------------------------------------------
        # Load data splits
        # ------------------------------------------------------------------
        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        # ------------------------------------------------------------------
        # Pre-tokenize AIDO.Cell inputs using 1-hot single-gene representation:
        #   - Perturbed gene (pert_id): expression = 1.0
        #   - All other genes: -1.0 (missing → treated as 0-count inside AIDO.Cell)
        # After AIDO.Cell's _prepare_inputs():
        #   - Perturbed gene value: log1p(1/1 * 10000) ≈ 9.21 (distinctive high expression)
        #   - Other genes: log1p(0) = 0.0
        # This creates a unique single-gene "profile" for each perturbation.
        # Since inputs are static per pert_id, pre-tokenizing saves compute per step.
        # Note: LoRA parameters are still updated at each training step via AIDO.Cell forward.
        # ------------------------------------------------------------------
        def tokenize_pert_ids(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
            pert_ids = df["pert_id"].tolist()
            expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
            tokenized = tokenizer(expr_dicts, return_tensors="pt")
            return (
                tokenized["input_ids"].float(),       # [N, 19264] float32
                tokenized["attention_mask"].long(),   # [N, 19264] int64
            )

        print("Pre-tokenizing AIDO.Cell inputs for train/val/test splits...")
        train_inp_ids, train_attn = tokenize_pert_ids(train_df)
        val_inp_ids,   val_attn   = tokenize_pert_ids(val_df)
        test_inp_ids,  test_attn  = tokenize_pert_ids(test_df)
        print("  Done. Train:", train_inp_ids.shape, " Val:", val_inp_ids.shape,
              " Test:", test_inp_ids.shape)

        # ------------------------------------------------------------------
        # Create datasets
        # ------------------------------------------------------------------
        self.train_ds = DEGDataset(train_df, string_map, train_inp_ids, train_attn)
        self.val_ds   = DEGDataset(val_df,   string_map, val_inp_ids,   val_attn)
        self.test_ds  = DEGDataset(test_df,  string_map, test_inp_ids,  test_attn)

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
# Neighborhood Attention Module (proven from node1 lineage, K=16, attn_dim=64)
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Center-context gated attention over top-K PPI neighbors.

    For each perturbed gene, aggregates the top-K neighbors from the STRING PPI
    graph using learned attention scores gated by the edge confidence weights.

    This module was the key addition in node2-1-1-1 that pushed AIDO.Cell from
    F1=0.4535 (raw concat, node2-1-1) to F1=0.5059 (K=16 neighborhood, node2-1-1-1).

    Architecture:
        attn_proj: [center(256) + neighbor(256)] → attn_dim(64) → score(1)
        attention = softmax(edge_weight + attn_proj_score)   # [B, K]
        aggregated = attention @ neighbor_emb                # [B, 256]
        gate = sigmoid(gate_proj(center_emb))                # [B, 256]
        output = center_emb + gate * aggregated              # [B, 256]
    """

    def __init__(
        self, embed_dim: int = 256, attn_dim: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim  = attn_dim

        # Attention projection: [center(256) + neighbor(256)] → attn_dim → 1
        self.attn_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, attn_dim),
            nn.GELU(),
            nn.Linear(attn_dim, 1),
        )
        # Gating: center embedding → gate vector
        self.gate_proj     = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout  = nn.Dropout(dropout)

    def forward(
        self,
        center_emb: torch.Tensor,       # [B, D]
        neighbor_emb: torch.Tensor,     # [B, K, D]
        neighbor_weights: torch.Tensor, # [B, K] STRING edge confidence (0–1)
        neighbor_mask: torch.Tensor,    # [B, K] bool: True = valid neighbor
    ) -> torch.Tensor:
        """Returns aggregated embedding [B, D]."""
        B, K, D = neighbor_emb.shape

        center_expanded = center_emb.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]
        pair_features   = torch.cat([center_expanded, neighbor_emb], dim=-1)  # [B, K, 2D]

        attn_scores  = self.attn_proj(pair_features).squeeze(-1)  # [B, K]
        attn_scores  = attn_scores + neighbor_weights              # add STRING prior
        attn_scores  = attn_scores.masked_fill(~neighbor_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)          # [B, K]
        attn_weights = self.attn_dropout(attn_weights)

        aggregated = torch.bmm(attn_weights.unsqueeze(1), neighbor_emb).squeeze(1)  # [B, D]
        gate   = torch.sigmoid(self.gate_proj(center_emb))  # [B, D]
        output = center_emb + gate * aggregated              # [B, D]
        return output


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------
class AIDOCellStringFusionModel(pl.LightningModule):
    """AIDO.Cell-100M (LoRA r=8) + Frozen STRING_GNN (K=16 neighborhood attn) + Extended Training.

    Architecture proven from node2-1-1-1 (F1=0.5059 — best in MCTS tree).
    Primary improvement: extended training from 63 to 300 epochs for full convergence.
    """

    def __init__(
        self,
        K: int = 16,
        attn_dim: int = 64,
        head_hidden: int = 256,
        dropout: float = 0.4,
        lr: float = 1e-4,
        weight_decay: float = 2e-2,
        warmup_epochs: int = 10,
        t_max: int = 290,
        eta_min: float = 5e-7,
        label_smoothing: float = 0.05,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Model components initialized in setup()

    def setup(self, stage: Optional[str] = None) -> None:
        # Guard against repeated setup calls
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True
        hp = self.hparams

        # ------------------------------------------------------------------
        # 1. Load AIDO.Cell-100M and apply LoRA (r=8, α=16 on Q/K/V)
        #    ~0.55M trainable LoRA params provide perturbation-specific context
        # ------------------------------------------------------------------
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Enable FlashAttention-2 for memory-efficient O(N) attention with long sequences.
        # The AIDO.Cell model uses seq_len=19266 (19264 genes + 2 summary tokens).
        # Standard HuggingFace attention materializes [B, H, T, T] ≈ 88.5 GB for B=32,
        # causing OOM on 80 GB GPUs. FlashAttention-2 uses cuSeqlen and avoids this.
        from transformers import AutoConfig
        aido_cfg = AutoConfig.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        aido_cfg.flash_attn_2 = True

        # Rank-0 triggers the download (local model, instant) then all ranks sync.
        if local_rank == 0:
            AutoModel.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True, config=aido_cfg)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        # Single model load per rank — no redundant copies.
        aido_base = AutoModel.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True, config=aido_cfg)
        aido_base.config.use_cache = False
        # Force bf16 computation for all layers. This is CRITICAL:
        # The CellFoundationAttention checks ln_outputs.dtype in (fp16, bf16) before deciding
        # whether to use HuggingFace BertSelfFlashAttention. With Lightning's bf16-mixed precision,
        # LayerNorm outputs may remain fp32, causing flash_self to be skipped and standard
        # attention to OOM (materializes [B,H,T,T] = 27 GiB for T=19266).
        # Setting model to bf16 ensures ln_outputs is bf16 and FlashAttention path is taken.
        aido_base = aido_base.to(device="cuda", dtype=torch.bfloat16)

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=["query", "key", "value"],
            # flash_self shares weight tensors with self, LoRA applies to both automatically
        )
        self.aido_model = get_peft_model(aido_base, lora_cfg)
        # Enable gradient checkpointing for memory efficiency.
        # With flash_attn_2=True, the extended_attention_mask is NOT used in the
        # attention computation (FlashAttention uses cuSeqlen params instead), so
        # recomputing it on backward pass (use_reentrant=False) does NOT cause OOM.
        self.aido_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("AIDO.Cell-100M LoRA trainable params:")
        self.aido_model.print_trainable_parameters()

        # ------------------------------------------------------------------
        # 2. Load STRING_GNN and precompute frozen static embeddings
        #    Frozen backbone: node1 lineage proven that frozen > DLR for small dataset
        # ------------------------------------------------------------------
        print("Loading STRING_GNN for static embedding precomputation...")
        string_model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        string_model.eval()
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index = graph["edge_index"].long()
        edge_weight = graph["edge_weight"].float()

        with torch.no_grad():
            out = string_model(edge_index=edge_index, edge_weight=edge_weight)
            static_emb = out.last_hidden_state.float()  # [18870, 256]
        self.register_buffer("_static_node_emb", static_emb)  # non-trainable buffer
        n_nodes = static_emb.shape[0]  # 18870
        del string_model  # free memory
        print(f"STRING_GNN static embeddings cached: {static_emb.shape}")

        # Pre-compute K=16 PPI neighbor topology (graph-level, fixed per model)
        print(f"Pre-computing top-{hp.K} PPI neighbors for {n_nodes} nodes...")
        nbr_idx, nbr_wgt = precompute_neighborhood(edge_index, edge_weight, n_nodes, K=hp.K)
        self.register_buffer("neighbor_indices", nbr_idx)  # [n_nodes, K]
        self.register_buffer("neighbor_weights", nbr_wgt)  # [n_nodes, K]

        # ------------------------------------------------------------------
        # 3. Neighborhood Attention Aggregator
        #    K=16, attn_dim=64: proven best config from node1-1-1-1-1 and node2-1-1-1
        # ------------------------------------------------------------------
        self.neighborhood_attn = NeighborhoodAttentionAggregator(
            embed_dim=STRING_DIM,
            attn_dim=hp.attn_dim,
            dropout=0.0,
        )

        # ------------------------------------------------------------------
        # 4. Learnable fallback embedding for genes not in STRING vocabulary
        # ------------------------------------------------------------------
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ------------------------------------------------------------------
        # 5. 2-layer MLP classification head (proven from node2-1-1-1)
        #    896 → 256 → GELU → Dropout(0.4) → 19920
        #    Dropout reduced from 0.5→0.4 per node2-1-1-1 feedback recommendation
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),  # 3 * 6640 = 19920
        )

        # Class weights for weighted cross-entropy
        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        # (AIDO.Cell LoRA params need float32 even under bf16-mixed precision)
        for _, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Validation / test accumulators (cleared each epoch)
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_tgts: List[torch.Tensor] = []
        self._test_idx:  List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # STRING embedding with K=16 PPI neighborhood attention
    # ------------------------------------------------------------------
    def _get_string_emb(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Get STRING_GNN embeddings enriched by K=16 neighborhood attention.

        Args:
            string_node_idx: [B] long — STRING node index, -1 for unknown genes
        Returns:
            [B, 256] float32 — STRING-PPI enriched perturbation embeddings
        """
        B        = string_node_idx.shape[0]
        node_emb = self._static_node_emb  # [18870, 256], on GPU
        emb      = torch.zeros(B, STRING_DIM, dtype=node_emb.dtype, device=node_emb.device)

        known   = string_node_idx >= 0
        unknown = ~known

        if known.any():
            known_idx = string_node_idx[known]     # [K_known]
            center    = node_emb[known_idx]        # [K_known, 256]
            nbr_idx   = self.neighbor_indices[known_idx]  # [K_known, K]
            nbr_wgt   = self.neighbor_weights[known_idx]  # [K_known, K]
            nbr_mask  = nbr_idx >= 0               # [K_known, K] validity mask

            # Clamp for safe embedding lookup (padding slots → 0, masked out later)
            nbr_idx_clamped = nbr_idx.clamp(min=0)
            n_known         = int(known.sum().item())
            K_neighbors     = nbr_idx.shape[1]
            flat_nbr_idx    = nbr_idx_clamped.view(-1)              # [K_known * K]
            flat_nbr_emb    = node_emb[flat_nbr_idx]                # [K_known * K, 256]
            neighbor_emb    = flat_nbr_emb.view(n_known, K_neighbors, STRING_DIM)  # [K_known, K, 256]

            # Zero out padding entries (invalid neighbors)
            neighbor_emb = neighbor_emb * nbr_mask.unsqueeze(-1).float()

            # Apply neighborhood attention aggregation
            aggregated = self.neighborhood_attn(
                center_emb       = center.float(),
                neighbor_emb     = neighbor_emb.float(),
                neighbor_weights = nbr_wgt.float(),
                neighbor_mask    = nbr_mask,
            )  # [K_known, 256]
            emb[known] = aggregated

        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(unknown.sum(), dtype=torch.long, device=node_emb.device)
            ).to(node_emb.dtype)
            emb[unknown] = fb

        return emb.float()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        aido_input_ids:  torch.Tensor,   # [B, 19264] float32 — raw expression values
        aido_attn_mask:  torch.Tensor,   # [B, 19264] int64   — unused; AIDO.Cell overrides to all-ones
        string_node_idx: torch.Tensor,   # [B] long           — STRING node index
    ) -> torch.Tensor:
        """Return logits of shape [B, 3, G]."""
        # 1. AIDO.Cell forward with LoRA — produces [B, 19266, 640]
        #    The +2 positions are the summary positions appended by _prepare_inputs()
        #    NOTE: Do NOT pass attention_mask — AIDO.Cell always overrides it to all-ones
        #    internally, and passing a [B, 19264] mask causes HuggingFace to expand it to
        #    [B, 1, T, T] = [32, 1, 19264, 19264] ≈ 88.5 GB, triggering OOM on 80 GB GPUs.
        #    Passing None avoids the expansion while achieving the same all-ones behavior.
        aido_out = self.aido_model(
            input_ids=aido_input_ids,
            attention_mask=None,
        )
        # Extract summary token: mean of the 2 appended summary positions (19264, 19265)
        # These positions accumulate perturbation-specific context through self-attention
        # (analogous to BERT's [CLS] token — proven by node2-1 to outperform mean-pool)
        aido_emb = aido_out.last_hidden_state[:, 19264:, :].mean(dim=1).float()  # [B, 640]

        # 2. STRING_GNN frozen embeddings with K=16 neighborhood attention
        string_emb = self._get_string_emb(string_node_idx)  # [B, 256]

        # 3. Concatenation fusion: [640 + 256] = 896-dim
        fused = torch.cat([aido_emb, string_emb], dim=-1)   # [B, 896]

        # 4. 2-layer MLP head → [B, 3, G]
        logits_flat = self.head(fused)                        # [B, 19920]
        logits = logits_flat.view(-1, N_CLASSES, N_GENES)    # [B, 3, 6640]
        return logits

    # ------------------------------------------------------------------
    # Loss: weighted cross-entropy + label smoothing (proven recipe)
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
        logits = self(batch["aido_input_ids"], batch["aido_attn_mask"], batch["string_node_idx"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["aido_input_ids"], batch["aido_attn_mask"], batch["string_node_idx"])
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

        # Gather across all DDP ranks
        all_preds = self.all_gather(local_preds)  # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)   # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)    # [W, N_local]

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
        logits = self(batch["aido_input_ids"], batch["aido_attn_mask"], batch["string_node_idx"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            self._test_tgts.append(batch["labels"].detach())
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)  # [N_local]
        local_tgts  = torch.cat(self._test_tgts,  dim=0) if self._test_tgts else None
        self._test_preds.clear(); self._test_tgts.clear(); self._test_idx.clear()

        all_preds = self.all_gather(local_preds)  # [W, N_local, 3, G]
        all_idx   = self.all_gather(local_idx)    # [W, N_local]

        if local_tgts is not None:
            all_tgts = self.all_gather(local_tgts)
        else:
            all_tgts = None

        all_preds = self.all_gather(local_preds)  # [W, N_local, 3, G]
        all_idx   = self.all_gather(local_idx)    # [W, N_local]
        all_tgts  = self.all_gather(local_tgts) if local_tgts is not None else None

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            # De-duplicate
            order  = torch.argsort(idx_flat)
            s_idx  = idx_flat[order]
            s_pred = preds_flat[order]
            mask   = torch.cat([
                torch.tensor([True], device=s_idx.device),
                s_idx[1:] != s_idx[:-1],
            ])
            preds_dedup = s_pred[mask]
            unique_sid  = s_idx[mask].tolist()

            # Compute test F1 if targets are available
            if all_tgts is not None:
                tgts_flat = all_tgts.view(-1, N_GENES)
                s_tgt     = tgts_flat[order]
                tgts_dedup = s_tgt[mask]
                test_f1   = compute_per_gene_f1(preds_dedup, tgts_dedup)
                self.log("test/f1", test_f1, sync_dist=False)  # Only log on rank 0 for file writing

            test_df = pd.read_csv(TEST_TSV, sep="\t")
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
            # Only rank 0 writes predictions to avoid duplicate outputs
            if self.trainer.is_global_zero:
                pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
                print(f"[Node1-2-1-2] Saved {len(rows)} test predictions to {out_dir}/test_predictions.tsv")

    # ------------------------------------------------------------------
    # Checkpoint helpers — save only trainable params (LoRA + head + attn) + buffers
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
            f"Checkpoint: {train}/{total} trainable params ({100 * train / total:.2f}%), "
            f"plus {bufs} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: AdamW + warmup(10) + CosineAnnealingLR(T_max=290, eta_min=5e-7)
    # All trainable params at same LR (LoRA + neighborhood_attn + head + fallback)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable_params, lr=hp.lr, weight_decay=hp.weight_decay)

        # Phase 1: linear warmup from 0.1×lr to lr over warmup_epochs epochs
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=hp.warmup_epochs
        )
        # Phase 2: CosineAnnealingLR (T_max=290, eta_min=5e-7)
        # Deeper final LR floor (5e-7 vs node2-1-1-1's 1e-6) for fine-grained late exploration
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=hp.t_max, eta_min=hp.eta_min
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_sched, cosine_sched], milestones=[hp.warmup_epochs]
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
        description="Node1-2-1-2: AIDO.Cell-100M (LoRA) + Frozen STRING_GNN (K=16 neigh attn)"
    )
    # Batch / compute
    parser.add_argument("--micro-batch-size",  type=int,   default=1)
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument("--max-epochs",        type=int,   default=300)
    # Architecture
    parser.add_argument("--K",                 type=int,   default=16, dest="K")
    parser.add_argument("--attn-dim",          type=int,   default=64, dest="attn_dim")
    parser.add_argument("--head-hidden",       type=int,   default=256, dest="head_hidden")
    parser.add_argument("--dropout",           type=float, default=0.4)
    parser.add_argument("--lora-r",            type=int,   default=8,  dest="lora_r")
    parser.add_argument("--lora-alpha",        type=int,   default=16, dest="lora_alpha")
    parser.add_argument("--lora-dropout",      type=float, default=0.05, dest="lora_dropout")
    # Optimization
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--weight-decay",      type=float, default=2e-2)
    parser.add_argument("--label-smoothing",   type=float, default=0.05, dest="label_smoothing")
    parser.add_argument("--warmup-epochs",     type=int,   default=10,  dest="warmup_epochs")
    parser.add_argument("--t-max",             type=int,   default=290, dest="t_max")
    parser.add_argument("--eta-min",           type=float, default=5e-7, dest="eta_min")
    parser.add_argument("--patience",          type=int,   default=15)
    parser.add_argument("--val-check-interval", type=float, default=1.0, dest="val_check_interval")
    # Infrastructure
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--debug-max-step",    type=int,   default=None, dest="debug_max_step")
    parser.add_argument("--fast-dev-run",      action="store_true", dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Debug / limit logic
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # DataModule
    # ------------------------------------------------------------------
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = AIDOCellStringFusionModel(
        K               = args.K,
        attn_dim        = args.attn_dim,
        head_hidden     = args.head_hidden,
        dropout         = args.dropout,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        t_max           = args.t_max,
        eta_min         = args.eta_min,
        label_smoothing = args.label_smoothing,
        lora_r          = args.lora_r,
        lora_alpha      = args.lora_alpha,
        lora_dropout    = args.lora_dropout,
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
        auto_insert_metric_name=False,
    )
    es_cb = EarlyStopping(
        monitor   = "val/f1",
        mode      = "max",
        patience  = args.patience,
        min_delta = 1e-4,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # ------------------------------------------------------------------
    # Loggers
    # ------------------------------------------------------------------
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # ------------------------------------------------------------------
    # DDP Strategy
    # find_unused_parameters=True: fallback_emb may not be used every batch
    # ------------------------------------------------------------------
    use_ddp = n_gpus > 1 and not fast_dev_run
    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=180))
        if use_ddp else "auto"
    )
    devices_for_trainer = 1 if (fast_dev_run and n_gpus > 1) else n_gpus

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Train + Test
    # ------------------------------------------------------------------
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
    print(f"[Node1-2-1-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
