"""Node 1-2: AIDO.Cell-100M LoRA + STRING_GNN K=16 2-Head + Smooth GenePriorBias + Bilinear Head.

Strategy: Fork from the proven best architecture (node2-1-1-1-1-1, F1=0.5128) and add
GenePriorBias with smooth activation to provide per-gene output calibration.

Architecture:
  1. AIDO.Cell-100M (LoRA r=8): perturbation-aware transcriptomic context → [B, 640]
     Single-gene perturbation input: pert_id = 1.0, all others = -1.0 (missing)
  2. STRING_GNN K=16 Multi-Head (2-head) Neighborhood Attention (frozen, 256-dim): PPI context
  3. Fusion: simple concat [640 + 256 = 896] (NO ESM2, NO fusion MLP bottleneck)
  4. Bilinear head: einsum("bd,cgd->bcg", h, gene_class_emb) / sqrt(D) + gene_bias
  5. GenePriorBias: learnable per-gene per-class additive bias with smooth linear ramp activation
     starting at bias_warmup_start=30, fully active by bias_warmup_start+bias_ramp_epochs=40
  6. Loss: weighted CE + label smoothing ε=0.05

Key improvements over parent (node1-1-2-2):
1. Remove ESM2 — the 3840→256 projection was the primary bottleneck degrading F1 by -0.028
2. Remove triple-concat fusion MLP (1152→512→256 bottleneck) — harmful for optimization
3. Simple 2-modality concat (640+256=896) → direct bilinear head (proven in node2-1-1-1-1-1)
4. Add GenePriorBias with SMOOTH activation (linear ramp 0→1 over 10 epochs starting at epoch 30)
   to avoid catastrophic disruption seen with binary on/off activation
5. Increase patience from 15 to 25 — parent stopped at epoch 38 before full convergence
6. Keep proven hyperparameters: lr=1e-4, wd=2e-2, label_smoothing=0.05, dropout=0.5

Inspired by:
- node2-1-1-1-1-1 (best in tree, F1=0.5128): AIDO+STRING K=16 2-head, simple concat 896→bilinear
- node1-1-1-1-2-1 (F1=0.4913): GenePriorBias contributes +0.017 in STRING-only
- node1-1-2-2 (parent, F1=0.4307): ESM2 projection is the confirmed bottleneck
- Feedback: "fork from node2-1-1-1-1-1 and add GenePriorBias, targeting F1 >= 0.515"
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
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES   = 6640
N_CLASSES = 3

# Class frequencies (remapped: -1→0, 0→1, +1→2)
CLASS_FREQ = [0.0429, 0.9251, 0.0320]  # [down, neutral, up]

AIDO_CELL_DIR  = Path("/home/Models/AIDO.Cell-100M")
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

# Embedding dimensions
AIDO_DIM    = 640   # AIDO.Cell-100M hidden dimension
STRING_DIM  = 256   # STRING_GNN output dimension
FUSED_DIM   = AIDO_DIM + STRING_DIM  # 640 + 256 = 896 (NO ESM2)

# Neighborhood attention config
NEIGHBOR_K   = 16
ATTN_DIM     = 64   # per-head dimension
N_HEADS      = 2    # multi-head attention


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def load_string_gnn_mapping() -> Dict[str, int]:
    """Load STRING_GNN node_names.json: Ensembl-ID → node-index."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    return {name: idx for idx, name in enumerate(node_names)}


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0,1,2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)   # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true  = (targets == c)
        is_pred  = (y_hat == c)
        present  = is_true.any(dim=0)

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
        self.pert_ids  = df["pert_id"].tolist()
        self.symbols   = df["symbol"].tolist()

        # STRING_GNN node index (-1 for unknowns)
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


def make_collate_fn(tokenizer) -> Any:
    """Build a collate function that tokenizes for AIDO.Cell on-the-fly."""
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Build AIDO.Cell expression dicts: perturbed gene = 1.0, all others = -1.0 (missing)
        expr_dicts = [
            {"gene_ids": [b["pert_id"]], "expression": [1.0]}
            for b in batch
        ]
        tokenized = tokenizer(expr_dicts, return_tensors="pt")  # input_ids: [B, 19264] float32

        out: Dict[str, Any] = {
            "sample_idx":      torch.tensor([b["sample_idx"]       for b in batch], dtype=torch.long),
            "pert_id":         [b["pert_id"]   for b in batch],
            "symbol":          [b["symbol"]    for b in batch],
            "string_node_idx": torch.stack([b["string_node_idx"]   for b in batch]),
            "input_ids":       tokenized["input_ids"],        # [B, 19264] float32
            "attention_mask":  tokenized["attention_mask"],   # [B, 19264] int64
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out

    return collate_fn


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.string_map: Optional[Dict[str, int]] = None
        self.tokenizer     = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Load STRING_GNN mapping
        if self.string_map is None:
            self.string_map = load_string_gnn_mapping()

        # Load AIDO.Cell tokenizer (rank-0 first to avoid duplicate downloads)
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

        self.train_ds = DEGDataset(train_df, self.string_map)
        self.val_ds   = DEGDataset(val_df,   self.string_map)
        self.test_ds  = DEGDataset(test_df,  self.string_map)

        self._collate_fn = make_collate_fn(self.tokenizer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=self._collate_fn, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate_fn, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        sampler = SequentialSampler(self.test_ds)
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=self._collate_fn, pin_memory=True,
            sampler=sampler,
        )


# ---------------------------------------------------------------------------
# Neural network modules
# ---------------------------------------------------------------------------
class MultiHeadNeighborhoodAttention(nn.Module):
    """Multi-head neighborhood attention over K nearest PPI neighbors.

    Architecture (2 heads, attn_dim=64 per head):
        Head i: q_i = W_q_i(center), k_i = W_k_i(neigh)
                attn_i = softmax(q_i @ k_i.T / sqrt(attn_dim) + log_conf)
                ctx_i  = attn_i @ neigh  [B, STRING_DIM]
        multi_ctx = concat([ctx_1, ctx_2])  [B, 2*STRING_DIM]
        projected = W_out(multi_ctx)        [B, STRING_DIM]
        gate      = sigmoid(W_gate([center, projected]))  [B, STRING_DIM]
        output    = gate * center + (1-gate) * projected  [B, STRING_DIM]
    """

    def __init__(self, in_dim: int = STRING_DIM, k: int = NEIGHBOR_K,
                 n_heads: int = N_HEADS, attn_dim: int = ATTN_DIM) -> None:
        super().__init__()
        self.k       = k
        self.n_heads = n_heads
        self.attn_dim = attn_dim
        self.scale   = attn_dim ** -0.5

        # Per-head projection matrices
        self.W_q = nn.ModuleList([nn.Linear(in_dim, attn_dim, bias=False) for _ in range(n_heads)])
        self.W_k = nn.ModuleList([nn.Linear(in_dim, attn_dim, bias=False) for _ in range(n_heads)])
        # Output projection from concatenated contexts
        self.W_out  = nn.Linear(n_heads * in_dim, in_dim, bias=False)
        # Gating
        self.W_gate = nn.Linear(2 * in_dim, in_dim, bias=False)
        self.norm   = nn.LayerNorm(in_dim)

    def forward(
        self,
        center: torch.Tensor,   # [B, in_dim]
        neigh:  torch.Tensor,   # [B, K, in_dim]
        conf:   torch.Tensor,   # [B, K] confidence weights
    ) -> torch.Tensor:
        B = center.shape[0]
        log_conf = torch.log(conf.clamp(min=1e-6)).unsqueeze(1)  # [B, 1, K]

        contexts = []
        for h in range(self.n_heads):
            q = self.W_q[h](center).unsqueeze(1)          # [B, 1, attn_dim]
            k = self.W_k[h](neigh)                         # [B, K, attn_dim]
            attn_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, 1, K]
            attn_logits = attn_logits + log_conf
            attn_weights = attn_logits.softmax(dim=-1)     # [B, 1, K]
            ctx = torch.bmm(attn_weights, neigh).squeeze(1)  # [B, in_dim]
            contexts.append(ctx)

        multi_ctx  = torch.cat(contexts, dim=-1)           # [B, n_heads * in_dim]
        projected  = self.W_out(multi_ctx)                  # [B, in_dim]
        gate       = torch.sigmoid(self.W_gate(torch.cat([center, projected], dim=-1)))
        out        = gate * center + (1.0 - gate) * projected  # [B, in_dim]
        return self.norm(out)


class GenePriorBias(nn.Module):
    """Learnable per-gene per-class additive calibration bias with smooth activation.

    The bias is ramped smoothly from 0 to 1 over bias_ramp_epochs starting at
    bias_warmup_start epoch, to avoid catastrophic disruption seen with binary
    on/off activation in previous nodes.

    Architecture:
        bias [N_CLASSES, N_GENES] initialized to zeros (no class-prior distortion)
        Scale factor: linearly ramped from 0 to 1 over the ramp window

    This is stored as a register_buffer (persistent) to ensure correct behavior
    at test inference — the scale factor is saved/loaded in the checkpoint.
    """

    def __init__(self, n_classes: int = N_CLASSES, n_genes: int = N_GENES,
                 bias_warmup_start: int = 30, bias_ramp_epochs: int = 10) -> None:
        super().__init__()
        self.bias_warmup_start = bias_warmup_start
        self.bias_ramp_epochs  = bias_ramp_epochs

        # Learnable per-gene per-class bias (zero init = no initial distortion)
        self.bias = nn.Parameter(torch.zeros(n_classes, n_genes))

        # Scale factor: 0 during warmup, linearly ramps to 1 (persistent for inference)
        self.register_buffer("bias_scale", torch.tensor(0.0))
        self.register_buffer("current_epoch_buf", torch.tensor(0, dtype=torch.long))

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch to update scale factor."""
        self.current_epoch_buf.fill_(epoch)
        if epoch < self.bias_warmup_start:
            scale = 0.0
        elif epoch < self.bias_warmup_start + self.bias_ramp_epochs:
            # Linear ramp from 0 to 1
            scale = float(epoch - self.bias_warmup_start) / float(self.bias_ramp_epochs)
        else:
            scale = 1.0
        self.bias_scale.fill_(scale)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Add scaled per-gene bias to logits.

        Args:
            logits: [B, N_CLASSES, N_GENES]
        Returns:
            logits + scale * bias: [B, N_CLASSES, N_GENES]
        """
        return logits + self.bias_scale * self.bias.unsqueeze(0)


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class DualBackboneBilinearModel(pl.LightningModule):
    """AIDO.Cell-100M LoRA + STRING_GNN K=16 2-head + Smooth GenePriorBias.

    Architecture:
        1. AIDO.Cell-100M (LoRA r=8): tokenized single-gene expression → [B, 19266, 640]
           → summary_token[:, 19264, :] → [B, 640]
        2. STRING_GNN K=16 Multi-Head Attention: frozen node embeddings + neighborhood attn → [B, 256]
        3. Fusion: simple concat([aido_emb, string_emb]) → [B, 896]
                   → LayerNorm(896) → Linear(896→bilinear_dim) → GELU → Dropout(head_dropout) → [B, D]
        4. Bilinear head: einsum("bd,cgd->bcg", h, gene_class_emb) / sqrt(D)
        5. GenePriorBias (smooth activation): + scale * bias [3, 6640]
        6. Loss: weighted CE + label smoothing ε=0.05
    """

    def __init__(
        self,
        bilinear_dim: int = 256,
        head_hidden: int = 896,
        head_dropout: float = 0.5,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lr: float = 1e-4,
        weight_decay: float = 2e-2,
        warmup_epochs: int = 10,
        min_lr_ratio: float = 0.01,
        label_smoothing: float = 0.05,
        bias_warmup_start: int = 30,
        bias_ramp_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Model initialized in setup()

    def setup(self, stage: Optional[str] = None) -> None:
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True
        hp = self.hparams

        # ── Load AIDO.Cell-100M with LoRA r=8 ────────────────────────────────
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoModel.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        aido_backbone = AutoModel.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        # Move to CUDA bf16 BEFORE LoRA wrapping (needed for FlashAttention)
        aido_backbone = aido_backbone.cuda().to(torch.bfloat16)
        # Apply LoRA to Q/K/V attention projections
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
        )
        aido_backbone = get_peft_model(aido_backbone, lora_cfg)
        aido_backbone.config.use_cache = False
        aido_backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        aido_backbone.train()
        aido_backbone.base_model.train()
        aido_backbone.base_model.model.train()
        self.aido_backbone = aido_backbone

        # ── Load STRING_GNN and pre-compute frozen node embeddings ────────────
        if local_rank == 0:
            AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        string_backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        string_backbone.eval()

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph["edge_index"].long()
        edge_weight = graph["edge_weight"].float()

        with torch.no_grad():
            gnn_out = string_backbone(edge_index=edge_index, edge_weight=edge_weight)
            string_emb_table = gnn_out.last_hidden_state.float()  # [18870, 256]
        self.register_buffer("string_emb_table", string_emb_table)

        # Build K=16 nearest-neighbor tables (sorted by edge weight / confidence)
        n_nodes = string_emb_table.shape[0]
        topk_neighbors = torch.zeros(n_nodes, NEIGHBOR_K, dtype=torch.long)
        topk_weights   = torch.zeros(n_nodes, NEIGHBOR_K, dtype=torch.float32)

        adj: Dict[int, List] = {i: [] for i in range(n_nodes)}
        ei = edge_index.numpy()
        ew = edge_weight.numpy()
        for e_idx in range(ei.shape[1]):
            src, dst = int(ei[0, e_idx]), int(ei[1, e_idx])
            if src != dst:  # skip self-loops
                adj[src].append((dst, float(ew[e_idx])))

        for i in range(n_nodes):
            neighbors = sorted(adj[i], key=lambda x: -x[1])[:NEIGHBOR_K]
            if len(neighbors) == 0:
                # Isolated node: use self as neighbor
                neighbors = [(i, 1.0)] * NEIGHBOR_K
            elif len(neighbors) < NEIGHBOR_K:
                # Pad by repeating last neighbor
                pad = [neighbors[-1]] * (NEIGHBOR_K - len(neighbors))
                neighbors = neighbors + pad
            topk_neighbors[i] = torch.tensor([n[0] for n in neighbors], dtype=torch.long)
            topk_weights[i]   = torch.tensor([n[1] for n in neighbors], dtype=torch.float32)

        self.register_buffer("topk_neighbors", topk_neighbors)  # [18870, K]
        self.register_buffer("topk_weights",   topk_weights)    # [18870, K]

        # Fallback for unknowns
        self.fallback_string = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_string.weight, std=0.02)

        # ── Neighborhood attention module ─────────────────────────────────────
        self.neigh_attn = MultiHeadNeighborhoodAttention(
            in_dim=STRING_DIM, k=NEIGHBOR_K, n_heads=N_HEADS, attn_dim=ATTN_DIM
        )

        # ── Fusion head: FUSED_DIM (896) → bilinear_dim ──────────────────────
        # Simple 2-layer projection (no separate fusion MLP bottleneck)
        self.head_proj = nn.Sequential(
            nn.LayerNorm(FUSED_DIM),
            nn.Linear(FUSED_DIM, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
        )

        # ── Bilinear gene-class embedding matrix [C, G, bilinear_dim] ────────
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # ── GenePriorBias with smooth activation ─────────────────────────────
        self.gene_prior_bias = GenePriorBias(
            n_classes=N_CLASSES,
            n_genes=N_GENES,
            bias_warmup_start=hp.bias_warmup_start,
            bias_ramp_epochs=hp.bias_ramp_epochs,
        )

        # ── Loss components ───────────────────────────────────────────────────
        self.register_buffer("class_weights", get_class_weights())

        # ── Cast trainable parameters to float32 ─────────────────────────────
        for _, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Accumulators ──────────────────────────────────────────────────────
        self._val_preds: List[torch.Tensor] = []
        self._val_tgts:  List[torch.Tensor] = []
        self._val_idx:   List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    # ── Embedding lookups ─────────────────────────────────────────────────────
    def _get_string_neighborhood_emb(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Get STRING_GNN neighborhood-attended embedding.

        Args:
            string_node_idx: [B] long, -1 for unknowns
        Returns:
            [B, STRING_DIM] float
        """
        B = string_node_idx.shape[0]
        known_mask   = string_node_idx >= 0
        unknown_mask = ~known_mask
        zero_idx     = torch.zeros(unknown_mask.sum(), dtype=torch.long,
                                   device=string_node_idx.device)

        # Initialize output
        string_emb = torch.zeros(B, STRING_DIM,
                                  dtype=self.string_emb_table.dtype,
                                  device=self.string_emb_table.device)
        if unknown_mask.any():
            string_emb[unknown_mask] = self.fallback_string(zero_idx).to(
                self.string_emb_table.dtype)

        if known_mask.any():
            known_idx = string_node_idx[known_mask]
            center = self.string_emb_table[known_idx].float()         # [K_known, STRING_DIM]
            neigh_idx = self.topk_neighbors[known_idx]                 # [K_known, K]
            conf      = self.topk_weights[known_idx].float()           # [K_known, K]
            neigh     = self.string_emb_table[neigh_idx.view(-1)].view(
                known_idx.shape[0], NEIGHBOR_K, STRING_DIM
            ).float()  # [K_known, K, STRING_DIM]
            attended = self.neigh_attn(center, neigh, conf)            # [K_known, STRING_DIM]
            string_emb[known_mask] = attended.to(string_emb.dtype)

        return string_emb.float()

    # ── Forward ────────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32
        attention_mask: torch.Tensor,  # [B, 19264] int64
        string_node_idx: torch.Tensor, # [B] long
    ) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        # Stream 1: AIDO.Cell-100M LoRA → summary token
        aido_out = self.aido_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Summary token is at position 19264 (first of the 2 appended tokens)
        aido_emb = aido_out.last_hidden_state[:, 19264, :].float()  # [B, 640]

        # Stream 2: STRING_GNN K=16 neighborhood attention
        string_emb = self._get_string_neighborhood_emb(string_node_idx)  # [B, 256]

        # Fusion: simple concat (NO ESM2, NO fusion MLP bottleneck)
        fused = torch.cat([aido_emb, string_emb], dim=-1)  # [B, 896]

        # Project to bilinear_dim
        h = self.head_proj(fused)  # [B, bilinear_dim]

        # Scaled bilinear output
        scale  = self.hparams.bilinear_dim ** 0.5
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb) / scale  # [B, 3, G]

        # Apply GenePriorBias (smooth-activated per-gene calibration)
        logits = self.gene_prior_bias(logits)  # [B, 3, G]

        return logits

    # ── Loss ──────────────────────────────────────────────────────────────────
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, 3]
            targets.reshape(-1),                       # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ── Epoch hooks ───────────────────────────────────────────────────────────
    def on_train_epoch_start(self) -> None:
        """Update GenePriorBias scale at the start of each training epoch."""
        current_epoch = self.current_epoch
        self.gene_prior_bias.set_epoch(current_epoch)
        # Log current bias scale
        self.log("train/bias_scale", self.gene_prior_bias.bias_scale.item(),
                 prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        """Ensure GenePriorBias scale is consistent during validation."""
        self.gene_prior_bias.set_epoch(self.current_epoch)

    def on_test_epoch_start(self) -> None:
        """At test time, bias should be fully active (scale=1.0)."""
        # When loading from best checkpoint, we use the saved scale value.
        # Only force to 1.0 if scale is 0 (e.g., fast_dev_run before any training).
        if self.gene_prior_bias.bias_scale.item() < 1.0:
            # Training may not have reached full activation - use whatever scale was saved
            pass  # Use the value from the checkpoint (set_epoch was called during training)

    # ── Training / validation / test steps ────────────────────────────────────
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["string_node_idx"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True,
                 on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["string_node_idx"])
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

        # Gather across all ranks
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
        mask   = torch.cat([torch.tensor([True], device=s_idx.device),
                            s_idx[1:] != s_idx[:-1]])
        preds_dedup = s_pred[mask]
        tgts_dedup  = s_tgt[mask]

        f1 = compute_per_gene_f1(preds_dedup, tgts_dedup)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["string_node_idx"])
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
        all_preds   = self.all_gather(local_preds)         # [W, N_local, 3, G]
        all_idx     = self.all_gather(local_idx)           # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            # De-duplicate
            order  = torch.argsort(idx_flat)
            s_idx  = idx_flat[order]
            s_pred = preds_flat[order]
            mask   = torch.cat([torch.ones(1, dtype=torch.bool, device=s_idx.device),
                                s_idx[1:] != s_idx[:-1]])
            preds_dedup = s_pred[mask]
            unique_sid  = s_idx[mask].tolist()

            # Reload test.tsv on rank 0
            test_df = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {
                i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                for i in range(len(test_df))
            }

            rows = []
            for i, sid in enumerate(unique_sid):
                pid, sym = idx_to_meta[int(sid)]
                # preds_dedup and unique_sid are aligned by construction (both follow mask order)
                pred_list = preds_dedup[i].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-2] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ── Checkpoint helpers ────────────────────────────────────────────────────
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
        self.print(f"Checkpoint: {train}/{total} params ({100*train/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        hp = self.hparams

        opt = torch.optim.AdamW(
            self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )

        # WarmupCosine schedule: linear warmup for warmup_epochs, then cosine decay
        # with floor at min_lr_ratio * lr
        def lr_lambda(epoch: int) -> float:
            if epoch < hp.warmup_epochs:
                return (epoch + 1) / hp.warmup_epochs
            # Cosine decay phase
            progress = (epoch - hp.warmup_epochs) / max(1, self._max_epochs - hp.warmup_epochs)
            cosine_factor = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265358979)).item())
            return hp.min_lr_ratio + (1.0 - hp.min_lr_ratio) * cosine_factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    @property
    def _max_epochs(self) -> int:
        """Access max_epochs from trainer or default to 300."""
        if self.trainer is not None and hasattr(self.trainer, "max_epochs"):
            return self.trainer.max_epochs or 300
        return 300


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-2 – AIDO.Cell-100M LoRA + STRING_GNN K=16 2-head + Smooth GenePriorBias + Bilinear Head"
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=16)
    parser.add_argument("--global-batch-size",   type=int,   default=128)
    parser.add_argument("--max-epochs",          type=int,   default=300)
    parser.add_argument("--lr",                  type=float, default=1e-4)
    parser.add_argument("--weight-decay",        type=float, default=2e-2)
    parser.add_argument("--bilinear-dim",        type=int,   default=256)
    parser.add_argument("--head-dropout",        type=float, default=0.5)
    parser.add_argument("--lora-r",              type=int,   default=8)
    parser.add_argument("--lora-alpha",          type=int,   default=16)
    parser.add_argument("--warmup-epochs",       type=int,   default=10)
    parser.add_argument("--min-lr-ratio",        type=float, default=0.01)
    parser.add_argument("--label-smoothing",     type=float, default=0.05)
    parser.add_argument("--num-workers",         type=int,   default=4)
    parser.add_argument("--patience",            type=int,   default=25)
    parser.add_argument("--bias-warmup-start",   type=int,   default=30,
                        dest="bias_warmup_start")
    parser.add_argument("--bias-ramp-epochs",    type=int,   default=10,
                        dest="bias_ramp_epochs")
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
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        max_steps = -1

    lim_test = 1.0
    val_check_interval = 1.0

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    # Model
    model = DualBackboneBilinearModel(
        bilinear_dim       = args.bilinear_dim,
        head_dropout       = args.head_dropout,
        lora_r             = args.lora_r,
        lora_alpha         = args.lora_alpha,
        lr                 = args.lr,
        weight_decay       = args.weight_decay,
        warmup_epochs      = args.warmup_epochs,
        min_lr_ratio       = args.min_lr_ratio,
        label_smoothing    = args.label_smoothing,
        bias_warmup_start  = args.bias_warmup_start,
        bias_ramp_epochs   = args.bias_ramp_epochs,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
    )
    es_cb = EarlyStopping(monitor="val/f1", mode="max",
                          patience=args.patience, min_delta=1e-3)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

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
