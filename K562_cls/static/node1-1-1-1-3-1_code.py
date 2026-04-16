"""Node 1-1-1-1-3-1 – AIDO.Cell-100M LoRA + STRING_GNN K=16 2-head Neighborhood Attention + GenePriorBias.

Addresses the primary bottleneck in node1-1-1-1-3 (F1=0.4846):
  - The STRING_GNN lineage has plateaued at ~0.485 F1.
  - The best node in the tree (node2-1-1-1-1-1, F1=0.5128) uses AIDO.Cell-100M LoRA
    fused with STRING_GNN K=16 2-head neighborhood attention + 2-layer MLP head.
  - GenePriorBias (proven +0.007 in parent lineage) is NOT in node2-1-1-1-1-1,
    making it an orthogonal addition that can push beyond 0.513.

Architecture:
  Stream 1: AIDO.Cell-100M with LoRA (r=8, alpha=16)
    - Input: sparse perturbation vector [B, 19264] float32
      (only perturbed gene set to 1.0; all other genes = -1.0 → 0 inside model)
    - Tokenized via AIDO.Cell tokenizer: gene_ids=[pert_id], expression=[1.0]
    - Forward → last_hidden_state [B, 19266, 640]
    - Mean-pool over gene positions [:, :19264, :] → [B, 640]
    - Gradient checkpointing enabled

  Stream 2: Frozen STRING_GNN K=16 2-head neighborhood attention
    - Pre-computed node embeddings [18870, 256] (STRING_GNN run once at setup)
    - Pre-computed top-16 neighbor indices + softmax edge weights
    - 2-head multi-head attention (attn_dim=32/head, head_dim=128) → fused [B, 256]
    - Learnable fallback embedding for pert_ids not in STRING vocabulary

  Fusion: Concatenate AIDO [640] + STRING [256] = [B, 896]

  Head: 2-layer MLP
    - LayerNorm(896) → Linear(896, 256) → GELU → Dropout(0.5)
    - LayerNorm(256) → Linear(256, 19920) → reshape [B, 3, 6640]

  GenePriorBias: Learnable [3, 6640] per-gene-per-class bias
    - Warmup=50 epochs (inactive first 50, activated via persistent register_buffer)
    - Proven +0.007 F1 across parent lineage

Loss: Weighted CE (sqrt-inverse-freq) + label smoothing ε=0.05

Optimizer: AdamW with discriminative LR
  - AIDO.Cell backbone (LoRA params): backbone_lr=5e-5
  - All other (head, nbr_agg, GenePriorBias): head_lr=2e-4
  - weight_decay=3e-2

LR Schedule: 10-epoch linear warmup → CosineAnnealingLR (T_max=200, eta_min=1e-6)
Training: patience=30, max_epochs=300 (captures late spikes like node2-1-1-1-1-1's epoch 77)

Expected: ~0.515–0.520 F1 (best in tree node2-1-1-1-1-1=0.5128 + GenePriorBias +0.007)
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from functools import partial
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
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3
AIDO_CELL_DIM = 640   # hidden_size of AIDO.Cell-100M
STRING_DIM = 256      # STRING_GNN hidden dimension
FUSION_DIM = AIDO_CELL_DIM + STRING_DIM  # 896

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
    n_nodes: int,
    K: int = 16,
) -> tuple:
    """Pre-compute top-K neighbor indices and softmax-normalized weights for each node."""
    src = edge_index[0]
    dst = edge_index[1]
    wts = edge_weight

    perm = torch.argsort(src, stable=True)
    src_sorted = src[perm]
    dst_sorted = dst[perm]
    wts_sorted = wts[perm]
    E = src.shape[0]

    node_ids = torch.arange(n_nodes, dtype=torch.long)
    boundaries = torch.searchsorted(src_sorted.contiguous(), node_ids.contiguous())
    boundaries = torch.cat([boundaries, torch.tensor([E], dtype=torch.long)])

    topk_neighbors = torch.zeros(n_nodes, K, dtype=torch.long)
    topk_weights   = torch.full((n_nodes, K), 1.0 / K, dtype=torch.float32)

    for i in range(n_nodes):
        start = int(boundaries[i].item())
        end   = int(boundaries[i + 1].item())

        if start == end:
            topk_neighbors[i] = torch.full((K,), i, dtype=torch.long)
            topk_weights[i]   = torch.full((K,), 1.0 / K, dtype=torch.float32)
            continue

        node_dsts = dst_sorted[start:end]
        node_wts  = wts_sorted[start:end]
        n_nbrs    = node_dsts.shape[0]

        if n_nbrs > K:
            top_idx   = torch.argsort(node_wts, descending=True)[:K]
            node_dsts = node_dsts[top_idx]
            node_wts  = node_wts[top_idx]
            n_nbrs    = K

        if n_nbrs < K:
            pad = K - n_nbrs
            node_dsts = torch.cat([node_dsts, node_dsts[-1:].expand(pad)])
            node_wts  = torch.cat([node_wts,  node_wts[-1:].expand(pad)])

        topk_neighbors[i] = node_dsts
        topk_weights[i]   = torch.softmax(node_wts.float(), dim=0)

    return topk_neighbors, topk_weights


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
    """K562 DEG prediction dataset."""

    def __init__(self, df: pd.DataFrame, string_map: Dict[str, int]) -> None:
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
            item["labels"] = self.labels[idx]  # [G] in {0, 1, 2}
        return item


def make_collate_fn(tokenizer):
    """Create collate function with AIDO.Cell tokenizer closure."""

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Create sparse perturbation expression dicts for AIDO.Cell
        # Each pert has only its gene expressed at 1.0; all others → -1.0 (missing)
        expr_dicts = [
            {"gene_ids": [b["pert_id"]], "expression": [1.0]}
            for b in batch
        ]
        tokenized = tokenizer(expr_dicts, return_tensors="pt")

        out: Dict[str, Any] = {
            "aido_input_ids":     tokenized["input_ids"],       # [B, 19264] float32
            "aido_attn_mask":     tokenized["attention_mask"],  # [B, 19264] int64
            "sample_idx":         torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":            [b["pert_id"]  for b in batch],
            "symbol":             [b["symbol"]   for b in batch],
            "string_node_idx":    torch.stack([b["string_node_idx"] for b in batch]),
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out

    return collate_fn


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.string_map: Optional[Dict[str, int]] = None
        self.tokenizer = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Guard against double setup (e.g., manual call before trainer + trainer's internal call)
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        # Load AIDO.Cell tokenizer (rank-0 first to trigger download if needed)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(AIDO_CELL_DIR), trust_remote_code=True
            )

        # Load STRING_GNN mapping
        if self.string_map is None:
            self.string_map = load_string_gnn_mapping()

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df, self.string_map)
        self.val_ds   = DEGDataset(val_df,   self.string_map)
        self.test_ds  = DEGDataset(test_df,  self.string_map)

    def _make_loader(self, ds, shuffle: bool, sampler=None) -> DataLoader:
        collate = make_collate_fn(self.tokenizer)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None),
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
            sampler=sampler,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(
            self.test_ds, shuffle=False,
            sampler=SequentialSampler(self.test_ds),
        )


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
class MultiHeadNeighborhoodAggregator(nn.Module):
    """2-head PPI neighborhood attention aggregator.

    Each head attends over top-K PPI neighbors in separate attention subspaces.
    Value projections map neighbor embeddings to per-head subspaces.
    Concatenated head outputs are gated with the center embedding.

    Parameters (embed_dim=256, attn_dim=32, n_heads=2):
        W_q: Linear(256, 64, bias=False)   = 16,384
        W_k: Linear(256, 64, bias=False)   = 16,384
        W_v: Linear(256, 256, bias=False)  = 65,536
        W_gate: Linear(512, 256)           = 131,328
        attn_dropout: —
        Total: ~229,632 params
    """

    def __init__(
        self,
        embed_dim:    int   = 256,
        attn_dim:     int   = 32,
        n_heads:      int   = 2,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads  = n_heads
        self.attn_dim = attn_dim
        self.head_dim = embed_dim // n_heads  # 128 per head

        # Joint Q, K projections for all heads: [D, H*attn_dim]
        self.W_q = nn.Linear(embed_dim, n_heads * attn_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, n_heads * attn_dim, bias=False)
        # V projections: each head gets head_dim subspace of neighbor embeddings
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)  # [D, H*head_dim]
        # Gate: blends center with multi-head context
        self.W_gate     = nn.Linear(embed_dim * 2, embed_dim)
        self.attn_drop  = nn.Dropout(attn_dropout)

    def forward(
        self,
        center_emb: torch.Tensor,  # [B, D]
        neigh_embs: torch.Tensor,  # [B, K, D]
        neigh_wts:  torch.Tensor,  # [B, K] — pre-softmaxed PPI confidence weights
    ) -> torch.Tensor:             # [B, D]
        B, K, D = neigh_embs.shape
        H  = self.n_heads
        da = self.attn_dim
        dh = self.head_dim

        # Q: [B, H*da] → [B, H, da]
        q = self.W_q(center_emb).view(B, H, da)

        # K: [B*K, H*da] → [B, K, H, da] → [B, H, K, da]
        k = self.W_k(neigh_embs.reshape(B * K, D)).view(B, K, H, da).permute(0, 2, 1, 3)

        # V: [B*K, D] → [B, K, H, dh] → [B, H, K, dh]
        v = self.W_v(neigh_embs.reshape(B * K, D)).view(B, K, H, dh).permute(0, 2, 1, 3)

        # Attention scores: [B, H, 1, da] × [B, H, da, K] → [B, H, K]
        scale = da ** 0.5
        attn = torch.matmul(q.unsqueeze(2), k.transpose(-1, -2)).squeeze(2) / scale  # [B, H, K]

        # Incorporate PPI confidence prior
        attn = attn * neigh_wts.unsqueeze(1)           # [B, H, K]
        attn = torch.softmax(attn, dim=-1)              # [B, H, K]
        attn = self.attn_drop(attn)

        # Weighted context: [B, H, K] × [B, H, K, dh] → [B, H, dh] → [B, D]
        context = torch.bmm(
            attn.view(B * H, 1, K),
            v.reshape(B * H, K, dh),
        ).view(B * H, dh)                              # [B*H, dh]
        context = context.view(B, H * dh)              # [B, D]   (H*dh = H*(D/H) = D)

        # Learnable gating
        gate  = torch.sigmoid(self.W_gate(torch.cat([center_emb, context], dim=-1)))
        fused = gate * center_emb + (1.0 - gate) * context
        return fused


class GenePriorBias(nn.Module):
    """Per-gene-per-class learnable logit bias with gradient warmup.

    During warmup phase (epochs 0 to bias_warmup_epochs-1), the bias is NOT added
    to logits (not in computation graph → zero gradients).
    At epoch bias_warmup_epochs, activate() is called to enable the bias.

    bias_active is a persistent register_buffer so it is saved/loaded with every
    checkpoint, ensuring correct test-time behavior when ckpt_path='best'.
    """

    def __init__(self, n_classes: int = 3, n_genes: int = 6640) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_classes, n_genes))
        self.register_buffer("bias_active", torch.tensor(False))

    def activate(self) -> None:
        self.bias_active.fill_(True)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if bool(self.bias_active.item()):
            return logits + self.bias.unsqueeze(0)  # [1, 3, G] broadcasts to [B, 3, G]
        return logits


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCellStringGNNModel(pl.LightningModule):
    """AIDO.Cell-100M LoRA + STRING_GNN K=16 2-head neighborhood attention + GenePriorBias.

    Architecture (two-stream dual encoder fusion):
        Stream 1: AIDO.Cell-100M with LoRA (r=8, alpha=16)
            - Sparse perturbation input: only pert gene = 1.0, rest = -1.0
            - Output: mean-pool over gene positions → [B, 640]
        Stream 2: Frozen STRING_GNN + K=16 2-head neighborhood attention
            - Pre-computed node embeddings [18870, 256]
            - K=16 top neighbors with softmax-normalized edge weights
            - 2-head attention (attn_dim=32/head, head_dim=128/head) → [B, 256]
        Fusion: concat [640+256] → [B, 896]
        Head: LN → 896→256 → GELU → Dropout(0.5) → LN → 256→19920 → reshape [B,3,6640]
        GenePriorBias: [3,6640] with 50-epoch warmup
    """

    def __init__(
        self,
        head_hidden:         int   = 256,
        head_dropout:        float = 0.5,
        backbone_lr:         float = 5e-5,
        head_lr:             float = 2e-4,
        weight_decay:        float = 3e-2,
        warmup_epochs:       int   = 10,
        T_max:               int   = 200,
        label_smoothing:     float = 0.05,
        K:                   int   = 16,
        attn_dim:            int   = 32,
        n_heads:             int   = 2,
        lora_r:              int   = 8,
        lora_alpha:          int   = 16,
        bias_warmup_epochs:  int   = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Model layers initialized in setup()

    def setup(self, stage: Optional[str] = None) -> None:
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        hp = self.hparams

        # ----------------------------------------------------------------
        # 1. Load AIDO.Cell-100M with LoRA
        # ----------------------------------------------------------------
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        aido_base = AutoModel.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        # Cast backbone to bfloat16 so Flash attention is triggered.
        # AIDO.Cell's config has _use_flash_attention_2=True, but the dispatch condition
        # requires ln_outputs.dtype in {fp16, bf16}. With float32 backbone, standard
        # attention runs matmul on [B, heads, 19264, 19264] → OOM (~110 GiB for batch=8).
        aido_base = aido_base.to(torch.bfloat16)
        aido_base.config.use_cache = False
        # Enable gradient checkpointing for memory efficiency
        aido_base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # AIDO.Cell has a custom GeneEmbedding (no standard word_embeddings).
        # PEFT 0.15.2's get_peft_model unconditionally calls enable_input_require_grads()
        # which internally calls get_input_embeddings() → raises NotImplementedError in AIDO.Cell.
        # Monkey-patch to a no-op; we register a forward hook on gene_embedding below instead.
        aido_base.enable_input_require_grads = lambda: None

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
        )
        self.aido_backbone = get_peft_model(aido_base, lora_cfg)
        self.print(f"[setup] AIDO.Cell LoRA trainable params:")
        self.aido_backbone.print_trainable_parameters()

        # Forward hook: ensure gradients flow through gene_embedding into LoRA adapters
        def _make_inputs_require_grad(module, inp, out):
            out.requires_grad_(True)
        self.aido_backbone.model.bert.gene_embedding.register_forward_hook(
            _make_inputs_require_grad
        )

        # ----------------------------------------------------------------
        # 2. Pre-compute STRING_GNN embeddings and neighborhood lookup tables
        # ----------------------------------------------------------------
        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        graph        = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index   = graph["edge_index"].long()
        edge_weight  = graph["edge_weight"].float()
        n_nodes      = getattr(backbone.config, "num_nodes", None)

        with torch.no_grad():
            gnn_out  = backbone(edge_index=edge_index, edge_weight=edge_weight)
            node_emb = gnn_out.last_hidden_state.float().detach()  # [18870, 256]

        if n_nodes is None:
            n_nodes = node_emb.shape[0]

        self.register_buffer("node_embeddings", node_emb)  # fixed lookup table

        self.print(f"[setup] Pre-computing top-K={hp.K} neighbors for {n_nodes} nodes ...")
        topk_nbrs, topk_wts = compute_topk_neighbors(
            edge_index=edge_index,
            edge_weight=edge_weight,
            n_nodes=n_nodes,
            K=hp.K,
        )
        self.register_buffer("topk_neighbors", topk_nbrs)  # [n_nodes, K]
        self.register_buffer("topk_weights",   topk_wts)   # [n_nodes, K]
        self.print(f"[setup] Neighborhood buffers: {topk_nbrs.shape}, {topk_wts.shape}")

        del backbone, graph, edge_index, edge_weight, gnn_out

        # ----------------------------------------------------------------
        # 3. Learnable fallback for pert_ids not in STRING graph
        # ----------------------------------------------------------------
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 4. 2-head neighborhood attention aggregator
        # ----------------------------------------------------------------
        self.nbr_agg = MultiHeadNeighborhoodAggregator(
            embed_dim    = STRING_DIM,
            attn_dim     = hp.attn_dim,
            n_heads      = hp.n_heads,
            attn_dropout = 0.1,
        )

        # ----------------------------------------------------------------
        # 5. 2-layer MLP head (FUSION_DIM → head_hidden → N_CLASSES*N_GENES)
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM),
            nn.Linear(FUSION_DIM, hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.LayerNorm(hp.head_hidden),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # ----------------------------------------------------------------
        # 6. GenePriorBias (activated at epoch bias_warmup_epochs)
        # ----------------------------------------------------------------
        self.gene_prior = GenePriorBias(n_classes=N_CLASSES, n_genes=N_GENES)

        # Class weights for weighted CE
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
        self._test_ids:   List[str]          = []
        self._test_syms:  List[str]          = []
        self._test_idx:   List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.hparams.bias_warmup_epochs:
            self.gene_prior.activate()
            self.print(
                f"[Epoch {self.current_epoch}] GenePriorBias activated — "
                f"per-gene calibration bias now in computation graph."
            )

    # ------------------------------------------------------------------
    # STRING stream: neighborhood-aggregated embedding
    # ------------------------------------------------------------------
    def _get_string_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Return neighborhood-aggregated STRING_GNN embeddings [B, D=256]."""
        B    = string_node_idx.shape[0]
        known   = string_node_idx >= 0
        unknown = ~known

        emb = torch.zeros(B, STRING_DIM, dtype=torch.float32,
                          device=self.node_embeddings.device)

        if known.any():
            known_idx = string_node_idx[known]
            center    = self.node_embeddings[known_idx].float()      # [n_known, D]
            nbr_idx   = self.topk_neighbors[known_idx]               # [n_known, K]
            nbr_embs  = self.node_embeddings[nbr_idx].float()        # [n_known, K, D]
            nbr_wts   = self.topk_weights[known_idx].float()         # [n_known, K]
            emb[known] = self.nbr_agg(center, nbr_embs, nbr_wts)

        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(unknown.sum(), dtype=torch.long,
                            device=self.node_embeddings.device)
            ).float()
            emb[unknown] = fb

        return emb

    # ------------------------------------------------------------------
    # AIDO.Cell stream: perturbation embedding via mean-pool
    # ------------------------------------------------------------------
    def _get_aido_embeddings(
        self,
        input_ids:   torch.Tensor,  # [B, 19264] float32
        attention_mask: torch.Tensor,  # [B, 19264] int64
    ) -> torch.Tensor:              # [B, 640]
        # Ensure float32 for LoRA params; AIDO processes in bf16 internally
        out = self.aido_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last_hidden_state: [B, 19266, 640] (G=19264 + 2 summary positions)
        # Mean-pool over gene positions only (exclude the 2 appended summary scalars)
        gene_emb = out.last_hidden_state[:, :19264, :].mean(dim=1)  # [B, 640]
        return gene_emb.float()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,  # [B, 19264] float32
        attention_mask: torch.Tensor,  # [B, 19264] int64
        string_node_idx: torch.Tensor,  # [B] long
    ) -> torch.Tensor:                   # [B, 3, G]
        aido_emb  = self._get_aido_embeddings(input_ids, attention_mask)  # [B, 640]
        string_emb = self._get_string_embeddings(string_node_idx)          # [B, 256]

        fused = torch.cat([aido_emb, string_emb], dim=-1)  # [B, 896]

        logits = self.head(fused)                           # [B, 3*G = 19920]
        logits = logits.reshape(-1, N_CLASSES, N_GENES)    # [B, 3, G]
        logits = self.gene_prior(logits)                    # GenePriorBias (if active)
        return logits

    # ------------------------------------------------------------------
    # Loss
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
        logits = self(
            batch["aido_input_ids"],
            batch["aido_attn_mask"],
            batch["string_node_idx"],
        )
        loss = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["aido_input_ids"],
            batch["aido_attn_mask"],
            batch["string_node_idx"],
        )
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
            torch.tensor([True], device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        preds_dedup = s_pred[mask]
        tgts_dedup  = s_tgt[mask]

        f1 = compute_per_gene_f1(preds_dedup, tgts_dedup)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(
            batch["aido_input_ids"],
            batch["aido_attn_mask"],
            batch["string_node_idx"],
        )
        probs = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        self._test_ids.extend(batch["pert_id"])
        self._test_syms.extend(batch["symbol"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        local_preds = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)  # [N_local]

        all_preds = self.all_gather(local_preds)           # [W, N_local, 3, G]
        all_idx   = self.all_gather(local_idx)             # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            # De-duplicate by sample index
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
                # Save as [3, 6640] (correct format for calc_metric.py)
                pred = pred_map[i].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "test_predictions.tsv"
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
            self.print(f"[Node1-1-1-1-3-1] Saved {len(rows)} test predictions to {out_path}")

        self._test_preds.clear()
        self._test_ids.clear()
        self._test_syms.clear()
        self._test_idx.clear()

    # ------------------------------------------------------------------
    # Checkpoint helpers — save only trainable params + buffers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        saved = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full:
                    saved[key] = full[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full:
                saved[key] = full[key]
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {trainable}/{total} params ({100 * trainable / total:.1f}%)")
        return saved

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: discriminative LR + linear warmup + CosineAnnealingLR
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        # AIDO backbone LoRA params get backbone_lr; everything else gets head_lr
        backbone_params = list(self.aido_backbone.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        other_params    = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in backbone_ids
        ]
        aido_trainable  = [p for p in backbone_params if p.requires_grad]

        param_groups = [
            {"params": aido_trainable, "lr": hp.backbone_lr,  "name": "aido_lora"},
            {"params": other_params,   "lr": hp.head_lr,      "name": "head_and_nbr"},
        ]
        opt = torch.optim.AdamW(param_groups, weight_decay=hp.weight_decay)

        # Phase 1: linear warmup from 0.1×lr to lr over warmup_epochs
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=hp.warmup_epochs,
        )
        # Phase 2: CosineAnnealingLR
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=hp.T_max, eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_sched, cosine_sched],
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
        description="Node1-1-1-1-3-1 – AIDO.Cell-100M LoRA + STRING_GNN K=16 2-head + GenePriorBias"
    )
    parser.add_argument("--micro-batch-size",      type=int,   default=8)
    parser.add_argument("--global-batch-size",     type=int,   default=64)
    parser.add_argument("--max-epochs",            type=int,   default=300)
    parser.add_argument("--backbone-lr",           type=float, default=5e-5,
                        dest="backbone_lr")
    parser.add_argument("--head-lr",               type=float, default=2e-4,
                        dest="head_lr")
    parser.add_argument("--weight-decay",          type=float, default=3e-2)
    parser.add_argument("--head-hidden",           type=int,   default=256,
                        dest="head_hidden")
    parser.add_argument("--head-dropout",          type=float, default=0.5,
                        dest="head_dropout")
    parser.add_argument("--warmup-epochs",         type=int,   default=10)
    parser.add_argument("--t-max",                 type=int,   default=200, dest="t_max")
    parser.add_argument("--label-smoothing",       type=float, default=0.05,
                        dest="label_smoothing")
    parser.add_argument("--patience",              type=int,   default=30)
    parser.add_argument("--k-neighbors",           type=int,   default=16, dest="k_neighbors")
    parser.add_argument("--attn-dim",              type=int,   default=32, dest="attn_dim")
    parser.add_argument("--n-heads",               type=int,   default=2,  dest="n_heads")
    parser.add_argument("--lora-r",                type=int,   default=8,  dest="lora_r")
    parser.add_argument("--lora-alpha",            type=int,   default=16, dest="lora_alpha")
    parser.add_argument("--bias-warmup-epochs",    type=int,   default=50,
                        dest="bias_warmup_epochs")
    parser.add_argument("--num-workers",           type=int,   default=4)
    parser.add_argument("--debug-max-step",        type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",          action="store_true", dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Debug/limit logic
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

    # DataModule — do NOT call dm.setup() manually; Trainer will call it after DDP init
    # so that distributed barriers in setup() work correctly.
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)

    # Model
    model = AIDOCellStringGNNModel(
        head_hidden        = args.head_hidden,
        head_dropout       = args.head_dropout,
        backbone_lr        = args.backbone_lr,
        head_lr            = args.head_lr,
        weight_decay       = args.weight_decay,
        warmup_epochs      = args.warmup_epochs,
        T_max              = args.t_max,
        label_smoothing    = args.label_smoothing,
        K                  = args.k_neighbors,
        attn_dim           = args.attn_dim,
        n_heads            = args.n_heads,
        lora_r             = args.lora_r,
        lora_alpha         = args.lora_alpha,
        bias_warmup_epochs = args.bias_warmup_epochs,
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
        patience  = args.patience,  # 30 — capture late spikes like node2-1-1-1-1-1's epoch 77
        min_delta = 1e-4,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy: explicitly use SingleDeviceStrategy for fast_dev_run to avoid DDP+AMP
    # deadlocks. When torchrun sets WORLD_SIZE>1 in env, Lightning "auto" strategy may
    # still pick DDP even with devices=1. SingleDeviceStrategy bypasses DDP completely.
    local_rank_for_device = int(os.environ.get("LOCAL_RANK", "0"))
    if fast_dev_run:
        # Each torchrun process uses its own GPU independently (no DDP coordination)
        strategy = SingleDeviceStrategy(device=torch.device(f"cuda:{local_rank_for_device}"))
        devices_for_trainer = 1
    elif n_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        devices_for_trainer = n_gpus
    else:
        strategy = "auto"
        devices_for_trainer = n_gpus

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
    print(f"[Node1-1-1-1-3-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
