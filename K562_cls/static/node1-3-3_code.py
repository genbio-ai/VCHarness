"""Node 1-3-3: AIDO.Cell-10M LoRA (online fine-tuned) + Frozen STRING_GNN (Neighborhood Attention K=16) + GenePriorBias.

Strategy: Introduce AIDO.Cell-10M with LoRA fine-tuning as a learnable perturbation-specific
transcriptional encoder, combined with frozen STRING_GNN + neighborhood attention (proven K=16
architecture from node1 lineage), a simple fusion MLP, bilinear gene-class head, and GenePriorBias
calibration.

Key innovations over parent (node1-3, F1=0.4669) and siblings:
- node1-3-1 (F1=0.4689): keeps frozen scFoundation (confirmed dead-end — 3 experiments)
- node1-3-2 (sibling): STRING-only + GenePriorBias + discriminative LR (no cell encoder)
- node1-3-3 (this): AIDO.Cell-10M LoRA online fine-tuning + frozen STRING_GNN + GenePriorBias
  Distinctly different: uses AIDO.Cell (best node in tree uses it) instead of frozen scFoundation

Memory connections:
- node2-1-1-1 (F1=0.5059): best overall = AIDO.Cell LoRA + STRING_GNN K=16 → directly inspired
- node1-2-2-1 (F1=0.4829): GenePriorBias proven +0.006 F1 over STRING-only baseline
- node1-1-1-1-1 (F1=0.4846): discriminative LR + neighborhood attention proven effective
- node1-3-1 feedback: confirms frozen embeddings are dead-end; fine-tuned encoder is key
- node4-2 (F1=0.4801): confirms that fine-tuned (not frozen) scFoundation works → generalizes to AIDO.Cell LoRA
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
N_GENES = 6640
N_CLASSES = 3
# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
AIDO_CELL_DIR  = Path("/home/Models/AIDO.Cell-10M")  # hidden_size=256, 8 layers


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
        targets: [N, G]    long   – class labels in {0, 1, 2}
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
# Pre-computation utilities (STRING_GNN only; AIDO.Cell is online)
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_string_gnn_embeddings() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load STRING_GNN and compute all node embeddings. Returns (emb[N,256], pert_id→idx)."""
    import json as _json

    model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
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
        neighbor_indices [N, K] long   — STRING_GNN node indices (-1 = padding)
        neighbor_weights [N, K] float  — normalized STRING confidence weights
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
# Neighborhood Attention Aggregator (identical to proven node1-2 design)
# ---------------------------------------------------------------------------
class NeighborhoodAttentionAggregator(nn.Module):
    """Aggregate top-K PPI neighbors for a center gene using learned attention."""

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
        neighbor_weights: torch.Tensor, # [B, K]  pre-normalized edge weights
        valid_mask: torch.Tensor,       # [B, K]  bool
    ) -> torch.Tensor:
        """Returns aggregated representation [B, D]."""
        B, K, D = neighbor_emb.shape
        center_exp = center_emb.unsqueeze(1).expand(-1, K, -1)   # [B, K, D]
        pair = torch.cat([center_exp, neighbor_emb], dim=-1)      # [B, K, 2D]
        attn_scores = self.attn_proj(pair).squeeze(-1)            # [B, K]

        # Combine learned scores with STRING confidence as prior
        attn_scores = attn_scores + neighbor_weights
        attn_scores = attn_scores.masked_fill(~valid_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)         # [B, K]
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
    def __init__(self, batch_size: int = 16, num_workers: int = 4) -> None:
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
class AidoCellDEGModel(pl.LightningModule):
    """AIDO.Cell-10M LoRA + Frozen STRING_GNN (Neighborhood Attention K=16)
    + GenePriorBias calibration + Bilinear gene-class head.

    Architecture:
      pert_id → AIDO.Cell-10M (LoRA, r=16) → mean_pool [:19264] → [B, 256]
             → STRING_GNN (frozen cached) → NeighborhoodAttn (K=16) → [B, 256]
             → per-modality LayerNorm → concat [B, 512]
             → FusionMLP: Linear(512→256) → GELU → LN → Dropout → [B, 256]
             → Bilinear: h[B,256] · gene_class_emb[3,6640,256] → logits[B,3,6640]
             → GenePriorBias [3, 6640] (active after epoch gene_prior_warmup)
    """

    def __init__(
        self,
        bilinear_dim: int = 256,
        attn_dim: int = 64,
        K: int = 16,
        dropout: float = 0.35,
        aido_lr: float = 5e-5,
        attn_lr: float = 3e-4,
        head_lr: float = 1e-4,
        weight_decay: float = 3e-2,
        warmup_epochs: int = 20,
        t_max: int = 120,
        label_smoothing: float = 0.05,
        gene_prior_warmup: int = 20,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # ---- Collect all pert_ids across all splits ----
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

        # ---- AIDO.Cell-10M with LoRA ----
        # Rank-0 downloads first; other ranks wait at barrier then load
        if local_rank == 0:
            AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.print("Loading AIDO.Cell-10M tokenizer...")
        self.aido_tokenizer = AutoTokenizer.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)

        self.print("Loading AIDO.Cell-10M model and applying LoRA...")
        aido_base = AutoModel.from_pretrained(str(AIDO_CELL_DIR), trust_remote_code=True)
        aido_base = aido_base.to(torch.bfloat16)

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,  # fine-tune all 8 layers
        )
        aido_base = get_peft_model(aido_base, lora_cfg)
        aido_base.config.use_cache = False
        aido_base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.aido_cell = aido_base
        # AIDO.Cell-10M hidden_size = 256; matches STRING_GNN output dim for clean fusion
        self._aido_hidden = 256

        # ---- Trainable fusion and prediction modules ----
        self.neighborhood_attn = NeighborhoodAttentionAggregator(
            emb_dim=256, attn_dim=hp.attn_dim
        )

        # Per-modality LayerNorm before fusion (scale-normalizes both embeddings)
        self.cell_norm = nn.LayerNorm(self._aido_hidden)
        self.gnn_norm  = nn.LayerNorm(256)

        # Simple concat fusion: [256 + 256 = 512] → 256
        # Simpler than GatedFusion (which failed) but still captures cross-modal interactions
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self._aido_hidden + 256, hp.bilinear_dim),
            nn.GELU(),
            nn.LayerNorm(hp.bilinear_dim),
            nn.Dropout(hp.dropout),
        )

        # Bilinear gene-class head: logits[b,c,g] = h[b] · gene_class_emb[c,g]
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # GenePriorBias [N_CLASSES, N_GENES]: per-gene per-class additive bias
        # Initialized from 0.3 × log(class_freq) for class-frequency prior
        class_freq = torch.tensor(CLASS_FREQ, dtype=torch.float32)
        log_freq   = torch.log(class_freq + 1e-9)         # [3]
        # gene_prior_bias shape [N_CLASSES, N_GENES] for direct addition to logits [B, N_CLASSES, N_GENES]
        bias_init  = 0.3 * log_freq.unsqueeze(1).expand(-1, N_GENES)  # [3, 6640]
        self.gene_prior_bias = nn.Parameter(bias_init.clone())         # [3, N_GENES]

        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        # LoRA adapters start as bf16 (from AIDO.Cell-10M bfloat16), cast to float32
        # Frozen AIDO.Cell backbone stays in bf16
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Accumulators for val/test
        self._val_preds: List[torch.Tensor] = []
        self._val_tgts:  List[torch.Tensor] = []
        self._val_idx:   List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    # ---- Embedding helpers ----

    def _get_neighborhood_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Return [B, 256] neighborhood-attention-aggregated STRING_GNN embeddings."""
        pos = torch.tensor(
            [self.pert_to_pos[pid] for pid in pert_ids], dtype=torch.long, device=self.device
        )
        gnn_node_idx = self.pert_gnn_idx[pos]      # [B]
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

    def _get_aido_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Return [B, 256] AIDO.Cell-10M embeddings via LoRA fine-tuning (online).

        Each perturbation is encoded as a single active gene (expression=1.0);
        all other 19,263 genes receive -1.0 (missing) from the tokenizer automatically.
        Output: mean pool of last_hidden_state[:, :19264, :] → [B, 256].
        """
        # Tokenize: each pert_id encoded as single-gene expression profile
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized = self.aido_tokenizer(expr_dicts, return_tensors="pt")

        # input_ids: [B, 19264] float32 (raw expression values, -1.0 for missing genes)
        # attention_mask: [B, 19264] int64 (always all-ones; overridden inside model)
        input_ids      = tokenized["input_ids"].to(self.device)       # float32
        attention_mask = tokenized["attention_mask"].to(self.device)  # int64

        outputs = self.aido_cell(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last_hidden_state: [B, 19266, 256]  (+2 summary tokens appended by _prepare_inputs)
        # Slice [:, :19264, :] to stay in gene-position space, then mean pool
        cell_emb = outputs.last_hidden_state[:, :19264, :].mean(dim=1)  # [B, 256]
        return cell_emb.float()   # cast bfloat16 → float32

    # ---- Forward ----

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        # Get embeddings from both encoders
        cell_emb = self._get_aido_emb(pert_ids)           # [B, 256] float32
        gnn_emb  = self._get_neighborhood_emb(pert_ids)   # [B, 256] float32

        # Per-modality normalization to equalize scales before fusion
        cell_emb = self.cell_norm(cell_emb)   # [B, 256]
        gnn_emb  = self.gnn_norm(gnn_emb)     # [B, 256]

        # Concat + MLP fusion
        fused = torch.cat([cell_emb, gnn_emb], dim=-1)   # [B, 512]
        h = self.fusion_mlp(fused)                         # [B, 256]

        # Bilinear gene-class head: captures perturbation-gene co-regulation
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb)  # [B, 3, G]

        # GenePriorBias: active only after warmup epochs to avoid cold-start failure
        # gene_prior_bias: [3, G]; not in computation graph during warmup → zero gradient
        if self.current_epoch >= self.hparams.gene_prior_warmup:
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
            self.print(f"[Node1-3-3] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ---- Checkpoint helpers ----

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters + persistent buffers (excludes frozen AIDO.Cell backbone)."""
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

        # Three discriminative LR groups:
        # 1. AIDO.Cell LoRA adapters: low LR (pre-trained model, fine-tune conservatively)
        # 2. Neighborhood attention: medium LR (learns PPI context routing)
        # 3. Fusion MLP + bilinear head + GenePriorBias: conservative LR (large param count)
        aido_params = []
        attn_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "aido_cell" in name:
                aido_params.append(param)
            elif "neighborhood_attn" in name:
                attn_params.append(param)
            else:
                # cell_norm, gnn_norm, fusion_mlp, gene_class_emb, gene_prior_bias, fallback_emb
                head_params.append(param)

        self.print(
            f"Param groups — AIDO LoRA: {sum(p.numel() for p in aido_params):,}, "
            f"Attn: {sum(p.numel() for p in attn_params):,}, "
            f"Head+Fusion: {sum(p.numel() for p in head_params):,}"
        )

        param_groups = [
            {"params": aido_params, "lr": hp.aido_lr,  "name": "aido_lora"},
            {"params": attn_params, "lr": hp.attn_lr,  "name": "neighborhood_attn"},
            {"params": head_params, "lr": hp.head_lr,  "name": "head_and_fusion"},
        ]

        opt = torch.optim.AdamW(param_groups, weight_decay=hp.weight_decay)

        warmup_sch = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=hp.warmup_epochs,
        )
        cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=hp.t_max, eta_min=5e-6,
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
        description="Node1-3-3: AIDO.Cell-10M LoRA + Frozen STRING_GNN + GenePriorBias"
    )
    # Batch size
    parser.add_argument("--micro-batch-size",  type=int,   default=16,
                        help="Per-GPU micro batch size (AIDO.Cell-10M safe at B=16 on H100)")
    parser.add_argument("--global-batch-size", type=int,   default=256,
                        help="Global batch size (must be multiple of micro_batch_size*8)")
    # Training schedule
    parser.add_argument("--max-epochs",        type=int,   default=220)
    parser.add_argument("--warmup-epochs",     type=int,   default=20)
    parser.add_argument("--t-max",             type=int,   default=120)
    # Discriminative learning rates
    parser.add_argument("--aido-lr",           type=float, default=5e-5,
                        help="LR for AIDO.Cell LoRA adapters")
    parser.add_argument("--attn-lr",           type=float, default=3e-4,
                        help="LR for neighborhood attention module")
    parser.add_argument("--head-lr",           type=float, default=1e-4,
                        help="LR for fusion MLP + bilinear head + GenePriorBias")
    parser.add_argument("--weight-decay",      type=float, default=3e-2)
    # Architecture
    parser.add_argument("--bilinear-dim",      type=int,   default=256)
    parser.add_argument("--attn-dim",          type=int,   default=64)
    parser.add_argument("--K",                 type=int,   default=16)
    parser.add_argument("--dropout",           type=float, default=0.35)
    parser.add_argument("--label-smoothing",   type=float, default=0.05)
    parser.add_argument("--gene-prior-warmup", type=int,   default=20)
    # LoRA
    parser.add_argument("--lora-r",            type=int,   default=16)
    parser.add_argument("--lora-alpha",        type=int,   default=32)
    parser.add_argument("--lora-dropout",      type=float, default=0.05)
    # EarlyStopping
    parser.add_argument("--patience",          type=int,   default=15)
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
        max_epochs_for_trainer = 1  # Quick debug: only 1 epoch
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
    )
    early_stop_callback = EarlyStopping(
        monitor="val/f1",
        patience=args.patience,
        mode="max",
        min_delta=2e-4,
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
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=180)),
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
    )

    # ---- Model and DataModule ----
    model = AidoCellDEGModel(
        bilinear_dim=args.bilinear_dim,
        attn_dim=args.attn_dim,
        K=args.K,
        dropout=args.dropout,
        aido_lr=args.aido_lr,
        attn_lr=args.attn_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        t_max=args.t_max,
        label_smoothing=args.label_smoothing,
        gene_prior_warmup=args.gene_prior_warmup,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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
    print(f"[Node1-3-3] Test results saved to {score_path}")


if __name__ == "__main__":
    main()
