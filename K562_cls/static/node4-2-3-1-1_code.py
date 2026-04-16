"""Node 4-2-3-1-1 – scFoundation (6 layers fine-tuned) + STRING_GNN (frozen, cached)
             + PPI Neighborhood Attention (K=16, attn_dim=64)
             + GenePriorBias (warmup=50, reintroduced to restore output-space calibration)
             + Gene-Frequency-Weighted Loss (diversity_factor=1.5, compressed range)
             + Manual SWA (start=epoch 100, running parameter average)
             + GatedFusion + Two-Layer Head

Strategy:
- scFoundation: encode perturbed gene (nnz=1, expression=1.0) → mean-pool → [B, 768].
  Fine-tune top-6 of 12 transformer layers + final LayerNorm.
- STRING_GNN: FULLY FROZEN. Embeddings precomputed once in setup() and cached as a buffer.
  Eliminates 5.43M trainable params contributing to overfitting on 1,388 samples.
- PPI Neighborhood Attention: aggregates top-K=16 STRING neighbors of the perturbed gene
  weighted by STRING confidence + center-context gating → enriched GNN emb [B, 256].
  +229K lightweight trainable params adding PPI neighborhood context.
- GenePriorBias: learnable [3, 6640] per-gene output calibration bias. Gradient frozen
  for the first bias_warmup=50 epochs (included in graph via zero-scale for DDP compat).
  After warmup, bias learns per-gene output corrections that are ORTHOGONAL to NbAttn.
  Warmup=50 is longer than parent node4-2-3's 30 (which failed) — gives NbAttn's +229K
  params time to stabilize before the bias starts correcting distributions.
- GatedFusion (fusion_dropout=0.3): combine scFoundation + enriched GNN emb → [B, 512]
- Two-layer classification head: 512 → 256 → 19,920
- Gene-Frequency-Weighted Loss: diversity_factor=1.5 (compressed from 2.0 to reduce
  extreme per-gene weight ratios, targeting [~0.5, ~2.0] range vs [~0.3, ~3.0]).
- Manual SWA: from swa_start=100, maintain online running average of all trainable
  parameter tensors. Applied at on_fit_end() if >=5 epochs collected. Does NOT
  interfere with the cosine LR schedule (unlike Lightning's SWA callback which
  overrides the schedule mid-training).
- Loss: Per-gene-weighted CrossEntropyLoss + label_smoothing=0.1
- LR: WarmupCosine schedule, warmup=5 epochs, min_lr_ratio=0.05
- Mixup alpha=0.2 on fused embeddings (stable; avoids train<val inversion at alpha=0.4)
- weight_decay=3e-2, max_epochs=350, patience=50

Key improvements over parent (node4-2-3-1, F1=0.4832):
1. Re-introduce GenePriorBias (warmup=50): parent feedback confirms NbAttn and
   GenePriorBias CAN coexist when warmup is long enough (node4-2-2-1: F1=0.4867,
   node4-2-1-1-1: F1=0.4868, both used GenePriorBias). The parent removed it entirely,
   losing per-gene output calibration. With warmup=50 (vs failed 30 in node4-2-3),
   NbAttn has 50 extra epochs to stabilize before bias corrections activate.
2. Manual SWA (start=100): smooths the combined NbAttn+GenePriorBias weight landscape.
   node4-2-1-1-1 used SWA and achieved F1=0.4868. Starting at epoch 100 gives 50+
   epochs of SWA averaging before likely convergence around epoch 180-230.
3. diversity_factor 2.0 → 1.5: compresses gene weight range, avoiding extreme 10x
   differences between genes at weight=3.0 vs 0.3.
4. patience 40 → 50 / max_epochs 300 → 350: more training budget.
"""

from __future__ import annotations

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
from collections import defaultdict
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES     = 6640
N_CLASSES   = 3
SCF_HIDDEN  = 768
GNN_HIDDEN  = 256
FUSION_DIM  = 512
HEAD_HIDDEN = 256

SCF_MODEL_DIR = "/home/Models/scFoundation"
GNN_MODEL_DIR = "/home/Models/STRING_GNN"

CLASS_FREQ = [0.0429, 0.9251, 0.0320]  # down(-1→0), neutral(0→1), up(1→2)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Inverse-sqrt-frequency class weights to handle 92.5% neutral class dominance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_gene_freq_weights(gene_diversity_factor: float = 1.5) -> torch.Tensor:
    """Compute per-gene loss weights based on DEG variability frequency.

    diversity_factor=1.5 (vs parent's 2.0) compresses the weight range from
    [~0.3, ~3.0] to roughly [~0.5, ~2.0], reducing extreme per-gene weighting.
    """
    train_df = pd.read_csv(TRAIN_TSV, sep="\t")
    deg_counts = np.zeros(N_GENES, dtype=np.float32)
    n_samples = len(train_df)

    for row in train_df["label"]:
        labels = np.array(json.loads(row))
        deg_counts += (labels != 0).astype(np.float32)

    gene_deg_freq = deg_counts / n_samples
    mean_freq = gene_deg_freq.mean() + 1e-9
    raw_weights = 1.0 + gene_diversity_factor * (gene_deg_freq / mean_freq - 1.0)
    raw_weights = np.clip(raw_weights, 0.3, 3.0)
    raw_weights = raw_weights / raw_weights.mean()

    return torch.tensor(raw_weights, dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float32 softmax probabilities
        targets: [N, G]   int64 class indices in {0, 1, 2}
    Returns:
        Scalar F1 averaged over genes.
    """
    y_hat = preds.argmax(dim=1)  # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)
    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0).float()
        tp  = (is_pred & is_true).float().sum(0)
        fp  = (is_pred & ~is_true).float().sum(0)
        fn  = (~is_pred & is_true).float().sum(0)
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(
            prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-8), torch.zeros_like(prec)
        )
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# PPI Neighborhood Attention Module
# ---------------------------------------------------------------------------
class NeighborhoodAttentionModule(nn.Module):
    """Aggregate top-K STRING PPI neighbors of the perturbed gene.

    center_emb   = gnn_embs[g]                    [B, 256]
    neighbor_embs= gnn_embs[top-K neighbors]      [B, K, 256]
    q = q_proj(center_emb)                        [B, 1, attn_dim]
    k = k_proj(neighbor_embs)                     [B, K, attn_dim]
    attn_score = q·k/sqrt(attn_dim) + log(conf)  [B, K]
    attn = softmax(attn_score)                    [B, K]
    gate = sigmoid(gate_proj(center_emb))         [B, 256]
    context = gate * (attn @ neighbor_embs)       [B, 256]
    enriched = LayerNorm(out_proj([center; context]) + center)  [B, 256]
    """

    def __init__(self, hidden_dim: int = GNN_HIDDEN, attn_dim: int = 64) -> None:
        super().__init__()
        self.q_proj    = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.k_proj    = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_proj  = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scale = math.sqrt(attn_dim)

    def forward(
        self,
        center_emb:       torch.Tensor,  # [B, 256]
        neighbor_embs:    torch.Tensor,  # [B, K, 256]
        neighbor_weights: torch.Tensor,  # [B, K]
    ) -> torch.Tensor:
        B, K, D = neighbor_embs.shape
        q = self.q_proj(center_emb).unsqueeze(1)          # [B, 1, attn_dim]
        k = self.k_proj(neighbor_embs)                     # [B, K, attn_dim]
        raw_scores = (q @ k.transpose(-2, -1)).squeeze(1) / self.scale  # [B, K]
        log_conf_bias = torch.log(neighbor_weights.clamp(min=1e-6))
        attn = torch.softmax(raw_scores + log_conf_bias, dim=-1)
        context_raw = (attn.unsqueeze(1) @ neighbor_embs).squeeze(1)    # [B, 256]
        gate    = torch.sigmoid(self.gate_proj(center_emb))
        context = gate * context_raw
        combined = torch.cat([center_emb, context], dim=-1)              # [B, 512]
        enriched = self.out_proj(combined) + center_emb
        return self.layer_norm(enriched)                                  # [B, 256]


# ---------------------------------------------------------------------------
# Gated Fusion Module
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Element-wise gated fusion of scFoundation and GNN embeddings."""

    def __init__(
        self,
        d_scf: int = SCF_HIDDEN,
        d_gnn: int = GNN_HIDDEN,
        d_out: int = FUSION_DIM,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        d_in = d_scf + d_gnn
        self.proj_scf   = nn.Linear(d_scf, d_out)
        self.proj_gnn   = nn.Linear(d_gnn, d_out)
        self.gate_scf   = nn.Linear(d_in,  d_out)
        self.gate_gnn   = nn.Linear(d_in,  d_out)
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, scf_emb: torch.Tensor, gnn_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([scf_emb, gnn_emb], dim=-1)
        gate_s   = torch.sigmoid(self.gate_scf(combined))
        gate_g   = torch.sigmoid(self.gate_gnn(combined))
        fused    = gate_s * self.proj_scf(scf_emb) + gate_g * self.proj_gnn(gnn_emb)
        return self.dropout(self.layer_norm(fused))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List[torch.Tensor]] = (
            [
                torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
                for row in df["label"].tolist()
            ]
            if has_label else None
        )

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx": idx,
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_collate_scf(tokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
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
        self.tokenizer   = None

    def setup(self, stage: Optional[str] = None) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(SCF_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(SCF_MODEL_DIR, trust_remote_code=True)

        self.train_ds = DEGDataset(pd.read_csv(TRAIN_TSV, sep="\t"))
        self.val_ds   = DEGDataset(pd.read_csv(VAL_TSV,   sep="\t"))
        self.test_ds  = DEGDataset(pd.read_csv(TEST_TSV,  sep="\t"))

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate_scf(self.tokenizer),
        )

    def train_dataloader(self) -> DataLoader: return self._loader(self.train_ds, True)
    def val_dataloader(self)   -> DataLoader: return self._loader(self.val_ds,   False)
    def test_dataloader(self)  -> DataLoader: return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FusionDEGModel(pl.LightningModule):
    """scFoundation (top-6 layers fine-tuned) + STRING_GNN (frozen, cached)
    + PPI NeighborhoodAttention (K=16, attn_dim=64)
    + GenePriorBias (learnable [3,6640], warmup=50 epochs)
    + GatedFusion + Two-Layer Head + Gene-Frequency-Weighted Loss
    + Manual SWA (running average from epoch 100).

    Key changes vs parent node4-2-3-1:
    1. GenePriorBias ADDED back: warmup=50 (longer than node4-2-3's failed warmup=30)
       gives NbAttn time to stabilize before per-gene output calibration activates.
    2. Manual SWA: running average of trainable params from swa_start=100.
       Applied in on_fit_end(). Does not override cosine LR schedule.
    3. diversity_factor 2.0→1.5: compressed gene weight range for better balance.
    4. patience 40→50, max_epochs 300→350: more convergence budget.
    5. find_unused_parameters=True: required for GenePriorBias DDP compat during warmup.
    """

    def __init__(
        self,
        scf_finetune_layers:   int   = 6,
        head_dropout:          float = 0.5,
        fusion_dropout:        float = 0.3,
        lr:                    float = 2e-4,
        weight_decay:          float = 3e-2,
        warmup_epochs:         int   = 5,
        max_epochs:            int   = 350,
        min_lr_ratio:          float = 0.05,
        mixup_alpha:           float = 0.2,
        label_smoothing:       float = 0.1,
        nb_attn_k:             int   = 16,
        nb_attn_dim:           int   = 64,
        gene_diversity_factor: float = 1.5,
        bias_warmup:           int   = 50,
        swa_start:             int   = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        # Guard: avoid reinitializing after SWA weights have been applied at fit end.
        # When trainer.test() is called after trainer.fit(), Lightning calls setup('test')
        # again. The guard preserves the SWA-averaged parameter values in memory.
        if getattr(self, '_is_setup', False):
            return

        hp = self.hparams
        gnn_dir = Path(GNN_MODEL_DIR)

        # ----------------------------------------------------------------
        # scFoundation backbone (top-k layers fine-tuned)
        # ----------------------------------------------------------------
        self.scf = AutoModel.from_pretrained(
            SCF_MODEL_DIR,
            trust_remote_code=True,
            _use_flash_attention_2=True,
        ).to(torch.bfloat16)
        self.scf.config.use_cache = False
        self.scf.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        for param in self.scf.parameters():
            param.requires_grad = False
        n_scf_layers = len(self.scf.encoder.transformer_encoder)
        for i in range(n_scf_layers - hp.scf_finetune_layers, n_scf_layers):
            for param in self.scf.encoder.transformer_encoder[i].parameters():
                param.requires_grad = True
        for param in self.scf.encoder.norm.parameters():
            param.requires_grad = True
        for name, param in self.scf.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        scf_train = sum(p.numel() for p in self.scf.parameters() if p.requires_grad)
        scf_total = sum(p.numel() for p in self.scf.parameters())
        print(f"[Node4-2-3-1-1] scFoundation: {scf_train:,}/{scf_total:,} trainable params")

        # ----------------------------------------------------------------
        # STRING_GNN: fully frozen, embeddings precomputed and cached
        # ----------------------------------------------------------------
        print("[Node4-2-3-1-1] Precomputing STRING_GNN embeddings (frozen)...")
        gnn_temp = AutoModel.from_pretrained(str(gnn_dir), trust_remote_code=True).float()
        gnn_temp.eval()
        graph_data  = torch.load(gnn_dir / "graph_data.pt", map_location="cpu")
        edge_index  = graph_data["edge_index"].long()
        edge_weight = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.float()
        with torch.no_grad():
            gnn_out  = gnn_temp(edge_index=edge_index, edge_weight=edge_weight)
            gnn_embs = gnn_out.last_hidden_state.float().detach()
        self.register_buffer("gnn_embs_cached", gnn_embs)
        del gnn_temp
        print(f"[Node4-2-3-1-1] GNN embeddings cached: {gnn_embs.shape}")

        node_names = json.loads((gnn_dir / "node_names.json").read_text())
        self._ensembl_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(node_names)
        }

        # ----------------------------------------------------------------
        # Precompute PPI Neighborhood Indices and Weights
        # ----------------------------------------------------------------
        print(f"[Node4-2-3-1-1] Precomputing PPI neighborhood (K={hp.nb_attn_k})...")
        n_nodes = gnn_embs.shape[0]
        K = hp.nb_attn_k
        src_nodes = edge_index[0].numpy()
        tgt_nodes = edge_index[1].numpy()
        e_weights = (
            edge_weight.numpy() if edge_weight is not None
            else np.ones(len(src_nodes), dtype=np.float32)
        )
        adj: Dict = defaultdict(list)
        for s, t, w in zip(src_nodes, tgt_nodes, e_weights):
            adj[int(s)].append((int(t), float(w)))

        nb_idx_arr = np.zeros((n_nodes, K), dtype=np.int64)
        nb_wt_arr  = np.zeros((n_nodes, K), dtype=np.float32)
        for node_i in range(n_nodes):
            neighbors = adj.get(node_i, [])
            if len(neighbors) == 0:
                nb_idx_arr[node_i, :] = node_i
                nb_wt_arr[node_i, :]  = 1.0
            else:
                neighbors.sort(key=lambda x: -x[1])
                top_k = neighbors[:K]
                for j, (nb_i, nb_w) in enumerate(top_k):
                    nb_idx_arr[node_i, j] = nb_i
                    nb_wt_arr[node_i, j]  = nb_w
                for j in range(len(top_k), K):
                    nb_idx_arr[node_i, j] = node_i
                    nb_wt_arr[node_i, j]  = 0.0

        wt_sum = nb_wt_arr.sum(axis=1, keepdims=True)
        wt_sum = np.where(wt_sum > 0, wt_sum, 1.0)
        nb_wt_arr = nb_wt_arr / wt_sum

        self.register_buffer("nb_idx", torch.from_numpy(nb_idx_arr))
        self.register_buffer("nb_wt",  torch.from_numpy(nb_wt_arr))

        # ----------------------------------------------------------------
        # PPI Neighborhood Attention Module
        # ----------------------------------------------------------------
        self.nb_attn = NeighborhoodAttentionModule(
            hidden_dim=GNN_HIDDEN, attn_dim=hp.nb_attn_dim
        )
        nb_attn_params = sum(p.numel() for p in self.nb_attn.parameters())
        print(f"[Node4-2-3-1-1] NeighborhoodAttention: {nb_attn_params:,} trainable params")

        # ----------------------------------------------------------------
        # Gated Fusion Module
        # ----------------------------------------------------------------
        self.fusion = GatedFusion(
            d_scf=SCF_HIDDEN, d_gnn=GNN_HIDDEN, d_out=FUSION_DIM,
            dropout=hp.fusion_dropout,
        )

        # ----------------------------------------------------------------
        # Two-layer Classification Head: 512 → 256 → 3*6640
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Dropout(hp.head_dropout),
            nn.Linear(FUSION_DIM, HEAD_HIDDEN),
            nn.LayerNorm(HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(hp.head_dropout * 0.5),
            nn.Linear(HEAD_HIDDEN, N_CLASSES * N_GENES),
        )

        # ----------------------------------------------------------------
        # GenePriorBias: learnable [N_CLASSES, N_GENES] output calibration
        # Initialized to zeros. Gradient flows only after bias_warmup epochs.
        # During warmup, multiplied by 0 in forward() for DDP compat.
        # After warmup, added directly to logits.
        # ----------------------------------------------------------------
        self.gene_prior_bias = nn.Parameter(
            torch.zeros(N_CLASSES, N_GENES, dtype=torch.float32),
            requires_grad=True,
        )
        bias_params = self.gene_prior_bias.numel()
        print(f"[Node4-2-3-1-1] GenePriorBias: {bias_params:,} params, "
              f"warmup={hp.bias_warmup} epochs")

        # ----------------------------------------------------------------
        # Gene-Frequency-Weighted Loss weights [N_GENES]
        # ----------------------------------------------------------------
        print("[Node4-2-3-1-1] Computing gene-frequency weights (diversity_factor=1.5)...")
        gene_freq_weights = compute_gene_freq_weights(hp.gene_diversity_factor)
        self.register_buffer("gene_freq_weights", gene_freq_weights)
        gfw_min = gene_freq_weights.min().item()
        gfw_max = gene_freq_weights.max().item()
        print(f"[Node4-2-3-1-1] Gene-freq weights: min={gfw_min:.3f}, max={gfw_max:.3f}")

        # Class weights for CE loss
        self.register_buffer("class_weights", get_class_weights())

        # Validation/test accumulators
        self._val_preds:     List[torch.Tensor] = []
        self._val_tgts:      List[torch.Tensor] = []
        self._val_idx:       List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []

        # SWA accumulators (initialized lazily)
        self._swa_state: Optional[Dict[str, torch.Tensor]] = None
        self._swa_n: int = 0

        self._is_setup = True

    # ---- GNN index lookup ----
    def _get_gnn_indices(self, pert_ids: List[str], device: torch.device) -> torch.Tensor:
        indices = [self._ensembl_to_idx.get(pid, 0) for pid in pert_ids]
        return torch.tensor(indices, dtype=torch.long, device=device)

    # ---- Enriched GNN embedding via neighborhood attention ----
    def _get_enriched_gnn_emb(self, node_indices: torch.Tensor) -> torch.Tensor:
        center_emb    = self.gnn_embs_cached[node_indices]           # [B, 256]
        batch_nb_idx  = self.nb_idx[node_indices]                    # [B, K]
        batch_nb_wt   = self.nb_wt[node_indices]                     # [B, K]
        B, K = batch_nb_idx.shape
        flat_nb_idx   = batch_nb_idx.view(-1)
        neighbor_embs = self.gnn_embs_cached[flat_nb_idx].view(B, K, GNN_HIDDEN)
        return self.nb_attn(center_emb, neighbor_embs, batch_nb_wt)  # [B, 256]

    # ---- Embedding computation ----
    def get_fused_emb(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids:       List[str],
    ) -> torch.Tensor:
        device = input_ids.device
        scf_out = self.scf(input_ids=input_ids, attention_mask=attention_mask)
        scf_emb = scf_out.last_hidden_state.float().mean(dim=1)
        node_indices = self._get_gnn_indices(pert_ids, device)
        gnn_emb = self._get_enriched_gnn_emb(node_indices)
        return self.fusion(scf_emb, gnn_emb)

    # ---- Forward ----
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids:       List[str],
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        fused  = self.get_fused_emb(input_ids, attention_mask, pert_ids)
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)

        # GenePriorBias gating:
        # - Training before warmup: multiply by 0.0 to keep gene_prior_bias in the
        #   computation graph (required for DDP find_unused_parameters compatibility).
        # - Training after warmup: add full bias — per-gene output calibration active.
        # - Inference (val/test): always apply bias. Before warmup, bias ≈ 0 (no-op).
        if self.training:
            if self.current_epoch >= self.hparams.bias_warmup:
                logits = logits + self.gene_prior_bias.unsqueeze(0)
            else:
                # Zero-contribution but included in computation graph
                logits = logits + 0.0 * self.gene_prior_bias.unsqueeze(0)
        else:
            logits = logits + self.gene_prior_bias.unsqueeze(0)

        return logits

    # ---- Gene-Frequency-Weighted Loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        loss_per = F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
            reduction="none",
        ).view(B, G)
        return (loss_per * self.gene_freq_weights.unsqueeze(0)).mean()

    # ---- Training ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pert_ids       = batch["pert_id"]
        labels         = batch["labels"]
        B = input_ids.shape[0]

        fused = self.get_fused_emb(input_ids, attention_mask, pert_ids)

        if self.hparams.mixup_alpha > 0.0 and B > 1 and self.training:
            lam  = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
            perm = torch.randperm(B, device=fused.device)
            fused_mix = lam * fused + (1 - lam) * fused[perm]
            head_out  = self.head(fused_mix).view(B, N_CLASSES, N_GENES)
            # Apply GenePriorBias (with warmup gating) to mixed logits
            if self.current_epoch >= self.hparams.bias_warmup:
                head_out = head_out + self.gene_prior_bias.unsqueeze(0)
            else:
                head_out = head_out + 0.0 * self.gene_prior_bias.unsqueeze(0)
            loss = lam * self._loss(head_out, labels) + (1 - lam) * self._loss(head_out, labels[perm])
        else:
            head_out = self.head(fused).view(B, N_CLASSES, N_GENES)
            if self.current_epoch >= self.hparams.bias_warmup:
                head_out = head_out + self.gene_prior_bias.unsqueeze(0)
            else:
                head_out = head_out + 0.0 * self.gene_prior_bias.unsqueeze(0)
            loss = self._loss(head_out, labels)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True,
                 on_step=True, on_epoch=False, batch_size=B)
        return loss

    # ---- Manual SWA: update running average of trainable params ----
    def on_train_epoch_end(self) -> None:
        if self.current_epoch < self.hparams.swa_start:
            return

        # Collect current trainable parameter values
        n = self._swa_n
        with torch.no_grad():
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                current = param.detach().float()
                if self._swa_state is None or name not in self._swa_state:
                    if self._swa_state is None:
                        self._swa_state = {}
                    self._swa_state[name] = current.clone()
                else:
                    # Online averaging: avg_n+1 = (n * avg_n + x_n+1) / (n+1)
                    self._swa_state[name] = (
                        self._swa_state[name] * (n / (n + 1)) +
                        current * (1.0 / (n + 1))
                    )
        self._swa_n = n + 1

        if (self.current_epoch - self.hparams.swa_start) % 20 == 0:
            self.print(
                f"[Node4-2-3-1-1] SWA update: epoch={self.current_epoch}, "
                f"n_averaged={self._swa_n}"
            )

    # ---- Apply SWA weights at end of training ----
    def on_fit_end(self) -> None:
        if self._swa_state is not None and self._swa_n >= 5:
            self.print(
                f"[Node4-2-3-1-1] Applying manual SWA: {self._swa_n} epochs averaged "
                f"(epochs {self.hparams.swa_start} to "
                f"{self.hparams.swa_start + self._swa_n - 1})"
            )
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    if name in self._swa_state:
                        param.copy_(
                            self._swa_state[name].to(param.dtype).to(param.device)
                        )
        else:
            n = self._swa_n if self._swa_state is not None else 0
            self.print(
                f"[Node4-2-3-1-1] SWA not applied: only {n} averaged epochs (need >=5)"
            )

    # ---- Validation ----
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True, batch_size=batch["input_ids"].shape[0])
            probs = torch.softmax(logits.float(), dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        local_idx   = torch.cat(self._val_idx,   0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        order = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([
            torch.ones(1, dtype=torch.bool, device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    # ---- Test ----
    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]),
                     sync_dist=True, batch_size=logits.shape[0])

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)

        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        if is_dist:
            all_preds    = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
            all_pert_ids = [None] * self.trainer.world_size
            all_symbols  = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(all_pert_ids, self._test_pert_ids)
            torch.distributed.all_gather_object(all_symbols,  self._test_symbols)
            flat_pids = [p for rank_pids in all_pert_ids for p in rank_pids]
            flat_syms = [s for rank_syms in all_symbols  for s in rank_syms]
        else:
            all_preds = local_preds
            flat_pids = self._test_pert_ids
            flat_syms = self._test_symbols

        if self.trainer.is_global_zero:
            seen_pids = {}
            dedup_rows = []
            for i in range(all_preds.shape[0]):
                pid = flat_pids[i]
                if pid not in seen_pids:
                    seen_pids[pid] = True
                    dedup_rows.append({
                        "idx":        pid,
                        "input":      flat_syms[i],
                        "prediction": json.dumps(all_preds[i].float().cpu().numpy().tolist()),
                    })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(dedup_rows).to_csv(
                out_dir / "test_predictions.tsv", sep="\t", index=False
            )
            print(f"[Node4-2-3-1-1] Saved {len(dedup_rows)} test predictions → "
                  f"{out_dir}/test_predictions.tsv")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    # ---- Checkpoint: save only trainable params + all buffers ----
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
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"[Node4-2-3-1-1] Checkpoint: {trained:,}/{total:,} params "
            f"({100*trained/total:.2f}%)"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer with WarmupCosine LR scheduler ----
    def configure_optimizers(self):
        hp = self.hparams

        # Separate param groups: GenePriorBias uses no weight decay
        # (bias parameters should not be L2 regularized — they calibrate output distributions)
        bias_params  = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name == "gene_prior_bias":
                bias_params.append(param)
            else:
                other_params.append(param)

        opt = torch.optim.AdamW([
            {"params": other_params, "lr": hp.lr, "weight_decay": hp.weight_decay},
            {"params": bias_params,  "lr": hp.lr, "weight_decay": 0.0},
        ])

        def lr_lambda(epoch: int) -> float:
            if epoch < hp.warmup_epochs:
                return max(1e-8, epoch / max(1, hp.warmup_epochs))
            progress = (epoch - hp.warmup_epochs) / max(1, hp.max_epochs - hp.warmup_epochs)
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return hp.min_lr_ratio + (1.0 - hp.min_lr_ratio) * cosine

        # Apply the same cosine schedule to both param groups
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=[lr_lambda, lr_lambda]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node4-2-3-1-1 – scFoundation + STRING_GNN (frozen) + "
                    "NbAttn + GenePriorBias(warmup=50) + Gene-Freq Loss + Manual SWA"
    )
    parser.add_argument("--micro-batch-size",      type=int,   default=8)
    parser.add_argument("--global-batch-size",     type=int,   default=64)
    parser.add_argument("--max-epochs",            type=int,   default=350)
    parser.add_argument("--lr",                    type=float, default=2e-4)
    parser.add_argument("--weight-decay",          type=float, default=3e-2)
    parser.add_argument("--scf-finetune-layers",   type=int,   default=6,
                        dest="scf_finetune_layers")
    parser.add_argument("--head-dropout",          type=float, default=0.5)
    parser.add_argument("--fusion-dropout",        type=float, default=0.3)
    parser.add_argument("--warmup-epochs",         type=int,   default=5)
    parser.add_argument("--min-lr-ratio",          type=float, default=0.05)
    parser.add_argument("--mixup-alpha",           type=float, default=0.2)
    parser.add_argument("--label-smoothing",       type=float, default=0.1)
    parser.add_argument("--nb-attn-k",             type=int,   default=16,
                        dest="nb_attn_k")
    parser.add_argument("--nb-attn-dim",           type=int,   default=64,
                        dest="nb_attn_dim")
    parser.add_argument("--gene-diversity-factor", type=float, default=1.5,
                        dest="gene_diversity_factor")
    parser.add_argument("--bias-warmup",           type=int,   default=50,
                        dest="bias_warmup")
    parser.add_argument("--swa-start",             type=int,   default=100,
                        dest="swa_start")
    parser.add_argument("--num-workers",           type=int,   default=4)
    parser.add_argument("--debug-max-step",        type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--debug_max_step",        type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",          action="store_true", dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = lim_val = lim_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = FusionDEGModel(
        scf_finetune_layers   = args.scf_finetune_layers,
        head_dropout          = args.head_dropout,
        fusion_dropout        = args.fusion_dropout,
        lr                    = args.lr,
        weight_decay          = args.weight_decay,
        warmup_epochs         = args.warmup_epochs,
        max_epochs            = args.max_epochs,
        min_lr_ratio          = args.min_lr_ratio,
        mixup_alpha           = args.mixup_alpha,
        label_smoothing       = args.label_smoothing,
        nb_attn_k             = args.nb_attn_k,
        nb_attn_dim           = args.nb_attn_dim,
        gene_diversity_factor = args.gene_diversity_factor,
        bias_warmup           = args.bias_warmup,
        swa_start             = args.swa_start,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
        auto_insert_metric_name=False,
    )
    # patience=50: more budget vs parent's 40
    es_cb  = EarlyStopping(monitor="val/f1", mode="max", patience=50, min_delta=1e-4)
    lr_cb  = LearningRateMonitor(logging_interval="epoch")
    pg_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # find_unused_parameters=True required: during warmup epochs, gene_prior_bias
    # contributes 0 to logits via "0.0 * bias", which DDP needs to handle correctly.
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
        val_check_interval      = 1.0,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)

    # After fit(), on_fit_end() has applied SWA-averaged weights to the model.
    # We test with the SWA model (in memory) rather than reloading the best checkpoint,
    # because the SWA average provides a smoother, better-generalizing parameter set.
    # Exception: debug modes still test directly (no meaningful SWA in short runs).
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=dm)
    else:
        # Use SWA model: do NOT reload ckpt_path='best' (which would overwrite SWA weights)
        # The _is_setup guard in setup() prevents model reinitialization during test.
        test_results = trainer.test(model, datamodule=dm)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        if test_results:
            test_loss = test_results[0].get("test/loss", float("nan"))
            score_path.write_text(f"test/loss={test_loss:.6f}\n")
        print(f"[Node4-2-3-1-1] Done. Check run/ for test_predictions.tsv")


if __name__ == "__main__":
    main()
