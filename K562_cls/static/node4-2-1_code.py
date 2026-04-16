"""Node 4-2-1 – scFoundation (6L) + STRING_GNN (frozen, cached, neighborhood attention) + GatedFusion

Improvements over node4-2 (F1=0.4801):
1. PPI Neighborhood Attention (K=16, attn_dim=64): Enrich frozen GNN embeddings by
   attention-weighting top-16 STRING PPI neighbors per perturbed gene.
   This injects the same proven improvement vector as node1-1-1-1 → node1-1-1-1-1
   (+0.010 F1) into the fusion model.  Only ~229K new trainable params.
2. Extended training: max_epochs=300, patience=35
   (node4-2 peaked at epoch 139 and stopped at 165 — budget was the bottleneck)
3. Reduced warmup: warmup_epochs=10 → 5 (faster cosine decay start)
4. Lower LR floor: min_lr_ratio=0.15 → 0.08 (deeper convergence in low-LR regime)
5. Stronger Mixup: alpha=0.2 → 0.4 (more interpolative mixing for 1,388-sample dataset)

Architecture:
  pert_id → scFoundation(top-6 layers fine-tuned)  → mean-pool → [B,768]
  pert_id → STRING_GNN(frozen,cached)[18870,256]
           → NeighborhoodAttention(K=16, attn_dim=64) → [B,256]
  → GatedFusion(768+256→512) + LayerNorm + Dropout(0.3)
  → [optional Mixup during training]
  → Head: Dropout(0.5)→Linear(512→256)→LN→GELU→Dropout(0.25)→Linear(256→19920)
  → [B, 3, 6640] logits
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
SCF_HIDDEN  = 768    # scFoundation hidden size
GNN_HIDDEN  = 256    # STRING_GNN hidden size
FUSION_DIM  = 512    # Gated fusion output dimension
HEAD_HIDDEN = 256    # Two-layer head intermediate dimension
K_NEIGHBORS = 16     # PPI neighborhood size (proven best in node1-1-1-1-1)
ATTN_DIM    = 64     # Neighborhood attention projection dimension (proven in node1-1-1-1-1)

SCF_MODEL_DIR = "/home/Models/scFoundation"
GNN_MODEL_DIR = "/home/Models/STRING_GNN"

CLASS_FREQ = [0.0429, 0.9251, 0.0320]   # down(-1→0), neutral(0→1), up(1→2)

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
    """Attention-weighted PPI neighborhood aggregation.

    For each perturbed gene, aggregates top-K STRING PPI neighbors using
    learned attention (dot-product with STRING confidence weighting) and
    center-context gating.  Operates on frozen cached GNN embeddings.

    Architecture (from node1-1-1-1-1's proven design):
      center_emb: [B, D]
      neighbor_embs: [B, K, D]
      q = q_proj(center)        [B, 1, A]
      k = k_proj(neighbors)     [B, K, A]
      attn = softmax(q·k/√A + log(edge_weights))   [B, K]
      gate = sigmoid(W_g · center)                   [B, D]
      context = gate ⊙ Σ_k attn_k · neighbor_k      [B, D]
      output = LayerNorm(out_proj([center; context]) + center)
    """

    def __init__(
        self,
        d_model:    int = GNN_HIDDEN,
        attn_dim:   int = ATTN_DIM,
        k_neighbors: int = K_NEIGHBORS,
    ) -> None:
        super().__init__()
        self.k     = k_neighbors
        self.scale = attn_dim ** -0.5
        self.q_proj  = nn.Linear(d_model, attn_dim, bias=False)
        self.k_proj  = nn.Linear(d_model, attn_dim, bias=False)
        self.gate    = nn.Linear(d_model, d_model)
        self.out     = nn.Linear(d_model * 2, d_model)
        self.norm    = nn.LayerNorm(d_model)

    def forward(
        self,
        all_embs:    torch.Tensor,  # [N_nodes, D] — cached frozen GNN embeddings
        center_idx:  torch.Tensor,  # [B]           — indices of perturbed genes
        nb_idx:      torch.Tensor,  # [B, K]         — top-K neighbor indices
        nb_weights:  torch.Tensor,  # [B, K]         — STRING confidence weights (0..1)
    ) -> torch.Tensor:              # [B, D]
        B, K = nb_idx.shape
        D = all_embs.shape[1]

        center = all_embs[center_idx]                             # [B, D]
        nbs    = all_embs[nb_idx.reshape(-1)].view(B, K, D)       # [B, K, D]

        # Dot-product attention biased by STRING confidence
        q    = self.q_proj(center).unsqueeze(1)                   # [B, 1, A]
        k    = self.k_proj(nbs)                                   # [B, K, A]
        attn = (q * k).sum(-1) * self.scale                       # [B, K]
        attn = attn + torch.log(nb_weights.clamp(min=1e-6))       # bias by log-confidence
        attn = torch.softmax(attn, dim=-1)                        # [B, K]

        # Center-context gating
        gate    = torch.sigmoid(self.gate(center))                # [B, D]
        context = (attn.unsqueeze(-1) * nbs).sum(dim=1)           # [B, D]
        context = gate * context                                  # [B, D]

        # Project concatenation, then add residual
        combined = torch.cat([center, context], dim=-1)           # [B, 2D]
        enriched = self.out(combined)                             # [B, D]
        return self.norm(enriched + center)                       # residual + norm


# ---------------------------------------------------------------------------
# Gated Fusion Module
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Element-wise gated fusion of two heterogeneous embeddings.

    output = LayerNorm(Dropout(gate_scf ⊙ proj_scf(scf) + gate_gnn ⊙ proj_gnn(gnn)))
    where gates are computed from the concatenation of both inputs.
    """

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
        combined = torch.cat([scf_emb, gnn_emb], dim=-1)     # [B, d_scf+d_gnn]
        gate_s   = torch.sigmoid(self.gate_scf(combined))     # [B, d_out]
        gate_g   = torch.sigmoid(self.gate_gnn(combined))     # [B, d_out]
        fused    = gate_s * self.proj_scf(scf_emb) + gate_g * self.proj_gnn(gnn_emb)
        return self.dropout(self.layer_norm(fused))            # [B, d_out]


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
    """Collate function that tokenizes inputs for scFoundation."""

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
    """scFoundation (top-6 layers fine-tuned) + STRING_GNN (frozen, cached, neighborhood attn)
    + GatedFusion.

    New vs. node4-2:
    1. NeighborhoodAttentionModule enriches cached GNN embeddings with K=16 PPI neighbor context.
    2. Extended training schedule (max_epochs=300, patience=35).
    3. Adjusted LR: warmup_epochs=5, min_lr_ratio=0.08 for deeper convergence.
    4. Stronger Mixup: alpha=0.4 for richer data augmentation.
    """

    def __init__(
        self,
        scf_finetune_layers: int = 6,
        head_dropout: float      = 0.5,
        fusion_dropout: float    = 0.3,
        lr: float                = 2e-4,
        weight_decay: float      = 3e-2,
        warmup_epochs: int       = 5,
        max_epochs: int          = 300,
        min_lr_ratio: float      = 0.08,
        mixup_alpha: float       = 0.4,
        label_smoothing: float   = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
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

        # Freeze all scF params, then unfreeze top-k transformer layers
        for param in self.scf.parameters():
            param.requires_grad = False
        n_scf_layers = len(self.scf.encoder.transformer_encoder)
        for i in range(n_scf_layers - hp.scf_finetune_layers, n_scf_layers):
            for param in self.scf.encoder.transformer_encoder[i].parameters():
                param.requires_grad = True
        # Unfreeze final LayerNorm
        for param in self.scf.encoder.norm.parameters():
            param.requires_grad = True
        # Cast trainable scF params to float32 for stable optimization
        for name, param in self.scf.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        scf_train = sum(p.numel() for p in self.scf.parameters() if p.requires_grad)
        scf_total = sum(p.numel() for p in self.scf.parameters())
        print(f"[Node4-2-1] scFoundation: {scf_train:,}/{scf_total:,} trainable params")

        # ----------------------------------------------------------------
        # STRING_GNN: fully frozen, embeddings precomputed and cached
        # ----------------------------------------------------------------
        print("[Node4-2-1] Precomputing STRING_GNN embeddings (frozen)...")
        gnn_temp    = AutoModel.from_pretrained(str(gnn_dir), trust_remote_code=True).float()
        gnn_temp.eval()
        graph_data  = torch.load(gnn_dir / "graph_data.pt", map_location="cpu")
        edge_index  = graph_data["edge_index"].long()
        edge_weight = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.float()
        with torch.no_grad():
            gnn_out  = gnn_temp(edge_index=edge_index, edge_weight=edge_weight)
            gnn_embs = gnn_out.last_hidden_state.float().detach()   # [18870, 256]
        # Register as a buffer → auto-moved to GPU by Lightning
        self.register_buffer("gnn_embs_cached", gnn_embs)
        del gnn_temp   # free memory
        print(f"[Node4-2-1] GNN embeddings cached: {gnn_embs.shape}")

        # ----------------------------------------------------------------
        # Precompute top-K PPI neighbors for each node
        # ----------------------------------------------------------------
        print(f"[Node4-2-1] Building K={K_NEIGHBORS} neighbor lookup table...")
        N_nodes = gnn_embs.shape[0]
        src_np  = edge_index[0].numpy()
        dst_np  = edge_index[1].numpy()
        wt_np   = (edge_weight.numpy().astype(np.float32)
                   if edge_weight is not None
                   else np.ones(src_np.shape[0], dtype=np.float32))

        # Sort edges by (src ascending, weight descending) for efficient top-K extraction
        sort_idx  = np.lexsort((-wt_np, src_np))
        src_s     = src_np[sort_idx]
        dst_s     = dst_np[sort_idx]
        wt_s      = wt_np[sort_idx]

        # Find edge range for each source node
        boundaries = np.searchsorted(src_s, np.arange(N_nodes + 1))

        nb_idx_arr = np.zeros((N_nodes, K_NEIGHBORS), dtype=np.int64)
        nb_wt_arr  = np.zeros((N_nodes, K_NEIGHBORS), dtype=np.float32)

        for node in range(N_nodes):
            start, end = int(boundaries[node]), int(boundaries[node + 1])
            count = min(end - start, K_NEIGHBORS)
            if count > 0:
                nb_idx_arr[node, :count] = dst_s[start:start + count]
                nb_wt_arr[node, :count]  = wt_s[start:start + count]
            # Pad remaining slots with self-loop (zero weight handled in attention log)
            for j in range(count, K_NEIGHBORS):
                nb_idx_arr[node, j] = node
                nb_wt_arr[node, j]  = 1e-6  # near-zero weight for padding

        self.register_buffer("nb_idx", torch.from_numpy(nb_idx_arr))
        self.register_buffer("nb_wt",  torch.from_numpy(nb_wt_arr))
        print(f"[Node4-2-1] Neighbor table built: {nb_idx_arr.shape}")

        # Build Ensembl ID → node index lookup
        node_names = json.loads((gnn_dir / "node_names.json").read_text())
        self._ensembl_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(node_names)
        }

        # ----------------------------------------------------------------
        # Neighborhood Attention Module (new in node4-2-1)
        # ----------------------------------------------------------------
        self.nb_attn = NeighborhoodAttentionModule(
            d_model=GNN_HIDDEN, attn_dim=ATTN_DIM, k_neighbors=K_NEIGHBORS
        )
        nb_params = sum(p.numel() for p in self.nb_attn.parameters())
        print(f"[Node4-2-1] NeighborhoodAttention: {nb_params:,} trainable params")

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

        # Class weights for weighted CE loss
        self.register_buffer("class_weights", get_class_weights())

        # Accumulators for validation / test
        self._val_preds:     List[torch.Tensor] = []
        self._val_tgts:      List[torch.Tensor] = []
        self._val_idx:       List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []

    # ---- GNN index lookup ----
    def _get_gnn_indices(self, pert_ids: List[str], device: torch.device) -> torch.Tensor:
        """Look up STRING_GNN node indices for a batch of Ensembl gene IDs."""
        indices = [self._ensembl_to_idx.get(pid, 0) for pid in pert_ids]
        return torch.tensor(indices, dtype=torch.long, device=device)

    # ---- Embedding computation ----
    def get_fused_emb(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids:       List[str],
    ) -> torch.Tensor:
        """Compute fused embedding: scFoundation + neighborhood-enriched GNN → GatedFusion."""
        device = input_ids.device

        # 1. scFoundation → mean-pool over sequence tokens
        scf_out = self.scf(input_ids=input_ids, attention_mask=attention_mask)
        scf_emb = scf_out.last_hidden_state.float().mean(dim=1)   # [B, 768]

        # 2. STRING_GNN: cached embeddings enriched with neighborhood attention
        node_indices = self._get_gnn_indices(pert_ids, device)
        nb_idx_b = self.nb_idx[node_indices]                       # [B, K]
        nb_wt_b  = self.nb_wt[node_indices]                        # [B, K]
        gnn_emb  = self.nb_attn(
            self.gnn_embs_cached, node_indices, nb_idx_b, nb_wt_b
        )                                                           # [B, 256]

        # 3. Gated fusion → [B, 512]
        return self.fusion(scf_emb, gnn_emb)

    # ---- Forward ----
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids:       List[str],
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        fused = self.get_fused_emb(input_ids, attention_mask, pert_ids)
        return self.head(fused).view(B, N_CLASSES, N_GENES)

    # ---- Loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted CE with label smoothing — proven better F1 alignment than focal loss."""
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, C]
            targets.reshape(-1),                       # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ---- Training ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pert_ids       = batch["pert_id"]
        labels         = batch["labels"]
        B = input_ids.shape[0]

        fused = self.get_fused_emb(input_ids, attention_mask, pert_ids)

        # Mixup augmentation (alpha=0.4 for stronger interpolation)
        if self.hparams.mixup_alpha > 0.0 and B > 1 and self.training:
            lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
            perm = torch.randperm(B, device=fused.device)
            fused_mix = lam * fused + (1 - lam) * fused[perm]
            logits = self.head(fused_mix).view(B, N_CLASSES, N_GENES)
            loss = lam * self._loss(logits, labels) + (1 - lam) * self._loss(logits, labels[perm])
        else:
            logits = self.head(fused).view(B, N_CLASSES, N_GENES)
            loss = self._loss(logits, labels)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    # ---- Validation ----
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
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
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        local_idx   = torch.cat(self._val_idx,   0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        # Deduplicate (DDP may duplicate samples at epoch boundaries)
        order  = torch.argsort(all_idx)
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
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

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
            n    = all_preds.shape[0]
            rows = []
            for i in range(n):
                rows.append({
                    "idx":        flat_pids[i],
                    "input":      flat_syms[i],
                    "prediction": json.dumps(all_preds[i].float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node4-2-1] Saved {len(rows)} test predictions → {out_dir}/test_predictions.tsv")

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
            f"[Node4-2-1] Checkpoint: {trained:,}/{total:,} params ({100*trained/total:.2f}%)"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer with WarmupCosine LR scheduler ----
    def configure_optimizers(self):
        hp = self.hparams
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable_params, lr=hp.lr, weight_decay=hp.weight_decay
        )

        def lr_lambda(epoch: int) -> float:
            """Linear warmup then cosine decay to min_lr_ratio floor."""
            if epoch < hp.warmup_epochs:
                return max(1e-8, epoch / max(1, hp.warmup_epochs))
            progress = (epoch - hp.warmup_epochs) / max(1, hp.max_epochs - hp.warmup_epochs)
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return hp.min_lr_ratio + (1.0 - hp.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
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
        description="Node4-2-1 – scFoundation + STRING_GNN (frozen + neighborhood attention) + GatedFusion"
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=8)
    parser.add_argument("--global-batch-size",   type=int,   default=64)
    parser.add_argument("--max-epochs",          type=int,   default=300)
    parser.add_argument("--lr",                  type=float, default=2e-4)
    parser.add_argument("--weight-decay",        type=float, default=3e-2)
    parser.add_argument("--scf-finetune-layers", type=int,   default=6,
                        dest="scf_finetune_layers")
    parser.add_argument("--head-dropout",        type=float, default=0.5)
    parser.add_argument("--fusion-dropout",      type=float, default=0.3)
    parser.add_argument("--warmup-epochs",       type=int,   default=5)
    parser.add_argument("--min-lr-ratio",        type=float, default=0.08)
    parser.add_argument("--mixup-alpha",         type=float, default=0.4)
    parser.add_argument("--label-smoothing",     type=float, default=0.1)
    parser.add_argument("--num-workers",         type=int,   default=4)
    parser.add_argument("--debug-max-step",      type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",        action="store_true", dest="fast_dev_run")
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
        scf_finetune_layers = args.scf_finetune_layers,
        head_dropout        = args.head_dropout,
        fusion_dropout      = args.fusion_dropout,
        lr                  = args.lr,
        weight_decay        = args.weight_decay,
        warmup_epochs       = args.warmup_epochs,
        max_epochs          = args.max_epochs,
        min_lr_ratio        = args.min_lr_ratio,
        mixup_alpha         = args.mixup_alpha,
        label_smoothing     = args.label_smoothing,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
    )
    es_cb  = EarlyStopping(monitor="val/f1", mode="max", patience=35, min_delta=1e-4)
    lr_cb  = LearningRateMonitor(logging_interval="epoch")
    pg_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    strategy = (
        DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120))
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
        gradient_clip_val       = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node4-2-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
