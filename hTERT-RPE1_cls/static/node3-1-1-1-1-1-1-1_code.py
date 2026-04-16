"""
Node 3-1-1-1-1-1-1-1 — Reverted Proven Architecture + Muon Optimizer

Key design philosophy:
  - REVERT to node1-2's proven configuration (rank=256, 6 layers, dropout=0.2) which is
    the current MCTS best (F1=0.4912) — larger capacity consistently degrades performance
  - ADD the one confirmed improvement from the parent: gradient clipping (clip_val=1.0)
    which prevents bf16 numerical instability spikes
  - ADD Muon optimizer for MLP hidden weight matrices to improve convergence speed and
    sample efficiency — this is the primary novel improvement direction
  - REMOVE label smoothing (suspected to hurt F1 optimization based on node2-1-1-1-1-1)
  - EXTENDED training: 300 epochs with T_max aligned to actual steps for full cosine decay
  - TWO-GROUP weight decay: main head wd=1e-3 (matching node1-2), out_gene_emb wd=3e-3
    (stronger for high-dimensional output table — from parent's insight)

Architecture (matching node1-2 exactly):
  - Frozen STRING_GNN (static embeddings precomputed once, 256-dim)
  - OOV fallback: learnable oov_embedding [256]
  - Input projection: LayerNorm + Linear(256 -> 512)
  - 6 x ResidualMLPBlock(512, expand=4, dropout=0.2)   [exact node1-2 config]
  - Output projection: Linear(512 -> 3*256=768)
  - Two-sided bilinear: [B, 3, 256] x [6640, 256] -> [B, 3, 6640]

Optimizer (NOVEL — Muon for hidden weights):
  - Muon group: hidden weight matrices in MLP (ndim>=2, not embeddings/heads) — lr=0.02
  - AdamW group 1: biases, LayerNorm, 1D params — lr=5e-4, wd=1e-3
  - AdamW group 2: out_gene_emb [6640, 256] — lr=5e-4, wd=3e-3 (stronger regularization)
  - AdamW group 3: oov_embedding, input_proj.weight — lr=5e-4, wd=1e-3
  All AdamW: betas=(0.9, 0.999), eps=1e-8
  Muon: momentum=0.95, ns_steps=5

Evidence base:
  - node1-2 (F1=0.4912): proven architecture, val/train ratio only 3.71x (healthy)
  - Parent node3-1-1-1-1-1-1 (F1=0.4788): capacity expansion (rank 512, 8 layers)
    worsened overfitting despite stronger regularization — confirms keep rank=256, 6 layers
  - node2-1-1-1-1-1 (F1=0.4780): label smoothing 0.05 suspected to reduce gradient signal
    for minority classes — removing label_smoothing may recover 0.005-0.010 F1
  - Parent feedback: gradient clipping confirmed to eliminate bf16 numerical instability
  - Muon skill: 1.35x training speedup, better sample efficiency — critical with only 1,416 samples
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Required for deterministic=True with CUDA >= 10.2

import json
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

# --- Constants ---

STRING_GNN_DIR = "/home/Models/STRING_GNN"

N_GENES_OUT = 6640    # Output gene positions to predict DEG labels for
N_CLASSES   = 3       # down (-1 -> 0), unchanged (0 -> 1), up (+1 -> 2)
GNN_DIM     = 256     # STRING_GNN node embedding dimension

# node1-2's proven bilinear rank — reverting from parent's failed 512 experiment
BILINEAR_RANK = 256

# Class weights: inverse-frequency based on train split label distribution
# down-regulated (-1 -> class 0): 8.14%, neutral (0 -> class 1): 88.86%, up (+1 -> class 2): 3.00%
CLASS_WEIGHTS = torch.tensor([10.91, 1.0, 29.62], dtype=torch.float32)


# --- Metric ---

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """
    Compute macro-averaged per-gene F1 score matching calc_metric.py logic.

    Args:
        pred_np: [N, 3, G] softmax probabilities (float)
        labels_np: [N, G] class indices in {0, 1, 2} (already shifted from {-1, 0, 1})
    Returns:
        float: mean per-gene macro-F1 over all G genes
    """
    pred_cls = pred_np.argmax(axis=1)  # [N, G]
    f1_vals = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# --- Focal Loss (no label smoothing — suspected to hurt minority class F1) ---

class FocalLoss(nn.Module):
    """
    Focal cross-entropy loss for multi-class classification.

    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    Using gamma=2.0 (same as node1-2 which achieved F1=0.4912).
    NO label smoothing — node2-1-1-1-1-1 (label_smoothing=0.05) fell 0.013 below node1-2.
    Removing smoothing preserves full gradient signal for minority class optimization.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, G] unnormalized logits (C=3 classes, G=6640 genes)
            targets: [B, G] class indices in {0, 1, 2}
        Returns:
            scalar loss
        """
        B, C, G = logits.shape
        logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        targets_flat = targets.reshape(-1)                      # [B*G]

        log_probs = F.log_softmax(logits_flat, dim=1)
        probs     = torch.exp(log_probs)

        target_log_prob = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        target_prob     = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - target_prob).pow(self.gamma)

        if self.weight is not None:
            class_w = self.weight.to(logits.device)[targets_flat]
        else:
            class_w = torch.ones_like(focal_weight)

        # Pure focal loss with no label smoothing
        loss_per_pos  = -target_log_prob
        weighted_loss = focal_weight * class_w * loss_per_pos
        denom = class_w.sum().clamp(min=1.0)
        return weighted_loss.sum() / denom


# --- Dataset ---

class PerturbDataset(Dataset):
    """Dataset wrapping per-perturbation static STRING_GNN embeddings + labels."""

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        gnn_embs: torch.Tensor,      # [N, GNN_DIM] float32 precomputed STRING_GNN embeddings
        in_vocab: torch.Tensor,      # [N] bool - True if pert_id found in STRING_GNN vocabulary
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long or None
    ):
        self.pert_ids = pert_ids
        self.symbols  = symbols
        self.gnn_embs = gnn_embs
        self.in_vocab = in_vocab
        self.labels   = labels

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id": self.pert_ids[idx],
            "symbol":  self.symbols[idx],
            "gnn_emb": self.gnn_embs[idx],
            "in_vocab": self.in_vocab[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def perturb_collate_fn(batch):
    out = {
        "pert_id": [b["pert_id"] for b in batch],
        "symbol":  [b["symbol"]  for b in batch],
        "gnn_emb": torch.stack([b["gnn_emb"]  for b in batch]),
        "in_vocab": torch.stack([b["in_vocab"] for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# --- DataModule ---

class PerturbDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "data", micro_batch_size: int = 8, num_workers: int = 4):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        # Load pre-computed GNN embedding cache
        # Cache is pre-computed in main() BEFORE DDP initialization.
        cache_path = Path(__file__).parent / "run" / "gnn_emb_cache.pt"
        gnn_cache: Dict[str, Tuple[torch.Tensor, bool]] = torch.load(
            cache_path, weights_only=False
        )

        def load_split(fname: str, has_lbl: bool) -> PerturbDataset:
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            pert_ids = df["pert_id"].tolist()
            symbols  = df["symbol"].tolist()

            gnn_embs_list: List[torch.Tensor] = []
            in_vocab_list: List[bool] = []
            for pid in pert_ids:
                emb, in_v = gnn_cache.get(pid, (torch.zeros(GNN_DIM), False))
                gnn_embs_list.append(emb)
                in_vocab_list.append(in_v)

            gnn_embs = torch.stack(gnn_embs_list, dim=0)      # [N, 256]
            in_vocab = torch.tensor(in_vocab_list, dtype=torch.bool)

            labels = None
            if has_lbl and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)

            return PerturbDataset(pert_ids, symbols, gnn_embs, in_vocab, labels)

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            collate_fn=perturb_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# --- Residual MLP Block ---

class ResidualMLPBlock(nn.Module):
    """
    One residual MLP block: LayerNorm -> Linear(d, hidden) -> GELU -> Dropout
                           -> Linear(hidden, d) -> Dropout -> residual add.

    Matching node1-2's proven configuration exactly:
    - dropout=0.2 (not 0.3 which caused overfitting regression in parent)
    - hidden = d * 4 = 512 * 4 = 2048

    The Linear weight matrices in this block will be optimized with Muon
    for better convergence, while biases use AdamW.
    """

    def __init__(self, d_in: int, expand: int = 4, dropout: float = 0.2):
        super().__init__()
        d_hidden = d_in * expand
        self.norm = nn.LayerNorm(d_in)
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_in)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.linear2(h)
        h = self.drop2(h)
        return x + h


# --- Main Model ---

class DeepBilinearPerturbModel(nn.Module):
    """
    Frozen STRING_GNN + Deep Bilinear MLP Head — node1-2 proven architecture.

    Reverting to node1-2's exact architecture after parent's rank-512/8-layer experiment
    caused severe overfitting regression (F1=0.4788 vs node1-2's 0.4912).

    Architecture:
    1. Frozen STRING_GNN embeddings precomputed as static cache [N_pids, 256]
    2. OOV fallback: learnable oov_embedding [256] for genes not in STRING vocabulary
    3. Input projection: LayerNorm(256) + Linear(256 -> 512)
    4. Deep residual MLP: 6 layers, hidden=512x4=2048, dropout=0.2  [node1-2 config]
    5. Output projection: Linear(512 -> 3*256=768) -> [B, 3, 256]
    6. Two-sided bilinear: einsum([B, 3, 256] x [6640, 256]) -> [B, 3, 6640]

    Separating linear layers from the Sequential for Muon optimizer compatibility:
    The weight matrices (ndim=2) in residual blocks are eligible for Muon.
    Biases, LayerNorm params, embeddings, and output heads use AdamW.
    """

    def __init__(
        self,
        n_genes_out: int = N_GENES_OUT,
        n_classes:   int = N_CLASSES,
        head_dim:    int = 512,
        bilinear_rank: int = BILINEAR_RANK,
        n_layers:    int = 6,
        dropout:     float = 0.2,
    ):
        super().__init__()

        self.n_genes_out   = n_genes_out
        self.n_classes     = n_classes
        self.bilinear_rank = bilinear_rank

        # --- OOV fallback embedding for genes not in STRING_GNN vocabulary ---
        self.oov_embedding = nn.Parameter(torch.zeros(GNN_DIM, dtype=torch.float32))
        nn.init.normal_(self.oov_embedding, std=0.02)

        # --- Input projection to head_dim ---
        # Separated into norm and linear for Muon parameter grouping
        self.input_norm = nn.LayerNorm(GNN_DIM)
        self.input_linear = nn.Linear(GNN_DIM, head_dim)

        # --- Deep residual MLP blocks (6 layers, matching node1-2) ---
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(head_dim, expand=4, dropout=dropout)
            for _ in range(n_layers)
        ])

        # --- Output projection: project to 3 * bilinear_rank ---
        # This is the classification head — use AdamW (not Muon)
        self.out_proj = nn.Linear(head_dim, n_classes * bilinear_rank)

        # --- Learnable output gene embeddings [N_GENES_OUT, bilinear_rank] ---
        # Matching node1-2: random init, rank=256 (not 512 which caused overfitting)
        # Use AdamW with stronger weight decay (wd=3e-3, vs wd=1e-3 for other params)
        self.out_gene_emb = nn.Parameter(
            torch.randn(n_genes_out, bilinear_rank) * 0.01
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Node-bilinear] Trainable params: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.2f}%)")

    def forward(
        self,
        gnn_emb:  torch.Tensor,   # [B, 256] static STRING_GNN embeddings
        in_vocab: torch.Tensor,   # [B] bool - True if in STRING_GNN vocabulary
    ) -> torch.Tensor:
        B = gnn_emb.shape[0]

        # --- OOV handling ---
        oov_fill  = self.oov_embedding.unsqueeze(0).expand(B, -1)  # [B, 256]
        in_v      = in_vocab.unsqueeze(1).expand_as(gnn_emb)       # [B, 256]
        pert_emb  = torch.where(in_v, gnn_emb, oov_fill)           # [B, 256]

        # --- Input projection ---
        h = self.input_norm(pert_emb)
        h = self.input_linear(h)                               # [B, 512]

        # --- 6 Residual MLP blocks (node1-2 depth) ---
        for block in self.blocks:
            h = block(h)

        # --- Output projection and bilinear interaction ---
        h = self.out_proj(h)                                   # [B, 3*256]
        h = h.view(B, self.n_classes, self.bilinear_rank)     # [B, 3, 256]

        # Two-sided bilinear: [B, 3, R] x [G, R]^T -> [B, 3, G]
        logits = torch.einsum("bcr,gr->bcg", h, self.out_gene_emb)  # [B, 3, 6640]

        return logits


# --- DDP Tensor Gathering ---

def _gather_tensors(
    local_p: torch.Tensor,
    local_l: torch.Tensor,
    device: torch.device,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather tensors from all DDP ranks, handling variable-size padding."""
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))
    pad    = max_sz - local_p.shape[0]
    p = local_p.to(device); l = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)
    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p); dist.all_gather(gl, l)
    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# --- LightningModule ---

class DeepBilinearPerturbLitModule(pl.LightningModule):

    def __init__(
        self,
        lr:             float = 5e-4,
        muon_lr:        float = 0.02,         # Muon optimizer LR for hidden weight matrices
        muon_wd:        float = 0.01,         # Muon weight decay
        weight_decay:   float = 1e-3,         # AdamW WD for non-Muon params (matching node1-2)
        gene_emb_wd:    float = 3e-3,         # Stronger WD for out_gene_emb [6640, 256]
        focal_gamma:    float = 2.0,
        max_steps:      int   = 13200,        # T_max: 44 steps/epoch * 300 epochs
        warmup_steps:   int   = 50,           # Same as node1-2
        head_dim:       int   = 512,
        bilinear_rank:  int   = BILINEAR_RANK,
        n_layers:       int   = 6,
        dropout:        float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []
        self._test_labels:   List[torch.Tensor] = []

    def setup(self, stage=None):
        self.model = DeepBilinearPerturbModel(
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            head_dim=self.hparams.head_dim,
            bilinear_rank=self.hparams.bilinear_rank,
            n_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
        )
        # Cast trainable parameters to float32 for stable optimization
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()
        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=CLASS_WEIGHTS,
        )

    def forward(self, gnn_emb: torch.Tensor, in_vocab: torch.Tensor) -> torch.Tensor:
        return self.model(gnn_emb, in_vocab)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.focal_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        logits = self(batch["gnn_emb"], batch["in_vocab"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["gnn_emb"], batch["in_vocab"])
        if "label" in batch:
            loss = self._loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        lp = torch.cat(self._val_preds,  0)
        ll = torch.cat(self._val_labels, 0)
        if self.trainer.world_size > 1:
            lp, ll = _gather_tensors(lp, ll, self.device, self.trainer.world_size)
        probs_np  = torch.softmax(lp, dim=1).numpy()
        labels_np = ll.numpy()
        f1 = compute_per_gene_f1(probs_np, labels_np)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["gnn_emb"], batch["in_vocab"])
        probs  = torch.softmax(logits, dim=1)  # [B, 3, G]
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (torch.cat(self._test_labels, 0) if self._test_labels
                        else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long))

        if self.trainer.world_size > 1:
            all_probs, all_labels = _gather_tensors(
                local_probs, dummy_labels, self.device, self.trainer.world_size
            )
            all_pert = [None] * self.trainer.world_size
            all_syms = [None] * self.trainer.world_size
            dist.all_gather_object(all_pert, self._test_pert_ids)
            dist.all_gather_object(all_syms, self._test_symbols)
            all_pert = [p for sub in all_pert for p in sub]
            all_syms = [s for sub in all_syms for s in sub]
        else:
            all_probs, all_labels = local_probs, dummy_labels
            all_pert, all_syms   = self._test_pert_ids, self._test_symbols

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id (DDP padding may create duplicates)
            seen_pids: set = set()
            dedup_perts, dedup_syms, dedup_probs_list, dedup_label_rows = [], [], [], []
            for pid, sym, prob_row, lbl_row in zip(
                all_pert, all_syms, all_probs.numpy(), all_labels.numpy()
            ):
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    dedup_perts.append(pid)
                    dedup_syms.append(sym)
                    dedup_probs_list.append(prob_row)
                    dedup_label_rows.append(lbl_row)

            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pid, sym, probs in zip(dedup_perts, dedup_syms, dedup_probs_list):
                    fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")

            self.print(f"[DeepBilinear] Saved {len(dedup_perts)} test predictions -> {pred_path}")

            if all_labels.any():
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[DeepBilinear] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams
        model = self.model

        # --- Muon optimizer parameter grouping ---
        # Muon targets: hidden weight matrices (ndim >= 2) in residual MLP blocks
        #   - model.blocks[i].linear1.weight [2048, 512]
        #   - model.blocks[i].linear2.weight [512, 2048]
        #   - model.input_linear.weight [512, 256]
        # These are the primary transformation matrices that benefit from orthogonalization.
        #
        # AdamW targets (NOT Muon-eligible per skill documentation):
        #   - model.out_proj.weight [768, 512] — output head (NOT Muon)
        #   - model.out_gene_emb [6640, 256] — output embedding table (NOT Muon)
        #   - model.oov_embedding [256] — 1D parameter (NOT Muon)
        #   - All biases (1D, NOT Muon)
        #   - All LayerNorm weights/biases (NOT Muon)

        muon_weight_ids = set()
        muon_weights = []

        # input_linear weight matrix — hidden transformation (Muon-eligible)
        muon_weight_ids.add(id(model.input_linear.weight))
        muon_weights.append(model.input_linear.weight)

        # Residual block weight matrices (Muon-eligible hidden matrices)
        for block in model.blocks:
            muon_weight_ids.add(id(block.linear1.weight))
            muon_weights.append(block.linear1.weight)
            muon_weight_ids.add(id(block.linear2.weight))
            muon_weights.append(block.linear2.weight)

        # out_gene_emb parameter id (separate AdamW group with stronger WD)
        gene_emb_id = id(model.out_gene_emb)

        # Collect all remaining params for AdamW
        adamw_main_params = []
        adamw_gene_emb_params = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in muon_weight_ids:
                continue  # handled by Muon
            if id(p) == gene_emb_id:
                adamw_gene_emb_params.append(p)
            else:
                adamw_main_params.append(p)

        param_groups = [
            # Muon group: hidden weight matrices
            {
                "params": muon_weights,
                "use_muon": True,
                "lr": hp.muon_lr,
                "weight_decay": hp.muon_wd,
                "momentum": 0.95,
            },
            # AdamW group 1: biases, norms, embeddings (oov), out_proj weight
            {
                "params": adamw_main_params,
                "use_muon": False,
                "lr": hp.lr,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": hp.weight_decay,
            },
            # AdamW group 2: out_gene_emb [6640, 256] — stronger WD
            # NOTE: MuonWithAuxAdam strictly enforces keys; "name" is not allowed.
            {
                "params": adamw_gene_emb_params,
                "use_muon": False,
                "lr": hp.lr,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": hp.gene_emb_wd,
            },
        ]

        # Use MuonWithAuxAdam in distributed context, SingleDeviceMuonWithAuxAdam otherwise.
        # MuonWithAuxAdam requires torch.distributed to be initialized (for gradient sync).
        # SingleDeviceMuonWithAuxAdam works on a single device without distributed context.
        if dist.is_available() and dist.is_initialized():
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        # Step-based cosine annealing with linear warmup.
        # T_max = 44 steps/epoch * 300 epochs = 13200 steps for full cosine decay.
        # This extends node1-2's training horizon (was 200 epochs, 8800 steps).
        # node1-2 showed val F1 improving as LR decayed late in training.
        # Longer T_max gives more time in the LR-decay regime for better convergence.
        def lr_lambda(step: int) -> float:
            if step < hp.warmup_steps:
                return float(step + 1) / float(hp.warmup_steps)
            progress = float(step - hp.warmup_steps) / float(
                max(1, hp.max_steps - hp.warmup_steps)
            )
            # Clamp progress to [0, 1] to prevent LR restart if training runs longer
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())
            eta_min_ratio = 1e-6 / hp.lr
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # --- Checkpoint: save only trainable parameters + buffers ---

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items()
              if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving ckpt: {trained:,}/{total:,} params ({100*trained/total:.2f}%)"
        )
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# --- STRING_GNN Embedding Cache ---

def build_static_gnn_embeddings(
    pert_ids: List[str],
    device: torch.device,
) -> Dict[str, Tuple[torch.Tensor, bool]]:
    """
    Precompute static STRING_GNN embeddings (frozen, no conditioning).

    One forward pass for all nodes to produce [18870, 256] embedding matrix,
    then look up the perturbed gene's row for each pert_id.

    Returns: {pert_id: (emb_256d_float32_cpu, in_vocab_bool)}
    """
    model_dir   = Path(STRING_GNN_DIR)
    node_names  = json.loads((model_dir / "node_names.json").read_text())
    name_to_idx = {n: i for i, n in enumerate(node_names)}

    gnn_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    gnn_model = gnn_model.to(device)
    gnn_model.eval()

    graph       = torch.load(model_dir / "graph_data.pt", weights_only=False)
    edge_index  = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"]
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    # Single forward pass: get all 18870 node embeddings at once
    with torch.no_grad():
        out = gnn_model(
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
    all_embs = out.last_hidden_state.cpu().float()  # [18870, 256]

    # Clean up GNN model from GPU memory
    del gnn_model
    torch.cuda.empty_cache()

    # Build lookup dict for all unique pert_ids
    unique_pids = list(set(pert_ids))
    result: Dict[str, Tuple[torch.Tensor, bool]] = {}
    for pid in unique_pids:
        pid_clean = pid.split(".")[0]
        if pid_clean in name_to_idx:
            node_idx = name_to_idx[pid_clean]
            result[pid] = (all_embs[node_idx], True)
        else:
            result[pid] = (torch.zeros(GNN_DIM, dtype=torch.float32), False)

    in_vocab_count = sum(1 for (_, iv) in result.values() if iv)
    print(f"[Pre-compute] Static STRING_GNN: {in_vocab_count}/{len(result)} genes in vocabulary "
          f"({100*in_vocab_count/len(result):.1f}%)")
    return result


def _precompute_gnn_cache(args):
    """
    Pre-compute STRING_GNN embeddings BEFORE DDP initialization.

    Uses a single GNN forward pass (no conditioning) for efficiency.
    Rank 0 computes and saves; other ranks poll for the sentinel file.
    """
    import time as _time

    rank       = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cache_path = Path(__file__).parent / "run" / "gnn_emb_cache.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path = cache_path.with_suffix(".pt.ready")

    if not sentinel_path.exists():
        if rank == 0:
            print("[Pre-compute] Building static STRING_GNN embeddings "
                  "(rank 0, one-time, single forward pass)...", flush=True)
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            data_dir = Path(args.data_dir)
            all_pids = list({
                pid
                for fname in ["train.tsv", "val.tsv", "test.tsv"]
                for pid in pd.read_csv(data_dir / fname, sep="\t")["pert_id"].tolist()
            })
            gnn_cache = build_static_gnn_embeddings(all_pids, device)
            tmp_path  = cache_path.with_suffix(".pt.tmp")
            torch.save(gnn_cache, tmp_path)
            tmp_path.rename(cache_path)
            sentinel_path.touch()
            print(f"[Pre-compute] Cached {len(gnn_cache)} embeddings -> {cache_path}", flush=True)
        else:
            print(f"[Rank {rank}] Waiting for STRING_GNN cache from rank 0...", flush=True)
            while not sentinel_path.exists():
                _time.sleep(3)
            print(f"[Rank {rank}] STRING_GNN cache ready.", flush=True)


# --- Argument Parsing ---

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 3-1-1-1-1-1-1-1 — Reverted Proven Architecture + Muon Optimizer"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--lr",                type=float, default=5e-4)
    p.add_argument("--muon-lr",           type=float, default=0.02)
    p.add_argument("--muon-wd",           type=float, default=0.01)
    p.add_argument("--weight-decay",      type=float, default=1e-3)
    p.add_argument("--gene-emb-wd",       type=float, default=3e-3)
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--head-dim",          type=int,   default=512)
    p.add_argument("--bilinear-rank",     type=int,   default=BILINEAR_RANK)
    p.add_argument("--n-layers",          type=int,   default=6)
    p.add_argument("--dropout",           type=float, default=0.2)
    p.add_argument("--micro-batch-size",  type=int,   default=8)
    p.add_argument("--global-batch-size", type=int,   default=32)
    p.add_argument("--max-epochs",        type=int,   default=300)
    p.add_argument("--warmup-steps",      type=int,   default=50)
    p.add_argument("--patience",          type=int,   default=50)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",    type=int,   default=None)
    p.add_argument("--fast-dev-run",      action="store_true", default=False)
    return p.parse_args()


# --- Main ---

def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute STRING_GNN static embeddings before DDP initialization.
    _precompute_gnn_cache(args)

    # --- Compute total training steps for step-based cosine schedule ---
    # T_max = steps_per_epoch * max_epochs
    # With global_batch_size=32 and 1416 training samples: 1416 // 32 = 44 steps/epoch
    # 300 epochs: 44 * 300 = 13200 steps for full cosine cycle
    # Extended from 200 epochs (8800 steps) for better LR-decay convergence
    train_size = 1416
    actual_steps_per_epoch = max(1, train_size // args.global_batch_size)
    total_max_steps = actual_steps_per_epoch * args.max_epochs  # e.g., 44 * 300 = 13200

    # Adjust for debug mode
    if args.debug_max_step is not None:
        total_max_steps = args.debug_max_step

    dm = PerturbDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    lit = DeepBilinearPerturbLitModule(
        lr=args.lr,
        muon_lr=args.muon_lr,
        muon_wd=args.muon_wd,
        weight_decay=args.weight_decay,
        gene_emb_wd=args.gene_emb_wd,
        focal_gamma=args.focal_gamma,
        max_steps=total_max_steps,
        warmup_steps=args.warmup_steps,
        head_dim=args.head_dim,
        bilinear_rank=args.bilinear_rank,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.patience, min_delta=1e-5
    )
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_train_steps: int  = -1
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_train_steps = args.debug_max_step
        limit_train     = args.debug_max_step
        limit_val       = 2
        limit_test      = 2
    if args.fast_dev_run:
        fast_dev_run = True

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_train_steps,
        accumulate_grad_batches=accum,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,         # Prevent bf16 numerical instability (confirmed fix)
        gradient_clip_algorithm="norm",
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)


if __name__ == "__main__":
    main()
