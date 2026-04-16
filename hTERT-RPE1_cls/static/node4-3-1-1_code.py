"""
Node 4-3-1-1: STRING_GNN Partial Fine-Tuning + Reverted Light Regularization + Longer Training

Architecture (unchanged from parent node4-3-1):
  - STRING_GNN backbone with partial fine-tuning:
      * Frozen layers (mps.0-6 + emb): pre-computed as a buffer
        (saves 7 GCN forward passes per step; valid since these layers don't change)
      * Trainable layers: mps.7 + post_mp (~67K params, backbone_lr=1e-5)
  - Deep 6-layer Residual Bilinear MLP head (rank=512, hidden=512, dropout=0.30):
      LayerNorm(256) -> Linear(256->512) -> 6x ResidualBlock(512, expand=4, drop=0.30)
      -> Linear(512->3x512) -> [B, 3, 512] @ out_gene_emb.T -> [B, 3, 6640]
  - Loss: Focal cross-entropy (gamma=2.0, class_weights=[2.0, 0.5, 4.0])
  - Optimizer: MuonWithAuxAdam
      - Muon (lr=0.006, wd=1e-3) for ResBlock + bilinear_proj 2D weight matrices
      - AdamW (lr=1e-5, wd=1e-3) for backbone mps.7+post_mp
      - AdamW (lr=5e-4, wd=1e-3) for out_gene_emb (light L2, same as node4-2)
      - AdamW (lr=5e-4, wd=1e-4) for other params (norms, biases, fallback_emb, input_proj)
  - Scheduler: Linear warmup (100 steps) + Cosine warm restarts (T_0=600 steps, T_mult=1)
  - No gradient clipping (matching node4-2's configuration which achieved highest F1=0.5069)

Key Improvements Over Parent (node4-3-1, F1=0.5016):
  1. dropout=0.30 (reverted from 0.35 back to node4-2's proven value). Feedback confirmed
     that the "balanced middle" regularization (0.35) failed to surpass node4-2 (0.30).
     The optimal is clearly at or near node4-2's lighter regime.
  2. out_gene_emb weight_decay=1e-3 (reverted from 5e-3 back to node4-2's proven value).
     Heavier L2 on the 3.4M-param output matrix consistently hurt performance in the node4
     lineage. The evidence (node4-3: 0.5036, node4-3-1: 0.5016) confirms lighter L2 is better.
  3. No gradient clipping (reverted from 1.0). Node4-2 had no gradient clipping and achieved
     the highest score in the node4 lineage (0.5069). Clipping may cap gradient signal
     needed for effective warm restart exploration.
  4. patience=80 (increased from 50, matching node4-2's patience). With 600-step cycles
     (~13-14 epochs/cycle), patience=80 allows early stopping only after ~6 cycles fail to
     improve, giving the ascending staircase mechanism more room to operate.
  5. max_epochs=350 (increased from 250). Provides ~25+ potential restart cycles at T_0=600
     steps, giving the staircase mechanism ample opportunity to accumulate improvements
     similar to node1-2-2-2-1 (which achieved F1=0.5099 via 6 ascending cycles).
  6. muon_lr=0.006 (slightly increased from 0.005). Minor increase per feedback recommendation
     to enable stronger per-cycle gradient exploration with Muon's orthogonalized updates.

Memory Sources:
  - node4-3-1 feedback: "Revert dropout to 0.30 + out_gene_emb wd to 1e-3 (PRIMARY)"
  - node4-3-1 feedback: "Remove gradient_clip_val (node4-2 had none, achieved 0.5069)"
  - node4-3-1 feedback: "Increase patience to 80 + extend max_epochs to 300-400"
  - node4-3-1 feedback: "Explore muon_lr 0.006-0.007 for stronger per-cycle exploration"
  - node4-2 (F1=0.5069): dropout=0.3, wd=1e-3, no clipping, patience=80, T_0=600
  - node1-2-2-2-1 (F1=0.5099): ascending staircase across 6 cycles with T_0=600
  - collected_memory: no label smoothing; class_weights=[2.0, 0.5, 4.0] optimal
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import math
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional

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
from muon import MuonWithAuxAdam
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# --- Constants ---

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_DIM        = 256
N_NODES        = 18870

# Focal loss class weights: down(-1)=2.0, neutral(0)=0.5, up(+1)=4.0
# Proven optimal in tree-best lineage (node2-1-3, node1-2-2-2-1, node4-2)
CLASS_WEIGHTS = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)

FOCAL_GAMMA = 2.0  # Focal loss gamma: focuses on hard minority-class examples


# --- Focal Loss ---

def focal_cross_entropy(
    logits: torch.Tensor,      # [B, C, G] -- raw logits
    targets: torch.Tensor,     # [B, G] -- class indices {0, 1, 2}
    class_weights: torch.Tensor,  # [C]
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal cross-entropy with per-class weighting for [B, C, G] logit layout.

    Implementation:
      1. Compute log-softmax probabilities: [B, C, G]
      2. Reshape to [B*G, C] for standard cross-entropy; targets to [B*G]
      3. Get per-sample CE and pt; apply focal modulation: (1-pt)^gamma * CE
      4. Apply class weights via class_weights[targets]; reduce by mean
    """
    B, C, G = logits.shape
    # log_softmax along class dim
    log_probs = F.log_softmax(logits.float(), dim=1)           # [B, C, G]
    # Reshape: [B*G, C] and [B*G]
    log_probs_flat = log_probs.permute(0, 2, 1).reshape(B * G, C)  # [B*G, C]
    targets_flat   = targets.reshape(B * G)                        # [B*G]

    # Standard NLL loss per sample (no reduction)
    ce = F.nll_loss(log_probs_flat, targets_flat, reduction="none")  # [B*G]

    # Compute pt = exp(-ce) = probability assigned to the correct class
    pt = torch.exp(-ce)  # [B*G]

    # Focal modulation: down-weight easy examples
    focal_weight = (1.0 - pt) ** gamma  # [B*G]

    # Class weights
    cw = class_weights.to(logits.device)[targets_flat]  # [B*G]

    loss = (focal_weight * cw * ce).mean()
    return loss


# --- Metric ---

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """
    Mirror of calc_metric.py:
      pred_np:   [N, 3, G] float (softmax probabilities)
      labels_np: [N, G]    int   (class indices 0/1/2)
    Returns mean per-gene macro-F1 over present classes.
    """
    pred_cls = pred_np.argmax(axis=1)  # [N, G]
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1   = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# --- Dataset ---

class StringGNNDataset(Dataset):
    def __init__(
        self,
        pert_ids:     List[str],
        symbols:      List[str],
        node_indices: torch.Tensor,          # [N] long, -1 for unknown
        labels:       Optional[torch.Tensor] = None,  # [N, 6640] long
    ):
        self.pert_ids     = pert_ids
        self.symbols      = symbols
        self.node_indices = node_indices
        self.labels       = labels

    def __len__(self): return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
            "node_index": self.node_indices[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    out = {
        "pert_id":    [b["pert_id"]    for b in batch],
        "symbol":     [b["symbol"]     for b in batch],
        "node_index": torch.stack([b["node_index"] for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# --- DataModule ---

class StringGNNDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data", micro_batch_size=8, num_workers=4):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        gnn_dir    = Path(STRING_GNN_DIR)
        node_names = json.loads((gnn_dir / "node_names.json").read_text())
        node_name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(node_names)}

        def load_split(fname: str, has_lbl: bool):
            df   = pd.read_csv(self.data_dir / fname, sep="\t")
            idxs = torch.tensor(
                [node_name_to_idx.get(pid, -1) for pid in df["pert_id"].tolist()],
                dtype=torch.long,
            )
            labels = None
            if has_lbl and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return StringGNNDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(), idxs, labels
            )

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

        # Load graph topology for backbone
        graph            = torch.load(gnn_dir / "graph_data.pt", weights_only=False)
        self.edge_index  = graph["edge_index"]
        self.edge_weight = graph.get("edge_weight", None)

        n_unknown = sum(
            1 for ds in (self.train_ds, self.val_ds, self.test_ds)
            for ni in ds.node_indices.tolist() if ni == -1
        )
        total = len(self.train_ds) + len(self.val_ds) + len(self.test_ds)
        print(f"[Node4-3-1-1] {n_unknown}/{total} samples not found in STRING_GNN "
              f"-> learned fallback embedding.")

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=self.num_workers,
            pin_memory=True, drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# --- Residual Block ---

class ResidualBlock(nn.Module):
    """
    Pre-norm residual block:
        LN(x) -> FC(d, d*expand) -> GELU -> Dropout -> FC(d*expand, d) + x
    """
    def __init__(self, hidden: int, expand: int = 4, dropout: float = 0.30):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1  = nn.Linear(hidden, hidden * expand, bias=True)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden * expand, hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        return x + h


# --- Model ---

class StringGNNResidualBilinear(nn.Module):
    """
    STRING_GNN partial fine-tuning (mps.7+post_mp) + deep residual bilinear head.

    Design:
      - STRING_GNN layers mps.0-6 are frozen; their output is cached as a buffer
        at first forward call to avoid recomputing frozen layers every step.
      - mps.7 and post_mp are trainable at a low LR to adapt the final representation.
      - A 6-layer residual MLP (rank-512 bilinear) maps the 256-dim PPI embedding
        to logits [B, 3, 6640] via a learned gene output embedding.
      - out_gene_emb uses light weight decay (1e-3, matching node4-2's proven setting)
        to preserve gene embedding expressiveness while preventing overfitting.
    """

    def __init__(
        self,
        edge_index:  torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        n_genes_out: int   = N_GENES_OUT,
        n_classes:   int   = N_CLASSES,
        gnn_dim:     int   = GNN_DIM,
        head_hidden: int   = 512,
        n_res_blocks: int  = 6,
        head_expand:  int  = 4,
        head_dropout: float = 0.30,
        bilinear_rank: int = 512,
    ):
        super().__init__()

        # Load STRING_GNN
        self.gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        # Register graph tensors as buffers
        self.register_buffer("edge_index", edge_index)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        # Freeze mps.0-6 and embedding table; keep mps.7 + post_mp trainable
        for name, param in self.gnn.named_parameters():
            if name.startswith("mps.") and not name.startswith("mps.7"):
                param.requires_grad = False
            elif name.startswith("emb."):
                param.requires_grad = False
            # mps.7.* and post_mp.* remain trainable

        # Buffer for pre-computed frozen intermediate embeddings [N_NODES, 256]
        # Populated on first forward pass; never invalidated (frozen layers don't change)
        self._frozen_emb_cache: Optional[torch.Tensor] = None

        # Fallback embedding for genes absent from STRING_GNN
        self.fallback_emb = nn.Parameter(torch.randn(gnn_dim) * 0.02)

        # -- Head --
        # Input projection: [B, 256] -> [B, head_hidden]
        self.input_proj = nn.Sequential(
            nn.LayerNorm(gnn_dim),
            nn.Linear(gnn_dim, head_hidden, bias=True),
        )

        # 6-layer residual MLP
        self.res_blocks = nn.ModuleList([
            ResidualBlock(head_hidden, expand=head_expand, dropout=head_dropout)
            for _ in range(n_res_blocks)
        ])

        # Bilinear projection: [B, head_hidden] -> [B, n_classes * bilinear_rank]
        self.bilinear_proj = nn.Linear(head_hidden, n_classes * bilinear_rank, bias=False)

        # Learnable output-gene embeddings [n_genes_out, bilinear_rank]
        # Light weight_decay (1e-3, matching node4-2) to prevent overfitting
        # while preserving gene embedding expressiveness
        self.out_gene_emb = nn.Embedding(n_genes_out, bilinear_rank)
        nn.init.xavier_uniform_(self.out_gene_emb.weight)

        self._bilinear_rank = bilinear_rank
        self._n_classes     = n_classes

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Node4-3-1-1] Trainable params: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.2f}%)")

    def _get_frozen_emb(self) -> torch.Tensor:
        """
        Return the intermediate embedding after mps.0-6 (frozen).
        Result is cached after the first call since frozen layers don't change.
        """
        if self._frozen_emb_cache is None:
            # Run frozen portion of the GNN: emb + mps.0-6
            with torch.no_grad():
                x = self.gnn.emb.weight  # [N, 256]
                for i in range(7):       # mps.0 through mps.6
                    x = self.gnn.mps[i](x, self.edge_index, self.edge_weight)
            self._frozen_emb_cache = x.detach()
        return self._frozen_emb_cache

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_indices: [B] long -- STRING_GNN node indices (-1 for unknown)
        Returns:
            logits: [B, 3, 6640]
        """
        # Step 1: Get frozen intermediate embeddings (cached after first call)
        frozen_x = self._get_frozen_emb()  # [N_NODES, 256]

        # Step 2: Apply trainable mps.7 and post_mp on top of frozen embeddings
        x = self.gnn.mps[7](frozen_x, self.edge_index, self.edge_weight)  # [N_NODES, 256]
        node_emb = self.gnn.post_mp(x)  # [N_NODES, 256]

        # Step 3: Index perturbed gene embeddings; handle unknowns with fallback
        known_mask = (node_indices >= 0)           # [B] bool
        safe_idx   = node_indices.clamp(min=0)     # [B]

        pert_emb = node_emb[safe_idx, :]           # [B, 256]
        if not known_mask.all():
            fallback = self.fallback_emb.unsqueeze(0).expand_as(pert_emb)
            pert_emb = torch.where(
                known_mask.unsqueeze(-1).expand_as(pert_emb),
                pert_emb, fallback,
            )

        # Step 4: Head -- input projection
        h = self.input_proj(pert_emb.float())  # [B, 512]

        # Step 5: Residual MLP blocks
        for block in self.res_blocks:
            h = block(h)  # [B, 512]

        # Step 6: Bilinear projection -> [B, 3, bilinear_rank]
        proj = self.bilinear_proj(h).view(-1, self._n_classes, self._bilinear_rank)  # [B, 3, 512]

        # Step 7: Gene interaction -> logits [B, 3, 6640]
        out_embs = self.out_gene_emb.weight  # [6640, 512]
        logits   = torch.matmul(proj, out_embs.T)  # [B, 3, 6640]
        return logits


# --- Gather helpers ---

def _gather_tensors(local_p, local_l, device, world_size):
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

class StringGNNLitModule(pl.LightningModule):

    def __init__(
        self,
        backbone_lr:    float = 1e-5,
        muon_lr:        float = 0.006,
        head_lr:        float = 5e-4,
        out_emb_wd:     float = 1e-3,
        backbone_wd:    float = 1e-3,
        muon_wd:        float = 1e-3,
        other_wd:       float = 1e-4,
        t0_steps:       int   = 600,
        warmup_steps:   int   = 100,
        focal_gamma:    float = 2.0,
        head_dropout:   float = 0.30,
        bilinear_rank:  int   = 512,
        n_res_blocks:   int   = 6,
        head_hidden:    int   = 512,
        head_expand:    int   = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str]  = []
        self._test_symbols:  List[str]  = []
        self._test_labels:   List[torch.Tensor] = []
        self._edge_index:  Optional[torch.Tensor] = None
        self._edge_weight: Optional[torch.Tensor] = None

    def setup(self, stage=None):
        dm = self.trainer.datamodule if self.trainer is not None else None
        if dm is not None and hasattr(dm, "edge_index"):
            self._edge_index  = dm.edge_index
            self._edge_weight = dm.edge_weight

        self.model = StringGNNResidualBilinear(
            edge_index   = self._edge_index,
            edge_weight  = self._edge_weight,
            head_hidden  = self.hparams.head_hidden,
            n_res_blocks = self.hparams.n_res_blocks,
            head_expand  = self.hparams.head_expand,
            head_dropout = self.hparams.head_dropout,
            bilinear_rank = self.hparams.bilinear_rank,
        )

        # Cast all trainable params to float32 for stable optimization
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(self, node_indices):
        return self.model(node_indices)

    def _loss(self, logits, labels):
        return focal_cross_entropy(
            logits, labels, self.class_weights, gamma=self.hparams.focal_gamma
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
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
        f1 = compute_per_gene_f1(lp.numpy(), ll.numpy())
        # val_f1 is already globally gathered via _gather_tensors; sync_dist=True avoids
        # Lightning's warning about logging on epoch level without synchronization.
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        probs  = torch.softmax(logits, dim=1)
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

        # Deduplicate by pert_id (DDP DistributedSampler may pad the dataset)
        seen_pids: set = set()
        keep: List[int] = []
        for i, pid in enumerate(all_pert):
            if pid not in seen_pids:
                seen_pids.add(pid)
                keep.append(i)
        if len(keep) < len(all_pert):
            self.print(f"[Node4-3-1-1] Deduplicating: {len(all_pert)} -> {len(keep)}")
            all_probs  = all_probs[keep]
            all_labels = all_labels[keep]
            all_pert   = [all_pert[i] for i in keep]
            all_syms   = [all_syms[i] for i in keep]

        if self.trainer.is_global_zero:
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pid, sym, probs in zip(all_pert, all_syms, all_probs.numpy()):
                    fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")
            self.print(f"[Node4-3-1-1] Saved test predictions -> {pred_path}")
            if all_labels.any():
                f1 = compute_per_gene_f1(all_probs.numpy(), all_labels.numpy())
                self.print(f"[Node4-3-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear(); self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # -- Parameter groups --
        # Group A: Backbone (mps.7 + post_mp) -- AdamW, low LR
        backbone_params = []
        for name, p in self.model.gnn.named_parameters():
            if p.requires_grad:
                backbone_params.append(p)

        # Group B (Muon): ResBlock 2D weight matrices (fc1, fc2, bilinear_proj)
        muon_params  = []
        # Group C: out_gene_emb -- AdamW, light wd (1e-3, matching node4-2's proven setting)
        out_emb_params = list(self.model.out_gene_emb.parameters())
        # Group D: other head params (norms, biases, fallback_emb, input_proj)
        other_params = []

        # Collect head parameters precisely
        head_param_ids = {id(p) for p in out_emb_params}
        head_param_ids |= {id(p) for p in backbone_params}

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in head_param_ids:
                continue  # already categorized
            # Classify by tensor rank: 2D weight matrices -> Muon
            if p.ndim >= 2 and "norm" not in name and "fallback" not in name:
                muon_params.append(p)
            else:
                other_params.append(p)

        param_groups = [
            # Group A: Backbone mps.7 + post_mp -- AdamW
            dict(params=backbone_params, use_muon=False,
                 lr=hp.backbone_lr, weight_decay=hp.backbone_wd,
                 betas=(0.9, 0.95), eps=1e-8),
            # Group B: ResBlock 2D weight matrices + bilinear_proj -- Muon
            dict(params=muon_params, use_muon=True,
                 lr=hp.muon_lr, weight_decay=hp.muon_wd, momentum=0.95),
            # Group C: out_gene_emb -- AdamW, light wd (1e-3)
            dict(params=out_emb_params, use_muon=False,
                 lr=hp.head_lr, weight_decay=hp.out_emb_wd,
                 betas=(0.9, 0.95), eps=1e-8),
            # Group D: norms, biases, fallback_emb, input_proj scalars -- AdamW
            dict(params=other_params, use_muon=False,
                 lr=hp.head_lr, weight_decay=hp.other_wd,
                 betas=(0.9, 0.95), eps=1e-8),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # -- Scheduler: linear warmup + cosine warm restarts --
        # T_0=600 steps (~14 epochs per cycle at ~44 optimizer steps/epoch on 2 GPUs)
        # T_0=600 produces an ascending staircase in node4-2 (0.489->0.501->0.507)
        # With 350 max_epochs, this allows ~25 potential restart cycles.
        # With patience=80 (~5-6 cycles), early stopping fires only after sufficient exploration.
        warmup = hp.warmup_steps
        T_0    = hp.t0_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            # Cosine warm restart: step within current cycle
            cycle_step = (current_step - warmup) % T_0
            return 0.5 * (1.0 + math.cos(math.pi * cycle_step / T_0))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

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
        self.print(f"Saving ckpt: {trained}/{total} params ({100*trained/total:.2f}%)")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# --- Main ---

def parse_args():
    p = argparse.ArgumentParser(description="Node 4-3-1-1: STRING_GNN Partial FT + Light Regularization + Long Training")
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--backbone-lr",       type=float, default=1e-5)
    p.add_argument("--muon-lr",           type=float, default=0.006)
    p.add_argument("--head-lr",           type=float, default=5e-4)
    p.add_argument("--out-emb-wd",        type=float, default=1e-3)
    p.add_argument("--backbone-wd",       type=float, default=1e-3)
    p.add_argument("--muon-wd",           type=float, default=1e-3)
    p.add_argument("--other-wd",          type=float, default=1e-4)
    p.add_argument("--t0-steps",          type=int,   default=600)
    p.add_argument("--warmup-steps",      type=int,   default=100)
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--head-dropout",      type=float, default=0.30)
    p.add_argument("--bilinear-rank",     type=int,   default=512)
    p.add_argument("--n-res-blocks",      type=int,   default=6)
    p.add_argument("--head-hidden",       type=int,   default=512)
    p.add_argument("--head-expand",       type=int,   default=4)
    p.add_argument("--micro-batch-size",  type=int,   default=8)
    p.add_argument("--global-batch-size", type=int,   default=32)
    p.add_argument("--max-epochs",        type=int,   default=350)
    p.add_argument("--patience",          type=int,   default=80)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",    type=int,   default=None)
    p.add_argument("--fast-dev-run",      action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm  = StringGNNDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = StringGNNLitModule(
        backbone_lr   = args.backbone_lr,
        muon_lr       = args.muon_lr,
        head_lr       = args.head_lr,
        out_emb_wd    = args.out_emb_wd,
        backbone_wd   = args.backbone_wd,
        muon_wd       = args.muon_wd,
        other_wd      = args.other_wd,
        t0_steps      = args.t0_steps,
        warmup_steps  = args.warmup_steps,
        focal_gamma   = args.focal_gamma,
        head_dropout  = args.head_dropout,
        bilinear_rank = args.bilinear_rank,
        n_res_blocks  = args.n_res_blocks,
        head_hidden   = args.head_hidden,
        head_expand   = args.head_expand,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath  = str(out_dir / "checkpoints"),
        filename = "best-{epoch:04d}-{val_f1:.4f}",
        monitor  = "val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max",
                           patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)
    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps    = -1
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False
    if args.debug_max_step is not None:
        max_steps   = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2; limit_test = 2
    if args.fast_dev_run:
        fast_dev_run = True

    accum    = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator     = "gpu",
        devices         = n_gpus,
        num_nodes       = 1,
        strategy        = strategy,
        precision       = "bf16-mixed",
        max_epochs      = args.max_epochs,
        max_steps       = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches = limit_train,
        limit_val_batches   = limit_val,
        limit_test_batches  = limit_test,
        val_check_interval  = (
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps = 2,
        callbacks   = [ckpt_cb, es_cb, lr_cb, pb_cb],
        logger      = [csv_logger, tb_logger],
        log_every_n_steps = 10,
        gradient_clip_val = None,   # No gradient clipping (matching node4-2 which achieved 0.5069)
        deterministic = False,   # nll_loss2d has no deterministic CUDA impl
        default_root_dir = str(out_dir),
        fast_dev_run = fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 4-3-1-1 -- STRING_GNN Partial FT + Light Regularization + Extended Training\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
