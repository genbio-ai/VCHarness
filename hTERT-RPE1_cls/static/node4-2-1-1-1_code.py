"""
Node 4-2-1-1-1: STRING_GNN Partial Fine-Tuning (mps.6+mps.7+post_mp)
             + Reduced-Capacity Head (4x ResBlock, hidden=256, rank=256)
             + Label Smoothing=0.07
             + Post-Training Quality-Filtered SWA (best + last checkpoints)

Architecture Strategy:
  Based on parent node4-2-1-1 (F1=0.4974, regression from grandparent node4-2-1 F1=0.5076).

  Parent failure analysis:
    - scFoundation pos_emb initialization was semantically wrong (positional != gene identity)
    - Low gene_emb_lr=1e-4 trapped embeddings in suboptimal basin, preventing decisive Cycle 2 spike
    - 17M trainable params on 1,416 samples (12,000:1 ratio) remains the fundamental bottleneck

  This node directly addresses the root cause bottleneck via architectural parameter reduction:
    1. Revert to Xavier init for out_gene_emb + restore gene_emb_lr=5e-4 (as in node4-2-1)
    2. REDUCE HEAD CAPACITY: 6x ResBlock(512)+rank512 (17M params) -> 4x ResBlock(256)+rank256 (~5M params)
       This directly cuts the parameter-to-sample ratio from 12,000:1 to ~3,500:1 (more appropriate)
    3. Add label smoothing=0.07 (from tree best lineage, improves minority class recall)
    4. Expand backbone fine-tuning to mps.6+mps.7+post_mp (~198K params, follows tree best node2 lineage)
    5. Post-training quality-filtered SWA (best + last checkpoint, from tree best lineage)

Memory Sources:
  - node4-2-1-1 feedback: "Abandon scFoundation pos_emb; restore xavier + gene_emb_lr=5e-4"
  - node4-2-1-1 feedback: "Option A (highest priority): reduce head to 4x ResBlock(256) + rank-256"
  - node4-2-1 feedback: "Reduce head capacity to ~5M params to address 12,000:1 param-to-sample ratio"
  - node2-1-1-1-2-1-1-1 (F1=0.5118): mps.6+7+post_mp backbone, dropout=0.45, label_smooth=0.10, SWA
  - node2-1-1-1-2-1-1-1-1 (F1=0.5124): quality-filtered SWA + SGDR proven most effective approach
  - node4-2-1 feedback: T_0=1200 warm restarts enabled decisive Cycle 2 breakthrough
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
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

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR  = "/home/Models/STRING_GNN"
N_GENES_OUT     = 6640
N_CLASSES       = 3
GNN_DIM         = 256
N_NODES         = 18870

# Class weights: down(-1)=2.0, neutral(0)=0.5, up(+1)=4.0
# Proven effective in tree best lineage (node2-1-3, node1-2-2-2-1, node4-2)
CLASS_WEIGHTS = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)

# Focal loss gamma: focuses training on hard examples
FOCAL_GAMMA = 2.0


# ─── Focal Loss with Label Smoothing ──────────────────────────────────────────

def focal_cross_entropy_with_smoothing(
    logits: torch.Tensor,       # [B, C, L]
    labels: torch.Tensor,       # [B, L] long
    class_weights: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.07,
) -> torch.Tensor:
    """
    Focal cross-entropy loss with label smoothing for multi-output 3-class classification.
    Logits: [B, 3, L], labels: [B, L], class_weights: [3]
    """
    B, C, L = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*L, C]
    labels_flat = labels.reshape(-1)                        # [B*L]

    # Label smoothing applied via cross_entropy
    ce_loss = F.cross_entropy(
        logits_flat, labels_flat,
        weight=class_weights.to(logits_flat.device),
        label_smoothing=label_smoothing,
        reduction="none",
    )  # [B*L]

    # Focal weight based on original (non-smoothed) class probabilities
    with torch.no_grad():
        probs = F.softmax(logits_flat.float(), dim=-1)
        pt    = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt).pow(gamma)

    loss = (focal_weight * ce_loss).mean()
    return loss


# ─── Per-Gene F1 ──────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Mirrors calc_metric.py: per-gene macro-F1 over present classes."""
    pred_cls = pred_np.argmax(axis=1)
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]; yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class StringGNNDataset(Dataset):
    def __init__(
        self,
        pert_ids:     List[str],
        symbols:      List[str],
        node_indices: torch.Tensor,
        labels:       Optional[torch.Tensor] = None,
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


# ─── DataModule ───────────────────────────────────────────────────────────────

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
            df  = pd.read_csv(self.data_dir / fname, sep="\t")
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

        graph = torch.load(gnn_dir / "graph_data.pt", weights_only=False)
        self.edge_index  = graph["edge_index"]
        self.edge_weight = graph.get("edge_weight", None)

        n_unknown = sum(
            1 for ds in (self.train_ds, self.val_ds, self.test_ds)
            for ni in ds.node_indices.tolist() if ni == -1
        )
        total = len(self.train_ds) + len(self.val_ds) + len(self.test_ds)
        print(f"[Node4-2-1-1-1] {n_unknown}/{total} samples not in STRING_GNN "
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


# ─── Residual Block ───────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, expand: int = 4, dropout: float = 0.4):
        super().__init__()
        mid = hidden_dim * expand
        self.fc1  = nn.Linear(hidden_dim, mid, bias=False)
        self.fc2  = nn.Linear(mid, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x + residual


# ─── Model ────────────────────────────────────────────────────────────────────

class StringGNNReducedHeadModel(nn.Module):
    """
    STRING_GNN (partial fine-tuning: mps.6 + mps.7 + post_mp) +
    Reduced-Capacity 4-layer Residual MLP + Rank-256 Bilinear Interaction Head.

    Key changes from parent node4-2-1-1:
      1. No scFoundation init — revert to Xavier uniform (proved superior in node4-2-1)
      2. Reduced head capacity: 6xResBlock(512)+rank512 -> 4xResBlock(256)+rank256 (~5M vs 17M params)
      3. Expanded backbone: mps.7+post_mp -> mps.6+mps.7+post_mp (~198K backbone params)
      4. gene_emb_lr restored to 5e-4 (from grandparent node4-2-1)
    """

    def __init__(
        self,
        edge_index:    torch.Tensor,
        edge_weight:   Optional[torch.Tensor],
        gnn_dim:       int = GNN_DIM,
        n_genes_out:   int = N_GENES_OUT,
        n_classes:     int = N_CLASSES,
        hidden_dim:    int = 256,
        n_layers:      int = 4,
        expand:        int = 4,
        bilinear_rank: int = 256,
        dropout:       float = 0.4,
    ):
        super().__init__()
        self.gnn_dim       = gnn_dim
        self.n_classes     = n_classes
        self.n_genes_out   = n_genes_out
        self.bilinear_rank = bilinear_rank

        # ── Backbone ──────────────────────────────────────────────────────────
        full_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        self.register_buffer("edge_index",  edge_index)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight)
        else:
            self.edge_weight = None

        # Freeze: emb + mps.0-5; Trainable: mps.6 + mps.7 + post_mp
        self.emb     = full_gnn.emb                                     # frozen
        self.mps_0_5 = nn.ModuleList([full_gnn.mps[i] for i in range(6)])  # frozen
        self.mps_6   = full_gnn.mps[6]                                  # trainable (NEW)
        self.mps_7   = full_gnn.mps[7]                                  # trainable
        self.post_mp = full_gnn.post_mp                                 # trainable

        for p in self.emb.parameters():
            p.requires_grad_(False)
        for layer in self.mps_0_5:
            for p in layer.parameters():
                p.requires_grad_(False)

        # Fallback embedding for genes absent from STRING_GNN
        self.fallback_emb = nn.Parameter(torch.randn(gnn_dim) * 0.02)

        # ── Head ──────────────────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.LayerNorm(gnn_dim),
            nn.Linear(gnn_dim, hidden_dim, bias=False),
        )

        # REDUCED: 4 layers, hidden=256 (was 6x512)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])

        # REDUCED: rank=256 (was 512)
        self.fc_bilinear = nn.Linear(hidden_dim, n_classes * bilinear_rank, bias=False)

        # Xavier uniform init (proved superior in node4-2-1 for decisive Cycle 2)
        self.out_gene_emb = nn.Embedding(n_genes_out, bilinear_rank)
        nn.init.xavier_uniform_(self.out_gene_emb.weight)

        # Cache for precomputed frozen intermediate embeddings
        self._frozen_emb_cache: Optional[torch.Tensor] = None

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Node4-2-1-1-1] Trainable: {n_trainable:,} / {n_total:,} params "
              f"({100*n_trainable/n_total:.2f}%)")

    def _compute_backbone_embs(self) -> torch.Tensor:
        ei = self.edge_index
        ew = self.edge_weight

        if self._frozen_emb_cache is None:
            x = self.emb.weight
            for layer in self.mps_0_5:
                x = layer(x, ei, ew)
            self._frozen_emb_cache = x.detach()

        x = self._frozen_emb_cache

        # Trainable tail: mps.6 + mps.7 + post_mp
        x = self.mps_6(x, ei, ew)
        x = self.mps_7(x, ei, ew)
        x = self.post_mp(x)
        return x

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        node_emb   = self._compute_backbone_embs()      # [N_nodes, 256]
        known_mask = (node_indices >= 0)
        safe_idx   = node_indices.clamp(min=0)
        pert_emb   = node_emb[safe_idx, :]              # [B, 256]
        if not known_mask.all():
            fallback = self.fallback_emb.unsqueeze(0).expand_as(pert_emb)
            pert_emb = torch.where(
                known_mask.unsqueeze(-1).expand_as(pert_emb),
                pert_emb, fallback,
            )
        pert_emb = pert_emb.float()
        h        = self.input_proj(pert_emb)
        for block in self.res_blocks:
            h = block(h)
        blin   = self.fc_bilinear(h).view(-1, self.n_classes, self.bilinear_rank)
        out_e  = self.out_gene_emb.weight
        logits = torch.matmul(blin, out_e.T)
        return logits


# ─── DDP gather helpers ───────────────────────────────────────────────────────

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


# ─── LightningModule ──────────────────────────────────────────────────────────

class StringGNNReducedHeadLitModule(pl.LightningModule):

    def __init__(
        self,
        backbone_lr:     float = 1e-5,
        head_lr:         float = 5e-4,
        muon_lr:         float = 0.005,
        weight_decay:    float = 1e-3,
        gene_emb_wd:     float = 1e-2,
        t0_steps:        int   = 1200,
        warmup_steps:    int   = 100,
        max_steps:       int   = 10000,
        focal_gamma:     float = FOCAL_GAMMA,
        label_smoothing: float = 0.07,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str]          = []
        self._test_symbols:  List[str]          = []
        self._test_labels:   List[torch.Tensor] = []
        # Track val_f1 per epoch for SWA checkpoint selection
        self.val_f1_history: Dict[int, float] = {}

    def setup(self, stage=None):
        dm = self.trainer.datamodule if self.trainer is not None else None
        if dm is None:
            raise RuntimeError("DataModule must be attached to the trainer.")
        self.model = StringGNNReducedHeadModel(
            edge_index  = dm.edge_index,
            edge_weight = dm.edge_weight,
        )
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(self, node_indices):
        return self.model(node_indices)

    def _loss(self, logits, labels):
        return focal_cross_entropy_with_smoothing(
            logits, labels,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
            label_smoothing=self.hparams.label_smoothing,
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
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self.val_f1_history[self.current_epoch] = f1
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_index"])
        probs  = torch.softmax(logits.float(), dim=1)
        self._test_preds.append(probs.detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (
            torch.cat(self._test_labels, 0) if self._test_labels
            else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        )
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

        # Deduplicate (DDP DistributedSampler pads)
        seen: set = set()
        keep: List[int] = []
        for i, pid in enumerate(all_pert):
            if pid not in seen:
                seen.add(pid); keep.append(i)
        if len(keep) < len(all_pert):
            self.print(f"[Node4-2-1-1-1] Deduplicating: {len(all_pert)} -> {len(keep)}")
            all_probs  = all_probs[keep]
            all_labels = all_labels[keep]
            all_pert   = [all_pert[i] for i in keep]
            all_syms   = [all_syms[i] for i in keep]

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pid, sym, probs in zip(all_pert, all_syms, all_probs.numpy()):
                    fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")
            self.print(f"[Node4-2-1-1-1] Saved predictions -> {pred_path}")
            if all_labels.any():
                f1 = compute_per_gene_f1(all_probs.numpy(), all_labels.numpy())
                self.print(f"[Node4-2-1-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear(); self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        backbone_params = (
            list(self.model.mps_6.parameters()) +
            list(self.model.mps_7.parameters()) +
            list(self.model.post_mp.parameters())
        )
        backbone_param_ids = {id(p) for p in backbone_params}

        gene_emb_params    = [self.model.out_gene_emb.weight]
        gene_emb_param_ids = {id(p) for p in gene_emb_params}

        head_2d_matrices = []
        head_other       = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in backbone_param_ids:
                continue
            if id(param) in gene_emb_param_ids:
                continue
            if (param.ndim >= 2
                    and "input_proj.1" not in name
                    and "fallback_emb" not in name):
                head_2d_matrices.append(param)
            else:
                head_other.append(param)

        param_groups = [
            {
                "params":       backbone_params,
                "use_muon":     False,
                "lr":           hp.backbone_lr,
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.weight_decay,
            },
            {
                "params":       head_2d_matrices,
                "use_muon":     True,
                "lr":           hp.muon_lr,
                "momentum":     0.95,
                "weight_decay": hp.weight_decay,
            },
            # out_gene_emb: strong WD + head_lr=5e-4 (restored from node4-2-1)
            {
                "params":       gene_emb_params,
                "use_muon":     False,
                "lr":           hp.head_lr,
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.gene_emb_wd,
            },
            {
                "params":       head_other,
                "use_muon":     False,
                "lr":           hp.head_lr,
                "betas":        (0.9, 0.95),
                "eps":          1e-8,
                "weight_decay": hp.weight_decay * 0.1,
            },
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        warmup = hp.warmup_steps
        T_0    = hp.t0_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / float(max(1, warmup))
            step_after = step - warmup
            cycle_pos  = step_after % T_0
            progress   = float(cycle_pos) / float(T_0)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
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


# ─── Post-Training Quality-Filtered SWA ───────────────────────────────────────

def run_quality_filtered_swa(
    datamodule: StringGNNDataModule,
    checkpoint_dir: Path,
    output_dir: Path,
    val_f1_history: Dict[int, float],
    min_val_f1: float = 0.490,
    top_k: int = 3,
):
    """
    Post-training quality-filtered SWA on rank 0 only.
    Averages the best single-epoch checkpoint and the last checkpoint.
    If more checkpoints are available, takes up to top_k best ones.
    Overwrites test_predictions.tsv with SWA predictions.
    """
    print(f"\n[SWA] Starting quality-filtered SWA (min_val_f1={min_val_f1}, top_k={top_k})")

    if not checkpoint_dir.exists():
        print(f"[SWA] Checkpoint directory not found. Skipping.")
        return

    all_ckpts = list(checkpoint_dir.glob("*.ckpt"))
    if not all_ckpts:
        print(f"[SWA] No checkpoints found. Skipping.")
        return

    # Parse checkpoint scores from filenames (best-EPOCH-F1.ckpt)
    scored_ckpts = []
    for ckpt_path in all_ckpts:
        stem = ckpt_path.stem
        if stem == "last":
            # Use the max val_f1 from history as a proxy for the last checkpoint
            if val_f1_history:
                last_epoch = max(val_f1_history.keys())
                last_f1    = val_f1_history.get(last_epoch, 0.0)
                scored_ckpts.append((ckpt_path, last_f1, last_epoch, "last"))
        elif stem.startswith("best-"):
            parts = stem.split("-")
            if len(parts) >= 3:
                try:
                    epoch  = int(parts[1])
                    val_f1 = float(parts[2])
                    scored_ckpts.append((ckpt_path, val_f1, epoch, "best"))
                except (ValueError, IndexError):
                    pass

    if not scored_ckpts:
        print(f"[SWA] Could not parse checkpoint scores. Skipping.")
        return

    # Sort by val_f1 descending; always include the best checkpoint
    scored_ckpts.sort(key=lambda x: -x[1])
    print(f"[SWA] Found {len(scored_ckpts)} checkpoints. Top:")
    for path, f1, epoch, tag in scored_ckpts[:5]:
        print(f"  epoch={epoch:3d}, val_f1={f1:.4f}, tag={tag}: {path.name}")

    # Quality filter: keep top_k with val_f1 >= min_val_f1
    qualified = [(p, f, e, t) for p, f, e, t in scored_ckpts if f >= min_val_f1]
    if len(qualified) == 0:
        # Fallback: just use the best checkpoint even if below threshold
        qualified = scored_ckpts[:1]
        print(f"[SWA] No checkpoints above threshold; using best only.")
    else:
        qualified = qualified[:top_k]

    # Deduplicate by path
    seen_paths: set = set()
    unique_q = []
    for item in qualified:
        if str(item[0]) not in seen_paths:
            seen_paths.add(str(item[0]))
            unique_q.append(item)
    qualified = unique_q

    print(f"[SWA] Using {len(qualified)} checkpoints for SWA averaging:")
    for path, f1, epoch, tag in qualified:
        print(f"  epoch={epoch}, val_f1={f1:.4f}: {path.name}")

    if len(qualified) <= 1:
        print(f"[SWA] Only 1 checkpoint after dedup; SWA averaging skipped.")
        return

    try:
        # Load and average state dicts
        avg_sd: Dict[str, torch.Tensor] = {}
        n_loaded = 0
        for ckpt_path, _, _, _ in qualified:
            raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            sd  = raw.get("state_dict", raw)
            if n_loaded == 0:
                avg_sd = {k: v.clone().float() for k, v in sd.items()}
            else:
                for k in avg_sd:
                    if k in sd:
                        avg_sd[k] = avg_sd[k] + sd[k].float()
            n_loaded += 1

        for k in avg_sd:
            avg_sd[k] = avg_sd[k] / n_loaded

        print(f"[SWA] Averaged {n_loaded} checkpoints.")

        # Build a fresh model for SWA inference
        # DataModule must be already setup (it is after trainer.fit)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        swa_model = StringGNNReducedHeadModel(
            edge_index  = datamodule.edge_index.to(device),
            edge_weight = datamodule.edge_weight.to(device) if datamodule.edge_weight is not None else None,
        )
        swa_model = swa_model.to(device)
        for p in swa_model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Load the averaged state dict
        # Lightning prefixes: "model." for the nn.Module inside LightningModule
        model_sd = {}
        for k, v in avg_sd.items():
            if k.startswith("model."):
                model_sd[k[len("model."):]] = v
        result = swa_model.load_state_dict(model_sd, strict=False)
        print(f"[SWA] Loaded averaged weights. "
              f"Missing: {len(result.missing_keys)}, unexpected: {len(result.unexpected_keys)}")

        swa_model.eval()

        # Run inference on test set
        test_loader = datamodule.test_dataloader()
        all_probs_list    = []
        all_pert_ids_list = []
        all_syms_list     = []
        all_labels_list   = []

        with torch.no_grad():
            for batch in test_loader:
                node_idx = batch["node_index"].to(device)
                logits   = swa_model(node_idx)
                probs    = torch.softmax(logits.float(), dim=1).cpu()
                all_probs_list.append(probs)
                all_pert_ids_list.extend(batch["pert_id"])
                all_syms_list.extend(batch["symbol"])
                if "label" in batch:
                    all_labels_list.append(batch["label"])

        all_probs = torch.cat(all_probs_list, 0)

        # Overwrite test_predictions.tsv with SWA predictions
        pred_path = output_dir / "test_predictions.tsv"
        with open(pred_path, "w") as fh:
            fh.write("idx\tinput\tprediction\n")
            for pid, sym, probs in zip(all_pert_ids_list, all_syms_list, all_probs.numpy()):
                fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")
        print(f"[SWA] Saved SWA predictions -> {pred_path}")

        if all_labels_list:
            all_labels = torch.cat(all_labels_list, 0)
            swa_f1 = compute_per_gene_f1(all_probs.numpy(), all_labels.numpy())
            print(f"[SWA] SWA self-computed test F1 = {swa_f1:.4f}")

        # Free GPU memory
        del swa_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        print(f"[SWA] WARNING: SWA failed: {e}. Keeping single-checkpoint predictions.")
        traceback.print_exc()


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 4-2-1-1-1 — STRING_GNN + 4xResBlock(256) + rank256 + LabelSmooth=0.07 + SWA"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--backbone-lr",        type=float, default=1e-5)
    p.add_argument("--head-lr",            type=float, default=5e-4)
    p.add_argument("--muon-lr",            type=float, default=0.005)
    p.add_argument("--weight-decay",       type=float, default=1e-3)
    p.add_argument("--gene-emb-wd",        type=float, default=1e-2)
    p.add_argument("--label-smoothing",    type=float, default=0.07)
    p.add_argument("--micro-batch-size",   type=int,   default=8)
    p.add_argument("--global-batch-size",  type=int,   default=32)
    p.add_argument("--max-epochs",         type=int,   default=250)
    p.add_argument("--patience",           type=int,   default=50)
    p.add_argument("--t0-steps",           type=int,   default=1200)
    p.add_argument("--warmup-steps",       type=int,   default=100)
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--swa-top-k",          type=int,   default=3,
                   help="Top-k checkpoints to SWA-average (default=3)")
    p.add_argument("--swa-min-val-f1",     type=float, default=0.490,
                   help="Min val_f1 threshold for SWA inclusion")
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    steps_per_epoch     = int(np.ceil(1416 / args.micro_batch_size))
    estimated_max_steps = args.max_epochs * steps_per_epoch // accum

    dm  = StringGNNDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = StringGNNReducedHeadLitModule(
        backbone_lr      = args.backbone_lr,
        head_lr          = args.head_lr,
        muon_lr          = args.muon_lr,
        weight_decay     = args.weight_decay,
        gene_emb_wd      = args.gene_emb_wd,
        t0_steps         = args.t0_steps,
        warmup_steps     = args.warmup_steps,
        max_steps        = estimated_max_steps,
        label_smoothing  = args.label_smoothing,
    )

    ckpt_dir = out_dir / "checkpoints"
    ckpt_cb  = ModelCheckpoint(
        dirpath  = str(ckpt_dir),
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
        limit_val   = 2
        limit_test  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator          = "gpu",
        devices              = n_gpus,
        num_nodes            = 1,
        strategy             = strategy,
        precision            = "bf16-mixed",
        max_epochs           = args.max_epochs,
        max_steps            = max_steps,
        accumulate_grad_batches = accum,
        gradient_clip_val    = 1.0,
        limit_train_batches  = limit_train,
        limit_val_batches    = limit_val,
        limit_test_batches   = limit_test,
        val_check_interval   = (
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps = 2,
        callbacks            = [ckpt_cb, es_cb, lr_cb, pb_cb],
        logger               = [csv_logger, tb_logger],
        log_every_n_steps    = 10,
        deterministic        = True,
        default_root_dir     = str(out_dir),
        fast_dev_run         = fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    # Post-training quality-filtered SWA on global rank 0 only
    if (trainer.is_global_zero
            and args.debug_max_step is None
            and not args.fast_dev_run):
        run_quality_filtered_swa(
            datamodule      = dm,
            checkpoint_dir  = ckpt_dir,
            output_dir      = out_dir,
            val_f1_history  = lit.val_f1_history,
            min_val_f1      = args.swa_min_val_f1,
            top_k           = args.swa_top_k,
        )

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 4-2-1-1-1 — STRING_GNN(mps.6+7+post_mp) + 4xResBlock(256)+rank256 "
            "+ LabelSmooth=0.07 + SWA\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
