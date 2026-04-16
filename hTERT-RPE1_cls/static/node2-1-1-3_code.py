"""
Node 2-1-1-3 — STRING_GNN Partial Fine-Tuning + Muon Optimizer + SGDR Warm Restarts + SWA

Architecture (same backbone as sibling node2-1-1-2, F1=0.5000):
  - STRING_GNN partial fine-tuning (mps.6, mps.7, post_mp, ~198K trainable params)
  - Frozen early layers (emb, mps.0-5) precomputed as buffer
  - 6-layer deep residual bilinear MLP head (rank=512, hidden=512)
  - Class-weighted focal loss (gamma=2.0, weights=[2.0, 0.5, 4.0])

Key improvements over sibling node2-1-1-2 (F1=0.5000):
  1. Muon optimizer for ResidualBlock 2D weight matrices (Muon lr=0.005)
     - Proven to push F1 from 0.5000 to 0.5060+ in node1-2-2-2 vs node1-2-3-2
     - Orthogonalized momentum updates enable faster convergence in small-data regime
  2. SGDR warm restarts (T_0=20, T_mult=2) for multi-cycle checkpoint diversity
     - Creates rich pool of high-quality checkpoints across LR cycles
     - Validated in tree-best lineage (node2-1-1-1-2-1-1-1-1-1-1-1, F1=0.5180)
  3. Quality-filtered SWA (top-K by val_f1, temperature-weighted) post-hoc ensemble
     - Top-20 checkpoints with threshold=0.480, temperature=3.0
     - Provides +0.004-0.007 F1 gain over single best checkpoint
  4. Stronger regularization package based on sibling + tree-best insights:
     - head_dropout: 0.25 → 0.40 (proven in tree best)
     - weight_decay: 1.5e-3 → 3e-3 (sibling feedback)
     - label_smoothing: 0 → 0.07 (tree best lineage proven)
  5. Extended patience=150 to capture multi-cycle SGDR improvement staircase

Design rationale:
  The sibling node2-1-1-2 achieved F1=0.5000 with plain AdamW cosine decay. The tree
  best (F1=0.5182) uses Muon + SGDR + SWA. This node bridges the gap by adding exactly
  those three proven techniques to the sibling's solid foundation.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
import glob
import re
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

try:
    from muon import MuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    print("[Node2-1-1-3] WARNING: muon not found, falling back to AdamW for all params")

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_HIDDEN     = 256   # STRING_GNN hidden size


# ─── Class-Weighted Focal Loss ────────────────────────────────────────────────

class ClassWeightedFocalLoss(nn.Module):
    """
    Class-weighted focal loss with label smoothing.
    Combines class weights for explicit minority class emphasis with
    focal (1-pt)^gamma modulation for down-weighting easy examples.
    Label smoothing reduces overconfidence.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.07,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, C] (2D, already reshaped)
        targets: [N] long
        """
        # Compute CE with label smoothing and class weights
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.class_weights if self.class_weights is not None else None,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        # Get probability of the true class (without smoothing) for focal weighting
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction='none'))
        # Focal weight: (1 - pt)^gamma down-weights easy examples
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Per-gene macro F1 matching calc_metric.py.  pred_np: [N,3,G], labels_np: [N,G]."""
    pred_cls = pred_np.argmax(axis=1)
    f1_vals = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    """Stores pert_ids, symbols, GNN node indices, and (optionally) labels."""

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        gnn_node_indices: List[int],   # -1 for OOV
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long or None
    ):
        self.pert_ids         = pert_ids
        self.symbols          = symbols
        self.gnn_node_indices = gnn_node_indices
        self.labels           = labels

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":       self.pert_ids[idx],
            "symbol":        self.symbols[idx],
            "gnn_node_idx":  self.gnn_node_indices[idx],  # int
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    pert_ids      = [b["pert_id"]      for b in batch]
    symbols       = [b["symbol"]       for b in batch]
    gnn_node_idxs = [b["gnn_node_idx"] for b in batch]
    out = {
        "pert_id":      pert_ids,
        "symbol":       symbols,
        "gnn_node_idx": torch.tensor(gnn_node_idxs, dtype=torch.long),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage: Optional[str] = None):
        # Build STRING_GNN node name → index lookup
        node_names = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.gnn_node_name_to_idx = {name: i for i, name in enumerate(node_names)}

        def load_split(fname: str, has_label: bool):
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            pert_ids = df["pert_id"].tolist()
            symbols  = df["symbol"].tolist()
            gnn_node_indices = [
                self.gnn_node_name_to_idx.get(pid, -1)
                for pid in pert_ids
            ]
            labels = None
            if has_label and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return PerturbDataset(pert_ids, symbols, gnn_node_indices, labels)

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  False)

    def _loader(self, ds, shuffle, drop_last=False):
        return DataLoader(
            ds, batch_size=self.micro_batch_size,
            shuffle=shuffle, collate_fn=collate_fn,
            num_workers=self.num_workers, pin_memory=True,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True, drop_last=True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Model Components ──────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Pre-norm residual feedforward block:
    LN -> Linear(d, d*expand) -> GELU -> Dropout -> Linear(d*expand, d) -> Dropout -> skip
    """
    def __init__(self, d: int, expand: int = 4, dropout: float = 0.40):
        super().__init__()
        d_inner = d * expand
        self.norm   = nn.LayerNorm(d)
        self.fc1    = nn.Linear(d, d_inner)
        self.act    = nn.GELU()
        self.drop1  = nn.Dropout(dropout)
        self.fc2    = nn.Linear(d_inner, d)
        self.drop2  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop2(self.fc2(self.drop1(self.act(self.fc1(self.norm(x))))))
        return x + h


class GNNBilinearHead(nn.Module):
    """
    6-layer deep residual bilinear MLP head (proven in node1-2, F1=0.4912
    and node2-1-1-2, F1=0.5000).

    Architecture:
      LayerNorm(256) -> Linear(256->512)
      -> 6x ResidualBlock(512, expand=4, dropout=0.40)
      -> LayerNorm(512) + Dropout(0.40)
      -> Linear(512 -> 3*512) -> reshape [B, 3, 512]
      -> einsum("bcr,gr->bcg", pert_proj, out_gene_emb) -> [B, 3, 6640]
    """
    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 512,
        n_classes: int = 3,
        bilinear_rank: int = 512,
        n_genes_out: int = N_GENES_OUT,
        n_blocks: int = 6,
        expand: int = 4,
        dropout: float = 0.40,
    ):
        super().__init__()
        self.n_classes    = n_classes
        self.bilinear_rank = bilinear_rank

        self.input_norm   = nn.LayerNorm(in_dim)
        self.input_proj   = nn.Linear(in_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=expand, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, n_classes * bilinear_rank)

        # Learnable output gene embeddings [n_genes_out, bilinear_rank]
        # Random init (std=0.02) -- no positional misalignment
        self.out_gene_emb = nn.Parameter(
            torch.randn(n_genes_out, bilinear_rank) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim] -> logits: [B, n_classes, n_genes_out]"""
        B = x.shape[0]
        h = self.input_proj(self.input_norm(x))
        for block in self.blocks:
            h = block(h)
        h = self.out_drop(self.out_norm(h))
        # Project to [B, 3*rank] and reshape
        pert_proj = self.out_proj(h).view(B, self.n_classes, self.bilinear_rank)
        # Bilinear interaction: [B, 3, rank] x [G, rank]^T -> [B, 3, G]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)
        return logits


class GNNPerturbModel(nn.Module):
    """
    STRING_GNN partial fine-tuning model:
    - Frozen early layers (emb, mps.0-5) precomputed as fixed buffer
    - Trainable tail (mps.6, mps.7, post_mp) updated at backbone_lr
    - Deep bilinear MLP head maps 256-dim embedding to [B, 3, 6640] logits
    """

    def __init__(
        self,
        bilinear_rank: int = 512,
        hidden_dim: int = 512,
        n_blocks: int = 6,
        expand: int = 4,
        head_dropout: float = 0.40,
    ):
        super().__init__()

        from transformers import AutoModel

        # Load full STRING_GNN
        full_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        for param in full_gnn.parameters():
            param.data = param.data.float()

        # Freeze early layers: emb + mps.0 through mps.5
        for name, param in full_gnn.named_parameters():
            is_early = (
                name.startswith("emb.")
                or any(name.startswith(f"mps.{i}.") for i in range(6))
            )
            param.requires_grad = not is_early

        self.gnn = full_gnn

        # Load graph data
        graph_data = torch.load(
            Path(STRING_GNN_DIR) / "graph_data.pt", weights_only=False
        )
        self.register_buffer("edge_index",  graph_data["edge_index"].long())
        edge_weight = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight.float())
        else:
            self.register_buffer("edge_weight", None)

        # OOV embedding (learnable, for genes not in STRING_GNN vocab)
        self.gnn_oov_emb = nn.Parameter(torch.zeros(GNN_HIDDEN))

        # Bilinear prediction head
        self.head = GNNBilinearHead(
            in_dim=GNN_HIDDEN,
            hidden_dim=hidden_dim,
            bilinear_rank=bilinear_rank,
            n_blocks=n_blocks,
            expand=expand,
            dropout=head_dropout,
        )

        # Frozen node states buffer (precomputed in setup(), stored on CPU first)
        # Will be registered after precomputation
        self._frozen_states_computed = False

    def precompute_frozen_states(self):
        """
        Run frozen early layers (emb + mps.0-5) once to get intermediate states.
        Stored as a buffer to avoid recomputation every forward pass.
        This dramatically speeds up training by avoiding the frozen part of GNN.
        """
        with torch.no_grad():
            # Get initial embeddings
            x = self.gnn.emb.weight.data.clone()

            # Get edge data on CPU
            edge_index_cpu  = self.edge_index.cpu()
            edge_weight_cpu = self.edge_weight.cpu() if self.edge_weight is not None else None

            # Run through frozen layers 0-5 on CPU
            for i in range(6):
                mp = self.gnn.mps[i].cpu()
                x = x.to(mp.conv.lin.weight.device)
                out = mp(x, edge_index_cpu, edge_weight_cpu)
                x = out + x  # residual connection

            self.register_buffer("frozen_node_states", x.float())  # [18870, 256]
            self._frozen_states_computed = True

    def forward(self, gnn_node_idxs: torch.Tensor) -> torch.Tensor:
        """
        gnn_node_idxs: [B] long — STRING_GNN node index (-1 = OOV)
        Returns logits: [B, 3, 6640]
        """
        B      = gnn_node_idxs.shape[0]
        device = gnn_node_idxs.device

        if not self._frozen_states_computed:
            raise RuntimeError("precompute_frozen_states() must be called before forward()")

        # Start from precomputed frozen intermediate states
        # Ensure float32 to avoid dtype mismatch under bf16-mixed autocast
        x = self.frozen_node_states.to(device).float()

        # Run trainable tail: mps.6, mps.7, post_mp
        edge_index  = self.edge_index.to(device)
        edge_weight = self.edge_weight.to(device) if self.edge_weight is not None else None

        x_6 = self.gnn.mps[6](x, edge_index, edge_weight)
        x = (x + x_6).float()  # residual, ensure float32

        x_7 = self.gnn.mps[7](x, edge_index, edge_weight)
        x = (x + x_7).float()  # residual, ensure float32

        all_node_embs = self.gnn.post_mp(x).float()  # [18870, 256], ensure float32

        # Per-sample embedding extraction
        in_vocab_mask = (gnn_node_idxs >= 0)
        embs = torch.zeros(B, GNN_HIDDEN, dtype=torch.float32, device=device)
        if in_vocab_mask.any():
            valid_idxs = gnn_node_idxs[in_vocab_mask]
            embs[in_vocab_mask] = all_node_embs[valid_idxs]
        oov_mask = ~in_vocab_mask
        if oov_mask.any():
            embs[oov_mask] = self.gnn_oov_emb.float().unsqueeze(0).expand(oov_mask.sum(), -1)

        # Head: [B, 256] -> [B, 3, 6640]
        logits = self.head(embs)
        return logits


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))

    pad = max_sz - local_p.shape[0]
    p   = local_p.to(device)
    l   = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)

    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p)
    dist.all_gather(gl, l)

    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class GNNLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_backbone: float = 5e-5,
        lr_head: float = 5e-4,
        muon_lr: float = 0.005,
        weight_decay: float = 3e-3,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.07,
        head_dropout: float = 0.40,
        bilinear_rank: int = 512,
        sgdr_t0: int = 20,          # SGDR T_0 in epochs
        sgdr_t_mult: float = 2.0,   # SGDR T_mult
        # SWA parameters
        swa_start_epoch: int = 15,  # start collecting periodic checkpoints
        swa_every_n_epochs: int = 3,
        swa_threshold: float = 0.480,   # minimum val_f1 to qualify
        swa_top_k: int = 20,
        swa_temperature: float = 3.0,
        # Training params
        warmup_steps: int = 100,
        max_steps_sgdr: int = 3000,     # total steps for SGDR LR reference
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []
        self._test_labels:   List[torch.Tensor] = []

    def setup(self, stage: Optional[str] = None):
        self.model = GNNPerturbModel(
            bilinear_rank=self.hparams.bilinear_rank,
            head_dropout=self.hparams.head_dropout,
        )

        # Precompute frozen GNN states (avoids re-running 6 GCN layers each step)
        self.model.precompute_frozen_states()

        # Build loss
        class_weights = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float32)
        self.loss_fn = ClassWeightedFocalLoss(
            gamma=self.hparams.focal_gamma,
            class_weights=class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

    def forward(self, gnn_node_idxs):
        return self.model(gnn_node_idxs)

    def _loss(self, logits, labels):
        # logits: [B, 3, 6640] -> [B*6640, 3];  labels: [B, 6640] -> [B*6640]
        logits_2d = logits.float().permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_1d = labels.reshape(-1)
        return self.loss_fn(logits_2d, labels_1d)

    def training_step(self, batch, batch_idx):
        logits = self(batch["gnn_node_idx"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["gnn_node_idx"])
        if "label" in batch:
            loss = self._loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        lp = torch.cat(self._val_preds,  0)   # [N_local, 3, 6640] on CPU
        ll = torch.cat(self._val_labels, 0)   # [N_local, 6640] on CPU
        if self.trainer.world_size > 1:
            # Use Lightning's all_gather (more reliable than raw dist.all_gather).
            # Move to device first since NCCL requires CUDA tensors.
            # DDP distributed sampler pads dataset so all ranks have equal N_local.
            lp_dev = lp.to(self.device)
            ll_dev = ll.to(self.device)
            # all_gather returns [world_size, N_local, ...]; reshape to [world_size*N_local, ...]
            lp = self.all_gather(lp_dev).reshape(-1, *lp.shape[1:]).cpu()
            ll = self.all_gather(ll_dev).reshape(-1, *ll.shape[1:]).cpu()
        f1 = compute_per_gene_f1(lp.numpy(), ll.numpy())
        # sync_dist=False: all ranks compute the same f1 from gathered data
        self.log("val_f1", f1, prog_bar=True, sync_dist=False)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["gnn_node_idx"])
        probs  = torch.softmax(logits, dim=1)
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (
            torch.cat(self._test_labels, 0)
            if self._test_labels
            else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        )

        if self.trainer.world_size > 1:
            # Use Lightning's all_gather for reliable DDP synchronization.
            # Move to device first since NCCL requires CUDA tensors.
            lp_dev = local_probs.to(self.device)
            ll_dev = dummy_labels.to(self.device)
            all_probs  = self.all_gather(lp_dev).reshape(-1, *local_probs.shape[1:]).cpu()
            all_labels = self.all_gather(ll_dev).reshape(-1, *dummy_labels.shape[1:]).cpu()
            all_pert_gathered = [None] * self.trainer.world_size
            all_syms_gathered = [None] * self.trainer.world_size
            dist.all_gather_object(all_pert_gathered, self._test_pert_ids)
            dist.all_gather_object(all_syms_gathered, self._test_symbols)
            all_pert = [p for sub in all_pert_gathered for p in sub]
            all_syms = [s for sub in all_syms_gathered for s in sub]
        else:
            all_probs, all_labels = local_probs, dummy_labels
            all_pert, all_syms    = self._test_pert_ids, self._test_symbols

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id (DDP pads with duplicates)
            seen_pids: set = set()
            dedup_indices: List[int] = []
            for i, pid in enumerate(all_pert):
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    dedup_indices.append(i)

            all_probs_np = all_probs.numpy()
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for i in dedup_indices:
                    fh.write(
                        f"{all_pert[i]}\t{all_syms[i]}\t"
                        f"{json.dumps(all_probs_np[i].tolist())}\n"
                    )
            self.print(f"[Node2-1-1-3] Saved {len(dedup_indices)} test predictions → {pred_path}")

            if self._test_labels:
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-1-1-3] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Parameter separation for Muon (2D matrices in residual blocks) vs AdamW (rest)
        # STRING_GNN backbone trainable params: mps.6, mps.7, post_mp
        backbone_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and n.startswith("gnn.")
        ]
        backbone_set = set(id(p) for p in backbone_params)

        # Head params: OOV embedding + head sub-modules
        all_head_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and id(p) not in backbone_set
        ]

        if MUON_AVAILABLE:
            # Separate head params into 2D weight matrices (for Muon) vs rest (for AdamW)
            # Muon handles: ResidualBlock fc1.weight, fc2.weight (2D matrices in hidden layers)
            # AdamW handles: norms, biases, out_gene_emb, OOV emb, input/output projections
            muon_params = [
                p for n, p in self.model.head.named_parameters()
                if p.requires_grad and p.ndim >= 2
                and any(x in n for x in ["fc1.weight", "fc2.weight"])
            ]
            muon_set = set(id(p) for p in muon_params)

            adamw_head_params = [
                p for p in all_head_params
                if id(p) not in muon_set
            ]

            # Also include OOV embedding in AdamW head params
            oov_set = {id(self.model.gnn_oov_emb)}

            param_groups = [
                # Backbone: trainable GNN tail (AdamW with low LR)
                dict(params=backbone_params,   use_muon=False,
                     lr=hp.lr_backbone, weight_decay=hp.weight_decay,
                     betas=(0.9, 0.95)),
                # Head 2D matrices: Muon with high LR
                dict(params=muon_params,       use_muon=True,
                     lr=hp.muon_lr, weight_decay=hp.weight_decay,
                     momentum=0.95),
                # Head other params: AdamW
                dict(params=adamw_head_params, use_muon=False,
                     lr=hp.lr_head, weight_decay=hp.weight_decay,
                     betas=(0.9, 0.95)),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
            self.print(f"[Node2-1-1-3] Using MuonWithAuxAdam: {len(muon_params)} Muon tensors, "
                       f"{len(backbone_params)} backbone tensors, "
                       f"{len(adamw_head_params)} AdamW head tensors")
        else:
            # Fallback to AdamW for all parameters
            optimizer = torch.optim.AdamW([
                {"params": backbone_params,  "lr": hp.lr_backbone, "weight_decay": hp.weight_decay},
                {"params": all_head_params,  "lr": hp.lr_head,     "weight_decay": hp.weight_decay},
            ])
            self.print("[Node2-1-1-3] Fallback: Using AdamW for all parameters")

        # SGDR cosine warm restarts via LambdaLR
        # T_0 cycles in steps, with T_mult factor
        warmup  = hp.warmup_steps

        def sgdr_lr_lambda(current_step: int) -> float:
            # Linear warmup
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))

            # SGDR: determine current cycle and position within cycle
            step_after_warmup = current_step - warmup
            T0    = hp.sgdr_t0 * _steps_per_epoch
            T_mult = hp.sgdr_t_mult

            # Find current cycle boundary
            cycle = 0
            t_cur = float(step_after_warmup)
            T_cur = float(T0)
            while t_cur >= T_cur:
                t_cur -= T_cur
                T_cur *= T_mult
                cycle += 1

            # Position within current cycle [0, 1]
            progress = t_cur / max(1.0, T_cur)
            # Clamp to prevent edge effects
            progress = min(1.0, progress)
            return max(1e-6, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, sgdr_lr_lambda)
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


# ─── SWA Logic ────────────────────────────────────────────────────────────────

def run_quality_filtered_swa(
    model_module: GNNLitModule,
    datamodule: PerturbDataModule,
    checkpoint_dir: str,
    output_dir: str,
    top_k: int = 20,
    threshold: float = 0.480,
    temperature: float = 3.0,
    device: str = "cuda",
) -> Optional[float]:
    """
    Post-hoc quality-filtered exponentially-weighted SWA.

    Scans checkpoint_dir for periodic checkpoints, extracts val_f1 from filenames,
    filters by threshold, selects top-K, computes temperature-weighted average
    state dict, runs test inference, and saves results.

    Returns the SWA test F1 score or None if insufficient checkpoints.
    """
    from transformers import AutoModel

    # Find all periodic checkpoint files
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "periodic-*.ckpt"))
    if len(ckpt_files) == 0:
        print("[SWA] No periodic checkpoints found. Skipping SWA.")
        return None

    # Parse val_f1 from filenames using regex
    # Expected format: periodic-epoch=NNNN-val_f1=X.XXXX.ckpt
    # PL may generate: periodic-epoch=0099-val_f1=val_f1=0.4912.ckpt or
    #                  periodic-epoch=0099-val_f1=0.4912.ckpt
    ckpt_scores: List[Tuple[str, float]] = []
    for ckpt_path in ckpt_files:
        fname = os.path.basename(ckpt_path)
        # Try to extract val_f1 value
        m = re.search(r'val_f1=(?:val_f1=)?([0-9]+\.[0-9]+)', fname)
        if m:
            val_f1 = float(m.group(1))
            ckpt_scores.append((ckpt_path, val_f1))
        else:
            print(f"[SWA] Could not parse val_f1 from: {fname}")

    if len(ckpt_scores) == 0:
        print("[SWA] No parseable periodic checkpoints. Skipping SWA.")
        return None

    print(f"[SWA] Found {len(ckpt_scores)} periodic checkpoints")
    print(f"[SWA] Val F1 range: [{min(s for _, s in ckpt_scores):.4f}, {max(s for _, s in ckpt_scores):.4f}]")

    # Filter by threshold
    qualified = [(p, s) for p, s in ckpt_scores if s >= threshold]
    if len(qualified) == 0:
        print(f"[SWA] No checkpoints above threshold {threshold:.3f}. Using top-5 regardless.")
        qualified = sorted(ckpt_scores, key=lambda x: x[1], reverse=True)[:5]

    print(f"[SWA] {len(qualified)} checkpoints above threshold {threshold:.3f}")

    # Select top-K by val_f1
    qualified_sorted = sorted(qualified, key=lambda x: x[1], reverse=True)
    selected = qualified_sorted[:top_k]
    print(f"[SWA] Selected top-{len(selected)} checkpoints:")
    for p, s in selected:
        print(f"  {os.path.basename(p)}: val_f1={s:.4f}")

    # Compute temperature-softmax weights (higher val_f1 → higher weight)
    scores_arr = np.array([s for _, s in selected])
    # Scale scores to reasonable range before softmax
    scaled = scores_arr * temperature
    weights = np.exp(scaled - scaled.max())
    weights = weights / weights.sum()
    print(f"[SWA] Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

    # Load and average state dicts (on CPU)
    avg_state_dict = None
    for i, (ckpt_path, val_f1) in enumerate(selected):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        w = float(weights[i])
        if avg_state_dict is None:
            avg_state_dict = {k: v.float() * w for k, v in state.items()}
        else:
            for k in avg_state_dict:
                if k in state:
                    avg_state_dict[k] = avg_state_dict[k] + state[k].float() * w

    if avg_state_dict is None:
        print("[SWA] Failed to build averaged state dict.")
        return None

    # Load averaged weights into model
    model_module.load_state_dict(avg_state_dict, strict=False)
    model_module = model_module.to(device)
    model_module.eval()

    # Run inference on test set
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    all_probs_list = []
    all_pids_list  = []
    all_syms_list  = []
    all_lbls_list  = []

    with torch.no_grad():
        for batch in test_loader:
            gnn_idx = batch["gnn_node_idx"].to(device)
            logits  = model_module(gnn_idx)
            probs   = torch.softmax(logits, dim=1).cpu().float()
            all_probs_list.append(probs)
            all_pids_list.extend(batch["pert_id"])
            all_syms_list.extend(batch["symbol"])
            if "label" in batch:
                all_lbls_list.append(batch["label"])

    all_probs_np = torch.cat(all_probs_list, 0).numpy()

    # Deduplicate
    seen_pids: set = set()
    dedup_idx: List[int] = []
    for i, pid in enumerate(all_pids_list):
        if pid not in seen_pids:
            seen_pids.add(pid)
            dedup_idx.append(i)

    out_path = Path(output_dir) / "test_predictions.tsv"
    with open(out_path, "w") as fh:
        fh.write("idx\tinput\tprediction\n")
        for i in dedup_idx:
            fh.write(
                f"{all_pids_list[i]}\t{all_syms_list[i]}\t"
                f"{json.dumps(all_probs_np[i].tolist())}\n"
            )
    print(f"[SWA] Saved {len(dedup_idx)} predictions → {out_path}")

    swa_f1 = None
    if all_lbls_list:
        all_lbls_np = torch.cat(all_lbls_list, 0).numpy()
        dedup_probs  = all_probs_np[dedup_idx]
        dedup_labels = all_lbls_np[dedup_idx]
        swa_f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
        print(f"[SWA] Self-computed SWA test F1 = {swa_f1:.4f}")

    return swa_f1


# ─── Global variable for steps_per_epoch (set in main) ────────────────────────

_steps_per_epoch = 12  # default, will be overridden in main()


# ─── Periodic Checkpoint Callback ─────────────────────────────────────────────

class PeriodicCheckpointCallback(pl.Callback):
    """
    Save model checkpoints every N epochs starting from start_epoch.
    These are used for quality-filtered SWA.
    Filename format: periodic-epoch=NNNN-val_f1=X.XXXX.ckpt
    """

    def __init__(self, every_n_epochs: int = 3, start_epoch: int = 15, dirpath: str = ""):
        self.every_n_epochs = every_n_epochs
        self.start_epoch    = start_epoch
        self.dirpath        = dirpath

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch < self.start_epoch:
            return
        if (epoch - self.start_epoch) % self.every_n_epochs != 0:
            return

        # Get current val_f1 from logged metrics
        val_f1 = trainer.callback_metrics.get("val_f1", None)
        if val_f1 is None:
            return
        val_f1_val = float(val_f1)

        # ALL ranks must call trainer.save_checkpoint because Lightning internally
        # calls self.strategy.barrier("Trainer.save_checkpoint") — a collective op.
        # If only rank 0 calls this, other ranks won't participate in the barrier,
        # causing an NCCL timeout/deadlock. Lightning's DDP strategy handles
        # rank-0-only file writing internally.
        os.makedirs(self.dirpath, exist_ok=True)
        fname = f"periodic-epoch={epoch:04d}-val_f1={val_f1_val:.4f}.ckpt"
        path  = os.path.join(self.dirpath, fname)
        trainer.save_checkpoint(path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-1-1-3 — STRING_GNN Partial FT + Muon + SGDR + SWA"
    )
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--lr-backbone",       type=float, default=5e-5)
    p.add_argument("--lr-head",           type=float, default=5e-4)
    p.add_argument("--muon-lr",           type=float, default=0.005)
    p.add_argument("--weight-decay",      type=float, default=3e-3)
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--label-smoothing",   type=float, default=0.07)
    p.add_argument("--head-dropout",      type=float, default=0.40)
    p.add_argument("--bilinear-rank",     type=int,   default=512)
    p.add_argument("--sgdr-t0",           type=int,   default=20,
                   help="SGDR T_0 in epochs")
    p.add_argument("--sgdr-t-mult",       type=float, default=2.0)
    p.add_argument("--swa-start-epoch",   type=int,   default=15)
    p.add_argument("--swa-every-n-epochs",type=int,   default=3)
    p.add_argument("--swa-threshold",     type=float, default=0.480)
    p.add_argument("--swa-top-k",         type=int,   default=20)
    p.add_argument("--swa-temperature",   type=float, default=3.0)
    p.add_argument("--warmup-steps",      type=int,   default=100)
    p.add_argument("--micro-batch-size",  type=int,   default=16)
    p.add_argument("--global-batch-size", type=int,   default=128)
    p.add_argument("--max-epochs",        type=int,   default=400)
    p.add_argument("--patience",          type=int,   default=150)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--val-check-interval",type=float, default=1.0)
    p.add_argument("--debug-max-step",    type=int,   default=None)
    p.add_argument("--fast-dev-run",      action="store_true", default=False)
    return p.parse_args()


def main():
    global _steps_per_epoch

    args = parse_args()
    pl.seed_everything(0)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute steps per epoch for SGDR schedule calibration
    train_size  = 1416
    accum       = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    steps_per_epoch_raw = max(1, train_size // (args.micro_batch_size * n_gpus))
    eff_steps_per_epoch = max(1, steps_per_epoch_raw // accum)
    _steps_per_epoch = eff_steps_per_epoch

    # Approximate max_steps_sgdr for SGDR LR reference (covers 4+ cycles of T_0=20 with T_mult=2)
    # T_0=20, T_1=40, T_2=80, T_3=160 → total ~300 epochs worth
    max_steps_sgdr = eff_steps_per_epoch * 300

    dm  = PerturbDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = GNNLitModule(
        lr_backbone      = args.lr_backbone,
        lr_head          = args.lr_head,
        muon_lr          = args.muon_lr,
        weight_decay     = args.weight_decay,
        focal_gamma      = args.focal_gamma,
        label_smoothing  = args.label_smoothing,
        head_dropout     = args.head_dropout,
        bilinear_rank    = args.bilinear_rank,
        sgdr_t0          = args.sgdr_t0,
        sgdr_t_mult      = args.sgdr_t_mult,
        swa_start_epoch  = args.swa_start_epoch,
        swa_every_n_epochs = args.swa_every_n_epochs,
        swa_threshold    = args.swa_threshold,
        swa_top_k        = args.swa_top_k,
        swa_temperature  = args.swa_temperature,
        warmup_steps     = args.warmup_steps,
        max_steps_sgdr   = max_steps_sgdr,
    )

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    # Periodic checkpoint callback for SWA
    periodic_cb = PeriodicCheckpointCallback(
        every_n_epochs=args.swa_every_n_epochs,
        start_epoch=args.swa_start_epoch,
        dirpath=str(ckpt_dir),
    )

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps_trainer: int = -1
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps_trainer = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2
        limit_test  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps_trainer,
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
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb, periodic_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    trainer.fit(lit, datamodule=dm)

    # Standard test with best checkpoint
    ckpt_path_arg = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt_path_arg)

    # ── Quality-filtered SWA post-hoc ensemble ────────────────────────────
    if (trainer.is_global_zero
            and args.debug_max_step is None
            and not args.fast_dev_run):
        print("\n" + "="*60)
        print("[Node2-1-1-3] Running quality-filtered SWA post-hoc ensemble...")
        print("="*60)
        swa_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Reload model in clean state for SWA
        lit_swa = GNNLitModule(
            lr_backbone     = args.lr_backbone,
            lr_head         = args.lr_head,
            muon_lr         = args.muon_lr,
            weight_decay    = args.weight_decay,
            focal_gamma     = args.focal_gamma,
            label_smoothing = args.label_smoothing,
            head_dropout    = args.head_dropout,
            bilinear_rank   = args.bilinear_rank,
            sgdr_t0         = args.sgdr_t0,
            sgdr_t_mult     = args.sgdr_t_mult,
            warmup_steps    = args.warmup_steps,
            max_steps_sgdr  = max_steps_sgdr,
        )
        lit_swa.setup(stage="test")

        swa_f1 = run_quality_filtered_swa(
            model_module    = lit_swa,
            datamodule      = dm,
            checkpoint_dir  = str(ckpt_dir),
            output_dir      = str(out_dir),
            top_k           = args.swa_top_k,
            threshold       = args.swa_threshold,
            temperature     = args.swa_temperature,
            device          = swa_device,
        )

        # Write summary
        summary_lines = [
            "Node 2-1-1-3 — STRING_GNN Partial FT + Muon + SGDR + Quality-Filtered SWA\n",
            f"SWA test F1: {swa_f1:.4f}\n" if swa_f1 is not None else "SWA: unavailable\n",
            "(Final score computed by EvaluateAgent via calc_metric.py)\n",
        ]
        (Path(__file__).parent / "test_score.txt").write_text("".join(summary_lines))
    else:
        if trainer.is_global_zero:
            (Path(__file__).parent / "test_score.txt").write_text(
                "Node 2-1-1-3 — STRING_GNN Partial FT + Muon + SGDR + SWA\n"
                "(Final score computed by EvaluateAgent via calc_metric.py)\n"
            )


if __name__ == "__main__":
    main()
