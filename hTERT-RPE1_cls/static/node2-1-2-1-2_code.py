"""
Node 2-1-2-1-2 — Frozen STRING_GNN + Rank-512 Bilinear Head + Muon Optimizer
              + SGDR Warm Restarts + Quality-Filtered Exponential SWA

Architecture:
  - STRING_GNN fully frozen (pre-computed once at setup, then discarded)
  - 6-layer residual MLP head (hidden=512, expand=4, rank=512, dropout=0.45)
  - Bilinear interaction: pert_repr [B, 3, 512] × out_gene_emb [6640, 512]
  - Class-weighted focal loss (gamma=2.0, weights=[down=2.0, neutral=0.5, up=4.0])
  - Label smoothing ε=0.07 to reduce overconfidence
  - MuonWithAuxAdam: Muon lr=0.005 for ResBlock 2D matrices, AdamW lr=5e-4 for others
  - SGDR cosine warm restarts (T_0=600 steps ≈ 27 epochs, T_mult=1)
  - Quality-filtered exponential SWA (top-25, threshold=0.497, temperature=2.5)
  - Periodic checkpoints every 3 epochs for SWA pool
  - patience=200 to allow full SGDR cycling

Key Design Rationale:
  - Parent (node2-1-2-1): Partial STRING_GNN FT + rank=512 + [2.0, 0.5, 4.0] = F1=0.5016
    - Early overfitting at epoch 22, no secondary improvement phase
    - Partial backbone FT provides no benefit on 1,416 samples
  - Sibling (node2-1-2-1-1): ESM2 backbone = F1=0.4359 (rejected direction)
  - Tree best: node2-1-1-1-2-1-1-1-1-1-1-1 (F1=0.5182) = Frozen backbone + Muon + SGDR + SWA
  - This node reverts to frozen STRING_GNN (proven stable) and adopts the
    high-performing Muon + SGDR + SWA strategy from the best tree lineage
  - MuonWithAuxAdam: proven +0.003-0.008 F1 gain over AdamW in multiple nodes
  - SGDR T_0=600 with T_mult=1: creates 6+ cycles with ascending staircase pattern
  - Quality-filtered SWA: +0.003-0.007 gain over best single checkpoint (proven in F1=0.5182)
  - Frozen backbone: avoids early overfitting, more stable training dynamics
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # required for deterministic einsum on CUDA >= 10.2

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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
from muon import MuonWithAuxAdam

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_DIM        = 256     # STRING_GNN hidden dim
HEAD_HIDDEN    = 512     # Residual MLP hidden dim
HEAD_EXPAND    = 4       # Expand factor in residual block
BILINEAR_RANK  = 512     # Bilinear interaction rank


# ─── Focal Loss with Class Weights + Label Smoothing ──────────────────────────

class FocalLossWithWeights(nn.Module):
    """
    Focal loss with optional per-class weights and label smoothing.

    class weights [down=2.0, neutral=0.5, up=4.0]:
    - Proven effective across multiple high-performing nodes
    - Addresses severe imbalance (88.9% neutral, 8.1% down, 3.0% up)

    Label smoothing ε=0.07:
    - Used in tree best (F1=0.5182) and its parent (F1=0.5180)
    - Reduces overconfidence without hurting argmax F1 significantly
    - Applied only to training targets; validation uses hard labels

    focal gamma=2.0: standard modulation proven across the tree.
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
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, C] (2D, already reshaped)
        targets: [N] long
        """
        # Cross-entropy with class weights and label smoothing
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        # Get probability of the true class for focal weighting
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction='none'))
        # Focal weight: down-weight easy examples
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


# ─── STRING_GNN Head ──────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Single residual MLP block: (LN → Linear(D→D*expand) → GELU → Dropout → Linear(D*expand→D)) + skip."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.45):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GNNBilinearHead(nn.Module):
    """
    Deep bilinear prediction head with rank-512 output interaction.

    Architecture:
      input [B, 256]
        → Linear(256→512) [proj_in]
        → 6 × ResidualBlock(512, expand=4, dropout=0.45)  [B, 512]
        → Linear(512→3*512=1536) [proj_out]
        → reshape [B, 3, 512]
        → einsum("bcr,gr->bcg", [B,3,512], out_gene_emb[6640,512])
        → logits [B, 3, 6640]

    dropout=0.45: proven in top tree nodes (F1=0.5180, 0.5182) to slow overfitting
    rank=512: +0.005 F1 gain from rank increase proven in node1-2-3
    """

    def __init__(
        self,
        in_dim:   int = GNN_DIM,         # 256
        hidden:   int = HEAD_HIDDEN,     # 512
        expand:   int = HEAD_EXPAND,     # 4
        n_blocks: int = 6,
        dropout:  float = 0.45,
        rank:     int = BILINEAR_RANK,   # 512
        n_genes:  int = N_GENES_OUT,     # 6640
        n_classes: int = N_CLASSES,       # 3
    ):
        super().__init__()
        self.rank     = rank
        self.n_classes = n_classes
        self.n_genes  = n_genes

        # Input projection from GNN_DIM (256) to HEAD_HIDDEN (512)
        self.proj_in = nn.Linear(in_dim, hidden)

        # Deep residual MLP blocks (Muon handles the 2D weight matrices inside)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, expand, dropout) for _ in range(n_blocks)
        ])

        # Output projection from HEAD_HIDDEN (512) to n_classes * rank (3*512=1536)
        self.proj_out = nn.Linear(hidden, n_classes * rank)

        # Learnable output gene embeddings [6640, 512]
        # rank=512 gives more expressive gene identity encoding
        self.out_gene_emb = nn.Parameter(torch.randn(n_genes, rank) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 256]
        returns: logits [B, 3, 6640]
        """
        h = self.proj_in(x)                             # [B, 512]
        for block in self.blocks:
            h = block(h)                                # [B, 512]

        proj = self.proj_out(h)                         # [B, 3*512]
        B = proj.shape[0]
        pert_proj = proj.view(B, self.n_classes, self.rank)  # [B, 3, 512]

        # Bilinear interaction: for each gene, compute class logits as dot product
        # of per-class perturbation vector with gene embedding
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, 6640]
        return logits


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    """Dataset with pert_ids, pre-computed embeddings, and labels."""

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        embeddings: torch.Tensor,           # [N, 256] pre-computed GNN embeddings
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long, class indices {0,1,2}
    ):
        self.pert_ids   = pert_ids
        self.symbols    = symbols
        self.embeddings = embeddings  # pre-computed, frozen
        self.labels     = labels

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":   self.pert_ids[idx],
            "symbol":    self.symbols[idx],
            "embedding": self.embeddings[idx],  # [256]
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    pert_ids   = [b["pert_id"]   for b in batch]
    symbols    = [b["symbol"]    for b in batch]
    embeddings = torch.stack([b["embedding"] for b in batch])
    out = {
        "pert_id":    pert_ids,
        "symbol":     symbols,
        "embedding":  embeddings,
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
        # Will be populated by the LightningModule after GNN pre-computation
        self._embeddings: Optional[Dict[str, torch.Tensor]] = None

    def set_embeddings(self, embeddings: Dict[str, torch.Tensor]):
        """
        Called by LightningModule after STRING_GNN pre-computation.
        embeddings: dict mapping pert_id -> 256-dim float tensor (CPU)
        """
        self._embeddings = embeddings

    def setup(self, stage: Optional[str] = None):
        assert self._embeddings is not None, "Call set_embeddings() before setup()"

        # OOV embedding: use zero vector (LightningModule has learnable oov_emb in model)
        # We store a sentinel tensor for OOV genes; actual replacement done in model forward
        # Here we store actual embeddings directly
        oov_vec = torch.zeros(GNN_DIM, dtype=torch.float32)  # placeholder

        def load_split(fname: str, has_label: bool) -> PerturbDataset:
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            pert_ids = df["pert_id"].tolist()
            symbols  = df["symbol"].tolist()

            # Look up pre-computed embeddings
            embs = []
            for pid in pert_ids:
                if pid in self._embeddings:
                    embs.append(self._embeddings[pid])
                else:
                    embs.append(oov_vec)  # OOV genes get zero vector
            embeddings = torch.stack(embs, dim=0)  # [N, 256]

            labels = None
            if has_label and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)  # {0,1,2}

            return PerturbDataset(pert_ids, symbols, embeddings, labels)

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  False)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size,
            shuffle=shuffle, collate_fn=collate_fn,
            num_workers=self.num_workers, pin_memory=True,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))

    pad = max_sz - local_p.shape[0]
    p = local_p.to(device)
    l = local_l.to(device)
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


# ─── Periodic Checkpoint Callback ─────────────────────────────────────────────

class PeriodicCheckpointCallback(pl.Callback):
    """
    Save checkpoints every `save_every_n_epochs` epochs (for SWA pool collection).
    The val_f1 is included in the filename for easy quality filtering.

    This implements the pattern from tree best nodes (F1=0.5182) where periodic
    checkpoints are saved to build a rich, diverse pool for quality-filtered SWA.
    """

    def __init__(self, dirpath: str, save_every_n_epochs: int = 3, start_epoch: int = 15):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.save_every_n_epochs = save_every_n_epochs
        self.start_epoch = start_epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch < self.start_epoch:
            return
        if (epoch - self.start_epoch) % self.save_every_n_epochs != 0:
            return

        # Get current val_f1
        val_f1 = trainer.callback_metrics.get("val_f1", None)
        if val_f1 is None:
            return

        val_f1_val = float(val_f1)
        # Create dir only on rank 0 to avoid race conditions
        if trainer.is_global_zero:
            self.dirpath.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.dirpath / f"periodic-epoch={epoch:04d}-val_f1={val_f1_val:.4f}.ckpt"

        # CRITICAL: call on ALL ranks — DDPStrategy.save_checkpoint internally
        # calls barrier() on all ranks.  Calling only on rank 0 causes a
        # collective-operation mismatch that corrupts NCCL buffers.
        trainer.save_checkpoint(str(ckpt_path))


# ─── LightningModule ──────────────────────────────────────────────────────────

class StringGNNFrozenMuonModel(pl.LightningModule):
    """
    Frozen STRING_GNN backbone + MuonWithAuxAdam + SGDR + quality-filtered SWA.

    Key improvements over parent (node2-1-2-1, F1=0.5016):
    1. Frozen backbone (pre-computed once): avoids early overfitting from partial FT
    2. MuonWithAuxAdam: proven +0.003-0.008 F1 gain over AdamW
    3. SGDR warm restarts (T_0=600): creates rich staircase improvement
    4. Quality-filtered SWA: +0.003-0.007 gain over best single checkpoint
    5. Dropout=0.45 + label_smoothing=0.07: stronger regularization from best nodes
    6. patience=200: allows full SGDR cycling

    Architecture follows the proven best lineage (node2-1-1-1-2-1-1-1-1-1-1-1, F1=0.5182).
    """

    def __init__(
        self,
        lr_head: float = 5e-4,
        lr_muon: float = 0.005,
        weight_decay: float = 2e-3,
        focal_gamma: float = 2.0,
        class_weight_down: float = 2.0,
        class_weight_neutral: float = 0.5,
        class_weight_up: float = 4.0,
        label_smoothing: float = 0.07,
        head_dropout: float = 0.45,
        sgdr_t0: int = 600,           # SGDR T_0 in steps (≈27 epochs with global_batch=128)
        sgdr_t_mult: int = 1,         # Constant cycle length
        eta_min_ratio: float = 1e-3,  # eta_min = lr * eta_min_ratio
        n_gnn_nodes: int = 18870,
        # SWA parameters
        swa_threshold: float = 0.497, # Minimum val_f1 for a checkpoint to qualify
        swa_top_k: int = 25,          # Take top-k checkpoints by val_f1
        swa_temperature: float = 2.5, # Temperature for exponential weighting
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
        # Initialize prediction head
        self.head = GNNBilinearHead(
            in_dim=GNN_DIM,
            hidden=HEAD_HIDDEN,
            expand=HEAD_EXPAND,
            n_blocks=6,
            dropout=self.hparams.head_dropout,
            rank=BILINEAR_RANK,
            n_genes=N_GENES_OUT,
            n_classes=N_CLASSES,
        )

        # Learnable OOV embedding for genes not in STRING vocabulary
        self.oov_emb = nn.Parameter(torch.randn(1, GNN_DIM) * 0.01)

        # Class weights tensor: [0=down, 1=neutral, 2=up]
        cw = torch.tensor([
            self.hparams.class_weight_down,
            self.hparams.class_weight_neutral,
            self.hparams.class_weight_up,
        ], dtype=torch.float32)

        self.focal_loss = FocalLossWithWeights(
            gamma=self.hparams.focal_gamma,
            class_weights=cw,
            label_smoothing=self.hparams.label_smoothing,
        ).to(self.device)

        # Cast all trainable parameters to float32 for optimizer stability
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, 256] pre-computed frozen STRING_GNN embeddings
        returns: logits [B, 3, 6640]
        """
        return self.head(embeddings.float())

    def _loss(self, logits, labels):
        # logits: [B, 3, 6640] -> [B*6640, 3];  labels: [B, 6640] -> [B*6640]
        logits_2d = logits.float().permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_1d = labels.reshape(-1)
        return self.focal_loss(logits_2d, labels_1d)

    def training_step(self, batch, batch_idx):
        logits = self(batch["embedding"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["embedding"])
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
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["embedding"])
        probs  = torch.softmax(logits.float(), dim=1)
        self._test_preds.append(probs.detach().cpu())
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
            all_pert, all_syms    = self._test_pert_ids, self._test_symbols

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id (DDP may pad with duplicates)
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
            self.print(f"[Node2-1-2-1-2] Saved {len(dedup_indices)} test predictions → {pred_path}")

            if self._test_labels:
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-1-2-1-2] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # MuonWithAuxAdam: proven superior to AdamW for this task
        # - Muon (lr=0.005) for 2D weight matrices in ResidualBlocks (hidden layers)
        # - AdamW (lr=5e-4) for all other parameters (proj_in, proj_out, embeddings, norms)

        # Identify 2D weight matrices in ResidualBlock hidden layers (suitable for Muon)
        muon_params = []
        adamw_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Muon for 2D hidden weight matrices in ResidualBlocks
            # (the Linear layers inside blocks, but NOT proj_in/proj_out/emb/norm)
            if (param.ndim >= 2 and
                    "blocks." in name and
                    ".net." in name and
                    ("weight" in name) and
                    "norm" not in name):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        self.print(f"[Node2-1-2-1-2] Muon params: {sum(p.numel() for p in muon_params):,}")
        self.print(f"[Node2-1-2-1-2] AdamW params: {sum(p.numel() for p in adamw_params):,}")

        param_groups = [
            # Muon group for hidden weight matrices in ResidualBlocks
            {
                "params": muon_params,
                "use_muon": True,
                "lr": hp.lr_muon,
                "weight_decay": hp.weight_decay,
                "momentum": 0.95,
            },
            # AdamW group for all other parameters
            {
                "params": adamw_params,
                "use_muon": False,
                "lr": hp.lr_head,
                "weight_decay": hp.weight_decay,
                "betas": (0.9, 0.95),
                "eps": 1e-10,
            },
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # SGDR cosine warm restarts
        # T_0=600 steps ≈ 27 epochs with global_batch=128, 1416 samples
        # T_mult=1: constant cycle length enables sustained staircase improvement
        # eta_min = lr * eta_min_ratio prevents hard LR=0 freeze
        eta_min_head = hp.lr_head * hp.eta_min_ratio
        eta_min_muon = hp.lr_muon * hp.eta_min_ratio

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.sgdr_t0,
            T_mult=hp.sgdr_t_mult,
            eta_min=eta_min_head,  # Note: CosineAnnealingWarmRestarts uses single eta_min
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and buffers (excludes frozen STRING_GNN backbone)."""
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items()
              if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving ckpt: {trained}/{total} trainable params "
            f"({100*trained/total:.2f}%) + {buffers} buffer values"
        )
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── STRING_GNN Pre-computation ───────────────────────────────────────────────

def precompute_string_gnn_embeddings(device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Pre-compute frozen STRING_GNN embeddings for all 18,870 nodes.
    Called once before training; the GNN model is then discarded from GPU.

    Returns: dict mapping Ensembl gene ID -> 256-dim float tensor (CPU)

    DDP: Each rank computes independently (deterministic GNN, same result on all ranks).
    The embedding dict is kept on CPU after computation to avoid GPU memory waste.
    """
    gnn_model_dir = Path(STRING_GNN_DIR)

    # Check for cached embeddings
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "string_gnn_emb_cache.pt"

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if cache_path.exists():
        if local_rank == 0:
            print(f"[Node2-1-2-1-2] Loading STRING_GNN embeddings from cache: {cache_path}")
        emb_dict = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        return emb_dict

    # Compute embeddings
    if local_rank == 0:
        print(f"[Node2-1-2-1-2] Pre-computing STRING_GNN embeddings on {device}...")

    gnn = AutoModel.from_pretrained(str(gnn_model_dir), trust_remote_code=True)
    gnn = gnn.to(device)
    gnn.eval()

    graph = torch.load(str(gnn_model_dir / "graph_data.pt"), map_location=device)
    edge_index = graph["edge_index"].to(device)
    edge_weight = graph.get("edge_weight")
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
    else:
        edge_weight = torch.ones(edge_index.shape[1], device=device)

    node_names = json.loads((gnn_model_dir / "node_names.json").read_text())

    with torch.no_grad():
        outputs = gnn(edge_index=edge_index, edge_weight=edge_weight)
        all_emb = outputs.last_hidden_state.float().cpu()  # [18870, 256] on CPU

    # Build dict: Ensembl ID -> embedding vector
    emb_dict: Dict[str, torch.Tensor] = {}
    for i, name in enumerate(node_names):
        emb_dict[name] = all_emb[i]

    # Cache to disk (rank 0 only)
    if local_rank == 0:
        torch.save(emb_dict, str(cache_path))
        print(f"[Node2-1-2-1-2] Cached STRING_GNN embeddings to {cache_path}")

    # Synchronize across ranks
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Free GNN from GPU
    del gnn, graph, all_emb
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if local_rank == 0:
        print(f"[Node2-1-2-1-2] STRING_GNN pre-computation complete. {len(emb_dict)} embeddings.")

    return emb_dict


# ─── Quality-Filtered Exponential SWA ─────────────────────────────────────────

def run_quality_filtered_swa(
    lit_module: StringGNNFrozenMuonModel,
    checkpoint_dir: Path,
    threshold: float,
    top_k: int,
    temperature: float,
    device: torch.device,
) -> Optional[float]:
    """
    Implements quality-filtered exponential SWA from the best tree lineage (F1=0.5182).

    Algorithm:
    1. Scan checkpoint_dir for periodic checkpoints with val_f1 in filename
    2. Filter by threshold (val_f1 >= threshold)
    3. Take top-k by val_f1
    4. Compute exponential weights: w_i = exp(temperature * (val_f1_i - max_val_f1))
    5. Normalize weights to sum=1
    6. Weighted average of state dicts

    Returns: average val_f1 of included checkpoints, or None if no checkpoints qualify.
    """
    import re

    # Find periodic checkpoint files
    periodic_ckpts = []
    for ckpt_file in checkpoint_dir.glob("periodic-epoch=*.ckpt"):
        # Parse val_f1 from filename: periodic-epoch=0027-val_f1=0.5023.ckpt
        match = re.search(r"val_f1=(\d+\.\d+)", ckpt_file.name)
        if match:
            val_f1_val = float(match.group(1))
            periodic_ckpts.append((val_f1_val, ckpt_file))

    if not periodic_ckpts:
        print("[Node2-1-2-1-2] No periodic checkpoints found for SWA.")
        return None

    # Filter by threshold
    qualifying = [(f1, p) for f1, p in periodic_ckpts if f1 >= threshold]
    print(f"[Node2-1-2-1-2] SWA pool: {len(qualifying)}/{len(periodic_ckpts)} checkpoints "
          f"with val_f1 >= {threshold}")

    if not qualifying:
        print(f"[Node2-1-2-1-2] No checkpoints qualify for SWA (threshold={threshold}).")
        return None

    # Take top-k by val_f1
    qualifying.sort(key=lambda x: x[0], reverse=True)
    selected = qualifying[:top_k]
    print(f"[Node2-1-2-1-2] Selected top-{len(selected)} checkpoints for SWA: "
          f"val_f1 range [{selected[-1][0]:.4f}, {selected[0][0]:.4f}]")

    # Compute exponential weights: w_i = exp(temperature * (val_f1_i - max_val_f1))
    f1_scores = np.array([f1 for f1, _ in selected])
    max_f1 = f1_scores.max()
    weights = np.exp(temperature * (f1_scores - max_f1))
    weights = weights / weights.sum()

    print(f"[Node2-1-2-1-2] SWA weights: top={weights[0]:.4f}, bottom={weights[-1]:.4f}")

    # Load state dicts and compute weighted average
    avg_state = None
    for (f1_val, ckpt_path), w in zip(selected, weights):
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            # Handle both direct state_dict and Lightning checkpoint format
            if "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            else:
                sd = ckpt

            if avg_state is None:
                avg_state = {k: v.float() * w for k, v in sd.items()}
            else:
                for k in avg_state:
                    if k in sd:
                        avg_state[k] = avg_state[k] + sd[k].float() * w
        except Exception as e:
            print(f"[Node2-1-2-1-2] Warning: Failed to load checkpoint {ckpt_path}: {e}")
            continue

    if avg_state is None:
        print("[Node2-1-2-1-2] SWA failed: could not load any checkpoints.")
        return None

    # Load SWA state into the model
    try:
        lit_module.load_state_dict(avg_state)
        print(f"[Node2-1-2-1-2] Successfully loaded SWA weights from {len(selected)} checkpoints.")
        return float(np.mean(f1_scores))
    except Exception as e:
        print(f"[Node2-1-2-1-2] Warning: Failed to load SWA state dict: {e}")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-1-2-1-2 — Frozen STRING_GNN + Muon + SGDR + Quality-Filtered SWA"
    )
    p.add_argument("--data-dir",             type=str,   default="data")
    p.add_argument("--lr-head",              type=float, default=5e-4,
                   help="AdamW learning rate for head parameters (non-Muon)")
    p.add_argument("--lr-muon",              type=float, default=0.005,
                   help="Muon learning rate for 2D ResidualBlock matrices")
    p.add_argument("--weight-decay",         type=float, default=2e-3,
                   help="Weight decay for both optimizer groups")
    p.add_argument("--focal-gamma",          type=float, default=2.0)
    p.add_argument("--class-weight-down",    type=float, default=2.0,
                   help="Class weight for down-regulated class (8.1% freq)")
    p.add_argument("--class-weight-neutral", type=float, default=0.5,
                   help="Class weight for neutral class (88.9% freq)")
    p.add_argument("--class-weight-up",      type=float, default=4.0,
                   help="Class weight for up-regulated class (3.0% freq)")
    p.add_argument("--label-smoothing",      type=float, default=0.07,
                   help="Label smoothing (ε=0.07, proven in F1=0.5182 node)")
    p.add_argument("--head-dropout",         type=float, default=0.45,
                   help="Dropout in ResidualBlocks (0.45, proven in F1=0.5182 node)")
    p.add_argument("--sgdr-t0",              type=int,   default=600,
                   help="SGDR T_0 in steps (≈27 epochs with global_batch=128)")
    p.add_argument("--sgdr-t-mult",          type=int,   default=1,
                   help="SGDR T_mult (1=constant cycles for sustained staircase)")
    p.add_argument("--eta-min-ratio",        type=float, default=1e-3,
                   help="eta_min = lr * eta_min_ratio (prevents hard LR=0 freeze)")
    p.add_argument("--micro-batch-size",     type=int,   default=16,
                   help="Micro batch size per GPU")
    p.add_argument("--global-batch-size",    type=int,   default=128,
                   help="Global batch size (must be multiple of micro_batch_size * 8)")
    p.add_argument("--max-epochs",           type=int,   default=350,
                   help="Max epochs to allow full SGDR cycling")
    p.add_argument("--patience",             type=int,   default=200,
                   help="Early stopping patience (200 allows ~7+ SGDR cycles)")
    p.add_argument("--swa-threshold",        type=float, default=0.497,
                   help="Minimum val_f1 for SWA checkpoint qualification")
    p.add_argument("--swa-top-k",            type=int,   default=25,
                   help="Number of top checkpoints to include in SWA")
    p.add_argument("--swa-temperature",      type=float, default=2.5,
                   help="Temperature for exponential SWA weighting")
    p.add_argument("--periodic-save-every",  type=int,   default=3,
                   help="Save periodic checkpoint every N epochs (for SWA pool)")
    p.add_argument("--periodic-start-epoch", type=int,   default=15,
                   help="Start saving periodic checkpoints from this epoch")
    p.add_argument("--num-workers",          type=int,   default=4)
    p.add_argument("--val-check-interval",   type=float, default=1.0)
    p.add_argument("--debug-max-step",       type=int,   default=None)
    p.add_argument("--fast-dev-run",         action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # ── Step 1: Pre-compute frozen STRING_GNN embeddings ──────────────────────
    # Use the local device for pre-computation
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    emb_dict = precompute_string_gnn_embeddings(device)

    # ── Step 2: Set up DataModule with pre-computed embeddings ────────────────
    dm = PerturbDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    dm.set_embeddings(emb_dict)

    # Free embedding dict memory after setting in DataModule
    # (DataModule will hold it until setup() materializes tensors)

    # ── Step 3: Set up LightningModule ────────────────────────────────────────
    lit = StringGNNFrozenMuonModel(
        lr_head             = args.lr_head,
        lr_muon             = args.lr_muon,
        weight_decay        = args.weight_decay,
        focal_gamma         = args.focal_gamma,
        class_weight_down   = args.class_weight_down,
        class_weight_neutral= args.class_weight_neutral,
        class_weight_up     = args.class_weight_up,
        label_smoothing     = args.label_smoothing,
        head_dropout        = args.head_dropout,
        sgdr_t0             = args.sgdr_t0,
        sgdr_t_mult         = args.sgdr_t_mult,
        eta_min_ratio       = args.eta_min_ratio,
        n_gnn_nodes         = 18870,
        swa_threshold       = args.swa_threshold,
        swa_top_k           = args.swa_top_k,
        swa_temperature     = args.swa_temperature,
    )

    # ── Step 4: Set up callbacks and trainer ─────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    periodic_ckpt_cb = PeriodicCheckpointCallback(
        dirpath=str(out_dir / "checkpoints"),
        save_every_n_epochs=args.periodic_save_every,
        start_epoch=args.periodic_start_epoch,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

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
        callbacks=[ckpt_cb, periodic_ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    trainer.fit(lit, datamodule=dm)

    # ── Step 6: Quality-Filtered SWA ─────────────────────────────────────────
    # DDP-safe SWA: rank 0 computes averaged weights and saves them to a shared
    # file; all ranks barrier, then load the saved weights together before
    # calling trainer.test() collectively.
    if not fast_dev_run and args.debug_max_step is None:
        swa_weights_path = out_dir / "swa_weights.pt"
        swa_performed = False

        # Step 6a: rank 0 computes and saves SWA weights
        if trainer.is_global_zero:
            print("[Node2-1-2-1-2] Running quality-filtered exponential SWA...")
            ckpt_dir = out_dir / "checkpoints"
            swa_avg_f1 = run_quality_filtered_swa(
                lit_module     = lit,
                checkpoint_dir = ckpt_dir,
                threshold      = args.swa_threshold,
                top_k          = args.swa_top_k,
                temperature    = args.swa_temperature,
                device         = device,
            )
            if swa_avg_f1 is not None:
                # Save SWA-averaged state so all ranks can load it
                torch.save(lit.state_dict(), str(swa_weights_path))
                print(f"[Node2-1-2-1-2] SWA complete. Average qualifying val_f1 = {swa_avg_f1:.4f}")
                swa_performed = True
            else:
                print("[Node2-1-2-1-2] SWA skipped. No qualifying checkpoints found.")

        # Step 6b: broadcast SWA flag and barrier so all ranks are consistent
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([int(swa_performed)], dtype=torch.long, device=device)
            dist.broadcast(flag, src=0)
            swa_performed = bool(flag.item())
            dist.barrier()

        # Step 6c: all ranks load SWA weights or best checkpoint, then test together
        if swa_performed:
            swa_state = torch.load(str(swa_weights_path), map_location="cpu",
                                   weights_only=False)
            lit.load_state_dict(swa_state)
            trainer.test(lit, datamodule=dm)
        else:
            trainer.test(lit, datamodule=dm, ckpt_path="best")
    else:
        # debug / fast_dev_run: use current model weights (no checkpoint loading)
        trainer.test(lit, datamodule=dm, ckpt_path=None)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            "Node 2-1-2-1-2 — Frozen STRING_GNN + Muon + SGDR + Quality-Filtered SWA\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
