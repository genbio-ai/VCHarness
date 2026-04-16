"""
Node 2-1-3-2-1 — Partial STRING_GNN Fine-Tuning (mps.6, mps.7, post_mp)
             + rank=512 Deep Residual Bilinear MLP Head
             + Muon Optimizer for ResBlock 2D Weight Matrices
             + Cosine Annealing with Warm Restarts (T_0=600 steps, T_mult=1)
             + Stronger Regularization (dropout=0.25, weight_decay=1.5e-3)
             + Stronger Class-Weighted Focal Loss [2.0, 0.5, 4.0]

Architecture (same as parent node2-1-3-2, with optimizer and schedule changes):
  - STRING_GNN backbone: partially fine-tuned (mps.6, mps.7, post_mp)
    while emb.weight and mps.0-5 remain frozen (pre-computed as partial_emb buffer).
  - Prediction head: rank=512 bilinear
    - 6-layer residual MLP (hidden=512, expand=4, dropout=0.25)
    - bilinear: pert_proj [B,3,512] x out_gene_emb [6640,512] -> [B,3,6640]
  - Loss: focal (gamma=2.0) + class weights [down=2.0, neutral=0.5, up=4.0]
  - Optimizer: MuonWithAuxAdam
    * Muon group: ResBlock 2D weight matrices (proj_in, proj_out, resblock linear weights)
      Muon lr=0.005, momentum=0.95
    * AdamW group: backbone params, embeddings, biases, 1D params, out_gene_emb, oov_emb
      AdamW lr=5e-4 (head), lr=5e-5 (backbone), weight_decay=1.5e-3
  - LR Schedule: Cosine annealing with warm restarts
    * T_0=600 steps (~50 epochs per cycle at 12 steps/epoch)
    * T_mult=1 (fixed cycle length)
    * Enables staircase-like improvement across multiple cycles
  - max_epochs=500, patience=80 (to capture late-cycle improvements)
  - Gradient clipping (max_norm=1.0) for bf16 numerical stability

Key Design Rationale:
  - Parent node2-1-3-2 (F1=0.5017) used standard AdamW with cosine annealing (total_steps=1200).
    The feedback identified that the best epoch was consistently ~32, but no secondary improvement
    phase materialized because the flat-LR period caused overfitting rather than exploration.
  - Tree-best node1-2-2-2-1 (F1=0.5099) showed that Muon optimizer for ResBlock 2D matrices
    combined with cosine warm restarts (T_0=600 steps) yielded staircase improvements across
    6 restart cycles, achieving val F1 peaks of 0.488→0.496→0.501→0.502→0.503→0.510 progressively.
    This is the only approach in the tree to achieve F1>0.505 without exceeding node2-1-3's 0.5047.
  - node1-2-2-2 (F1=0.5060) confirmed that Muon+focal+class_weights[2,0.5,4] is a winning combo
    even with frozen backbone, and node1-2-2-2-1 showed adding partial backbone fine-tuning
    (mps.7+post_mp) further pushed to 0.5099.
  - This node combines the partial STRING_GNN fine-tuning from the parent lineage (node2-1-3-2,
    mps.6+mps.7+post_mp, proven to give ~+0.003 over frozen backbone) with the Muon+warm-restarts
    strategy from the best tree node (node1-2-2-2-1). This combination has not been tried before.
  - Dropout increased to 0.25 (from 0.2 in parent) to address the identified over-parameterization
    problem (17M params / 1416 samples = 12,100 params/sample) and reduce post-peak overfitting.
  - Weight decay 1e-3 → 1.5e-3 for additional regularization.
  - No label smoothing (consistently hurts per-gene macro F1 in this task, confirmed across 5+ nodes).
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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_DIM        = 256    # STRING_GNN hidden/output dim
HEAD_HIDDEN    = 512    # Residual MLP hidden dim
HEAD_EXPAND    = 4      # Expand factor in residual block
BILINEAR_RANK  = 512    # Bilinear interaction rank
N_GNN_NODES    = 18870  # Total STRING_GNN nodes


# ─── Focal Loss with Class Weights ────────────────────────────────────────────

class FocalLossWithWeights(nn.Module):
    """
    Focal loss with per-class weights.

    Class weights [down=2.0, neutral=0.5, up=4.0] from node1-2-3's proven recipe (F1=0.4969).
    Combined with node2-1-3's partial backbone: achieved F1=0.5047 (tree best at that time).
    """
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, C] (2D, already reshaped)
        targets: [N] long
        """
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            reduction='none'
        )
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction='none'))
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


# ─── Model Components ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Single residual MLP block: (LN → Linear(D→D*expand) → GELU → Dropout → Linear(D*expand→D)) + skip."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.25):
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
    Deep bilinear prediction head with rank=512.

    Architecture (same as node2-1-3-2, parent):
      input [B, 256]
        → Linear(256→512) [proj_in]
        → 6 × ResidualBlock(512, expand=4, dropout=0.25)
        → Linear(512→3*512) [proj_out]
        → reshape [B, 3, 512]
        → einsum("bcr,gr->bcg", [B,3,512], out_gene_emb[6640,512])
        → logits [B, 3, 6640]

    Key change from parent: dropout=0.25 (up from 0.20) to address over-parameterization.
    """

    def __init__(
        self,
        in_dim:    int = GNN_DIM,       # 256 (STRING_GNN output dim)
        hidden:    int = HEAD_HIDDEN,   # 512
        expand:    int = HEAD_EXPAND,   # 4
        n_blocks:  int = 6,
        dropout:   float = 0.25,
        rank:      int = BILINEAR_RANK, # 512
        n_genes:   int = N_GENES_OUT,   # 6640
        n_classes: int = N_CLASSES,     # 3
    ):
        super().__init__()
        self.rank      = rank
        self.n_classes = n_classes
        self.n_genes   = n_genes

        # Input projection: GNN_DIM (256) -> HEAD_HIDDEN (512)
        self.proj_in = nn.Linear(in_dim, hidden)

        # Deep residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, expand, dropout) for _ in range(n_blocks)
        ])

        # Output projection: HEAD_HIDDEN (512) -> n_classes * rank (3*512=1536)
        self.proj_out = nn.Linear(hidden, n_classes * rank)

        # Learnable output gene embeddings [6640, 512]
        self.out_gene_emb = nn.Parameter(torch.randn(n_genes, rank) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 256] — STRING_GNN embedding for each sample's perturbed gene
        returns: logits [B, 3, 6640]
        """
        h = self.proj_in(x)            # [B, 512]
        for block in self.blocks:
            h = block(h)               # [B, 512]

        proj = self.proj_out(h)        # [B, 3*512]
        B = proj.shape[0]
        pert_proj = proj.view(B, self.n_classes, self.rank)  # [B, 3, 512]

        # Bilinear: logits[b,c,g] = sum_r pert_proj[b,c,r] * out_gene_emb[g,r]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, 6640]
        return logits


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    """Simple dataset storing pert_ids, symbols, STRING_GNN indices, and labels."""

    def __init__(
        self,
        pert_ids:    List[str],
        symbols:     List[str],
        gnn_indices: List[int],
        labels:      Optional[torch.Tensor] = None,  # [N, 6640] long, class indices {0,1,2}
    ):
        self.pert_ids    = pert_ids
        self.symbols     = symbols
        self.gnn_indices = gnn_indices
        self.labels      = labels

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":   self.pert_ids[idx],
            "symbol":    self.symbols[idx],
            "gnn_index": self.gnn_indices[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    pert_ids    = [b["pert_id"]   for b in batch]
    symbols     = [b["symbol"]    for b in batch]
    gnn_indices = torch.tensor([b["gnn_index"] for b in batch], dtype=torch.long)
    out = {
        "pert_id":   pert_ids,
        "symbol":    symbols,
        "gnn_index": gnn_indices,
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir:         str = "data",
        micro_batch_size: int = 16,
        num_workers:      int = 4,
    ):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage: Optional[str] = None):
        # Load STRING_GNN node name mapping: pert_id (Ensembl gene ID) -> node index
        node_names_path = Path(STRING_GNN_DIR) / "node_names.json"
        node_names = json.loads(node_names_path.read_text())
        self.node_name_to_idx = {name: i for i, name in enumerate(node_names)}
        self.n_gnn_nodes = len(node_names)  # 18870

        def load_split(fname: str, has_label: bool) -> PerturbDataset:
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            pert_ids = df["pert_id"].tolist()
            symbols  = df["symbol"].tolist()

            # Map pert_id (Ensembl gene ID) -> STRING_GNN node index
            # OOV sentinel = n_gnn_nodes (learnable fallback embedding)
            gnn_indices = [
                self.node_name_to_idx.get(pid, self.n_gnn_nodes)
                for pid in pert_ids
            ]

            labels = None
            if has_label and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)  # {0,1,2}

            return PerturbDataset(pert_ids, symbols, gnn_indices, labels)

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


# ─── Full Model ───────────────────────────────────────────────────────────────

class PartialStringGNNPerturbModel(nn.Module):
    """
    Partially fine-tuned STRING_GNN + rank=512 deep bilinear MLP head.

    Training strategy (partial fine-tuning):
    - Frozen: STRING_GNN emb.weight + mps.0-5 (pre-computed as partial_emb buffer)
    - Trainable: mps.6, mps.7, post_mp (last 2 GNN layers + output projection)

    Same as node2-1-3-2 architecture.
    Key changes: dropout=0.25 in ResidualBlocks.
    """

    def __init__(
        self,
        n_gnn_nodes:  int   = N_GNN_NODES,  # 18870
        head_dropout: float = 0.25,
        oov_init_std: float = 0.01,
    ):
        super().__init__()
        self.n_gnn_nodes = n_gnn_nodes

        # OOV learnable embedding for genes not in STRING vocabulary
        self.oov_emb = nn.Parameter(torch.randn(1, GNN_DIM) * oov_init_std)

        # Deep bilinear prediction head with rank=512
        self.head = GNNBilinearHead(
            in_dim=GNN_DIM,
            hidden=HEAD_HIDDEN,
            expand=HEAD_EXPAND,
            n_blocks=6,
            dropout=head_dropout,
            rank=BILINEAR_RANK,
            n_genes=N_GENES_OUT,
            n_classes=N_CLASSES,
        )

        # These will be set in setup_model():
        # - self.gnn_last_layers: trainable mps.6, mps.7, post_mp
        # - self.partial_emb: buffer with partial GNN states after mps.5
        # - self.edge_index, self.edge_weight: graph topology buffers
        self._model_initialized = False

    def setup_model(self, device: torch.device):
        """
        Load STRING_GNN, freeze early layers, store partial embeddings as buffer,
        and register the trainable last 2 GNN layers + post_mp for gradient updates.

        Called once during LightningModule.setup().
        """
        if self._model_initialized:
            return

        gnn_model_dir = Path(STRING_GNN_DIR)
        gnn_full = AutoModel.from_pretrained(str(gnn_model_dir), trust_remote_code=True)
        gnn_full.eval()
        gnn_full = gnn_full.to(device)

        # Load graph topology
        graph = torch.load(str(gnn_model_dir / "graph_data.pt"), map_location=device)
        edge_index  = graph["edge_index"].to(device)
        edge_weight = graph.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.float().to(device)

        # Register graph topology as buffers (move with model to correct device)
        self.register_buffer("edge_index",  edge_index.cpu())
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight.cpu())
        else:
            self.register_buffer("edge_weight", None)

        # Compute partial GNN forward through frozen layers (mps.0-5, not post_mp)
        # Following the STRING_GNN's forward: emb -> mps blocks (each with residual) -> post_mp
        # GNNLayer.forward: norm -> conv -> act -> dropout (returns transformed, NOT residual)
        # StringGNNModel.forward: x = mp(x, ...) + x  (adds residual outside layer)
        with torch.no_grad():
            # Initial node states from embedding table
            x = gnn_full.emb.weight.clone()  # [18870, 256]

            # Run frozen early layers mps.0 through mps.5
            # GNNLayer returns only the transformed value; residual add done here
            for i in range(6):  # mps[0..5] frozen
                layer = gnn_full.mps[i]
                x = layer(x, edge_index, edge_weight=edge_weight) + x  # residual add

        # Store partial embeddings [18870, 256] as frozen buffer
        partial_emb = x.detach().float().cpu()
        self.register_buffer("partial_emb", partial_emb)

        # Extract the last 2 trainable GNN layers + post_mp
        # These will be fine-tuned during training
        self.gnn_mps_6   = gnn_full.mps[6]
        self.gnn_mps_7   = gnn_full.mps[7]
        self.gnn_post_mp = gnn_full.post_mp

        # Ensure trainable layers are in float32 for optimizer stability
        self.gnn_mps_6.float()
        self.gnn_mps_7.float()
        self.gnn_post_mp.float()

        # Free the full GNN model
        del gnn_full, graph
        if device.type == "cuda":
            torch.cuda.empty_cache()

        self._model_initialized = True

    def _get_adapted_embeddings(self) -> torch.Tensor:
        """
        Run the trainable last 2 GNN layers on the frozen partial embeddings.
        Returns adapted embeddings [18870, 256] in float32.
        """
        dev = self.partial_emb.device
        # Get edge_index and edge_weight on the right device
        ei = self.edge_index.to(dev)
        ew = self.edge_weight.to(dev).float() if self.edge_weight is not None else None

        # Always use float32 for GNN layer computation to avoid dtype conflicts
        x = self.partial_emb.float()  # [18870, 256], always float32

        # mps.6 (trainable) + residual
        x = self.gnn_mps_6(x, ei, edge_weight=ew) + x

        # mps.7 (trainable) + residual
        x = self.gnn_mps_7(x, ei, edge_weight=ew) + x

        # post_mp (trainable)
        x = self.gnn_post_mp(x)  # [18870, 256], float32
        return x

    def forward(self, gnn_indices: torch.Tensor) -> torch.Tensor:
        """
        gnn_indices: [B] long — STRING_GNN node index for each perturbed gene
                     OOV genes have index = n_gnn_nodes (handled via oov_emb)

        returns: logits [B, 3, 6640]
        """
        # Get adapted embeddings through trainable last 2 GNN layers
        adapted_emb = self._get_adapted_embeddings()  # [18870, 256] float32

        B = gnn_indices.shape[0]
        oov_mask = (gnn_indices >= self.n_gnn_nodes)  # [B] bool
        safe_idx = gnn_indices.clone()
        safe_idx[oov_mask] = 0  # temporary valid index

        emb = adapted_emb[safe_idx].float()  # [B, 256] float32

        # Replace OOV embeddings with learnable fallback
        if oov_mask.any():
            emb = emb.clone()  # ensure we don't modify adapted_emb in-place
            emb[oov_mask] = self.oov_emb.float().expand(oov_mask.sum(), -1)

        # Pass through the head (float32 input, float32 head weights)
        logits = self.head(emb)  # [B, 3, 6640]
        return logits


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


# ─── LightningModule ──────────────────────────────────────────────────────────

class PartialStringGNNLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_backbone:          float = 5e-5,  # conservative for partial GNN fine-tuning
        lr_head_adamw:        float = 5e-4,  # AdamW LR for non-2D head params (embeddings etc)
        lr_muon:              float = 0.005, # Muon LR for 2D ResBlock weight matrices
        weight_decay:         float = 1.5e-3, # increased from 1e-3 to address over-parameterization
        focal_gamma:          float = 2.0,
        class_weight_down:    float = 2.0,   # node1-2-3 + node2-1-3 proven recipe
        class_weight_neutral: float = 0.5,
        class_weight_up:      float = 4.0,
        head_dropout:         float = 0.25,  # increased from 0.20 to reduce overfitting
        warmup_steps:         int   = 50,    # short warmup before first restart cycle
        restart_period:       int   = 600,   # T_0 for cosine warm restarts (steps)
                                             # ~50 epochs at 12 steps/epoch
        n_gnn_nodes:          int   = N_GNN_NODES,
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
        self.model = PartialStringGNNPerturbModel(
            n_gnn_nodes=self.hparams.n_gnn_nodes,
            head_dropout=self.hparams.head_dropout,
        )
        self.model.setup_model(self.device)

        # Class weights tensor: [0=down, 1=neutral, 2=up]
        cw = torch.tensor([
            self.hparams.class_weight_down,
            self.hparams.class_weight_neutral,
            self.hparams.class_weight_up,
        ], dtype=torch.float32)
        self.focal_loss = FocalLossWithWeights(
            gamma=self.hparams.focal_gamma,
            class_weights=cw,
        )

        # Cast all trainable parameters to float32 for optimizer stability
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

    def forward(self, gnn_indices):
        return self.model(gnn_indices)

    def _loss(self, logits, labels):
        # logits: [B, 3, 6640] -> [B*6640, 3];  labels: [B, 6640] -> [B*6640]
        logits_2d = logits.float().permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_1d = labels.reshape(-1)
        return self.focal_loss(logits_2d, labels_1d)

    def training_step(self, batch, batch_idx):
        logits = self(batch["gnn_index"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["gnn_index"])
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
        logits = self(batch["gnn_index"])
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
            self.print(f"[Node2-1-3-2-1] Saved {len(dedup_indices)} test predictions -> {pred_path}")

            if self._test_labels:
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-1-3-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # ─── Parameter Grouping for MuonWithAuxAdam ───────────────────────────
        #
        # Muon group: 2D weight matrices from ResBlock hidden layers
        #   - ResidualBlock linear weights: blocks.*.net.1.weight, blocks.*.net.4.weight
        #   - proj_in.weight (256->512)
        #   - proj_out.weight (512->1536)
        #   These are hidden weight matrices appropriate for Muon orthogonalization.
        #
        # AdamW group (use_muon=False): everything else
        #   - Backbone params (mps.6, mps.7, post_mp, oov_emb)
        #   - Biases, LayerNorm params (1D params)
        #   - out_gene_emb (embedding table)
        #   - proj_in.bias, proj_out.bias

        # Identify all 2D weight matrices in the head's hidden layers (for Muon)
        # Exclude out_gene_emb (embedding), exclude bias terms
        head_muon_params   = []
        head_adamw_params  = []
        backbone_params    = []

        # Names of backbone trainable components
        backbone_keywords = ["gnn_mps_6", "gnn_mps_7", "gnn_post_mp", "oov_emb"]

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            is_backbone = any(kw in name for kw in backbone_keywords)

            if is_backbone:
                backbone_params.append(param)
            else:
                # head parameter
                # Use Muon only for proper 2D weight matrices in hidden layers
                # (not out_gene_emb which is an embedding, not biases)
                is_2d_weight = (
                    param.ndim == 2
                    and "out_gene_emb" not in name
                    and "bias" not in name
                    and "norm" not in name
                )
                if is_2d_weight:
                    head_muon_params.append(param)
                else:
                    head_adamw_params.append(param)

        # Import Muon for distributed training
        try:
            from muon import MuonWithAuxAdam
            use_muon = len(head_muon_params) > 0
        except ImportError:
            # Fallback to AdamW if Muon is not available
            use_muon = False
            head_adamw_params = head_adamw_params + head_muon_params
            head_muon_params = []
            self.print("[Warning] Muon not available, falling back to AdamW for all params")

        if use_muon:
            # MuonWithAuxAdam: single optimizer handling both Muon and AdamW groups
            # Uses aux_lr (AdamW internally) for all groups; Muon groups get
            # their own Muon-style update on top.
            param_groups = [
                # Muon group for ResBlock 2D weight matrices
                dict(
                    params=head_muon_params,
                    use_muon=True,
                    lr=hp.lr_muon,          # 0.005 — higher than AdamW, proven in node1-2-2-2-1
                    momentum=0.95,
                    weight_decay=0.0,       # Muon with no explicit wd (orthogonalization handles this)
                ),
                # AdamW group for head non-matrix params (biases, LayerNorm, out_gene_emb)
                dict(
                    params=head_adamw_params,
                    use_muon=False,
                    lr=hp.lr_head_adamw,    # 5e-4 — standard head AdamW lr
                    betas=(0.9, 0.999),
                    weight_decay=hp.weight_decay,
                ),
                # AdamW group for backbone params (conservative lr)
                dict(
                    params=backbone_params,
                    use_muon=False,
                    lr=hp.lr_backbone,      # 5e-5 — conservative partial fine-tuning lr
                    betas=(0.9, 0.999),
                    weight_decay=hp.weight_decay,
                ),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            # Fallback: standard AdamW with two LR groups
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params,   "lr": hp.lr_backbone,
                     "weight_decay": hp.weight_decay},
                    {"params": head_adamw_params,  "lr": hp.lr_head_adamw,
                     "weight_decay": hp.weight_decay},
                ],
                betas=(0.9, 0.999),
            )

        # ─── Cosine Annealing with Warm Restarts ──────────────────────────────
        # CosineAnnealingWarmRestarts: T_0=600 steps, T_mult=1 (fixed cycle length)
        # At 12 steps/epoch, T_0=600 steps ≈ 50 epochs per cycle.
        # With max_epochs=500, we get ~8 restart cycles.
        # node1-2-2-2-1 showed staircase improvement across 6 cycles with T_0=600.
        #
        # Linear warmup for first `warmup_steps` steps (50 steps = ~4 epochs),
        # then switch to cosine warm restarts.

        warmup = hp.warmup_steps
        T0 = hp.restart_period

        def warmup_factor(step: int) -> float:
            if step < warmup:
                return float(step) / float(max(1, warmup))
            return 1.0  # After warmup, CosineAnnealingWarmRestarts handles the schedule

        # We use a combined approach: LambdaLR for warmup, then CosineAnnealingWarmRestarts
        # Implementation: we use a single LambdaLR that includes cosine restart logic
        def lr_lambda_with_warm_restarts(step: int) -> float:
            if step < warmup:
                # Linear warmup
                return float(step) / float(max(1, warmup))
            # Cosine annealing with warm restarts
            step_after_warmup = step - warmup
            # Current position within the cycle
            cycle_step = step_after_warmup % T0
            # Cosine decay within the cycle: 1 -> 0
            cosine_val = 0.5 * (1.0 + np.cos(np.pi * cycle_step / T0))
            return max(1e-7, cosine_val)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_with_warm_restarts)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "step",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and buffers (partial GNN + head)."""
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-1-3-2-1 — Partial STRING_GNN + rank=512 Head + Muon + Warm Restarts"
    )
    p.add_argument("--data-dir",            type=str,   default="data")
    p.add_argument("--lr-backbone",         type=float, default=5e-5,
                   help="AdamW LR for trainable backbone GNN layers (mps.6, mps.7, post_mp)")
    p.add_argument("--lr-head-adamw",       type=float, default=5e-4,
                   help="AdamW LR for head non-matrix params (embeddings, biases)")
    p.add_argument("--lr-muon",             type=float, default=0.005,
                   help="Muon LR for ResBlock 2D weight matrices (proven in node1-2-2-2-1)")
    p.add_argument("--weight-decay",        type=float, default=1.5e-3,
                   help="Weight decay (increased from 1e-3 in parent to address over-parameterization)")
    p.add_argument("--focal-gamma",         type=float, default=2.0)
    p.add_argument("--class-weight-down",   type=float, default=2.0)
    p.add_argument("--class-weight-neutral", type=float, default=0.5)
    p.add_argument("--class-weight-up",     type=float, default=4.0)
    p.add_argument("--head-dropout",        type=float, default=0.25,
                   help="Head dropout (increased from 0.20 to reduce post-peak overfitting)")
    p.add_argument("--warmup-steps",        type=int,   default=50,
                   help="Linear warmup steps before first cosine restart cycle")
    p.add_argument("--restart-period",      type=int,   default=600,
                   help="T_0 for cosine warm restarts in steps (~50 epochs at 12 steps/epoch)")
    p.add_argument("--micro-batch-size",    type=int,   default=16,
                   help="Micro batch size per GPU")
    p.add_argument("--global-batch-size",   type=int,   default=128,
                   help="Global batch size (multiple of micro_batch_size * 8 GPUs)")
    p.add_argument("--max-epochs",          type=int,   default=500,
                   help="Max epochs to allow multiple warm restart cycles (T_0=600 steps ≈ 8 cycles)")
    p.add_argument("--patience",            type=int,   default=80,
                   help="Early stopping patience to capture late-cycle improvements")
    p.add_argument("--num-workers",         type=int,   default=4)
    p.add_argument("--val-check-interval",  type=float, default=1.0)
    p.add_argument("--debug-max-step",      type=int,   default=None,
                   help="Limit train/val/test steps (debug mode)")
    p.add_argument("--fast-dev-run",        action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm  = PerturbDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = PartialStringGNNLitModule(
        lr_backbone          = args.lr_backbone,
        lr_head_adamw        = args.lr_head_adamw,
        lr_muon              = args.lr_muon,
        weight_decay         = args.weight_decay,
        focal_gamma          = args.focal_gamma,
        class_weight_down    = args.class_weight_down,
        class_weight_neutral = args.class_weight_neutral,
        class_weight_up      = args.class_weight_up,
        head_dropout         = args.head_dropout,
        warmup_steps         = args.warmup_steps,
        restart_period       = args.restart_period,
        n_gnn_nodes          = N_GNN_NODES,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
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
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,  # Prevents bf16 numerical instability
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 2-1-3-2-1 — Partial STRING_GNN Fine-Tuning (mps.6, mps.7, post_mp) "
            "+ rank=512 Deep Bilinear Head + Muon Optimizer (ResBlock 2D matrices) "
            "+ Cosine Annealing with Warm Restarts (T_0=600 steps) "
            "+ Stronger Regularization (dropout=0.25, wd=1.5e-3)\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
