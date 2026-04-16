"""
Node 2-1-3-1 — Partially Fine-Tuned STRING_GNN (last 2 GNN layers + post_mp)
             + rank=512 Deep Residual Bilinear MLP Head
             + Stronger Class-Weighted Focal Loss [2.0, 0.5, 4.0]
             + FIXED LR Schedule (total_steps=1200, aligned with actual training duration)
             + Stronger Regularization (dropout=0.3, weight_decay=2e-3)

Architecture (identical to node2-1-3, the parent):
  - STRING_GNN backbone: partially fine-tuned (mps.6, mps.7, post_mp)
    while emb.weight and mps.0-5 remain frozen.
    Intermediate embeddings after mps.0-5 are pre-computed as a buffer.
    During training, only mps.6 + mps.7 + post_mp run live (efficient partial forward).
  - Prediction head: rank=512 bilinear
    - 6-layer residual MLP (hidden=512, expand=4, dropout=0.3 [INCREASED from 0.2])
    - bilinear: pert_proj [B,3,512] x out_gene_emb [6640,512] -> [B,3,6640]
  - Loss: focal (gamma=2.0) + class weights [down=2.0, neutral=0.5, up=4.0]
    (same as node2-1-3, proven effective for minority class learning)
  - Two optimizer groups: backbone (last 2 GNN layers + post_mp) lr=5e-5,
    head (residual MLP + output gene embeddings + oov_emb) lr=5e-4
  - Cosine annealing LR (total_steps=1200 [FIXED from 6600], warmup=100)
    + patience=50
  - weight_decay=2e-3 [INCREASED from 1e-3]
  - No label smoothing (consistently hurts per-gene macro F1)
  - Gradient clipping (max_norm=1.0) for bf16 numerical stability

Key Changes vs Node2-1-3 (parent, Test F1=0.5047):
  1. **total_steps=1200** (was 6600):
     - Parent's total_steps=6600 was designed for ~600 epochs, but training stopped
       at epoch 83 due to patience. The LR cosine schedule barely decayed
       (4.82e-4 at stop vs 5e-4 peak = only 3.6% decay).
     - At total_steps=1200 with ~11 steps/epoch (128 batch / 1416 samples × 8 GPUs):
       * warmup=100: LR ramps to 5e-4 in first ~9 epochs
       * After step 100, cosine decay from 5e-4 → ~0 over steps 100→1200
       * At epoch 50 (step~550): LR ≈ 3.2e-4 (36% of peak)
       * At epoch 83 (step~913): LR ≈ 1.5e-4 (30% of peak)
       * At epoch 100 (step~1100): LR ≈ ~0 (cosine fully annealed)
     - This provides the beneficial late-training LR annealing that node2-1-2
       benefited from but node2-1-3 missed. Feedback identifies this as the
       **primary bottleneck** and **highest impact fix**.
  2. **head_dropout=0.3** (was 0.2):
     - Parent showed widening train/val loss gap (0.130 at epoch 32 → 0.258 at epoch 82)
       with 16.9M head params on only 1,416 training samples.
     - Increased dropout regularizes the 6-layer residual MLP head.
     - Feedback explicitly recommends increasing dropout from 0.2 to 0.3.
  3. **weight_decay=2e-3** (was 1e-3):
     - Parent's weight_decay=1e-3 is standard; however, given the ~12:1 val/train loss ratio
       at epoch 82 (0.299 vs 0.041), stronger L2 regularization is warranted.
     - Feedback recommends weight_decay=2e-3 for both parameter groups.
  4. **max_epochs=300** (was 200):
     - With total_steps=1200, the cosine schedule completes at ~109 epochs.
     - Training may continue for secondary improvement after LR reaches minimum.
     - Extended budget (300 epochs) gives patience=50 sufficient room to capture
       any secondary improvement phase that node2-1-2 exhibited but node2-1-3 missed.
     - In practice, the combination of proper LR decay + stronger regularization
       should produce a cleaner convergence pattern.

Key Design Invariants (retained from node2-1-3):
  - Partial STRING_GNN backbone: mps.6, mps.7, post_mp trainable (~198K params)
  - rank=512 bilinear head (vs rank=256 in node2-1-2)
  - Class weights [down=2.0, neutral=0.5, up=4.0] (node2-1-3's proven recipe)
  - Two-LR optimizer (backbone lr=5e-5, head lr=5e-4)
  - Cosine annealing with linear warmup (warmup_steps=100)
  - Float32 safety for GNN layers in bf16 mixed-precision context
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
BILINEAR_RANK  = 512    # Bilinear interaction rank (same as node2-1-3)
N_GNN_NODES    = 18870  # Total STRING_GNN nodes


# ─── Focal Loss with Class Weights ────────────────────────────────────────────

class FocalLossWithWeights(nn.Module):
    """
    Focal loss with per-class weights.

    Class weights [down=2.0, neutral=0.5, up=4.0] retained from node2-1-3
    which yielded the new tree best F1=0.5047.
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
    """Single residual MLP block: (LN → Linear(D→D*expand) → GELU → Dropout → Linear(D*expand→D)) + skip.

    Dropout increased to 0.3 (from 0.2 in node2-1-3) for stronger regularization.
    The parent showed widening train/val loss gap (from 0.130 at epoch 32 to 0.258 at epoch 82)
    indicating overfitting with 16.9M head params on only 1,416 training samples.
    """

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.3):
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
    Deep bilinear prediction head with rank=512 (retained from node2-1-3).

    Architecture:
      input [B, 256]
        → Linear(256→512) [proj_in]
        → 6 × ResidualBlock(512, expand=4, dropout=0.3) [INCREASED dropout]
        → Linear(512→3*512) [proj_out]
        → reshape [B, 3, 512]
        → einsum("bcr,gr->bcg", [B,3,512], out_gene_emb[6640,512])
        → logits [B, 3, 6640]

    rank=512 vs node2-1-2's rank=256:
    - node2-1-3 (rank=512, partial backbone) achieved F1=0.5047 — current tree best
    - Higher rank provides more expressive bilinear decomposition
    """

    def __init__(
        self,
        in_dim:    int = GNN_DIM,       # 256 (STRING_GNN output dim)
        hidden:    int = HEAD_HIDDEN,   # 512
        expand:    int = HEAD_EXPAND,   # 4
        n_blocks:  int = 6,
        dropout:   float = 0.3,         # INCREASED from 0.2 (node2-1-3) to 0.3
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

        # Deep residual MLP blocks (dropout=0.3 for stronger regularization)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, expand, dropout) for _ in range(n_blocks)
        ])

        # Output projection: HEAD_HIDDEN (512) -> n_classes * rank (3*512=1536)
        self.proj_out = nn.Linear(hidden, n_classes * rank)

        # Learnable output gene embeddings [6640, 512]
        # Higher-dimensional bilinear decomposition (rank=512 vs rank=256 in node2-1-2)
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

    Training strategy (partial fine-tuning, identical to node2-1-3):
    - Frozen: STRING_GNN emb.weight + mps.0-5 (early message-passing layers)
    - Trainable: mps.6, mps.7, post_mp (last 2 GNN layers + output projection)

    Efficient partial forward:
      - Pre-compute: frozen early embedding states up to layer 5 → buffer [N_GNN_NODES, 256]
      - Fine-tune: mps.6, mps.7, post_mp at each training step on pre-computed states
        (Only 2 GCN layers + 1 linear, much cheaper than full GNN)
    """

    def __init__(
        self,
        n_gnn_nodes:  int   = N_GNN_NODES,  # 18870
        head_dropout: float = 0.3,           # INCREASED from 0.2 (node2-1-3)
        oov_init_std: float = 0.01,
    ):
        super().__init__()
        self.n_gnn_nodes = n_gnn_nodes

        # OOV learnable embedding for genes not in STRING vocabulary (~6.4%)
        self.oov_emb = nn.Parameter(torch.randn(1, GNN_DIM) * oov_init_std)

        # Deep bilinear prediction head with rank=512 and increased dropout
        self.head = GNNBilinearHead(
            in_dim=GNN_DIM,
            hidden=HEAD_HIDDEN,
            expand=HEAD_EXPAND,
            n_blocks=6,
            dropout=head_dropout,   # 0.3 (increased from 0.2 in node2-1-3)
            rank=BILINEAR_RANK,
            n_genes=N_GENES_OUT,
            n_classes=N_CLASSES,
        )

        # These will be set in setup_model():
        # - self.gnn_mps_6: trainable GCN layer 6
        # - self.gnn_mps_7: trainable GCN layer 7
        # - self.gnn_post_mp: trainable output projection
        # - self.partial_emb: buffer with GNN states after mps.0-5
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
        This runs on ALL 18870 nodes at once — ~0.26M ops but cheap (2 GCN layers).
        Returns adapted embeddings [18870, 256].

        Note: We explicitly use float32 for GNN computations to avoid dtype conflicts
        between the float32 GNN layer weights and bf16 mixed-precision context.
        """
        dev = self.partial_emb.device
        # Get edge_index and edge_weight on the right device
        ei = self.edge_index.to(dev)
        ew = self.edge_weight.to(dev).float() if self.edge_weight is not None else None

        # Always use float32 for GNN layer computation to avoid dtype conflicts
        x = self.partial_emb.float()  # [18870, 256], always float32

        # mps.6 (trainable) — residual add (same as StringGNNModel.forward)
        x = self.gnn_mps_6(x, ei, edge_weight=ew) + x

        # mps.7 (trainable) — residual add
        x = self.gnn_mps_7(x, ei, edge_weight=ew) + x

        # post_mp (trainable)
        x = self.gnn_post_mp(x)  # [18870, 256], float32
        return x  # float32; caller's head will handle it via its own precision

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
        lr_backbone:          float = 5e-5,   # conservative for partial GNN fine-tuning
        lr_head:              float = 5e-4,   # proven optimal for head (node2-1-3)
        weight_decay:         float = 2e-3,   # INCREASED from 1e-3 (node2-1-3) for regularization
        focal_gamma:          float = 2.0,
        # Class weights retained from node2-1-3 (proven effective, new tree best)
        class_weight_down:    float = 2.0,
        class_weight_neutral: float = 0.5,
        class_weight_up:      float = 4.0,
        head_dropout:         float = 0.3,    # INCREASED from 0.2 (node2-1-3)
        warmup_steps:         int   = 100,    # linear warmup
        total_steps:          int   = 1200,   # FIXED from 6600 — aligned with actual ~100 epoch training
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
            self.print(f"[Node2-1-3-1] Saved {len(dedup_indices)} test predictions -> {pred_path}")

            if self._test_labels:
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-1-3-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Two-LR parameter groups (identical structure to node2-1-3):
        # 1. Backbone: trainable GNN layers (mps.6, mps.7, post_mp) + oov_emb
        #    Conservative lr (5e-5) — partial fine-tuning of pretrained backbone
        # 2. Head: residual MLP + out_gene_emb
        #    Standard lr (5e-4) — fresh prediction head

        backbone_params = []
        head_params     = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Match trainable backbone components
            if any(key in name for key in ["gnn_mps_6", "gnn_mps_7", "gnn_post_mp", "oov_emb"]):
                backbone_params.append(param)
            else:
                # head params: GNNBilinearHead components
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": hp.lr_backbone,
                 "weight_decay": hp.weight_decay},
                {"params": head_params,     "lr": hp.lr_head,
                 "weight_decay": hp.weight_decay},
            ],
            betas=(0.9, 0.999),
        )

        # Cosine annealing with linear warmup
        # total_steps=1200 (FIXED from 6600):
        #   - Parent's schedule was calibrated for ~600 epochs but stopped at 83 (patience)
        #   - LR barely decayed (only 3.6% reduction from peak), depriving the model of
        #     beneficial late-training LR annealing
        #   - At 1200 steps with ~11 steps/epoch and global_batch=128 on 8 GPUs:
        #     * Step 100 (epoch ~9): warmup complete, LR = peak (5e-4 / 5e-5)
        #     * Step 600 (epoch ~55): cosine at 50%, LR ≈ 2.5e-4 / 2.5e-5
        #     * Step 1100 (epoch ~100): cosine near 0, LR ≈ 0
        #   - This provides real LR annealing aligned with actual training duration
        warmup = hp.warmup_steps
        total  = hp.total_steps

        def lr_lambda(current_step: int):
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            progress = float(current_step - warmup) / float(max(1, total - warmup))
            return max(1e-7, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
        description="Node 2-1-3-1 — Partial STRING_GNN Fine-Tuning + rank=512 Bilinear Head "
                    "+ Fixed LR Schedule (total_steps=1200) + Stronger Regularization"
    )
    p.add_argument("--data-dir",             type=str,   default="data")
    p.add_argument("--lr-backbone",          type=float, default=5e-5,
                   help="LR for trainable backbone GNN layers (mps.6, mps.7, post_mp)")
    p.add_argument("--lr-head",              type=float, default=5e-4,
                   help="LR for prediction head parameters")
    p.add_argument("--weight-decay",         type=float, default=2e-3,
                   help="Weight decay (INCREASED from 1e-3 to 2e-3 vs node2-1-3 for regularization)")
    p.add_argument("--focal-gamma",          type=float, default=2.0)
    p.add_argument("--class-weight-down",    type=float, default=2.0,
                   help="Class weight for down-regulated class (retained from node2-1-3)")
    p.add_argument("--class-weight-neutral", type=float, default=0.5,
                   help="Class weight for neutral class (retained from node2-1-3)")
    p.add_argument("--class-weight-up",      type=float, default=4.0,
                   help="Class weight for up-regulated class (retained from node2-1-3)")
    p.add_argument("--head-dropout",         type=float, default=0.3,
                   help="Head dropout (INCREASED from 0.2 to 0.3 vs node2-1-3 for regularization)")
    p.add_argument("--warmup-steps",         type=int,   default=100,
                   help="LR warmup steps")
    p.add_argument("--total-steps",          type=int,   default=1200,
                   help="Total steps for cosine LR (FIXED from 6600 — aligned with actual ~100 epoch training)")
    p.add_argument("--micro-batch-size",     type=int,   default=16,
                   help="Micro batch size per GPU")
    p.add_argument("--global-batch-size",    type=int,   default=128,
                   help="Global batch size (multiple of micro_batch_size * 8 GPUs)")
    p.add_argument("--max-epochs",           type=int,   default=300,
                   help="Max epochs (INCREASED from 200 to 300 to allow secondary improvement after LR decay)")
    p.add_argument("--patience",             type=int,   default=50,
                   help="Early stopping patience (same as node2-1-3)")
    p.add_argument("--num-workers",          type=int,   default=4)
    p.add_argument("--val-check-interval",   type=float, default=1.0)
    p.add_argument("--debug-max-step",       type=int,   default=None,
                   help="Limit train/val/test steps (debug mode)")
    p.add_argument("--fast-dev-run",         action="store_true", default=False)
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
        lr_head              = args.lr_head,
        weight_decay         = args.weight_decay,
        focal_gamma          = args.focal_gamma,
        class_weight_down    = args.class_weight_down,
        class_weight_neutral = args.class_weight_neutral,
        class_weight_up      = args.class_weight_up,
        head_dropout         = args.head_dropout,
        warmup_steps         = args.warmup_steps,
        total_steps          = args.total_steps,
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
            "Node 2-1-3-1 — Partial STRING_GNN Fine-Tuning (mps.6, mps.7, post_mp) "
            "+ rank=512 Deep Bilinear Head + Class Weights [2.0, 0.5, 4.0] "
            "+ FIXED LR Schedule (total_steps=1200) "
            "+ Stronger Regularization (dropout=0.3, weight_decay=2e-3)\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
