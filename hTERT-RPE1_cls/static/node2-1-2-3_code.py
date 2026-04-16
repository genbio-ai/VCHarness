"""
Node 2-1-2-3 — Frozen STRING_GNN + Deep Residual Bilinear MLP Head (rank=256)
              with Muon lr=0.002 + AdamW + Stronger Class-Weighted Focal Loss
              + Calibrated LR Schedule (total_steps=1650)

Architecture:
  - Frozen STRING_GNN (5.43M params, 256-dim PPI topology embeddings)
  - Pre-computed GNN forward pass once at setup → stored as buffer
  - 6-layer residual MLP head (hidden=512, expand=4, rank=256, dropout=0.2)
  - Bilinear interaction: pert_repr [B, 3, 256] × out_gene_emb [6640, 256]
  - Stronger class-weighted focal loss (gamma=2.0, weights=[down=2.0, neutral=0.5, up=4.0])
  - MuonWithAuxAdam: Muon lr=0.002 for hidden 2D weight matrices, AdamW lr=5e-4 for embeddings
  - Calibrated cosine LR schedule (total_steps=1650, warmup=100, clamped progress)
  - Gradient clipping (max_norm=1.0) for bf16 numerical stability
  - Patience=50 to enable secondary LR-decay improvement phase

Key Design Rationale:
  - Frozen backbone: avoids sibling-1's early convergence (partial backbone caused epoch-22 peak)
  - rank=256: parent's proven rank; Muon benefit confirmed at this scale in node1-1-2-1-1 (+0.0111 F1)
  - Muon lr=0.002: reduced from sibling-2's over-aggressive 0.005 (which caused epoch-12 peak at rank=512)
    * Direct address of sibling-2 failure: lr=0.005 was proven at rank=256 only, not rank=512
    * lr=0.002 at rank=256 should delay best epoch to 30-50 (vs sibling-2's epoch 12)
  - Extended warmup 100 steps: gentler Muon onset, prevents premature optimization before convergence
  - Stronger class weights [2.0, 0.5, 4.0]: proven across tree best (node2-1-3: F1=0.5047),
    node1-1-2-1-1 (F1=0.5023), node2-1-2-1 (F1=0.5016) — consistently beneficial vs parent's mild weights
  - total_steps=1650: calibrated for ~150-epoch window (11 steps/epoch x 150 = 1650)
    * Enables meaningful cosine LR decay within actual training duration
    * Supports secondary improvement phase (parent's phase was at epoch 20-51)
  - Progress clamping: prevents unintended second cosine cycle
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
GNN_DIM        = 256    # STRING_GNN hidden dim
HEAD_HIDDEN    = 512    # Residual MLP hidden dim
HEAD_EXPAND    = 4      # Expand factor in residual block
BILINEAR_RANK  = 256    # Bilinear interaction rank (same as parent - proven at this scale)


# ─── Focal Loss with Class Weights ────────────────────────────────────────────

class FocalLossWithWeights(nn.Module):
    """
    Focal loss with optional per-class weights.

    Stronger class weights [down=2.0, neutral=0.5, up=4.0] address class imbalance
    while remaining within the safe range confirmed across multiple tree nodes.

    Focal gamma=2.0 down-weights easy neutral examples: (1-0.9)^2 = 0.01x for
    confident correct neutral predictions, allowing the model to focus on harder
    down-regulated and up-regulated examples.
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
        # Cross-entropy with optional class weights (no reduction)
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            reduction='none'
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

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.2):
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
    Deep bilinear prediction head using frozen STRING_GNN PPI embeddings.

    Architecture (mirroring parent node2-1-2's proven design at rank=256):
      input [B, 256]
        → 6 × ResidualBlock(512, expand=4, dropout=0.2)  [B, 512]
        → Linear(256→512) at proj_in  [B, 512]
        → Linear(512→3*256) at proj_out  [B, 3*256]
        → reshape [B, 3, 256]
        → einsum("bcr,gr->bcg", [B,3,256], out_gene_emb[6640,256])
        → logits [B, 3, 6640]

    This bilinear factorization provides an inductive structural prior.
    Rank=256 is proven at this scale for Muon optimization (node1-1-2-1-1: F1=0.5023).
    """

    def __init__(
        self,
        in_dim:   int = GNN_DIM,         # 256
        hidden:   int = HEAD_HIDDEN,     # 512
        expand:   int = HEAD_EXPAND,     # 4
        n_blocks: int = 6,
        dropout:  float = 0.2,
        rank:     int = BILINEAR_RANK,   # 256
        n_genes:  int = N_GENES_OUT,     # 6640
        n_classes: int = N_CLASSES,       # 3
    ):
        super().__init__()
        self.rank     = rank
        self.n_classes = n_classes
        self.n_genes  = n_genes

        # Input projection from GNN_DIM (256) to HEAD_HIDDEN (512)
        self.proj_in = nn.Linear(in_dim, hidden)

        # Deep residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, expand, dropout) for _ in range(n_blocks)
        ])

        # Output projection from HEAD_HIDDEN (512) to n_classes * rank (3*256=768)
        self.proj_out = nn.Linear(hidden, n_classes * rank)

        # Learnable output gene embeddings [6640, 256]
        # These learn to encode each gene's response profile
        self.out_gene_emb = nn.Parameter(torch.randn(n_genes, rank) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 256]
        returns: logits [B, 3, 6640]
        """
        h = self.proj_in(x)                             # [B, 512]
        for block in self.blocks:
            h = block(h)                                # [B, 512]

        proj = self.proj_out(h)                         # [B, 3*256]
        B = proj.shape[0]
        pert_proj = proj.view(B, self.n_classes, self.rank)  # [B, 3, 256]

        # Bilinear interaction: for each gene, compute class logits as dot product
        # of per-class perturbation vector with gene embedding
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, 6640]
        return logits


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    """Simple dataset with pert_ids, STRING indices, and labels."""

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        gnn_indices: List[int],           # STRING_GNN node index for each sample
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long, class indices {0,1,2}
    ):
        self.pert_ids   = pert_ids
        self.symbols    = symbols
        self.gnn_indices = gnn_indices
        self.labels     = labels

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
    pert_ids   = [b["pert_id"]   for b in batch]
    symbols    = [b["symbol"]    for b in batch]
    gnn_indices = torch.tensor([b["gnn_index"] for b in batch], dtype=torch.long)
    out = {
        "pert_id":    pert_ids,
        "symbol":     symbols,
        "gnn_index":  gnn_indices,
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
        # Load STRING_GNN node name mapping: pert_id (Ensembl gene ID) → node index
        node_names_path = Path(STRING_GNN_DIR) / "node_names.json"
        node_names = json.loads(node_names_path.read_text())
        # node_names is a list of Ensembl gene IDs; build reverse lookup
        self.node_name_to_idx = {name: i for i, name in enumerate(node_names)}
        self.n_gnn_nodes = len(node_names)  # 18870

        def load_split(fname: str, has_label: bool) -> PerturbDataset:
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            pert_ids = df["pert_id"].tolist()
            symbols  = df["symbol"].tolist()

            # Map pert_id (Ensembl gene ID) to STRING_GNN node index
            # Use out-of-vocabulary sentinel = n_gnn_nodes (will get learnable OOV emb)
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

class StringGNNPerturbModel(nn.Module):
    """
    Frozen STRING_GNN backbone + deep bilinear MLP head.

    STRING_GNN runs a single forward pass over the full PPI graph (18,870 nodes,
    786,012 directed edges) at setup time. Embeddings are stored as a buffer so
    that no GNN forward pass is needed during training batches.

    Each training/inference step simply indexes the buffer by gnn_index to get
    the 256-dim PPI embedding for the perturbed gene, then passes it through the
    deep residual bilinear head.

    The OOV embedding (n_gnn_nodes index) handles genes not in STRING vocabulary
    (~6.4% of dataset) as a learnable fallback.
    """

    def __init__(
        self,
        n_gnn_nodes: int = 18870,
        head_dropout: float = 0.2,
        oov_init_std: float = 0.01,
    ):
        super().__init__()

        self.n_gnn_nodes = n_gnn_nodes

        # OOV learnable embedding for genes not in STRING vocabulary
        # Initialized to near-zero to have minimal disruptive effect initially
        self.oov_emb = nn.Parameter(torch.randn(1, GNN_DIM) * oov_init_std)

        # Deep bilinear prediction head (rank=256, same as proven parent architecture)
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

        # GNN embeddings buffer — will be registered in setup_embeddings()
        # [N_GNN_NODES, 256] frozen
        self._embeddings_initialized = False

    def setup_embeddings(self, device: torch.device):
        """
        Run STRING_GNN forward pass once to compute all node embeddings.
        Called once during LightningModule.setup() before training.
        The embeddings are stored as a frozen buffer for batch lookups.
        """
        if self._embeddings_initialized:
            return

        gnn_model_dir = Path(STRING_GNN_DIR)
        gnn = AutoModel.from_pretrained(str(gnn_model_dir), trust_remote_code=True)
        gnn.eval()
        gnn = gnn.to(device)

        graph = torch.load(str(gnn_model_dir / "graph_data.pt"), map_location=device, weights_only=False)
        edge_index  = graph["edge_index"].to(device)
        edge_weight = graph.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        with torch.no_grad():
            outputs = gnn(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
        # [18870, 256] on device
        emb = outputs.last_hidden_state.float().cpu()

        # Register as non-trainable buffer (will be moved to correct device by Lightning)
        self.register_buffer("gnn_embeddings", emb)

        # Free GNN model memory
        del gnn, graph, edge_index, edge_weight, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        self._embeddings_initialized = True

    def forward(self, gnn_indices: torch.Tensor) -> torch.Tensor:
        """
        gnn_indices: [B] long — STRING_GNN node index for each perturbed gene
                     OOV genes have index = n_gnn_nodes (handled via oov_emb)

        returns: logits [B, 3, 6640]
        """
        B = gnn_indices.shape[0]
        # Look up embeddings from buffer
        oov_mask = (gnn_indices >= self.n_gnn_nodes)  # [B] bool
        safe_idx = gnn_indices.clone()
        safe_idx[oov_mask] = 0  # temporary valid index

        emb = self.gnn_embeddings[safe_idx]    # [B, 256]
        emb = emb.to(gnn_indices.device)

        # Replace OOV embeddings
        if oov_mask.any():
            emb[oov_mask] = self.oov_emb.expand(oov_mask.sum(), -1)

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


def _build_muon_optimizer(model, muon_lr: float, adamw_lr: float, weight_decay: float):
    """
    Build MuonWithAuxAdam optimizer if Muon is available, otherwise fall back to AdamW.

    Muon targets: 2D weight matrices in hidden layers (proj_in, ResidualBlock linears, proj_out)
    AdamW targets: embeddings (out_gene_emb, oov_emb), LayerNorm params, biases

    Key rule: out_gene_emb is 2D but is an embedding matrix (gene identity representation),
    not a hidden transformation — it goes to AdamW, not Muon.
    """
    try:
        from muon import MuonWithAuxAdam

        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Classify parameter
            is_muon_target = (
                param.ndim >= 2                         # must be 2D+ matrix
                and "out_gene_emb" not in name          # exclude gene embedding (embedding-like)
                and "oov_emb" not in name               # exclude OOV embedding
                and "norm" not in name                  # exclude LayerNorm
                and "gnn_embeddings" not in name        # exclude frozen buffer (shouldn't be here)
            )

            if is_muon_target:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        param_groups = [
            dict(
                params=muon_params,
                use_muon=True,
                lr=muon_lr,
                weight_decay=weight_decay,
                momentum=0.95,
            ),
            dict(
                params=adamw_params,
                use_muon=False,
                lr=adamw_lr,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            ),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
        return optimizer, "MuonWithAuxAdam"

    except ImportError:
        # Fallback to standard AdamW if Muon not installed
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=adamw_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        return optimizer, "AdamW (Muon not available)"


# ─── LightningModule ──────────────────────────────────────────────────────────

class StringGNNLitModule(pl.LightningModule):

    def __init__(
        self,
        lr: float = 5e-4,                      # AdamW LR for embeddings
        muon_lr: float = 0.002,                # Muon LR for hidden matrices (reduced from 0.005)
        weight_decay: float = 1e-3,
        focal_gamma: float = 2.0,
        # Stronger class weights: proven in node2-1-3 (F1=0.5047), node1-1-2-1-1 (F1=0.5023)
        # down=2.0 (8.1% freq), neutral=0.5 (88.9% freq), up=4.0 (3.0% freq)
        class_weight_down: float = 2.0,
        class_weight_neutral: float = 0.5,
        class_weight_up: float = 4.0,
        head_dropout: float = 0.2,
        warmup_steps: int = 100,               # Extended from 50 for gentler Muon onset
        total_steps: int = 1650,               # Calibrated: 11 steps/epoch x 150 epochs
        n_gnn_nodes: int = 18870,
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
        self.model = StringGNNPerturbModel(
            n_gnn_nodes=self.hparams.n_gnn_nodes,
            head_dropout=self.hparams.head_dropout,
        )
        # Pre-compute STRING_GNN embeddings (frozen buffer)
        self.model.setup_embeddings(self.device)

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
            self.print(f"[Node2-1-2-3] Saved {len(dedup_indices)} test predictions → {pred_path}")

            if self._test_labels:
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-1-2-3] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Build Muon optimizer (with AdamW fallback if Muon not installed)
        optimizer, opt_name = _build_muon_optimizer(
            self.model,
            muon_lr=hp.muon_lr,
            adamw_lr=hp.lr,
            weight_decay=hp.weight_decay,
        )
        self.print(f"[Node2-1-2-3] Using optimizer: {opt_name}")

        # Calibrated cosine annealing with extended linear warmup
        # total_steps=1650 calibrated for ~150-epoch window (11 steps/epoch x 150)
        # warmup_steps=100 provides gentler Muon onset (vs 50 in sibling-2 which peaked at epoch 12)
        warmup = hp.warmup_steps
        total  = hp.total_steps

        def lr_lambda(current_step: int):
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            # Clamped progress to prevent unintended second cosine cycle
            progress = min(1.0, float(current_step - warmup) / float(max(1, total - warmup)))
            return max(1e-7, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters and buffers."""
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
        description="Node 2-1-2-3 — Frozen STRING_GNN + Rank=256 Bilinear Head + Muon lr=0.002"
    )
    p.add_argument("--data-dir",             type=str,   default="data")
    p.add_argument("--lr",                   type=float, default=5e-4,
                   help="AdamW learning rate for embedding parameters (out_gene_emb, oov_emb, biases)")
    p.add_argument("--muon-lr",              type=float, default=0.002,
                   help="Muon LR for 2D hidden weight matrices (reduced from 0.005 to avoid overfitting at rank=256)")
    p.add_argument("--weight-decay",         type=float, default=1e-3)
    p.add_argument("--focal-gamma",          type=float, default=2.0)
    p.add_argument("--class-weight-down",    type=float, default=2.0,
                   help="Class weight for down-regulated class (8.1% freq); proven in node2-1-3")
    p.add_argument("--class-weight-neutral", type=float, default=0.5,
                   help="Class weight for neutral class (88.9% freq); proven in node2-1-3")
    p.add_argument("--class-weight-up",      type=float, default=4.0,
                   help="Class weight for up-regulated class (3.0% freq); proven in node2-1-3")
    p.add_argument("--head-dropout",         type=float, default=0.2)
    p.add_argument("--warmup-steps",         type=int,   default=100,
                   help="LR warmup steps (extended from 50 for gentler Muon onset)")
    p.add_argument("--total-steps",          type=int,   default=1650,
                   help="Total steps for cosine LR schedule (calibrated: 11 steps/epoch x 150 epochs)")
    p.add_argument("--micro-batch-size",     type=int,   default=16,
                   help="Micro batch size per GPU")
    p.add_argument("--global-batch-size",    type=int,   default=128,
                   help="Global batch size (multiple of micro_batch_size * 8 GPUs)")
    p.add_argument("--max-epochs",           type=int,   default=200)
    p.add_argument("--patience",             type=int,   default=50,
                   help="Early stopping patience (50 allows secondary LR-decay improvement phase)")
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
    lit = StringGNNLitModule(
        lr                  = args.lr,
        muon_lr             = args.muon_lr,
        weight_decay        = args.weight_decay,
        focal_gamma         = args.focal_gamma,
        class_weight_down   = args.class_weight_down,
        class_weight_neutral= args.class_weight_neutral,
        class_weight_up     = args.class_weight_up,
        head_dropout        = args.head_dropout,
        warmup_steps        = args.warmup_steps,
        total_steps         = args.total_steps,
        n_gnn_nodes         = 18870,
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
            "Node 2-1-2-3 — Frozen STRING_GNN + Rank=256 Bilinear Head "
            "+ Muon lr=0.002 + Stronger Class-Weighted Focal Loss [2.0, 0.5, 4.0]\n"
            "+ Calibrated LR Schedule (total_steps=1650, warmup=100)\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
