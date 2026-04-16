"""
Node 2-1-2-3-1 — Frozen STRING_GNN + Deep Residual Bilinear MLP Head (rank=256)
              with Muon lr=0.005 + AdamW + Mild Class-Weighted Focal Loss [1.5, 0.8, 2.5]
              + SGDR Warm Restarts + Dropout=0.3 + Extended Warmup (150 steps)
              + Quality-Filtered Post-Hoc SWA

Architecture:
  - Frozen STRING_GNN (5.43M params, 256-dim PPI topology embeddings)
  - Pre-computed GNN forward pass once at setup -> stored as buffer
  - 6-layer residual MLP head (hidden=512, expand=4, rank=256, dropout=0.3)
  - Bilinear interaction: pert_repr [B, 3, 256] x out_gene_emb [6640, 256]
  - Mild class-weighted focal loss (gamma=2.0, weights=[down=1.5, neutral=0.8, up=2.5])
  - MuonWithAuxAdam: Muon lr=0.005 for hidden 2D weight matrices, AdamW lr=5e-4 for embeddings
  - SGDR cosine warm restarts (T_0=600 steps ~27 epochs, T_mult=1.5)
  - Extended warmup (150 steps) for gentle Muon onset
  - Dropout=0.3 to reduce overfitting (up from parent's 0.2)
  - Gradient clipping (max_norm=1.0) for bf16 numerical stability
  - Patience=80 to allow secondary LR-decay improvement phase
  - Quality-filtered post-hoc SWA (top-k=10, threshold=0.497, temperature=3.0)

Key Design Rationale:
  - Parent failure: Muon lr=0.002 at rank=256 fell BELOW AdamW baseline (F1=0.4961 vs 0.5011)
  - node1-1-2-1-1 proves: Muon lr=0.005 + rank=256 -> F1=0.5023 (best bilinear node)
  - Restoring Muon lr=0.005 is primary fix (from feedback: "restore to 0.003-0.005")
  - Mild weights [1.5, 0.8, 2.5]: parent (node2-1-2) used these and achieved F1=0.5011
    with secondary improvement phase. Strong weights [2.0, 0.5, 4.0] + Muon lr=0.002
    amplified gradient variance for rare up-class (3%), preventing secondary phase.
  - SGDR warm restarts (T_0=600, T_mult=1.5): tree-best nodes show SGDR staircases
    provide +0.002-0.006 F1 gains via local minima escape
  - Dropout=0.3: parent feedback recommends increasing from 0.2 to 0.3
  - Extended warmup=150 steps: delays Muon reaching peak LR, prevents premature memorization
  - SWA: tree-best nodes (F1=0.5182) achieve +0.004-0.006 gain over best single checkpoint
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

# --- Constants ----------------------------------------------------------------

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_DIM        = 256    # STRING_GNN hidden dim
HEAD_HIDDEN    = 512    # Residual MLP hidden dim
HEAD_EXPAND    = 4      # Expand factor in residual block
BILINEAR_RANK  = 256    # Bilinear interaction rank (proven at this scale for Muon)


# --- Focal Loss with Class Weights -------------------------------------------

class FocalLossWithWeights(nn.Module):
    """
    Focal loss with optional per-class weights.

    Mild class weights [down=1.5, neutral=0.8, up=2.5] per feedback analysis:
    - Parent (node2-1-2) used [1.5, 0.8, 2.5] with AdamW and achieved F1=0.5011
      with a secondary improvement phase (best epoch=51)
    - Stronger [2.0, 0.5, 4.0] combined with Muon lr=0.002 at rank=256 caused
      premature convergence (best epoch=26) and failed secondary improvement phase
    - Mild weights balance minority-class gradient signals without amplifying noise
      from the rare up-class (3.0% frequency) that drove parent's overfitting

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
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            reduction='none'
        )
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction='none'))
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


# --- Metric ------------------------------------------------------------------

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


# --- STRING_GNN Head ----------------------------------------------------------

class ResidualBlock(nn.Module):
    """Single residual MLP block: (LN -> Linear(D->D*expand) -> GELU -> Dropout -> Linear(D*expand->D)) + skip."""

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
    Deep bilinear prediction head using frozen STRING_GNN PPI embeddings.

    Architecture (rank=256 proven for Muon optimization, node1-1-2-1-1: F1=0.5023):
      input [B, 256]
        -> proj_in: Linear(256->512)   [B, 512]
        -> 6 x ResidualBlock(512, expand=4, dropout=0.3)  [B, 512]
        -> proj_out: Linear(512->3*256)  [B, 3*256]
        -> reshape [B, 3, 256]
        -> einsum("bcr,gr->bcg", [B,3,256], out_gene_emb[6640,256])
        -> logits [B, 3, 6640]

    Dropout=0.3 (vs parent's 0.2): feedback recommends higher dropout to extend
    the generalization window and allow SGDR warm restarts to escape overfitting.
    """

    def __init__(
        self,
        in_dim:   int = GNN_DIM,         # 256
        hidden:   int = HEAD_HIDDEN,     # 512
        expand:   int = HEAD_EXPAND,     # 4
        n_blocks: int = 6,
        dropout:  float = 0.3,
        rank:     int = BILINEAR_RANK,   # 256
        n_genes:  int = N_GENES_OUT,     # 6640
        n_classes: int = N_CLASSES,      # 3
    ):
        super().__init__()
        self.rank     = rank
        self.n_classes = n_classes
        self.n_genes  = n_genes

        self.proj_in = nn.Linear(in_dim, hidden)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, expand, dropout) for _ in range(n_blocks)
        ])

        self.proj_out = nn.Linear(hidden, n_classes * rank)

        # Learnable output gene embeddings [6640, 256]
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

        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, 6640]
        return logits


# --- Dataset -----------------------------------------------------------------

class PerturbDataset(Dataset):
    """Simple dataset with pert_ids, STRING indices, and labels."""

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        gnn_indices: List[int],
        labels: Optional[torch.Tensor] = None,
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
    pert_ids    = [b["pert_id"]   for b in batch]
    symbols     = [b["symbol"]    for b in batch]
    gnn_indices = torch.tensor([b["gnn_index"] for b in batch], dtype=torch.long)
    out = {
        "pert_id":    pert_ids,
        "symbol":     symbols,
        "gnn_index":  gnn_indices,
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# --- DataModule --------------------------------------------------------------

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
        node_names_path = Path(STRING_GNN_DIR) / "node_names.json"
        node_names = json.loads(node_names_path.read_text())
        self.node_name_to_idx = {name: i for i, name in enumerate(node_names)}
        self.n_gnn_nodes = len(node_names)  # 18870

        def load_split(fname: str, has_label: bool) -> PerturbDataset:
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            pert_ids = df["pert_id"].tolist()
            symbols  = df["symbol"].tolist()

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


# --- Full Model --------------------------------------------------------------

class StringGNNPerturbModel(nn.Module):
    """
    Frozen STRING_GNN backbone + deep bilinear MLP head.

    STRING_GNN runs a single forward pass at setup time.
    Embeddings are stored as a buffer so no GNN forward pass needed during training.
    """

    def __init__(
        self,
        n_gnn_nodes: int = 18870,
        head_dropout: float = 0.3,
        oov_init_std: float = 0.01,
    ):
        super().__init__()

        self.n_gnn_nodes = n_gnn_nodes

        # OOV learnable embedding for genes not in STRING vocabulary
        self.oov_emb = nn.Parameter(torch.randn(1, GNN_DIM) * oov_init_std)

        # Deep bilinear prediction head (rank=256, proven for Muon optimization)
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

        self._embeddings_initialized = False

    def setup_embeddings(self, device: torch.device):
        """
        Run STRING_GNN forward pass once to compute all node embeddings.
        Called once during LightningModule.setup() before training.
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
        emb = outputs.last_hidden_state.float().cpu()

        self.register_buffer("gnn_embeddings", emb)

        del gnn, graph, edge_index, edge_weight, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        self._embeddings_initialized = True

    def forward(self, gnn_indices: torch.Tensor) -> torch.Tensor:
        """
        gnn_indices: [B] long
        returns: logits [B, 3, 6640]
        """
        B = gnn_indices.shape[0]
        oov_mask = (gnn_indices >= self.n_gnn_nodes)
        safe_idx = gnn_indices.clone()
        safe_idx[oov_mask] = 0

        emb = self.gnn_embeddings[safe_idx]    # [B, 256]
        emb = emb.to(gnn_indices.device)

        if oov_mask.any():
            emb[oov_mask] = self.oov_emb.expand(oov_mask.sum(), -1)

        logits = self.head(emb)  # [B, 3, 6640]
        return logits


# --- Helpers -----------------------------------------------------------------

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
    Build MuonWithAuxAdam optimizer.

    Muon targets: 2D weight matrices in hidden layers (proj_in, ResidualBlock linears, proj_out)
    AdamW targets: embeddings (out_gene_emb, oov_emb), LayerNorm params, biases

    Muon lr=0.005 is the proven optimal at rank=256 (node1-1-2-1-1: F1=0.5023).
    The parent's mistake was reducing to 0.002, which fell below AdamW baseline.
    """
    try:
        from muon import MuonWithAuxAdam

        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_muon_target = (
                param.ndim >= 2
                and "out_gene_emb" not in name
                and "oov_emb" not in name
                and "norm" not in name
                and "gnn_embeddings" not in name
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
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=adamw_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        return optimizer, "AdamW (Muon not available)"


# --- SGDR LR Scheduler -------------------------------------------------------

class SGDRScheduler:
    """
    Cosine Annealing Warm Restarts (SGDR) at step level.

    T_0: initial cycle length in steps (default=600 steps ~27 epochs with batch=128/1416 samples)
    T_mult: cycle length multiplier (default=1.5 for gradually longer cycles)
    eta_min: minimum LR multiplier (floor, prevent total LR collapse)
    warmup_steps: linear warmup before first cosine cycle

    SGDR staircase mechanism from tree-best nodes (F1=0.51+):
    - Multiple warm restarts create escape events from local optima
    - T_mult=1.5 allows slightly deeper convergence per cycle while maintaining restart frequency
    - Cycle peaks: T_0=600, then 900, 1350, 2025...

    Warmup=150 steps: gentler Muon onset, delays peak LR to allow better initialization
    """
    def __init__(self, warmup_steps: int, T_0: int, T_mult: float, eta_min: float = 1e-7):
        self.warmup_steps = warmup_steps
        self.T_0    = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

    def __call__(self, current_step: int) -> float:
        # Linear warmup phase
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))

        # SGDR cosine cycles
        step = current_step - self.warmup_steps
        T_cur = self.T_0
        cycle_start = 0
        while cycle_start + T_cur <= step:
            cycle_start += T_cur
            T_cur = int(T_cur * self.T_mult)

        T_cur_pos = step - cycle_start
        progress = float(T_cur_pos) / float(max(1, T_cur))
        return max(self.eta_min, 0.5 * (1.0 + np.cos(np.pi * progress)))


# --- LightningModule ---------------------------------------------------------

class StringGNNLitModule(pl.LightningModule):

    def __init__(
        self,
        lr: float = 5e-4,                      # AdamW LR for embeddings
        muon_lr: float = 0.005,                # Muon LR for hidden matrices (restored to proven 0.005)
        weight_decay: float = 1e-3,
        focal_gamma: float = 2.0,
        # Mild class weights from parent node2-1-2 (F1=0.5011 with secondary improvement phase)
        # Reverted from [2.0, 0.5, 4.0] which caused premature convergence with Muon
        class_weight_down: float = 1.5,
        class_weight_neutral: float = 0.8,
        class_weight_up: float = 2.5,
        head_dropout: float = 0.3,
        warmup_steps: int = 150,               # Extended for gentler Muon onset
        sgdr_T0: int = 600,                    # SGDR first cycle steps (~27 epochs)
        sgdr_T_mult: float = 1.5,              # Gradually longer cycles
        n_gnn_nodes: int = 18870,
        # SWA params
        swa_threshold: float = 0.497,
        swa_top_k: int = 10,
        swa_temperature: float = 3.0,
        swa_every_n_epochs: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []
        self._test_labels:   List[torch.Tensor] = []

        # SWA checkpoint tracking (accumulated during training)
        self._swa_checkpoints: List[Dict] = []  # list of {val_f1: float, state_dict: dict}

    def setup(self, stage: Optional[str] = None):
        self.model = StringGNNPerturbModel(
            n_gnn_nodes=self.hparams.n_gnn_nodes,
            head_dropout=self.hparams.head_dropout,
        )
        self.model.setup_embeddings(self.device)

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

        # Collect SWA checkpoints on global rank 0 every N epochs
        if (
            self.trainer.is_global_zero
            and not self.trainer.sanity_checking
            and self.current_epoch > 0
            and self.current_epoch % self.hparams.swa_every_n_epochs == 0
            and f1 >= self.hparams.swa_threshold
        ):
            # Save a lightweight copy of head parameters (trainable params only)
            state = {
                k: v.cpu().clone()
                for k, v in self.model.state_dict().items()
                if not k.startswith("gnn_embeddings")  # skip large frozen buffer
            }
            self._swa_checkpoints.append({"val_f1": f1, "state_dict": state})

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
            self.print(f"[Node2-1-2-3-1] Saved {len(dedup_indices)} test predictions -> {pred_path}")

            if self._test_labels:
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-1-2-3-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        optimizer, opt_name = _build_muon_optimizer(
            self.model,
            muon_lr=hp.muon_lr,
            adamw_lr=hp.lr,
            weight_decay=hp.weight_decay,
        )
        self.print(f"[Node2-1-2-3-1] Using optimizer: {opt_name}")

        # SGDR cosine warm restarts with linear warmup
        # T_0=600 steps ~27 epochs; T_mult=1.5 for gradually longer cycles
        # Warmup=150 steps for gentler Muon onset
        scheduler_fn = SGDRScheduler(
            warmup_steps=hp.warmup_steps,
            T_0=hp.sgdr_T0,
            T_mult=hp.sgdr_T_mult,
            eta_min=1e-7,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def apply_swa_and_save_predictions(self, dm: "PerturbDataModule", out_dir: Path, trainer: "pl.Trainer") -> Optional[float]:
        """
        Quality-filtered SWA: select top-k checkpoints by val_f1, apply exponential
        weighting (higher val_f1 -> higher weight), average model weights, run test.

        This follows the pattern from tree-best nodes (F1=0.5182) where SWA
        provides +0.004-0.006 gain over best single checkpoint.

        Only runs on global rank 0 after training completes.
        Returns test F1 if labels available, else None.
        """
        if not self._swa_checkpoints:
            self.print("[Node2-1-2-3-1] No SWA checkpoints collected, skipping SWA.")
            return None

        # Sort by val_f1 descending and take top-k
        sorted_ckpts = sorted(self._swa_checkpoints, key=lambda x: x["val_f1"], reverse=True)
        top_k = self.hparams.swa_top_k
        selected = sorted_ckpts[:top_k]
        self.print(f"[Node2-1-2-3-1] SWA pool: {len(selected)}/{len(self._swa_checkpoints)} checkpoints, "
                   f"val_f1 range [{selected[-1]['val_f1']:.4f}, {selected[0]['val_f1']:.4f}]")

        if len(selected) < 2:
            self.print("[Node2-1-2-3-1] Insufficient SWA checkpoints, skipping SWA.")
            return None

        # Compute exponential weights based on val_f1 scores
        scores = torch.tensor([c["val_f1"] for c in selected], dtype=torch.float32)
        temp = self.hparams.swa_temperature
        weights = torch.softmax(scores * temp, dim=0)
        self.print(f"[Node2-1-2-3-1] SWA weights: {[f'{w:.3f}' for w in weights.tolist()]}")

        # Average the state dicts
        avg_state_dict = {}
        for key in selected[0]["state_dict"].keys():
            avg_tensor = sum(w * c["state_dict"][key].float() for w, c in zip(weights, selected))
            avg_state_dict[key] = avg_tensor

        # Load averaged weights into model
        # Restore gnn_embeddings buffer (not in SWA dict because too large)
        current_full = self.model.state_dict()
        for k, v in avg_state_dict.items():
            current_full[k] = v
        self.model.load_state_dict(current_full, strict=False)

        # Run test inference with averaged model
        self.model.eval()
        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

        test_loader = dm.test_dataloader()
        with torch.no_grad():
            for batch in test_loader:
                gnn_idx = batch["gnn_index"].to(self.device)
                logits = self.model(gnn_idx)
                probs  = torch.softmax(logits.float(), dim=1)
                self._test_preds.append(probs.detach().cpu())
                self._test_pert_ids.extend(batch["pert_id"])
                self._test_symbols.extend(batch["symbol"])
                if "label" in batch:
                    self._test_labels.append(batch["label"].cpu())

        all_probs  = torch.cat(self._test_preds, 0)
        all_pert   = self._test_pert_ids
        all_syms   = self._test_symbols

        # Dedup
        seen_pids: set = set()
        dedup_indices: List[int] = []
        for i, pid in enumerate(all_pert):
            if pid not in seen_pids:
                seen_pids.add(pid)
                dedup_indices.append(i)

        pred_path_swa = out_dir / "test_predictions.tsv"
        all_probs_np = all_probs.numpy()
        with open(pred_path_swa, "w") as fh:
            fh.write("idx\tinput\tprediction\n")
            for i in dedup_indices:
                fh.write(
                    f"{all_pert[i]}\t{all_syms[i]}\t"
                    f"{json.dumps(all_probs_np[i].tolist())}\n"
                )
        self.print(f"[Node2-1-2-3-1] SWA: saved {len(dedup_indices)} test predictions -> {pred_path_swa}")

        test_f1 = None
        if self._test_labels:
            labels_cat = torch.cat(self._test_labels, 0)
            dedup_probs  = all_probs_np[dedup_indices]
            dedup_labels = labels_cat[dedup_indices].numpy()
            test_f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
            self.print(f"[Node2-1-2-3-1] SWA self-computed test F1 = {test_f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

        return test_f1

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


# --- Main --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-1-2-3-1 — Frozen STRING_GNN + Rank=256 Bilinear Head "
                    "+ Muon lr=0.005 + Mild Class Weights + SGDR + SWA"
    )
    p.add_argument("--data-dir",             type=str,   default="data")
    p.add_argument("--lr",                   type=float, default=5e-4,
                   help="AdamW learning rate for embedding parameters")
    p.add_argument("--muon-lr",              type=float, default=0.005,
                   help="Muon LR for 2D hidden weight matrices (proven optimal at rank=256)")
    p.add_argument("--weight-decay",         type=float, default=1e-3)
    p.add_argument("--focal-gamma",          type=float, default=2.0)
    p.add_argument("--class-weight-down",    type=float, default=1.5,
                   help="Class weight for down-regulated class; mild weights from node2-1-2")
    p.add_argument("--class-weight-neutral", type=float, default=0.8,
                   help="Class weight for neutral class; mild weights from node2-1-2")
    p.add_argument("--class-weight-up",      type=float, default=2.5,
                   help="Class weight for up-regulated class; mild weights from node2-1-2")
    p.add_argument("--head-dropout",         type=float, default=0.3,
                   help="Dropout in residual blocks (increased from 0.2 to delay overfitting)")
    p.add_argument("--warmup-steps",         type=int,   default=150,
                   help="LR warmup steps (extended for gentler Muon onset)")
    p.add_argument("--sgdr-T0",              type=int,   default=600,
                   help="SGDR first cycle steps (~27 epochs with batch=128/1416 samples)")
    p.add_argument("--sgdr-T-mult",          type=float, default=1.5,
                   help="SGDR cycle length multiplier (gradually longer cycles)")
    p.add_argument("--swa-threshold",        type=float, default=0.497,
                   help="Minimum val_f1 for SWA checkpoint inclusion")
    p.add_argument("--swa-top-k",            type=int,   default=10,
                   help="Maximum checkpoints to include in SWA ensemble")
    p.add_argument("--swa-temperature",      type=float, default=3.0,
                   help="Softmax temperature for SWA weighting (higher -> more concentrated)")
    p.add_argument("--swa-every-n-epochs",   type=int,   default=3,
                   help="Collect SWA checkpoints every N epochs")
    p.add_argument("--micro-batch-size",     type=int,   default=16,
                   help="Micro batch size per GPU")
    p.add_argument("--global-batch-size",    type=int,   default=128,
                   help="Global batch size (multiple of micro_batch_size * 8 GPUs)")
    p.add_argument("--max-epochs",           type=int,   default=300)
    p.add_argument("--patience",             type=int,   default=80,
                   help="Early stopping patience (80 allows SGDR staircase to develop)")
    p.add_argument("--num-workers",          type=int,   default=4)
    p.add_argument("--val-check-interval",   type=float, default=1.0)
    p.add_argument("--debug-max-step",       type=int,   default=None,
                   help="Limit train/val/test steps (debug mode)")
    p.add_argument("--fast-dev-run",         action="store_true", default=False)
    p.add_argument("--skip-swa",             action="store_true", default=False,
                   help="Skip SWA post-processing (for debugging)")
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
        sgdr_T0             = args.sgdr_T0,
        sgdr_T_mult         = args.sgdr_T_mult,
        n_gnn_nodes         = 18870,
        swa_threshold       = args.swa_threshold,
        swa_top_k           = args.swa_top_k,
        swa_temperature     = args.swa_temperature,
        swa_every_n_epochs  = args.swa_every_n_epochs,
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
        gradient_clip_val=1.0,
    )

    trainer.fit(lit, datamodule=dm)

    # Step 1: Test with best checkpoint (standard evaluation)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    # Step 2: Apply SWA on rank 0 if checkpoint pool is rich enough
    if (
        trainer.is_global_zero
        and not args.skip_swa
        and not fast_dev_run
        and args.debug_max_step is None
        and len(lit._swa_checkpoints) >= 3
    ):
        lit.print(f"[Node2-1-2-3-1] Applying SWA with {len(lit._swa_checkpoints)} collected checkpoints...")
        dm.setup()  # Ensure test dataloader is ready
        swa_f1 = lit.apply_swa_and_save_predictions(dm, out_dir, trainer)
        if swa_f1 is not None:
            lit.print(f"[Node2-1-2-3-1] SWA test F1 = {swa_f1:.4f}")
    elif trainer.is_global_zero and len(lit._swa_checkpoints) < 3:
        lit.print(f"[Node2-1-2-3-1] SWA skipped: only {len(lit._swa_checkpoints)} checkpoints collected "
                  f"(need >= 3, threshold={args.swa_threshold:.3f})")

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 2-1-2-3-1 — Frozen STRING_GNN + Rank=256 Bilinear Head\n"
            "+ Muon lr=0.005 (restored from parent's sub-optimal 0.002)\n"
            "+ Mild Class Weights [1.5, 0.8, 2.5] (from node2-1-2's proven config)\n"
            "+ SGDR Warm Restarts (T_0=600, T_mult=1.5) for staircase improvement\n"
            "+ Dropout=0.3 (increased from 0.2 per feedback)\n"
            "+ Extended Warmup=150 steps + Quality-Filtered SWA\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
