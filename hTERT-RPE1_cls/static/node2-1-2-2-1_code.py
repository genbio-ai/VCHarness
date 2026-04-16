"""
Node 2-1-2-2-1 — Frozen STRING_GNN + Rank-512 Bilinear Head
            + Muon Optimizer (Reduced lr=0.002) + Extended Warmup
            + Stronger Class-Weighted Focal Loss + Calibrated LR Schedule

Architecture:
  - Frozen STRING_GNN (5.43M params, 256-dim PPI topology embeddings)
  - Pre-computed GNN forward pass once at setup -> stored as buffer
  - 6-layer residual MLP head (hidden=512, expand=4, rank=512, dropout=0.2)
  - Bilinear interaction: pert_repr [B, 3, 512] x out_gene_emb [6640, 512]
  - Class-weighted focal loss (gamma=2.0, weights=[down=2.0, neutral=0.5, up=4.0])
  - MuonWithAuxAdam: Muon lr=0.002 for hidden weight matrices (REDUCED from parent's 0.005),
    AdamW lr=5e-4 for output gene embeddings, OOV embedding, norms, biases
  - Cosine annealing LR (total_steps=1200, warmup=150, EXTENDED from parent's 50)
  - Gradient clipping (max_norm=1.0) for bf16 numerical stability
  - Patience=50 to enable secondary LR-decay improvement phase

Key Design Rationale:
  - Parent node2-1-2-2 used Muon lr=0.005 + rank=512 and FAILED due to over-aggressive
    optimization: model peaked at epoch 12 (vs parent's epoch 51), train/val loss
    ratio 0.027 at epoch 62, secondary improvement phase never emerged.
  - Root cause: Muon lr=0.005 was proven at rank=256 (node1-1-2-1-1, F1=0.5023),
    but doubling to rank=512 doubles head capacity, amplifying memorization risk.
  - Fix: Reduce Muon lr to 0.002 (~2.5x reduction). At lower lr, Muon optimizes
    more gradually, delaying the peak epoch to ~30-50 where:
      (a) Secondary improvement phase can emerge
      (b) Cosine LR decay provides meaningful variation within the training window
  - Extended warmup from 50 to 150 steps: provides a gentler initial optimization
    trajectory, reducing the risk of catastrophic early memorization that occurred
    in the parent. With warmup=150, Muon lr reaches peak ~epoch 14 vs epoch 5 in parent.
  - Calibrated total_steps=1200 (vs parent's 1650): tighter alignment with actual
    training window of ~100 epochs (1416/128 ≈ 11 steps/epoch × 110 = ~1210 steps).
    At epoch 32 (expected best for this class of nodes): 29% cosine progress, LR ≈ 0.00157.
    At epoch 80: 73% cosine progress, LR ≈ 0.00053. Meaningful decay throughout.
  - Keeps backbone FULLY FROZEN (pre-computed buffer): avoids early convergence
    seen in partial fine-tuning nodes (node2-1-2-1 peaked at epoch 22 vs parent's 51)
  - rank=512 maintained: the bilinear rank itself is not the problem; the optimizer
    aggressiveness was. rank=512 provides +0.005 F1 benefit (node1-2-3 evidence).
  - Class weights [2.0, 0.5, 4.0]: proven in tree best node2-1-3 (F1=0.5047).
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

# Muon optimizer (installed via pip install git+https://github.com/KellerJordan/Muon)
try:
    from muon import MuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    print("[Warning] Muon not available, falling back to AdamW")

# ─── Constants ────────────────────────────────────────────────────────────────

STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES_OUT    = 6640
N_CLASSES      = 3
GNN_DIM        = 256    # STRING_GNN hidden dim
HEAD_HIDDEN    = 512    # Residual MLP hidden dim
HEAD_EXPAND    = 4      # Expand factor in residual block
BILINEAR_RANK  = 512    # Bilinear interaction rank


# ─── Focal Loss with Class Weights ────────────────────────────────────────────

class FocalLossWithWeights(nn.Module):
    """
    Focal loss with optional per-class weights.

    Class weights [down=2.0, neutral=0.5, up=4.0] address class imbalance
    (down=8.1%, neutral=88.9%, up=3.0%) while remaining within the safe range
    proven in node2-1-3 (F1=0.5047, tree best) and node1-2-3 (F1=0.4969).

    focal gamma=2.0 down-weights easy neutral examples: (1-0.9)^2 = 0.01x for
    confident correct neutral predictions, focusing on harder minority classes.
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
    """Single residual MLP block: (LN -> Linear(D->D*expand) -> GELU -> Dropout -> Linear(D*expand->D)) + skip."""

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

    Architecture (rank=512):
      input [B, 256]
        -> proj_in: Linear(256->512)  [B, 512]
        -> 6 x ResidualBlock(512, expand=4, dropout=0.2)  [B, 512]
        -> proj_out: Linear(512->3*512=1536)  [B, 3*512]
        -> reshape [B, 3, 512]
        -> einsum("bcr,gr->bcg", [B,3,512], out_gene_emb[6640,512])
        -> logits [B, 3, 6640]

    The rank=512 bilinear factorization provides a richer gene-perturbation
    interaction space than rank=256 (+0.005 F1 in node1-2-3 vs node1-2).

    The Muon optimizer is applied to the 2D weight matrices in ResidualBlocks
    (proj_in, net.1, net.4 within each block, proj_out) — ideal Muon targets
    per the skill documentation (hidden layer matrices, not embeddings/output).
    The key change in this node: Muon lr=0.002 (vs parent's 0.005) prevents
    the catastrophic early memorization seen in node2-1-2-2.
    """

    def __init__(
        self,
        in_dim:   int = GNN_DIM,         # 256
        hidden:   int = HEAD_HIDDEN,     # 512
        expand:   int = HEAD_EXPAND,     # 4
        n_blocks: int = 6,
        dropout:  float = 0.2,
        rank:     int = BILINEAR_RANK,   # 512
        n_genes:  int = N_GENES_OUT,     # 6640
        n_classes: int = N_CLASSES,       # 3
    ):
        super().__init__()
        self.rank      = rank
        self.n_classes = n_classes
        self.n_genes   = n_genes

        # Input projection from GNN_DIM (256) to HEAD_HIDDEN (512)
        self.proj_in = nn.Linear(in_dim, hidden)

        # Deep residual MLP blocks — 2D weight matrices within are Muon targets
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, expand, dropout) for _ in range(n_blocks)
        ])

        # Output projection from HEAD_HIDDEN (512) to n_classes * rank (3*512=1536)
        self.proj_out = nn.Linear(hidden, n_classes * rank)

        # Learnable output gene embeddings [6640, 512]
        # These learn to encode each gene's response profile (bilinear right side)
        # Use AdamW (not Muon) as gene embeddings should not be orthogonalized
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
    """Simple dataset with pert_ids, STRING indices, and labels."""

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        gnn_indices: List[int],           # STRING_GNN node index for each sample
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long, class indices {0,1,2}
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
        data_dir: str = "data",
        micro_batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage: Optional[str] = None):
        # Load STRING_GNN node name mapping: pert_id (Ensembl gene ID) -> node index
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
    Frozen STRING_GNN backbone + rank-512 bilinear MLP head.

    STRING_GNN runs a single forward pass over the full PPI graph (18,870 nodes,
    786,012 directed edges) at setup time. Embeddings are stored as a frozen
    buffer so that no GNN forward pass is needed during training batches.

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
        bilinear_rank: int = BILINEAR_RANK,
        oov_init_std: float = 0.01,
    ):
        super().__init__()

        self.n_gnn_nodes = n_gnn_nodes

        # OOV learnable embedding for genes not in STRING vocabulary
        # Initialized to near-zero to have minimal disruptive effect initially
        self.oov_emb = nn.Parameter(torch.randn(1, GNN_DIM) * oov_init_std)

        # Deep bilinear prediction head (rank=512)
        self.head = GNNBilinearHead(
            in_dim=GNN_DIM,
            hidden=HEAD_HIDDEN,
            expand=HEAD_EXPAND,
            n_blocks=6,
            dropout=head_dropout,
            rank=bilinear_rank,
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

        graph = torch.load(str(gnn_model_dir / "graph_data.pt"), map_location=device)
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


# ─── LightningModule ──────────────────────────────────────────────────────────

class StringGNNLitModule(pl.LightningModule):

    def __init__(
        self,
        lr: float = 5e-4,
        muon_lr: float = 0.002,          # Reduced from parent's 0.005 — critical fix
        weight_decay: float = 1e-3,
        focal_gamma: float = 2.0,
        # Strong class weights: proven in node2-1-3 (tree best F1=0.5047)
        class_weight_down: float = 2.0,
        class_weight_neutral: float = 0.5,
        class_weight_up: float = 4.0,
        head_dropout: float = 0.2,
        warmup_steps: int = 150,         # Extended from parent's 50 — gentler ramp-up
        total_steps: int = 1200,         # Recalibrated: ~11 steps/epoch × 110 = ~1210
        n_gnn_nodes: int = 18870,
        bilinear_rank: int = BILINEAR_RANK,
        use_muon: bool = True,
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
            bilinear_rank=self.hparams.bilinear_rank,
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
            self.print(f"[Node2-1-2-2-1] Saved {len(dedup_indices)} test predictions -> {pred_path}")

            if self._test_labels:
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-1-2-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # ── Muon optimizer for hidden weight matrices + AdamW for rest ──────────
        #
        # KEY CHANGE vs parent node2-1-2-2:
        # Muon lr REDUCED from 0.005 to 0.002 to prevent the aggressive early
        # memorization that caused parent to peak at epoch 12 with train/val
        # loss ratio of 0.027 at epoch 62.
        #
        # Muon lr=0.005 was proven at rank=256 (node1-1-2-1-1, F1=0.5023),
        # but doubling the head capacity to rank=512 amplified memorization.
        # At lr=0.002, Muon should still provide better optimization than AdamW
        # (due to orthogonalized momentum) while delaying peak epoch to ~30-50,
        # enabling the secondary improvement phase that was critical for parent
        # node2-1-2's success (epochs 20-51 added +0.006 F1).
        #
        # Parameter categorization:
        # - Muon (use_muon=True): 2D weight matrices from ResidualBlocks (Linear
        #   layers inside blocks, proj_in, proj_out) — all 2D parameter matrices
        #   in the hidden layers
        # - AdamW (use_muon=False): out_gene_emb (embedding matrix), oov_emb,
        #   LayerNorm parameters (1D), biases (1D), focal_loss buffers

        if hp.use_muon and MUON_AVAILABLE:
            # Identify 2D hidden weight matrices for Muon
            # Include: proj_in, blocks (net.1, net.4), proj_out weight matrices
            # Exclude: out_gene_emb (embedding/output), oov_emb (1D), biases, norms
            muon_params = []
            adam_params = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                # Muon targets: 2D weight matrices in hidden layers
                is_weight_matrix = (
                    param.ndim >= 2
                    and "out_gene_emb" not in name  # output embedding — use AdamW
                    and "norm" not in name            # LayerNorm — use AdamW
                )
                if is_weight_matrix:
                    muon_params.append(param)
                else:
                    adam_params.append(param)

            # Also include focal_loss parameters if any are trainable
            for name, param in self.focal_loss.named_parameters():
                if param.requires_grad:
                    adam_params.append(param)

            param_groups = [
                # Muon group: hidden weight matrices (proj_in, residual blocks, proj_out)
                # lr=0.002 — reduced from 0.005 to prevent over-aggressive memorization
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=hp.muon_lr,        # 0.002 — critical fix for rank=512 scale
                    weight_decay=hp.weight_decay,
                    momentum=0.95,
                ),
                # AdamW group: embeddings (out_gene_emb, oov_emb), norms, biases
                dict(
                    params=adam_params,
                    use_muon=False,
                    lr=hp.lr,             # 5e-4 — proven optimal for embeddings
                    betas=(0.9, 0.999),
                    weight_decay=hp.weight_decay,
                )
            ]

            optimizer = MuonWithAuxAdam(param_groups)

        else:
            # Fallback to AdamW if Muon not available
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=hp.lr,
                weight_decay=hp.weight_decay,
                betas=(0.9, 0.999),
            )

        # Cosine annealing with extended linear warmup
        # total_steps=1200 calibrated for ~110 epoch window:
        #   steps_per_epoch = 1416 // 128 = 11 steps
        #   expected training window = ~110 epochs = ~1210 steps
        # warmup_steps=150 provides a gentler ramp-up:
        #   At step 150 (~epoch 14): Muon lr reaches peak 0.002
        #   vs parent's step 50 (~epoch 5): Muon lr reached peak 0.005
        #   This delays aggressive optimization by 9 epochs, reducing early memorization risk
        #
        # At expected best epoch ~32 (step ~352): cosine progress = (352-150)/(1200-150) = 19.2%
        #   -> LR = 0.002 * 0.5*(1+cos(0.192*pi)) = 0.002 * 0.5*(1+0.938) = 0.00194
        # At epoch 50 (step ~550): progress = (550-150)/1050 = 38.1%
        #   -> LR = 0.002 * 0.5*(1+cos(0.381*pi)) = 0.002 * 0.5*(1+0.757) = 0.00176
        # At epoch 80 (step ~880): progress = (880-150)/1050 = 69.5%
        #   -> LR = 0.002 * 0.5*(1+cos(0.695*pi)) = 0.002 * 0.5*(1+0.222) = 0.00122
        # Meaningful LR decay throughout training, supporting secondary improvement phase

        warmup = hp.warmup_steps
        total  = hp.total_steps

        def lr_lambda(current_step: int):
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            # Clamp progress to [0, 1] to prevent unintended second cosine cycle
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
        description="Node 2-1-2-2-1 — Frozen STRING_GNN + Rank-512 + "
                    "Muon lr=0.002 (reduced) + Extended Warmup + Calibrated Schedule"
    )
    p.add_argument("--data-dir",             type=str,   default="data")
    p.add_argument("--lr",                   type=float, default=5e-4,
                   help="Learning rate for AdamW parameters (embeddings, norms, biases)")
    p.add_argument("--muon-lr",              type=float, default=0.002,
                   help="Learning rate for Muon parameters — REDUCED from 0.005 to 0.002 "
                        "to prevent over-aggressive memorization at rank=512 scale")
    p.add_argument("--weight-decay",         type=float, default=1e-3)
    p.add_argument("--focal-gamma",          type=float, default=2.0)
    p.add_argument("--class-weight-down",    type=float, default=2.0,
                   help="Class weight for down-regulated class (8.1% freq)")
    p.add_argument("--class-weight-neutral", type=float, default=0.5,
                   help="Class weight for neutral class (88.9% freq)")
    p.add_argument("--class-weight-up",      type=float, default=4.0,
                   help="Class weight for up-regulated class (3.0% freq)")
    p.add_argument("--head-dropout",         type=float, default=0.2)
    p.add_argument("--warmup-steps",         type=int,   default=150,
                   help="Warmup steps — EXTENDED from 50 to 150 for gentler Muon ramp-up")
    p.add_argument("--total-steps",          type=int,   default=1200,
                   help="Total steps for cosine LR schedule (~110 epochs with 11 steps/epoch)")
    p.add_argument("--bilinear-rank",        type=int,   default=512,
                   help="Bilinear interaction rank")
    p.add_argument("--micro-batch-size",     type=int,   default=16,
                   help="Micro batch size per GPU")
    p.add_argument("--global-batch-size",    type=int,   default=128,
                   help="Global batch size (multiple of micro_batch_size * 8 GPUs)")
    p.add_argument("--max-epochs",           type=int,   default=200)
    p.add_argument("--patience",             type=int,   default=50,
                   help="Early stopping patience (50 allows secondary LR-decay improvement)")
    p.add_argument("--num-workers",          type=int,   default=4)
    p.add_argument("--val-check-interval",   type=float, default=1.0)
    p.add_argument("--no-muon",              action="store_true", default=False,
                   help="Disable Muon optimizer, use AdamW instead (fallback)")
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

    use_muon = not args.no_muon

    dm  = PerturbDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = StringGNNLitModule(
        lr                   = args.lr,
        muon_lr              = args.muon_lr,
        weight_decay         = args.weight_decay,
        focal_gamma          = args.focal_gamma,
        class_weight_down    = args.class_weight_down,
        class_weight_neutral = args.class_weight_neutral,
        class_weight_up      = args.class_weight_up,
        head_dropout         = args.head_dropout,
        warmup_steps         = args.warmup_steps,
        total_steps          = args.total_steps,
        n_gnn_nodes          = 18870,
        bilinear_rank        = args.bilinear_rank,
        use_muon             = use_muon,
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
            "Node 2-1-2-2-1 — Frozen STRING_GNN + Rank-512 Bilinear Head "
            "+ Muon lr=0.002 (reduced) + Extended Warmup=150 "
            "+ Class Weights [2.0, 0.5, 4.0]\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
