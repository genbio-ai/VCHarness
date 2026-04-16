"""
Node 2-1-2-1 — Partial STRING_GNN Fine-tuning + Rank-512 Bilinear Head
               + Stronger Class-Weighted Focal Loss

Architecture:
  - STRING_GNN partially fine-tuned (last 2 GCN layers + post_mp, ~45K backbone params)
  - Full GNN forward pass each training batch (partial backbone unfreezing)
  - 6-layer residual MLP head (hidden=512, expand=4, rank=512, dropout=0.2)
  - Bilinear interaction: pert_repr [B, 3, 512] × out_gene_emb [6640, 512]
  - Class-weighted focal loss (gamma=2.0, weights=[down=2.0, neutral=0.5, up=4.0])
  - Two-group AdamW: backbone LR=1e-5, head LR=5e-4, cosine annealing
  - Gradient clipping (max_norm=1.0) for bf16 numerical stability
  - Patience=50 to enable secondary LR-decay improvement phase

Key Design Rationale:
  - Parent (node2-1-2): frozen STRING_GNN + rank=256 + [1.5, 0.8, 2.5] = F1=0.5011 (tree best)
  - The frozen backbone is the primary performance bottleneck. Partial fine-tuning of the
    last 2 GCN layers + post_mp allows the PPI embeddings to adapt to the DEG prediction
    task while avoiding catastrophic forgetting/overfitting on 1,416 samples.
  - rank=512 (vs parent's rank=256): node1-2-3 demonstrated +0.005 F1 from rank increase
  - class weights [2.0, 0.5, 4.0]: node1-2-3's proven weights that yielded F1=0.4969;
    now applied to the stronger parent foundation (F1=0.5011) for further gains
  - Low backbone LR (1e-5): prevents overfitting of the 5.43M backbone on small dataset
  - High head LR (5e-4): retains fast convergence for the prediction head
  - find_unused_parameters=False: with partial backbone training, all params are used
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
GNN_DIM        = 256     # STRING_GNN hidden dim
HEAD_HIDDEN    = 512     # Residual MLP hidden dim
HEAD_EXPAND    = 4       # Expand factor in residual block
BILINEAR_RANK  = 512     # Bilinear interaction rank (increased from parent's 256)


# ─── Focal Loss with Class Weights ────────────────────────────────────────────

class FocalLossWithWeights(nn.Module):
    """
    Focal loss with optional per-class weights.

    class weights [down=2.0, neutral=0.5, up=4.0] from node1-2-3 (F1=0.4969):
    - More aggressive than parent [1.5, 0.8, 2.5] targeting the very rare up-regulated class
    - Consistent with the gradient: going from no-weights (node1-2: F1=0.4912) to
      mild weights (parent node2-1-2: F1=0.5011) to these weights is the proven direction
    - Proven in node1-2-3 without instability (vs node1-2-2-1's too-aggressive [3.0, 0.3, 7.0])

    focal gamma=2.0: standard modulation proven across the tree.
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
    Deep bilinear prediction head with rank-512 output interaction.

    Architecture:
      input [B, 256]
        → Linear(256→512) [proj_in]
        → 6 × ResidualBlock(512, expand=4, dropout=0.2)  [B, 512]
        → Linear(512→3*512=1536) [proj_out]
        → reshape [B, 3, 512]
        → einsum("bcr,gr->bcg", [B,3,512], out_gene_emb[6640,512])
        → logits [B, 3, 6640]

    rank=512 (vs parent's rank=256): node1-2-3 showed +0.005 F1 gain from this change.
    The wider bilinear interaction allows more expressive gene-perturbation combinations.
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
        self.rank     = rank
        self.n_classes = n_classes
        self.n_genes  = n_genes

        # Input projection from GNN_DIM (256) to HEAD_HIDDEN (512)
        self.proj_in = nn.Linear(in_dim, hidden)

        # Deep residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, expand, dropout) for _ in range(n_blocks)
        ])

        # Output projection from HEAD_HIDDEN (512) to n_classes * rank (3*512=1536)
        self.proj_out = nn.Linear(hidden, n_classes * rank)

        # Learnable output gene embeddings [6640, 512]
        # These learn to encode each gene's response profile
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

class StringGNNPartialFinetuneModel(nn.Module):
    """
    Partially fine-tuned STRING_GNN backbone + deep bilinear MLP head (rank=512).

    Key difference from parent (node2-1-2):
    - Parent: GNN fully frozen, embeddings pre-computed once at setup()
    - This node: GNN last 2 layers (mps.6, mps.7) + post_mp are trainable
      The first 6 GCN layers (mps.0-5) and embedding table (emb) remain frozen.
      Total additional trainable backbone params: ~45K (from the small GCN linear layers)

    The partial fine-tuning allows the backbone embeddings to adapt to the DEG task
    while preventing overfitting on the small 1,416-sample training set. The frozen
    early layers preserve general PPI topology structure; the trainable last layers
    learn task-specific adaptation.

    Architecture:
    - STRING_GNN backbone (partially fine-tuned: last 2 layers + post_mp)
    - Full GNN forward pass at each training step (no pre-computation buffer)
    - OOV embedding (learnable) for genes not in STRING vocabulary
    - rank-512 bilinear prediction head
    """

    def __init__(
        self,
        n_gnn_nodes: int = 18870,
        head_dropout: float = 0.2,
        oov_init_std: float = 0.01,
    ):
        super().__init__()
        self.n_gnn_nodes = n_gnn_nodes

        # Learnable OOV embedding for genes not in STRING vocabulary
        self.oov_emb = nn.Parameter(torch.randn(1, GNN_DIM) * oov_init_std)

        # Deep bilinear prediction head (rank=512)
        self.head = GNNBilinearHead(
            in_dim=GNN_DIM,
            hidden=HEAD_HIDDEN,
            expand=HEAD_EXPAND,
            n_blocks=6,
            dropout=head_dropout,
            rank=BILINEAR_RANK,    # 512 (vs parent's 256)
            n_genes=N_GENES_OUT,
            n_classes=N_CLASSES,
        )

        # GNN backbone — will be loaded and partially frozen in setup_gnn()
        self.gnn = None
        self._gnn_initialized = False

        # Graph tensors — stored for repeated forward passes during training
        self.register_buffer("edge_index",  torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer("edge_weight", torch.zeros(0))
        self._graph_loaded = False

    def setup_gnn(self, device: torch.device):
        """
        Load STRING_GNN, selectively freeze layers, and load graph tensors.
        Called once during LightningModule.setup().

        Frozen (early layers — preserve general PPI topology):
          - emb (Embedding table, 18870×256)
          - mps.0 through mps.5 (first 6 GCN layers)

        Trainable (task-specific adaptation):
          - mps.6, mps.7 (last 2 GCN layers)
          - post_mp (output projection)
          ~45K params total in backbone
        """
        if self._gnn_initialized:
            return

        gnn_model_dir = Path(STRING_GNN_DIR)
        gnn = AutoModel.from_pretrained(str(gnn_model_dir), trust_remote_code=True)
        gnn = gnn.to(device)

        # Freeze early layers (embedding table + first 6 GCN layers)
        for name, param in gnn.named_parameters():
            should_freeze = (
                name.startswith("emb.") or
                name.startswith("mps.0.") or
                name.startswith("mps.1.") or
                name.startswith("mps.2.") or
                name.startswith("mps.3.") or
                name.startswith("mps.4.") or
                name.startswith("mps.5.")
            )
            param.requires_grad = not should_freeze

        # Report parameter counts
        total_gnn = sum(p.numel() for p in gnn.parameters())
        trainable_gnn = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
        print(f"[Node2-1-2-1] STRING_GNN: {trainable_gnn}/{total_gnn} trainable "
              f"({100*trainable_gnn/total_gnn:.2f}%) — last 2 layers + post_mp")

        # Cast trainable backbone params to float32 for optimizer stability
        for param in gnn.parameters():
            if param.requires_grad:
                param.data = param.data.float()

        self.gnn = gnn

        # Load graph tensors into registered buffers
        graph = torch.load(str(gnn_model_dir / "graph_data.pt"), map_location=device)
        edge_index = graph["edge_index"].to(device)
        edge_weight = graph.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        else:
            edge_weight = torch.ones(edge_index.shape[1], device=device)

        # Re-register buffers with actual graph data
        # (initial buffers were placeholders)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self._graph_loaded = True
        self._gnn_initialized = True

        # Move ALL model components to device.
        # setup_gnn() only moves the gnn backbone; head and oov_emb are still on CPU
        # after __init__. Calling self.to(device) here ensures everything is on device
        # before the first forward pass.
        self.to(device)

        del graph
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def forward(self, gnn_indices: torch.Tensor) -> torch.Tensor:
        """
        gnn_indices: [B] long — STRING_GNN node index for each perturbed gene
                     OOV genes have index = n_gnn_nodes (handled via oov_emb)

        This runs the full GNN forward pass each batch to allow gradient flow
        through the trainable last 2 GCN layers + post_mp.

        returns: logits [B, 3, 6640]
        """
        # Run GNN forward pass (gradient flows through last 2 layers + post_mp)
        edge_index  = self.edge_index
        edge_weight = self.edge_weight if self.edge_weight.numel() > 0 else None

        outputs = self.gnn(
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        all_emb = outputs.last_hidden_state.float()  # [18870, 256]

        B = gnn_indices.shape[0]
        # Look up embeddings for the batch
        oov_mask = (gnn_indices >= self.n_gnn_nodes)  # [B] bool
        safe_idx = gnn_indices.clone()
        safe_idx[oov_mask] = 0  # temporary valid index

        emb = all_emb[safe_idx]    # [B, 256]

        # Replace OOV embeddings
        if oov_mask.any():
            emb = emb.clone()
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
        backbone_lr: float = 1e-5,
        weight_decay: float = 1e-3,
        focal_gamma: float = 2.0,
        # Stronger class weights from node1-2-3 (F1=0.4969):
        # down=2.0 (8.1% of labels), neutral=0.5 (88.9%), up=4.0 (3.0%)
        # Proven effective in node1-2-3, now applied on top of this stronger foundation
        class_weight_down: float = 2.0,
        class_weight_neutral: float = 0.5,
        class_weight_up: float = 4.0,
        head_dropout: float = 0.2,
        warmup_steps: int = 50,
        total_steps: int = 6600,
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
        self.model = StringGNNPartialFinetuneModel(
            n_gnn_nodes=self.hparams.n_gnn_nodes,
            head_dropout=self.hparams.head_dropout,
        )
        # Load STRING_GNN and partially freeze it
        self.model.setup_gnn(self.device)

        # Class weights tensor: [0=down, 1=neutral, 2=up]
        cw = torch.tensor([
            self.hparams.class_weight_down,
            self.hparams.class_weight_neutral,
            self.hparams.class_weight_up,
        ], dtype=torch.float32)

        self.focal_loss = FocalLossWithWeights(
            gamma=self.hparams.focal_gamma,
            class_weights=cw,
        ).to(self.device)  # class_weights buffer must be on same device as logits

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
            self.print(f"[Node2-1-2-1] Saved {len(dedup_indices)} test predictions → {pred_path}")

            if self._test_labels:
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-1-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Two-group optimizer: different LRs for backbone vs. head
        # - backbone_lr = 1e-5: very low to prevent overfitting on 1,416 samples
        # - head LR = 5e-4: retained from parent for fast convergence
        backbone_params = list(self.model.gnn.parameters()) if self.model.gnn is not None else []
        backbone_trainable = [p for p in backbone_params if p.requires_grad]

        param_groups = []
        if backbone_trainable:
            param_groups.append({
                "params": backbone_trainable,
                "lr": hp.backbone_lr,
                "weight_decay": hp.weight_decay,
            })
        param_groups.append({
            "params": list(self.model.head.parameters()) + [self.model.oov_emb],
            "lr": hp.lr,
            "weight_decay": hp.weight_decay,
        })

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
        )

        # Cosine annealing with linear warmup
        # total_steps=6600 calibrated for ~200 epochs on 1,416 samples with
        # global_batch=128 (micro=16, 8 GPUs), enabling secondary LR-decay improvement
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
    p = argparse.ArgumentParser(description="Node 2-1-2-1 — Partial STRING_GNN Fine-tuning + Rank-512 Head")
    p.add_argument("--data-dir",             type=str,   default="data")
    p.add_argument("--lr",                   type=float, default=5e-4,
                   help="Learning rate for head parameters")
    p.add_argument("--backbone-lr",          type=float, default=1e-5,
                   help="Learning rate for STRING_GNN backbone (last 2 layers + post_mp)")
    p.add_argument("--weight-decay",         type=float, default=1e-3)
    p.add_argument("--focal-gamma",          type=float, default=2.0)
    p.add_argument("--class-weight-down",    type=float, default=2.0,
                   help="Class weight for down-regulated class (8.1% freq)")
    p.add_argument("--class-weight-neutral", type=float, default=0.5,
                   help="Class weight for neutral class (88.9% freq)")
    p.add_argument("--class-weight-up",      type=float, default=4.0,
                   help="Class weight for up-regulated class (3.0% freq)")
    p.add_argument("--head-dropout",         type=float, default=0.2)
    p.add_argument("--warmup-steps",         type=int,   default=50)
    p.add_argument("--total-steps",          type=int,   default=6600,
                   help="Total steps for cosine LR schedule (enables secondary improvement phase)")
    p.add_argument("--micro-batch-size",     type=int,   default=16,
                   help="Micro batch size per GPU")
    p.add_argument("--global-batch-size",    type=int,   default=128,
                   help="Global batch size (multiple of micro_batch_size * 8 GPUs)")
    p.add_argument("--max-epochs",           type=int,   default=200)
    p.add_argument("--patience",             type=int,   default=50,
                   help="Early stopping patience (50 allows secondary LR-decay improvement)")
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
        backbone_lr         = args.backbone_lr,
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

    # find_unused_parameters=False: with partial backbone training, all included
    # params (last 2 GCN layers + post_mp) receive gradients each step
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
            "Node 2-1-2-1 — Partial STRING_GNN Fine-tuning + Rank-512 Bilinear Head "
            "+ Stronger Class-Weighted Focal Loss\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
