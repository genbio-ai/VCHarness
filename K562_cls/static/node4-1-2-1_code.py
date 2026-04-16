"""Node 4-1-2-1 – scFoundation (top-6 layers) + STRING_GNN (discriminative LR) +
   K=16 PPI Neighborhood Attention + Gated Fusion + Restored Single Linear Head +
   Weighted CE Loss.

Changes from parent node4-1-2 (test F1=0.4290, severely regressed):

1. REMOVE embedding noise augmentation (std=0.1 → 0.0) [PRIMARY FIX]:
   The embedding noise was the primary cause of regression. It creates a train/test
   distribution mismatch — model trains on corrupted embeddings but is evaluated on
   clean ones. The pretrained STRING_GNN node embeddings are especially vulnerable
   since they are already meaningful; corrupting them with Gaussian noise removes
   useful PPI signal without any mechanism for the model to recover it.

2. RESTORE single linear head (512→19920, 10.2M params) [CAPACITY FIX]:
   Parent's narrow 2-layer head (512→256→19920, 5.3M params) was the secondary cause
   of underfitting. With 1,388 training samples and 19,920 outputs per sample, the
   model needs sufficient output capacity. Returning to node4-1's proven single linear
   head. Training loss was 4.6x higher in the parent (0.074 vs node4-1's 0.016).

3. RESTORE weight_decay=2e-2 [OPTIMIZATION FIX]:
   Parent's 1.5e-2 contributed to underfitting. 2e-2 is node4-1's proven value that
   balances regularization with convergence capacity.

4. ADD K=16 PPI Neighborhood Attention on STRING_GNN branch [NEW IMPROVEMENT]:
   For each perturbed gene, aggregates its top-16 PPI neighbors via scaled dot-product
   attention weighted by STRING confidence log-priors, then gates between center
   embedding and aggregated context. This enriches the GNN representation with local
   PPI topology context. Proven in:
   - node2-1-1-1: +0.052 F1 gain (primary driver of achieving 0.5059)
   - node4-2 lineage: +0.003-0.004 F1 gain per application (node4-2-2-1: 0.4867)
   Adds only ~33K trainable params (q_proj + k_proj + gate_fc).

5. SWITCH loss: LabelSmoothingFocalLoss → weighted CE + label smoothing [ALIGNMENT FIX]:
   Node4-1's feedback confirmed the focal+label_smooth combination diverges from F1 —
   val loss steadily increased (0.054→0.089) while val F1 remained stable, indicating
   the focal component optimizes a different objective than per-gene macro-F1. Switching
   to F.cross_entropy(weight=class_weights, label_smoothing=0.1) removes this divergence.
   The node4-2 lineage used this same loss and consistently outperformed node4-1 lineage.

6. INCREASE min_lr_ratio: 0.08 → 0.12 [SCHEDULE IMPROVEMENT]:
   Node4-1 found its best checkpoint at epoch 80 when LR was still ~20% of peak (~4e-5).
   After LR dropped below 2e-5 (10% of peak), no further improvement was observed.
   A floor of 12% (LR=2.4e-5) maintains sufficient exploration capacity while providing
   enough decay for fine-grained convergence.

7. KEEP: max_epochs=200, patience=25, 6 scFoundation fine-tuned layers, GNN discriminative
   LR (gnn_lr_ratio=5.0), WarmupCosine schedule with 10-epoch warmup, GatedFusion.

Differentiation from siblings:
- node4-1-1: CAWR + frozen GNN + BN head → failed (underfitting)
- node4-1-2: embedding noise + narrow head → failed (underfitting/mismatch)
- node4-1-2-1: PPI neighborhood attention + weighted CE + restored single head [this]

Expected performance: targeting F1 > 0.4629 (node4-1 baseline), leveraging PPI attention
benefit (+0.003-0.005 from memory) on top of the restored proven architecture.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES    = 6640
N_CLASSES  = 3
SCF_HIDDEN = 768    # scFoundation hidden size
GNN_HIDDEN = 256    # STRING_GNN hidden size
FUSION_DIM = 512    # output dimension of gated fusion

SCF_MODEL_DIR = "/home/Models/scFoundation"
GNN_MODEL_DIR = "/home/Models/STRING_GNN"

CLASS_FREQ = [0.0429, 0.9251, 0.0320]  # down, neutral, up (remapped 0,1,2)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1, matching calc_metric.py definition."""
    y_hat       = preds.argmax(dim=1)
    G           = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)
    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0).float()
        tp  = (is_pred & is_true).float().sum(0)
        fp  = (is_pred & ~is_true).float().sum(0)
        fn  = (~is_pred & is_true).float().sum(0)
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(prec + rec > 0, 2*prec*rec/(prec+rec+1e-8), torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


def _build_topk_neighbors(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    n_nodes: int,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build top-K neighbor table from PPI graph for neighborhood attention.

    For each node, selects the top-K neighbors with highest STRING confidence
    score (edge_weight), sorted in descending order.

    Returns:
        topk_ids:  [n_nodes, K] long  – top-K neighbor node indices (0-padded)
        topk_logw: [n_nodes, K] float – log(STRING confidence), -1e9 for padding
        topk_mask: [n_nodes, K] bool  – True for valid (non-padded) entries
    """
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    wts = (edge_weight.cpu().float().numpy() if edge_weight is not None
           else np.ones(src.shape[0], dtype=np.float32))

    topk_ids  = np.zeros((n_nodes, K), dtype=np.int64)
    topk_logw = np.full((n_nodes, K), -1e9, dtype=np.float32)
    topk_mask = np.zeros((n_nodes, K), dtype=bool)

    # Sort edges by (src ASC, weight DESC) to get top-K per source node
    sort_idx = np.lexsort((-wts, src))
    src_s, dst_s, wts_s = src[sort_idx], dst[sort_idx], wts[sort_idx]

    # Find boundaries where source node changes
    boundaries = np.concatenate(
        [[0], np.where(np.diff(src_s) != 0)[0] + 1, [len(src_s)]]
    )

    for i in range(len(boundaries) - 1):
        start, end = int(boundaries[i]), int(boundaries[i + 1])
        n = int(src_s[start])
        if n >= n_nodes:
            continue
        k = min(end - start, K)
        topk_ids[n, :k]  = dst_s[start:start + k]
        topk_logw[n, :k] = np.log(np.clip(wts_s[start:start + k], 1e-9, None))
        topk_mask[n, :k] = True

    return (
        torch.tensor(topk_ids,  dtype=torch.long),
        torch.tensor(topk_logw, dtype=torch.float32),
        torch.tensor(topk_mask, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# PPI Neighborhood Attention Module
# ---------------------------------------------------------------------------
class PpiNeighborhoodAttention(nn.Module):
    """K-hop PPI neighborhood attention for STRING graph.

    For each perturbed gene in a batch, retrieves its top-K PPI neighbors
    (pre-sorted by STRING confidence), computes scaled dot-product attention
    weighted by log(STRING confidence), and gates between the center embedding
    and the aggregated neighbor context.

    Trainable parameters: q_proj (d×attn_dim), k_proj (d×attn_dim), gate_fc (2d→1)
    Total: ~33K params for d=256, attn_dim=64 (negligible relative to backbone).

    Buffers (registered after build_and_register() call):
        topk_ids:  [N, K] long  – neighbor node indices
        topk_logw: [N, K] float – log(STRING confidence), -1e9 for padding
        topk_mask: [N, K] bool  – True for valid neighbors

    Architecture adapted from node2-1-1-1 (F1=0.5059) and node4-2-2-1 (F1=0.4867)
    where PPI neighborhood attention was the primary driver of improvement.
    """

    def __init__(self, d_gnn: int = 256, K: int = 16, attn_dim: int = 64) -> None:
        super().__init__()
        self.K        = K
        self.attn_dim = attn_dim
        self.q_proj   = nn.Linear(d_gnn, attn_dim, bias=False)
        self.k_proj   = nn.Linear(d_gnn, attn_dim, bias=False)
        self.gate_fc  = nn.Linear(d_gnn * 2, 1)

    def forward(
        self,
        node_embs:      torch.Tensor,  # [N, d]
        center_indices: torch.Tensor,  # [B]
    ) -> torch.Tensor:                 # [B, d]
        # Retrieve center embeddings
        center = node_embs[center_indices]          # [B, d]

        # Retrieve neighbor data for this batch
        nbr_ids  = self.topk_ids[center_indices]    # [B, K]
        nbr_logw = self.topk_logw[center_indices]   # [B, K]
        nbr_mask = self.topk_mask[center_indices]   # [B, K]

        # Get neighbor embeddings (clamp to avoid OOB on 0-padded entries)
        nbr_embs = node_embs[nbr_ids.clamp(min=0)]  # [B, K, d]

        # Scaled dot-product attention + log-confidence prior
        Q      = self.q_proj(center).unsqueeze(1)              # [B, 1, attn_dim]
        Kmat   = self.k_proj(nbr_embs)                         # [B, K, attn_dim]
        scores = (Q * Kmat).sum(-1) / (self.attn_dim ** 0.5)  # [B, K]
        scores = scores + nbr_logw                              # add log-confidence
        scores = scores.masked_fill(~nbr_mask, float('-inf'))  # mask padding

        # Stable softmax: handle nodes with no valid neighbors (avoid all-NaN)
        has_nbrs = nbr_mask.any(dim=1, keepdim=True)  # [B, 1]
        scores_safe = scores.clone()
        no_nbr_mask = ~has_nbrs.squeeze(1)
        if no_nbr_mask.any():
            # For nodes with no neighbors, set one score to 0 to avoid NaN in softmax
            # Context will be zeroed out anyway by the has_nbrs mask below
            scores_safe[no_nbr_mask, 0] = 0.0

        attn = torch.softmax(scores_safe, dim=-1)  # [B, K]
        attn = attn * nbr_mask.float()              # zero out padding positions
        attn = attn * has_nbrs.float()              # zero out no-neighbor rows

        # Aggregate neighbor context
        context = (attn.unsqueeze(-1) * nbr_embs).sum(dim=1)  # [B, d]

        # Learned gate: output = gate * center + (1 - gate) * context
        gate = torch.sigmoid(self.gate_fc(torch.cat([center, context], dim=-1)))  # [B, 1]
        return gate * center + (1.0 - gate) * context                             # [B, d]


# ---------------------------------------------------------------------------
# Gated Fusion Module
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Learnable gated fusion of two heterogeneous embeddings.

    output = gate_scf * proj_scf(scf_emb) + gate_gnn * proj_gnn(gnn_emb)
    """

    def __init__(
        self,
        d_scf: int = SCF_HIDDEN,
        d_gnn: int = GNN_HIDDEN,
        d_out: int = FUSION_DIM,
        fusion_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        d_in = d_scf + d_gnn
        self.proj_scf   = nn.Linear(d_scf, d_out)
        self.proj_gnn   = nn.Linear(d_gnn, d_out)
        self.gate_scf   = nn.Linear(d_in,  d_out)
        self.gate_gnn   = nn.Linear(d_in,  d_out)
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout    = nn.Dropout(fusion_dropout)

    def forward(
        self, scf_emb: torch.Tensor, gnn_emb: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([scf_emb, gnn_emb], dim=-1)      # [B, d_scf+d_gnn]
        gate_s   = torch.sigmoid(self.gate_scf(combined))     # [B, d_out]
        gate_g   = torch.sigmoid(self.gate_gnn(combined))     # [B, d_out]
        fused    = gate_s * self.proj_scf(scf_emb) + gate_g * self.proj_gnn(gnn_emb)
        return self.dropout(self.layer_norm(fused))            # [B, d_out]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List] = (
            [torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
             for row in df["label"].tolist()]
            if has_label else None
        )

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx": idx,
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_collate_scf(tokenizer):
    """Collate function that tokenizes for scFoundation."""

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]

        # scFoundation: encode single-gene perturbation as expression=1.0
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")

        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out

    return collate_fn


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer   = None

    def setup(self, stage: Optional[str] = None) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(SCF_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(SCF_MODEL_DIR, trust_remote_code=True)

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        self.train_ds = DEGDataset(train_df)
        self.val_ds   = DEGDataset(val_df)
        self.test_ds  = DEGDataset(test_df)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=make_collate_scf(self.tokenizer))

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# LR Scheduler: Linear Warmup + Cosine Annealing
# ---------------------------------------------------------------------------
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup for warmup_steps steps, then cosine decay to min_lr_ratio."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.12,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = float(step + 1) / float(max(1, self.warmup_steps))
        else:
            progress = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
        return [base_lr * scale for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FusionDEGModel(pl.LightningModule):
    """scFoundation (top-6 layers fine-tuned) + STRING_GNN (discriminative LR) +
    K=16 PPI Neighborhood Attention + Gated Fusion + Single Linear Head.

    Key changes from parent node4-1-2:
    - REMOVED: Gaussian embedding noise augmentation (was causing train/test mismatch)
    - RESTORED: Single linear head (512→19920, 10.2M params) vs parent's narrow 2-layer
    - RESTORED: weight_decay=2e-2 (parent used 1.5e-2, contributing to underfitting)
    - NEW: K=16 PPI neighborhood attention on STRING_GNN branch (proven in node2/node4-2)
    - SWITCHED: loss from focal+label_smooth to weighted CE + label_smooth
    - INCREASED: min_lr_ratio 0.08 → 0.12 for better late-epoch exploration
    """

    def __init__(
        self,
        scf_finetune_layers: int  = 6,
        head_dropout: float       = 0.5,
        fusion_dropout: float     = 0.2,
        lr: float                 = 2e-4,
        gnn_lr_ratio: float       = 5.0,
        weight_decay: float       = 2e-2,
        label_smoothing: float    = 0.1,
        warmup_epochs: int        = 10,
        max_epochs: int           = 200,
        min_lr_ratio: float       = 0.12,
        ppi_K: int                = 16,
        ppi_attn_dim: int         = 64,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        gnn_dir = Path(GNN_MODEL_DIR)

        # ---- scFoundation backbone ----
        self.scf = AutoModel.from_pretrained(
            SCF_MODEL_DIR,
            trust_remote_code=True,
            _use_flash_attention_2=True,
        )
        self.scf = self.scf.to(torch.bfloat16)
        self.scf.config.use_cache = False
        self.scf.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Freeze all scF params, then unfreeze top-k transformer layers
        for param in self.scf.parameters():
            param.requires_grad = False

        n_scf_layers = len(self.scf.encoder.transformer_encoder)
        for i in range(n_scf_layers - hp.scf_finetune_layers, n_scf_layers):
            for param in self.scf.encoder.transformer_encoder[i].parameters():
                param.requires_grad = True
        # Also unfreeze the final LayerNorm
        for param in self.scf.encoder.norm.parameters():
            param.requires_grad = True

        # Cast unfrozen scF params to float32 for stable optimization
        for name, param in self.scf.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        scf_train = sum(p.numel() for p in self.scf.parameters() if p.requires_grad)
        scf_total = sum(p.numel() for p in self.scf.parameters())
        print(f"[Node4-1-2-1] scFoundation: {scf_train:,}/{scf_total:,} trainable")

        # ---- STRING_GNN ---- (full fine-tuning with discriminative lower LR)
        self.gnn = AutoModel.from_pretrained(str(gnn_dir), trust_remote_code=True)
        self.gnn = self.gnn.to(torch.float32)

        # Load graph data and node name→index mapping
        graph_data = torch.load(gnn_dir / "graph_data.pt", map_location="cpu")
        node_names = json.loads((gnn_dir / "node_names.json").read_text())
        self.register_buffer("edge_index", graph_data["edge_index"].long())
        edge_weight = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight.float())
        else:
            self.edge_weight = None

        # Build Ensembl ID → node index lookup
        self._ensembl_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(node_names)
        }
        self._n_nodes = len(node_names)

        gnn_train = sum(p.numel() for p in self.gnn.parameters())
        print(
            f"[Node4-1-2-1] STRING_GNN: {gnn_train:,} params "
            f"(full fine-tune, discriminative lr={hp.lr/hp.gnn_lr_ratio:.2e})"
        )

        # ---- K=16 PPI Neighborhood Attention [NEW] ----
        # Build top-K neighbor table from PPI graph
        print(f"[Node4-1-2-1] Building top-{hp.ppi_K} PPI neighbor table …", flush=True)
        topk_ids, topk_logw, topk_mask = _build_topk_neighbors(
            self.edge_index, self.edge_weight, self._n_nodes, K=hp.ppi_K
        )
        # Initialize PPI attention module
        self.ppi_attn = PpiNeighborhoodAttention(
            d_gnn=GNN_HIDDEN, K=hp.ppi_K, attn_dim=hp.ppi_attn_dim
        )
        # Register neighbor buffers on the submodule so they move with the model
        self.ppi_attn.register_buffer("topk_ids",  topk_ids)
        self.ppi_attn.register_buffer("topk_logw", topk_logw)
        self.ppi_attn.register_buffer("topk_mask", topk_mask)

        n_ppi = sum(p.numel() for p in self.ppi_attn.parameters())
        print(
            f"[Node4-1-2-1] PPI attention: K={hp.ppi_K}, attn_dim={hp.ppi_attn_dim}, "
            f"{n_ppi:,} trainable params"
        )

        # ---- Gated Fusion ----
        self.fusion = GatedFusion(
            d_scf=SCF_HIDDEN,
            d_gnn=GNN_HIDDEN,
            d_out=FUSION_DIM,
            fusion_dropout=hp.fusion_dropout,
        )

        # ---- Restored single linear head (512→19920) ----
        # Returns to node4-1's proven 10.2M-parameter head.
        # The narrow 2-layer bottleneck in the parent caused underfitting.
        self.head = nn.Sequential(
            nn.Dropout(hp.head_dropout),
            nn.Linear(FUSION_DIM, N_CLASSES * N_GENES),
        )

        head_params = sum(p.numel() for p in self.head.parameters())
        print(
            f"[Node4-1-2-1] Head: single linear {FUSION_DIM}→{N_CLASSES*N_GENES}, "
            f"{head_params:,} params"
        )

        # ---- Class weights ----
        self.register_buffer("class_weights", get_class_weights())

        # Accumulators
        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []

    # ---- GNN node index lookup ----
    def _get_gnn_indices(self, pert_ids: List[str], device: torch.device) -> torch.Tensor:
        """Return LongTensor of node indices, 0 for unknowns."""
        indices = [self._ensembl_to_idx.get(pid, 0) for pid in pert_ids]
        return torch.tensor(indices, dtype=torch.long, device=device)

    # ---- forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids: List[str],
    ) -> torch.Tensor:
        B      = input_ids.shape[0]
        device = input_ids.device

        # 1. scFoundation: [B, nnz+2, 768] → mean pool → [B, 768]
        scf_out = self.scf(input_ids=input_ids, attention_mask=attention_mask)
        scf_emb = scf_out.last_hidden_state.float().mean(dim=1)   # [B, 768]

        # 2. STRING_GNN: run full graph → [N_nodes, 256]
        ew = self.edge_weight.to(device) if self.edge_weight is not None else None
        gnn_out   = self.gnn(
            edge_index  = self.edge_index.to(device),
            edge_weight = ew,
        )
        node_embs    = gnn_out.last_hidden_state               # [N, 256]
        node_indices = self._get_gnn_indices(pert_ids, device) # [B]

        # 3. K=16 PPI Neighborhood Attention [NEW]
        # Enriches each gene's GNN embedding with aggregated PPI neighbor context.
        # Adds local topology awareness beyond the GNN's own message-passing.
        gnn_emb = self.ppi_attn(node_embs, node_indices)      # [B, 256]

        # 4. Gated Fusion → [B, 512]
        fused = self.fusion(scf_emb, gnn_emb)

        # 5. Single linear head → [B, 3, G]
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return logits

    # ---- loss: weighted CE + label smoothing (no focal component) ----
    # Replaces parent's LabelSmoothingFocalLoss which diverged from the F1 metric.
    # F.cross_entropy with weight + label_smoothing aligns better with per-gene F1.
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        targets_flat = targets.reshape(-1)                      # [B*G]
        return F.cross_entropy(
            logits_flat,
            targets_flat,
            weight          = self.class_weights,
            label_smoothing = self.hparams.label_smoothing,
        )

    # ---- training / validation / test steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits.float(), dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        local_idx   = torch.cat(self._val_idx,   0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        all_preds   = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)

        # Collect pert_ids and symbols from all ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_pert_ids: List[List[str]] = [None] * self.trainer.world_size
            all_symbols:  List[List[str]] = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(all_pert_ids, self._test_pert_ids)
            torch.distributed.all_gather_object(all_symbols,  self._test_symbols)
            flat_pids = [p for rank_pids in all_pert_ids for p in rank_pids]
            flat_syms = [s for rank_syms in all_symbols  for s in rank_syms]
        else:
            flat_pids = list(self._test_pert_ids)
            flat_syms = list(self._test_symbols)

        if self.trainer.is_global_zero:
            n = all_preds.shape[0]
            rows = []
            for i in range(n):
                rows.append({
                    "idx":        flat_pids[i],
                    "input":      flat_syms[i],
                    "prediction": json.dumps(all_preds[i].float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node4-1-2-1] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    # ---- checkpoint: save only trainable params + buffers ----
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full:
                    trainable[key] = full[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full:
                trainable[key] = full[key]
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {trained}/{total} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- optimizer: discriminative LRs ----
    def configure_optimizers(self):
        hp = self.hparams
        scf_lr = hp.lr
        gnn_lr = hp.lr / hp.gnn_lr_ratio  # lower LR for GNN

        scf_params  = [p for p in self.scf.parameters()     if p.requires_grad]
        gnn_params  = list(self.gnn.parameters())
        head_params = (
            list(self.fusion.parameters())  +
            list(self.ppi_attn.parameters()) +  # PPI attention params at main LR
            list(self.head.parameters())
        )

        param_groups = [
            {"params": scf_params,  "lr": scf_lr, "name": "scf"},
            {"params": gnn_params,  "lr": gnn_lr, "name": "gnn"},
            {"params": head_params, "lr": scf_lr, "name": "head"},
        ]

        opt = torch.optim.AdamW(param_groups, weight_decay=hp.weight_decay)

        sch = WarmupCosineScheduler(
            opt,
            warmup_steps  = hp.warmup_epochs,
            total_steps   = hp.max_epochs,
            min_lr_ratio  = hp.min_lr_ratio,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval":  "epoch",
                "frequency": 1,
                "monitor":   "val/f1",
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node4-1-2-1 – scFoundation + STRING_GNN + PPI Neighborhood Attention"
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=8)
    parser.add_argument("--global-batch-size",   type=int,   default=64)
    parser.add_argument("--max-epochs",          type=int,   default=200)
    parser.add_argument("--lr",                  type=float, default=2e-4)
    parser.add_argument("--weight-decay",        type=float, default=2e-2)
    parser.add_argument("--scf-finetune-layers", type=int,   default=6,
                        dest="scf_finetune_layers")
    parser.add_argument("--head-dropout",        type=float, default=0.5)
    parser.add_argument("--fusion-dropout",      type=float, default=0.2,
                        dest="fusion_dropout")
    parser.add_argument("--gnn-lr-ratio",        type=float, default=5.0,
                        dest="gnn_lr_ratio")
    parser.add_argument("--label-smoothing",     type=float, default=0.1,
                        dest="label_smoothing")
    parser.add_argument("--warmup-epochs",       type=int,   default=10,
                        dest="warmup_epochs")
    parser.add_argument("--min-lr-ratio",        type=float, default=0.12,
                        dest="min_lr_ratio")
    parser.add_argument("--ppi-k",               type=int,   default=16,
                        dest="ppi_K")
    parser.add_argument("--ppi-attn-dim",        type=int,   default=64,
                        dest="ppi_attn_dim")
    parser.add_argument("--num-workers",         type=int,   default=4)
    parser.add_argument("--debug-max-step",      type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",        action="store_true", dest="fast_dev_run")
    parser.add_argument("--val-check-interval",  type=float, default=1.0,
                        dest="val_check_interval")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = lim_val = lim_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = FusionDEGModel(
        scf_finetune_layers = args.scf_finetune_layers,
        head_dropout        = args.head_dropout,
        fusion_dropout      = args.fusion_dropout,
        lr                  = args.lr,
        gnn_lr_ratio        = args.gnn_lr_ratio,
        weight_decay        = args.weight_decay,
        label_smoothing     = args.label_smoothing,
        warmup_epochs       = args.warmup_epochs,
        max_epochs          = args.max_epochs,
        min_lr_ratio        = args.min_lr_ratio,
        ppi_K               = args.ppi_K,
        ppi_attn_dim        = args.ppi_attn_dim,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath  = str(output_dir / "checkpoints"),
        filename = "best-{epoch:03d}-{val/f1:.4f}",
        monitor  = "val/f1", mode="max", save_top_k=1,
    )
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=25, min_delta=1e-4)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = n_gpus,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        max_steps               = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches     = lim_train,
        limit_val_batches       = lim_val,
        limit_test_batches      = lim_test,
        val_check_interval      = (
            args.val_check_interval
            if (args.debug_max_step is None and not fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node4-1-2-1] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
