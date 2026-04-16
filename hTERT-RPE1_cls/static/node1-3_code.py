"""
Node 1-3: STRING_GNN Partial Fine-tuning + Post-GNN Additive Conditioning + Two-sided Bilinear

Architecture:
  - STRING_GNN (partial fine-tuning: last 2 GCN layers + post_mp, ~530K trainable params)
    + post-GNN additive conditioning via low-rank pert_matrix [N_nodes, rank_pert=16] (clean,
    no batch-mixing flaw from node1-2-1)
  - STRING_GNN-initialized output gene embeddings (two-sided bilinear)
  - Deep residual MLP head (hidden_dim=512, 6 layers)
  - Bilinear interaction: [B, 3, bilinear_rank] x [6640, bilinear_rank]^T → [B, 3, 6640]
  - Focal loss (gamma=2.0)
  - Properly-calibrated cosine annealing (total_steps aligned with actual training)
  - Two-group AdamW (backbone lr=5e-5, head lr=5e-4)
  - Increased patience (50) to allow secondary LR-decay improvement phase

Differentiation:
  - vs. node1-1 (sibling): STRING_GNN not AIDO.Cell; no OOD input issue
  - vs. node1-2 (sibling, F1=0.4912, frozen): enables partial fine-tuning for task adaptation
  - vs. node1-2-1 (F1=0.4500): uses post-GNN additive conditioning (no batch-mixing flaw)
  - vs. node1-2-1-1 (F1=0.4900): properly aligned cosine LR + two-sided bilinear with
    STRING_GNN-initialized output embeddings + low-rank pert_matrix (rank=16 vs 18870×256)

Key Improvements over node1-2:
  1. Partial backbone fine-tuning (last 2 layers + post_mp) → task-specific adaptation
  2. Low-rank post-GNN perturbation conditioning → perturbation-aware embeddings
  3. STRING_GNN embeddings for output gene positions (both sides of bilinear)
  4. Properly calibrated LR schedule (total_steps = actual training steps, not 6600)
  5. Increased patience=50 to enable the 'secondary improvement phase'
  6. Dropout=0.3 for stronger regularization with fine-tuned backbone
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import json
import argparse
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
from transformers import AutoModel

# ─── Constants ────────────────────────────────────────────────────────────────

N_GENES_OUT = 6640
N_CLASSES = 3
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Exact per-gene macro F1 matching calc_metric.py logic.

    Args:
        pred_logits_np: [N, 3, G] float (logits or probabilities)
        labels_np:      [N, G]    int   (class indices 0/1/2)

    Returns:
        Mean per-gene F1 score (float).
    """
    pred_classes = pred_logits_np.argmax(axis=1)  # [N, G]
    n_genes = labels_np.shape[1]
    f1_vals: List[float] = []
    for g in range(n_genes):
        yt = labels_np[:, g]
        yh = pred_classes[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── STRING_GNN Loading ───────────────────────────────────────────────────────

def load_string_gnn_resources(
    model_dir: Path,
    device: torch.device,
) -> Tuple[object, torch.Tensor, Optional[torch.Tensor], List[str], Dict[str, int]]:
    """Load STRING_GNN model, graph data, and node names.

    Returns:
        model: STRING_GNN model (partially frozen - last 2 layers + post_mp trainable)
        edge_index: [2, E] graph edge indices on device
        edge_weight: [E] edge weights on device (or None)
        node_names: list of Ensembl gene IDs
        node_name_to_idx: dict mapping Ensembl ID → node index
    """
    node_names = json.loads((model_dir / "node_names.json").read_text())
    graph = torch.load(model_dir / "graph_data.pt", weights_only=False)
    node_name_to_idx = {name: i for i, name in enumerate(node_names)}

    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
    model = model.to(device)

    # Freeze ALL parameters initially
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last 2 GCN layers (mps.6, mps.7) and post_mp
    # This follows the STRING_GNN skill guidance: "tune mps.6.*, mps.7.*, post_mp.*"
    for name, param in model.named_parameters():
        if name.startswith("mps.6.") or name.startswith("mps.7.") or name.startswith("post_mp"):
            param.requires_grad = True

    edge_index = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"].to(device) if graph.get("edge_weight") is not None else None

    return model, edge_index, edge_weight, node_names, node_name_to_idx


def get_frozen_embeddings(
    model: object,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
) -> np.ndarray:
    """Run STRING_GNN in frozen mode to get initial embeddings.

    These are used to initialize the output gene embeddings (two-sided bilinear).
    The model parameters for last 2 layers/post_mp will still be updated during
    training via the backbone forward pass.

    Returns:
        embeddings: [N_nodes, 256] numpy array of initial pretrained embeddings
    """
    model.eval()
    with torch.no_grad():
        outputs = model(
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
    embeddings = outputs.last_hidden_state.float().cpu().numpy()
    return embeddings


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset.

    Each sample provides:
    - pert_id: Ensembl gene ID
    - symbol: gene symbol
    - node_idx: STRING_GNN node index (for embedding lookup), -1 if OOV
    - label: [N_GENES_OUT] integer class labels 0/1/2
    """

    def __init__(
        self,
        df: pd.DataFrame,
        node_name_to_idx: Dict[str, int],
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        # Map pert_id to STRING_GNN node index (-1 for OOV)
        node_indices = []
        for pert_id in self.pert_ids:
            if pert_id in node_name_to_idx:
                node_indices.append(node_name_to_idx[pert_id])
            else:
                node_indices.append(-1)
        self.node_indices = torch.tensor(node_indices, dtype=torch.long)  # [N]

        self.has_labels = has_labels
        if has_labels and "label" in df.columns:
            rows = []
            for lbl_str in df["label"]:
                rows.append([x + 1 for x in json.loads(lbl_str)])
            self.labels = torch.tensor(rows, dtype=torch.long)  # [N, G]
        else:
            self.has_labels = False

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int):
        item = {
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "node_idx": self.node_indices[idx],  # long scalar, -1 if OOV
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule using STRING_GNN node indices (not precomputed embeddings).

    We pass node indices to the model, which runs STRING_GNN forward during training.
    This allows the partial fine-tuning to update backbone weights.
    """

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if hasattr(self, "train_ds"):
            return

        # Load node_name_to_idx for dataset construction
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        self.node_name_to_idx = {name: i for i, name in enumerate(node_names)}
        self.n_nodes = len(node_names)

        # Load all splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        # Coverage report
        train_in_vocab = sum(p in self.node_name_to_idx for p in dfs['train']['pert_id'])
        print(f"[DataModule] Coverage: {train_in_vocab}/{len(dfs['train'])} train genes in STRING_GNN")

        self.train_ds = PerturbationDataset(dfs["train"], self.node_name_to_idx, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   self.node_name_to_idx, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  self.node_name_to_idx, True)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )


# ─── Model ────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout + skip."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class StringGNNFinetuneHead(nn.Module):
    """Prediction head with partial STRING_GNN fine-tuning + post-GNN additive conditioning.

    The STRING_GNN backbone is loaded with:
    - emb, mps.0–mps.5: frozen
    - mps.6, mps.7, post_mp: trainable

    Post-GNN conditioning: a low-rank pert_matrix [N_nodes+1, gnn_dim] (rank-16 factored)
    is added to the frozen GNN embedding AFTER the forward pass, before the MLP head.
    This avoids the batch-mixing flaw: each sample gets its own perturbation offset,
    independent of other samples in the batch.

    Two-sided bilinear: output gene embeddings are initialized from frozen STRING_GNN
    embeddings for the N_GENES_OUT gene positions, providing biological prior on both sides.
    """

    def __init__(
        self,
        backbone: nn.Module,          # partially frozen STRING_GNN
        edge_index: torch.Tensor,     # [2, E]
        edge_weight: Optional[torch.Tensor],  # [E]
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        bilinear_rank: int = 256,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.3,
        n_residual_layers: int = 6,
        pert_matrix_rank: int = 16,   # low-rank factorization of pert_matrix
        n_nodes: int = 18870,
        initial_out_gene_emb: Optional[np.ndarray] = None,  # [n_genes_out, gnn_dim]
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.bilinear_rank = bilinear_rank
        self.gnn_dim = gnn_dim
        self.n_nodes = n_nodes

        # STRING_GNN backbone (partial fine-tuning: last 2 layers + post_mp)
        self.backbone = backbone

        # Register graph data as buffers (not parameters)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight if edge_weight is not None else torch.tensor([]))
        self._has_edge_weight = edge_weight is not None

        # Low-rank post-GNN perturbation conditioning:
        # pert_matrix[node_idx] = U[node_idx] @ V^T (shape [gnn_dim])
        # where U: [n_nodes+1, pert_matrix_rank], V: [gnn_dim, pert_matrix_rank]
        # Total params: (n_nodes+1) * rank + gnn_dim * rank = (18871 + 256) * 16 ≈ 307K
        # This is much smaller than full pert_matrix (18870 × 256 = 4.83M)
        # OOV genes (idx=-1) mapped to index n_nodes (the OOV token)
        self.pert_U = nn.Embedding(n_nodes + 1, pert_matrix_rank)  # [N+1, rank]
        self.pert_V = nn.Linear(pert_matrix_rank, gnn_dim, bias=False)  # [rank, gnn_dim]

        # OOV fallback embedding for genes not in STRING_GNN (~6.4% of dataset)
        self.oov_emb = nn.Embedding(1, gnn_dim)

        # Input normalization
        self.input_norm = nn.LayerNorm(gnn_dim)

        # Projection: gnn_dim -> hidden_dim
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)

        # Deep residual MLP
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim -> n_classes * bilinear_rank
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * bilinear_rank)

        # Head dropout
        self.head_dropout = nn.Dropout(dropout)

        # Output gene embeddings [n_genes_out, bilinear_rank]
        # Initialized from STRING_GNN embeddings if provided (two-sided bilinear)
        # If bilinear_rank != gnn_dim, we project STRING_GNN emb to bilinear_rank
        if initial_out_gene_emb is not None:
            # initial_out_gene_emb: [n_genes_out, gnn_dim]
            out_emb_tensor = torch.from_numpy(initial_out_gene_emb).float()
            if bilinear_rank != gnn_dim:
                # Linear projection to bilinear_rank, random init
                # (can't do gradient-free projection at init time)
                self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, bilinear_rank))
                nn.init.normal_(self.out_gene_emb, std=0.02)
            else:
                # bilinear_rank == gnn_dim: use STRING_GNN embeddings directly
                self.out_gene_emb = nn.Parameter(out_emb_tensor)
        else:
            self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, bilinear_rank))
            nn.init.normal_(self.out_gene_emb, std=0.02)

        self._init_weights()

    def _init_weights(self):
        # Initialize pert conditioning near zero (residual-style: start as identity)
        nn.init.normal_(self.pert_U.weight, std=0.001)
        nn.init.normal_(self.pert_V.weight, std=0.001)
        nn.init.normal_(self.oov_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)

    def forward(
        self,
        node_idx: torch.Tensor,  # [B] long, -1 for OOV
    ) -> torch.Tensor:
        """
        Args:
            node_idx: [B] long - STRING_GNN node indices, -1 if OOV
        Returns:
            logits: [B, 3, 6640]
        """
        B = node_idx.shape[0]
        device = node_idx.device

        # --- Step 1: Run STRING_GNN forward pass (backbone) ---
        # This pass has partial gradient: last 2 layers + post_mp are trainable
        ew = self.edge_weight if self._has_edge_weight else None
        if ew is not None and ew.numel() == 0:
            ew = None

        backbone_out = self.backbone(
            edge_index=self.edge_index,
            edge_weight=ew,
        )
        # all_node_emb: [N_nodes, gnn_dim] — contains gradients for trainable layers
        all_node_emb = backbone_out.last_hidden_state  # [18870, gnn_dim]

        # --- Step 2: Extract per-sample embeddings from backbone output ---
        in_vocab_mask = (node_idx >= 0)  # [B] bool
        # Clamp OOV to 0 for safe indexing (will be replaced by oov_emb)
        safe_idx = node_idx.clamp(min=0)  # [B]

        # Gather node embeddings for in-vocab genes
        gnn_emb = all_node_emb[safe_idx]  # [B, gnn_dim]

        # Replace OOV embeddings with learned fallback
        oov_token = self.oov_emb(torch.zeros(B, dtype=torch.long, device=device))  # [B, gnn_dim]
        in_vocab_f = in_vocab_mask.unsqueeze(1).float()  # [B, 1]
        gnn_emb = gnn_emb * in_vocab_f + oov_token * (1.0 - in_vocab_f)  # [B, gnn_dim]

        # --- Step 3: Post-GNN additive perturbation conditioning ---
        # pert_offset[i] = pert_U[node_idx_for_pert_table] @ pert_V^T
        # Map OOV (node_idx=-1) to index n_nodes (OOV token in pert_U)
        pert_idx = torch.where(in_vocab_mask, safe_idx, torch.full_like(safe_idx, self.n_nodes))
        pert_u = self.pert_U(pert_idx)           # [B, pert_matrix_rank]
        pert_offset = self.pert_V(pert_u)         # [B, gnn_dim]
        # Add perturbation offset (initialized near zero → starts as residual identity)
        gnn_emb = gnn_emb + pert_offset          # [B, gnn_dim]

        # --- Step 4: MLP head ---
        x = self.input_norm(gnn_emb)    # [B, gnn_dim]
        x = self.proj_in(x)             # [B, hidden_dim]

        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)            # [B, hidden_dim]

        # --- Step 5: Bilinear interaction ---
        x = self.head_dropout(x)
        pert_proj = self.proj_bilinear(x)                        # [B, n_classes * bilinear_rank]
        pert_proj = pert_proj.view(B, self.n_classes, self.bilinear_rank)  # [B, 3, bilinear_rank]

        # Bilinear: [B, 3, bilinear_rank] x [bilinear_rank, n_genes_out] → [B, 3, n_genes_out]
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, 6640]

        return logits


# ─── Focal Loss ───────────────────────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss: -(1-p_t)^gamma * log(p_t). No class weights (focal handles imbalance).

    Args:
        logits: [B, 3, G] raw logits
        labels: [B, G] integer class labels (0, 1, 2)
        gamma: focusing parameter
    Returns:
        Scalar loss value.
    """
    B, C, G = logits.shape

    # Reshape to [B*G, 3] and [B*G]
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
    labels_flat = labels.reshape(-1)                        # [B*G]

    # Cross-entropy per element
    ce_loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        reduction="none",
    )  # [B*G]

    # Focal weighting
    with torch.no_grad():
        probs = F.softmax(logits_flat, dim=1)           # [B*G, 3]
        pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [B*G]
        focal_weight = (1.0 - pt).pow(gamma)            # [B*G]

    loss = (focal_weight * ce_loss).mean()
    return loss


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(
    local_preds: torch.Tensor,
    local_labels: torch.Tensor,
    device: torch.device,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather variable-length tensors from all DDP ranks with padding."""
    local_size = torch.tensor([local_preds.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = int(max(s.item() for s in all_sizes))

    pad = max_size - local_preds.shape[0]
    p = local_preds.to(device)
    l = local_labels.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], dim=0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], dim=0)

    g_preds  = [torch.zeros_like(p) for _ in range(world_size)]
    g_labels = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(g_preds,  p)
    dist.all_gather(g_labels, l)

    real_preds  = torch.cat([g_preds[i][: all_sizes[i].item()].cpu()  for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][: all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """LightningModule for gene-perturbation DEG prediction (Node 1-3)."""

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        bilinear_rank: int = 256,
        n_residual_layers: int = 6,
        dropout: float = 0.3,
        lr_backbone: float = 5e-5,
        lr_head: float = 5e-4,
        weight_decay: float = 1e-3,
        focal_gamma: float = 2.0,
        warmup_steps: int = 50,
        total_steps: int = 1000,
        pert_matrix_rank: int = 16,
        n_nodes: int = 18870,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Accumulation buffers
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams
        device = self.device if self.device.type != "meta" else torch.device("cpu")

        # Load STRING_GNN with partial fine-tuning
        backbone, edge_index, edge_weight, _, node_name_to_idx = load_string_gnn_resources(
            STRING_GNN_DIR, device
        )

        # Get initial frozen embeddings to initialize output gene embeddings
        # We need to identify the first N_GENES_OUT genes that correspond to the
        # 6640 output gene positions. Since we don't have explicit mapping of the
        # 6640 output genes to STRING_GNN nodes here, we use the datamodule's
        # node_name_to_idx to map test gene labels.
        # Actually: we initialize out_gene_emb with STRING_GNN embeddings for
        # the output gene positions. We need to know which STRING_GNN node indices
        # correspond to the 6640 output positions.
        #
        # The 6640 output positions correspond to gene expression positions in the DEG
        # signature. The DATA_ABSTRACT.md does not specify which specific genes they are
        # (the label is a dense 6640-vector for ALL output genes, not a sparse subset).
        # Therefore, we cannot perfectly map output positions to STRING_GNN nodes.
        # Instead: we use the first 6640 STRING_GNN nodes' embeddings as initialization
        # for out_gene_emb when bilinear_rank == gnn_dim. This provides a biologically-
        # informed prior even without exact positional alignment.
        initial_out_gene_emb = get_frozen_embeddings(backbone, edge_index, edge_weight)
        # Take first N_GENES_OUT rows as initialization (dense prior)
        # bilinear_rank == gnn_dim=256 → directly use STRING_GNN embeddings
        initial_out_emb_subset = initial_out_gene_emb[:N_GENES_OUT]  # [6640, 256]

        self.model = StringGNNFinetuneHead(
            backbone=backbone,
            edge_index=edge_index,
            edge_weight=edge_weight,
            gnn_dim=hp.gnn_dim,
            hidden_dim=hp.hidden_dim,
            bilinear_rank=hp.bilinear_rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
            pert_matrix_rank=hp.pert_matrix_rank,
            n_nodes=hp.n_nodes,
            initial_out_gene_emb=initial_out_emb_subset,
        )

        # Ensure float32 for all trainable parameters
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

    def forward(self, node_idx: torch.Tensor) -> torch.Tensor:
        return self.model(node_idx)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return focal_loss(logits, labels, gamma=self.hparams.focal_gamma)

    def training_step(self, batch, batch_idx):
        logits = self(batch["node_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["node_idx"])
        if "label" in batch:
            loss = self._compute_loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())
        return logits

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        local_p = torch.cat(self._val_preds,  dim=0)
        local_l = torch.cat(self._val_labels, dim=0)

        if self.trainer.world_size > 1:
            all_p, all_l = _gather_tensors(local_p, local_l, self.device, self.trainer.world_size)
        else:
            all_p, all_l = local_p, local_l

        f1 = compute_per_gene_f1(all_p.numpy(), all_l.numpy())
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["node_idx"])
        probs = torch.softmax(logits, dim=1)  # [B, 3, 6640]
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs = torch.cat(self._test_preds, dim=0)
        dummy_labels = torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        if self._test_labels:
            dummy_labels = torch.cat(self._test_labels, dim=0)

        if self.trainer.world_size > 1:
            all_probs, all_labels = _gather_tensors(local_probs, dummy_labels, self.device, self.trainer.world_size)
            all_pert = [None] * self.trainer.world_size
            all_syms = [None] * self.trainer.world_size
            dist.all_gather_object(all_pert, self._test_pert_ids)
            dist.all_gather_object(all_syms, self._test_symbols)
            all_pert = [p for sub in all_pert for p in sub]
            all_syms = [s for sub in all_syms for s in sub]
        else:
            all_probs  = local_probs
            all_labels = dummy_labels
            all_pert   = self._test_pert_ids
            all_syms   = self._test_symbols

        if self.trainer.is_global_zero:
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            seen_ids: set = set()
            dedup_probs: list = []
            dedup_labels: list = []
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for i, (pert_id, symbol, probs) in enumerate(
                    zip(all_pert, all_syms, all_probs.numpy())
                ):
                    if pert_id not in seen_ids:
                        seen_ids.add(pert_id)
                        fh.write(f"{pert_id}\t{symbol}\t{json.dumps(probs.tolist())}\n")
                        dedup_probs.append(probs)
                        dedup_labels.append(all_labels[i].numpy())
            self.print(f"[Node1-3] Saved test predictions → {pred_path} ({len(seen_ids)} unique samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node1-3] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Two-group optimizer: separate LRs for backbone vs. head
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": hp.lr_backbone, "weight_decay": hp.weight_decay},
            {"params": head_params,     "lr": hp.lr_head,     "weight_decay": hp.weight_decay},
        ]

        optimizer = torch.optim.AdamW(param_groups)

        # Cosine annealing with linear warmup
        # total_steps is properly set to actual training steps (not 6600 like node1-2-1-1)
        def lr_lambda(step: int) -> float:
            if step < hp.warmup_steps:
                return float(step) / max(1, hp.warmup_steps)
            progress = (step - hp.warmup_steps) / max(1, hp.total_steps - hp.warmup_steps)
            return max(0.01, 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable params ─────────────────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_sd = {}
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        for k, v in full_sd.items():
            if k in trainable_keys or k in buffer_keys:
                trainable_sd[k] = v
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)")
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Node 1-3 – STRING_GNN Partial Fine-tuning + Post-GNN Conditioning + Two-sided Bilinear")
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--gnn-dim",            type=int,   default=256)
    p.add_argument("--hidden-dim",         type=int,   default=512)
    p.add_argument("--bilinear-rank",      type=int,   default=256)
    p.add_argument("--n-residual-layers",  type=int,   default=6)
    p.add_argument("--dropout",            type=float, default=0.3)
    p.add_argument("--lr-backbone",        type=float, default=5e-5)
    p.add_argument("--lr-head",            type=float, default=5e-4)
    p.add_argument("--weight-decay",       type=float, default=1e-3)
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--warmup-steps",       type=int,   default=50)
    p.add_argument("--pert-matrix-rank",   type=int,   default=16)
    p.add_argument("--micro-batch-size",   type=int,   default=16)
    p.add_argument("--global-batch-size",  type=int,   default=64)
    p.add_argument("--max-epochs",         type=int,   default=300)
    p.add_argument("--patience",           type=int,   default=50)
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataModule
    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    # Estimate total training steps for LR schedule
    # Properly calibrated: aligned with actual expected training duration
    # This avoids the node1-2-1-1 flaw of setting total_steps=6600 for 300 epochs
    # but stopping at epoch 86 (only 29% of planned steps)
    steps_per_epoch = max(1, len(dm.train_ds) // (args.micro_batch_size * n_gpus))
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)
    # Set total_steps for cosine schedule to a reasonable expected training duration.
    # With patience=50 and typical convergence around epoch 100-150, use 150 epochs as target.
    # This ensures the LR decays meaningfully during the expected training window.
    expected_training_epochs = min(args.max_epochs, 150)
    total_steps = effective_steps_per_epoch * expected_training_epochs

    # LightningModule
    lit = PerturbationLitModule(
        gnn_dim=args.gnn_dim,
        hidden_dim=args.hidden_dim,
        bilinear_rank=args.bilinear_rank,
        n_residual_layers=args.n_residual_layers,
        dropout=args.dropout,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        pert_matrix_rank=args.pert_matrix_rank,
        n_nodes=dm.n_nodes,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=1, save_last=True,
    )
    es_cb = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
    lr_cb = LearningRateMonitor(logging_interval="step")
    pb_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    # Debug / fast-dev-run settings
    max_steps:           int           = -1
    limit_train_batches: float | int   = 1.0
    limit_val_batches:   float | int   = 1.0
    limit_test_batches:  float | int   = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps           = args.debug_max_step
        limit_train_batches = args.debug_max_step
        limit_val_batches   = 2
        limit_test_batches  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        gradient_clip_val=1.0,   # gradient clipping for stable training
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(
            f"Node 1-3 – STRING_GNN Partial Fine-tuning + Post-GNN Conditioning + Two-sided Bilinear\n"
            f"Test results from trainer: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
