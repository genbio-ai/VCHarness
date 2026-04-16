"""
Node 2-1-1-1-1-1 — Frozen STRING_GNN + Deep Bilinear MLP Head
             with Pre-Initialized Output Gene Embeddings and MuonWithAuxAdam

Architecture (returns to tree-best foundation, adds pre-initialized out_gene_emb):
  - STRING_GNN backbone: fully frozen (precomputed embeddings as fixed features)
  - Output gene embedding initialization: STRING_GNN embeddings for label gene positions
      * label_genes.txt maps 6640 positions to Ensembl IDs
      * For genes in STRING_GNN vocab: initialize out_gene_emb from pretrained embeddings
      * For OOV output genes: random init (std=0.02)
      * Then project 256-dim STRING_GNN embeddings → rank=512 via trainable linear
  - 6-layer deep residual bilinear MLP head (same proven design as node1-2 and best nodes)
  - MuonWithAuxAdam optimizer (proven best optimizer in tree: F1=0.5035 nodes)
  - Class-weighted focal loss: gamma=2.0, weights=[2.0, 0.5, 4.0] for (down, neutral, up)
  - Single cosine cycle schedule calibrated to ~80 epochs (T_max≈880 steps)

Key differences from parent (node2-1-1-1, F1=0.4518):
  1. Return to fully frozen STRING_GNN (no partial fine-tuning)
  2. Remove dead-gradient LowRankPertMatrix entirely
  3. Remove over-aggressive class weights [12.28, 1.0, 33.33] → use [2.0, 0.5, 4.0]
  4. Use MuonWithAuxAdam optimizer instead of pure AdamW (proven +0.02 gain)
  5. Pre-initialize output gene embeddings from STRING_GNN vocab for 6640 output genes
  6. Correct cosine schedule calibration

Key differences from sibling (node2-1-1-1-1, F1=0.4761):
  1. MuonWithAuxAdam optimizer instead of pure AdamW (expected +0.02 improvement)
  2. Pre-initialized out_gene_emb from STRING_GNN (biological prior for output space geometry)
  3. rank=512 instead of rank=256 (more capacity for bilinear interaction)
  4. No inductive conditioning MLP (which provided negligible signal per feedback)
  5. Class weights [2.0, 0.5, 4.0] (tree-validated) not removed entirely

Differentiation from sibling node2-1-1-1-1 (inductive conditioning approach):
  - sibling: frozen STRING + conditioning MLP (adds 66K params computing offset from PPI emb)
  - this node: frozen STRING + pre-init out_gene_emb (improves output space geometry directly)
  - Key insight: inductive MLP conditioning on frozen embeddings has failed in 2 attempts;
    the feedback suggests improving the bilinear head's output gene embedding initialization
    as a cleaner alternative that doesn't compete with the head learning pathway
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # required for deterministic=True with CUDA >= 10.2

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
LABEL_GENES_FILE = Path("data/label_genes.txt")

# Class weights from tree-validated approach (node1-1-2-1-1 onward, F1=0.5023):
# These are softer weights [2.0, 0.5, 4.0] that are empirically better than
# inverse-freq weights [12.28, 1.0, 33.33] for this task.
# Note: using neutral weight < 1 has been effective; "down" and "up" are upweighted.
CLASS_WEIGHTS_LIST = [2.0, 0.5, 4.0]  # [down, neutral, up]


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


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """Perturbation DEG dataset.

    Stores precomputed STRING_GNN embeddings for fast batch retrieval.
    Embeddings are computed once from the frozen GNN in DataModule.setup().
    """

    def __init__(
        self,
        df: pd.DataFrame,
        gnn_embeddings: np.ndarray,        # [N_nodes, 256] - frozen GNN embeddings
        node_name_to_idx: Dict[str, int],
        embed_dim: int = 256,
        has_labels: bool = True,
    ):
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.has_labels = has_labels

        n_samples = len(df)
        embeddings = np.zeros((n_samples, embed_dim), dtype=np.float32)

        for i, pert_id in enumerate(self.pert_ids):
            if pert_id in node_name_to_idx:
                node_idx = node_name_to_idx[pert_id]
                embeddings[i] = gnn_embeddings[node_idx]
            # OOV: left as zeros — model handles via oov_embedding parameter

        self.embeddings = torch.from_numpy(embeddings)   # [N, 256]
        # Track in-vocab mask for OOV gene handling
        in_vocab = [pert_id in node_name_to_idx for pert_id in self.pert_ids]
        self.in_vocab = torch.tensor(in_vocab, dtype=torch.bool)  # [N]

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
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
            "embedding":  self.embeddings[idx],    # [256]
            "in_vocab":   self.in_vocab[idx],       # bool
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


# ─── DataModule ───────────────────────────────────────────────────────────────

class PerturbationDataModule(pl.LightningDataModule):
    """Single-fold DataModule using precomputed frozen STRING_GNN embeddings."""

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
        # Guard against double initialization
        if hasattr(self, "train_ds"):
            return

        # Run STRING_GNN ONCE in frozen mode to get base embeddings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[DataModule] Computing STRING_GNN base embeddings (frozen forward pass)...")
        node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", weights_only=False)

        model = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        edge_index = graph["edge_index"].to(device)
        edge_weight = graph["edge_weight"].to(device) if graph.get("edge_weight") is not None else None

        with torch.no_grad():
            outputs = model(edge_index=edge_index, edge_weight=edge_weight)

        gnn_embeddings = outputs.last_hidden_state.float().cpu().numpy()  # [N_nodes, 256]
        node_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(node_names)}

        self.gnn_embeddings = gnn_embeddings
        self.node_name_to_idx = node_name_to_idx
        self.n_gnn_nodes = len(node_names)

        del model
        torch.cuda.empty_cache()

        print(f"[DataModule] STRING_GNN base embeddings shape: {gnn_embeddings.shape}")

        # Load all splits
        dfs: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")

        n_train_cov = sum(p in node_name_to_idx for p in dfs["train"]["pert_id"])
        print(f"[DataModule] Coverage: {n_train_cov}/{len(dfs['train'])} train genes in STRING_GNN")

        embed_dim = gnn_embeddings.shape[1]
        self.train_ds = PerturbationDataset(dfs["train"], gnn_embeddings, node_name_to_idx, embed_dim, True)
        self.val_ds   = PerturbationDataset(dfs["val"],   gnn_embeddings, node_name_to_idx, embed_dim, True)
        self.test_ds  = PerturbationDataset(dfs["test"],  gnn_embeddings, node_name_to_idx, embed_dim, True)

        # Also load label gene mappings for output gene embedding initialization
        label_genes_path = self.data_dir / "label_genes.txt"
        self.label_gene_ids: List[str] = []
        if label_genes_path.exists():
            with open(label_genes_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        self.label_gene_ids.append(parts[0])  # Ensembl ID
        print(f"[DataModule] Loaded {len(self.label_gene_ids)} label gene IDs from label_genes.txt")

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


# ─── Model Components ─────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual MLP block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout + skip."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.2):
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


class GNNBilinearHead(nn.Module):
    """Prediction head with bilinear interaction.

    Left side: perturbation GNN embedding -> Deep MLP -> [B, 3, rank]
    Right side: output gene embeddings [n_genes_out, rank] (pre-initialized from STRING_GNN)
    Interaction: einsum("bcr,gr->bcg") -> logits [B, 3, n_genes_out]

    Key improvement: out_gene_emb initialized from STRING_GNN embeddings for 6640 output genes,
    providing biological priors for the output space geometry without adding conditioning complexity.
    The 256-dim STRING_GNN embeddings are projected to rank=512 via a pre-computed linear mapping.
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_genes_out: int = N_GENES_OUT,
        n_classes: int = N_CLASSES,
        dropout: float = 0.2,
        n_residual_layers: int = 6,
        out_gene_emb_init: Optional[np.ndarray] = None,   # [n_genes_out, gnn_dim] or None
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_genes_out = n_genes_out
        self.rank = rank

        # Input normalization
        self.input_norm = nn.LayerNorm(gnn_dim)

        # Learned OOV embedding for genes not in STRING_GNN vocab
        self.oov_embedding = nn.Parameter(torch.zeros(gnn_dim))
        nn.init.normal_(self.oov_embedding, std=0.02)

        # Projection: gnn_dim -> hidden_dim
        self.proj_in = nn.Linear(gnn_dim, hidden_dim)

        # Deep residual MLP
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, expand=4, dropout=dropout)
             for _ in range(n_residual_layers)]
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Bilinear projection: hidden_dim -> n_classes * rank
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * rank)

        # Output gene embeddings: [n_genes_out, rank]
        # Initialize from STRING_GNN embeddings if available, otherwise random
        self.out_gene_emb = nn.Parameter(torch.empty(n_genes_out, rank))

        # Head dropout
        self.head_dropout = nn.Dropout(dropout)

        self._init_weights(out_gene_emb_init)

    def _init_weights(self, out_gene_emb_init: Optional[np.ndarray]):
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)

        if out_gene_emb_init is not None and out_gene_emb_init.shape[0] == self.n_genes_out:
            # Project 256-dim STRING_GNN embeddings to rank using random linear projection
            # This preserves the relative geometry of the STRING_GNN embedding space
            # while adapting to the required rank dimension
            gnn_dim = out_gene_emb_init.shape[1]
            if gnn_dim != self.rank:
                # Random projection preserving geometry (not learnable -- just initialization)
                proj_matrix = np.random.randn(gnn_dim, self.rank).astype(np.float32)
                proj_matrix /= np.sqrt(gnn_dim)  # normalize
                projected = out_gene_emb_init @ proj_matrix  # [n_genes_out, rank]
                # Scale to std=0.02 as used in random init (keeps values small)
                projected_std = projected.std()
                if projected_std > 0:
                    projected = projected * (0.02 / projected_std)
                self.out_gene_emb.data = torch.from_numpy(projected)
                print(f"[GNNBilinearHead] Initialized out_gene_emb from STRING_GNN embeddings "
                      f"(projected {gnn_dim}→{self.rank}, std={self.out_gene_emb.data.std():.4f})")
            else:
                emb_tensor = torch.from_numpy(out_gene_emb_init.astype(np.float32))
                emb_std = emb_tensor.std().item()
                if emb_std > 0:
                    emb_tensor = emb_tensor * (0.02 / emb_std)
                self.out_gene_emb.data = emb_tensor
                print(f"[GNNBilinearHead] Initialized out_gene_emb from STRING_GNN embeddings "
                      f"(same dim={gnn_dim}, scaled to std=0.02)")
        else:
            nn.init.normal_(self.out_gene_emb, std=0.02)
            print(f"[GNNBilinearHead] Initialized out_gene_emb randomly (std=0.02)")

    def forward(self, gnn_emb: torch.Tensor, in_vocab: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gnn_emb: [B, gnn_dim] - precomputed frozen STRING_GNN embeddings (zeros for OOV)
            in_vocab: [B] bool - True if gene is in STRING_GNN vocab
        Returns:
            logits: [B, 3, n_genes_out]
        """
        B = gnn_emb.shape[0]

        # Replace OOV gene embeddings with learned OOV parameter
        if (~in_vocab).any():
            oov_emb = self.oov_embedding.unsqueeze(0).expand(B, -1)  # [B, gnn_dim]
            gnn_emb = gnn_emb.clone()
            gnn_emb[~in_vocab] = oov_emb[~in_vocab]

        x = self.input_norm(gnn_emb)
        x = self.proj_in(x)   # [B, hidden_dim]

        for blk in self.res_blocks:
            x = blk(x)
        x = self.norm_out(x)   # [B, hidden_dim]

        x = self.head_dropout(x)
        pert_proj = self.proj_bilinear(x)                          # [B, n_classes * rank]
        pert_proj = pert_proj.view(B, self.n_classes, self.rank)   # [B, 3, rank]

        # Bilinear interaction with output gene embeddings
        logits = torch.einsum("bcr,gr->bcg", pert_proj, self.out_gene_emb)  # [B, 3, n_genes_out]
        return logits


# ─── Focal Loss ───────────────────────────────────────────────────────────────

def focal_loss_with_weights(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Focal loss with optional class weights.

    Args:
        logits:        [B, C, G] logits
        labels:        [B, G] long class indices (0/1/2)
        gamma:         focal exponent
        class_weights: [C] per-class weights (optional)
    Returns:
        Scalar loss
    """
    B, C, G = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, 3]
    labels_flat = labels.reshape(-1)                        # [B*G]

    ce_loss = F.cross_entropy(
        logits_flat, labels_flat,
        weight=class_weights,
        reduction="none"
    )  # [B*G]

    with torch.no_grad():
        probs = F.softmax(logits_flat, dim=1)
        pt = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt).pow(gamma)

    return (focal_weight * ce_loss).mean()


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

    real_preds  = torch.cat([g_preds[i][:all_sizes[i].item()].cpu()  for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][:all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


# ─── LightningModule ──────────────────────────────────────────────────────────

class PerturbationLitModule(pl.LightningModule):
    """LightningModule for gene-perturbation DEG prediction.

    Frozen STRING_GNN backbone (precomputed embeddings) + deep bilinear MLP head.
    Pre-initialized output gene embeddings from STRING_GNN for biological geometry.

    Optimizer: MuonWithAuxAdam
    - Muon (lr=0.005) for hidden weight matrices in ResidualBlocks (2D, non-embedding)
    - AdamW (lr=5e-4) for norms, biases, projection weights, output gene embeddings

    This matches the optimizer configuration from tree-best nodes (F1=0.5035):
    node1-1-2-1-1-1-1 and adjacent nodes with MuonWithAuxAdam.
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        hidden_dim: int = 512,
        rank: int = 512,
        n_residual_layers: int = 6,
        dropout: float = 0.2,
        lr_muon: float = 0.005,
        lr_adamw: float = 5e-4,
        weight_decay: float = 2e-3,
        focal_gamma: float = 2.0,
        use_class_weights: bool = True,
        warmup_steps: int = 50,
        total_steps: int = 880,
        out_gene_emb_init: Optional[np.ndarray] = None,   # passed from main()
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["out_gene_emb_init"])
        # Store init data (not saved in hparams to avoid serialization issues)
        self._out_gene_emb_init = out_gene_emb_init
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams

        # Build bilinear prediction head with pre-initialized output gene embeddings
        self.head = GNNBilinearHead(
            gnn_dim=hp.gnn_dim,
            hidden_dim=hp.hidden_dim,
            rank=hp.rank,
            n_genes_out=N_GENES_OUT,
            n_classes=N_CLASSES,
            dropout=hp.dropout,
            n_residual_layers=hp.n_residual_layers,
            out_gene_emb_init=self._out_gene_emb_init,
        )

        # Class weights for focal loss
        if hp.use_class_weights:
            cw = torch.tensor(CLASS_WEIGHTS_LIST, dtype=torch.float32)
            self.register_buffer("class_weights", cw)
            print(f"[Setup] Using class weights (tree-validated [2.0, 0.5, 4.0]): "
                  f"down={CLASS_WEIGHTS_LIST[0]:.1f}, "
                  f"neutral={CLASS_WEIGHTS_LIST[1]:.1f}, up={CLASS_WEIGHTS_LIST[2]:.1f}")
        else:
            self.class_weights = None

        # Cast all trainable parameters to float32 for stable optimization
        for p in self.head.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        print(f"[Setup] Head trainable params: {head_trainable:,}")

    def forward(self, embedding: torch.Tensor, in_vocab: torch.Tensor) -> torch.Tensor:
        logits = self.head(embedding, in_vocab)  # [B, 3, 6640]
        return logits

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cw = None
        if self.hparams.use_class_weights and hasattr(self, "class_weights") and self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
        return focal_loss_with_weights(logits, labels, gamma=self.hparams.focal_gamma, class_weights=cw)

    def training_step(self, batch, batch_idx):
        logits = self(
            batch["embedding"].float(),
            batch["in_vocab"],
        )
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(
            batch["embedding"].float(),
            batch["in_vocab"],
        )
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
        logits = self(
            batch["embedding"].float(),
            batch["in_vocab"],
        )
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

            self.print(f"[Node2-1-1-1-1-1] Saved test predictions → {pred_path} ({len(seen_ids)} samples)")

            if dedup_probs and dedup_labels:
                dedup_probs_np  = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node2-1-1-1-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams

        try:
            from muon import MuonWithAuxAdam

            # Separate parameters by type for MuonWithAuxAdam:
            # Muon: hidden weight matrices in ResidualBlocks (2D, non-embedding, non-norm)
            # AdamW: everything else (norms, biases, projection weights, out_gene_emb, oov_embedding)
            muon_params = []
            adamw_params = []

            for name, param in self.head.named_parameters():
                if not param.requires_grad:
                    continue
                # Apply Muon to 2D weight matrices in ResidualBlock hidden layers only
                # NOT to: embeddings (oov_embedding), output params (out_gene_emb),
                #         norm params, biases, projection matrices (proj_in, proj_bilinear)
                is_hidden_matrix = (
                    param.ndim >= 2  # Must be at least 2D
                    and "res_blocks" in name   # Must be in residual blocks
                    and "weight" in name        # Must be a weight (not bias)
                    and "norm" not in name      # Not LayerNorm weight
                )
                if is_hidden_matrix:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

            n_muon = sum(p.numel() for p in muon_params)
            n_adamw = sum(p.numel() for p in adamw_params)
            print(f"[Optimizer] MuonWithAuxAdam: Muon={n_muon:,} params, AdamW={n_adamw:,} params")
            print(f"[Optimizer] Muon lr={hp.lr_muon}, AdamW lr={hp.lr_adamw}, wd={hp.weight_decay}")

            param_groups = [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=hp.lr_muon,
                    weight_decay=hp.weight_decay,
                    momentum=0.95,
                ),
                dict(
                    params=adamw_params,
                    use_muon=False,
                    lr=hp.lr_adamw,
                    betas=(0.9, 0.95),
                    eps=1e-10,
                    weight_decay=hp.weight_decay,
                ),
            ]
            optimizer = MuonWithAuxAdam(param_groups)

        except ImportError:
            # Fallback to AdamW if Muon is not available
            print("[Optimizer] WARNING: MuonWithAuxAdam not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                self.head.parameters(),
                lr=hp.lr_adamw,
                weight_decay=hp.weight_decay,
            )

        # Cosine annealing with linear warmup
        # Calibrated to ~80 epochs (T_max=880 steps for effective_batch=64 on 1 GPU)
        def lr_lambda(step: int) -> float:
            if step < hp.warmup_steps:
                return float(step) / max(1, hp.warmup_steps)
            progress = (step - hp.warmup_steps) / max(1, hp.total_steps - hp.warmup_steps)
            # Clamp progress to [0, 1] to prevent schedule rebound (lesson from node1-1-2-1-1-1)
            progress = min(progress, 1.0)
            return max(1e-7 / hp.lr_adamw, 0.5 * (1.0 + np.cos(np.pi * progress)))

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
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        trainable_sd = {k: v for k, v in full_sd.items() if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)")
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 2-1-1-1-1-1 – Frozen STRING_GNN + Deep Bilinear MLP with "
                    "Pre-Initialized Out Gene Embeddings and MuonWithAuxAdam"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--gnn-dim",            type=int,   default=256)
    p.add_argument("--hidden-dim",         type=int,   default=512)
    p.add_argument("--rank",               type=int,   default=512,
                   help="Bilinear rank for out_gene_emb (512 matches best nodes in tree)")
    p.add_argument("--n-residual-layers",  type=int,   default=6)
    p.add_argument("--dropout",            type=float, default=0.2)
    p.add_argument("--lr-muon",            type=float, default=0.005,
                   help="Muon LR for hidden ResidualBlock weight matrices")
    p.add_argument("--lr-adamw",           type=float, default=5e-4,
                   help="AdamW LR for output embeddings, projections, norms")
    p.add_argument("--weight-decay",       type=float, default=2e-3,
                   help="Weight decay (same as best nodes in tree)")
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--use-class-weights",  action="store_true", default=True,
                   help="Use tree-validated class weights [2.0, 0.5, 4.0]")
    p.add_argument("--no-class-weights",   dest="use_class_weights", action="store_false")
    p.add_argument("--warmup-steps",       type=int,   default=50)
    p.add_argument("--micro-batch-size",   type=int,   default=16)
    p.add_argument("--global-batch-size",  type=int,   default=64)
    p.add_argument("--max-epochs",         type=int,   default=300)
    p.add_argument("--patience",           type=int,   default=80,
                   help="Patience for EarlyStopping (matching tree-best nodes)")
    p.add_argument("--num-workers",        type=int,   default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
    return p.parse_args()


def build_out_gene_emb_init(
    gnn_embeddings: np.ndarray,
    node_name_to_idx: Dict[str, int],
    label_gene_ids: List[str],
    rank: int,
) -> Optional[np.ndarray]:
    """Build output gene embedding initializations from STRING_GNN.

    For each of the 6640 output genes (from label_genes.txt), look up the
    corresponding STRING_GNN embedding if available. Return [6640, 256] array.
    OOV genes (not in STRING_GNN) get random embeddings.

    Args:
        gnn_embeddings: [N_gnn, 256] precomputed STRING_GNN embeddings
        node_name_to_idx: Ensembl ID -> STRING_GNN node index
        label_gene_ids: list of 6640 Ensembl IDs for output gene positions
        rank: bilinear rank (used for logging only; projection happens in GNNBilinearHead)

    Returns:
        [6640, 256] numpy array with STRING_GNN embeddings for in-vocab genes,
        random init for OOV genes.
    """
    gnn_dim = gnn_embeddings.shape[1]
    out_init = np.random.randn(len(label_gene_ids), gnn_dim).astype(np.float32) * 0.02

    n_found = 0
    for i, gene_id in enumerate(label_gene_ids):
        if gene_id in node_name_to_idx:
            node_idx = node_name_to_idx[gene_id]
            out_init[i] = gnn_embeddings[node_idx]
            n_found += 1

    print(f"[Init] Output gene embeddings: {n_found}/{len(label_gene_ids)} "
          f"({100*n_found/len(label_gene_ids):.1f}%) initialized from STRING_GNN "
          f"(will be projected to rank={rank} in GNNBilinearHead)")
    return out_init


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

    # Build output gene embedding initializations from STRING_GNN embeddings
    # This provides biological priors for the output space geometry
    out_gene_emb_init = None
    if hasattr(dm, "label_gene_ids") and len(dm.label_gene_ids) == N_GENES_OUT:
        out_gene_emb_init = build_out_gene_emb_init(
            dm.gnn_embeddings,
            dm.node_name_to_idx,
            dm.label_gene_ids,
            rank=args.rank,
        )
    else:
        print("[Main] WARNING: label_gene_ids not loaded or wrong length; using random out_gene_emb init")

    # Estimate total training steps for LR schedule
    # Following tree-best nodes: calibrate to ~80 epochs
    steps_per_epoch = max(1, len(dm.train_ds) // (args.micro_batch_size * n_gpus))
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)
    # Calibrate to ~80 epochs (matches best nodes in tree with patience=80)
    calibrated_total_steps = effective_steps_per_epoch * 80
    total_steps = max(800, calibrated_total_steps)

    print(f"[Main] effective_steps_per_epoch={effective_steps_per_epoch}, "
          f"calibrated total_steps={total_steps} "
          f"(targeting ~80 epochs for LR schedule alignment, matching tree-best nodes)")

    # LightningModule
    lit = PerturbationLitModule(
        gnn_dim=args.gnn_dim,
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        n_residual_layers=args.n_residual_layers,
        dropout=args.dropout,
        lr_muon=args.lr_muon,
        lr_adamw=args.lr_adamw,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        use_class_weights=args.use_class_weights,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        out_gene_emb_init=out_gene_emb_init,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max",
        save_top_k=1, save_last=True,
    )
    es_cb = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.patience,  # 80 (matches tree-best nodes)
        min_delta=1e-5,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    pb_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    # Debug / fast-dev-run settings
    max_steps:           int | None   = -1
    limit_train_batches: float | int  = 1.0
    limit_val_batches:   float | int  = 1.0
    limit_test_batches:  float | int  = 1.0
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
        gradient_clip_val=1.0,
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
            f"Node 2-1-1-1-1-1 – Frozen STRING_GNN + Deep Bilinear MLP "
            f"with Pre-Initialized Out Gene Embeddings and MuonWithAuxAdam\n"
            f"Test results from trainer: {test_results}\n"
            f"(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
