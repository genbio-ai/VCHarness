"""Node 1-1-1-2-1-1: Frozen STRING_GNN + PostLN MLP + Muon Optimizer + Manifold Mixup + CosineWarmRestarts.

Architecture:
  - Input: perturbed gene identifier (ENSEMBL ID)
  - STRING_GNN fully frozen (256-dim embeddings, no fine-tuning)
  - Per-gene lookup: use the frozen GNN embedding for the perturbed gene
  - Fallback embedding: learnable 256-dim vector for genes not in STRING (~6.1% of perturbations)
  - 3-block PostLN Residual MLP: 512-dim hidden with 1024-dim intermediate
  - Output Head: LayerNorm(512) -> Dropout(0.15) -> Linear(512 -> 6640 * 3)
  - Loss: Focal loss (gamma=2.0) + class weights (no label smoothing)
  - Augmentation: Manifold Mixup in hidden space (alpha=0.2, prob=0.65)
  - Optimizer: MuonWithAuxAdam — Muon for MLP hidden weight matrices, AdamW for head/norms/biases
  - LR schedule: CosineAnnealingWarmRestarts (T_0=80, T_mult=2)

Key changes vs parent node1-1-1-2-1:
  1. Fully frozen STRING_GNN (vs parent's partial fine-tuning) — Muon + partial GNN fine-tuning is
     confirmed incompatible in 2 independent nodes (node1-2-1-1-1-1-1: F1=0.4850, train_loss=0.18-0.22;
     node3-1-1-2-1-1: F1=0.4332, massive val-test gap), because Muon's Newton-Schulz orthogonalization
     requires stationary gradient distributions that are violated when GNN embedding space shifts
     each step. Frozen STRING enables clean Muon training dynamics.
  2. Muon+AdamW optimizer (MuonWithAuxAdam) — Muon for 2D hidden weight matrices (input_proj linear,
     block linears), AdamW for remaining params (head linear, norms, biases, fallback embedding).
     Muon lr=0.01, AdamW lr=3e-4. This combination achieved F1=0.5243 in the tree-best node
     (node3-3-1-2-1-1-1) with frozen STRING + Manifold Mixup + CosineWarmRestarts — the
     same training recipe as this node but without partial GNN fine-tuning.
  3. Increased Mixup prob to 0.65 (from 0.50) — tree-best node (F1=0.5243) used prob=0.65.
     Parent feedback explicitly recommended 0.65–0.75 as next step. More aggressive augmentation
     is warranted since frozen STRING embeddings cannot be adapted, making data augmentation
     even more important as a regularization mechanism.
  4. Reduced early_stop_patience to 80 (from 160) — parent's best checkpoint was at epoch 274,
     but the model continued 126 more epochs with increasing val loss and high variance. Third
     cycle (240-400) showed clear overfitting: train_loss ~0.007, val_loss ~0.063, F1 0.467–0.485.
     Reduced patience prevents wasted computation and limits checkpoint degradation.
  5. Increased weight_decay to 8e-4 (from 5e-4) — tree-best STRING-only Muon nodes consistently
     use 8e-4 (node1-3-3 F1=0.4950, node1-3-2-2-1-1-1-1-1-1 F1=0.4968, node3-3-1-2-1-1-1 F1=0.5243).
     Matches the proven Muon+frozen-STRING weight decay level.
  6. Simplified optimizer setup — no GNN parameter groups needed since STRING is fully frozen.
     Only two groups: Muon (hidden 2D weight matrices in MLP) and AdamW (everything else).
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640          # number of response genes
N_CLASSES = 3           # {-1->0, 0->1, 1->2}
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
STRING_EMB_DIM = 256    # STRING_GNN output embedding dimension


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Dataset for gene perturbation -> differential expression prediction."""

    def __init__(
        self,
        df: pd.DataFrame,
        node_name2idx: Dict[str, int],
        n_nodes: int,
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        # Map each perturbation to STRING_GNN node index (-1 = not in STRING)
        gnn_indices = []
        for pid in self.pert_ids:
            idx = node_name2idx.get(pid, -1)
            gnn_indices.append(idx)
        self.gnn_indices = torch.tensor(gnn_indices, dtype=torch.long)

        if "label" in df.columns:
            labels = np.array([json.loads(x) for x in df["label"].tolist()], dtype=np.int64)
            # Shift {-1,0,1} -> {0,1,2}
            self.labels = torch.tensor(labels + 1, dtype=torch.long)  # [N, 6640]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "gnn_idx": self.gnn_indices[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]  # [6640]
        return item


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class PerturbDataModule(LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        micro_batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

        self.node_name2idx: Dict[str, int] = {}
        self.n_nodes: int = 0
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        # Build STRING_GNN node name to index mapping
        if not self.node_name2idx:
            node_names = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
            self.node_name2idx = {name: i for i, name in enumerate(node_names)}
            self.n_nodes = len(node_names)

        self.train_ds = PerturbDataset(train_df, self.node_name2idx, self.n_nodes)
        self.val_ds = PerturbDataset(val_df, self.node_name2idx, self.n_nodes)
        self.test_ds = PerturbDataset(test_df, self.node_name2idx, self.n_nodes)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Loss function: Pure focal loss (no label smoothing)
# ---------------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Multi-class focal loss.

    logits:  [N, C] float
    targets: [N]    long  values in {0,...,C-1}
    class_weights: [C] float
    gamma: focusing parameter (0 = standard CE)

    Label smoothing is intentionally absent: focal loss already handles class imbalance
    through the (1-p_t)^gamma modulation. Combining label smoothing with focal loss
    creates conflicting objectives (focal = confident; smoothing = diffuse).
    """
    log_softmax = F.log_softmax(logits, dim=-1)  # [N, C]
    softmax = log_softmax.exp()                   # [N, C]

    log_pt = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
    pt = softmax.gather(1, targets.unsqueeze(1)).squeeze(1)           # [N]

    focal_weight = (1 - pt) ** gamma  # [N]
    sample_weight = class_weights[targets]  # [N]

    loss = -focal_weight * sample_weight * log_pt  # [N]
    return loss.mean()


def focal_loss_mixup(
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for Manifold Mixup (linear interpolation of loss for mixed targets).

    When Manifold Mixup is active, the hidden representations are interpolated:
       h_mix = lam * h_a + (1 - lam) * h_b
    The loss is computed as a convex combination of two focal losses:
       L = lam * focal(h_mix, a) + (1 - lam) * focal(h_mix, b)
    This is the standard Mixup loss formulation applied to focal loss.
    """
    loss_a = focal_loss(logits, targets_a, class_weights, gamma)
    loss_b = focal_loss(logits, targets_b, class_weights, gamma)
    return lam * loss_a + (1 - lam) * loss_b


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class PostLNResidualBlock(nn.Module):
    """Post-LayerNorm residual MLP block.

    PostLN applies LayerNorm AFTER the sublayer, which is the original Transformer
    design. For shallow networks (3 blocks), PostLN provides stronger gradient flow
    through the skip connection and better direct signal transmission from input to
    output compared to PreLN.

    Evidence from search tree: PostLN is validated across multiple successful nodes:
    node1-1-1 (F1=0.474), node1-1 (F1=0.472), node1-1-1-2-1 (F1=0.4912 parent).
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PostLN: apply sublayer, add residual, then normalize
        return self.norm(x + self.net(x))


class StringGNNFrozenPostLNMLP(nn.Module):
    """Fully frozen STRING_GNN + PostLN MLP with Muon-friendly parameter structure.

    Architecture:
    - Fully frozen STRING_GNN backbone (no gradient updates to GNN parameters)
    - PostLN residual blocks (proven better than PreLN for shallow 3-block MLP)
    - Head dropout p=0.15 (breakthrough innovation from node1-3-2-2-1, preserved)
    - No per-gene bias (confirmed negligible in multiple nodes)
    - No label smoothing (removed to prevent focal-smoothing conflict)

    Parameter groups for Muon+AdamW:
    - Muon (hidden 2D weight matrices): input_proj Linear, block net Linear layers (4 linears per block * 3)
    - AdamW (everything else): LayerNorms, Dropout (no params), head Linear, fallback_emb
    """

    def __init__(
        self,
        n_nodes: int,
        gnn_emb_dim: int = STRING_EMB_DIM,
        hidden_dim: int = 512,
        n_blocks: int = 3,
        n_genes: int = N_GENES,
        n_classes: int = N_CLASSES,
        dropout: float = 0.35,
        head_dropout: float = 0.15,
    ) -> None:
        super().__init__()

        # Fallback embedding for genes not in STRING_GNN
        self.fallback_emb = nn.Embedding(1, gnn_emb_dim)
        nn.init.normal_(self.fallback_emb.weight, std=0.01)

        # Project STRING_GNN embeddings (256) -> hidden_dim
        # input_proj_linear is the 2D weight matrix suitable for Muon
        self.input_proj_linear = nn.Linear(gnn_emb_dim, hidden_dim)
        self.input_proj_norm = nn.LayerNorm(hidden_dim)
        self.input_proj_act = nn.GELU()

        # PostLN residual MLP blocks (key: PostLN for better gradient flow in shallow networks)
        self.blocks = nn.ModuleList([
            PostLNResidualBlock(hidden_dim, hidden_dim * 2, dropout)
            for _ in range(n_blocks)
        ])

        # Output head with head dropout before the final linear projection
        # Head dropout p=0.15 from node1-3-2-2-1 (F1: 0.4750 -> 0.4777 breakthrough)
        # With PostLN, the final norm is already applied after the last block.
        self.head_dropout = nn.Dropout(head_dropout)
        self.head_linear = nn.Linear(hidden_dim, n_genes * n_classes)

        self.n_genes = n_genes
        self.n_classes = n_classes

    def forward(
        self,
        gnn_node_emb: torch.Tensor,  # [N_nodes, 256] - precomputed STRING_GNN output
        gnn_idx: torch.Tensor,        # [B] - index into gnn_node_emb, or -1 for missing
    ) -> torch.Tensor:
        """Standard forward pass (no Mixup)."""
        emb = self._get_embeddings(gnn_node_emb, gnn_idx)
        x = self.input_proj_act(self.input_proj_norm(self.input_proj_linear(emb)))
        for block in self.blocks:
            x = block(x)
        logits = self.head_linear(self.head_dropout(x))
        return logits.view(gnn_idx.size(0), self.n_classes, self.n_genes)  # [B, 3, 6640]

    def forward_with_mixup(
        self,
        gnn_node_emb: torch.Tensor,
        gnn_idx: torch.Tensor,
        mixup_lam: float,
        mixup_perm: torch.Tensor,
        mixup_block_idx: int,
    ) -> torch.Tensor:
        """Forward pass with Manifold Mixup at the specified block index.

        Manifold Mixup interpolates hidden representations at a randomly chosen
        layer: h_mix = lam * h_a + (1-lam) * h_b[perm]
        This creates more diverse training examples in the hidden space than
        input-level mixing, improving generalization on small datasets.

        With frozen STRING embeddings (no task-specific adaptation), Manifold Mixup
        is even more critical as the primary data augmentation mechanism.

        Evidence: node3-3-1-2-1-1-1 (F1=0.5243) uses Manifold Mixup prob=0.65 +
        frozen STRING + Muon — the combination this node implements.
        """
        emb = self._get_embeddings(gnn_node_emb, gnn_idx)
        x = self.input_proj_act(self.input_proj_norm(self.input_proj_linear(emb)))

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == mixup_block_idx:
                # Apply Manifold Mixup AFTER this block
                x = mixup_lam * x + (1 - mixup_lam) * x[mixup_perm]

        logits = self.head_linear(self.head_dropout(x))
        return logits.view(gnn_idx.size(0), self.n_classes, self.n_genes)  # [B, 3, 6640]

    def _get_embeddings(
        self,
        gnn_node_emb: torch.Tensor,
        gnn_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve STRING_GNN embeddings with fallback for out-of-graph genes."""
        B = gnn_idx.size(0)
        device = gnn_idx.device

        emb = torch.zeros(B, gnn_node_emb.size(1), device=device, dtype=gnn_node_emb.dtype)

        in_gnn_mask = gnn_idx >= 0
        not_in_gnn_mask = ~in_gnn_mask

        if in_gnn_mask.any():
            valid_idx = gnn_idx[in_gnn_mask]
            emb[in_gnn_mask] = gnn_node_emb[valid_idx]

        if not_in_gnn_mask.any():
            fallback = self.fallback_emb(
                torch.zeros(not_in_gnn_mask.sum(), device=device, dtype=torch.long)
            )
            emb[not_in_gnn_mask] = fallback.to(emb.dtype)

        return emb

    def get_muon_params(self) -> List[torch.Tensor]:
        """Return 2D weight matrix parameters suitable for Muon optimizer.

        Per Muon skill: use Muon for hidden layer 2D weight matrices.
        Do NOT use Muon for: input embeddings, output heads, LayerNorm, biases.
        """
        muon_params = []
        # input_proj_linear weight (2D: [hidden_dim, gnn_emb_dim])
        muon_params.append(self.input_proj_linear.weight)
        # Block linear weights (each block has 2 linears: dim->hidden_dim, hidden_dim->dim)
        for block in self.blocks:
            for module in block.net:
                if isinstance(module, nn.Linear):
                    muon_params.append(module.weight)
        return muon_params

    def get_adamw_params(self) -> List[torch.Tensor]:
        """Return all parameters NOT handled by Muon (biases, norms, head, embeddings).

        These use standard AdamW: input_proj_linear.bias, LayerNorms, head linear,
        fallback_emb, block biases.
        """
        muon_param_set = set(id(p) for p in self.get_muon_params())
        adamw_params = [p for p in self.parameters() if id(p) not in muon_param_set]
        return adamw_params


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.35,
        head_dropout: float = 0.15,
        muon_lr: float = 0.01,
        adamw_lr: float = 3e-4,
        weight_decay: float = 8e-4,
        focal_gamma: float = 2.0,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.65,
        t0: int = 80,
        t_mult: int = 2,
        max_epochs: int = 400,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.muon_lr = muon_lr
        self.adamw_lr = adamw_lr
        self.weight_decay = weight_decay
        self.focal_gamma = focal_gamma
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.t0 = t0
        self.t_mult = t_mult
        self.max_epochs = max_epochs

        self.model: Optional[StringGNNFrozenPostLNMLP] = None
        self.gnn_model = None  # STRING_GNN backbone (fully frozen)

        # Buffers for validation and test
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

        # Will hold the cached GNN node embeddings [n_nodes, 256]
        # Since STRING_GNN is fully frozen, we can cache once and reuse across all epochs
        self._gnn_emb_cache: Optional[torch.Tensor] = None

    def setup(self, stage: str = "fit") -> None:
        from transformers import AutoModel

        # Class weights: inversely proportional to frequency
        # class0 (down-reg) 4.77%, class1 (neutral) 92.82%, class2 (up-reg) 2.41%
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = 1.0 / freq
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        self.register_buffer("class_weights", class_weights)

        # Load STRING_GNN backbone -- fully frozen
        if self.gnn_model is None:
            self.gnn_model = AutoModel.from_pretrained(
                str(STRING_GNN_DIR), trust_remote_code=True
            )
            # Freeze ALL parameters — this is the key difference from the parent
            # Frozen STRING_GNN is compatible with Muon optimizer on the MLP head
            for param in self.gnn_model.parameters():
                param.requires_grad = False

            total_gnn = sum(p.numel() for p in self.gnn_model.parameters())
            self.print(
                f"STRING_GNN loaded: {total_gnn:,} total params, ALL FROZEN "
                f"(enables clean Muon optimizer dynamics for the MLP head)"
            )

        # Initialize the MLP prediction model -- only once
        if self.model is None:
            self.model = StringGNNFrozenPostLNMLP(
                n_nodes=self.n_nodes,
                gnn_emb_dim=STRING_EMB_DIM,
                hidden_dim=self.hidden_dim,
                n_blocks=self.n_blocks,
                dropout=self.dropout,
                head_dropout=self.head_dropout,
            )
            # Cast trainable parameters to float32 for stable optimization
            for v in self.model.parameters():
                if v.requires_grad:
                    v.data = v.data.float()
            mlp_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            muon_param_count = sum(p.numel() for p in self.model.get_muon_params())
            adamw_param_count = sum(p.numel() for p in self.model.get_adamw_params())
            self.print(
                f"StringGNNFrozenPostLNMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
                f"head_dropout={self.head_dropout} | dropout={self.dropout} | "
                f"MLP trainable params={mlp_trainable:,} "
                f"(Muon: {muon_param_count:,}, AdamW: {adamw_param_count:,})"
            )

    def _get_gnn_embeddings(self) -> torch.Tensor:
        """Get fully cached STRING_GNN node embeddings.

        Since STRING_GNN is FULLY FROZEN (no gradient updates), we can compute
        the embeddings once and cache them permanently across all epochs.
        This is more efficient than the parent which had to recompute every training
        step to allow gradient flow through the unfrozen layers.
        """
        if self._gnn_emb_cache is None:
            device = next(self.model.parameters()).device
            self.gnn_model = self.gnn_model.to(device)
            graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location=device, weights_only=False)
            edge_index = graph["edge_index"]
            edge_weight = graph.get("edge_weight", None)
            with torch.no_grad():
                outputs = self.gnn_model(
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
            self._gnn_emb_cache = outputs.last_hidden_state.float().detach()
            self.print(
                f"Computed and cached frozen STRING_GNN embeddings: {self._gnn_emb_cache.shape}"
            )
        return self._gnn_emb_cache

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Pure focal loss (no label smoothing).

        logits: [B, 3, 6640]
        labels: [B, 6640]  values in {0,1,2}
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return focal_loss(logits_flat, labels_flat, self.class_weights, gamma=self.focal_gamma)

    def _compute_loss_mixup(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Focal loss for Manifold Mixup.

        logits:   [B, 3, 6640]
        labels_a: [B, 6640]  original labels
        labels_b: [B, 6640]  permuted labels for Mixup
        lam: float  Mixup lambda
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_a_flat = labels_a.reshape(-1)
        labels_b_flat = labels_b.reshape(-1)
        return focal_loss_mixup(
            logits_flat, labels_a_flat, labels_b_flat, lam,
            self.class_weights, gamma=self.focal_gamma
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        gnn_emb = self._get_gnn_embeddings()
        labels = batch["label"]  # [B, 6640]
        B = labels.size(0)

        # Manifold Mixup: randomly decide whether to apply on this batch
        use_mixup = (
            self.training
            and self.mixup_prob > 0
            and B > 1
            and torch.rand(1).item() < self.mixup_prob
        )

        if use_mixup:
            # Sample Mixup lambda from Beta distribution
            lam_np = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = float(lam_np)
            # Random permutation of batch indices
            perm = torch.randperm(B, device=labels.device)
            # Randomly choose which block to apply Mixup at (0 to n_blocks-1)
            mixup_block_idx = torch.randint(0, self.n_blocks, (1,)).item()

            logits = self.model.forward_with_mixup(
                gnn_emb, batch["gnn_idx"], lam, perm, mixup_block_idx
            )
            labels_b = labels[perm]
            loss = self._compute_loss_mixup(logits, labels, labels_b, lam)
        else:
            logits = self.model(gnn_emb, batch["gnn_idx"])
            loss = self._compute_loss(logits, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        gnn_emb = self._get_gnn_embeddings()
        logits = self.model(gnn_emb, batch["gnn_idx"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds_local = torch.cat(self._val_preds, dim=0)   # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0) # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather from all DDP ranks
        all_preds = self.all_gather(preds_local)   # [ws, N_local, 3, 6640]
        all_labels = self.all_gather(labels_local)  # [ws, N_local, 6640]

        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
        all_labels = all_labels.view(-1, N_GENES)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()

        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        gnn_emb = self._get_gnn_embeddings()
        logits = self.model(gnn_emb, batch["gnn_idx"])
        self._test_preds.append(logits.detach().cpu())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> Optional[Dict[str, Any]]:
        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        all_preds = self.all_gather(preds_local)
        ws = self.trainer.world_size
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        # Compute test F1 if labels are available (gathered across ranks)
        test_f1 = None
        if self._test_labels:
            labels_local = torch.cat(self._test_labels, dim=0)
            all_labels = self.all_gather(labels_local)
            all_labels = all_labels.view(-1, N_GENES).cpu().numpy()
            preds_np = all_preds.float().cpu().numpy()
            test_f1 = _compute_per_gene_f1(preds_np, all_labels)
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=True)
        self._test_labels.clear()

        all_pert_ids: List[str] = []
        all_symbols: List[str] = []
        if self.trainer.is_global_zero:
            _pert_ids_gathered: List[List[str]] = [[] for _ in range(ws)]
            _symbols_gathered: List[List[str]] = [[] for _ in range(ws)]
            torch.distributed.gather_object(self._test_pert_ids, _pert_ids_gathered, dst=0)
            torch.distributed.gather_object(self._test_symbols, _symbols_gathered, dst=0)
            for p_list, s_list in zip(_pert_ids_gathered, _symbols_gathered):
                all_pert_ids.extend(p_list)
                all_symbols.extend(s_list)
        else:
            torch.distributed.gather_object(self._test_pert_ids, dst=0)
            torch.distributed.gather_object(self._test_symbols, dst=0)

        self._test_pert_ids.clear()
        self._test_symbols.clear()

        if self.trainer.is_global_zero:
            if all_preds.size(0) == 0:
                self.print("Warning: No test predictions gathered. Skipping save.")
            else:
                preds_np = all_preds.float().cpu().numpy()

                # Deduplicate by pert_id (DistributedSampler may pad with replicated samples)
                seen = set()
                keep_mask = []
                for pid in all_pert_ids:
                    if pid not in seen:
                        seen.add(pid)
                        keep_mask.append(True)
                    else:
                        keep_mask.append(False)

                n_unique = sum(keep_mask)
                if n_unique < len(keep_mask):
                    self.print(
                        f"Deduplication: {n_unique}/{len(keep_mask)} unique samples after "
                        f"removing DistributedSampler padding."
                    )

                unique_preds = preds_np[keep_mask]
                unique_pert_ids = [p for p, k in zip(all_pert_ids, keep_mask) if k]
                unique_symbols = [s for s, k in zip(all_symbols, keep_mask) if k]

                _save_test_predictions(
                    pert_ids=unique_pert_ids,
                    symbols=unique_symbols,
                    preds=unique_preds,
                    out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
                )

        return {"test_f1": test_f1} if test_f1 is not None else {}

    def configure_optimizers(self):
        """Configure Muon+AdamW dual optimizer with CosineAnnealingWarmRestarts.

        Muon is used for hidden 2D weight matrices in the MLP (input_proj_linear,
        block linear layers). These are the parameters that benefit most from
        orthogonalized gradient updates as they are the main computational bottleneck
        of the MLP trunk.

        AdamW is used for all other parameters: head linear, LayerNorms, biases,
        fallback embedding. The head linear is explicitly excluded from Muon per
        the Muon skill specification ("output classifier heads should use AdamW").

        This combination is validated by tree-best STRING-only nodes:
        - node3-3-1-2-1-1-1 (F1=0.5243): frozen ESM2+STRING + Muon lr=0.01 + AdamW lr=3e-4
        - node1-3-2-2-1-1-1-1-1-1 (F1=0.4968): frozen STRING + Muon lr=0.01 + AdamW lr=3e-4
        Both used frozen STRING (consistent with this node's fully frozen GNN).
        """
        from muon import MuonWithAuxAdam

        muon_params = self.model.get_muon_params()
        adamw_params = self.model.get_adamw_params()

        muon_param_count = sum(p.numel() for p in muon_params)
        adamw_param_count = sum(p.numel() for p in adamw_params)
        self.print(
            f"Optimizer setup: Muon({muon_param_count:,} params, lr={self.muon_lr}) + "
            f"AdamW({adamw_param_count:,} params, lr={self.adamw_lr})"
        )

        param_groups = [
            # Muon group for hidden weight matrices
            {
                "params": muon_params,
                "use_muon": True,
                "lr": self.muon_lr,
                "weight_decay": self.weight_decay,
                "momentum": 0.95,
            },
            # AdamW group for other parameters (head, norms, biases, embeddings)
            {
                "params": adamw_params,
                "use_muon": False,
                "lr": self.adamw_lr,
                "betas": (0.9, 0.95),
                "weight_decay": self.weight_decay,
            },
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # CosineAnnealingWarmRestarts: allows warm restarts to escape local optima.
        # T_0=80: first restart at epoch 80
        # T_mult=2: each subsequent cycle is 2x longer (80 -> 160 -> 320...)
        # This schedule is validated in tree-best nodes:
        # - node3-3-1-2-1-1-1: F1=0.5243 (frozen ESM2+STRING+Muon+CosineWarmRestarts)
        # - node1-3-2-2-1-1-1-1-1-1: F1=0.4968 (frozen STRING+Muon+CosineWarmRestarts)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.t0,
            T_mult=self.t_mult,
            eta_min=1e-7,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters (skip frozen STRING_GNN backbone)."""
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                trainable[key] = full_sd[key]
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} trainable params "
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute per-gene macro F1 averaged over genes (matches calc_metric.py logic).

    preds:  [N, 3, 6640] float -- class logits
    labels: [N, 6640]    int   -- class indices in {0,1,2}
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N, 6640]
    n_genes = labels.shape[1]

    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        f1_vals.append(float(per_class_f1[present].mean()))
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    """Save test predictions in required TSV format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
        pred_list = preds[i].tolist()  # [3][6640]
        rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-1-1-2-1-1: Frozen STRING_GNN + PostLN MLP + Muon Optimizer + Manifold Mixup + CosineWarmRestarts"
    )
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=400)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=8e-4)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--head-dropout", type=float, default=0.15)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.65)
    p.add_argument("--t0", type=int, default=80)
    p.add_argument("--t-mult", type=int, default=2)
    p.add_argument("--early-stop-patience", type=int, default=80)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- DataModule ---
    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup("fit")

    # --- Model ---
    model = PerturbModule(
        n_nodes=datamodule.n_nodes,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        t0=args.t0,
        t_mult=args.t_mult,
        max_epochs=args.max_epochs,
    )

    # --- Trainer config ---
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if fast_dev_run:
        limit_train = None
        limit_val = None
        limit_test = None
        max_steps = -1
    elif args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        max_steps = -1

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-val_f1={val/f1:.4f}-epoch={epoch:03d}",
        monitor="val/f1",
        mode="max",
        save_top_k=5,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.early_stop_patience,
        min_delta=1e-5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=20)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=1.0,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Save score
    if trainer.is_global_zero and test_results:
        merged = {}
        for r in test_results:
            if isinstance(r, dict):
                merged.update(r)
        score_path = Path(__file__).parent / "test_score.txt"
        f1_val = merged.get("test_f1", merged.get("test/f1", "N/A"))
        score_path.write_text(f"f1_score: {f1_val}\n{json.dumps(merged, indent=2)}")
        print(f"Test results -> {score_path}: {merged}")


if __name__ == "__main__":
    main()
