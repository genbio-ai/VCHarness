"""Node 1-3-3-2: ESM2-650M + STRING_GNN Dual-Branch with Gated Fusion

Architecture and training recipe based on proven learnings from tree memory:
  - node3-1-1-1-1-2-1-1-1 (F1=0.5283, tree-best): frozen ESM2-650M + STRING + gated fusion
    FUSION_DIM=512, Muon, CosineWarmRestarts T_0=80/T_mult=2, Mixup prob=0.65, top-3 ensemble
  - node3-1-1-1-1-2-1-1 (F1=0.5265): FUSION_DIM=512 better than 256
  - node3-1-1-1-1-2 (F1=0.5243): confirmed frozen ESM2-650M + STRING dual-branch recipe
  - node2-1-2-1-1-1-1 (F1=0.5169): cross-lineage transfer confirms recipe generalizes
  - node1-3-3 (parent, F1=0.4950): Muon LR=0.01 + CosineWarmRestarts T_0=80 + Mixup

Key differentiation from sibling:
  - node1-3-3-1: STRING-only + T_0=150 + top-5 ensemble → REGRESSED to 0.4930
  - node1-3-3-2 (this): ESM2-650M + STRING dual-branch → targets >0.51

Why dual-branch over STRING-only:
  - Both parent and sibling feedbacks confirm STRING-only ceiling ~0.495
  - tree-best is 0.5283 with frozen ESM2-650M + STRING dual-branch
  - ESM2 captures evolutionary sequence information not in STRING PPI graph
  - Gated fusion allows model to learn optimal ESM2/STRING balance

Key technical choices:
  - FROZEN ESM2-650M (not LoRA): frozen > LoRA for 1,273 samples (node3-3-1-2-1-1 regression)
  - FUSION_DIM=512: proven optimal (0.5265 vs 0.5243 for 256-dim)
  - hidden_dim=512: matches FUSION_DIM for proper residual connections
  - Mixup applied in FUSED SPACE (prob=0.65): tree-best nodes use 0.65-0.75
  - T_0=80 (not 150): sibling showed T_0=150 reduces restarts and hurts performance
  - Top-5 ensemble: proven +0.003-0.008 F1 gain with diverse dual-branch checkpoints
  - Correct class weights: freq=[0.0477, 0.9282, 0.0241] (bug in node2-1-2-1-1-1-1)
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
N_GENES = 6640
N_CLASSES = 3
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
# Pre-computed ESM2-650M embeddings (shape [18870, 3840] = 3 layers * 1280-dim)
# averaged to [18870, 1280] during loading — proper protein-sequence-based embeddings
ESM2_EMB_FILE = STRING_GNN_DIR / "esm2_embeddings_t33_650M.pt"
ESM2_DIM = 1280   # esm2_t33_650M per-layer embedding dimension (3840 raw → 1280 mean-pooled)
STRING_DIM = 256  # STRING_GNN output dimension
FUSION_DIM = 512  # Gated fusion output dimension (proven optimal in node3 lineage)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Gene-perturbation → differential-expression dataset."""

    def __init__(self, df: pd.DataFrame, gene2str_idx: Dict[str, int]) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        # Map ENSEMBL pert_id → STRING node index; -1 = not in STRING graph
        self.str_indices = torch.tensor(
            [gene2str_idx.get(pid, -1) for pid in self.pert_ids], dtype=torch.long
        )
        if "label" in df.columns:
            labels = np.array([json.loads(x) for x in df["label"]], dtype=np.int64)
            self.labels = torch.tensor(labels + 1, dtype=torch.long)  # {-1,0,1} → {0,1,2}
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "str_idx": self.str_indices[idx],
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
        micro_batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.gene2str_idx: Dict[str, int] = {}
        self.train_ds = self.val_ds = self.test_ds = None

    def setup(self, stage: str = "fit") -> None:
        # Build ENSEMBL-ID → STRING-node-index mapping once
        if not self.gene2str_idx:
            node_names: List[str] = json.loads(
                (STRING_GNN_DIR / "node_names.json").read_text()
            )
            self.gene2str_idx = {ensg: i for i, ensg in enumerate(node_names)}

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df, self.gene2str_idx)
        self.val_ds = PerturbDataset(val_df, self.gene2str_idx)
        self.test_ds = PerturbDataset(test_df, self.gene2str_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------
class PreNormResidualBlock(nn.Module):
    """Pre-LayerNorm residual MLP block (h → 2h → h).

    Proven effective in STRING-only lineage (node1-3-3 and tree-best nodes).
    """

    def __init__(self, dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(self.norm(x)))


class GatedFusion(nn.Module):
    """Learnable sigmoidal gated fusion of STRING and ESM2 branches.

    Architecture proven optimal in node3-1-1-1-1-2 lineage (F1=0.5265).
    Fusion dim=512 established as optimal (vs 256-dim).

    gate = sigmoid(Linear(2*fusion_dim → 1)(concat(str_proj, esm_proj)))
    fused = gate * str_proj + (1 - gate) * esm_proj
    """

    def __init__(self, fusion_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(fusion_dim * 2, 1)

    def forward(
        self,
        str_proj: torch.Tensor,  # [B, fusion_dim]
        esm_proj: torch.Tensor,  # [B, fusion_dim]
    ) -> torch.Tensor:
        combined = torch.cat([str_proj, esm_proj], dim=-1)  # [B, 2*fusion_dim]
        gate = torch.sigmoid(self.gate(combined))            # [B, 1]
        return gate * str_proj + (1 - gate) * esm_proj      # [B, fusion_dim]


class DualBranchModel(nn.Module):
    """ESM2-650M + STRING_GNN dual-branch model with gated fusion.

    Architecture:
      STRING_GNN embed [B, 256] + ESM2-650M embed [B, 1280]
      → proj_str [256→FUSION_DIM] + proj_esm [1280→FUSION_DIM]
      → GatedFusion → [B, FUSION_DIM]
      → input_norm → 3x PreNormResidualBlock(FUSION_DIM)
      → head_norm → head_dropout → Linear(FUSION_DIM→19920) + gene_bias
      → Reshape [B, 3, 6640]

    Key design choices (from node3 lineage memory):
    - FUSION_DIM=512 (proven optimal in node3-1-1-1-1-2-1-1, F1=0.5265)
    - hidden_dim=512 (= FUSION_DIM for proper residual connections)
    - head_dropout=0.15 (optimal from parent; 0.20 over-regularizes)
    - Per-gene additive bias (learns baseline DE priors)
    """

    def __init__(
        self,
        fusion_dim: int = FUSION_DIM,
        n_blocks: int = 3,
        dropout: float = 0.35,
        head_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim

        # Fallback embeddings for genes not in STRING graph (~6% of perturbations)
        self.fallback_str_emb = nn.Parameter(torch.zeros(STRING_DIM))
        self.fallback_esm_emb = nn.Parameter(torch.zeros(ESM2_DIM))
        nn.init.normal_(self.fallback_str_emb, std=0.02)
        nn.init.normal_(self.fallback_esm_emb, std=0.02)

        # Branch projections
        self.proj_str = nn.Linear(STRING_DIM, fusion_dim)
        self.proj_esm = nn.Linear(ESM2_DIM, fusion_dim)

        # Gated fusion
        self.fusion = GatedFusion(fusion_dim)

        # Input normalization (after fusion, before MLP blocks)
        self.input_norm = nn.LayerNorm(fusion_dim)

        # Residual MLP blocks (PreLN)
        self.blocks = nn.ModuleList(
            [PreNormResidualBlock(fusion_dim, dropout) for _ in range(n_blocks)]
        )

        # Output head: LayerNorm + head_dropout + flat Linear + per-gene bias
        self.head_norm = nn.LayerNorm(fusion_dim)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head_linear = nn.Linear(fusion_dim, N_GENES * N_CLASSES)
        # Per-gene additive bias: captures baseline DE priors per response gene
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES * N_CLASSES))

    def forward(
        self,
        str_idx: torch.Tensor,       # [B]  STRING node indices, -1 = not in graph
        string_embs: torch.Tensor,   # [N_STR, 256] frozen STRING_GNN buffer
        esm2_embs: torch.Tensor,     # [N_STR, 1280] frozen ESM2 buffer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, fused) where fused is the pre-MLP representation."""
        valid_mask = str_idx >= 0                    # [B] bool
        safe_idx = str_idx.clamp(min=0)              # replace -1 with 0 (overwritten below)

        # Look up frozen STRING embeddings [B, 256]
        str_emb = string_embs[safe_idx].to(torch.float32)
        # Look up frozen ESM2 embeddings [B, 1280]
        esm_emb = esm2_embs[safe_idx].to(torch.float32)

        # Overwrite samples whose gene is absent from the STRING graph
        if not valid_mask.all():
            n_invalid = int((~valid_mask).sum())
            fallback_str = self.fallback_str_emb.to(str_emb).unsqueeze(0).expand(n_invalid, -1)
            fallback_esm = self.fallback_esm_emb.to(esm_emb).unsqueeze(0).expand(n_invalid, -1)
            str_emb = str_emb.clone()
            esm_emb = esm_emb.clone()
            str_emb[~valid_mask] = fallback_str
            esm_emb[~valid_mask] = fallback_esm

        # Project each branch to fusion_dim
        str_proj = self.proj_str(str_emb)   # [B, fusion_dim]
        esm_proj = self.proj_esm(esm_emb)   # [B, fusion_dim]

        # Gated fusion: gate learns optimal STRING/ESM2 balance per sample
        fused = self.fusion(str_proj, esm_proj)  # [B, fusion_dim]

        # Input normalization + residual MLP
        x = self.input_norm(fused)              # [B, fusion_dim]
        for block in self.blocks:
            x = block(x)                        # [B, fusion_dim]

        # Output head with dropout
        h = self.head_norm(x)                   # [B, fusion_dim]
        h = self.head_dropout(h)                # Regularize output head
        logits = self.head_linear(h) + self.gene_bias.to(h)  # [B, N_GENES * N_CLASSES]
        return logits.view(-1, N_CLASSES, N_GENES), fused  # ([B, 3, 6640], [B, fusion_dim])


# ---------------------------------------------------------------------------
# Manifold Mixup helper (applied in fused representation space)
# ---------------------------------------------------------------------------
def manifold_mixup(
    x: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Manifold Mixup in the fused representation space.

    Applied BEFORE the MLP blocks (in the fused ESM2+STRING space).
    Tree-best nodes use prob=0.65-0.75 and this location for mixing.

    Returns:
        mixed_x: mixed embedding [B, D]
        labels_a: original labels [B, 6640]
        labels_b: shuffled labels [B, 6640]
        lam: mixing coefficient (float)
    """
    batch_size = x.shape[0]
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    labels_a = labels
    labels_b = labels[index]
    return mixed_x, labels_a, labels_b, lam


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        fusion_dim: int = FUSION_DIM,
        n_blocks: int = 3,
        dropout: float = 0.35,
        head_dropout: float = 0.15,
        muon_lr: float = 0.01,
        adamw_lr: float = 3e-4,
        weight_decay: float = 8e-4,
        t0: int = 80,
        t_mult: int = 2,
        mixup_prob: float = 0.65,
        mixup_alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.fusion_dim = fusion_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.muon_lr = muon_lr
        self.adamw_lr = adamw_lr
        self.weight_decay = weight_decay
        self.t0 = t0
        self.t_mult = t_mult
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

        self.model: Optional[DualBranchModel] = None

        # Accumulation buffers for epoch-level metrics
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Class weights: inversely proportional to class frequencies
        # {-1,0,1}→{0,1,2} shift: 0=down(4.77%), 1=neutral(92.82%), 2=up(2.41%)
        # CRITICAL: correct order [down, neutral, up] not [neutral, down, up]
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq)
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        self.register_buffer("class_weights", class_weights)

        if self.model is not None:
            return  # already initialized (guard for re-entrant setup calls)

        # ---- Load STRING_GNN node embeddings (once per rank) ----
        from transformers import AutoModel
        gnn = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        gnn.eval()
        graph = torch.load(
            STRING_GNN_DIR / "graph_data.pt", map_location="cpu", weights_only=False
        )
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)
        with torch.no_grad():
            gnn_out = gnn(edge_index=edge_index, edge_weight=edge_weight)
        string_embs = gnn_out.last_hidden_state.detach().float().cpu()  # [18870, 256]
        del gnn, gnn_out
        # Register as frozen buffer (moved to device by Lightning automatically)
        self.register_buffer("string_embs", string_embs)
        self.print(f"Loaded STRING_GNN embeddings: {string_embs.shape}")

        # ---- Load ESM2-650M node embeddings (pre-computed from protein sequences) ----
        # Uses pre-computed file at STRING_GNN_DIR/esm2_embeddings_t33_650M.pt
        # which contains proper protein-sequence-based embeddings (not ENSEMBL ID chars)
        self._load_esm2_embeddings()
        self.print(f"Loaded ESM2-650M embeddings: {self.esm2_embs.shape}")

        # ---- Build model ----
        self.model = DualBranchModel(
            fusion_dim=self.fusion_dim,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
        )

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        self.print(
            f"Node1-3-3-2 DualBranchModel | fusion_dim={self.fusion_dim} | blocks={self.n_blocks} | "
            f"dropout={self.dropout} | head_dropout={self.head_dropout} | "
            f"muon_lr={self.muon_lr} | t0={self.t0}/t_mult={self.t_mult} | "
            f"mixup_prob={self.mixup_prob}/alpha={self.mixup_alpha} | "
            f"trainable={n_trainable:,}/{n_total:,}"
        )

    def _load_esm2_embeddings(self) -> None:
        """Load frozen ESM2-650M embeddings from pre-computed file.

        The file at STRING_GNN_DIR/esm2_embeddings_t33_650M.pt stores proper
        protein-sequence-based ESM2-650M embeddings with shape [18870, 3840]
        (3 layers × 1280-dim). We mean-pool across the 3 layer dimension to get
        [18870, 1280] (= ESM2_DIM) and register as a frozen buffer.

        This is the correct approach: avoids slow online inference, avoids the
        incorrect approach of tokenizing ENSEMBL IDs as amino-acid sequences.
        """
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.print(f"Loading pre-computed ESM2-650M embeddings from {ESM2_EMB_FILE}...")
        # Raw shape: [18870, 3840] = [N_nodes, 3 * 1280]
        raw_emb = torch.load(str(ESM2_EMB_FILE), map_location="cpu", weights_only=False)
        raw_emb = raw_emb.float()  # ensure float32

        raw_dim = raw_emb.shape[1]
        n_layers = raw_dim // ESM2_DIM  # should be 3
        if n_layers * ESM2_DIM == raw_dim:
            # Average 3 layer representations → [18870, 1280]
            esm2_embs = raw_emb.view(-1, n_layers, ESM2_DIM).mean(dim=1)
            self.print(
                f"  ESM2 raw shape {raw_emb.shape} → mean-pooled {n_layers} layers "
                f"→ {esm2_embs.shape}"
            )
        else:
            # Fallback: take first ESM2_DIM dims if dim not divisible
            esm2_embs = raw_emb[:, :ESM2_DIM]
            self.print(
                f"  Warning: raw ESM2 dim {raw_dim} not divisible by {ESM2_DIM}, "
                f"using first {ESM2_DIM} dims."
            )

        del raw_emb
        # Register as frozen buffer (not saved in checkpoint — reloaded in setup)
        self.register_buffer("esm2_embs", esm2_embs)
        self.print(f"ESM2 embeddings loaded: {esm2_embs.shape}")

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted cross-entropy loss (no label smoothing).

        logits: [B, 3, 6640]
        labels: [B, 6640]  — values in {0, 1, 2}
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                                # [B*6640]
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=0.0,  # No label smoothing — allows lower train loss
        )

    def _compute_mixed_loss(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Mixed cross-entropy loss for Manifold Mixup training.

        logits: [B, 3, 6640]
        labels_a, labels_b: [B, 6640]
        lam: mixing coefficient
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_a_flat = labels_a.reshape(-1)
        labels_b_flat = labels_b.reshape(-1)
        loss_a = F.cross_entropy(
            logits_flat, labels_a_flat, weight=self.class_weights, label_smoothing=0.0
        )
        loss_b = F.cross_entropy(
            logits_flat, labels_b_flat, weight=self.class_weights, label_smoothing=0.0
        )
        return lam * loss_a + (1 - lam) * loss_b

    def _forward_from_fused(
        self, fused: torch.Tensor
    ) -> torch.Tensor:
        """Run MLP head on a fused representation (used in Mixup path).

        fused: [B, fusion_dim] — from GatedFusion output (possibly mixed)
        Returns: logits [B, 3, 6640]
        """
        x = self.model.input_norm(fused)
        for block in self.model.blocks:
            x = block(x)
        h = self.model.head_norm(x)
        h = self.model.head_dropout(h)
        logits = self.model.head_linear(h) + self.model.gene_bias.to(h)
        return logits.view(-1, N_CLASSES, N_GENES)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        str_idx = batch["str_idx"]
        labels = batch["label"]

        # Full forward pass to get logits and fused representation
        logits, fused = self.model(str_idx, self.string_embs, self.esm2_embs)

        # Apply Manifold Mixup in the fused representation space
        # (after GatedFusion, before MLP blocks) — proven location in tree-best nodes
        apply_mixup = (
            self.training
            and np.random.random() < self.mixup_prob
            and labels is not None
        )

        if apply_mixup:
            # Mix in the fused space and re-run MLP
            mixed_fused, labels_a, labels_b, lam = manifold_mixup(
                fused, labels, alpha=self.mixup_alpha
            )
            # Re-run from fused representation through MLP head
            logits_mixed = self._forward_from_fused(mixed_fused)
            loss = self._compute_mixed_loss(logits_mixed, labels_a, labels_b, lam)
        else:
            loss = self._compute_loss(logits, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits, _ = self.model(batch["str_idx"], self.string_embs, self.esm2_embs)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds_local = torch.cat(self._val_preds, dim=0)    # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0)  # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        # Gather across DDP ranks for accurate global F1
        all_preds = self.all_gather(preds_local)   # [world_size, N_local, 3, 6640]
        all_labels = self.all_gather(labels_local) # [world_size, N_local, 6640]
        ws = self.trainer.world_size
        if ws > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
            all_labels = all_labels.view(-1, N_GENES)
        else:
            if all_preds.dim() == 4:
                all_preds = all_preds.squeeze(0)
                all_labels = all_labels.squeeze(0)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits, _ = self.model(batch["str_idx"], self.string_embs, self.esm2_embs)
        self._test_preds.append(logits.detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        all_preds = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640]
        ws = self.trainer.world_size
        if ws > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
        else:
            if all_preds.dim() == 4:
                all_preds = all_preds.squeeze(0)

        # Gather string metadata from all ranks → rank 0 only
        if ws > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            if self.trainer.is_global_zero:
                _pert_gathered: List[List[str]] = [[] for _ in range(ws)]
                _syms_gathered: List[List[str]] = [[] for _ in range(ws)]
                torch.distributed.gather_object(self._test_pert_ids, _pert_gathered, dst=0)
                torch.distributed.gather_object(self._test_symbols, _syms_gathered, dst=0)
                all_pert_ids: List[str] = []
                all_symbols: List[str] = []
                for p_list, s_list in zip(_pert_gathered, _syms_gathered):
                    all_pert_ids.extend(p_list)
                    all_symbols.extend(s_list)
            else:
                torch.distributed.gather_object(self._test_pert_ids, dst=0)
                torch.distributed.gather_object(self._test_symbols, dst=0)
                all_pert_ids, all_symbols = [], []
        else:
            all_pert_ids = self._test_pert_ids
            all_symbols = self._test_symbols

        if self.trainer.is_global_zero:
            preds_np = all_preds.float().cpu().numpy()  # [N_total, 3, 6640]
            _save_test_predictions(
                pert_ids=all_pert_ids,
                symbols=all_symbols,
                preds=preds_np,
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        """Configure Muon+AdamW optimizer with CosineAnnealingWarmRestarts.

        Muon applied to hidden 2D weight matrices in residual blocks.
        AdamW applied to all other parameters (projections, fusion gate, head,
        gene_bias, fallback_embs, norms, biases).

        CosineAnnealingWarmRestarts T_0=80/T_mult=2:
        - Restarts at epochs 80, 240, 560, ...
        - Proven in parent (F1=0.4950) and tree-best dual-branch nodes
        """
        try:
            from muon import MuonWithAuxAdam
            use_muon = True
        except ImportError:
            use_muon = False
            self.print("Muon not installed — falling back to AdamW for all parameters")

        if use_muon:
            muon_params = []
            adamw_params = []

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                # Apply Muon only to 2D weight matrices inside residual blocks
                # Specifically: model.blocks.*.net.0.weight and model.blocks.*.net.3.weight
                is_block_weight = (
                    "model.blocks" in name
                    and "net" in name
                    and ("net.0.weight" in name or "net.3.weight" in name)
                    and param.ndim >= 2
                )
                if is_block_weight:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

            param_groups = [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=self.muon_lr,
                    momentum=0.95,
                    weight_decay=self.weight_decay,
                ),
                dict(
                    params=adamw_params,
                    use_muon=False,
                    lr=self.adamw_lr,
                    betas=(0.9, 0.95),
                    weight_decay=self.weight_decay,
                ),
            ]

            optimizer = MuonWithAuxAdam(param_groups)
            self.print(
                f"Using Muon+AdamW: {len(muon_params)} Muon params, "
                f"{len(adamw_params)} AdamW params"
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.adamw_lr, weight_decay=self.weight_decay
            )

        # CosineAnnealingWarmRestarts: T_0=80 proven in parent (F1=0.4950)
        # T_mult=2 doubles cycle length: restarts at 80, 240, 560...
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.t0,
            T_mult=self.t_mult,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint: save only trainable params + small essential buffers
    # (string_embs and esm2_embs are large frozen tensors recomputed in setup())
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        saved: Dict[str, Any] = {}
        # Trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    saved[key] = full_sd[key]
        # Essential small buffers (class_weights); exclude large frozen embeddings
        large_frozen = {"string_embs", "esm2_embs"}
        for name, buf in self.named_buffers():
            leaf = name.split(".")[-1]
            if leaf not in large_frozen:
                key = prefix + name
                if key in full_sd:
                    saved[key] = full_sd[key]

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        self.print(
            f"Saving checkpoint: {n_trainable:,}/{n_total:,} trainable params "
            f"({100*n_trainable/n_total:.1f}%)"
        )
        return saved

    def load_state_dict(self, state_dict, strict=True):
        # strict=False: string_embs and esm2_embs not in checkpoint
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-F1 averaged over all genes — matches calc_metric.py logic.

    preds:  [N, 3, 6640] float — class logits
    labels: [N, 6640]    int   — class indices in {0,1,2}
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N, 6640]
    n_genes = labels.shape[1]
    f1_vals: List[float] = []
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
    """Save test predictions in required TSV format (idx / input / prediction)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # De-duplicate by pert_id (keep first occurrence)
    seen_ids = set()
    for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
        if pid not in seen_ids:
            seen_ids.add(pid)
            rows.append({
                "idx": pid,
                "input": sym,
                "prediction": json.dumps(preds[i].tolist()),  # [3][6640] list
            })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


def _run_ensemble_inference_on_gpu(
    checkpoint_paths: List[Path],
    datamodule: "PerturbDataModule",
    module_hparams: Dict[str, Any],
    out_path: Path,
    device_id: int = 0,
) -> None:
    """Run top-K checkpoint ensemble inference on GPU (rank 0 only).

    Loads each checkpoint into a fresh DualBranchModel, runs forward pass,
    averages logits across all checkpoints, and saves ensemble predictions.

    This is executed AFTER DDP training completes, so no process-group conflicts.
    We run on device_id=0 (the main GPU for rank 0).
    """
    if not checkpoint_paths:
        print("No checkpoints found for ensemble — skipping.")
        return

    device = torch.device(f"cuda:{device_id}")
    print(f"Running top-{len(checkpoint_paths)} checkpoint ensemble on {device}")

    # We need the full PerturbModule to call setup() and get the buffers
    # But we cannot use DDP here. Use a fresh PerturbModule with no DDP.
    ensemble_module = PerturbModule(**module_hparams)
    # Call setup manually to initialize string_embs and esm2_embs
    ensemble_module.setup("test")

    string_embs = ensemble_module.string_embs.to(device).float()
    esm2_embs = ensemble_module.esm2_embs.to(device).float()

    # Get test data
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    # Collect test pert_ids and symbols (consistent order)
    all_pert_ids: List[str] = []
    all_symbols: List[str] = []
    for batch in test_loader:
        all_pert_ids.extend(batch["pert_id"])
        all_symbols.extend(batch["symbol"])

    ensemble_logits: Optional[np.ndarray] = None
    n_ckpts_loaded = 0

    for ckpt_path in checkpoint_paths:
        print(f"  Loading checkpoint: {ckpt_path.name}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)

        # Build a fresh model and load weights
        fresh_model = DualBranchModel(
            fusion_dim=module_hparams["fusion_dim"],
            n_blocks=module_hparams["n_blocks"],
            dropout=module_hparams["dropout"],
            head_dropout=module_hparams["head_dropout"],
        ).to(device).float()

        # Strip "model." prefix if present (Lightning convention)
        filtered = {
            k[len("model."):] if k.startswith("model.") else k: v
            for k, v in state.items()
            if k.startswith("model.") or k in fresh_model.state_dict()
        }
        missing, unexpected = fresh_model.load_state_dict(filtered, strict=False)
        if missing:
            print(f"    Missing keys (expected): {missing[:3]}...")

        fresh_model.eval()
        batch_preds: List[np.ndarray] = []

        with torch.no_grad():
            for batch in test_loader:
                str_idx = batch["str_idx"].to(device)
                logits, _ = fresh_model(str_idx, string_embs, esm2_embs)
                batch_preds.append(logits.cpu().float().numpy())

        ckpt_preds = np.concatenate(batch_preds, axis=0)  # [N_test, 3, 6640]
        del fresh_model

        if ensemble_logits is None:
            ensemble_logits = ckpt_preds
        else:
            ensemble_logits = ensemble_logits + ckpt_preds
        n_ckpts_loaded += 1

    if ensemble_logits is None or n_ckpts_loaded == 0:
        print("No valid checkpoints loaded — skipping ensemble.")
        return

    # Average logits across checkpoints
    ensemble_logits = ensemble_logits / n_ckpts_loaded
    print(f"Ensemble from {n_ckpts_loaded} checkpoints, logits averaged")

    # De-duplicate and save
    _save_test_predictions(
        pert_ids=all_pert_ids,
        symbols=all_symbols,
        preds=ensemble_logits,
        out_path=out_path,
    )
    print(f"Ensemble predictions saved → {out_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-3-3-2: ESM2-650M + STRING_GNN Dual-Branch Gated Fusion"
    )
    p.add_argument("--micro-batch-size",    type=int,   default=32)
    p.add_argument("--global-batch-size",   type=int,   default=256)
    p.add_argument("--max-epochs",          type=int,   default=700)
    p.add_argument("--muon-lr",             type=float, default=0.01)
    p.add_argument("--adamw-lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",        type=float, default=8e-4)
    p.add_argument("--fusion-dim",          type=int,   default=FUSION_DIM)
    p.add_argument("--n-blocks",            type=int,   default=3)
    p.add_argument("--dropout",             type=float, default=0.35)
    p.add_argument("--head-dropout",        type=float, default=0.15)
    p.add_argument("--t0",                  type=int,   default=80)
    p.add_argument("--t-mult",              type=int,   default=2)
    p.add_argument("--mixup-prob",          type=float, default=0.65)
    p.add_argument("--mixup-alpha",         type=float, default=0.2)
    p.add_argument("--early-stop-patience", type=int,   default=160)
    p.add_argument("--num-workers",         type=int,   default=4)
    p.add_argument("--val-check-interval",  type=float, default=1.0)
    p.add_argument("--debug_max_step",      type=int,   default=None)
    p.add_argument("--fast_dev_run",        action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    pl.seed_everything(0)
    # Additional seed for numpy random (used in manifold mixup)
    np.random.seed(0)

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

    # --- LightningModule ---
    model = PerturbModule(
        fusion_dim=args.fusion_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        t0=args.t0,
        t_mult=args.t_mult,
        mixup_prob=args.mixup_prob,
        mixup_alpha=args.mixup_alpha,
    )

    # --- Trainer configuration ---
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
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
        filename="best-{epoch:03d}-{val_f1:.4f}",  # Underscore not slash in filename!
        monitor="val/f1",
        mode="max",
        save_top_k=5,   # Save top-5 for ensemble; enough diversity across cosine cycles
        save_last=True,
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
        strategy=DDPStrategy(
            find_unused_parameters=True,  # Safe: mixup path skips some head params
            timeout=timedelta(seconds=120),
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,  # Stabilize updates on large output head parameters
    )

    # --- Fit ---
    trainer.fit(model, datamodule=datamodule)

    # --- Standard test (single best checkpoint) ---
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # --- Top-K Checkpoint Ensemble (rank 0 only, after DDP teardown) ---
    if (
        trainer.is_global_zero
        and not args.fast_dev_run
        and args.debug_max_step is None
    ):
        ckpt_dir = output_dir / "checkpoints"
        # Find top-K checkpoint files — use underscore filename pattern
        ckpt_files = sorted(
            ckpt_dir.glob("best-*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Filter out 'last.ckpt' if present
        ckpt_files = [f for f in ckpt_files if "last" not in f.name]

        if len(ckpt_files) >= 2:
            print(f"Running top-{len(ckpt_files)} checkpoint ensemble...")
            module_hparams = {
                "fusion_dim": args.fusion_dim,
                "n_blocks": args.n_blocks,
                "dropout": args.dropout,
                "head_dropout": args.head_dropout,
                "muon_lr": args.muon_lr,
                "adamw_lr": args.adamw_lr,
                "weight_decay": args.weight_decay,
                "t0": args.t0,
                "t_mult": args.t_mult,
                "mixup_prob": args.mixup_prob,
                "mixup_alpha": args.mixup_alpha,
            }
            _run_ensemble_inference_on_gpu(
                checkpoint_paths=ckpt_files,
                datamodule=datamodule,
                module_hparams=module_hparams,
                out_path=output_dir / "test_predictions.tsv",
                device_id=0,
            )
        else:
            print(f"Only {len(ckpt_files)} checkpoint(s) found — skipping ensemble.")

    # --- Save test score (rank 0 only) ---
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")


if __name__ == "__main__":
    main()
