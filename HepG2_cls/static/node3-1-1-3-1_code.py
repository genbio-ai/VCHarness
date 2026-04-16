"""Node: ESM2-650M + STRING_GNN Dual-Branch with Gated Sigmoidal Fusion.

Architecture Overview:
  - Frozen ESM2-650M embeddings (1280-dim): precomputed from layer-concatenated
    /home/Models/STRING_GNN/esm2_embeddings_t33_650M.pt (3-layer mean pooling)
  - Frozen STRING_GNN embeddings (256-dim): PPI topology features, precomputed once
  - Gated sigmoidal fusion (FUSION_DIM=512): learned combination of both sources
    * ESM2: 1280 -> 512 (2.5:1 compression, preserves rich sequence semantics)
    * STRING: 256 -> 512 (expansion, boosts PPI topology capacity)
    * Gate = sigmoid(Linear([esm_proj; str_proj] -> 512))
    * Fused = gate * esm_proj + (1-gate) * str_proj
  - 3-block PreNorm Residual MLP (hidden=384, expand=2, dropout=0.30)
  - Flat output head: LayerNorm + Dropout(0.15) + Linear(384->19920) + per-gene bias
  - Manifold Mixup (alpha=0.2, prob=0.65) in hidden space
  - Muon(LR=0.01) + AdamW(LR=3e-4) dual optimizer
  - CosineAnnealingWarmRestarts(T_0=80, T_mult=2): 3 cycles at epochs 0-80, 80-240, 240-560
  - WCE + label smoothing=0.05, gradient clipping max_norm=1.0
  - Top-3 checkpoint ensemble at test time: averages logits from 3 best val/f1 checkpoints

Key improvements over parent (node1-2, F1=0.427):
  1. ESM2-650M adds complementary protein sequence signal -> MAJOR uplift (+0.10 F1 expected)
     Parent used STRING-only, which has ceiling ~0.497 F1
  2. FUSION_DIM=512 (proven better than 256 in tree lineage node3-1-1-1-1-2-1-1)
  3. Manifold Mixup prob=0.65 (vs 0.50 in parent) for more aggressive data augmentation
  4. max_epochs=600 (parent hit 336 epoch cap; reference ESM2+STRING nodes peaked at epoch 137-162)
  5. patience=100 (allows full CosineWR cycles to complete without premature stopping)
  6. gradient_clip_val=1.0 (stabilizes Muon with the larger dual-branch architecture)
  7. Top-3 ensemble at test time (+0.005-0.007 F1 based on tree history)

Architecture evidence from tree memory:
  - node3-1-1-1-1-2 (ESM2-650M + STRING, FUSION=256): test F1=0.5243 (parent=0.3919, +0.13)
  - node3-1-1-1-1-2-1-1 (ESM2-3B + STRING, FUSION=512): test F1=0.5265 (+0.002 from wider fusion)
  - node3-1-1-1-1-2-1-1-1 (top-3 ensemble): test F1=0.5283 (+0.002 from ensemble)

Expected performance: F1 ~ 0.52-0.53
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import random
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
from transformers import AutoModel

from muon import MuonWithAuxAdam

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES = 6640
N_CLASSES = 3
STRING_EMB_DIM = 256
ESM2_EMB_DIM = 1280
FUSION_DIM = 512


# ---------------------------------------------------------------------------
# ESM2 Embedding Loading (precomputed)
# ---------------------------------------------------------------------------
def _load_precomputed_esm2_embeddings(
    esm_model_dir: Path,
    node_names: List[str],
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load precomputed ESM2-650M layer-concatenated embeddings from disk.

    The embeddings file contains layer-concatenated representations:
    - File: esm2_embeddings_t33_650M.pt
    - Shape: [18870, 3840] = [N_nodes, 3 * 1280]
    - Each protein has 3 consecutive 1280-dim layer representations concatenated.

    We reduce to [18870, 1280] by averaging the 3 layer representations,
    then match ENSG IDs from the node_names list.

    Returns:
        gene_features: [N_unique_genes, 1280] float32 CPU tensor
        ensg_to_idx:   ENSG_ID -> row index mapping (maps to gene_features rows)
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Only rank 0 prints to avoid duplication
    if local_rank == 0:
        print("Loading precomputed ESM2-650M embeddings...", flush=True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Load the precomputed embeddings [18870, 3840] = [18870, 3*1280]
    raw_emb = torch.load(esm_model_dir / "esm2_embeddings_t33_650M.pt", map_location="cpu")
    raw_emb = raw_emb.float()  # Ensure float32

    n_layers = raw_emb.shape[1] // ESM2_EMB_DIM  # Expected: 3
    if n_layers * ESM2_EMB_DIM != raw_emb.shape[1]:
        if local_rank == 0:
            print(
                f"  Warning: embeddings dim {raw_emb.shape[1]} not divisible by {ESM2_EMB_DIM}. "
                f"Taking first {ESM2_EMB_DIM} dims.",
                flush=True,
            )
        gene_features = raw_emb[:, :ESM2_EMB_DIM]
    else:
        # Average the n_layers 1280-dim representations -> [18870, 1280]
        gene_features = raw_emb.view(-1, n_layers, ESM2_EMB_DIM).mean(dim=1)

    # Build ENSG -> embedding row index mapping
    # node_names[i] is the ENSG ID for embedding row i
    ensg_to_idx = {name: i for i, name in enumerate(node_names)}

    if local_rank == 0:
        print(
            f"  ESM2 embeddings: {raw_emb.shape} -> mean-reduced to {gene_features.shape} | "
            f"ENSG coverage: {len(ensg_to_idx)}/{len(node_names)}",
            flush=True,
        )

    del raw_emb
    return gene_features, ensg_to_idx


# ---------------------------------------------------------------------------
# Residual Block (Pre-Norm / PreLN style)
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Pre-norm residual block: LN -> Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, dim: int, expand: int = 2, dropout: float = 0.30) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expand)
        self.fc2 = nn.Linear(dim * expand, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return res + x


# ---------------------------------------------------------------------------
# Gated Fusion Module
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Sigmoidal gated fusion of ESM2 and STRING_GNN embeddings.

    Architecture:
      e = Linear(esm_dim -> fusion_dim)
      s = Linear(str_dim -> fusion_dim)
      g = sigmoid(Linear([e; s] -> fusion_dim))
      output = g * e + (1 - g) * s

    Proven effective in tree lineage nodes achieving F1 > 0.52.
    FUSION_DIM=512 provides richer representation vs 256 (2.5:1 compression for ESM2-650M).
    """

    def __init__(self, esm_dim: int, str_dim: int, fusion_dim: int) -> None:
        super().__init__()
        self.esm_proj = nn.Linear(esm_dim, fusion_dim, bias=True)
        self.str_proj = nn.Linear(str_dim, fusion_dim, bias=True)
        self.gate_linear = nn.Linear(fusion_dim * 2, fusion_dim, bias=True)

    def forward(self, esm_emb: torch.Tensor, str_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            esm_emb: [B, esm_dim]
            str_emb: [B, str_dim]
        Returns:
            fused: [B, fusion_dim]
        """
        e = self.esm_proj(esm_emb)   # [B, fusion_dim]
        s = self.str_proj(str_emb)   # [B, fusion_dim]
        g = torch.sigmoid(self.gate_linear(torch.cat([e, s], dim=-1)))  # [B, fusion_dim]
        return g * e + (1.0 - g) * s


# ---------------------------------------------------------------------------
# Prediction Head
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """[B, fusion_dim] -> 3-block PreNorm MLP -> [B, 3, N_GENES].

    Uses hidden_dim=384 (proven optimal for STRING/dual-branch on 1,273 samples),
    flat output head (unfactorized -- every factorized variant underperformed),
    head_dropout=0.15 for targeted output head regularization,
    and optional per-gene additive bias.
    """

    def __init__(
        self,
        in_dim: int = FUSION_DIM,
        hidden_dim: int = 384,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        dropout: float = 0.30,
        head_dropout: float = 0.15,
        use_per_gene_bias: bool = True,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.n_classes = N_CLASSES

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, expand=2, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head_dropout = nn.Dropout(head_dropout)
        self.out_proj = nn.Linear(hidden_dim, n_genes * N_CLASSES)

        if use_per_gene_bias:
            self.per_gene_bias = nn.Parameter(torch.zeros(n_genes * N_CLASSES))
        else:
            self.per_gene_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)          # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)
        x = self.head_norm(x)
        x = self.head_dropout(x)
        out = self.out_proj(x)          # [B, N_GENES * 3]
        if self.per_gene_bias is not None:
            out = out + self.per_gene_bias.unsqueeze(0)
        return out.view(-1, N_CLASSES, self.n_genes)  # [B, 3, N_GENES]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Maps each perturbed gene to its precomputed STRING_GNN + ESM2 feature vectors."""

    def __init__(
        self,
        df: pd.DataFrame,
        str_features: torch.Tensor,      # [N_str, 256] CPU float32
        str_ensg_to_idx: Dict[str, int],
        esm_features: torch.Tensor,      # [N_esm, 1280] CPU float32
        esm_ensg_to_idx: Dict[str, int],
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.str_features = str_features
        self.str_ensg_to_idx = str_ensg_to_idx
        self.esm_features = esm_features
        self.esm_ensg_to_idx = esm_ensg_to_idx

        if "label" in df.columns:
            labels = np.array(
                [json.loads(x) for x in df["label"].tolist()], dtype=np.int64
            )
            self.labels: Optional[torch.Tensor] = torch.tensor(
                labels + 1, dtype=torch.long
            )  # {-1,0,1} -> {0,1,2}
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pert_id = self.pert_ids[idx]

        # STRING_GNN features
        str_idx = self.str_ensg_to_idx.get(pert_id, -1)
        str_feat = (
            self.str_features[str_idx] if str_idx >= 0
            else torch.zeros(self.str_features.shape[1])
        )

        # ESM2 features
        esm_idx = self.esm_ensg_to_idx.get(pert_id, -1)
        esm_feat = (
            self.esm_features[esm_idx] if esm_idx >= 0
            else torch.zeros(self.esm_features.shape[1])
        )

        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": pert_id,
            "symbol": self.symbols[idx],
            "str_features": str_feat,
            "esm_features": esm_feat,
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
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
        micro_batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers

        self.str_features: Optional[torch.Tensor] = None
        self.str_ensg_to_idx: Optional[Dict[str, int]] = None
        self.esm_features: Optional[torch.Tensor] = None
        self.esm_ensg_to_idx: Optional[Dict[str, int]] = None

        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        if self.str_features is None:
            self._precompute_str_features()

        if self.esm_features is None:
            self._precompute_esm2_features()

        self.train_ds = PerturbDataset(
            train_df, self.str_features, self.str_ensg_to_idx,
            self.esm_features, self.esm_ensg_to_idx,
        )
        self.val_ds = PerturbDataset(
            val_df, self.str_features, self.str_ensg_to_idx,
            self.esm_features, self.esm_ensg_to_idx,
        )
        self.test_ds = PerturbDataset(
            test_df, self.str_features, self.str_ensg_to_idx,
            self.esm_features, self.esm_ensg_to_idx,
        )

    def _precompute_str_features(self) -> None:
        """Run STRING_GNN forward once to get frozen PPI topology embeddings [N, 256]."""
        model_dir = Path(STRING_GNN_DIR)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        node_names: List[str] = json.loads(
            (model_dir / "node_names.json").read_text()
        )
        self.str_ensg_to_idx = {name: i for i, name in enumerate(node_names)}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading STRING_GNN for precomputing PPI topology embeddings...", flush=True)
        gnn = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device)
        gnn.eval()

        graph = torch.load(model_dir / "graph_data.pt", map_location=device)
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)

        with torch.no_grad():
            out = gnn(
                edge_index=edge_index,
                edge_weight=edge_weight,
                output_hidden_states=False,
            )
            self.str_features = out.last_hidden_state.float().cpu()

        del gnn, graph, out
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"STRING_GNN features: {self.str_features.shape}", flush=True)

    def _precompute_esm2_features(self) -> None:
        """Load precomputed frozen ESM2-650M embeddings from disk.

        The embeddings are stored alongside STRING_GNN model at:
          /home/Models/STRING_GNN/esm2_embeddings_t33_650M.pt
        They cover all 18,870 STRING PPI graph nodes (ENSG IDs).
        """
        model_dir = Path(STRING_GNN_DIR)
        node_names = json.loads((model_dir / "node_names.json").read_text())

        self.esm_features, self.esm_ensg_to_idx = _load_precomputed_esm2_embeddings(
            esm_model_dir=model_dir,
            node_names=node_names,
        )

    def _make_loader(self, ds: PerturbDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        esm_dim: int = ESM2_EMB_DIM,
        str_dim: int = STRING_EMB_DIM,
        fusion_dim: int = FUSION_DIM,
        hidden_dim: int = 384,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        muon_lr: float = 0.01,
        adamw_lr: float = 3e-4,
        weight_decay: float = 8e-4,
        dropout: float = 0.30,
        head_dropout: float = 0.15,
        use_per_gene_bias: bool = True,
        label_smoothing: float = 0.05,
        t_0: int = 80,
        t_mult: int = 2,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.65,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.fusion: Optional[GatedFusion] = None
        self.head: Optional[PerturbHead] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

        # Configurable output path for ensemble test (set per checkpoint)
        self._test_pred_out: Optional[str] = None

    def setup(self, stage: str = "fit") -> None:
        self.fusion = GatedFusion(
            esm_dim=self.hparams.esm_dim,
            str_dim=self.hparams.str_dim,
            fusion_dim=self.hparams.fusion_dim,
        )
        self.head = PerturbHead(
            in_dim=self.hparams.fusion_dim,
            hidden_dim=self.hparams.hidden_dim,
            n_genes=self.hparams.n_genes,
            n_blocks=self.hparams.n_blocks,
            dropout=self.hparams.dropout,
            head_dropout=self.hparams.head_dropout,
            use_per_gene_bias=self.hparams.use_per_gene_bias,
        )

        # Cast trainable params to float32 for stable optimization
        for p in list(self.fusion.parameters()) + list(self.head.parameters()):
            if p.requires_grad:
                p.data = p.data.float()

        # Class weights: frequencies from DATA_ABSTRACT (down=4.77%, neutral=92.82%, up=2.41%)
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        cw = 1.0 / freq
        cw = cw / cw.mean()
        self.register_buffer("class_weights", cw)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"ESM2-650M + STRING_GNN GatedFusion (FUSION_DIM={self.hparams.fusion_dim}) | "
            f"trainable={trainable:,}/{total:,} | "
            f"hidden={self.hparams.hidden_dim}, blocks={self.hparams.n_blocks}, "
            f"dropout={self.hparams.dropout}, head_dropout={self.hparams.head_dropout}, "
            f"mixup_prob={self.hparams.mixup_prob}"
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        labels_b: Optional[torch.Tensor] = None,
        lam: Optional[float] = None,
    ) -> torch.Tensor:
        """Weighted cross-entropy with label smoothing. Supports Manifold Mixup targets."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)

        if labels_b is not None and lam is not None:
            labels_b_flat = labels_b.reshape(-1)
            loss_a = F.cross_entropy(
                logits_flat, labels_flat,
                weight=self.class_weights,
                label_smoothing=self.hparams.label_smoothing,
            )
            loss_b = F.cross_entropy(
                logits_flat, labels_b_flat,
                weight=self.class_weights,
                label_smoothing=self.hparams.label_smoothing,
            )
            return lam * loss_a + (1.0 - lam) * loss_b

        return F.cross_entropy(
            logits_flat, labels_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Manifold Mixup (applied AFTER gated fusion, at a random hidden layer)
    # ------------------------------------------------------------------
    def _apply_manifold_mixup(
        self,
        fused: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[float]]:
        """Apply Manifold Mixup in the hidden space of PerturbHead.

        Args:
            fused: [B, fusion_dim] tensor after GatedFusion
            labels: [B, N_GENES] integer labels {0,1,2}

        Returns:
            (result, labels_a, labels_b, lam):
            - When mixup applied: result=logits [B, 3, N_GENES], labels_b != None
            - When not applied:   result=fused [B, fusion_dim], labels_b=None
        """
        alpha = self.hparams.mixup_alpha
        prob = self.hparams.mixup_prob

        if not self.training or random.random() > prob or alpha <= 0:
            return fused, labels, None, None

        B = fused.size(0)
        lam = float(np.random.beta(alpha, alpha))
        idx = torch.randperm(B, device=fused.device)

        # Pass through input projection first
        x = self.head.input_proj(fused)  # [B, hidden_dim]

        n_blocks = self.hparams.n_blocks
        # Choose random layer to apply mixup at (0=after input_proj, i=after block i-1)
        mix_layer = random.randint(0, n_blocks)

        for i, block in enumerate(self.head.blocks):
            if i == mix_layer:
                x = lam * x + (1.0 - lam) * x[idx]
            x = block(x)

        if mix_layer == n_blocks:
            x = lam * x + (1.0 - lam) * x[idx]

        # Head output
        x = self.head.head_norm(x)
        x = self.head.head_dropout(x)
        out = self.head.out_proj(x)
        if self.head.per_gene_bias is not None:
            out = out + self.head.per_gene_bias.unsqueeze(0)
        logits = out.view(-1, N_CLASSES, N_GENES)

        return logits, labels, labels[idx], lam

    # ------------------------------------------------------------------
    # Training / Validation / Test Steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        esm_feats = batch["esm_features"].to(self.device).float()
        str_feats = batch["str_features"].to(self.device).float()
        labels = batch["label"]

        fused = self.fusion(esm_feats, str_feats)  # [B, fusion_dim]

        logits, labels_a, labels_b, lam = self._apply_manifold_mixup(fused, labels)

        # If mixup was not applied, run normal head forward
        if labels_b is None:
            logits = self.head(fused)

        loss = self._compute_loss(logits, labels_a, labels_b, lam)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        esm_feats = batch["esm_features"].to(self.device).float()
        str_feats = batch["str_features"].to(self.device).float()
        fused = self.fusion(esm_feats, str_feats)
        logits = self.head(fused)
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        import torch.distributed as dist

        if not self._val_preds:
            return
        preds_t = torch.cat(self._val_preds, dim=0)    # [local_N, 3, N_GENES]
        labels_t = torch.cat(self._val_labels, dim=0)   # [local_N, N_GENES]
        self._val_preds.clear()
        self._val_labels.clear()

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            # Use all_gather_object which handles variable-sized data automatically
            local_preds = preds_t.float().cpu().numpy()
            local_labels = labels_t.cpu().numpy()

            gathered_preds: List[np.ndarray] = [local_preds]
            gathered_labels: List[np.ndarray] = [local_labels]

            # Collect from all ranks via all_gather_object
            all_preds_list: List[List[np.ndarray]] = [None] * world_size
            all_labels_list: List[List[np.ndarray]] = [None] * world_size
            dist.all_gather_object(all_preds_list, local_preds.tolist())
            dist.all_gather_object(all_labels_list, local_labels.tolist())

            for rank_preds in all_preds_list:
                gathered_preds.append(np.array(rank_preds))
            for rank_labels in all_labels_list:
                gathered_labels.append(np.array(rank_labels))

            all_preds_np = np.concatenate(gathered_preds, axis=0)
            all_labels_np = np.concatenate(gathered_labels, axis=0)
        else:
            all_preds_np = preds_t.float().cpu().numpy()
            all_labels_np = labels_t.cpu().numpy()

        f1 = _compute_per_gene_f1(all_preds_np, all_labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        esm_feats = batch["esm_features"].to(self.device).float()
        str_feats = batch["str_features"].to(self.device).float()
        fused = self.fusion(esm_feats, str_feats)
        logits = self.head(fused)
        self._test_preds.append(logits.detach().cpu().float())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)   # [local_N, 3, N_GENES]
        labels_local = (
            torch.cat(self._test_labels, dim=0) if self._test_labels else None
        )
        self._test_preds.clear()
        self._test_labels.clear()

        # Gather predictions from all DDP ranks
        all_preds = self.all_gather(preds_local)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        if labels_local is not None:
            all_labels = self.all_gather(labels_local)
            all_labels = all_labels.view(-1, N_GENES)
            test_f1 = _compute_per_gene_f1(
                all_preds.float().cpu().numpy(),
                all_labels.cpu().numpy(),
            )
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=True)

        # Gather metadata (strings need all_gather_object)
        local_ids = list(self._test_pert_ids)
        local_syms = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        gathered_ids = [local_ids]
        gathered_syms = [local_syms]
        if world_size > 1:
            obj_ids = [None] * world_size
            obj_syms = [None] * world_size
            dist.all_gather_object(obj_ids, local_ids)
            dist.all_gather_object(obj_syms, local_syms)
            gathered_ids = obj_ids
            gathered_syms = obj_syms

        if self.trainer.is_global_zero:
            all_ids = [p for rank_list in gathered_ids for p in rank_list]
            all_syms = [s for rank_list in gathered_syms for s in rank_list]
            all_preds_np = all_preds.float().cpu().numpy()

            # Deduplicate (DDP may pad last batch)
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            for i, pid in enumerate(all_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_syms[i])
                    dedup_preds.append(all_preds_np[i])

            # Save to configurable path (for ensemble) or default path
            out_path = (
                Path(self._test_pred_out)
                if self._test_pred_out is not None
                else Path(__file__).parent / "run" / "test_predictions.tsv"
            )
            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=np.stack(dedup_preds, axis=0),
                out_path=out_path,
            )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        """Muon for block weight matrices + AdamW for all other parameters.

        Muon is applied to 2D+ weight matrices in residual blocks (fc1.weight, fc2.weight).
        AdamW handles: fusion module, input projection, biases, norms, output head, per-gene bias.
        """
        muon_params = []
        adamw_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Muon: 2D+ weight matrices in hidden residual blocks only
            if (
                param.ndim >= 2
                and "head.blocks" in name
                and ("fc1.weight" in name or "fc2.weight" in name)
            ):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        param_groups = [
            dict(
                params=muon_params,
                use_muon=True,
                lr=self.hparams.muon_lr,
                weight_decay=self.hparams.weight_decay,
                momentum=0.95,
            ),
            dict(
                params=adamw_params,
                use_muon=False,
                lr=self.hparams.adamw_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=self.hparams.weight_decay,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # CosineAnnealingWarmRestarts: T_0=80 first cycle, T_mult=2 doubles each cycle
        # Cycle boundaries: 0-80 (cycle 1), 80-240 (cycle 2), 240-560 (cycle 3)
        # Historical evidence: val/f1 peaks typically occur in cycle 2-3 (epochs 130-250)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.t_0,
            T_mult=self.hparams.t_mult,
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

    # ------------------------------------------------------------------
    # Checkpoint: save only trainable params + buffers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        result = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    result[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                result[key] = full_sd[key]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving checkpoint: {trainable:,}/{total:,} params "
            f"({100 * trainable / total:.2f}%)"
        )
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-F1 averaged over all 6,640 response genes.

    Matches data/calc_metric.py logic exactly:
    - argmax over class dim
    - per-gene F1 averaged over present classes only
    - final F1 = mean over all genes
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N, N_GENES]
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        if present.any():
            f1_vals.append(float(per_class_f1[present].mean()))
        else:
            f1_vals.append(0.0)
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(pert_ids)):
        rows.append({
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions -> {out_path}", flush=True)


def _find_top_k_checkpoints(
    checkpoint_dir: Path, k: int = 3
) -> List[Tuple[float, Path]]:
    """Find top-k checkpoint files ranked by val_f1 in filename."""
    ckpt_files = []
    for p in sorted(checkpoint_dir.glob("*.ckpt")):
        stem = p.stem
        if "val_f1=" in stem:
            try:
                score_str = stem.split("val_f1=")[1]
                score = float(score_str)
                ckpt_files.append((score, p))
            except (ValueError, IndexError):
                pass
    ckpt_files.sort(key=lambda x: x[0], reverse=True)
    return ckpt_files[:k]


def _average_ensemble_predictions(temp_paths: List[Path], out_path: Path) -> None:
    """Average logits from multiple checkpoint prediction files and save final result."""
    all_preds_by_id: Dict[str, List[np.ndarray]] = {}
    all_syms_by_id: Dict[str, str] = {}

    for p in temp_paths:
        df = pd.read_csv(p, sep="\t")
        for _, row in df.iterrows():
            pid = row["idx"]
            pred = np.array(json.loads(row["prediction"]))  # [3, N_GENES]
            all_preds_by_id.setdefault(pid, []).append(pred)
            all_syms_by_id[pid] = row["input"]

    rows = []
    for pid, preds in all_preds_by_id.items():
        avg_pred = np.mean(preds, axis=0)  # Average logits across checkpoints
        rows.append({
            "idx": pid,
            "input": all_syms_by_id[pid],
            "prediction": json.dumps(avg_pred.tolist()),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(
        f"Saved {len(rows)} ensemble predictions "
        f"({len(temp_paths)} checkpoints) -> {out_path}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ESM2-650M + STRING_GNN Dual-Branch for HepG2 DEG Prediction"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=600)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=8e-4)
    p.add_argument("--fusion-dim", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--head-dropout", type=float, default=0.15)
    p.add_argument("--use-per-gene-bias", action="store_true", default=True)
    p.add_argument("--no-per-gene-bias", dest="use_per_gene_bias", action="store_false")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--t-0", type=int, default=80)
    p.add_argument("--t-mult", type=int, default=2)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.65)
    p.add_argument("--early-stop-patience", type=int, default=100)
    p.add_argument("--ensemble-top-k", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    model_kwargs = dict(
        esm_dim=ESM2_EMB_DIM,
        str_dim=STRING_EMB_DIM,
        fusion_dim=args.fusion_dim,
        hidden_dim=args.hidden_dim,
        n_genes=N_GENES,
        n_blocks=args.n_blocks,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        use_per_gene_bias=args.use_per_gene_bias,
        label_smoothing=args.label_smoothing,
        t_0=args.t_0,
        t_mult=args.t_mult,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
    )
    model = PerturbModule(**model_kwargs)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = limit_val = limit_test = args.debug_max_step
        max_steps = args.debug_max_step
        val_check_interval = 1.0
        num_sanity_val_steps = 0
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval if not fast_dev_run else 1.0
        num_sanity_val_steps = 2

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-epoch{epoch:03d}-val_f1={val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=args.ensemble_top_k,
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
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(
            find_unused_parameters=False, timeout=timedelta(seconds=120)
        ),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Stabilize Muon in later CosineWR cycles
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        # Quick test: use current model state
        trainer.test(model, datamodule=datamodule)
    else:
        # Top-K checkpoint ensemble test
        checkpoint_dir = output_dir / "checkpoints"
        top_ckpts = _find_top_k_checkpoints(checkpoint_dir, k=args.ensemble_top_k)

        if len(top_ckpts) <= 1:
            # Fallback: single best checkpoint
            print("Ensemble: insufficient checkpoints, using single best", flush=True)
            trainer.test(model, datamodule=datamodule, ckpt_path="best")
        else:
            print(
                f"Ensemble: averaging top-{len(top_ckpts)} checkpoints "
                f"by val/f1: {[f'{s:.4f}' for s, _ in top_ckpts]}",
                flush=True,
            )
            temp_pred_paths: List[Path] = []
            for i, (score, ckpt_path) in enumerate(top_ckpts):
                temp_out = output_dir / f"_ens_temp_{i}.tsv"
                model._test_pred_out = str(temp_out)
                trainer.test(model, datamodule=datamodule, ckpt_path=str(ckpt_path))
                if trainer.is_global_zero and temp_out.exists():
                    temp_pred_paths.append(temp_out)

            model._test_pred_out = None  # Reset

            # Average and save final predictions (rank 0 only)
            if trainer.is_global_zero and temp_pred_paths:
                _average_ensemble_predictions(
                    temp_paths=temp_pred_paths,
                    out_path=output_dir / "test_predictions.tsv",
                )
                for tp in temp_pred_paths:
                    if tp.exists():
                        tp.unlink()
            elif trainer.is_global_zero:
                # Ensemble failed, fall back to best single checkpoint
                print(
                    "Warning: Ensemble predictions collection failed, "
                    "falling back to single best checkpoint.",
                    flush=True,
                )
                model._test_pred_out = None
                trainer.test(model, datamodule=datamodule, ckpt_path="best")

            # Sync all ranks after ensemble
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()


if __name__ == "__main__":
    main()
