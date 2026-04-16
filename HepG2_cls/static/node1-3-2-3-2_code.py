"""Node 1-2: ESM2-650M (precomputed, frozen) + STRING_GNN dual-branch + Gated Fusion + Manifold Mixup

Key changes from parent (node1-3-2-3, F1=0.4635):
  1. PIVOT to ESM2-650M + STRING dual-branch architecture
     — Leverage precomputed ESM2-650M embeddings [18870, 3840] available at STRING_GNN_DIR
     — STRING-only nodes conclusively capped at ~0.46-0.47 F1 (both parent and sibling failed)
     — Best dual-branch nodes: node3-1-1-1-1-2 (F1=0.5243), node3-3-1-2-1 (F1=0.5170)
  2. ESM2 two-stage projection: 3840→1024→512 (richer than single-stage, less compression)
     — node3-3-1-2-1-1-1-1 used single-stage 1280→256 (aggressive) and got F1=0.5219
     — Two-stage preserves more information, addressing the compression bottleneck
  3. STRING projection: 256→512
  4. Gated fusion (learnable sigmoidal gate over 512-dim concatenation → 512)
  5. Manifold Mixup (alpha=0.2, prob=0.65) applied after fusion — proven effective for ESM2+STRING
     — node3-3-1-2-1 (F1=0.5170) with prob=0.65 outperformed prob=0.50
  6. CosineAnnealingWarmRestarts (T_0=80, T_mult=2) — proven schedule for dual-branch + Mixup
  7. Muon optimizer for hidden MLP block weights, AdamW for rest
  8. head_dropout=0.15, weight_decay=8e-4 — proven configuration from high-F1 nodes

Tree context:
  node1-3-2-3 (parent)      | F1=0.4635 | STRING+Mixup+RLROP — STRING ceiling
  node1-3-2-3-1 (sibling)   | F1=0.4635 | STRING+Mixup+CosineWR — STRING ceiling
  node3-3-1-2-1 (reference) | F1=0.5170 | ESM2-150M+STRING+gated+Mixup+CosineWR
  node3-1-1-1-1-2 (ref2)   | F1=0.5243 | ESM2-650M+STRING+gated+Mixup+CosineWR (top-5 ensemble)
  This node targets F1 > 0.510 via dual-branch ESM2-650M+STRING fusion
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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
ESM2_650M_EMB_PATH = STRING_GNN_DIR / "esm2_embeddings_t33_650M.pt"  # [18870, 3840]
ESM2_DIM = 3840   # ESM2-650M precomputed embedding dimension
STRING_DIM = 256  # STRING_GNN last hidden state dimension
FUSION_DIM = 512  # Fused dimension for both branches


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
        micro_batch_size: int = 16,
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
class ResidualBlock(nn.Module):
    """Pre-LayerNorm residual MLP block (hidden_dim → hidden_dim*2 → hidden_dim)."""

    def __init__(self, dim: int, inner_dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(self.norm(x)))


class GatedFusion(nn.Module):
    """Learnable sigmoidal gated fusion of two branches.

    Both branches are projected to FUSION_DIM before the gate.
    The gate is a sigmoid applied to a linear transformation of the concatenation.
    Output: element-wise weighted combination of the two projected inputs.

    This architecture matches the proven fusion in node3-1-1-1-1-2 (F1=0.5243)
    and node3-3-1-2-1 (F1=0.5170).
    """

    def __init__(self, fusion_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(fusion_dim * 2, fusion_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: [B, fusion_dim] — first branch (e.g., ESM2 projected)
        x2: [B, fusion_dim] — second branch (e.g., STRING projected)
        Returns: [B, fusion_dim] fused representation
        """
        concat = torch.cat([x1, x2], dim=-1)  # [B, 2*fusion_dim]
        gate = self.sigmoid(self.gate(concat))  # [B, fusion_dim]
        return gate * x1 + (1 - gate) * x2     # element-wise weighted sum


class DualBranchPerturbMLP(nn.Module):
    """ESM2-650M (precomputed, frozen) + STRING_GNN (frozen) dual-branch MLP.

    Architecture:
      ① ESM2-650M lookup [18870, 3840] (precomputed frozen buffer)
         Fallback: learnable 3840-dim for genes not in STRING graph
         Two-stage projection: 3840→1024→FUSION_DIM (preserves more info than 1-stage)
      ② STRING_GNN lookup [18870, 256] (frozen buffer)
         Fallback: learnable 256-dim
         Single projection: 256→FUSION_DIM
      ③ Gated fusion: sigmoid gate over concatenation → [B, FUSION_DIM]
         [Manifold Mixup applied here during training]
      ④ n_blocks × ResidualBlock(FUSION_DIM, inner=FUSION_DIM*2)
      ⑤ head_dropout + LN + Linear(FUSION_DIM→N_GENES*N_CLASSES) + per-gene bias
      ⑥ reshape → [B, 3, 6640]
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.30,
        head_dropout: float = 0.15,
    ) -> None:
        super().__init__()

        # ESM2-650M branch: fallback learnable embedding for genes not in STRING graph
        self.esm2_fallback = nn.Parameter(torch.zeros(ESM2_DIM))
        nn.init.normal_(self.esm2_fallback, std=0.02)

        # Two-stage ESM2 projection: 3840 → 1024 → FUSION_DIM
        # Rationale: node3-3-1-2-1-1-1-1 single-stage 2560→256 (10x) was aggressive;
        # two-stage 3840→1024→512 (7.5x) preserves richer representations
        self.esm2_proj = nn.Sequential(
            nn.Linear(ESM2_DIM, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, FUSION_DIM),
            nn.LayerNorm(FUSION_DIM),
            nn.GELU(),
        )

        # STRING branch: fallback learnable embedding
        self.string_fallback = nn.Parameter(torch.zeros(STRING_DIM))
        nn.init.normal_(self.string_fallback, std=0.02)

        # STRING projection: 256 → FUSION_DIM
        self.string_proj = nn.Sequential(
            nn.Linear(STRING_DIM, FUSION_DIM),
            nn.LayerNorm(FUSION_DIM),
            nn.GELU(),
        )

        # Gated fusion
        self.gated_fusion = GatedFusion(fusion_dim=FUSION_DIM)

        # Feed-forward from fused representation to hidden_dim
        # Note: if hidden_dim == FUSION_DIM, this is an identity-like projection
        # We project FUSION_DIM → hidden_dim before the residual blocks
        self.fusion_to_hidden = nn.Sequential(
            nn.Linear(FUSION_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Residual MLP blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim * 2, dropout) for _ in range(n_blocks)]
        )

        # Output head: head_dropout + LayerNorm + flat Linear + per-gene additive bias
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, N_GENES * N_CLASSES),
        )
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES * N_CLASSES))

    def _lookup_esm2(
        self,
        str_idx: torch.Tensor,       # [B]
        esm2_embs: torch.Tensor,     # [18870, 3840] frozen buffer
    ) -> torch.Tensor:
        """Look up ESM2 embeddings with fallback for genes not in STRING."""
        valid_mask = str_idx >= 0
        safe_idx = str_idx.clamp(min=0)
        emb = esm2_embs[safe_idx].to(self.esm2_fallback)  # [B, 3840]
        if not valid_mask.all():
            fallback = self.esm2_fallback.unsqueeze(0).expand(
                int((~valid_mask).sum()), -1
            )
            emb = emb.clone()
            emb[~valid_mask] = fallback
        return emb  # [B, 3840]

    def _lookup_string(
        self,
        str_idx: torch.Tensor,       # [B]
        string_embs: torch.Tensor,   # [18870, 256] frozen buffer
    ) -> torch.Tensor:
        """Look up STRING embeddings with fallback for genes not in STRING."""
        valid_mask = str_idx >= 0
        safe_idx = str_idx.clamp(min=0)
        emb = string_embs[safe_idx].to(self.string_fallback)  # [B, 256]
        if not valid_mask.all():
            fallback = self.string_fallback.unsqueeze(0).expand(
                int((~valid_mask).sum()), -1
            )
            emb = emb.clone()
            emb[~valid_mask] = fallback
        return emb  # [B, 256]

    def get_fused_embedding(
        self,
        str_idx: torch.Tensor,
        esm2_embs: torch.Tensor,
        string_embs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute fused embedding (after gated fusion). Used by Manifold Mixup."""
        esm2_raw = self._lookup_esm2(str_idx, esm2_embs)    # [B, 3840]
        string_raw = self._lookup_string(str_idx, string_embs)  # [B, 256]

        esm2_proj = self.esm2_proj(esm2_raw)        # [B, FUSION_DIM]
        string_proj = self.string_proj(string_raw)   # [B, FUSION_DIM]

        fused = self.gated_fusion(esm2_proj, string_proj)  # [B, FUSION_DIM]
        return fused  # [B, FUSION_DIM]

    def forward_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        """Run from fused embedding through MLP to logits."""
        x = self.fusion_to_hidden(fused)  # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)
        x = self.head_dropout(x)
        logits = self.head(x) + self.gene_bias.to(x)  # [B, N_GENES * N_CLASSES]
        return logits.view(-1, N_CLASSES, N_GENES)     # [B, 3, 6640]

    def forward(
        self,
        str_idx: torch.Tensor,
        esm2_embs: torch.Tensor,
        string_embs: torch.Tensor,
        mixup_lam: Optional[float] = None,
        mixup_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional Manifold Mixup in the fused embedding space."""
        fused = self.get_fused_embedding(str_idx, esm2_embs, string_embs)

        # Apply Manifold Mixup: mix in the fused representation space
        if mixup_lam is not None and mixup_index is not None:
            fused = mixup_lam * fused + (1 - mixup_lam) * fused[mixup_index]

        return self.forward_from_fused(fused)


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.30,
        head_dropout: float = 0.15,
        lr: float = 3e-4,
        muon_lr: float = 0.01,
        weight_decay: float = 8e-4,
        label_smoothing: float = 0.0,
        grad_clip_norm: float = 1.0,
        use_muon: bool = True,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.65,
        cosine_t0: int = 80,
        cosine_tmult: int = 2,
        cosine_eta_min: float = 1e-6,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.lr = lr
        self.muon_lr = muon_lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.grad_clip_norm = grad_clip_norm
        self.use_muon = use_muon
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.cosine_t0 = cosine_t0
        self.cosine_tmult = cosine_tmult
        self.cosine_eta_min = cosine_eta_min

        # Model is initialized here in __init__ (not setup) to ensure all parameters
        # exist at optimizer construction time during DDP setup.
        self.model = DualBranchPerturbMLP(
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
        )

        # Accumulation buffers for epoch-level metrics
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Class weights: inversely proportional to class frequencies
        # class0=neutral(92.82%), class1=down(4.77%), class2=up(2.41%) after {-1,0,1}→{0,1,2}
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq)
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        self.register_buffer("class_weights", class_weights)

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

        # ---- Load precomputed ESM2-650M embeddings (once per rank) ----
        # Shape: [18870, 3840] — precomputed from ESM2-650M, indexed by STRING node index
        esm2_embs = torch.load(
            ESM2_650M_EMB_PATH, map_location="cpu", weights_only=False
        ).float()  # [18870, 3840]
        self.register_buffer("esm2_embs", esm2_embs)
        self.print(f"Loaded ESM2-650M precomputed embeddings: {esm2_embs.shape}")

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        self.print(
            f"Node1-2 DualBranchPerturbMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
            f"dropout={self.dropout} | head_dropout={self.head_dropout} | "
            f"use_muon={self.use_muon} | muon_lr={self.muon_lr} | "
            f"mixup_alpha={self.mixup_alpha} | mixup_prob={self.mixup_prob} | "
            f"cosine_t0={self.cosine_t0} | cosine_tmult={self.cosine_tmult} | "
            f"trainable={n_trainable:,}/{n_total:,}"
        )

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor,
        lam: Optional[float] = None, index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Weighted cross-entropy loss with optional Manifold Mixup targets.

        For Mixup: loss = lam * CE(logits, labels) + (1-lam) * CE(logits, shuffled_labels)
        logits: [B, 3, 6640]
        labels: [B, 6640]  — values in {0, 1, 2}
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                                # [B*6640]

        if lam is not None and index is not None:
            labels_shuffled_flat = labels[index].reshape(-1)  # [B*6640]
            loss_a = F.cross_entropy(
                logits_flat, labels_flat,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
            loss_b = F.cross_entropy(
                logits_flat, labels_shuffled_flat,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
            return lam * loss_a + (1 - lam) * loss_b
        else:
            return F.cross_entropy(
                logits_flat,
                labels_flat,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Manifold Mixup: apply with probability mixup_prob in the fused embedding space
        if self.training and self.mixup_alpha > 0 and self.mixup_prob > 0:
            if torch.rand(1).item() < self.mixup_prob:
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                B = batch["str_idx"].shape[0]
                index = torch.randperm(B, device=batch["str_idx"].device)
                logits = self.model(
                    batch["str_idx"], self.esm2_embs, self.string_embs,
                    mixup_lam=lam, mixup_index=index,
                )
                loss = self._compute_loss(logits, batch["label"], lam=lam, index=index)
            else:
                logits = self.model(batch["str_idx"], self.esm2_embs, self.string_embs)
                loss = self._compute_loss(logits, batch["label"])
        else:
            logits = self.model(batch["str_idx"], self.esm2_embs, self.string_embs)
            loss = self._compute_loss(logits, batch["label"])

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["str_idx"], self.esm2_embs, self.string_embs)
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
            # With ws=1 all_gather prepends a size-1 dim
            if all_preds.dim() == 4:
                all_preds = all_preds.squeeze(0)
                all_labels = all_labels.squeeze(0)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["str_idx"], self.esm2_embs, self.string_embs)
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

        # Gather string metadata
        if ws > 1:
            all_pert_ids, all_symbols = _gather_string_lists(
                self._test_pert_ids, self._test_symbols, ws, self.device
            )
        else:
            all_pert_ids = self._test_pert_ids
            all_symbols = self._test_symbols

        if self.trainer.is_global_zero:
            preds_np = all_preds.float().cpu().numpy()  # [N_total, 3, 6640]
            # Deduplicate in case DistributedSampler added padding rows
            seen = set()
            dedup_rows = []
            for pid, sym, pred in zip(all_pert_ids, all_symbols, preds_np):
                if pid not in seen:
                    seen.add(pid)
                    dedup_rows.append((pid, sym, pred))
            dedup_pert_ids = [r[0] for r in dedup_rows]
            dedup_symbols = [r[1] for r in dedup_rows]
            dedup_preds = np.stack([r[2] for r in dedup_rows])
            _save_test_predictions(
                pert_ids=dedup_pert_ids,
                symbols=dedup_symbols,
                preds=dedup_preds,
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        """Configure optimizer: Muon for hidden MLP weight matrices, AdamW for everything else.

        Muon is applied to 2D weight matrices in the residual blocks only.
        All projection/fusion/head parameters use AdamW.
        CosineAnnealingWarmRestarts provides deterministic LR cycles independent of val/F1.
        """
        if self.use_muon:
            try:
                from muon import MuonWithAuxAdam
                muon_available = True
            except ImportError:
                self.print("Warning: muon not installed, falling back to AdamW")
                muon_available = False
        else:
            muon_available = False

        if muon_available and self.use_muon:
            # Identify hidden MLP weight matrices (Linear weights in residual blocks only)
            hidden_weight_names = set()
            for name, param in self.model.named_parameters():
                if (param.ndim >= 2
                        and "blocks." in name
                        and ".weight" in name
                        and "norm" not in name):
                    hidden_weight_names.add(name)

            hidden_weights = []
            other_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                model_name = name[len("model."):] if name.startswith("model.") else name
                if model_name in hidden_weight_names:
                    hidden_weights.append(param)
                else:
                    other_params.append(param)

            self.print(
                f"Muon params: {sum(p.numel() for p in hidden_weights):,} | "
                f"AdamW params: {sum(p.numel() for p in other_params):,}"
            )

            param_groups = [
                dict(
                    params=hidden_weights,
                    use_muon=True,
                    lr=self.muon_lr,
                    weight_decay=self.weight_decay,
                    momentum=0.95,
                ),
                dict(
                    params=other_params,
                    use_muon=False,
                    lr=self.lr,
                    betas=(0.9, 0.95),
                    weight_decay=self.weight_decay,
                ),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        # CosineAnnealingWarmRestarts: deterministic, independent of val/F1 trajectory
        # Proven schedule for dual-branch + Mixup training:
        # node3-1-1-1-1-2 (F1=0.5243), node3-3-1-2-1 (F1=0.5170)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cosine_t0,
            T_mult=self.cosine_tmult,
            eta_min=self.cosine_eta_min,
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
    # (string_embs and esm2_embs are large frozen tensors — exclude from checkpoint)
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        saved: Dict[str, Any] = {}
        if full_sd is None:
            return {}
        # Trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    saved[key] = full_sd[key]
        # Essential small buffers (class_weights); exclude large frozen buffers
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
        # strict=False: string_embs and esm2_embs not in checkpoint but populated by setup()
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


def _gather_string_lists(
    local_ids: List[str],
    local_syms: List[str],
    world_size: int,
    device: torch.device,
) -> tuple:
    """Gather string lists across DDP ranks using tensor encoding."""
    MAX_LEN = 64

    def encode_strings(strings: List[str], max_len: int) -> torch.Tensor:
        result = torch.zeros(len(strings), max_len, dtype=torch.int32)
        for i, s in enumerate(strings):
            chars = [min(ord(c), 127) for c in s[:max_len]]
            result[i, :len(chars)] = torch.tensor(chars, dtype=torch.int32)
        return result

    def decode_strings(tensor: torch.Tensor) -> List[str]:
        strings = []
        for row in tensor:
            chars = [chr(int(c)) for c in row.tolist() if c > 0]
            strings.append("".join(chars))
        return strings

    ids_tensor = encode_strings(local_ids, MAX_LEN).to(device)
    syms_tensor = encode_strings(local_syms, MAX_LEN).to(device)

    local_n = torch.tensor([len(local_ids)], device=device, dtype=torch.long)
    all_ns = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
    torch.distributed.all_gather(all_ns, local_n)
    max_n = int(max(n.item() for n in all_ns))

    if ids_tensor.shape[0] < max_n:
        pad_ids = torch.zeros(max_n - ids_tensor.shape[0], MAX_LEN, dtype=torch.int32, device=device)
        pad_syms = torch.zeros(max_n - syms_tensor.shape[0], MAX_LEN, dtype=torch.int32, device=device)
        ids_tensor = torch.cat([ids_tensor, pad_ids], dim=0)
        syms_tensor = torch.cat([syms_tensor, pad_syms], dim=0)

    all_ids_tensors = [torch.zeros(max_n, MAX_LEN, dtype=torch.int32, device=device) for _ in range(world_size)]
    all_syms_tensors = [torch.zeros(max_n, MAX_LEN, dtype=torch.int32, device=device) for _ in range(world_size)]
    torch.distributed.all_gather(all_ids_tensors, ids_tensor)
    torch.distributed.all_gather(all_syms_tensors, syms_tensor)

    all_pert_ids: List[str] = []
    all_symbols: List[str] = []
    for rank_idx in range(world_size):
        n_items = int(all_ns[rank_idx].item())
        all_pert_ids.extend(decode_strings(all_ids_tensors[rank_idx][:n_items].cpu()))
        all_symbols.extend(decode_strings(all_syms_tensors[rank_idx][:n_items].cpu()))

    return all_pert_ids, all_symbols


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    """Save test predictions in required TSV format (idx / input / prediction)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
        rows.append({
            "idx": pid,
            "input": sym,
            "prediction": json.dumps(preds[i].tolist()),  # [3][6640] list
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-2: ESM2-650M (precomputed) + STRING dual-branch + Gated Fusion + Manifold Mixup"
    )
    p.add_argument("--micro-batch-size",    type=int,   default=16)
    p.add_argument("--global-batch-size",   type=int,   default=256)
    p.add_argument("--max-epochs",          type=int,   default=700)
    p.add_argument("--lr",                  type=float, default=3e-4)
    p.add_argument("--muon-lr",             type=float, default=0.01)
    p.add_argument("--weight-decay",        type=float, default=8e-4)
    p.add_argument("--hidden-dim",          type=int,   default=384)
    p.add_argument("--n-blocks",            type=int,   default=3)
    p.add_argument("--dropout",             type=float, default=0.30)
    p.add_argument("--head-dropout",        type=float, default=0.15)
    p.add_argument("--label-smoothing",     type=float, default=0.0)
    p.add_argument("--grad-clip-norm",      type=float, default=1.0)
    p.add_argument("--mixup-alpha",         type=float, default=0.2)
    p.add_argument("--mixup-prob",          type=float, default=0.65)
    p.add_argument("--cosine-t0",           type=int,   default=80)
    p.add_argument("--cosine-tmult",        type=int,   default=2)
    p.add_argument("--cosine-eta-min",      type=float, default=1e-6)
    p.add_argument("--early-stop-patience", type=int,   default=120)
    p.add_argument("--no-muon",             action="store_true",
                   help="Disable Muon optimizer, fall back to AdamW for all params")
    p.add_argument("--no-mixup",            action="store_true",
                   help="Disable Manifold Mixup augmentation")
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

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve mixup settings
    mixup_alpha = 0.0 if args.no_mixup else args.mixup_alpha
    mixup_prob = 0.0 if args.no_mixup else args.mixup_prob

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
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        lr=args.lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip_norm,
        use_muon=not args.no_muon,
        mixup_alpha=mixup_alpha,
        mixup_prob=mixup_prob,
        cosine_t0=args.cosine_t0,
        cosine_tmult=args.cosine_tmult,
        cosine_eta_min=args.cosine_eta_min,
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
        filename="best-epoch={epoch:03d}-val_f1={val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=5,   # Save top-5 for potential ensemble (not used in default test)
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
        strategy=DDPStrategy(
            find_unused_parameters=True,  # Some forward paths may skip blocks
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
        gradient_clip_val=args.grad_clip_norm,
    )

    # --- Fit ---
    trainer.fit(model, datamodule=datamodule)

    # --- Test ---
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # --- Save test score (rank 0 only) ---
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")


if __name__ == "__main__":
    main()
