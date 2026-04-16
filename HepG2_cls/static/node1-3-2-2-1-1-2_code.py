"""Node 1-3-2-2-1-1-2: STRING-only + Flat Head + Hidden=384 + Muon(LR=0.01) + CosineWarmRestarts + Manifold Mixup

Key changes from parent (node1-3-2-2-1-1, F1=0.4746 — REGRESSION):
  1. Revert head_dropout: 0.20 → 0.15 — parent feedback confirms p=0.20 crossed the
     over-regularization threshold, yielding -0.0031 regression. p=0.15 is the proven optimal.
  2. Revert trunk dropout: 0.28 → 0.30 — restored to grandparent's proven setting.
  3. Switch scheduler: RLROP(patience=10) → CosineAnnealingWarmRestarts(T_0=80, T_mult=2)
     — The node1-3-2-2-1-1-1-1-1-1 lineage achieved F1=0.4968 using this schedule,
     which escapes local optima through warm restarts at epochs 80 and 240.
     RLROP at the current tree position (~F1=0.477) has stalled; warm restarts are needed.
  4. Add Manifold Mixup (alpha=0.2, prob=0.5) — node1-3-2-2-1-1-1-1-1-1 confirmed this
     reduces train-val loss gap significantly (0.172→0.152) by doubling effective training
     set through embedding interpolation, transformative for 1,273-sample datasets.
  5. Muon LR: 0.02 → 0.01 — node1-3-2-2-1-1-1-1 confirmed that LR=0.01 enables proper
     LR adaptation via halvings (3 halvings at epochs 96/132/148) while LR=0.02 with RLROP
     kept LR constant throughout training due to val/f1 oscillations.
  6. Remove label smoothing (0.05 → 0.0) — node1-3-2-2-1-1-1-1 showed removing label
     smoothing allowed train/loss to reach 0.058 (3.5× lower), enabling more precise fitting
     and F1=0.4914 vs parent's 0.4879 with smoothing.
  7. weight_decay: 5e-4 → 8e-4 — matching the best-performing node1-3-2-2-1-1-1-1-1-1
     configuration (F1=0.4968) which found this weight_decay optimal with CosineWR.
  8. Class weights: keep α=1.0 (inverse frequency) — node1-3-2-2-1-1-1 confirmed that
     softened weights α=0.5 create spurious val/f1 inflation misleading checkpoint selection.
  9. top-5 checkpoint ensemble at test — proven +0.002-0.004 F1 in node1-3-2-2-1-1-1-1
     by spanning multiple CosineWR phases providing genuine diversity.
 10. max_epochs=500, early_stop_patience=120 — CosineWR needs long horizon; best epoch
     in ancestor was 329/469 with T_0=80/T_mult=2. Early stop 120 allows node to see
     3 full restart cycles before stopping.

Tree context:
  node1-3-2-2-1 (great-grandparent) | F1=0.4777 | RLROP + head_dropout=0.15 [old tree best]
  node1-3-2-2-1-1 (parent)          | F1=0.4746 | RLROP + head_dropout=0.20 [REGRESSION]
  node1-3-2-2-1-1-1 (sibling)       | TBD       | RLROP + head_dropout=0.15 + softened weights
  node1-3-2-2-1-1-1-1-1-1 (cousin)  | F1=0.4968 | CosineWR + Manifold Mixup [proven best lineage]
  This node targets F1 > 0.4968 via CosineWR + Manifold Mixup + Muon(lr=0.01) + no label smoothing
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
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
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3
STRING_GNN_DIR = Path("/home/Models/STRING_GNN")


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
class ResidualBlock(nn.Module):
    """Pre-LayerNorm residual MLP block (hidden_dim → hidden_dim*2 → hidden_dim)."""

    def __init__(self, dim: int, dropout: float = 0.30) -> None:
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


class PerturbMLP(nn.Module):
    """STRING-only MLP with output head dropout for gene perturbation response prediction.

    Architecture (per sample):
      ① STRING_GNN embedding lookup [256-dim, frozen buffer]
         (fallback learnable 256-dim for ~6% genes not in STRING)
      ② Input projection: Linear(256→hidden_dim) + LN + GELU
      ③ n_blocks × ResidualBlock(hidden_dim)
      ④ LN(hidden_dim) → Dropout(head_dropout) → Linear(hidden_dim → 6640*3) + per-gene-bias
      ⑤ reshape → [B, 3, 6640]

    Key differences vs parent (node1-3-2-2-1-1):
      - head_dropout=0.15 (reverted from 0.20): proven optimal from grandparent
      - dropout=0.30 (reverted from 0.28): proven optimal from grandparent
      - Training uses CosineWarmRestarts + Manifold Mixup (not RLROP)
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.30,
        head_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        # Learnable fallback embedding for genes not in STRING graph
        self.fallback_emb = nn.Parameter(torch.zeros(256))
        nn.init.normal_(self.fallback_emb, std=0.02)

        # Input projection: 256 → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Residual MLP blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        # Flat output head + per-gene additive bias
        # head_dropout=0.15 (proven optimal from node1-3-2-2-1, F1=0.4777)
        # Flat head confirmed superior over factorized heads across 4+ nodes.
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(head_dropout),           # Output head regularization (p=0.15)
            nn.Linear(hidden_dim, N_GENES * N_CLASSES),
        )
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES * N_CLASSES))

    def get_embedding(
        self,
        str_idx: torch.Tensor,       # [B]  STRING node indices, -1 = not in graph
        string_embs: torch.Tensor,   # [18870, 256] frozen buffer
    ) -> torch.Tensor:
        """Get hidden representation after residual blocks — used for Manifold Mixup."""
        valid_mask = str_idx >= 0
        safe_idx = str_idx.clamp(min=0)

        emb = string_embs[safe_idx].to(self.fallback_emb)
        if not valid_mask.all():
            fallback = self.fallback_emb.unsqueeze(0).expand(int((~valid_mask).sum()), -1)
            emb = emb.clone()
            emb[~valid_mask] = fallback

        x = self.input_proj(emb)
        for block in self.blocks:
            x = block(x)
        return x  # [B, hidden_dim]

    def head_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply output head to hidden representation."""
        logits = self.head(x) + self.gene_bias.to(x)  # [B, N_GENES * N_CLASSES]
        return logits.view(-1, N_CLASSES, N_GENES)     # [B, 3, 6640]

    def forward(
        self,
        str_idx: torch.Tensor,       # [B]  STRING node indices, -1 = not in graph
        string_embs: torch.Tensor,   # [18870, 256] frozen buffer
    ) -> torch.Tensor:
        x = self.get_embedding(str_idx, string_embs)
        return self.head_forward(x)


# ---------------------------------------------------------------------------
# Manifold Mixup utility
# ---------------------------------------------------------------------------
def manifold_mixup(
    x: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Manifold Mixup in the hidden representation space.

    Args:
        x: [B, hidden_dim] hidden representations
        labels: [B, 6640] integer class labels
        alpha: Beta distribution parameter

    Returns:
        x_mix: [B, hidden_dim] mixed representations
        labels_a: original labels [B, 6640]
        labels_b: shuffled labels [B, 6640]
        lam: mixing coefficient (scalar)
    """
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)

    x_mix = lam * x + (1.0 - lam) * x[idx]
    labels_a = labels
    labels_b = labels[idx]
    return x_mix, labels_a, labels_b, lam


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
        cosine_t0: int = 80,
        cosine_t_mult: int = 2,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.5,
        max_epochs: int = 500,
        grad_clip_norm: float = 1.0,
        use_muon: bool = True,
        save_top_k: int = 5,
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
        self.cosine_t0 = cosine_t0
        self.cosine_t_mult = cosine_t_mult
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.max_epochs = max_epochs
        self.grad_clip_norm = grad_clip_norm
        self.use_muon = use_muon
        self.save_top_k = save_top_k

        # Build model in __init__ so configure_optimizers sees all parameters
        # (required for DDP to properly detect all parameters before wrapping)
        self.model = PerturbMLP(
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout,
            head_dropout=head_dropout,
        )

        # Accumulation buffers for epoch-level metrics
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Class weights: inversely proportional to class frequencies (α=1.0 proven)
        # α=0.5 (softened) was shown to create spurious val/f1 signal in node1-3-2-2-1-1-1
        # class0=neutral(92.82%), class1=down(4.77%), class2=up(2.41%) after {-1,0,1}→{0,1,2}
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq)
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        self.register_buffer("class_weights", class_weights)

        if hasattr(self, "string_embs"):
            return  # already loaded (guard for re-entrant setup calls)

        # ---- Load STRING_GNN node embeddings (rank-0 downloads first) ----
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            # Ensure model is downloaded/cached by rank 0 first
            from transformers import AutoModel as _AM
            _AM.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

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

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        self.print(
            f"Node1-3-2-2-1-1-2 PerturbMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
            f"dropout={self.dropout} | head_dropout={self.head_dropout} | "
            f"muon_lr={self.muon_lr} | cosine_T0={self.cosine_t0}/Tmult={self.cosine_t_mult} | "
            f"mixup_alpha={self.mixup_alpha}/prob={self.mixup_prob} | "
            f"use_muon={self.use_muon} | max_epochs={self.max_epochs} | "
            f"trainable={n_trainable:,}/{n_total:,}"
        )

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted cross-entropy with optional label smoothing.

        logits: [B, 3, 6640]
        labels: [B, 6640]  — values in {0, 1, 2}
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                                # [B*6640]
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def _compute_mixup_loss(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Mixup loss: convex combination of two CE losses.

        logits:   [B, 3, 6640]
        labels_a: [B, 6640]  — values in {0, 1, 2}
        labels_b: [B, 6640]  — values in {0, 1, 2}
        lam: mixing coefficient
        """
        loss_a = self._compute_loss(logits, labels_a)
        loss_b = self._compute_loss(logits, labels_b)
        return lam * loss_a + (1.0 - lam) * loss_b

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        use_mixup = (
            self.training
            and self.mixup_prob > 0.0
            and random.random() < self.mixup_prob
            and batch["label"].size(0) > 1  # need at least 2 samples for mixup
        )

        if use_mixup:
            # Manifold Mixup: mix in the hidden representation space
            hidden = self.model.get_embedding(batch["str_idx"], self.string_embs)
            hidden_mix, labels_a, labels_b, lam = manifold_mixup(
                hidden, batch["label"], alpha=self.mixup_alpha
            )
            logits = self.model.head_forward(hidden_mix)
            loss = self._compute_mixup_loss(logits, labels_a, labels_b, lam)
        else:
            logits = self.model(batch["str_idx"], self.string_embs)
            loss = self._compute_loss(logits, batch["label"])

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["str_idx"], self.string_embs)
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
            if all_labels.dim() == 3:
                all_labels = all_labels.squeeze(0)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        f1 = _compute_per_gene_f1(preds_np, labels_np)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["str_idx"], self.string_embs)
        self._test_preds.append(logits.detach().cpu())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
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

        # Gather labels for metric computation (if available)
        test_f1 = None
        if self._test_labels:
            labels_local = torch.cat(self._test_labels, dim=0)  # [N_local, 6640]
            self._test_labels.clear()
            all_labels = self.all_gather(labels_local)  # [world_size, N_local, 6640]
            if ws > 1:
                all_labels = all_labels.view(-1, N_GENES)
            else:
                if all_labels.dim() == 3:
                    all_labels = all_labels.squeeze(0)
            preds_np = all_preds.float().cpu().numpy()
            labels_np = all_labels.cpu().numpy()
            test_f1 = _compute_per_gene_f1(preds_np, labels_np)
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=True)

        # Gather string metadata: encode as char-ordinal tensors for all_gather compatibility
        def _encode_strings(strings: List[str], max_len: int = 32) -> torch.Tensor:
            """Encode list of strings as int tensor [N, max_len]."""
            out = torch.zeros(len(strings), max_len, dtype=torch.long)
            for i, s in enumerate(strings):
                chars = [ord(c) for c in s[:max_len]]
                out[i, :len(chars)] = torch.tensor(chars, dtype=torch.long)
            return out

        def _decode_strings(tensor: torch.Tensor) -> List[str]:
            """Decode int tensor [N, max_len] back to list of strings."""
            result = []
            for row in tensor:
                chars = [chr(c) for c in row.tolist() if c > 0]
                result.append("".join(chars))
            return result

        pert_tensor = _encode_strings(self._test_pert_ids, max_len=32).to(self.device)
        sym_tensor = _encode_strings(self._test_symbols, max_len=32).to(self.device)

        all_pert_tensor = self.all_gather(pert_tensor)  # [ws, N_local, 32]
        all_sym_tensor = self.all_gather(sym_tensor)    # [ws, N_local, 32]

        if ws > 1:
            all_pert_tensor = all_pert_tensor.view(-1, 32)
            all_sym_tensor = all_sym_tensor.view(-1, 32)
        else:
            if all_pert_tensor.dim() == 3:
                all_pert_tensor = all_pert_tensor.squeeze(0)
                all_sym_tensor = all_sym_tensor.squeeze(0)

        self._test_pert_ids.clear()
        self._test_symbols.clear()

        if self.trainer.is_global_zero:
            all_pert_ids = _decode_strings(all_pert_tensor.cpu())
            all_symbols = _decode_strings(all_sym_tensor.cpu())
            preds_np = all_preds.float().cpu().numpy()  # [N_total, 3, 6640]

            # De-duplicate in case of DDP padding artifacts
            seen = set()
            dedup_pert = []
            dedup_sym = []
            dedup_idx = []
            for i, pid in enumerate(all_pert_ids):
                if pid and pid not in seen:
                    seen.add(pid)
                    dedup_pert.append(pid)
                    dedup_sym.append(all_symbols[i])
                    dedup_idx.append(i)
            preds_np = preds_np[dedup_idx]

            _save_test_predictions(
                pert_ids=dedup_pert,
                symbols=dedup_sym,
                preds=preds_np,
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    def configure_optimizers(self):
        """Configure optimizer: Muon for hidden MLP weight matrices, AdamW for everything else.

        LR schedule: CosineAnnealingWarmRestarts (T_0=80, T_mult=2).
        - The node1-3-2-2-1-1-1-1-1-1 lineage achieved F1=0.4968 using this schedule.
        - CosineWR escapes local optima through warm restarts at epochs 80 and 240,
          enabling the model to find better optima in subsequent restart cycles.
        - Muon LR=0.01 (not 0.02): node1-3-2-2-1-1-1-1 confirmed that LR=0.02 with
          RLROP kept LR constant throughout (val/f1 oscillations reset patience before
          reaching 8 consecutive non-improvements), while LR=0.01 allowed proper LR
          adaptation when the CosineWR provides structure for convergence.

        Muon is applied only to 2D weight matrices in hidden layers (not input/output layers,
        not biases/norms/embeddings). Per Muon skill documentation:
        - Muon: hidden Linear weight matrices in blocks (p.ndim >= 2, no norm)
        - AdamW: all other params (input_proj, head, gene_bias, fallback_emb, norms, biases)
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
                # Strip "model." prefix for model params
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
                # Muon for hidden weight matrices (residual block Linear weights)
                dict(
                    params=hidden_weights,
                    use_muon=True,
                    lr=self.muon_lr,
                    weight_decay=self.weight_decay,
                    momentum=0.95,
                ),
                # AdamW for all other params (input_proj, head, gene_bias, norms, etc.)
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
            # Fallback: pure AdamW for all parameters
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        # CosineAnnealingWarmRestarts: T_0=80, T_mult=2
        # - First restart at epoch 80, second at epoch 80+160=240
        # - Best epochs in ancestor lineage: 138 (cycle 1), 329 (cycle 2), 469 (cycle 3)
        # - This scheduler escapes local optima that RLROP gets trapped in
        # - min_lr=1e-6 to prevent training collapse at cosine minimum
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cosine_t0,
            T_mult=self.cosine_t_mult,
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
    # (string_embs are large frozen tensors recomputed in setup() —
    #  excluding them keeps checkpoint files small)
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # Guard for SWA internal copy scenario: self.model may be None in deepcopy
        if self.model is None:
            return super().state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )

        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        saved: Dict[str, Any] = {}
        # Trainable parameters from model
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                key = prefix + "model." + name
                if key in full_sd:
                    saved[key] = full_sd[key]
        # Also include non-model trainable params (if any)
        for name, param in self.named_parameters():
            if param.requires_grad and not name.startswith("model."):
                key = prefix + name
                if key in full_sd:
                    saved[key] = full_sd[key]
        # Essential small buffers (class_weights); exclude large frozen string_embs
        large_frozen = {"string_embs"}
        for name, buf in self.model.named_buffers():
            leaf = name.split(".")[-1]
            if leaf not in large_frozen:
                key = prefix + "model." + name
                if key in full_sd:
                    saved[key] = full_sd[key]
        # Module-level buffers (e.g., class_weights)
        for name, buf in self.named_buffers():
            if not name.startswith("model.") and name not in large_frozen:
                key = prefix + name
                if key in full_sd:
                    saved[key] = full_sd[key]

        n_total = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        pct = f"{100*n_trainable/n_total:.1f}%" if n_total > 0 else "N/A"
        self.print(
            f"Saving checkpoint: {n_trainable:,}/{n_total:,} trainable params ({pct})"
        )
        return saved

    def load_state_dict(self, state_dict, strict=True):
        # strict=False: string_embs is not in checkpoint but was populated by setup()
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
    for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
        rows.append({
            "idx": pid,
            "input": sym,
            "prediction": json.dumps(preds[i].tolist()),  # [3][6640] list
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


def _run_ensemble_test(
    model: PerturbModule,
    datamodule: PerturbDataModule,
    checkpoint_cb: ModelCheckpoint,
    trainer: pl.Trainer,
    output_dir: Path,
) -> None:
    """Run top-k checkpoint ensemble for test inference.

    Loads each of the top-k checkpoints saved during training, runs test inference
    on each, averages the raw logits, then saves the ensemble prediction.
    Only runs on rank 0 (DDP-safe since test_step aggregates all predictions).

    This function is a SELF-CONTAINED ensemble — it uses ONLY checkpoints from
    THIS node's own training run. No cross-node artifacts are used.
    """
    best_k = checkpoint_cb.best_k_models  # dict: path → val/f1
    if len(best_k) < 2:
        print("Warning: fewer than 2 checkpoints available — skipping ensemble")
        return

    # Sort by val/f1 descending
    sorted_ckpts = sorted(best_k.items(), key=lambda x: x[1], reverse=True)
    ckpt_paths = [str(p) for p, _ in sorted_ckpts]
    print(f"Ensemble: averaging {len(ckpt_paths)} checkpoints: {[Path(p).name for p in ckpt_paths]}")

    all_logits = []  # list of [N_test, 3, 6640] arrays
    for ckpt_path in ckpt_paths:
        # Use SingleDeviceStrategy to avoid NCCL conflicts with the parent DDP process group.
        # Test predictions are already gathered via all_gather in on_test_epoch_end.
        test_trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            strategy=SingleDeviceStrategy(device=torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")),
            precision="bf16-mixed",
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            deterministic=True,
            default_root_dir=str(output_dir),
        )
        # Load checkpoint and run test
        ckpt_model = PerturbModule.load_from_checkpoint(
            ckpt_path,
            strict=False,
        )
        # CRITICAL: setup("test") must be called to initialize string_embs buffer
        # since load_from_checkpoint does not trigger Lightning's normal setup flow
        ckpt_model.setup("test")
        test_trainer.test(ckpt_model, datamodule=datamodule)

        # Collect the predictions just saved
        pred_path = output_dir / "test_predictions.tsv"
        if pred_path.exists() and test_trainer.is_global_zero:
            df = pd.read_csv(pred_path, sep="\t")
            logits_list = [json.loads(x) for x in df["prediction"]]
            logits_arr = np.array(logits_list)  # [N, 3, 6640]
            all_logits.append((df[["idx", "input"]], logits_arr))

    if len(all_logits) < 2:
        print("Warning: could not load predictions from multiple checkpoints")
        return

    # Average logits across all checkpoints
    meta_df = all_logits[0][0].reset_index(drop=True)
    avg_logits = np.mean([arr for _, arr in all_logits], axis=0)  # [N, 3, 6640]

    # Save ensemble predictions (overwrites single-checkpoint result)
    _save_test_predictions(
        pert_ids=meta_df["idx"].tolist(),
        symbols=meta_df["input"].tolist(),
        preds=avg_logits,
        out_path=output_dir / "test_predictions.tsv",
    )
    print(f"Ensemble test predictions saved (averaged {len(all_logits)} checkpoints)")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-3-2-2-1-1-2: STRING-only + Flat Head + hidden=384 + Muon(LR=0.01) + CosineWR + Manifold Mixup"
    )
    p.add_argument("--micro-batch-size",    type=int,   default=32)
    p.add_argument("--global-batch-size",   type=int,   default=256)
    p.add_argument("--max-epochs",          type=int,   default=500)
    p.add_argument("--lr",                  type=float, default=3e-4)
    p.add_argument("--muon-lr",             type=float, default=0.01,
                   help="Muon LR=0.01 (not 0.02): confirmed correct setting for LR adaptation in warm restart schedule")
    p.add_argument("--weight-decay",        type=float, default=8e-4,
                   help="weight_decay=8e-4 matching node1-3-2-2-1-1-1-1-1-1 (F1=0.4968)")
    p.add_argument("--hidden-dim",          type=int,   default=384)
    p.add_argument("--n-blocks",            type=int,   default=3)
    p.add_argument("--dropout",             type=float, default=0.30)
    p.add_argument("--head-dropout",        type=float, default=0.15,
                   help="head_dropout=0.15 reverted from parent's 0.20 (over-regularized)")
    p.add_argument("--label-smoothing",     type=float, default=0.0,
                   help="No label smoothing: node1-3-2-2-1-1-1-1 showed removing it allows lower train loss")
    p.add_argument("--cosine-t0",           type=int,   default=80,
                   help="CosineAnnealingWarmRestarts T_0: first restart at epoch 80")
    p.add_argument("--cosine-t-mult",       type=int,   default=2,
                   help="CosineAnnealingWarmRestarts T_mult: doubles period each restart")
    p.add_argument("--mixup-alpha",         type=float, default=0.2,
                   help="Manifold Mixup alpha (Beta distribution parameter)")
    p.add_argument("--mixup-prob",          type=float, default=0.5,
                   help="Probability of applying Manifold Mixup to a given batch")
    p.add_argument("--save-top-k",          type=int,   default=5,
                   help="Save top-k checkpoints by val/f1 for ensemble")
    p.add_argument("--early-stop-patience", type=int,   default=120,
                   help="EarlyStopping patience: 120 allows ≥3 full warm restart cycles")
    p.add_argument("--grad-clip-norm",      type=float, default=1.0)
    p.add_argument("--no-muon",             action="store_true",
                   help="Disable Muon optimizer, fall back to AdamW for all params")
    p.add_argument("--no-ensemble",         action="store_true",
                   help="Disable top-k checkpoint ensemble at test time")
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

    # --- DataModule ---
    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup("fit")

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
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_t_mult,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        max_epochs=args.max_epochs,
        grad_clip_norm=args.grad_clip_norm,
        use_muon=not args.no_muon,
        save_top_k=args.save_top_k,
    )

    # --- Trainer configuration ---
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        # Limit train/val to debug_max_step GLOBAL steps (after grad accumulation).
        # Test MUST always use full test set (1.0) — never limited by debug_max_step.
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = 1.0
        max_steps = args.debug_max_step
    else:
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        max_steps = -1

    # save_top_k=5: keep top-5 checkpoints for ensemble
    # Use filename without "/" to avoid directory creation issues
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-val_f1={val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=args.save_top_k if not (args.fast_dev_run or args.debug_max_step is not None) else 1,
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
            find_unused_parameters=True,  # Required: self.model forward may not register all params as used in DDP
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

    # --- Test (single best checkpoint first) ---
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # --- Save test score (rank 0 only) BEFORE ensemble to ensure it persists ---
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")

    # --- Top-k Checkpoint Ensemble (production run only, single-GPU only) ---
    # After single-checkpoint test, run ensemble to average top-k logits.
    # This is a within-node ensemble: ALL checkpoints from THIS node's training only.
    # No cross-node artifacts used.
    # NOTE: _run_ensemble_test creates a new pl.Trainer with SingleDeviceStrategy,
    # which causes NCCL conflicts when called inside a DDP process. We guard with
    # n_gpus == 1 to ensure ensemble only runs in single-GPU mode.
    if (
        not args.fast_dev_run
        and args.debug_max_step is None
        and not args.no_ensemble
        and trainer.is_global_zero
        and n_gpus == 1
    ):
        try:
            _run_ensemble_test(
                model=model,
                datamodule=datamodule,
                checkpoint_cb=checkpoint_cb,
                trainer=trainer,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"Warning: ensemble test failed ({e}). Single-checkpoint result retained.")


if __name__ == "__main__":
    main()
