"""Node 1-3-2-2-1-1-1: Revert to proven-best config (head_dropout=0.15)
                       + Softened class weights (α=0.5 temperature)
                       + Top-3 checkpoint ensemble at test time

Key changes from parent (node1-3-2-2-1-1, F1=0.4746 — REGRESSION):
  1. Revert head_dropout: 0.20 → 0.15
     — Parent's head_dropout=0.20 over-regularized the 7.6M-param flat head,
       causing premature convergence to a suboptimal local minimum at epoch 60
       (vs grandparent's peak at epoch 91). The train-val loss gap compressed
       from 0.064 to 0.052 yet test F1 *dropped* by 0.0031 — confirming harmful
       over-regularization. Reverting to 0.15 (the tree-best configuration from
       node1-3-2-2-1, F1=0.4777).

  2. Revert trunk dropout: 0.28 → 0.30
     — Companion restoration consistent with the proven-best config.

  3. Revert RLROP patience: 10 → 8
     — RLROP patience=10 was designed for the noisier head_dropout=0.20 training
       dynamics. With head_dropout=0.15 restored, patience=8 is optimal.

  4. Revert max_epochs: 300 → 200 (early_stop_patience 35 → 25)
     — Grandparent's peak was at epoch 91/200; no need for 300 epochs.

  5. NEW: Softened class weights (α=0.5 temperature)
     — Replace extreme inverse-frequency weights w_i = 1/freq_i with
       w_i = (1/freq_i)^0.5 (square root / α=0.5 temperature).
     — Current weights [1.08, 20.96, 41.49] → ratio 38.5x (class0:class2)
     — Softened weights [1.04, 4.58, 6.44] → ratio 6.2x
     — Motivation: the extreme 38.5x weight for class 2 (up-regulated, 2.41%)
       may produce large, noisy gradient spikes for individual up-regulated
       predictions. Softening reduces the gradient variance while still
       strongly prioritizing minority classes over the dominant neutral class.
       This targets more stable and efficient learning of all three classes.

  6. NEW: Top-3 checkpoint ensemble at test time
     — save_top_k=3 (was 1) to preserve the 3 best validation checkpoints.
     — At test time, load all 3 checkpoints, run inference for each, and
       average the logits before saving test_predictions.tsv.
     — This within-node ensemble provides a free generalization boost by
       averaging across different weight configurations explored during
       training, similar to stochastic weight averaging but leveraging
       natural checkpoint diversity from the RLROP schedule.
     — Ensemble is self-contained: all checkpoints from the same run.

Tree context:
  node1-3-2 (great-grandparent) | F1=0.4756 | STRING + hidden=384 + Muon + RLROP
  node1-3-2-2 (grandparent)     | F1=0.4750 | STRING + hidden=416 + Muon + Cosine LR
  node1-3-2-2-1 (grandparent)   | F1=0.4777 | STRING + hidden=384 + head_dropout=0.15 [TREE BEST]
  node1-3-2-2-1-1 (parent)      | F1=0.4746 | STRING + hidden=384 + head_dropout=0.20 [REGRESSION]
  This node targets F1 > 0.4777 via proven-best core + softened weights + ensemble
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
import re
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
    """STRING-only MLP for gene perturbation response prediction.

    Architecture (per sample):
      ① STRING_GNN embedding lookup [256-dim, frozen buffer]
         (fallback learnable 256-dim for ~6% genes not in STRING)
      ② Input projection: Linear(256→hidden_dim) + LN + GELU
      ③ n_blocks × ResidualBlock(hidden_dim)
      ④ LN(hidden_dim) → Dropout(head_dropout) → Linear(hidden_dim → 6640*3) + per-gene-bias
      ⑤ reshape → [B, 3, 6640]

    This is the proven-best architecture from node1-3-2-2-1 (F1=0.4777),
    with head_dropout=0.15 restored from parent's over-regularizing 0.20.
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
        # Flat output head + per-gene additive bias.
        # head_dropout=0.15 — the proven-best value from node1-3-2-2-1 (F1=0.4777).
        # Flat head confirmed superior over factorized heads across 4+ nodes.
        # Per-gene bias confirmed in node1-1-1.
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(head_dropout),           # Output head regularization
            nn.Linear(hidden_dim, N_GENES * N_CLASSES),
        )
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES * N_CLASSES))

    def forward(
        self,
        str_idx: torch.Tensor,       # [B]  STRING node indices, -1 = not in graph
        string_embs: torch.Tensor,   # [18870, 256] frozen buffer
    ) -> torch.Tensor:
        valid_mask = str_idx >= 0                    # [B] bool
        safe_idx = str_idx.clamp(min=0)              # replace -1 with 0 (overwritten below)

        # ①  Look up frozen STRING embeddings [B, 256]
        emb = string_embs[safe_idx].to(self.fallback_emb)  # cast to compute dtype

        # Overwrite samples whose gene is absent from the STRING graph
        if not valid_mask.all():
            fallback = self.fallback_emb.unsqueeze(0).expand(
                int((~valid_mask).sum()), -1
            )
            emb = emb.clone()
            emb[~valid_mask] = fallback

        # ②–④  Projection → residual MLP → output
        x = self.input_proj(emb)            # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)
        logits = self.head(x) + self.gene_bias.to(x)  # [B, N_GENES * N_CLASSES]
        return logits.view(-1, N_CLASSES, N_GENES)      # [B, 3, 6640]


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
        muon_lr: float = 0.02,
        weight_decay: float = 5e-4,
        label_smoothing: float = 0.05,
        class_weight_alpha: float = 0.5,
        rlrop_patience: int = 8,
        rlrop_factor: float = 0.5,
        max_epochs: int = 200,
        grad_clip_norm: float = 1.0,
        use_muon: bool = True,
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
        self.class_weight_alpha = class_weight_alpha
        self.rlrop_patience = rlrop_patience
        self.rlrop_factor = rlrop_factor
        self.max_epochs = max_epochs
        self.grad_clip_norm = grad_clip_norm
        self.use_muon = use_muon

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
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def _load_string_embeddings(self, skip_barrier: bool = False) -> None:
        """Load STRING_GNN embeddings and register as frozen buffer.

        This is a standalone helper that does NOT require a Trainer attachment.
        Use this in ensemble-test contexts where the module is not part of a Trainer.

        Args:
            skip_barrier: If True, skips the distributed barrier call. This is
                required when called outside the DDP collective context (e.g., in
                ensemble test where only rank 0 runs the code). The barrier is
                only needed for multi-rank DDP training synchronization.
        """
        if hasattr(self, "string_embs") and self.string_embs is not None:
            return  # already loaded

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            from transformers import AutoModel as _AM
            _AM.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        # Only barrier in DDP training context; skip when outside the collective
        # (e.g., ensemble test runs only on rank 0, rank 1 is not in that code path)
        if not skip_barrier and torch.distributed.is_available() and torch.distributed.is_initialized():
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
        self.register_buffer("string_embs", string_embs)

    def setup(self, stage: str = "fit") -> None:
        # Softened class weights: w_i = (1/freq_i)^alpha  [alpha=0.5 → sqrt]
        # Current alpha=1 gives: [1.08, 20.96, 41.49] normalized → 38.5x ratio (class0:class2)
        # Alpha=0.5 gives:        [1.04, 4.58, 6.44]  normalized → 6.2x ratio  (much gentler)
        #
        # Motivation: the extreme 38.5x weight for class2 (up-regulated, 2.41%) may
        # produce large gradient spikes and training instability. The softer 6.2x
        # ratio still strongly prioritizes minority classes while maintaining
        # more stable gradient magnitudes. This is the primary NEW change from
        # the proven-best config (node1-3-2-2-1), targeting improved gradient
        # quality for all three class boundaries.
        alpha = self.class_weight_alpha
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq).pow(alpha)   # temperature-scaled
        class_weights = class_weights / class_weights.sum() * N_CLASSES
        self.register_buffer("class_weights", class_weights)

        if hasattr(self, "string_embs") and self.string_embs is not None:
            return  # already loaded (guard for re-entrant setup calls)

        # ---- Load STRING_GNN node embeddings (rank-0 downloads first) ----
        self._load_string_embeddings()

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        # Guard self.print() — it accesses self.trainer which is None in ensemble-test context
        if hasattr(self, "trainer") and self.trainer is not None:
            self.print(
                f"Node1-3-2-2-1-1-1 PerturbMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
                f"dropout={self.dropout} | head_dropout={self.head_dropout} | "
                f"class_weight_alpha={self.class_weight_alpha} | "
                f"use_muon={self.use_muon} | max_epochs={self.max_epochs} | "
                f"trainable={n_trainable:,}/{n_total:,}"
            )

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted cross-entropy with label smoothing.

        logits: [B, 3, 6640]
        labels: [B, 6640]  — values in {0, 1, 2}

        Uses temperature-scaled class weights (alpha=0.5) to reduce gradient
        instability from extreme inverse-frequency weighting.
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                                # [B*6640]
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
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

        LR schedule: ReduceLROnPlateau (patience=8, factor=0.5).
        - Restored to patience=8 (proven optimal from tree-best node1-3-2-2-1).
        - Parent's patience=10 was designed for the noisier head_dropout=0.20;
          with head_dropout=0.15 restored, patience=8 is the right choice.

        Muon is applied only to 2D weight matrices in hidden layers (not input/output layers,
        not biases/norms/embeddings). Per Muon skill documentation.
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

        # ReduceLROnPlateau: patience=8 (restored to proven-best value)
        # - Monitors val/f1 (maximize)
        # - min_lr=1e-6 prevents training collapse
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",           # monitor val/f1 (higher = better)
            factor=self.rlrop_factor,
            patience=self.rlrop_patience,
            min_lr=1e-6,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/f1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint: save only trainable params + small essential buffers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if self.model is None:
            return super().state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )

        # Get the UNWRAPPED model's state dict to avoid DDP's "module." prefix.
        # DDP wraps the module as "module.model.*" but self.model is the unwrapped
        # PerturbMLP. Calling self.model.state_dict() gives us clean "model.*" keys
        # that are compatible with loading into a fresh unwrapped module.
        unwrapped_sd = self.model.state_dict()
        saved: Dict[str, Any] = {}

        # Trainable parameters from the model (all PerturbMLP params require grad)
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in unwrapped_sd:
                saved[f"{prefix}model.{name}"] = unwrapped_sd[name]

        # Module-level trainable params (if any, beyond self.model)
        for name, param in self.named_parameters():
            if param.requires_grad and not name.startswith("model."):
                if name in unwrapped_sd:
                    saved[f"{prefix}{name}"] = unwrapped_sd[name]

        # Essential small buffers (e.g., class_weights); exclude large frozen string_embs
        large_frozen = {"string_embs"}
        for name, buf in self.model.named_buffers():
            leaf = name.split(".")[-1]
            if leaf not in large_frozen and name in unwrapped_sd:
                saved[f"{prefix}model.{name}"] = unwrapped_sd[name]

        # Module-level buffers (e.g., class_weights)
        for name, buf in self.named_buffers():
            if not name.startswith("model.") and name not in large_frozen:
                if name in unwrapped_sd:
                    saved[f"{prefix}{name}"] = unwrapped_sd[name]

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


def _discover_local_checkpoints(
    ckpt_dir: Path, top_k: int
) -> List[Path]:
    """Discover the top-k best checkpoints from the local checkpoint directory.

    The checkpoint callback saves files as subdirectories:
      best-epoch=XXX-val/f1=0.XXXX.ckpt

    We scan the directory, extract val/f1 from filenames, and return the top-k.
    This avoids relying on best_k_models (which may contain stale absolute paths
    from a previous training run location).
    """
    import re
    all_ckpts: List[tuple[float, Path]] = []
    for subdir in ckpt_dir.iterdir():
        if not subdir.is_dir():
            continue
        # Subdir name like "best-epoch=068-val"
        epoch_match = re.match(r"best-epoch=(\d+)-val", subdir.name)
        if not epoch_match:
            continue
        epoch = int(epoch_match.group(1))
        # Find the .ckpt file inside (e.g. f1=0.4873.ckpt)
        ckpt_files = list(subdir.glob("f1=*.ckpt"))
        if not ckpt_files:
            continue
        ckpt_file = ckpt_files[0]
        # Extract f1 score: "f1=0.4873.ckpt"
        f1_match = re.match(r"f1=([\d.]+)\.ckpt$", ckpt_file.name)
        if not f1_match:
            continue
        f1 = float(f1_match.group(1))
        all_ckpts.append((f1, ckpt_file))

    if not all_ckpts:
        return []

    # Sort by f1 descending (best first), take top_k
    all_ckpts.sort(key=lambda x: x[0], reverse=True)
    return [ckpt for _, ckpt in all_ckpts[:top_k]]


def _run_ensemble_test(
    model_kwargs: Dict[str, Any],
    ckpt_paths: List[Path],
    test_dataloader: DataLoader,
    out_path: Path,
    device: torch.device,
) -> None:
    """Ensemble test by averaging logits from multiple checkpoints.

    This is a within-node ensemble: all checkpoints are from the same training run.
    No cross-node artifacts are used. All predictions are self-contained.

    Args:
        model_kwargs: Keyword arguments to instantiate PerturbModule
        ckpt_paths:   Paths to checkpoints to ensemble (sorted by val/f1 ascending)
        test_dataloader: DataLoader for the test set
        out_path:     Path to save final ensemble predictions TSV
        device:       Device for inference (typically cuda:0)
    """
    all_preds_list: List[np.ndarray] = []
    pert_ids_final: Optional[List[str]] = None
    symbols_final: Optional[List[str]] = None

    print(f"\n[Ensemble] Averaging logits from {len(ckpt_paths)} checkpoints:")
    for ckpt_path in ckpt_paths:
        print(f"  Loading: {ckpt_path.name}")

        # Create a fresh model instance (not wrapped in DDP)
        m = PerturbModule(**model_kwargs)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        # The checkpoint may have 'state_dict' key (from Lightning) or be raw
        sd = state.get("state_dict", state)
        m.load_state_dict(sd, strict=False)

        # Set up string embeddings (without DDP) — use _load_string_embeddings to avoid
        # trainer dependency (setup() calls self.print which requires self.trainer).
        # skip_barrier=True because ensemble test runs only on rank 0 (no DDP collective).
        m._load_string_embeddings(skip_barrier=True)
        m.eval()
        m = m.to(device)
        if hasattr(m, "string_embs"):
            m.string_embs = m.string_embs.to(device)

        # Run inference on the test set
        preds_list: List[torch.Tensor] = []
        pert_ids_batch: List[str] = []
        symbols_batch: List[str] = []

        with torch.no_grad():
            for batch in test_dataloader:
                str_idx = batch["str_idx"].to(device)
                logits = m.model(str_idx, m.string_embs)  # [B, 3, 6640]
                preds_list.append(logits.float().cpu())
                pert_ids_batch.extend(batch["pert_id"])
                symbols_batch.extend(batch["symbol"])

        batch_preds = torch.cat(preds_list, dim=0).numpy()  # [N, 3, 6640]
        all_preds_list.append(batch_preds)

        if pert_ids_final is None:
            pert_ids_final = pert_ids_batch
            symbols_final = symbols_batch

        # Free memory before loading next checkpoint
        del m
        torch.cuda.empty_cache()

    # Average logits across all checkpoints
    avg_preds = np.mean(all_preds_list, axis=0)  # [N, 3, 6640]

    # Deduplicate (consistent with the Lightning test pipeline)
    seen: set = set()
    dedup_pert: List[str] = []
    dedup_sym: List[str] = []
    dedup_idx: List[int] = []
    for i, pid in enumerate(pert_ids_final or []):
        if pid and pid not in seen:
            seen.add(pid)
            dedup_pert.append(pid)
            dedup_sym.append((symbols_final or [])[i])
            dedup_idx.append(i)
    avg_preds = avg_preds[dedup_idx]

    _save_test_predictions(
        pert_ids=dedup_pert,
        symbols=dedup_sym,
        preds=avg_preds,
        out_path=out_path,
    )
    print(f"[Ensemble] Saved {len(dedup_pert)} predictions using {len(ckpt_paths)}-checkpoint ensemble → {out_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-3-2-2-1-1-1: STRING-only + Flat Head + hidden=384 + Muon + RLROP + "
                    "head_dropout=0.15 + softened class weights (alpha=0.5) + top-3 ensemble"
    )
    p.add_argument("--micro-batch-size",      type=int,   default=32)
    p.add_argument("--global-batch-size",     type=int,   default=256)
    p.add_argument("--max-epochs",            type=int,   default=200)
    p.add_argument("--lr",                    type=float, default=3e-4)
    p.add_argument("--muon-lr",               type=float, default=0.02)
    p.add_argument("--weight-decay",          type=float, default=5e-4)
    p.add_argument("--hidden-dim",            type=int,   default=384)
    p.add_argument("--n-blocks",              type=int,   default=3)
    p.add_argument("--dropout",               type=float, default=0.30)
    p.add_argument("--head-dropout",          type=float, default=0.15,
                   help="Dropout before flat output head (restored to proven-best 0.15 from node1-3-2-2-1)")
    p.add_argument("--label-smoothing",       type=float, default=0.05)
    p.add_argument("--class-weight-alpha",    type=float, default=0.5,
                   help="Temperature exponent for class weights: w_i=(1/freq_i)^alpha. "
                        "0.5=sqrt (softened, NEW), 1.0=inverse freq (original)")
    p.add_argument("--rlrop-patience",        type=int,   default=8,
                   help="ReduceLROnPlateau patience (restored to proven-best 8)")
    p.add_argument("--rlrop-factor",          type=float, default=0.5,
                   help="ReduceLROnPlateau LR reduction factor")
    p.add_argument("--early-stop-patience",   type=int,   default=25,
                   help="EarlyStopping patience (restored to proven-best 25)")
    p.add_argument("--grad-clip-norm",        type=float, default=1.0)
    p.add_argument("--no-muon",               action="store_true",
                   help="Disable Muon optimizer, fall back to AdamW for all params")
    p.add_argument("--ensemble-top-k",        type=int,   default=3,
                   help="Number of best checkpoints to ensemble at test time (0=disable)")
    p.add_argument("--num-workers",           type=int,   default=4)
    p.add_argument("--val-check-interval",    type=float, default=1.0)
    p.add_argument("--debug_max_step",        type=int,   default=None)
    p.add_argument("--fast_dev_run",          action="store_true")
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
        class_weight_alpha=args.class_weight_alpha,
        rlrop_patience=args.rlrop_patience,
        rlrop_factor=args.rlrop_factor,
        max_epochs=args.max_epochs,
        grad_clip_norm=args.grad_clip_norm,
        use_muon=not args.no_muon,
    )

    # --- Trainer configuration ---
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = args.debug_max_step
        limit_val = args.debug_max_step
        limit_test = 1.0
        max_steps = args.debug_max_step
    else:
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        max_steps = -1

    # save_top_k=3: keep the 3 best checkpoints by val/f1 for ensemble
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=3,     # NEW: save top-3 for ensemble (was 1)
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
            find_unused_parameters=True,
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
    test_results = None
    if args.fast_dev_run or args.debug_max_step is not None:
        # Debug mode: standard Lightning test (all ranks, no ensemble)
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        # Production mode:
        # Step 1: Run standard test with single best checkpoint (all ranks participate)
        # This writes the initial test_predictions.tsv via on_test_epoch_end
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

        # Step 2: Ensemble test on rank 0 (overwrite test_predictions.tsv if >1 checkpoints)
        # This is a within-node ensemble; no cross-node artifacts are used.
        if trainer.is_global_zero and args.ensemble_top_k > 1:
            ckpt_dir = output_dir / "checkpoints"
            # Discover checkpoints by scanning the local checkpoint directory.
            # We cannot use checkpoint_cb.best_k_models here because it may contain
            # stale absolute paths from a previous training run location.
            top_ckpts = _discover_local_checkpoints(ckpt_dir, top_k=args.ensemble_top_k)
            if len(top_ckpts) >= 2:
                print(f"\n[Ensemble] Running {len(top_ckpts)}-checkpoint ensemble test...")
                _run_ensemble_test(
                    model_kwargs=dict(
                        hidden_dim=args.hidden_dim,
                        n_blocks=args.n_blocks,
                        dropout=args.dropout,
                        head_dropout=args.head_dropout,
                        lr=args.lr,
                        muon_lr=args.muon_lr,
                        weight_decay=args.weight_decay,
                        label_smoothing=args.label_smoothing,
                        class_weight_alpha=args.class_weight_alpha,
                        rlrop_patience=args.rlrop_patience,
                        rlrop_factor=args.rlrop_factor,
                        max_epochs=args.max_epochs,
                        grad_clip_norm=args.grad_clip_norm,
                        use_muon=not args.no_muon,
                    ),
                    ckpt_paths=top_ckpts,
                    test_dataloader=datamodule.test_dataloader(),
                    out_path=output_dir / "test_predictions.tsv",
                    device=torch.device("cuda", 0),
                )
            else:
                print(f"\n[Ensemble] Only {len(top_ckpts)} checkpoint(s) available; "
                      "skipping ensemble (need >= 2)")

        # Synchronize all ranks after rank-0 ensemble
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    # --- Save test score (rank 0 only) ---
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")


if __name__ == "__main__":
    main()
