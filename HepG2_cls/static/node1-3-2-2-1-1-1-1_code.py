"""Node 1-3-2-2-1-1-1-1: Lower Muon LR (0.02→0.01) + No Label Smoothing (0.05→0.0)
                        + Extended RLROP patience (8→15) + Top-5 checkpoint ensemble

Key changes from parent (node1-3-2-2-1-1-1, test F1=0.4879 — TREE BEST):
  1. Lower Muon LR: 0.02 → 0.01
     — The parent's RLROP never fired across 93 epochs because val/f1 was too
       noisy at Muon LR=0.02 for patience=8 to trigger. node3-3-1-1 (F1=0.4793)
       confirmed that Muon LR=0.01 provides stable, monotonic convergence with 5
       successful RLROP halvings. Smaller gradient steps reduce oscillation amplitude,
       enabling patience=15 to distinguish genuine plateaus from noise.

  2. Remove label smoothing: 0.05 → 0.0
     — label_smoothing=0.05 raises the minimum achievable cross-entropy by ~0.055
       nats, creating a training loss floor that limits precise fitting. node3-3-1-1
       removed label smoothing and saw train/loss drop from 0.94 → 0.213 (+0.057 F1).
       The α=0.5 softened class weights already provide smooth class guidance;
       removing label smoothing allows proper cross-entropy minimization.

  3. Increase RLROP patience: 8 → 15
     — With 141 validation samples and α=0.5 class weights, val/f1 oscillates
       frequently (43% epoch-to-epoch decrease rate in parent). Patience=8 is
       insufficient to distinguish noise from true plateaus. Patience=15 accommodates
       natural oscillations while still allowing RLROP to fire 3-5 times in 300 epochs.

  4. Extend max_epochs: 200 → 300 (early_stop_patience: 25 → 35)
     — With RLROP patience=15 enabling ~3-5 halvings across 300 epochs, more training
       budget is needed to fully converge across multiple LR phases. Early stopping
       patience=35 prevents premature termination after the final RLROP halving.

  5. Save top-5 checkpoints (was top-3) + richer ensemble
     — Parent's top-3 checkpoints were from epochs 66-79 (narrow 13-epoch window),
       providing minimal diversity. With RLROP halvings across ~250-300 epochs, the
       top-5 checkpoints should span different LR phases, capturing diverse model
       states for a more effective ensemble.

What is preserved from parent:
  — head_dropout=0.15 (proven optimal)
  — trunk dropout=0.30 (proven optimal)
  — class_weight_alpha=0.5 (softened, contributed to parent's 0.4879)
  — AdamW lr=3e-4 (proven optimal for non-Muon params)
  — weight_decay=5e-4 (proven optimal)
  — grad_clip_norm=1.0 (proven stable)

Tree context:
  node1-3-2-2-1 (great-grandparent) | F1=0.4777 | STRING + hidden=384 + head_dropout=0.15 [prev best]
  node1-3-2-2-1-1 (grandparent)     | F1=0.4746 | head_dropout=0.20 (regression)
  node1-3-2-2-1-1-1 (parent)        | F1=0.4879 | head_dropout=0.15 + alpha=0.5 + top-3 ensemble [TREE BEST]
  node3-3-1-1 (cousin)              | F1=0.4793 | Muon LR=0.01 + no label smooth + head_dropout=0.05
  This node targets F1 > 0.4879 via proven core + lower Muon LR + no label smooth + extended RLROP + top-5 ensemble
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
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
        # Map ENSEMBL pert_id → STRING-node-index; -1 = not in STRING graph
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

    Key changes from parent:
      - Identical architecture (head_dropout=0.15 retained as proven optimal)
      - Training improvements: lower Muon LR + no label smoothing + extended RLROP
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
        # head_dropout=0.15 — the proven-best value (confirmed across multiple nodes).
        # Flat head confirmed superior over factorized heads across 4+ nodes.
        # Per-gene bias confirmed in node1-1-1.
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(head_dropout),
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
        muon_lr: float = 0.01,        # KEY CHANGE: 0.02 → 0.01 (enables RLROP to fire)
        weight_decay: float = 5e-4,
        label_smoothing: float = 0.0, # KEY CHANGE: 0.05 → 0.0 (removes training loss floor)
        class_weight_alpha: float = 0.5,  # Softened weights (contributed to parent's 0.4879)
        rlrop_patience: int = 15,     # KEY CHANGE: 8 → 15 (accommodates noisy val/f1)
        rlrop_factor: float = 0.5,
        max_epochs: int = 300,
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
            skip_barrier: If True, skips the distributed barrier call. Required
                when called outside the DDP collective context (e.g., ensemble test
                where only rank 0 runs the code).
        """
        if hasattr(self, "string_embs") and self.string_embs is not None:
            return  # already loaded

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            from transformers import AutoModel as _AM
            _AM.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        # Only barrier in DDP training context; skip when outside the collective
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
        # alpha=0.5: weights [1.04, 4.58, 6.44] → ratio class0:class2 = 6.2× (moderate)
        # This prevents extreme gradient spikes from the class2 (up-regulated, 2.41%) weight.
        # The parent (node1-3-2-2-1-1-1) achieved F1=0.4879 with this exact alpha=0.5 configuration.
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
        if hasattr(self, "trainer") and self.trainer is not None:
            self.print(
                f"Node1-3-2-2-1-1-1-1 PerturbMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
                f"dropout={self.dropout} | head_dropout={self.head_dropout} | "
                f"class_weight_alpha={self.class_weight_alpha} | label_smoothing={self.label_smoothing} | "
                f"muon_lr={self.muon_lr} | rlrop_patience={self.rlrop_patience} | "
                f"use_muon={self.use_muon} | max_epochs={self.max_epochs} | "
                f"trainable={n_trainable:,}/{n_total:,}"
            )

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted cross-entropy without label smoothing.

        logits: [B, 3, 6640]
        labels: [B, 6640]  — values in {0, 1, 2}

        Uses temperature-scaled class weights (alpha=0.5) to moderate gradient
        instability from extreme inverse-frequency weighting.
        label_smoothing=0.0 (no floor) — removed to allow proper cross-entropy minimization.
        This combination allows the model to fit training data precisely while still
        prioritizing minority classes through the softened class weights.
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                                # [B*6640]
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,  # 0.0: no artificial loss floor
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

        LR schedule: ReduceLROnPlateau (patience=15, factor=0.5).
        - patience=15: KEY CHANGE from parent's 8 — accommodates noisy val/f1 on 141 samples.
          With 43% epoch-to-epoch val/f1 decrease rate, patience=8 never triggered in parent.
          patience=15 distinguishes genuine plateaus from oscillations.
        - Muon LR=0.01: KEY CHANGE from parent's 0.02 — proven stable in node3-3-1-1 (F1=0.4793),
          produces stable monotonic convergence and enables RLROP halvings.

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
                    lr=self.muon_lr,       # KEY: 0.01 (was 0.02) — stable convergence
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

        # ReduceLROnPlateau: patience=15 (KEY CHANGE from parent's 8)
        # - patience=15 accommodates val/f1 oscillations on the 141-sample validation set
        # - monitors val/f1 (maximize); min_lr=1e-7 prevents training collapse
        # - With Muon LR=0.01 reducing oscillation amplitude, patience=15 should trigger
        #   3-5 halvings across 300 epochs, enabling progressive LR refinement
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",           # monitor val/f1 (higher = better)
            factor=self.rlrop_factor,
            patience=self.rlrop_patience,
            min_lr=1e-7,
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
        """Save trainable parameters and essential buffers, excluding large frozen buffers.

        Uses Lightning's base state_dict() which correctly handles DDP wrapping
        (adds/strips 'module.' prefix as needed). We then filter to keep only:
        - Trainable parameters (all model weights that require gradients)
        - Small persistent buffers (e.g., class_weights)
        - Excludes large frozen buffers (string_embs: [18870, 256])
        """
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        saved: Dict[str, Any] = {}

        # Keys to always exclude (large frozen buffers)
        exclude_keys = {"string_embs"}
        # Keys that are trainable params or essential small buffers
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        essential_buffers = {
            name for name, buf in self.named_buffers()
            if name not in exclude_keys
        }
        keep_keys = trainable_keys | essential_buffers

        for key, val in full_sd.items():
            # Strip top-level prefix for matching
            rel_key = key[len(prefix):] if key.startswith(prefix) else key
            if rel_key in keep_keys or any(k in rel_key for k in keep_keys):
                saved[key] = val

        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_saved = sum(v.numel() for v in saved.values())
        pct = f"{100*n_trainable/n_total:.1f}%" if n_total > 0 else "N/A"
        self.print(
            f"Saving checkpoint: {n_trainable:,}/{n_total:,} trainable params ({pct}), "
            f"{n_saved:,} total saved"
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

    The checkpoint callback saves files with pattern:
      best-epoch=XXX-val/f1=0.XXXX.ckpt  (flat file) or
      best-epoch=XXX-val/  (subdirectory containing f1=0.XXXX.ckpt)

    We scan the directory for both flat .ckpt files and subdirectories,
    extract val/f1 from filenames, and return the top-k by val/f1 score.
    """
    all_ckpts: List[tuple] = []  # (f1_score, path)

    if not ckpt_dir.exists():
        return []

    for entry in ckpt_dir.iterdir():
        if entry.is_file() and entry.suffix == ".ckpt":
            # Try flat file: "best-epoch=068-val_f1=0.4873.ckpt" or similar
            f1_match = re.search(r"val_f1[=_]([\d.]+)\.ckpt$", entry.name)
            if f1_match:
                f1 = float(f1_match.group(1))
                all_ckpts.append((f1, entry))
        elif entry.is_dir():
            # Try subdirectory: "best-epoch=XXX-val_f1=0.XXXX" containing "val_f1=0.XXXX.ckpt"
            epoch_match = re.match(r"best-epoch=(\d+)-val_f1", entry.name)
            if not epoch_match:
                continue
            ckpt_files = list(entry.glob("val_f1=*.ckpt"))
            if not ckpt_files:
                continue
            ckpt_file = ckpt_files[0]
            f1_match = re.match(r"val_f1=([\d.]+)\.ckpt$", ckpt_file.name)
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
        ckpt_paths:   Paths to checkpoints to ensemble (top-k by val/f1)
        test_dataloader: DataLoader for the test set
        out_path:     Path to save final ensemble predictions TSV
        device:       Device for inference (typically cuda:0)
    """
    all_preds_list: List[np.ndarray] = []
    pert_ids_final: Optional[List[str]] = None
    symbols_final: Optional[List[str]] = None

    print(f"\n[Ensemble] Averaging logits from {len(ckpt_paths)} checkpoints:")
    for ckpt_path in ckpt_paths:
        print(f"  Loading: {ckpt_path.name if ckpt_path.is_file() else ckpt_path}")

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
        description="Node1-3-2-2-1-1-1-1: STRING-only + Flat Head + hidden=384 + "
                    "Muon LR=0.01 + No Label Smoothing + RLROP patience=15 + top-5 ensemble"
    )
    p.add_argument("--micro-batch-size",      type=int,   default=32)
    p.add_argument("--global-batch-size",     type=int,   default=256)
    p.add_argument("--max-epochs",            type=int,   default=300,
                   help="Extended from parent's 200 to allow RLROP halvings to complete")
    p.add_argument("--lr",                    type=float, default=3e-4)
    p.add_argument("--muon-lr",               type=float, default=0.01,
                   help="KEY CHANGE: 0.02→0.01 for stable convergence (per node3-3-1-1's recipe)")
    p.add_argument("--weight-decay",          type=float, default=5e-4)
    p.add_argument("--hidden-dim",            type=int,   default=384)
    p.add_argument("--n-blocks",              type=int,   default=3)
    p.add_argument("--dropout",               type=float, default=0.30)
    p.add_argument("--head-dropout",          type=float, default=0.15,
                   help="Dropout before flat output head (proven optimal)")
    p.add_argument("--label-smoothing",       type=float, default=0.0,
                   help="KEY CHANGE: 0.05→0.0 (removes training loss floor per node3-3-1-1)")
    p.add_argument("--class-weight-alpha",    type=float, default=0.5,
                   help="Temperature exponent for class weights: w_i=(1/freq_i)^alpha. "
                        "0.5=sqrt (softened, from parent's proven 0.4879 config)")
    p.add_argument("--rlrop-patience",        type=int,   default=15,
                   help="KEY CHANGE: 8→15 (accommodates noisy 141-sample val/f1 signal)")
    p.add_argument("--rlrop-factor",          type=float, default=0.5,
                   help="ReduceLROnPlateau LR reduction factor")
    p.add_argument("--early-stop-patience",   type=int,   default=35,
                   help="Extended from parent's 25 to allow post-halving convergence")
    p.add_argument("--grad-clip-norm",        type=float, default=1.0)
    p.add_argument("--no-muon",               action="store_true",
                   help="Disable Muon optimizer, fall back to AdamW for all params")
    p.add_argument("--ensemble-top-k",        type=int,   default=5,
                   help="Number of best checkpoints to ensemble at test time "
                        "(5 for diverse temporal coverage across RLROP phases)")
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

    # save_top_k=5: save top-5 checkpoints for diverse temporal ensemble
    # (parent saved top-3 but they were all from epochs 66-79; top-5 across RLROP halvings
    # will capture more diverse model states from different LR regimes)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-val_f1={val_f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=5,     # KEY CHANGE: 3→5 for richer ensemble diversity
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.early_stop_patience,  # 35 (extended from parent's 25)
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

        # Step 2: Ensemble test on rank 0 (overwrite test_predictions.tsv with top-5 ensemble)
        # This is a within-node ensemble; no cross-node artifacts are used.
        # With RLROP halvings expected across ~250-300 epochs, the top-5 checkpoints
        # should span different LR phases, providing genuine temporal diversity.
        if trainer.is_global_zero and args.ensemble_top_k > 1:
            ckpt_dir = output_dir / "checkpoints"
            # Discover checkpoints by scanning the local checkpoint directory.
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
