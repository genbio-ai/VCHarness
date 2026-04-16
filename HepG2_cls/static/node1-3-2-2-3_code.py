"""Node 1-3-2-2-3: STRING-only + Flat Head + Hidden Dim=384 + Muon lr=0.01 +
   CosineAnnealingWarmRestarts(T_0=80, T_mult=2) + head_dropout=0.15 + wd=8e-4

Key changes from parent (node1-3-2-2, F1=0.4750):
  1. hidden_dim: 416 → 384    — revert to proven best; 416 was marginally worse
  2. head_dropout: 0.0 → 0.15 — breakthrough from sibling node1-3-2-2-1 (+0.0027 F1)
  3. trunk dropout: 0.32 → 0.30 — matches proven sibling1 balance
  4. Muon lr: 0.02 → 0.01     — proven in F1=0.497 deep-tree nodes; gentler hidden
                                  block optimization pairs better with head_dropout=0.15
  5. LR schedule: CosineAnnealing → CosineAnnealingWarmRestarts(T_0=80, T_mult=2)
     — warm restarts periodically reset LR to escape local optima; proven to push
       STRING-only to F1≈0.497 in deep tree; fundamentally distinct from both siblings' RLROP
  6. weight_decay: 5e-4 → 8e-4 — higher WD proven in F1=0.4968 deep-tree node
  7. max_epochs: 200 → 500     — warm restart cycle 3 begins at epoch 240; peak often
                                  near epoch 469 in confirmed successful runs
  8. early_stop_patience: 30 → 160 — must survive cycle restarts; cycle 3 ~320 epochs long
  9. Keep all proven elements: STRING-only (frozen 256-dim PPI), flat output head,
     3 PreLN residual blocks, per-gene bias, weighted CE + label_smoothing=0.05,
     gradient clipping max_norm=1.0, global_batch_size=256

Tree context:
  node1-3-2-2 (parent)     | F1=0.4750 | hidden=416, cosine LR, no head dropout
  node1-3-2-2-1 (sibling1) | F1=0.4777 | hidden=384, RLROP(p=8), head_drop=0.15 [TREE BEST]
  node1-3-2-2-2 (sibling2) | F1=0.4746 | hidden=384, RLROP(p=10), head_drop=0.20 [REGRESSION]
  node1-3-2-2-3 (this)     | F1=???    | hidden=384, WarmRestarts, head_drop=0.15, muon_lr=0.01

Why CosineAnnealingWarmRestarts instead of RLROP:
  - Both siblings used RLROP; this node takes a completely different scheduling path
  - Deep-tree node (F1=0.4968) with T_0=80/T_mult=2 + 500 epochs proves warm restarts
    find better optima than RLROP's reactive halvings on this dataset
  - Warm restarts reset LR at cycle boundaries (epochs 80, 240, 560...), escaping
    local optima and allowing exploration of new parameter basins

Why Muon lr=0.01:
  - Sibling 2 feedback (Priority 4): "Test Muon lr=0.01 while keeping head_dropout=0.15"
  - F1=0.4968 deep-tree node confirmed Muon lr=0.01 effective
  - Lower Muon lr provides gentler hidden block orthogonalization, less risk of
    overshooting during the long warm restart cycles
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
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
      ③ n_blocks × ResidualBlock(hidden_dim, dropout)
      ④ LN(hidden_dim) + Dropout(head_dropout) + Linear(hidden_dim → 6640*3)
         + per-gene-bias [6640*3]
      ⑤ reshape → [B, 3, 6640]

    Key changes vs parent (node1-3-2-2):
      - hidden_dim=384 (not 416): reverted to proven optimal
      - dropout=0.30 (not 0.32): consistent with proven sibling
      - head_dropout=0.15 (not 0.0): targeted output head regularization —
        this was the breakthrough innovation in node1-3-2-2-1 (+0.0021 F1)
        that broke the F1~0.475 ceiling across 10+ STRING-only nodes
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
        # Flat output head + output-head dropout (key regularization for 7.6M-param head)
        # LayerNorm → Dropout(head_dropout) → Linear → per-gene bias
        # head_dropout=0.15 was the breakthrough in sibling node1-3-2-2-1 (+0.0021 F1)
        # It directly regularizes the dominant overfitting source (the large flat head)
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout > 0.0 else nn.Identity()
        self.head_linear = nn.Linear(hidden_dim, N_GENES * N_CLASSES)
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

        # ②–③  Projection → residual MLP
        x = self.input_proj(emb)            # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)

        # ④  Output head with head dropout
        x = self.head_norm(x)               # [B, hidden_dim]
        x = self.head_dropout(x)            # [B, hidden_dim] — stochastic masking during train
        logits = self.head_linear(x) + self.gene_bias.to(x)  # [B, N_GENES * N_CLASSES]
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
        muon_lr: float = 0.01,
        weight_decay: float = 8e-4,
        label_smoothing: float = 0.05,
        t0_epochs: int = 80,
        t_mult: int = 2,
        max_epochs: int = 500,
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
        self.t0_epochs = t0_epochs
        self.t_mult = t_mult
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

    def setup(self, stage: str = "fit") -> None:
        # Class weights: inversely proportional to class frequencies
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
            f"Node1-3-2-2-3 PerturbMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
            f"trunk_dropout={self.dropout} | head_dropout={self.head_dropout} | "
            f"muon_lr={self.muon_lr} | weight_decay={self.weight_decay} | "
            f"T_0={self.t0_epochs} | T_mult={self.t_mult} | max_epochs={self.max_epochs} | "
            f"trainable={n_trainable:,}/{n_total:,}"
        )

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted cross-entropy with label smoothing.

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
        # This avoids gather_object synchronization issues in DDP
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

        # Use larger max_len for ENSEMBL IDs (e.g., "ENSG00000123456" = 15 chars)
        # and gene symbols (up to ~30 chars for long names)
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

        LR schedule: CosineAnnealingWarmRestarts(T_0=80, T_mult=2) via LambdaLR.

        Warm restarts provide periodic LR resets that help the model escape local optima
        and explore new parameter basins. This is distinct from both siblings' RLROP:
        - Sibling 1: RLROP(patience=8) — reactive, triggers on val/f1 plateaus
        - Sibling 2: RLROP(patience=10) — same mechanism, longer patience
        - This node: Warm restarts — proactive, fixed cycle boundaries (80, 240, 560 epochs)

        Deep-tree evidence: node1-3-2-2-1-1-1-1-1-1 (F1=0.4968) achieved best val_f1=0.4999
        at epoch 469 (cycle 3) using T_0=80/T_mult=2 with 500 epochs.

        Muon (lr=0.01): Applied only to 2D weight matrices in hidden residual blocks.
        - Lower than siblings' 0.02: gentler orthogonalization pairs well with head_dropout
        - Proven in F1=0.497 deep-tree nodes; recommended by sibling 2 feedback (Priority 4)

        AdamW (lr=3e-4): Applied to all other parameters (input_proj, head, norms, biases).
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

        # CosineAnnealingWarmRestarts via LambdaLR.
        # Implemented as a multiplicative factor so it works uniformly for both
        # Muon (lr=0.01) and AdamW (lr=3e-4) param groups.
        #
        # Cycle schedule with T_0=80 and T_mult=2:
        #   Cycle 1: epochs  0–79    (length=80)   — first peak at epoch 0
        #   Cycle 2: epochs 80–239   (length=160)  — warm restart at epoch 80
        #   Cycle 3: epochs 240–559  (length=320)  — warm restart at epoch 240
        #   Cycle 4: epochs 560–...  (length=640)  — warm restart at epoch 560
        #
        # eta_min_frac=1e-3: LR decays to 0.1% of peak at each cycle trough.
        t0 = self.t0_epochs
        t_mult = self.t_mult
        eta_min_frac = 1e-3  # final LR = initial_lr * eta_min_frac at each trough

        def cosine_warmrestart_lambda(current_epoch: int) -> float:
            """Cosine annealing with warm restarts (T_0, T_mult).

            Returns the LR multiplier in [eta_min_frac, 1.0] for the current epoch.
            """
            T_cur = current_epoch
            cycle_len = t0
            # Advance through completed cycles
            while T_cur >= cycle_len:
                T_cur -= cycle_len
                cycle_len = cycle_len * t_mult
            # T_cur is now the position within the current cycle [0, cycle_len)
            progress = T_cur / max(1, cycle_len)
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_frac + (1.0 - eta_min_frac) * cosine_val

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=cosine_warmrestart_lambda
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
        # Guard for internal copy scenario: self.model may be None in deepcopy
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


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Node1-3-2-2-3: STRING-only + Flat Head + Head Dropout=0.15 + "
            "Muon lr=0.01 + CosineAnnealingWarmRestarts(T_0=80, T_mult=2)"
        )
    )
    p.add_argument("--micro-batch-size",       type=int,   default=32)
    p.add_argument("--global-batch-size",      type=int,   default=256)
    p.add_argument("--max-epochs",             type=int,   default=500,
                   help="Extended for warm restart cycles (cycle 3 peaks ~epoch 469)")
    p.add_argument("--lr",                     type=float, default=3e-4)
    p.add_argument("--muon-lr",                type=float, default=0.01,
                   help="Muon lr=0.01 (vs siblings' 0.02) — proven in F1=0.497 deep-tree nodes")
    p.add_argument("--weight-decay",           type=float, default=8e-4,
                   help="weight_decay=8e-4 (vs 5e-4 in siblings) — proven in F1=0.4968 node")
    p.add_argument("--hidden-dim",             type=int,   default=384)
    p.add_argument("--n-blocks",               type=int,   default=3)
    p.add_argument("--dropout",                type=float, default=0.30)
    p.add_argument("--head-dropout",           type=float, default=0.15,
                   help="Output head dropout — breakthrough from sibling1 (+0.0027 F1)")
    p.add_argument("--label-smoothing",        type=float, default=0.05)
    p.add_argument("--t0-epochs",              type=int,   default=80,
                   help="T_0 for CosineAnnealingWarmRestarts (first cycle length)")
    p.add_argument("--t-mult",                 type=int,   default=2,
                   help="T_mult for CosineAnnealingWarmRestarts (cycle length multiplier)")
    p.add_argument("--early-stop-patience",    type=int,   default=160,
                   help="Patience=160 to survive warm restart cycles (cycle 3 ~320 epochs)")
    p.add_argument("--grad-clip-norm",         type=float, default=1.0)
    p.add_argument("--no-muon",                action="store_true",
                   help="Disable Muon optimizer, fall back to AdamW for all params")
    p.add_argument("--num-workers",            type=int,   default=4)
    p.add_argument("--val-check-interval",     type=float, default=1.0)
    p.add_argument("--debug_max_step",         type=int,   default=None)
    p.add_argument("--fast_dev_run",           action="store_true")
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
        t0_epochs=args.t0_epochs,
        t_mult=args.t_mult,
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
        limit_test = 1.0   # Always evaluate full test set (bug fix from sibling analysis)
        max_steps = args.debug_max_step
    else:
        limit_train = 1.0
        limit_val = 1.0
        limit_test = 1.0
        max_steps = -1

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val_f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=3,   # Save top-3 for 500-epoch run; deep tree peak at epoch 469
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
            find_unused_parameters=True,  # True needed: fallback_emb + norm/bias params may not be used in every forward pass
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
        gradient_clip_val=args.grad_clip_norm,  # Gradient clipping for stable optimization
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
