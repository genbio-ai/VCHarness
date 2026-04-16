"""Node 1-3-2-3-1: STRING-only + Flat Head + Hidden=384 + CosineWarmRestarts + Manifold Mixup

Key changes from parent (node1-3-2-3, F1=0.4635):
  1. REMOVE Manifold Mixup in isolation — but apply it in the PROVEN configuration:
     - CosineAnnealingWarmRestarts (T_0=80, T_mult=2) replaces ReduceLROnPlateau
     - Manifold Mixup (alpha=0.2, prob=0.5) — same parameters as parent
     - CosineWarmRestarts enables Mixup to work by providing proper LR dynamics
  2. Fix LR scheduler: RLROP → CosineAnnealingWarmRestarts (T_0=80, T_mult=2)
     — Parent's RLROP never fired (LR stuck at 0.01 all 83 epochs)
     — CosineWarmRestarts with warm restarts escapes local optima at T_0=80, 240, 560...
     — node1-3-3 proved: CosineWarmRestarts + Mixup → F1=0.4950 (+0.02 over parent)
  3. Increase weight_decay: 5e-4 → 8e-4 (proven in node1-3-2-2-1-1-1-1-1-1, F1=0.4968)
  4. Increase head_dropout: 0.05 → 0.15 (proven in node1-3-2-2-1, F1=0.4777; and
     node1-3-2-2-1-1-1-1-1-1, F1=0.4968 used 0.15)
  5. Remove checkpoint ensemble: single best checkpoint only
     — Parent's ensemble degraded F1 by 0.0002 (near-identical adjacent-epoch checkpoints)
  6. Extend max_epochs: 300 → 500
     — node1-3-3 needed 500 epochs to fully converge with CosineWarmRestarts
     — CosineWarmRestarts T_0=80, T_mult=2 completes cycles at 80, 240, 560 epochs
     — Extended training allows 3rd warm restart cycle to find optimal basin
  7. Extend early_stop_patience: 30 → 80
     — Allow plateau exploration across full warm-restart cycles
     — node1-3-3: stopped at epoch 500 (still improving, no early stop)
  8. Muon LR stays at 0.01 (proven optimal for h=384, no LS)
  9. No label smoothing (proven superior in node3-3-1, node3-3-1-2)

Tree context:
  node1-3-2-3 (parent)      | F1=0.4635 | STRING+h=384+Muon(0.01)+RLROP+Mixup → RLROP never fired
  node1-3-3 (reference)     | F1=0.4950 | STRING+h=384+Muon+CosineWR+Mixup+wd=8e-4+head_drop=0.15
  node1-3-2-2-1-1-1-1-1-1  | F1=0.4968 | STRING+h=384+Muon+CosineWR+Mixup+wd=8e-4+head_drop=0.15
  This node targets F1 > 0.495 via proven CosineWarmRestarts + Mixup + extended training
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


class PerturbMLP(nn.Module):
    """STRING-only MLP with Manifold Mixup support for gene perturbation response prediction.

    Architecture (per sample):
      ① STRING_GNN embedding lookup [256-dim, frozen buffer]
         (fallback learnable 256-dim for ~6% genes not in STRING)
      ② Input projection: Linear(256→hidden_dim) + LN + GELU
         [Manifold Mixup applied here during training]
      ③ n_blocks × ResidualBlock(hidden_dim)
      ④ Head dropout + LN(hidden_dim) + Linear(hidden_dim → 6640*3) + per-gene-bias [6640*3]
      ⑤ reshape → [B, 3, 6640]

    Key changes from parent (node1-3-2-3):
      - head_dropout increased: 0.05 → 0.15 (proven in node1-3-3, node1-3-2-2-1-1-1-1-1-1)
      - LR scheduler changed: RLROP → CosineAnnealingWarmRestarts (T_0=80, T_mult=2)
      - weight_decay increased: 5e-4 → 8e-4
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.35,
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
        # Output head: head_dropout + LayerNorm + flat Linear + per-gene additive bias
        # head_dropout=0.15 is the proven sweet spot from node1-3-3/node1-3-2-2-1-1-1-1-1-1
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, N_GENES * N_CLASSES),
        )
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES * N_CLASSES))

    def get_embedding(
        self,
        str_idx: torch.Tensor,       # [B]  STRING node indices, -1 = not in graph
        string_embs: torch.Tensor,   # [18870, 256] frozen buffer
    ) -> torch.Tensor:
        """Compute the projected embedding (after input_proj). Used by Mixup."""
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

        # ②  Project to hidden_dim
        x = self.input_proj(emb)  # [B, hidden_dim]
        return x

    def forward_from_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP from projected hidden state (after input_proj)."""
        for block in self.blocks:
            x = block(x)
        x = self.head_dropout(x)
        logits = self.head(x) + self.gene_bias.to(x)  # [B, N_GENES * N_CLASSES]
        return logits.view(-1, N_CLASSES, N_GENES)      # [B, 3, 6640]

    def forward(
        self,
        str_idx: torch.Tensor,       # [B]  STRING node indices, -1 = not in graph
        string_embs: torch.Tensor,   # [18870, 256] frozen buffer
        mixup_lam: Optional[float] = None,
        mixup_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional Manifold Mixup in the projected embedding space."""
        x = self.get_embedding(str_idx, string_embs)

        # Apply Manifold Mixup: mix the projected embeddings before the residual blocks
        if mixup_lam is not None and mixup_index is not None:
            x = mixup_lam * x + (1 - mixup_lam) * x[mixup_index]

        return self.forward_from_embedding(x)


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.35,
        head_dropout: float = 0.15,
        lr: float = 3e-4,
        muon_lr: float = 0.01,
        weight_decay: float = 8e-4,
        label_smoothing: float = 0.0,
        cosine_T0: int = 80,
        cosine_T_mult: int = 2,
        grad_clip_norm: float = 1.0,
        use_muon: bool = True,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.5,
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
        self.cosine_T0 = cosine_T0
        self.cosine_T_mult = cosine_T_mult
        self.grad_clip_norm = grad_clip_norm
        self.use_muon = use_muon
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob

        # Model is initialized here in __init__ (not setup) to ensure all parameters
        # exist at optimizer construction time during DDP setup.
        # STRING_GNN embeddings (frozen buffer) are loaded in setup() since they require
        # GPU allocation and are not needed for optimizer construction.
        self.model = PerturbMLP(
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

        # Cast trainable parameters to float32 for stable optimization
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        self.print(
            f"Node1-3-2-3-1 PerturbMLP | hidden={self.hidden_dim} | blocks={self.n_blocks} | "
            f"dropout={self.dropout} | head_dropout={self.head_dropout} | "
            f"use_muon={self.use_muon} | muon_lr={self.muon_lr} | wd={self.weight_decay} | "
            f"CosineWR(T0={self.cosine_T0},T_mult={self.cosine_T_mult}) | "
            f"mixup_alpha={self.mixup_alpha} | mixup_prob={self.mixup_prob} | "
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
            # Mixed targets: lam * loss_a + (1-lam) * loss_b
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
        # Manifold Mixup: apply with probability mixup_prob
        if self.training and self.mixup_alpha > 0 and self.mixup_prob > 0:
            if torch.rand(1).item() < self.mixup_prob:
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                B = batch["str_idx"].shape[0]
                index = torch.randperm(B, device=batch["str_idx"].device)
                logits = self.model(
                    batch["str_idx"], self.string_embs,
                    mixup_lam=lam, mixup_index=index,
                )
                loss = self._compute_loss(logits, batch["label"], lam=lam, index=index)
            else:
                logits = self.model(batch["str_idx"], self.string_embs)
                loss = self._compute_loss(logits, batch["label"])
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

        # Gather string metadata: encode strings as integer tensors, all_gather, then decode on rank 0
        if ws > 1:
            MAX_CHARS = 32  # ENSEMBL IDs are <= 15 chars, symbols <= 15 chars

            def encode_strings(strings: List[str]) -> torch.Tensor:
                """Encode strings to int32 tensor: [N, MAX_CHARS]."""
                result = torch.zeros(len(strings), MAX_CHARS, dtype=torch.int32, device=self.device)
                for i, s in enumerate(strings):
                    for j, c in enumerate(s[:MAX_CHARS]):
                        result[i, j] = ord(c)
                return result

            def decode_strings(tensor: torch.Tensor, n: int) -> List[str]:
                """Decode first n rows of tensor back to strings."""
                strings = []
                for i in range(n):
                    chars = [chr(tensor[i, j].item()) for j in range(MAX_CHARS) if tensor[i, j] > 0]
                    strings.append("".join(chars))
                return strings

            # Get per-rank counts first (flatten the [1, ws] tensor from all_gather)
            local_count_t = torch.tensor([len(self._test_pert_ids)], device=self.device, dtype=torch.long)
            all_counts_flat = self.all_gather(local_count_t).cpu().flatten().tolist()  # [ws]
            max_per_rank = max(all_counts_flat)

            # Encode strings and pad to max_per_rank
            ids_enc = encode_strings(self._test_pert_ids)
            syms_enc = encode_strings(self._test_symbols)
            ids_padded = F.pad(ids_enc, (0, 0, 0, max(0, max_per_rank - len(self._test_pert_ids))))
            syms_padded = F.pad(syms_enc, (0, 0, 0, max(0, max_per_rank - len(self._test_symbols))))

            # all_gather: result [ws, max_per_rank, MAX_CHARS]
            all_ids_enc = self.all_gather(ids_padded)
            all_syms_enc = self.all_gather(syms_padded)

            # Decode on global zero using per-rank counts
            all_pert_ids: List[str] = []
            all_symbols: List[str] = []
            if self.trainer.is_global_zero:
                for rank_idx, count in enumerate(all_counts_flat):
                    all_pert_ids.extend(decode_strings(all_ids_enc[rank_idx], count))
                    all_symbols.extend(decode_strings(all_syms_enc[rank_idx], count))
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
        """Configure optimizer: Muon for hidden MLP weight matrices (lr=0.01), AdamW for everything else.

        Key changes vs parent (node1-3-2-3):
          - LR scheduler changed: ReduceLROnPlateau → CosineAnnealingWarmRestarts (T_0=80, T_mult=2)
          - CosineWarmRestarts enables warm restarts to escape local optima
          - Proven superior for STRING-only + Mixup: node1-3-3 (F1=0.4950), node1-3-2-2-1-1-1-1-1-1 (F1=0.4968)
          - Muon LR=0.01 for hidden 2D weight matrices (unchanged from parent)
          - AdamW LR=3e-4 for other params (unchanged from parent)
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
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        # CosineAnnealingWarmRestarts: T_0=80, T_mult=2
        # Cycle lengths: 80, 160, 320, ... epochs
        # Warm restarts proven to escape local optima in STRING+Mixup configuration:
        #   - node1-3-3 (F1=0.4950): used T_0=80, T_mult=2, 500 epochs
        #   - node1-3-2-2-1-1-1-1-1-1 (F1=0.4968): used T_0=80, T_mult=2, 350+ epochs
        # The key advantage: LR is periodically reset to the initial value (0.01/3e-4),
        # preventing convergence to suboptimal local basins that RLROP cannot escape.
        # This is critical for Mixup-augmented training where the loss landscape is
        # smoother and RLROP patience=8 never fires (as seen in the parent).
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cosine_T0,
            T_mult=self.cosine_T_mult,
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
        # Essential small buffers (class_weights); exclude large frozen string_embs
        large_frozen = {"string_embs"}
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
    """Save test predictions in required TSV format (idx / input / prediction).

    Handles deduplication robustly: in DDP with DistributedSampler, the same
    pert_id may appear multiple times (from padding). We keep only the first
    occurrence of each pert_id, ensuring unique IDs in the output.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_ids: set[str] = set()
    for i, (pid, sym) in enumerate(zip(pert_ids, symbols)):
        # Skip duplicates (from DistributedSampler's padding rows)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
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
        description="Node1-3-2-3-1: STRING-only + Flat Head + h=384 + CosineWarmRestarts + Manifold Mixup"
    )
    p.add_argument("--micro-batch-size",    type=int,   default=32)
    p.add_argument("--global-batch-size",   type=int,   default=256)
    p.add_argument("--max-epochs",          type=int,   default=500)
    p.add_argument("--lr",                  type=float, default=3e-4)
    p.add_argument("--muon-lr",             type=float, default=0.01)
    p.add_argument("--weight-decay",        type=float, default=8e-4)
    p.add_argument("--hidden-dim",          type=int,   default=384)
    p.add_argument("--n-blocks",            type=int,   default=3)
    p.add_argument("--dropout",             type=float, default=0.35)
    p.add_argument("--head-dropout",        type=float, default=0.15)
    p.add_argument("--label-smoothing",     type=float, default=0.0)
    p.add_argument("--cosine-T0",           type=int,   default=80,
                   help="CosineAnnealingWarmRestarts T_0 (first cycle length in epochs)")
    p.add_argument("--cosine-T-mult",       type=int,   default=2,
                   help="CosineAnnealingWarmRestarts T_mult (cycle length multiplier)")
    p.add_argument("--early-stop-patience", type=int,   default=80,
                   help="Extended patience to survive across warm restart cycles")
    p.add_argument("--grad-clip-norm",      type=float, default=1.0)
    p.add_argument("--mixup-alpha",         type=float, default=0.2)
    p.add_argument("--mixup-prob",          type=float, default=0.5)
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
        cosine_T0=args.cosine_T0,
        cosine_T_mult=args.cosine_T_mult,
        grad_clip_norm=args.grad_clip_norm,
        use_muon=not args.no_muon,
        mixup_alpha=mixup_alpha,
        mixup_prob=mixup_prob,
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

    # Single best checkpoint — no ensemble (ensemble was counterproductive in parent)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-epoch={epoch:03d}-val_f1={val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,   # Single best checkpoint only
        save_last=True,
        auto_insert_metric_name=False,  # Prevent nested dir from "/" in metric name
    )
    early_stop_cb = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.early_stop_patience,  # Extended to 80 to survive warm restart valleys
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
            find_unused_parameters=True,   # Required for Muon + AMP interaction
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

    # --- Test with single best checkpoint ---
    if args.fast_dev_run or args.debug_max_step is not None:
        # No best checkpoint in debug mode
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        # Use single best checkpoint (no ensemble — ensemble was harmful in parent)
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # --- Save test score (rank 0 only) ---
    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")


if __name__ == "__main__":
    main()
