"""Node 1-3-1-3: Proven node1-3-2/node1-3-3 Recipe — Muon + PreLN + hidden=384 + CosineWarmRestarts + Manifold Mixup

This node transfers the proven recipe from the successful node1-3-2/node1-3-3 lineage
(F1=0.4756→0.4950) to address the systematic failures of node1-3-1's children:
  - node1-3-1-1 (sibling): hidden=384 + flat head + AdamW only → F1=0.431 (no Muon, no WR, no Mixup)
  - node1-3-1-2 (sibling): hidden=512 + flat head + Muon + dropout=0.40 → F1=0.430 (wrong hidden, RLROP)
  - node1-3-1 (parent): hidden=512 + factorized head + AdamW → F1=0.430 (factorized head failure)

Key recipe from node1-3-2/node1-3-3 lineage (the proven STRING-only high-performance baseline):
  1. MuonWithAuxAdam: Muon (lr=0.01) for hidden 2D weight matrices + AdamW (lr=3e-4) for head/norm/bias
  2. hidden_dim=384 with PreLN (Pre-LayerNorm) residual blocks — proven capacity sweet spot
  3. Flat output head: LayerNorm(384) → Dropout(0.15) → Linear(384→19920) + per-gene bias
  4. CosineAnnealingWarmRestarts (T_0=80, T_mult=2) — warm restarts escape local optima (+0.017 at restart)
  5. Manifold Mixup (alpha=0.2, prob=0.5) — transforms 1,273 training samples into a continuous manifold
  6. Weighted cross-entropy (no label smoothing) — proven optimal with Muon (label smoothing hurts Muon)
  7. weight_decay=8e-4, head_dropout=0.15 — proven regularization balance
  8. 500 epochs max — warm restart cycles need sufficient epochs (node1-3-3 peaked at epoch 468)
  9. top-5 checkpoint ensemble — improves generalization on the small dataset
  10. gradient clipping max_norm=1.0 — stabilizes Muon's aggressive orthogonal updates
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
class PreLNResidualBlock(nn.Module):
    """Pre-LayerNorm residual MLP block (dim → dim*2 → dim).

    Uses pre-normalization (normalize BEFORE the sublayer) which is the proven
    configuration for the Muon optimizer in the node1-3-2 lineage:
    - PreLN provides stable gradient flow for Newton-Schulz orthogonalization
    - Muon's orthogonal gradient updates work best with PreLN architecture
    - All high-performing Muon STRING-only nodes (F1=0.4756-0.4968) use PreLN
    """

    def __init__(self, dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2, bias=True)
        self.linear2 = nn.Linear(dim * 2, dim, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalize
        h = self.norm(x)
        # Feedforward
        h = self.linear1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.linear2(h)
        h = self.drop2(h)
        return self.act(x + h)


class StringOnlyPreLNMLP(nn.Module):
    """STRING-only MLP with PreLN residual blocks and flat output head.

    Architecture (input → output):
      ① STRING_GNN embedding lookup [B, 256] frozen buffer
         (fallback learnable 256-dim for ~6% genes absent from STRING)
      ② Input projection: Linear(256→hidden_dim) + LayerNorm(hidden_dim) + GELU
      ③ n_blocks × PreLNResidualBlock(hidden_dim)
      ④ Flat output head: LayerNorm(hidden_dim) → Dropout(head_dropout) → Linear(hidden_dim→N_GENES*N_CLASSES)
         + per-gene additive bias [N_GENES*N_CLASSES]
      ⑤ Reshape → [B, 3, 6640]

    Key design choices aligned with proven node1-3-2/node1-3-3 lineage:
      - hidden_dim=384 (proven overfitting sweet spot, better than 512)
      - PreLN residual blocks (proven Muon-compatible architecture)
      - Flat head (factorized head confirmed harmful across 3 experiments)
      - head_dropout=0.15 (confirmed optimal, 0.20 over-regularizes)
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
        # Note: linear weight exposed separately for Muon optimizer assignment
        self.input_proj_linear = nn.Linear(256, hidden_dim, bias=True)
        self.input_proj_norm = nn.LayerNorm(hidden_dim)
        self.input_proj_act = nn.GELU()

        # PreLN Residual MLP blocks
        self.blocks = nn.ModuleList(
            [PreLNResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )

        # Flat output head with dropout regularization
        # LayerNorm → Dropout → Linear(hidden_dim→N_GENES*N_CLASSES)
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head_drop = nn.Dropout(head_dropout)
        self.head_linear = nn.Linear(hidden_dim, N_GENES * N_CLASSES, bias=True)

        # Per-gene additive bias: captures baseline DE tendencies per response gene
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES * N_CLASSES))

    def forward(
        self,
        str_idx: torch.Tensor,       # [B]  STRING node indices, -1 = not in graph
        string_embs: torch.Tensor,   # [18870, 256] frozen buffer
    ) -> torch.Tensor:
        valid_mask = str_idx >= 0                    # [B] bool
        safe_idx = str_idx.clamp(min=0)              # replace -1 with 0 (overwritten below)

        # ① Look up frozen STRING embeddings [B, 256]
        emb = string_embs[safe_idx].to(torch.float32)  # [B, 256]

        # Overwrite samples whose gene is absent from the STRING graph with fallback
        if not valid_mask.all():
            fallback = self.fallback_emb.to(emb).unsqueeze(0).expand(
                int((~valid_mask).sum()), -1
            )
            emb = emb.clone()
            emb[~valid_mask] = fallback

        # ② Input projection
        x = self.input_proj_act(self.input_proj_norm(self.input_proj_linear(emb)))

        # ③ PreLN residual MLP blocks
        for block in self.blocks:
            x = block(x)

        # ④ Flat output head + per-gene bias
        x = self.head_norm(x)
        x = self.head_drop(x)
        logits = self.head_linear(x) + self.gene_bias.to(x)  # [B, N_GENES * N_CLASSES]
        return logits.view(-1, N_CLASSES, N_GENES)      # [B, 3, 6640]


# ---------------------------------------------------------------------------
# Manifold Mixup
# ---------------------------------------------------------------------------
def manifold_mixup_hidden(
    x: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
    prob: float = 0.5,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, float]]]:
    """Apply Manifold Mixup in hidden space with probability `prob`.

    Returns:
        x_mixed: possibly mixed hidden representation
        mix_info: None if no mixing, else (labels_a, labels_b, lam) for loss computation
    """
    if not x.requires_grad:
        return x, None
    if random.random() > prob:
        return x, None

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    rand_idx = torch.randperm(batch_size, device=x.device)

    x_mixed = lam * x + (1 - lam) * x[rand_idx]
    labels_a = labels
    labels_b = labels[rand_idx]

    return x_mixed, (labels_a, labels_b, lam)


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
        muon_weight_decay: float = 8e-4,
        lr_patience: int = 8,   # kept for potential RLROP fallback
        lr_factor: float = 0.5,
        t0: int = 80,
        t_mult: int = 2,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.5,
        n_ensemble_ckpts: int = 5,
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
        self.muon_weight_decay = muon_weight_decay
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.t0 = t0
        self.t_mult = t_mult
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.n_ensemble_ckpts = n_ensemble_ckpts

        self.model: Optional[StringOnlyPreLNMLP] = None

        # Accumulation buffers for epoch-level metrics
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Class weights: inversely proportional to class frequencies
        # After {-1,0,1}→{0,1,2}: class0=down(4.77%), class1=neutral(92.82%), class2=up(2.41%)
        # NOTE: Correct ordering — class 0 is down-regulated (freq=4.77%), NOT neutral
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

        # ---- Build model (STRING-only, PreLN, flat head) ----
        self.model = StringOnlyPreLNMLP(
            hidden_dim=self.hidden_dim,
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
            f"Node1-3-1-3 StringOnlyPreLNMLP | hidden={self.hidden_dim} | "
            f"blocks={self.n_blocks} | head_dropout={self.head_dropout} | "
            f"muon_lr={self.muon_lr} | t0={self.t0} | t_mult={self.t_mult} | "
            f"mixup_prob={self.mixup_prob} | "
            f"trainable={n_trainable:,}/{n_total:,}"
        )

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted cross-entropy (no label smoothing — proven optimal with Muon).

        logits: [B, 3, 6640]
        labels: [B, 6640]  — values in {0, 1, 2}
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES)  # [B*6640, 3]
        labels_flat = labels.reshape(-1)                                # [B*6640]
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=0.0,  # No label smoothing (proven better with Muon)
        )

    def _compute_mixup_loss(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Mixup loss = lam * CE(logits, a) + (1-lam) * CE(logits, b)."""
        loss_a = self._compute_loss(logits, labels_a)
        loss_b = self._compute_loss(logits, labels_b)
        return lam * loss_a + (1.0 - lam) * loss_b

    def _forward_with_mixup(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Any]:
        """Forward pass with Manifold Mixup in hidden space.

        Mixup is applied to the hidden state AFTER the input projection,
        BEFORE the residual blocks — the 'manifold' is the pre-block hidden space.
        This interpolates between learned representations rather than raw inputs.
        """
        str_idx = batch["str_idx"]
        labels = batch.get("label", None)

        valid_mask = str_idx >= 0
        safe_idx = str_idx.clamp(min=0)
        emb = self.string_embs[safe_idx].to(torch.float32)

        if not valid_mask.all():
            fallback = self.model.fallback_emb.to(emb).unsqueeze(0).expand(
                int((~valid_mask).sum()), -1
            )
            emb = emb.clone()
            emb[~valid_mask] = fallback

        # Input projection
        x = self.model.input_proj_act(
            self.model.input_proj_norm(self.model.input_proj_linear(emb))
        )

        # Manifold Mixup in hidden space (only during training)
        mix_info = None
        if labels is not None and self.training:
            x, mix_info = manifold_mixup_hidden(
                x, labels, alpha=self.mixup_alpha, prob=self.mixup_prob
            )

        # Residual blocks
        for block in self.model.blocks:
            x = block(x)

        # Flat output head
        x = self.model.head_norm(x)
        x = self.model.head_drop(x)
        logits = self.model.head_linear(x) + self.model.gene_bias.to(x)
        logits = logits.view(-1, N_CLASSES, N_GENES)

        return logits, mix_info

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits, mix_info = self._forward_with_mixup(batch)

        if mix_info is not None:
            labels_a, labels_b, lam = mix_info
            loss = self._compute_mixup_loss(logits, labels_a, labels_b, lam)
        else:
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
            if all_preds.dim() == 4:
                all_preds = all_preds.squeeze(0)
                all_labels = all_labels.squeeze(0)

        preds_np = all_preds.float().cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        f1 = _compute_per_gene_f1(preds_np, labels_np)
        # sync_dist=True: even though F1 is already globally computed from all_gather,
        # Lightning still warns when epoch-level metrics are logged without sync_dist=True.
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["str_idx"], self.string_embs)
        self._test_preds.append(logits.detach().cpu())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        # ------------------------------------------------------------------ #
        # Rank-local tensors                                                   #
        # ------------------------------------------------------------------ #
        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        ws = self.trainer.world_size
        out_dir = Path(__file__).parent / "run"
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # ------------------------------------------------------------------
        # Step 1: each rank saves its OWN local predictions to a temp file.
        # Using preds_local (not all-gathered) ensures that pert_ids and
        # predictions are aligned correctly on every rank (fixed DDP bug).
        # ------------------------------------------------------------------
        rank_preds_path = out_dir / f"test_predictions_rank{local_rank}.tsv"
        preds_np = preds_local.float().cpu().numpy()  # [N_local, 3, 6640]
        _save_test_predictions(
            pert_ids=self._test_pert_ids,
            symbols=self._test_symbols,
            preds=preds_np,
            out_path=rank_preds_path,
        )

        # Save labels if available — use local labels only (no all-gather needed)
        rank_meta_path = out_dir / f"test_predictions_rank{local_rank}_meta.tsv"
        if self._test_labels:
            labels_local_np = torch.cat(self._test_labels, dim=0).cpu().numpy()  # [N_local, 6640]
            label_rows = []
            for i, (pid, sym) in enumerate(zip(self._test_pert_ids, self._test_symbols)):
                label_rows.append({
                    "idx": pid,
                    "input": sym,
                    "label": json.dumps(labels_local_np[i].tolist()),
                })
            pd.DataFrame(label_rows).to_csv(rank_meta_path, sep="\t", index=False)

        # Step 2: barrier — ensure all ranks have written their files
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Step 3: rank 0 reads all rank files and merges into final predictions
        if self.trainer.is_global_zero:
            all_pert_ids: List[str] = []
            all_symbols: List[str] = []
            all_preds_merged: List[np.ndarray] = []
            all_labels_list: List[np.ndarray] = []

            for r in range(ws):
                rp = out_dir / f"test_predictions_rank{r}.tsv"
                if rp.exists():
                    df_r = pd.read_csv(rp, sep="\t")
                    all_pert_ids.extend(df_r["idx"].tolist())
                    all_symbols.extend(df_r["input"].tolist())
                    for _, row in df_r.iterrows():
                        all_preds_merged.append(np.array(json.loads(row["prediction"])))
                rm = out_dir / f"test_predictions_rank{r}_meta.tsv"
                if rm.exists():
                    df_m = pd.read_csv(rm, sep="\t")
                    for _, row in df_m.iterrows():
                        all_labels_list.append(np.array(json.loads(row["label"])))

            # Deduplicate by pert_id (DDP DistributedSampler may pad dataset
            # by repeating some samples to make it evenly divisible).
            seen_pids: set = set()
            dedup_pids: List[str] = []
            dedup_syms: List[str] = []
            dedup_preds_list: List[np.ndarray] = []
            dedup_labels: List[np.ndarray] = []
            for pid, sym, pred in zip(all_pert_ids, all_symbols, all_preds_merged):
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    dedup_pids.append(pid)
                    dedup_syms.append(sym)
                    dedup_preds_list.append(pred)
            # Rebuild all_labels_list aligned to dedup order
            if all_labels_list:
                pid_to_label: Dict[str, np.ndarray] = {}
                label_idx = 0
                for r in range(ws):
                    rm = out_dir / f"test_predictions_rank{r}_meta.tsv"
                    if rm.exists():
                        df_m_r = pd.read_csv(rm, sep="\t")
                        for _, row in df_m_r.iterrows():
                            pid_to_label[row["idx"]] = np.array(json.loads(row["label"]))
                dedup_labels = [pid_to_label[pid] for pid in dedup_pids if pid in pid_to_label]

            all_pert_ids = dedup_pids
            all_symbols = dedup_syms
            all_preds_merged = dedup_preds_list
            if dedup_labels:
                all_labels_list = dedup_labels

            final_preds = np.stack(all_preds_merged) if all_preds_merged else np.array([]).reshape(0, 3, N_GENES)
            _save_test_predictions(
                pert_ids=all_pert_ids,
                symbols=all_symbols,
                preds=final_preds,
                out_path=out_dir / "test_predictions.tsv",
            )
            # Clean up per-rank temp files
            for r in range(ws):
                (out_dir / f"test_predictions_rank{r}.tsv").unlink(missing_ok=True)
                (out_dir / f"test_predictions_rank{r}_meta.tsv").unlink(missing_ok=True)

            # Compute inline test F1 if ground truth is available.
            # Use self.print (rank-0 only) to avoid a DDP sync_dist hang —
            # the authoritative test score is written to test_score.txt in main().
            if all_labels_list:
                all_labels_arr = np.stack(all_labels_list)
                test_f1 = _compute_per_gene_f1(final_preds, all_labels_arr)
                self.print(f"Inline test/f1 (on_test_epoch_end, rank-0 gather): {test_f1:.4f}")

        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        """MuonWithAuxAdam dual optimizer with CosineAnnealingWarmRestarts scheduler.

        Muon optimizer for hidden 2D weight matrices (proven in node1-3-2 lineage):
          - input_proj_linear.weight [384, 256]
          - block[i].linear1.weight [768, 384]
          - block[i].linear2.weight [384, 768]
        AdamW for all other parameters:
          - biases, LayerNorm params
          - head_linear.weight/bias (output head — explicitly excluded from Muon)
          - gene_bias, fallback_emb

        NOTE: If Muon unavailable or DDP multi-GPU, fall back to standard AdamW.
        DDP + Muon requires all_reduce synchronization; single-GPU runs Muon natively.
        """
        # Identify Muon parameter names: hidden 2D weight matrices in trunk only
        muon_param_names = set()
        muon_param_names.add("model.input_proj_linear.weight")
        for i in range(len(self.model.blocks)):
            muon_param_names.add(f"model.blocks.{i}.linear1.weight")
            muon_param_names.add(f"model.blocks.{i}.linear2.weight")

        # Check if running multi-GPU DDP (Muon not safe with NCCL in all configs)
        n_gpus = int(os.environ.get("WORLD_SIZE", 1))
        use_muon = (n_gpus == 1)  # safe fallback: only use Muon on single-GPU

        if use_muon:
            try:
                from muon import MuonWithAuxAdam
                muon_params = []
                adamw_params = []
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                    if name in muon_param_names:
                        muon_params.append(param)
                    else:
                        adamw_params.append(param)

                # MuonWithAuxAdam.__init__ takes a single param_groups list.
                # Each group dict must contain exactly the keys expected by the source:
                #   use_muon=True  → {params, lr, momentum, weight_decay, use_muon}
                #   use_muon=False → {params, lr, betas, eps, weight_decay, use_muon}
                muon_group = {
                    "params": muon_params,
                    "use_muon": True,
                    "lr": self.muon_lr,
                    "weight_decay": self.muon_weight_decay,
                    "momentum": 0.95,
                }
                adamw_group = {
                    "params": adamw_params,
                    "use_muon": False,
                    "lr": self.lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-10,
                    "weight_decay": self.weight_decay,
                }
                optimizer = MuonWithAuxAdam([muon_group, adamw_group])
                self.print(f"Using MuonWithAuxAdam: {len(muon_params)} Muon params, {len(adamw_params)} AdamW params")
            except ImportError:
                self.print("WARNING: Muon not available, falling back to AdamW")
                use_muon = False

        if not use_muon:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            self.print(f"Using AdamW (multi-GPU mode or Muon unavailable), lr={self.lr}")

        # CosineAnnealingWarmRestarts scheduler — proven to improve generalization
        # via warm restart cycles that escape local optima (T_0=80 → restart every 80 epochs)
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
    # (string_embs is a large frozen tensor recomputed in setup() —
    #  excluding it keeps checkpoint files small)
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
        # strict=False: string_embs is not in checkpoint but populated by setup()
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


def _load_top_k_checkpoints(
    ckpt_dir: Path,
    k: int = 5,
) -> List[Path]:
    """Load top-k checkpoint paths sorted by val/f1 descending.

    Uses regex pattern to match Lightning's checkpoint filenames like:
    'best-epoch=XXX-val_f1=0.XXXX.ckpt' or similar patterns.
    """
    import re
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    # Filter out 'last.ckpt'
    ckpt_files = [f for f in ckpt_files if "last" not in f.name]

    val_f1_list = []
    for f in ckpt_files:
        # Try multiple pattern variants to handle Lightning's filename quirks
        match = re.search(r"val[/_]f1[=_]([\d.]+)", f.name)
        if match:
            try:
                val_f1 = float(match.group(1))
                val_f1_list.append((val_f1, f))
            except ValueError:
                pass

    # Sort by val_f1 descending
    val_f1_list.sort(key=lambda x: x[0], reverse=True)
    top_k = [f for _, f in val_f1_list[:k]]

    if not top_k and ckpt_files:
        # Fallback: use all available checkpoints
        top_k = ckpt_files[:k]

    return top_k


def _ensemble_predictions_from_checkpoints(
    ckpt_paths: List[Path],
    model: PerturbModule,
    datamodule: PerturbDataModule,
    trainer: pl.Trainer,
    out_dir: Path,
) -> Optional[np.ndarray]:
    """Run inference on multiple checkpoints and average logits for ensemble prediction.

    Returns averaged predictions as numpy array [N, 3, 6640] or None if no checkpoints.
    """
    if not ckpt_paths:
        return None

    all_run_preds = []
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"Ensemble checkpoint {i+1}/{len(ckpt_paths)}: {ckpt_path.name}")
        # Reset test buffers
        model._test_preds.clear()
        model._test_labels.clear()
        model._test_pert_ids.clear()
        model._test_symbols.clear()

        # Run test with this checkpoint
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path=str(ckpt_path))

        # Load the predictions saved by this test run
        pred_path = out_dir / "test_predictions.tsv"
        if pred_path.exists():
            df = pd.read_csv(pred_path, sep="\t")
            run_preds = np.array([json.loads(p) for p in df["prediction"]])  # [N, 3, 6640]
            all_run_preds.append((df["idx"].tolist(), df["input"].tolist(), run_preds))

    if not all_run_preds:
        return None

    # Average predictions across checkpoints
    # Align by pert_id from first run
    first_ids = all_run_preds[0][0]
    first_syms = all_run_preds[0][1]
    avg_preds = np.mean([rp for _, _, rp in all_run_preds], axis=0)  # [N, 3, 6640]

    return first_ids, first_syms, avg_preds


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-3-1-3: STRING-Only PreLN MLP with Muon + CosineWarmRestarts + Manifold Mixup"
    )
    p.add_argument("--micro-batch-size",    type=int,   default=32)
    p.add_argument("--global-batch-size",   type=int,   default=256)
    p.add_argument("--max-epochs",          type=int,   default=500,
                   help="500 epochs to allow 3+ warm restart cycles (T_0=80, T_mult=2)")
    p.add_argument("--lr",                  type=float, default=3e-4,
                   help="AdamW lr for head/norm/bias params")
    p.add_argument("--muon-lr",             type=float, default=0.01,
                   help="Muon lr for hidden 2D weight matrices (proven optimal in node1-3-2)")
    p.add_argument("--weight-decay",        type=float, default=8e-4,
                   help="AdamW weight decay (proven optimal: 8e-4)")
    p.add_argument("--muon-weight-decay",   type=float, default=8e-4,
                   help="Muon weight decay (same as AdamW for balanced regularization)")
    p.add_argument("--hidden-dim",          type=int,   default=384,
                   help="Hidden dimension (384 = proven capacity sweet spot)")
    p.add_argument("--n-blocks",            type=int,   default=3,
                   help="Number of PreLN residual blocks (3 = proven optimal)")
    p.add_argument("--dropout",             type=float, default=0.30,
                   help="Trunk dropout (0.30 = proven for node1-3-2 lineage)")
    p.add_argument("--head-dropout",        type=float, default=0.15,
                   help="Flat output head dropout (0.15 = proven optimal, 0.20 over-regularizes)")
    p.add_argument("--t0",                  type=int,   default=80,
                   help="CosineWarmRestarts T_0 — first cycle length in epochs")
    p.add_argument("--t-mult",              type=int,   default=2,
                   help="CosineWarmRestarts T_mult — cycle length multiplier")
    p.add_argument("--mixup-alpha",         type=float, default=0.2,
                   help="Manifold Mixup Beta distribution alpha parameter")
    p.add_argument("--mixup-prob",          type=float, default=0.5,
                   help="Manifold Mixup application probability per batch")
    p.add_argument("--n-ensemble-ckpts",    type=int,   default=5,
                   help="Number of top-k checkpoints for test-time ensemble")
    p.add_argument("--lr-patience",         type=int,   default=8,
                   help="LR patience (kept for compatibility, CosineWR is the primary scheduler)")
    p.add_argument("--lr-factor",           type=float, default=0.5)
    p.add_argument("--early-stop-patience", type=int,   default=120,
                   help="Early stopping patience — 120 allows >3 warm restart cycles")
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
        muon_weight_decay=args.muon_weight_decay,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        t0=args.t0,
        t_mult=args.t_mult,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        n_ensemble_ckpts=args.n_ensemble_ckpts,
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

    # Save top-k checkpoints for ensemble (save_top_k=5)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-val_f1={val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=args.n_ensemble_ckpts,  # Save top-k for ensemble
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/f1",
        mode="max",
        patience=args.early_stop_patience,  # 120 epochs patience
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
            timeout=timedelta(seconds=600),
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
        gradient_clip_val=1.0,  # Gradient clipping — stabilizes Muon's aggressive updates
        gradient_clip_algorithm="norm",
    )

    # --- Fit ---
    trainer.fit(model, datamodule=datamodule)

    # --- Test with checkpoint ensemble ---
    if args.fast_dev_run or args.debug_max_step is not None:
        # Simple test for debug modes
        trainer.test(model, datamodule=datamodule)
    else:
        ckpt_dir = Path(output_dir) / "checkpoints"
        top_k_ckpts = _load_top_k_checkpoints(ckpt_dir, k=args.n_ensemble_ckpts)

        if len(top_k_ckpts) >= 2:
            print(f"\nRunning top-{len(top_k_ckpts)} checkpoint ensemble...")

            # Collect predictions from each checkpoint
            all_run_preds_np = []
            all_pert_ids_first = None
            all_syms_first = None

            for i, ckpt_path in enumerate(top_k_ckpts):
                print(f"  Loading checkpoint {i+1}/{len(top_k_ckpts)}: {ckpt_path.name}")
                # Reset test buffers before each run
                model._test_preds.clear()
                model._test_labels.clear()
                model._test_pert_ids.clear()
                model._test_symbols.clear()
                # Test with this checkpoint
                trainer.test(model, datamodule=datamodule, ckpt_path=str(ckpt_path))

                # Load predictions saved by this run
                pred_path_i = output_dir / "test_predictions.tsv"
                if pred_path_i.exists():
                    df_i = pd.read_csv(pred_path_i, sep="\t")
                    preds_i = np.array([json.loads(p) for p in df_i["prediction"]])
                    all_run_preds_np.append(preds_i)
                    if all_pert_ids_first is None:
                        all_pert_ids_first = df_i["idx"].tolist()
                        all_syms_first = df_i["input"].tolist()

            if len(all_run_preds_np) >= 1:
                # Average logits across all checkpoints (ensemble in logit space)
                avg_preds = np.mean(all_run_preds_np, axis=0)  # [N, 3, 6640]

                if trainer.is_global_zero:
                    # Save ensemble predictions
                    ensemble_pred_path = output_dir / "test_predictions.tsv"
                    _save_test_predictions(
                        pert_ids=all_pert_ids_first,
                        symbols=all_syms_first,
                        preds=avg_preds,
                        out_path=ensemble_pred_path,
                    )
                    print(f"Saved {len(top_k_ckpts)}-checkpoint ensemble to {ensemble_pred_path}")
        else:
            # Fall back to single best checkpoint
            print("Fewer than 2 checkpoints available, using single best checkpoint.")
            trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # --- Save test score using ground truth and predictions (rank 0 only) ---
    if trainer.is_global_zero:
        pred_path = output_dir / "test_predictions.tsv"
        if pred_path.exists():
            # Load ground truth test labels
            test_gt_df = pd.read_csv(str(data_dir / "test.tsv"), sep="\t")
            test_pred_df = pd.read_csv(pred_path, sep="\t")
            # Align predictions to ground truth by pert_id
            gt_map = dict(zip(test_gt_df["pert_id"], test_gt_df["label"]))
            pred_map = dict(zip(test_pred_df["idx"], test_pred_df["prediction"]))
            common_ids = [pid for pid in gt_map if pid in pred_map]
            y_true = np.array([json.loads(gt_map[pid]) for pid in common_ids]) + 1  # shift
            y_pred = np.array([json.loads(pred_map[pid]) for pid in common_ids])
            test_f1 = _compute_per_gene_f1(y_pred, y_true)
            score_path = Path(__file__).parent / "test_score.txt"
            score_path.write_text(json.dumps({"test_f1": test_f1}, indent=2))
            print(f"Test F1 = {test_f1:.4f} → {score_path}")


if __name__ == "__main__":
    main()
