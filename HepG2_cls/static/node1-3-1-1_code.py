"""Node 1-3-1-1: STRING-Only + FLAT Output Head + Reduced hidden_dim (512→384) + Head Dropout

Changes from parent node1-3-1 (F1=0.430):
  1. REVERT factorized head → flat Linear(384→19920) (Priority 1 from feedback)
     The 256-dim bottleneck failed in all 3 independent experiments. Flat head is
     unambiguously superior. Flat head with 384-dim: 384×19920+19920 = 7.68M params.
  2. REDUCE hidden_dim: 512→384 (Priority 2 from feedback)
     Directly addresses the dominant overfitting bottleneck. Reduces MLP trunk params
     from 3.15M (3×(512×1024+1024×512)) to 1.77M (3×(384×768+768×384)) — a 44% reduction.
     Total trainable params: ~9.5M → ~9.5M (flat head offset: 384×19920 vs 512×256+256×19920)
     Actually saves substantial capacity in the trunk without sacrificing head expressiveness.
  3. ADD mild output head dropout=0.10 (Priority 3 from feedback)
     Applied after LayerNorm in the flat head (LayerNorm→Dropout→Linear→gene_bias).
     Mild regularization on the flat 7.68M-param output mapping without a capacity bottleneck.
  4. KEEP: STRING-only 256-dim, 3 blocks, weighted CE + label_smoothing=0.05,
     RLROP patience=8, per-gene bias, AdamW lr=3e-4 wd=5e-4, global_batch_size=256.

Node ancestry and test F1:
  node1 (root)       → 0.405  Learned embeddings + 8-block MLP
  node1-1            → 0.472  STRING 256-dim + 5-block + focal loss
  node1-1-1          → 0.474  STRING 256-dim + 3-block + flat head + weighted CE (TREE BEST)
  node1-3 (parent²) → 0.463  STRING+ESM2 additive 256-dim + flat head
  node1-3-1 (parent) → 0.430  STRING-only + FACTORIZED head (512→256→19920) ← HARMFUL
  node1-3-1-1 (this) → TBD   STRING-only + FLAT head + hidden_dim=384 + head_dropout=0.10
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
    """Pre-LayerNorm residual MLP block (dim → dim*2 → dim)."""

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


class StringOnlyPerturbMLP(nn.Module):
    """STRING-only MLP with flat output head and reduced hidden_dim for HepG2 perturbation.

    Architecture (input → output):
      ① STRING_GNN embedding lookup [B, 256] frozen buffer
         (fallback learnable 256-dim for ~6% genes absent from STRING)
      ② Input projection: Linear(256→hidden_dim) + LayerNorm + GELU
      ③ n_blocks × ResidualBlock(hidden_dim)
      ④ Flat output head: LayerNorm(hidden_dim) → Dropout(head_dropout) → Linear(hidden_dim→N_GENES*N_CLASSES)
         + per-gene additive bias [N_GENES*N_CLASSES]
      ⑤ Reshape → [B, 3, 6640]

    Key changes vs node1-3-1 parent:
      - FLAT head (hidden_dim→19920) vs factorized (512→256→19920) in parent
        * Factorized head confirmed harmful in 3 independent experiments
        * Flat head allows full hidden_dim capacity for all 19,920 output logits
      - hidden_dim reduced 512→384:
        * Trunk params: 3×(512×1024+1024×512)=3.15M → 3×(384×768+768×384)=1.77M (44% reduction)
        * Directly addresses overfitting without destroying head expressiveness
      - head_dropout=0.10 applied on flat head (LayerNorm→Dropout→Linear):
        * Mild regularization on the 7.68M-param flat output mapping
        * No capacity bottleneck (dropout does not reduce dimensionality)
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.35,
        head_dropout: float = 0.10,
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

        # Flat output head: LayerNorm → Dropout → Linear(hidden_dim→N_GENES*N_CLASSES)
        # NOTE: No bottleneck. Mild dropout adds regularization without capacity loss.
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, N_GENES * N_CLASSES),
        )

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

        # ② Input projection → residual MLP → flat head
        x = self.input_proj(emb)            # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)
        # ④ Flat output head (LayerNorm → Dropout → Linear) + per-gene bias
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
        dropout: float = 0.35,
        head_dropout: float = 0.10,
        lr: float = 3e-4,
        weight_decay: float = 5e-4,
        label_smoothing: float = 0.05,
        lr_patience: int = 8,
        lr_factor: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor

        self.model: Optional[StringOnlyPerturbMLP] = None

        # Accumulation buffers for epoch-level metrics
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # Class weights: inversely proportional to class frequencies
        # After {-1,0,1}→{0,1,2}: class0=neutral(92.82%), class1=down(4.77%), class2=up(2.41%)
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
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

        # ---- Build model (STRING-only, flat head, hidden_dim=384) ----
        self.model = StringOnlyPerturbMLP(
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
            f"Node1-3-1-1 StringOnlyPerturbMLP | hidden={self.hidden_dim} | "
            f"blocks={self.n_blocks} | dropout={self.dropout} | head_dropout={self.head_dropout} | "
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
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
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
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

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
            elif all_preds.dim() == 3:
                all_preds = all_preds.squeeze(0)

        # Gather test labels if available
        all_labels_np: Optional[np.ndarray] = None
        if self._test_labels:
            labels_local = torch.cat(self._test_labels, dim=0)
            all_labels = self.all_gather(labels_local)
            if ws > 1:
                all_labels = all_labels.view(-1, N_GENES)
            else:
                if all_labels.dim() == 3:
                    all_labels = all_labels.squeeze(0)  # [1, N, 6640] → [N, 6640]
                elif all_labels.dim() == 2:
                    all_labels = all_labels.squeeze(0)  # [N, 6640]
            all_labels_np = all_labels.cpu().numpy()

        # Rank-aware file saving: each rank writes its own file, rank 0 merges.
        out_dir = Path(__file__).parent / "run"
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        ws = self.trainer.world_size

        # Step 1: each rank saves its portion to a temp file
        rank_preds_path = out_dir / f"test_predictions_rank{local_rank}.tsv"
        rank_meta_path = out_dir / f"test_predictions_rank{local_rank}_meta.tsv"

        preds_np = all_preds.float().cpu().numpy()
        _save_test_predictions(
            pert_ids=self._test_pert_ids,
            symbols=self._test_symbols,
            preds=preds_np,
            out_path=rank_preds_path,
        )
        # Save metadata separately (labels if available)
        if all_labels_np is not None:
            label_rows = []
            for i, (pid, sym) in enumerate(zip(self._test_pert_ids, self._test_symbols)):
                label_rows.append({
                    "idx": pid,
                    "input": sym,
                    "label": json.dumps(all_labels_np[i].tolist()),
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

            # Compute and log test F1 if ground truth is available
            if all_labels_list:
                all_labels_arr = np.stack(all_labels_list)
                test_f1 = _compute_per_gene_f1(final_preds, all_labels_arr)
                self.log("test/f1", test_f1, prog_bar=True, sync_dist=False)

        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.lr_factor,
            patience=self.lr_patience,  # patience=8 prevents premature LR halving
            min_lr=1e-6,
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


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-3-1-1: STRING-Only + Flat Head + hidden_dim=384 + head_dropout=0.10"
    )
    p.add_argument("--micro-batch-size",    type=int,   default=32)
    p.add_argument("--global-batch-size",   type=int,   default=256)
    p.add_argument("--max-epochs",          type=int,   default=200)
    p.add_argument("--lr",                  type=float, default=3e-4)
    p.add_argument("--weight-decay",        type=float, default=5e-4)
    p.add_argument("--hidden-dim",          type=int,   default=384,
                   help="Hidden dimension of residual blocks (reduced from 512 to 384)")
    p.add_argument("--n-blocks",            type=int,   default=3)
    p.add_argument("--dropout",             type=float, default=0.35)
    p.add_argument("--head-dropout",        type=float, default=0.10,
                   help="Dropout rate applied in flat output head (before Linear layer)")
    p.add_argument("--label-smoothing",     type=float, default=0.05)
    p.add_argument("--lr-patience",         type=int,   default=8,
                   help="ReduceLROnPlateau patience (8 prevents premature LR halving)")
    p.add_argument("--lr-factor",           type=float, default=0.5)
    p.add_argument("--early-stop-patience", type=int,   default=25)
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
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
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
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
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
            find_unused_parameters=True,  # Enable to prevent false-positive unused-param errors with DDP
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
    )

    # --- Fit ---
    trainer.fit(model, datamodule=datamodule)

    # --- Test ---
    if args.fast_dev_run or args.debug_max_step is not None:
        trainer.test(model, datamodule=datamodule)
    else:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
