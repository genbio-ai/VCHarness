"""Node 1-1-2 – Dual Backbone (STRING_GNN + ESM2-650M) + Asymmetric Focal Loss.

Strategy: Combine frozen STRING_GNN (PPI topology, 256-dim) with frozen pre-computed
ESM2-650M protein sequence embeddings (3840-dim) into a 4096-dim fused perturbation
representation. This dual-source encoding directly addresses the core domain mismatch
bottleneck identified in node1-1: STRING_GNN encodes static PPI topology, whereas the
task requires predicting dynamic transcriptional response. The ESM2 protein sequence
embeddings capture intrinsic protein properties (evolutionary conservation, domain
structure, binding motifs) that are orthogonal to PPI topology and complementary for
predicting regulatory effects.

Key improvements over node1-1 (parent):
1. Dual backbone: frozen STRING_GNN (256-dim) + frozen ESM2-650M (3840-dim) → 4096-dim
2. Asymmetric focal loss: γ_neutral=3.0, γ_down=1.0, γ_up=1.0 (distinct from uniform)
3. Pre-computed frozen embeddings for both backbones (fast training)
4. Gene bias term [3, G] to capture per-gene DEG priors
5. Scaled bilinear output (/ sqrt(bilinear_dim)) for stable softmax
6. Linear warmup (20 epochs) + CosineAnnealingWarmRestarts (T_0=100, T_mult=2)
7. Extended training (300 epochs, patience=20)
8. Fallback learnable embedding for pert_ids not in STRING/ESM2

Key differences from sibling (node1-1-1):
- Sibling: single STRING_GNN backbone, uniform focal loss γ=2.0, deep residual MLP, bilinear_dim=512
- This node: DUAL backbone (STRING_GNN + ESM2-650M), asymmetric focal loss, lean projection MLP, bilinear_dim=256
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES = 6640
N_CLASSES = 3

# Remapped class frequencies (after -1→0, 0→1, 1→2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

# Asymmetric focal gamma: aggressively suppress easy neutral predictions,
# moderate focus on hard DEG cases (down/up)
# Neutral class (majority 92.5%): γ=3.0 → (1-0.95)^3 = 0.000125 suppression
# DEG classes (minority 3-4%): γ=1.0 → standard focal modulation
CLASS_GAMMA = [1.0, 3.0, 1.0]  # [down, neutral, up]

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

# ESM2-650M embedding dimension
ESM2_DIM = 3840
STRING_DIM = 256
FUSED_DIM = STRING_DIM + ESM2_DIM  # 4096


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights; neutral class stays ≈ 1."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def load_string_gnn_mapping() -> Dict[str, int]:
    """Load STRING_GNN node_names.json and return Ensembl-ID → node-index mapping."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    return {name: idx for idx, name in enumerate(node_names)}


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0,1,2}
    Returns:
        Scalar float: mean over all G genes.
    """
    y_hat = preds.argmax(dim=1)           # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)           # [N, G]
        is_pred = (y_hat == c)             # [N, G]
        present = is_true.any(dim=0)       # [G]

        tp = (is_pred & is_true).float().sum(0)
        fp = (is_pred & ~is_true).float().sum(0)
        fn = (~is_pred & is_true).float().sum(0)

        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(
            prec + rec > 0,
            2 * prec * rec / (prec + rec + 1e-8),
            torch.zeros_like(prec),
        )
        f1_per_gene += f1_c * present.float()
        n_present   += present.float()

    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


def asymmetric_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    class_gamma: torch.Tensor,
) -> torch.Tensor:
    """Asymmetric focal loss with per-class gamma values.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma_t * log(p_t)

    Args:
        logits:        [N, C] – unnormalized logits
        targets:       [N]    – integer class labels
        class_weights: [C]    – alpha (class-frequency-based weights)
        class_gamma:   [C]    – gamma for each class (asymmetric)
    Returns:
        Scalar mean loss.
    """
    log_p = F.log_softmax(logits, dim=-1)          # [N, C]
    p = log_p.exp()                                  # [N, C]

    # Probability and log-probability of the true class
    p_t     = p.gather(1, targets.unsqueeze(1)).squeeze(1)       # [N]
    log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)   # [N]

    # Per-sample gamma (from true class)
    gamma_t = class_gamma[targets]        # [N]
    # Per-sample alpha (from true class)
    alpha_t = class_weights[targets]      # [N]

    # Focal modulation
    focal_weight = (1.0 - p_t) ** gamma_t   # [N]

    loss = -alpha_t * focal_weight * log_p_t  # [N]
    return loss.mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    """K562 DEG prediction dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        string_map: Dict[str, int],
    ) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()

        # STRING_GNN node index for each sample (-1 means not in STRING)
        self.string_node_indices = torch.tensor(
            [string_map.get(p, -1) for p in self.pert_ids], dtype=torch.long
        )

        has_label = "label" in df.columns and df["label"].notna().all()
        if has_label:
            self.labels = [
                torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
                for row in df["label"].tolist()
            ]
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx":        idx,
            "pert_id":           self.pert_ids[idx],
            "symbol":            self.symbols[idx],
            "string_node_idx":   self.string_node_indices[idx],  # long scalar
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]  # [G] in {0,1,2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx":      torch.tensor([b["sample_idx"]       for b in batch], dtype=torch.long),
        "pert_id":         [b["pert_id"]  for b in batch],
        "symbol":          [b["symbol"]   for b in batch],
        "string_node_idx": torch.stack([b["string_node_idx"]   for b in batch]),
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.string_map: Optional[Dict[str, int]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.string_map is None:
            self.string_map = load_string_gnn_mapping()

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")

        self.train_ds = DEGDataset(train_df, self.string_map)
        self.val_ds   = DEGDataset(val_df,   self.string_map)
        self.test_ds  = DEGDataset(test_df,  self.string_map)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        sampler = SequentialSampler(self.test_ds)
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True,
            sampler=sampler,
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DualBackboneBilinearModel(pl.LightningModule):
    """Dual-backbone DEG prediction model.

    Architecture:
        1. Pre-computed frozen STRING_GNN embeddings [18870, 256]
        2. Pre-computed frozen ESM2-650M protein embeddings [18870, 3840]
        3. Lookup: string_emb[B, 256] + esm2_emb[B, 3840] → concat [B, 4096]
        4. Projection MLP: 4096 → hidden_dim → bilinear_dim
        5. Bilinear output: logits[b,c,g] = pert_emb[b] · gene_class_emb[c,g] / sqrt(D)
        6. Gene bias: + gene_bias[c, g]
        7. Loss: asymmetric focal loss (γ_neutral=3.0, γ_down/up=1.0)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        bilinear_dim: int = 256,
        dropout: float = 0.25,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_epochs: int = 20,
        T_0: int = 100,
        T_mult: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Model initialized in setup()

    def setup(self, stage: Optional[str] = None) -> None:
        # Guard against repeated setup calls (Lightning may call setup multiple times)
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        hp = self.hparams

        # ── Load STRING_GNN and pre-compute frozen node embeddings ──────────
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone.eval()

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph["edge_index"].long()
        edge_weight = graph["edge_weight"].float()

        # One-time STRING_GNN forward pass to get node embeddings (no grad)
        with torch.no_grad():
            gnn_out = backbone(edge_index=edge_index, edge_weight=edge_weight)
            string_emb_table = gnn_out.last_hidden_state.float()  # [18870, 256]

        # Register as buffer so it moves to GPU automatically
        self.register_buffer("string_emb_table", string_emb_table)

        # ── Load pre-computed ESM2-650M embeddings ───────────────────────────
        # Already pre-computed for all 18870 STRING_GNN nodes
        esm2_emb_table = torch.load(
            STRING_GNN_DIR / "esm2_embeddings_t33_650M.pt", map_location="cpu"
        ).float()  # [18870, 3840]
        self.register_buffer("esm2_emb_table", esm2_emb_table)

        # ── Fallback learnable embeddings for unknowns ───────────────────────
        # Used for pert_ids not found in the STRING_GNN node_names (~6.4% of train)
        self.fallback_string = nn.Embedding(1, STRING_DIM)
        self.fallback_esm2   = nn.Embedding(1, ESM2_DIM)
        nn.init.normal_(self.fallback_string.weight, std=0.02)
        nn.init.normal_(self.fallback_esm2.weight,   std=0.02)

        # ── Projection MLP: FUSED_DIM (4096) → hidden_dim → bilinear_dim ────
        self.proj = nn.Sequential(
            nn.LayerNorm(FUSED_DIM),
            nn.Linear(FUSED_DIM, hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
            nn.LayerNorm(hp.hidden_dim),
            nn.Linear(hp.hidden_dim, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # ── Bilinear gene-class embedding matrix: [C, G, bilinear_dim] ───────
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # ── Learnable per-gene-class bias: captures DEG priors ───────────────
        # E.g., housekeeping genes (always neutral) vs. frequently DEG genes
        self.gene_bias = nn.Parameter(torch.zeros(N_CLASSES, N_GENES))

        # ── Loss components ───────────────────────────────────────────────────
        self.register_buffer("class_weights", get_class_weights())
        self.register_buffer(
            "class_gamma",
            torch.tensor(CLASS_GAMMA, dtype=torch.float32)
        )

        # ── Cast trainable parameters to float32 ─────────────────────────────
        for _, param in self.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Accumulators for val/test ─────────────────────────────────────────
        self._val_preds: List[torch.Tensor] = []
        self._val_tgts:  List[torch.Tensor] = []
        self._val_idx:   List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    # ── Embedding lookup ──────────────────────────────────────────────────────
    def _get_fused_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Lookup pre-computed STRING_GNN + ESM2 embeddings and fuse.

        Args:
            string_node_idx: [B] long, -1 for unknowns
        Returns:
            [B, FUSED_DIM=4096] float
        """
        B = string_node_idx.shape[0]
        known_mask   = string_node_idx >= 0
        unknown_mask = ~known_mask
        zero_idx     = torch.zeros(unknown_mask.sum(), dtype=torch.long,
                                   device=string_node_idx.device)

        # STRING_GNN embeddings
        string_emb = torch.zeros(B, STRING_DIM,
                                  dtype=self.string_emb_table.dtype,
                                  device=self.string_emb_table.device)
        if known_mask.any():
            string_emb[known_mask] = self.string_emb_table[string_node_idx[known_mask]]
        if unknown_mask.any():
            string_emb[unknown_mask] = self.fallback_string(zero_idx).to(
                self.string_emb_table.dtype)

        # ESM2-650M embeddings
        esm2_emb = torch.zeros(B, ESM2_DIM,
                                dtype=self.esm2_emb_table.dtype,
                                device=self.esm2_emb_table.device)
        if known_mask.any():
            esm2_emb[known_mask] = self.esm2_emb_table[string_node_idx[known_mask]]
        if unknown_mask.any():
            esm2_emb[unknown_mask] = self.fallback_esm2(zero_idx).to(
                self.esm2_emb_table.dtype)

        # Concatenate: [B, 4096]
        fused = torch.cat([string_emb.float(), esm2_emb.float()], dim=-1)
        return fused

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        fused    = self._get_fused_embeddings(string_node_idx)  # [B, 4096]
        h        = self.proj(fused)                              # [B, bilinear_dim]

        # Scaled bilinear: divide by sqrt(D) to prevent logit magnitude growth
        scale    = (self.hparams.bilinear_dim ** 0.5)
        logits   = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb) / scale

        # Add per-gene-class bias (captures DEG priors independent of perturbation)
        logits   = logits + self.gene_bias.unsqueeze(0)          # [B, 3, G]
        return logits

    # ── Loss ─────────────────────────────────────────────────────────────────
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return asymmetric_focal_loss(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, 3]
            targets.reshape(-1),                       # [B*G]
            self.class_weights,
            self.class_gamma,
        )

    # ── Training / validation / test steps ───────────────────────────────────
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["string_node_idx"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True,
                 on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits, dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, dim=0)   # [N_local, 3, G]
        local_tgts  = torch.cat(self._val_tgts,  dim=0)   # [N_local, G]
        local_idx   = torch.cat(self._val_idx,   dim=0)   # [N_local]
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        # Gather across all ranks
        all_preds = self.all_gather(local_preds)   # [W, N_local, 3, G]
        all_tgts  = self.all_gather(local_tgts)    # [W, N_local, G]
        all_idx   = self.all_gather(local_idx)     # [W, N_local]

        preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
        tgts_flat  = all_tgts.view(-1, N_GENES)
        idx_flat   = all_idx.view(-1)

        # De-duplicate (DDP padding may introduce repeated samples)
        order  = torch.argsort(idx_flat)
        s_idx  = idx_flat[order]
        s_pred = preds_flat[order]
        s_tgt  = tgts_flat[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device),
                            s_idx[1:] != s_idx[:-1]])
        preds_dedup = s_pred[mask]
        tgts_dedup  = s_tgt[mask]

        f1 = compute_per_gene_f1(preds_dedup, tgts_dedup)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["string_node_idx"])
        probs  = torch.softmax(logits, dim=1).detach()
        self._test_preds.append(probs)
        self._test_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, dim=0)  # [N_local, 3, G]
        local_idx   = torch.cat(self._test_idx,   dim=0)  # [N_local]
        all_preds   = self.all_gather(local_preds)         # [W, N_local, 3, G]
        all_idx     = self.all_gather(local_idx)           # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            # De-duplicate
            order  = torch.argsort(idx_flat)
            s_idx  = idx_flat[order]
            s_pred = preds_flat[order]
            mask   = torch.cat([torch.ones(1, dtype=torch.bool, device=s_idx.device),
                                s_idx[1:] != s_idx[:-1]])
            preds_dedup = s_pred[mask]
            unique_sid  = s_idx[mask].tolist()

            # Reload test.tsv on rank 0
            test_df = pd.read_csv(TEST_TSV, sep="\t")
            idx_to_meta = {i: (test_df.iloc[i]["pert_id"], test_df.iloc[i]["symbol"])
                           for i in range(len(test_df))}

            rows = []
            for sid in unique_sid:
                pid, sym = idx_to_meta[int(sid)]
                dedup_pos = (s_idx == sid).nonzero(as_tuple=True)[0][0].item()
                pred_list = preds_dedup[dedup_pos].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred_list)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-1-2] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_idx.clear()

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                key = prefix + name
                if key in full:
                    trainable[key] = full[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full:
                trainable[key] = full[key]
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {train}/{total} params ({100*train/total:.1f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        hp = self.hparams

        opt = torch.optim.AdamW(
            self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )

        # Linear warmup for first warmup_epochs epochs
        # then CosineAnnealingWarmRestarts (T_0=100, T_mult=2)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=hp.T_0,
            T_mult=hp.T_mult,
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[hp.warmup_epochs],
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node1-1-2 – Dual Backbone STRING_GNN + ESM2-650M + Asymmetric Focal Loss"
    )
    parser.add_argument("--micro-batch-size",  type=int,   default=32)
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument("--max-epochs",        type=int,   default=300)
    parser.add_argument("--lr",                type=float, default=3e-4)
    parser.add_argument("--weight-decay",      type=float, default=1e-2)
    parser.add_argument("--hidden-dim",        type=int,   default=512)
    parser.add_argument("--bilinear-dim",      type=int,   default=256)
    parser.add_argument("--dropout",           type=float, default=0.25)
    parser.add_argument("--warmup-epochs",     type=int,   default=20)
    parser.add_argument("--t-0",               type=int,   default=100)
    parser.add_argument("--t-mult",            type=int,   default=2)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--debug-max-step",    type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",      action="store_true",
                        dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Limit logic
    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        max_steps = -1

    # Always process ALL test batches regardless of debug mode, since Lightning's
    # DistributedSampler shards the test set across ranks and deduplication in
    # on_test_epoch_end requires all samples to be collected from all ranks.
    lim_test = 1.0

    # val_check_interval: validate once per epoch in normal mode;
    # in debug mode (limit_train batches) also once per epoch.
    val_check_interval = 1.0

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    # Model
    model = DualBackboneBilinearModel(
        hidden_dim    = args.hidden_dim,
        bilinear_dim  = args.bilinear_dim,
        dropout       = args.dropout,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        warmup_epochs = args.warmup_epochs,
        T_0           = args.t_0,
        T_mult        = args.t_mult,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
    )
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=20, min_delta=1e-4)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy: find_unused_parameters=True is required because some parameters
    # (e.g., fallback embeddings for unknown pert_ids) may not appear in every
    # forward pass in DDP gradient bucketing.
    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator             = "gpu",
        devices                 = n_gpus,
        num_nodes               = 1,
        strategy                = strategy,
        precision               = "bf16-mixed",
        max_epochs              = args.max_epochs,
        max_steps               = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches     = lim_train,
        limit_val_batches       = lim_val,
        limit_test_batches      = lim_test,
        val_check_interval      = val_check_interval,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 10,
        deterministic           = True,
        default_root_dir        = str(output_dir),
        fast_dev_run            = fast_dev_run,
        gradient_clip_val       = 1.0,
    )

    trainer.fit(model, datamodule=dm)

    ckpt_path = "best" if (args.debug_max_step is None and not fast_dev_run) else None
    test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)

    # Save test score
    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node1-1-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
