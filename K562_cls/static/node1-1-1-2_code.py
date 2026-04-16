"""Node 1-1-1-2 – Frozen STRING_GNN + Mixup Augmentation + Per-Gene Log Prior.

Key innovations over sibling node1-1-1-1 (test F1=0.4746):

1. Mixup augmentation on perturbation embeddings (training only).
   - Creates virtual training samples from convex combinations of real embeddings.
   - Critical for the small dataset (1,388 samples): effectively infinite virtual samples.
   - Applied at the embedding level: mixed_emb = lam * emb_a + (1-lam) * emb_b
   - Loss = lam * CE(logits, labels_a) + (1-lam) * CE(logits, labels_b)
   - Disabled at validation/test (clean bilinear head behavior).

2. Fixed per-gene class log-prior (non-learnable) + learnable temperature.
   - Pre-computed from training data: prior_freq[c, g] = P(gene g == class c | train)
   - Stored as a fixed buffer (no gradients): gene_log_prior[3, 6640]
   - Added to bilinear logits: logits += softplus(T) * gene_log_prior (T init=0)
   - Addresses gene-level class imbalance on top of global class weights.
   - Unlike node1-1-1's gene_bias (19,920 learnable params -> severe overfitting),
     this uses ZERO learnable per-gene parameters; only 1 scalar temperature.

3. Refined hyperparameters from sibling's feedback:
   - dropout: 0.35 -> 0.4 (stronger, explicitly recommended by sibling's feedback)
   - lr: 3e-4 -> 2e-4 (finer convergence after sibling peaked at epoch 49)
   - warmup_epochs: 20 -> 25 (proportionally longer warmup with lower LR)
   - patience: 7 -> 5 (tighter, prevents mild post-peak oscillation observed in sibling)
   - mixup_alpha: 0.4 (new, creates soft convex-hull augmentation)

Retained from sibling (node1-1-1-1, all proven effective):
- Frozen STRING_GNN backbone with pre-computed embeddings [18870, 256]
- Simple 2-layer MLP head (bilinear_dim=256, no ResBlocks, no gene_bias)
- Weighted cross-entropy + label_smoothing=0.05
- weight_decay=2e-2
- LinearWarmup(25 epochs) + CosineAnnealingLR(T_max=100)
- max_epochs=150
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

# Remapped class frequencies (after -1->0, 0->1, 1->2):
# class 0 (down): 4.29%, class 1 (neutral): 92.51%, class 2 (up): 3.20%
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

STRING_GNN_DIR = Path("/home/Models/STRING_GNN")
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"

STRING_DIM = 256  # STRING_GNN hidden dimension


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency weights for cross-entropy."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    mean_w = sum(w) / len(w)
    return torch.tensor([x / mean_w for x in w], dtype=torch.float32)


def load_string_gnn_mapping() -> Dict[str, int]:
    """Load STRING_GNN node_names.json -> Ensembl-ID to node-index mapping."""
    node_names: List[str] = json.loads((STRING_GNN_DIR / "node_names.json").read_text())
    return {name: idx for idx, name in enumerate(node_names)}


def compute_gene_log_prior(train_tsv: Path, n_genes: int = N_GENES, n_classes: int = N_CLASSES,
                           smoothing: float = 1e-4) -> np.ndarray:
    """Pre-compute per-gene class log-prior from training data.

    Returns:
        log_prior: float32 array of shape [3, N_GENES]
                   log_prior[c, g] = log(freq(class c for gene g in training) + smoothing)
    """
    df = pd.read_csv(train_tsv, sep="\t")
    # Parse labels: JSON list of {-1, 0, 1}, remap to {0, 1, 2}
    labels_list = [np.array(json.loads(r), dtype=np.int8) + 1 for r in df["label"].tolist()]
    labels_arr = np.stack(labels_list, axis=0)  # [N_train, N_GENES], values in {0,1,2}

    gene_class_freq = np.zeros((n_classes, n_genes), dtype=np.float32)
    n_samples = labels_arr.shape[0]
    for c in range(n_classes):
        gene_class_freq[c] = (labels_arr == c).sum(axis=0) / n_samples  # [N_GENES]

    # Log prior with Laplace smoothing
    log_prior = np.log(gene_class_freq + smoothing)  # [3, N_GENES]
    return log_prior


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float  – softmax probabilities
        targets: [N, G]    long   – class labels in {0, 1, 2}
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
            "string_node_idx":   self.string_node_indices[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]  # [G] in {0, 1, 2}
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_idx":      torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
        "pert_id":         [b["pert_id"]  for b in batch],
        "symbol":          [b["symbol"]   for b in batch],
        "string_node_idx": torch.stack([b["string_node_idx"] for b in batch]),
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
        self.batch_size  = batch_size
        self.num_workers = num_workers
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
class FrozenStringGNNMixupModel(pl.LightningModule):
    """Frozen STRING_GNN with Mixup augmentation and per-gene class log-prior.

    Architecture:
        1. STRING_GNN run ONCE at setup() to pre-compute node embeddings [18870, 256]
           -> stored as a fixed buffer; no GNN gradients during training.
        2. Lookup pre-computed embedding by pert_id string_node_idx -> [B, STRING_DIM]
           (learnable fallback for ~6.4% unknown pert_ids)
        3. [Training only] Mixup augmentation:
              mixed_emb = lam * emb_a + (1-lam) * emb_b   (lam ~ Beta(alpha, alpha))
        4. Simple 2-layer MLP head (matches sibling's proven architecture):
              LayerNorm(256) -> Linear(256->256) -> GELU -> Dropout(0.4)
              -> LayerNorm(256) -> Linear(256->256) -> GELU -> Dropout(0.4)
        5. Bilinear output:
              logits[b,c,g] = (h[b] . gene_class_emb[c,g]) / sqrt(D)  [B, 3, G]
        6. Per-gene log-prior adjustment:
              logits += softplus(prior_temperature) * gene_log_prior
              where gene_log_prior[3, G] is a fixed buffer pre-computed from train data
        7. Loss: Weighted CE + label_smoothing=0.05 (with Mixup split)
    """

    def __init__(
        self,
        bilinear_dim: int = 256,
        dropout: float = 0.4,
        lr: float = 2e-4,
        weight_decay: float = 2e-2,
        warmup_epochs: int = 25,
        T_max: int = 100,
        label_smoothing: float = 0.05,
        mixup_alpha: float = 0.4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Model layers initialized in setup()

    def setup(self, stage: Optional[str] = None) -> None:
        # Guard against repeated setup calls
        if getattr(self, "_setup_done", False):
            return
        self._setup_done = True

        hp = self.hparams

        # ----------------------------------------------------------------
        # 1. Pre-compute STRING_GNN node embeddings (backbone frozen)
        # All ranks load independently (local model, no download needed).
        # ----------------------------------------------------------------
        backbone = AutoModel.from_pretrained(str(STRING_GNN_DIR), trust_remote_code=True)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        graph = torch.load(STRING_GNN_DIR / "graph_data.pt", map_location="cpu")
        edge_index  = graph["edge_index"].long()
        edge_weight = graph["edge_weight"].float()

        # One-time forward pass on CPU -> fixed lookup table [18870, 256]
        with torch.no_grad():
            gnn_out  = backbone(edge_index=edge_index, edge_weight=edge_weight)
            node_emb = gnn_out.last_hidden_state.float().detach()  # [18870, 256]

        self.register_buffer("node_embeddings", node_emb)
        del backbone, graph, edge_index, edge_weight, gnn_out

        # ----------------------------------------------------------------
        # 2. Per-gene class log-prior (FIXED buffer, pre-computed from train data)
        # ----------------------------------------------------------------
        log_prior_np = compute_gene_log_prior(TRAIN_TSV)  # [3, N_GENES]
        self.register_buffer("gene_log_prior", torch.tensor(log_prior_np))  # [3, N_GENES]

        # Single learnable temperature for prior blending (init=0 -> prior ignored initially)
        # Using softplus to ensure non-negative temperature (prior is informative, not harmful)
        self.prior_temperature = nn.Parameter(torch.zeros(1))

        # ----------------------------------------------------------------
        # 3. Learnable fallback for unknown pert_ids (~6.4% of training data)
        # ----------------------------------------------------------------
        self.fallback_emb = nn.Embedding(1, STRING_DIM)
        nn.init.normal_(self.fallback_emb.weight, std=0.02)

        # ----------------------------------------------------------------
        # 4. Simple 2-layer MLP head (proven architecture from sibling node1-1-1-1)
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(STRING_DIM),
            nn.Linear(STRING_DIM, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
            nn.LayerNorm(hp.bilinear_dim),
            nn.Linear(hp.bilinear_dim, hp.bilinear_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )

        # ----------------------------------------------------------------
        # 5. Bilinear gene-class embedding (no gene_bias — removed, prevented overfitting)
        # ----------------------------------------------------------------
        self.gene_class_emb = nn.Parameter(
            torch.randn(N_CLASSES, N_GENES, hp.bilinear_dim) * 0.02
        )

        # Class weights for cross-entropy
        self.register_buffer("class_weights", get_class_weights())

        # Cast all trainable parameters to float32 for stable optimization
        for _, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Accumulators for val/test (cleared each epoch)
        self._val_preds:  List[torch.Tensor] = []
        self._val_tgts:   List[torch.Tensor] = []
        self._val_idx:    List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_idx:   List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Embedding lookup
    # ------------------------------------------------------------------
    def _get_pert_embeddings(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Lookup pre-computed embeddings; use learnable fallback for unknowns.

        Args:
            string_node_idx: [B] long tensor, -1 for pert_ids not in STRING.
        Returns:
            [B, STRING_DIM] float32 perturbation embeddings.
        """
        B = string_node_idx.shape[0]
        emb = torch.zeros(
            B, STRING_DIM,
            dtype=self.node_embeddings.dtype,
            device=self.node_embeddings.device,
        )
        known   = string_node_idx >= 0
        unknown = ~known

        if known.any():
            emb[known] = self.node_embeddings[string_node_idx[known]]

        if unknown.any():
            fb = self.fallback_emb(
                torch.zeros(unknown.sum(), dtype=torch.long, device=self.node_embeddings.device)
            ).to(self.node_embeddings.dtype)
            emb[unknown] = fb

        return emb.float()

    # ------------------------------------------------------------------
    # Forward (from pre-fetched embedding)
    # ------------------------------------------------------------------
    def _forward_from_emb(self, pert_emb: torch.Tensor) -> torch.Tensor:
        """Compute logits from perturbation embedding.

        Args:
            pert_emb: [B, STRING_DIM] float32
        Returns:
            logits: [B, 3, G] float32
        """
        h = self.head(pert_emb)  # [B, bilinear_dim]

        # Bilinear interaction with scaling
        scale  = h.shape[-1] ** -0.5
        logits = torch.einsum("bd,cgd->bcg", h, self.gene_class_emb) * scale  # [B, 3, G]

        # Per-gene class log-prior adjustment
        # softplus(prior_temperature) >= 0 ensures non-negative weighting
        T = F.softplus(self.prior_temperature)  # scalar >= 0
        logits = logits + T * self.gene_log_prior.unsqueeze(0)  # [B, 3, G]

        return logits

    def forward(self, string_node_idx: torch.Tensor) -> torch.Tensor:
        """Return logits [B, 3, G]."""
        pert_emb = self._get_pert_embeddings(string_node_idx)
        return self._forward_from_emb(pert_emb)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted cross-entropy with label smoothing.

        Args:
            logits:  [B, 3, G]
            targets: [B, G]   long, values in {0, 1, 2}
        """
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, 3]
            targets.reshape(-1),                       # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        pert_emb = self._get_pert_embeddings(batch["string_node_idx"])  # [B, STRING_DIM]
        labels = batch["labels"]  # [B, G]

        alpha = self.hparams.mixup_alpha
        if alpha > 0 and self.training and pert_emb.shape[0] > 1:
            # -- Mixup augmentation --
            # Sample lambda from Beta(alpha, alpha)
            lam = float(np.random.beta(alpha, alpha))
            B = pert_emb.shape[0]
            perm = torch.randperm(B, device=pert_emb.device)

            mixed_emb = lam * pert_emb + (1.0 - lam) * pert_emb[perm]  # [B, STRING_DIM]
            labels_a  = labels          # [B, G]
            labels_b  = labels[perm]    # [B, G]

            logits = self._forward_from_emb(mixed_emb)
            loss = lam * self._loss(logits, labels_a) + (1.0 - lam) * self._loss(logits, labels_b)
        else:
            logits = self._forward_from_emb(pert_emb)
            loss   = self._loss(logits, labels)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
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

        # Gather across all DDP ranks
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
        mask   = torch.cat([
            torch.tensor([True], device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
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

        all_preds = self.all_gather(local_preds)  # [W, N_local, 3, G]
        all_idx   = self.all_gather(local_idx)    # [W, N_local]

        if self.trainer.is_global_zero:
            preds_flat = all_preds.view(-1, N_CLASSES, N_GENES)
            idx_flat   = all_idx.view(-1)

            pred_map: Dict[int, torch.Tensor] = {}
            for i in range(len(idx_flat)):
                gid = int(idx_flat[i].item())
                if gid not in pred_map:
                    pred_map[gid] = preds_flat[i]

            test_df = pd.read_csv(TEST_TSV, sep="\t")
            rows = []
            for i in range(len(test_df)):
                if i not in pred_map:
                    continue
                pid  = test_df.iloc[i]["pert_id"]
                sym  = test_df.iloc[i]["symbol"]
                pred = pred_map[i].float().cpu().numpy().tolist()
                rows.append({"idx": pid, "input": sym, "prediction": json.dumps(pred)})

            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-1-1-2] Saved {len(rows)} test predictions.")

        self._test_preds.clear()
        self._test_idx.clear()

    # ------------------------------------------------------------------
    # Checkpoint helpers — save only trainable params + buffers
    # ------------------------------------------------------------------
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
        self.print(f"Checkpoint: {train}/{total} params ({100 * train / total:.1f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------------
    # Optimizer: head-only AdamW + linear warmup + CosineAnnealingLR
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        hp = self.hparams

        # Only head parameters — backbone is frozen
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=hp.lr, weight_decay=hp.weight_decay)

        # Phase 1: linear warmup from 0.1×lr to lr over warmup_epochs epochs
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_epochs,
        )
        # Phase 2: cosine annealing (T_max=100, no warm restarts)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=hp.T_max,
            eta_min=1e-6,
        )
        # Sequential: warmup first, then cosine
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_sched, cosine_sched],
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
        description="Node1-1-1-2 – Frozen STRING_GNN + Mixup + Per-Gene Log Prior"
    )
    parser.add_argument("--micro-batch-size",  type=int,   default=32)
    parser.add_argument("--global-batch-size", type=int,   default=256)
    parser.add_argument("--max-epochs",        type=int,   default=150)
    parser.add_argument("--lr",                type=float, default=2e-4)
    parser.add_argument("--weight-decay",      type=float, default=2e-2)
    parser.add_argument("--bilinear-dim",      type=int,   default=256)
    parser.add_argument("--dropout",           type=float, default=0.4)
    parser.add_argument("--warmup-epochs",     type=int,   default=25)
    parser.add_argument("--t-max",             type=int,   default=100, dest="t_max")
    parser.add_argument("--label-smoothing",   type=float, default=0.05,
                        dest="label_smoothing")
    parser.add_argument("--mixup-alpha",       type=float, default=0.4,
                        dest="mixup_alpha")
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

    # Limit / debug logic
    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        lim_test  = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = 1.0
        lim_val   = 1.0
        lim_test  = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    # DataModule
    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    dm.setup()

    # Model
    model = FrozenStringGNNMixupModel(
        bilinear_dim    = args.bilinear_dim,
        dropout         = args.dropout,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        T_max           = args.t_max,
        label_smoothing = args.label_smoothing,
        mixup_alpha     = args.mixup_alpha,
    )

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
    )
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=5, min_delta=1e-4)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    # Loggers
    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    # Strategy
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
        val_check_interval      = 1.0,
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
    print(f"[Node1-1-1-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
