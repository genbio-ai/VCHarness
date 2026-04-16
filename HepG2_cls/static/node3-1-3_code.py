"""node3-1-3: STRING_GNN frozen + 3-block PreNormResNet (h=384) + Muon+AdamW + Manifold Mixup

Architecture Overview:
  - Frozen STRING_GNN PPI embeddings (256-dim): precomputed once, no backbone gradient
  - 3-block Pre-LayerNorm residual MLP (hidden_dim=384, expand=2, head_dropout=0.05)
  - Additive per-gene bias: [1, 3, 6640] = 19,920 params for gene-specific base rates
  - MuonWithAuxAdam: Muon (LR=0.01) for hidden block weight matrices (fc1/fc2),
    AdamW (LR=3e-4, wd=8e-4) for input projection, output head, biases, LayerNorm
  - Manifold Mixup (alpha=0.2, prob=0.5) applied in feature embedding space
  - ReduceLROnPlateau (factor=0.5, patience=8, mode=max) monitoring val/f1
  - Weighted cross-entropy (no focal loss, no label smoothing)
  - Gradient clipping (max_norm=2.0)

Key Improvements over both siblings (node3-1-1, node3-1-2):
  1. Muon+AdamW optimizer: the critical differentiator that achieves train/loss~0.2
     vs AdamW-only train/loss~0.93-1.5 on identical architectures, driving STRING-only
     results from ~0.38 to ~0.48-0.50 (node3-3-1-1: F1=0.479, node3-3-1-2: F1=0.497)
  2. h=384 with head_dropout=0.05: all tree-best STRING nodes use h=384, not h=512
     which over-regularizes with AdamW. Muon needs lighter head dropout.
  3. Manifold Mixup: the key augmentation that pushed node3-3-1-2 to F1=0.4966 (best
     STRING-only), providing implicit regularization without reducing model capacity
  4. No label smoothing: creates a training loss floor with Muon, preventing convergence
     (node3-3-1-1-1-1: removing label smoothing was the key fix for train/loss=0.94→0.2)
  5. ReduceLROnPlateau (patience=8): plateau-triggered LR reduction is uniquely effective
     for this task vs cosine annealing (node3-3-1: RLROP > cosine with Muon)
  6. Gradient clipping (max_norm=2.0): prevents large gradient spikes from instability
  7. No LR warmup: node3-1-1's 5-epoch warmup caused 28% val/f1 collapse; no warmup
     is consistent with all successful STRING+Muon nodes

Primary memory sources:
  - node3-3-1-2 (F1=0.4966): STRING+Muon+AdamW+Manifold Mixup+RLROP — exact recipe
  - node3-3-1-1 (F1=0.4793): Muon LR=0.01, no label smoothing, head_dropout=0.05
  - node3-1-2 sibling feedback: RLROP patience=10 caused 4 halvings (too many → use 8)
  - node3-1-1 sibling feedback: warmup+low LR+over-regularization underfit critically
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from transformers import AutoModel

try:
    from muon import MuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    print("WARNING: muon package not found, falling back to AdamW only", flush=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES = 6640
N_CLASSES = 3
STRING_EMB_DIM = 256  # STRING_GNN output embedding dimension


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class PreNormResBlock(nn.Module):
    """Pre-LayerNorm residual block: LN → Linear → GELU → Dropout → Linear → Dropout + residual."""

    def __init__(self, dim: int, expand: int = 2, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expand)
        self.fc2 = nn.Linear(dim * expand, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class PerturbHead(nn.Module):
    """STRING_GNN embedding → 3-block PreNorm MLP → [B, 3, N_GENES] with per-gene bias."""

    def __init__(
        self,
        in_dim: int = STRING_EMB_DIM,
        hidden_dim: int = 384,
        n_genes: int = N_GENES,
        n_blocks: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        # Input projection (LayerNorm + Linear + GELU — no dropout for first proj)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        # Pre-norm residual blocks (Muon handles these inner weight matrices)
        self.blocks = nn.ModuleList([
            PreNormResBlock(hidden_dim, expand=2, dropout=dropout)
            for _ in range(n_blocks)
        ])
        # Output LayerNorm before head projection
        self.out_norm = nn.LayerNorm(hidden_dim)
        # Flat output head: hidden_dim → n_genes * n_classes
        self.out_proj = nn.Linear(hidden_dim, n_genes * N_CLASSES)
        # Per-gene additive bias: gene-specific output offsets (initialized to zero)
        self.per_gene_bias = nn.Parameter(torch.zeros(1, N_CLASSES, n_genes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)              # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)                    # [B, hidden_dim]
        x = self.out_norm(x)
        logits = self.out_proj(x).view(-1, N_CLASSES, self.n_genes)  # [B, 3, N_GENES]
        return logits + self.per_gene_bias  # Gene-specific base rate offsets


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Maps each perturbed gene to its precomputed STRING_GNN embedding."""

    def __init__(
        self,
        df: pd.DataFrame,
        gene_features: torch.Tensor,
        ensg_to_idx: Dict[str, int],
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.gene_features = gene_features      # [N_NODES, STRING_EMB_DIM] CPU float32
        self.ensg_to_idx = ensg_to_idx

        if "label" in df.columns:
            labels = np.array(
                [json.loads(x) for x in df["label"].tolist()], dtype=np.int64
            )
            self.labels: Optional[torch.Tensor] = torch.tensor(
                labels + 1, dtype=torch.long
            )  # shift {-1,0,1} → {0,1,2}
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pert_id = self.pert_ids[idx]
        gnn_idx = self.ensg_to_idx.get(pert_id, -1)
        if gnn_idx >= 0:
            feat = self.gene_features[gnn_idx]  # [STRING_EMB_DIM]
        else:
            # ~7% of genes not in STRING graph → zero vector fallback
            feat = torch.zeros(STRING_EMB_DIM)
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": pert_id,
            "symbol": self.symbols[idx],
            "features": feat,
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
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
        micro_batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.gene_features: Optional[torch.Tensor] = None
        self.ensg_to_idx: Optional[Dict[str, int]] = None

    def setup(self, stage: str = "fit") -> None:
        if self.gene_features is None:
            self._precompute_features()
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")
        self.train_ds = PerturbDataset(train_df, self.gene_features, self.ensg_to_idx)
        self.val_ds = PerturbDataset(val_df, self.gene_features, self.ensg_to_idx)
        self.test_ds = PerturbDataset(test_df, self.gene_features, self.ensg_to_idx)

    def _precompute_features(self) -> None:
        """Run STRING_GNN forward pass once to get frozen PPI embeddings [N, 256]."""
        model_dir = Path(STRING_GNN_DIR)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        node_names: List[str] = json.loads(
            (model_dir / "node_names.json").read_text()
        )
        self.ensg_to_idx = {name: i for i, name in enumerate(node_names)}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading STRING_GNN for precomputing frozen PPI embeddings...", flush=True)
        gnn = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device)
        gnn.eval()
        graph = torch.load(model_dir / "graph_data.pt", map_location=device)
        edge_index = graph["edge_index"]
        edge_weight = graph.get("edge_weight", None)
        with torch.no_grad():
            out = gnn(
                edge_index=edge_index,
                edge_weight=edge_weight,
                output_hidden_states=False,
            )
            self.gene_features = out.last_hidden_state.float().cpu()  # [N, 256]
        del gnn, graph, out
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"STRING_GNN features precomputed: {self.gene_features.shape}", flush=True)

    def _make_loader(self, ds: PerturbDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = 384,
        n_blocks: int = 3,
        dropout: float = 0.05,
        muon_lr: float = 0.01,
        adamw_lr: float = 3e-4,
        weight_decay: float = 8e-4,
        rlrop_patience: int = 8,
        rlrop_factor: float = 0.5,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.head: Optional[PerturbHead] = None
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        self.head = PerturbHead(
            in_dim=STRING_EMB_DIM,
            hidden_dim=self.hparams.hidden_dim,
            n_genes=N_GENES,
            n_blocks=self.hparams.n_blocks,
            dropout=self.hparams.dropout,
        )
        # Cast to float32 for stable optimization
        for p in self.head.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Class weights (freq: down=4.77%, neutral=92.82%, up=2.41%)
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = 1.0 / freq
        class_weights = class_weights / class_weights.mean()
        self.register_buffer("class_weights", class_weights)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        muon_cnt = sum(
            p.numel() for n, p in self.head.blocks.named_parameters()
            if p.requires_grad and p.ndim >= 2 and "weight" in n
        )
        self.print(
            f"PerturbHead | trainable={trainable:,}/{total:,} | "
            f"h={self.hparams.hidden_dim}, blocks={self.hparams.n_blocks}, "
            f"dropout={self.hparams.dropout} | "
            f"Muon={muon_cnt:,} params in residual block weight matrices"
        )

    # ------------------------------------------------------------------ #
    # Manifold Mixup in feature embedding space                           #
    # ------------------------------------------------------------------ #
    def _manifold_mixup(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Apply Manifold Mixup to STRING_GNN embeddings with probability mixup_prob.

        Returns mixed features, original labels, and (labels_b, lam) or None.
        """
        if not self.training or torch.rand(1, device=features.device).item() > self.hparams.mixup_prob:
            return features, labels, None
        lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
        batch_size = features.size(0)
        perm = torch.randperm(batch_size, device=features.device)
        mixed_features = lam * features + (1.0 - lam) * features[perm]
        return mixed_features, labels, (labels[perm], lam)

    # ------------------------------------------------------------------ #
    # Loss                                                                #
    # ------------------------------------------------------------------ #
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mixup_data: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """Weighted cross-entropy (no focal loss, no label smoothing) with optional Mixup."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        if mixup_data is not None:
            labels_b, lam = mixup_data
            labels_b_flat = labels_b.reshape(-1)
            loss_a = F.cross_entropy(logits_flat, labels_flat, weight=self.class_weights)
            loss_b = F.cross_entropy(logits_flat, labels_b_flat, weight=self.class_weights)
            return lam * loss_a + (1.0 - lam) * loss_b
        return F.cross_entropy(logits_flat, labels_flat, weight=self.class_weights)

    # ------------------------------------------------------------------ #
    # Training / Validation / Test Steps                                  #
    # ------------------------------------------------------------------ #
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        feats = batch["features"].to(self.device).float()
        labels = batch["label"]
        feats, labels, mixup_data = self._manifold_mixup(feats, labels)
        logits = self.head(feats)
        loss = self._compute_loss(logits, labels, mixup_data)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        feats = batch["features"].to(self.device).float()
        logits = self.head(feats)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds, dim=0).numpy()    # [local_N, 3, N_GENES]
        labels = torch.cat(self._val_labels, dim=0).numpy()  # [local_N, N_GENES]
        self._val_preds.clear()
        self._val_labels.clear()
        f1 = _compute_per_gene_f1(preds, labels)
        # sync_dist=True averages across DDP ranks (consistent monitor for RLROP)
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        feats = batch["features"].to(self.device).float()
        logits = self.head(feats)
        self._test_preds.append(logits.detach().cpu().float())
        if "label" in batch:
            self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        import torch.distributed as dist

        preds_local = torch.cat(self._test_preds, dim=0)      # [local_N, 3, N_GENES]
        labels_local = (
            torch.cat(self._test_labels, dim=0) if self._test_labels else None
        )
        self._test_preds.clear()
        self._test_labels.clear()

        # Gather predictions from all DDP ranks
        all_preds = self.all_gather(preds_local)               # [world_size, local_N, 3, N_GENES]
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        if labels_local is not None:
            all_labels = self.all_gather(labels_local)
            all_labels = all_labels.view(-1, N_GENES)
            test_f1 = _compute_per_gene_f1(
                all_preds.float().cpu().numpy(),
                all_labels.cpu().numpy(),
            )
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=True)

        # Gather string metadata across ranks
        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        gathered_pert_ids = [local_pert_ids]
        gathered_symbols = [local_symbols]
        if world_size > 1:
            obj_pert = [None] * world_size
            obj_sym = [None] * world_size
            dist.all_gather_object(obj_pert, local_pert_ids)
            dist.all_gather_object(obj_sym, local_symbols)
            gathered_pert_ids = obj_pert
            gathered_symbols = obj_sym

        if self.trainer.is_global_zero:
            all_pert_ids = [p for rank_list in gathered_pert_ids for p in rank_list]
            all_symbols = [s for rank_list in gathered_symbols for s in rank_list]
            all_preds_np = all_preds.float().cpu().numpy()

            # Deduplicate by pert_id (handles DDP data padding)
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(all_preds_np[i])

            _save_test_predictions(
                pert_ids=dedup_ids,
                symbols=dedup_syms,
                preds=np.stack(dedup_preds, axis=0),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

    # ------------------------------------------------------------------ #
    # Optimizer: Muon for hidden block weights, AdamW for everything else #
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        # Muon parameters: inner weight matrices of residual blocks (2D, fc1/fc2)
        # These benefit most from orthogonalized gradient updates
        muon_params = []
        muon_param_ids: set = set()
        for name, param in self.head.blocks.named_parameters():
            if param.requires_grad and param.ndim >= 2 and "weight" in name:
                muon_params.append(param)
                muon_param_ids.add(id(param))

        # AdamW parameters: input projection, out_norm, out_proj, per_gene_bias,
        # all LayerNorm params, all biases
        adamw_params = [
            p for p in self.head.parameters()
            if p.requires_grad and id(p) not in muon_param_ids
        ]

        if MUON_AVAILABLE and muon_params:
            param_groups = [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=self.hparams.muon_lr,
                    weight_decay=0.0,    # Muon's orthogonal update handles implicit reg
                    momentum=0.95,
                ),
                dict(
                    params=adamw_params,
                    use_muon=False,
                    lr=self.hparams.adamw_lr,
                    betas=(0.9, 0.95),
                    weight_decay=self.hparams.weight_decay,
                ),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
            self.print(
                f"Using MuonWithAuxAdam: Muon LR={self.hparams.muon_lr} "
                f"({len(muon_params)} matrices), "
                f"AdamW LR={self.hparams.adamw_lr} ({len(adamw_params)} params)"
            )
        else:
            # Fallback to pure AdamW if Muon unavailable
            self.print("WARNING: Falling back to pure AdamW (Muon unavailable)")
            all_params = [p for p in self.head.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                all_params, lr=self.hparams.adamw_lr, weight_decay=self.hparams.weight_decay
            )

        # ReduceLROnPlateau: monitor val/f1, halve LR on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.rlrop_factor,
            patience=self.hparams.rlrop_patience,
            min_lr=1e-7,
            threshold=1e-5,   # Require at least 1e-5 improvement to avoid over-triggering
            threshold_mode="rel",
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

    # ------------------------------------------------------------------ #
    # Checkpoint: save only trainable parameters                          #
    # ------------------------------------------------------------------ #
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        result = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    result[key] = full_sd[key]
        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                result[key] = full_sd[key]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving checkpoint: {trainable:,}/{total:,} params "
            f"({100 * trainable / total:.2f}%)"
        )
        return result

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-F1 averaged over all 6640 response genes.

    Matches data/calc_metric.py exactly:
      - argmax(preds, axis=1) → hard predictions
      - Per-gene F1 averaged over present classes only
      - Final F1 = mean over all genes
    """
    from sklearn.metrics import f1_score as sk_f1
    y_hat = preds.argmax(axis=1)   # [N, N_GENES]
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        if present.any():
            f1_vals.append(float(per_class_f1[present].mean()))
        else:
            f1_vals.append(0.0)
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(pert_ids)):
        rows.append({
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STRING_GNN + Muon+AdamW + Manifold Mixup for HepG2 DEG Prediction"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=400)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=8e-4)
    p.add_argument("--rlrop-patience", type=int, default=8)
    p.add_argument("--rlrop-factor", type=float, default=0.5)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--early-stop-patience", type=int, default=40)
    p.add_argument("--gradient-clip-val", type=float, default=2.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    model = PerturbModule(
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        rlrop_patience=args.rlrop_patience,
        rlrop_factor=args.rlrop_factor,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
    )

    if args.debug_max_step is not None:
        limit_train = limit_val = limit_test = args.debug_max_step
        max_steps = args.debug_max_step
        val_check_interval = 1.0
        num_sanity_val_steps = 0
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval if not args.fast_dev_run else 1.0
        num_sanity_val_steps = 2

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best_epoch{epoch:03d}_f1{val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=3,     # Keep top-3 for potential ensemble use
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
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        gradient_clip_val=args.gradient_clip_val,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    # Test with best checkpoint
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    if trainer.is_global_zero and test_results:
        result = test_results[0]
        primary_metric = result.get("test/f1", float("nan"))
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(str(float(primary_metric)))
        print(f"Test results → {score_path} (f1_score={primary_metric:.4f})", flush=True)


if __name__ == "__main__":
    main()
