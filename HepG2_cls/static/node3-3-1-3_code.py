"""Node 1-2: STRING_GNN Frozen Embeddings + 3-Block Pre-Norm MLP (h=384)
             + Muon (LR=0.01) + Multi-Checkpoint Ensemble + Trunk Reg 0.35
================================================================
Parent  : node3-3-1  (STRING+3-block+h=384+Muon LR=0.02+label_smooth=0.05,
                      test F1=0.4226, severe underfitting: train/loss=0.94)

Sibling : node3-3-1-1 (STRING+3-block+Muon LR=0.01+no label_smooth+
                       head_drop=0.05, test F1=0.4793, new STRING-only best)

Design philosophy: same proven underfitting-fix recipe as sibling, but
differentiated by:
  1. Trunk dropout 0.30 → 0.35 to address sibling's growing train-val gap
  2. Small label smoothing re-introduction (0.0 → 0.01) for calibration
  3. Multi-checkpoint ensemble (top-3 checkpoints) at test time

Key changes from parent node3-3-1:
------------------------------------
1. Muon LR 0.02 → 0.01  (fixes oscillation, same as sibling)
2. Label smoothing 0.05 → 0.01  (sibling removed fully; we re-add minimally)
3. Head dropout 0.15 → 0.05  (frees output head to fit training data)
4. Trunk dropout 0.30 → 0.35  (extra regularization for train-val gap)
5. Gradient clip 1.0 → 2.0  (preserves Muon update scale)
6. RLROP patience 8 → 10  (slower convergence needs more patience)
7. Max epochs 300 → 400  (more training time)
8. save_top_k=3 + multi-checkpoint ensemble at test time  (novel innovation)

Unchanged from proven recipe:
------------------------------
- STRING_GNN frozen PPI graph embeddings (256-dim)
- Per-gene additive bias (19,920 learnable parameters)
- Pre-norm residual block structure
- Correct class-weight order: [0.0477, 0.9282, 0.0241] (down/neutral/up)
- Weight decay = 0.01
- min_lr = 1e-7, early-stop patience = 30
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import glob
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
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRING_GNN_DIR = "/home/Models/STRING_GNN"
N_GENES = 6640        # number of response genes per perturbation
N_CLASSES = 3         # down (-1→0), neutral (0→1), up (1→2)
GNN_DIM = 256         # STRING_GNN output embedding dimension
HIDDEN_DIM = 384      # MLP hidden dimension — proven optimal
INNER_DIM = 768       # MLP inner (expansion) dimension (2x hidden per PreLN block)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    """Each sample is one gene perturbation experiment in HepG2 cells."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()

        if "label" in df.columns:
            # Labels in {-1,0,1} → shift to {0,1,2}
            labels = np.array(
                [json.loads(x) for x in df["label"].tolist()], dtype=np.int64
            )
            self.labels: Optional[torch.Tensor] = torch.tensor(
                labels + 1, dtype=torch.long
            )
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
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

        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df)
        self.val_ds = PerturbDataset(val_df)
        self.test_ds = PerturbDataset(test_df)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------
class PreNormResBlock(nn.Module):
    """Pre-LayerNorm residual block.

    Architecture (proven stable):
        output = x + LN(x) → Linear(dim→inner) → GELU → Dropout
                               → Linear(inner→dim) → Dropout
    """

    def __init__(self, dim: int, inner_dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        inner_dim: int = INNER_DIM,
        n_blocks: int = 3,
        dropout: float = 0.35,          # Increased from 0.30 to address train-val gap
        head_dropout: float = 0.05,     # Reduced from 0.15 (proven by sibling)
        muon_lr: float = 0.01,          # Reduced from 0.02 (key underfitting fix)
        adamw_lr: float = 3e-4,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.01,  # Minimal re-introduction (vs sibling's 0.0)
        rlrop_factor: float = 0.5,
        rlrop_patience: int = 10,       # Extended from 8 (slower convergence at LR=0.01)
        min_lr: float = 1e-7,
        grad_clip_norm: float = 2.0,    # Loosened from 1.0 (Muon update scale)
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Populated in setup()
        self.input_proj: Optional[nn.Sequential] = None
        self.blocks: Optional[nn.ModuleList] = None
        self.output_head: Optional[nn.Sequential] = None
        self.gene_bias: Optional[nn.Parameter] = None

        # STRING_GNN gene-ID → embedding-row index
        self.gnn_id_to_idx: Dict[str, int] = {}

        # Metric accumulators
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """Build model and precompute frozen STRING_GNN node embeddings."""
        from transformers import AutoModel

        self.print("Loading STRING_GNN and computing frozen node embeddings …")
        gnn_model = AutoModel.from_pretrained(
            STRING_GNN_DIR, trust_remote_code=True
        )
        gnn_model.eval()
        gnn_model = gnn_model.to(self.device)

        graph = torch.load(
            Path(STRING_GNN_DIR) / "graph_data.pt",
            map_location=self.device,
        )
        edge_index = graph["edge_index"].to(self.device)
        edge_weight = graph.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        with torch.no_grad():
            gnn_out = gnn_model(edge_index=edge_index, edge_weight=edge_weight)

        # Register as a non-trainable float32 buffer [18870, 256]
        all_emb = gnn_out.last_hidden_state.detach().float()
        self.register_buffer("gnn_embeddings", all_emb)

        # Free GNN model memory
        del gnn_model, gnn_out, graph, edge_index, edge_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.print(f"STRING_GNN embeddings shape: {all_emb.shape}")

        # Build ENSG-ID → row-index mapping
        node_names: List[str] = json.loads(
            (Path(STRING_GNN_DIR) / "node_names.json").read_text()
        )
        self.gnn_id_to_idx = {name: i for i, name in enumerate(node_names)}
        n_covered = len(self.gnn_id_to_idx)
        self.print(f"STRING_GNN covers {n_covered} Ensembl gene IDs")

        # ---- MLP architecture ----
        hp = self.hparams
        self.input_proj = nn.Sequential(
            nn.LayerNorm(GNN_DIM),
            nn.Linear(GNN_DIM, hp.hidden_dim),
            nn.GELU(),
            nn.Dropout(hp.dropout),
        )
        self.blocks = nn.ModuleList(
            [
                PreNormResBlock(hp.hidden_dim, hp.inner_dim, hp.dropout)
                for _ in range(hp.n_blocks)
            ]
        )
        # Output head with minimal dropout (p=0.05) — balances fitting and generalization
        self.output_head = nn.Sequential(
            nn.LayerNorm(hp.hidden_dim),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.hidden_dim, N_GENES * N_CLASSES),
        )

        # Per-gene additive bias: one offset per (gene × class) pair
        self.gene_bias = nn.Parameter(torch.zeros(N_GENES, N_CLASSES))

        # ---- Class weights (CORRECT ordering after +1 label shift) ----
        # class 0 = down-regulated  (4.77%)  → high weight
        # class 1 = neutral         (92.82%) → low weight
        # class 2 = up-regulated    (2.41%)  → highest weight
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq) / (1.0 / freq).mean()
        self.register_buffer("class_weights", class_weights)

        # Cast trainable params to float32 for stable optimization
        for k, v in self.named_parameters():
            if v.requires_grad:
                v.data = v.data.float()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Architecture: STRING_GNN({GNN_DIM}) → Proj → "
            f"{hp.n_blocks}×PreNormResBlock({hp.hidden_dim},{hp.inner_dim},drop={hp.dropout}) "
            f"→ HeadDropout({hp.head_dropout}) → Linear({hp.hidden_dim},{N_GENES}×{N_CLASSES}) + gene_bias"
        )
        self.print(f"Trainable params: {trainable:,} / {total:,}")

    # ------------------------------------------------------------------
    def _get_gene_emb(self, pert_ids: List[str]) -> torch.Tensor:
        """Batch lookup of frozen STRING_GNN embeddings for ENSG IDs.

        Genes absent from STRING_GNN (~7% of samples) receive a zero vector.
        """
        emb_list: List[torch.Tensor] = []
        for pid in pert_ids:
            row = self.gnn_id_to_idx.get(pid)
            if row is not None:
                emb_list.append(self.gnn_embeddings[row])
            else:
                emb_list.append(
                    torch.zeros(GNN_DIM, device=self.device, dtype=torch.float32)
                )
        return torch.stack(emb_list, dim=0)  # [B, 256]

    def forward(self, pert_ids: List[str]) -> torch.Tensor:
        """Return logits of shape [B, N_CLASSES, N_GENES]."""
        x = self._get_gene_emb(pert_ids)              # [B, 256]
        x = self.input_proj(x)                         # [B, 384]
        for block in self.blocks:
            x = block(x)                               # [B, 384]
        logits = self.output_head(x)                   # [B, N_GENES * N_CLASSES]
        logits = logits.view(-1, N_CLASSES, N_GENES)   # [B, 3, 6640]
        # gene_bias: [N_GENES, N_CLASSES].T → [N_CLASSES, N_GENES] → [1, 3, N_GENES]
        logits = logits + self.gene_bias.T.unsqueeze(0)
        return logits

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Weighted CE + minimal label smoothing on [B, N_CLASSES, N_GENES] logits."""
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(
            logits_flat,
            labels_flat,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["pert_id"])
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        loss = self._compute_loss(logits, batch["label"])
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        preds_local = torch.cat(self._val_preds, dim=0)    # [N_local, 3, 6640]
        labels_local = torch.cat(self._val_labels, dim=0)  # [N_local, 6640]
        self._val_preds.clear()
        self._val_labels.clear()

        import torch.distributed as dist

        is_dist = dist.is_available() and dist.is_initialized()

        if is_dist and self.trainer.world_size > 1:
            world_size = dist.get_world_size()
            preds_np_local = preds_local.numpy()
            labels_np_local = labels_local.numpy()

            obj_preds = [None] * world_size
            obj_labels = [None] * world_size
            dist.all_gather_object(obj_preds, preds_np_local)
            dist.all_gather_object(obj_labels, labels_np_local)

            preds_np = np.concatenate(obj_preds, axis=0)
            labels_np = np.concatenate(obj_labels, axis=0)
            f1 = _compute_per_gene_f1(preds_np, labels_np)
            # Log on all ranks so EarlyStopping / RLROP can access the metric.
            self.log("val/f1", f1, prog_bar=True)
        else:
            preds_np = preds_local.numpy()
            labels_np = labels_local.numpy()
            f1 = _compute_per_gene_f1(preds_np, labels_np)
            self.log("val/f1", f1, prog_bar=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["pert_id"])
        self._test_preds.append(logits.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        """Gather test predictions from all ranks and deduplicate.

        Uses self.trainer.world_size to detect single-device vs DDP context.
        When running via _run_ensemble_test (single-device ckpt_trainer),
        world_size==1 so no distributed gathering is needed.
        """
        preds_local = torch.cat(self._test_preds, dim=0)  # [N_local, 3, 6640]
        self._test_preds.clear()

        local_pert_ids = list(self._test_pert_ids)
        local_symbols = list(self._test_symbols)
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        # Detect multi-GPU DDP context via trainer's world_size
        # SingleDeviceStrategy → world_size==1 → skip distributed gathering
        world_size = getattr(self.trainer, "world_size", 1)

        if world_size > 1:
            # Multi-GPU DDP: gather predictions and metadata
            gathered = self.all_gather(preds_local)  # [world_size, N_local, 3, 6640]
            all_preds = gathered.view(-1, N_CLASSES, N_GENES)  # [N_total, 3, 6640]

            import torch.distributed as dist
            obj_pids = [None] * world_size
            obj_syms = [None] * world_size
            dist.all_gather_object(obj_pids, local_pert_ids)
            dist.all_gather_object(obj_syms, local_symbols)
            all_pert_ids = [pid for lst in obj_pids for pid in lst]
            all_symbols = [sym for lst in obj_syms for sym in lst]
        else:
            # Single-device (ensemble inference or debug): use local data directly
            all_preds = preds_local
            all_pert_ids = local_pert_ids
            all_symbols = local_symbols

        if self.trainer.is_global_zero:
            # De-duplicate (DDP may replicate samples across ranks)
            seen: set = set()
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            preds_np = all_preds.cpu().numpy()  # [N_total, 3, 6640]
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(preds_np[i])

            # Store for possible multi-checkpoint averaging in main()
            self._final_pert_ids = dedup_ids
            self._final_symbols = dedup_syms
            self._final_preds = np.stack(dedup_preds, axis=0)  # [N, 3, 6640]

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        from muon import MuonWithAuxAdam

        hp = self.hparams

        # Separate parameters for Muon vs AdamW:
        # Muon: 2D weight matrices in the hidden residual blocks
        # AdamW: all other parameters (norms, biases, input_proj, output_head, gene_bias)
        muon_params = [
            p for name, p in self.blocks.named_parameters()
            if p.ndim >= 2 and p.requires_grad
        ]
        # All other trainable params go to AdamW
        muon_param_ids = set(id(p) for p in muon_params)
        adamw_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in muon_param_ids
        ]

        param_groups = [
            # Muon group for hidden block weight matrices
            dict(
                params=muon_params,
                use_muon=True,
                lr=hp.muon_lr,
                weight_decay=hp.weight_decay,
                momentum=0.95,
            ),
            # AdamW group for embeddings, norms, biases, head
            dict(
                params=adamw_params,
                use_muon=False,
                lr=hp.adamw_lr,
                betas=(0.9, 0.95),
                weight_decay=hp.weight_decay,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)

        # ReduceLROnPlateau: monitors val/f1 (mode='max'); halves LR at plateaus.
        # patience=10 extended from 8 to accommodate slower LR=0.01 convergence.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=hp.rlrop_factor,
            patience=hp.rlrop_patience,
            min_lr=hp.min_lr,
            threshold=1e-5,
            threshold_mode="abs",
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
    # Checkpoint helpers (use full state_dict for reliable loading)
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        n_tensors = len(sd)
        total_elems = sum(v.numel() for v in sd.values())
        print(f"Saving checkpoint: {n_tensors} tensors ({total_elems:,} elements)")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Per-gene macro-averaged F1 exactly matching calc_metric.py.

    preds  : [N_samples, 3, N_genes]  — logits / class scores
    labels : [N_samples, N_genes]     — integer class labels in {0, 1, 2}
    """
    from sklearn.metrics import f1_score as sk_f1

    y_hat = preds.argmax(axis=1)  # [N_samples, N_genes]
    n_genes = labels.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels[:, g]
        yh = y_hat[:, g]
        per_class_f1 = sk_f1(
            yt, yh, labels=[0, 1, 2], average=None, zero_division=0
        )
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        f1_vals.append(float(per_class_f1[present].mean()))
    return float(np.mean(f1_vals))


def _save_test_predictions(
    pert_ids: List[str],
    symbols: List[str],
    preds: np.ndarray,
    out_path: Path,
) -> None:
    """Save test predictions in the TSV format required by calc_metric.py."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert len(pert_ids) == len(preds), (
        f"Length mismatch: {len(pert_ids)} pert_ids vs {len(preds)} pred rows"
    )
    rows = [
        {
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),  # [3, 6640] as JSON
        }
        for i in range(len(pert_ids))
    ]
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


def _run_ensemble_test(
    datamodule: PerturbDataModule,
    checkpoint_dir: Path,
    output_dir: Path,
    base_model_args: dict,
) -> None:
    """Load top-K checkpoints and ensemble their test predictions by averaging logits.

    This is the key novel contribution of node1-2. By averaging predictions
    from the top-3 checkpoints (which correspond to different points in the
    training trajectory), we reduce prediction variance and improve robustness.

    The ensemble function runs on rank 0 only (called inside a
    is_global_zero guard in main). Each checkpoint is loaded into a model
    and inference is run on a single-device trainer to avoid DDP complexity.
    """
    # Find all saved checkpoints by val/f1 score (exclude last.ckpt)
    # Filename format: best-epoch=XXX-valf1=Y.YYYY.ckpt
    ckpt_pattern = str(checkpoint_dir / "best-epoch=*.ckpt")
    ckpt_files = sorted(
        glob.glob(ckpt_pattern),
        key=lambda f: float(Path(f).stem.split("valf1=")[-1]) if "valf1=" in Path(f).stem else 0.0,
        reverse=True,
    )

    if len(ckpt_files) == 0:
        print("WARNING: No best checkpoints found for ensemble. Output not saved.")
        return

    print(f"Found {len(ckpt_files)} checkpoint(s) for ensemble: {[Path(f).name for f in ckpt_files]}")

    # Collect predictions from each checkpoint using single-device inference
    all_preds_list = []  # list of np arrays [N, 3, 6640]
    ref_pert_ids = None
    ref_symbols = None

    for ckpt_path in ckpt_files:
        print(f"Running test inference with checkpoint: {Path(ckpt_path).name}")
        model_ckpt = PerturbModule(**base_model_args)

        # Use SingleDeviceStrategy for ensemble inference to avoid DDP complexity
        from lightning.pytorch.strategies import SingleDeviceStrategy
        ckpt_trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            strategy="auto",
            precision="bf16-mixed",
            logger=False,
            enable_checkpointing=False,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
            deterministic=True,
            default_root_dir=str(output_dir),
        )
        ckpt_trainer.test(model_ckpt, datamodule=datamodule, ckpt_path=ckpt_path)

        # Retrieve predictions from the module attribute set in on_test_epoch_end
        if hasattr(model_ckpt, "_final_preds") and model_ckpt._final_preds is not None:
            all_preds_list.append(model_ckpt._final_preds)
            if ref_pert_ids is None:
                ref_pert_ids = model_ckpt._final_pert_ids
                ref_symbols = model_ckpt._final_symbols
            print(f"  Collected predictions shape: {model_ckpt._final_preds.shape}")
        else:
            print(f"  WARNING: No predictions found for checkpoint {Path(ckpt_path).name}")

        # Free memory
        del model_ckpt, ckpt_trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_preds_list) == 0:
        print("ERROR: No predictions collected from any checkpoint. Saving empty result.")
        return

    # Average logits across all collected checkpoints
    print(f"Ensembling {len(all_preds_list)} checkpoint predictions by averaging logits...")
    ensemble_preds = np.mean(all_preds_list, axis=0)  # [N, 3, 6640]
    print(f"Ensemble predictions shape: {ensemble_preds.shape}")

    # Save ensemble predictions
    _save_test_predictions(
        pert_ids=ref_pert_ids,
        symbols=ref_symbols,
        preds=ensemble_preds,
        out_path=output_dir / "test_predictions.tsv",
    )
    print(f"Multi-checkpoint ensemble ({len(all_preds_list)} checkpoints) saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node1-2: STRING_GNN + 3-Block MLP (h=384) + Muon (LR=0.01) + "
                    "Multi-Checkpoint Ensemble + Trunk Reg 0.35"
    )
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--global-batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=400)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--head-dropout", type=float, default=0.05)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--inner-dim", type=int, default=768)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--rlrop-factor", type=float, default=0.5)
    p.add_argument("--rlrop-patience", type=int, default=10)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--grad-clip-norm", type=float, default=2.0)
    p.add_argument("--save-top-k", type=int, default=3,
                   help="Number of checkpoints to save for multi-checkpoint ensemble")
    p.add_argument("--early-stop-patience", type=int, default=30)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--test-only", action="store_true",
                   help="Skip training and run ensemble test only (use existing checkpoints)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    # Compute data_dir robustly: walk up from the real script path until we find
    # a directory containing both 'data/' and 'mcts/' subdirectories.
    # This handles the case where mcts/node is a symlink (Path resolves symlinks
    # in __file__ but parent traversal may not follow the same symlink chain).
    _script_real = Path(__file__).resolve()  # resolve symlinks in __file__
    _root = _script_real.parent
    while not ((_root / "data").is_dir() and (_root / "mcts").is_dir()):
        _parent = _root.parent
        if _parent == _root:
            raise RuntimeError(
                f"Could not find project root (with data/ and mcts/) from {Path(__file__)}"
            )
        _root = _parent
    data_dir = _root / "data"

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DataModule
    # ------------------------------------------------------------------
    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    base_model_args = dict(
        hidden_dim=args.hidden_dim,
        inner_dim=args.inner_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        rlrop_factor=args.rlrop_factor,
        rlrop_patience=args.rlrop_patience,
        min_lr=args.min_lr,
        grad_clip_norm=args.grad_clip_norm,
    )
    model = PerturbModule(**base_model_args)

    # ------------------------------------------------------------------
    # Trainer configuration
    # ------------------------------------------------------------------
    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = limit_val = limit_test = args.debug_max_step
        max_steps = args.debug_max_step
        val_check_interval = 1.0
        num_sanity_val_steps = 0
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval
        num_sanity_val_steps = 2

    # Save top-3 checkpoints for multi-checkpoint ensemble
    # In debug mode, save only 1 to avoid overhead
    # NOTE: filename uses 'valf1' (no slash) to ensure valid filesystem names
    save_top_k = args.save_top_k if (args.debug_max_step is None and not fast_dev_run) else 1
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-epoch={epoch:03d}-valf1={val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=save_top_k,
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

    callbacks = [checkpoint_cb, early_stop_cb, lr_monitor, progress_bar]

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

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
        val_check_interval=val_check_interval if (
            args.debug_max_step is None and not fast_dev_run
        ) else 1.0,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=callbacks,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        gradient_clip_val=args.grad_clip_norm,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    if args.test_only:
        # Skip training: use existing checkpoints for ensemble test only.
        # Use single-device trainer to avoid DDP process-group interference.
        print("test-only mode: skipped training, running ensemble test only.")
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if trainer.is_global_zero:
            _run_ensemble_test(
                datamodule=datamodule,
                checkpoint_dir=output_dir / "checkpoints",
                output_dir=output_dir,
                base_model_args=base_model_args,
            )
        return
    else:
        trainer.fit(model, datamodule=datamodule)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    if args.fast_dev_run or args.debug_max_step is not None:
        # Debug mode: use current model state (no checkpoint loading)
        test_results = trainer.test(model, datamodule=datamodule)
        # Save predictions (rank 0 only)
        if trainer.is_global_zero and hasattr(model, "_final_preds"):
            _save_test_predictions(
                pert_ids=model._final_pert_ids,
                symbols=model._final_symbols,
                preds=model._final_preds,
                out_path=output_dir / "test_predictions.tsv",
            )
    else:
        # Production mode: multi-checkpoint ensemble on rank 0 only
        # First sync all ranks to ensure all checkpoints are flushed to disk
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Only rank 0 runs the ensemble (uses single-GPU inference internally)
        if trainer.is_global_zero:
            checkpoint_dir = output_dir / "checkpoints"
            _run_ensemble_test(
                datamodule=datamodule,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                base_model_args=base_model_args,
            )


if __name__ == "__main__":
    main()
