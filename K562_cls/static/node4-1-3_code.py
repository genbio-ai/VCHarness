"""Node 4-1-3 – scFoundation (top-6 layers fine-tuned) + STRING_GNN (discriminative lr)
+ Gated Fusion + Wider 2-Layer Head (512→1024→19920) + Raised Cosine Floor.

This node builds directly on the proven parent (node4-1, test F1=0.4629), which is the best
performer in the entire MCTS tree, and applies only orthogonal improvements that avoid the
failure modes of both sibling nodes (node4-1-1: underfitting from frozen GNN + CAWR;
node4-1-2: train/test mismatch from embedding noise + narrow bottleneck).

Key changes from parent (node4-1):
1. Wider 2-layer head: 512 → 1024 → 19920 (vs single 512→19920 in parent)
   - Adds beneficial nonlinearity without the narrow bottleneck failure from sibling-2 (512→256)
   - Keeps parameter count comparable (~20M vs parent's ~10M) while allowing richer features
   - Uses LayerNorm (not BatchNorm) to avoid instability with small per-GPU batch sizes
2. Raised cosine floor: min_lr_ratio 0.05 → 0.12
   - Parent's best checkpoint was at epoch 80 when LR was still ~20% of peak
   - After LR dropped below 2e-5 (5% of peak), no further improvement was observed
   - Raising to 12% maintains productive exploration capacity in later epochs
3. Extended training: max_epochs 150 → 200, patience 20 → 25
   - Allows more convergence time in the high-LR regime before final decay phase
4. Slightly reduced focal_gamma: 2.0 → 1.5
   - Reduces the focal penalty's aggressiveness on hard examples
   - May allow better utilization of the minority DEG class signal

Preserved from parent (unchanged):
- scFoundation: top-6 of 12 layers fine-tuned, gradient checkpointing
- STRING_GNN: full fine-tuning with discriminative LR (lr/5 = 4e-5)
- GatedFusion architecture with fusion_dropout=0.2
- head_dropout=0.5
- weight_decay=2e-2
- label_smoothing=0.1
- WarmupCosine schedule (no CAWR — proven harmful for this task)
- 10-epoch linear warmup
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES    = 6640
N_CLASSES  = 3
SCF_HIDDEN = 768    # scFoundation hidden size
GNN_HIDDEN = 256    # STRING_GNN hidden size
FUSION_DIM = 512    # output dimension of gated fusion

SCF_MODEL_DIR = "/home/Models/scFoundation"
GNN_MODEL_DIR = "/home/Models/STRING_GNN"

CLASS_FREQ = [0.0429, 0.9251, 0.0320]  # down, neutral, up (remapped 0,1,2)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    y_hat       = preds.argmax(dim=1)
    G           = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)
    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0).float()
        tp  = (is_pred & is_true).float().sum(0)
        fp  = (is_pred & ~is_true).float().sum(0)
        fn  = (~is_pred & is_true).float().sum(0)
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(prec + rec > 0, 2*prec*rec/(prec+rec+1e-8), torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


class LabelSmoothingFocalLoss(nn.Module):
    """Focal loss with label smoothing.

    Combines focal loss (to address class imbalance) with label smoothing
    (to reduce overconfidence on the dominant neutral class). The label
    smoothing creates soft targets: epsilon/K for each class, with
    (1 - epsilon) * one_hot for the true class.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 1.5,
        smoothing: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: [N] long, logits: [N, C]
        log_probs = F.log_softmax(logits, dim=-1)
        # Hard cross-entropy for focal weighting
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal_weight = (1 - pt) ** self.gamma

        # Label-smoothed NLL: smooth targets to avoid overconfidence
        # Soft cross-entropy: -(1-eps)*log_p(y) - eps/K * sum(log_p)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / self.num_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / self.num_classes)

        smooth_ce = -(smooth_targets * log_probs).sum(dim=-1)

        # Apply focal weighting to the smoothed loss
        loss = focal_weight * smooth_ce

        # Apply class weight scaling (weight[targets] to scale per-sample)
        if self.weight is not None:
            sample_weight = self.weight[targets]
            loss = loss * sample_weight / self.weight.mean()

        return loss.mean()


# ---------------------------------------------------------------------------
# Gated Fusion Module
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Learnable gated fusion of two heterogeneous embeddings.

    output = gate_scf * proj_scf(scf_emb) + gate_gnn * proj_gnn(gnn_emb)

    Includes fusion dropout after LayerNorm to combat overfitting.
    """

    def __init__(
        self,
        d_scf: int = SCF_HIDDEN,
        d_gnn: int = GNN_HIDDEN,
        d_out: int = FUSION_DIM,
        fusion_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        d_in = d_scf + d_gnn
        self.proj_scf   = nn.Linear(d_scf, d_out)
        self.proj_gnn   = nn.Linear(d_gnn, d_out)
        self.gate_scf   = nn.Linear(d_in,  d_out)
        self.gate_gnn   = nn.Linear(d_in,  d_out)
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout    = nn.Dropout(fusion_dropout)

    def forward(
        self, scf_emb: torch.Tensor, gnn_emb: torch.Tensor
    ) -> torch.Tensor:
        combined  = torch.cat([scf_emb, gnn_emb], dim=-1)     # [B, d_scf+d_gnn]
        gate_s    = torch.sigmoid(self.gate_scf(combined))     # [B, d_out]
        gate_g    = torch.sigmoid(self.gate_gnn(combined))     # [B, d_out]
        fused     = gate_s * self.proj_scf(scf_emb) + gate_g * self.proj_gnn(gnn_emb)
        return self.dropout(self.layer_norm(fused))             # [B, d_out]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List] = (
            [torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
             for row in df["label"].tolist()]
            if has_label else None
        )

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "sample_idx": idx,
            "pert_id":    self.pert_ids[idx],
            "symbol":     self.symbols[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_collate_scf(tokenizer):
    """Collate function that tokenizes for scFoundation."""

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]

        # scFoundation: missing genes → 0.0 (not -1.0!)
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")  # [B, 19264] float32

        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out

    return collate_fn


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer   = None

    def setup(self, stage: Optional[str] = None) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(SCF_MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(SCF_MODEL_DIR, trust_remote_code=True)

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        self.train_ds = DEGDataset(train_df)
        self.val_ds   = DEGDataset(val_df)
        self.test_ds  = DEGDataset(test_df)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=make_collate_scf(self.tokenizer))

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# LR Scheduler: Linear Warmup + Cosine Annealing (single cycle, higher floor)
# ---------------------------------------------------------------------------
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup for warmup_steps steps, then cosine decay to min_lr.

    This is a single-cycle schedule (no warm restarts — CAWR was shown to be
    harmful in sibling node4-1-1). The min_lr_ratio is raised to 0.12 from the
    parent's 0.05 to maintain productive exploration in later epochs.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.12,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps   = warmup_steps
        self.total_steps    = total_steps
        self.min_lr_ratio   = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = float(step + 1) / float(max(1, self.warmup_steps))
        else:
            progress = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
        return [base_lr * scale for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FusionDEGModel(pl.LightningModule):
    """scFoundation (top-6 layers fine-tuned) + STRING_GNN (discriminative lr)
    + Gated Fusion + Wider 2-Layer Head (512→1024→19920).

    Key improvements over node4-1:
    - Wider 2-layer head: 512→1024→19920 avoids the narrow bottleneck failure
      of sibling node4-1-2 (which used 512→256→19920) while adding beneficial
      nonlinearity to the classification head
    - Raised cosine floor (min_lr_ratio=0.12) for better late-epoch exploration
    - Extended training (max_epochs=200, patience=25) for more convergence time
    - Slightly reduced focal_gamma (1.5 vs 2.0) to allow better minority class learning
    """

    def __init__(
        self,
        scf_finetune_layers: int = 6,
        head_dropout: float      = 0.5,
        head_hidden: int         = 1024,   # wider intermediate dimension
        fusion_dropout: float    = 0.2,
        lr: float                = 2e-4,
        gnn_lr_ratio: float      = 5.0,   # GNN branch lr = lr / gnn_lr_ratio
        weight_decay: float      = 2e-2,
        label_smoothing: float   = 0.1,
        focal_gamma: float       = 1.5,   # reduced from 2.0
        warmup_epochs: int       = 10,
        max_epochs: int          = 200,
        min_lr_ratio: float      = 0.12,  # raised from 0.05
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        gnn_dir = Path(GNN_MODEL_DIR)

        # ---- scFoundation backbone ----
        self.scf = AutoModel.from_pretrained(
            SCF_MODEL_DIR,
            trust_remote_code=True,
            _use_flash_attention_2=True,
        )
        self.scf = self.scf.to(torch.bfloat16)
        self.scf.config.use_cache = False
        self.scf.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Freeze all scF params, then unfreeze top-k transformer layers
        for param in self.scf.parameters():
            param.requires_grad = False

        # scFoundation encoder layers: self.scf.encoder.transformer_encoder (ModuleList, 12 layers)
        n_scf_layers = len(self.scf.encoder.transformer_encoder)
        for i in range(n_scf_layers - hp.scf_finetune_layers, n_scf_layers):
            for param in self.scf.encoder.transformer_encoder[i].parameters():
                param.requires_grad = True
        # Also unfreeze the final LayerNorm
        for param in self.scf.encoder.norm.parameters():
            param.requires_grad = True

        # Cast unfrozen scF params to float32 for stable optimization
        for name, param in self.scf.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        scf_train = sum(p.numel() for p in self.scf.parameters() if p.requires_grad)
        scf_total = sum(p.numel() for p in self.scf.parameters())
        print(f"[Node4-1-3] scFoundation: {scf_train:,}/{scf_total:,} trainable")

        # ---- STRING_GNN ---- (full fine-tuning but with discriminative lower LR)
        self.gnn = AutoModel.from_pretrained(str(gnn_dir), trust_remote_code=True)
        self.gnn = self.gnn.to(torch.float32)

        # Load graph data and node name→index mapping
        graph_data   = torch.load(gnn_dir / "graph_data.pt", map_location="cpu")
        node_names   = json.loads((gnn_dir / "node_names.json").read_text())
        self.register_buffer(
            "edge_index",  graph_data["edge_index"].long()
        )
        edge_weight = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            self.register_buffer("edge_weight", edge_weight.float())
        else:
            self.edge_weight = None

        # Build Ensembl ID → node index lookup
        self._ensembl_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(node_names)
        }
        self._n_nodes = len(node_names)

        # ---- Gated Fusion (with dropout) ----
        self.fusion = GatedFusion(
            d_scf=SCF_HIDDEN,
            d_gnn=GNN_HIDDEN,
            d_out=FUSION_DIM,
            fusion_dropout=hp.fusion_dropout,
        )

        # ---- Classification head: wider 2-layer (512→1024→19920) ----
        # This avoids the narrow bottleneck failure seen in node4-1-2 (512→256→19920)
        # while adding beneficial nonlinearity to the output mapping.
        # LayerNorm (not BatchNorm) avoids instability with small per-GPU batch sizes.
        self.head = nn.Sequential(
            nn.Dropout(hp.head_dropout),
            nn.Linear(FUSION_DIM, hp.head_hidden),   # 512 → 1024
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(0.3),                          # lighter second dropout
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),  # 1024 → 19920
        )

        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"[Node4-1-3] Head: {head_params:,} params (512→{hp.head_hidden}→19920)")

        # ---- Loss: Label Smoothing + Focal ----
        self.register_buffer("class_weights", get_class_weights())
        self.criterion = LabelSmoothingFocalLoss(
            gamma=hp.focal_gamma,
            smoothing=hp.label_smoothing,
            num_classes=N_CLASSES,
        )

        # Accumulators
        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds:  List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []

    # ---- look up GNN node indices for a batch of pert_ids ----
    def _get_gnn_indices(self, pert_ids: List[str], device: torch.device) -> torch.Tensor:
        """Return LongTensor of node indices, 0 for unknowns."""
        indices = [self._ensembl_to_idx.get(pid, 0) for pid in pert_ids]
        return torch.tensor(indices, dtype=torch.long, device=device)

    # ---- forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids: List[str],
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        device = input_ids.device

        # 1. scFoundation: [B, nnz+2, 768] → mean pool → [B, 768]
        scf_out  = self.scf(input_ids=input_ids, attention_mask=attention_mask)
        scf_emb  = scf_out.last_hidden_state.float().mean(dim=1)   # [B, 768]

        # 2. STRING_GNN: run full graph → look up perturbed gene embeddings
        ew = self.edge_weight.to(device) if self.edge_weight is not None else None
        gnn_out  = self.gnn(
            edge_index  = self.edge_index.to(device),
            edge_weight = ew,
        )
        node_embs   = gnn_out.last_hidden_state                     # [N_nodes, 256]
        node_indices = self._get_gnn_indices(pert_ids, device)      # [B]
        gnn_emb     = node_embs[node_indices]                       # [B, 256]

        # 3. Gated fusion (with dropout) → [B, 512]
        fused = self.fusion(scf_emb, gnn_emb)

        # 4. Classification head (wider 2-layer) → [B, 3, G]
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return logits

    # ---- loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        self.criterion.weight = self.class_weights
        return self.criterion(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
        )

    # ---- steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits.float(), dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        local_idx   = torch.cat(self._val_idx,   0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "labels" in batch:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)

        # Only use all_gather in DDP mode (n_gpus > 1)
        if self.trainer.world_size > 1:
            all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
            # Collect pert_ids from all ranks
            all_pert_ids: List[List[str]] = [None] * self.trainer.world_size
            all_symbols: List[List[str]] = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(all_pert_ids, self._test_pert_ids)
            torch.distributed.all_gather_object(all_symbols, self._test_symbols)
            flat_pids  = [p for rank_pids in all_pert_ids  for p in rank_pids]
            flat_syms  = [s for rank_syms in all_symbols for s in rank_syms]
        else:
            all_preds = local_preds
            flat_pids = self._test_pert_ids
            flat_syms = self._test_symbols

        if self.trainer.is_global_zero:
            n = all_preds.shape[0]
            rows = []
            for i in range(n):
                rows.append({
                    "idx":        flat_pids[i],
                    "input":      flat_syms[i],
                    "prediction": json.dumps(all_preds[i].float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node4-1-3] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    # ---- checkpoint: save only trainable params ----
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
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {trained}/{total} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- optimizer: discriminative LRs ----
    def configure_optimizers(self):
        hp = self.hparams
        scf_lr  = hp.lr
        gnn_lr  = hp.lr / hp.gnn_lr_ratio   # lower LR for GNN to reduce overfitting

        # Separate parameter groups: GNN branch at lower LR
        scf_params  = [p for p in self.scf.parameters() if p.requires_grad]
        gnn_params  = list(self.gnn.parameters())
        head_params = (
            list(self.fusion.parameters()) +
            list(self.head.parameters())
        )

        param_groups = [
            {"params": scf_params,  "lr": scf_lr,  "name": "scf"},
            {"params": gnn_params,  "lr": gnn_lr,  "name": "gnn"},
            {"params": head_params, "lr": scf_lr,  "name": "head"},
        ]

        opt = torch.optim.AdamW(
            param_groups,
            weight_decay=hp.weight_decay,
        )

        # Warmup cosine schedule based on epoch-level steps
        # Single cycle (no CAWR — proven harmful in sibling node4-1-1)
        # min_lr_ratio raised to 0.12 for better late-epoch exploration
        total_epochs  = hp.max_epochs
        warmup_epochs = hp.warmup_epochs

        sch = WarmupCosineScheduler(
            opt,
            warmup_steps=warmup_epochs,
            total_steps=total_epochs,
            min_lr_ratio=hp.min_lr_ratio,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/f1",
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(description="Node4-1-3 – scFoundation + STRING_GNN + Gated Fusion + Wider 2-Layer Head")
    parser.add_argument("--micro-batch-size",     type=int,   default=8)
    parser.add_argument("--global-batch-size",    type=int,   default=64)
    parser.add_argument("--max-epochs",           type=int,   default=200)
    parser.add_argument("--lr",                   type=float, default=2e-4)
    parser.add_argument("--weight-decay",         type=float, default=2e-2)
    parser.add_argument("--scf-finetune-layers",  type=int,   default=6,
                        dest="scf_finetune_layers")
    parser.add_argument("--head-dropout",         type=float, default=0.5)
    parser.add_argument("--head-hidden",          type=int,   default=1024,
                        dest="head_hidden")
    parser.add_argument("--fusion-dropout",       type=float, default=0.2,
                        dest="fusion_dropout")
    parser.add_argument("--gnn-lr-ratio",         type=float, default=5.0,
                        dest="gnn_lr_ratio")
    parser.add_argument("--label-smoothing",      type=float, default=0.1,
                        dest="label_smoothing")
    parser.add_argument("--focal-gamma",          type=float, default=1.5,
                        dest="focal_gamma")
    parser.add_argument("--warmup-epochs",        type=int,   default=10,
                        dest="warmup_epochs")
    parser.add_argument("--min-lr-ratio",         type=float, default=0.12,
                        dest="min_lr_ratio")
    parser.add_argument("--num-workers",          type=int,   default=4)
    parser.add_argument("--debug-max-step",       type=int,   default=None, dest="debug_max_step")
    parser.add_argument("--fast-dev-run",         action="store_true", dest="fast_dev_run")
    parser.add_argument("--val-check-interval",   type=float, default=1.0, dest="val_check_interval")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        # --debug_max_step limits training steps globally while processing
        # full batches per epoch for val/test. Use lim_train = debug_max_step
        # (batches per epoch), full val/test (1.0 = all samples).
        lim_train = args.debug_max_step
        lim_val   = 1.0   # full validation set per epoch
        lim_test  = 1.0   # full test set for predictions
        max_steps = args.debug_max_step
        check_val_every_n_epoch = 1
    else:
        lim_train = 1.0
        lim_val   = 1.0
        lim_test  = 1.0
        max_steps = -1
        check_val_every_n_epoch = 1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = FusionDEGModel(
        scf_finetune_layers = args.scf_finetune_layers,
        head_dropout        = args.head_dropout,
        head_hidden         = args.head_hidden,
        fusion_dropout      = args.fusion_dropout,
        lr                  = args.lr,
        gnn_lr_ratio        = args.gnn_lr_ratio,
        weight_decay        = args.weight_decay,
        label_smoothing     = args.label_smoothing,
        focal_gamma         = args.focal_gamma,
        warmup_epochs       = args.warmup_epochs,
        max_epochs          = args.max_epochs,
        min_lr_ratio        = args.min_lr_ratio,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
    )
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=25, min_delta=1e-4)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    strategy = (
        DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120))
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
        val_check_interval      = args.val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        check_val_every_n_epoch = check_val_every_n_epoch,
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

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node4-1-3] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
