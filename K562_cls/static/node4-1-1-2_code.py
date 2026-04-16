"""Node 4-1-1-2 – scFoundation (top-6 fine-tuned) + STRING_GNN (frozen, cached) + GatedFusion + 2-layer head.

Key improvements over node4-1-1 (parent, test F1=0.3895):
1. Restore 6 scFoundation fine-tuned layers (parent used only 4 — underfitting was the primary bottleneck)
2. Keep STRING_GNN frozen with cached embeddings (retains parent's frozen strategy; eliminates GNN overfitting)
3. Replace CAWR with standard single cosine decay (CAWR warm restarts caused F1 collapse in parent)
4. Replace focal loss + class weights with weighted CE + label smoothing only
   (node4-2 proved this recipe yields F1=0.4801 — focal loss + class weights + smoothing conflicts)
5. Add Mixup augmentation (alpha=0.2) — from node4-2's proven recipe (+0.0172 improvement)
6. Gaussian noise augmentation (std=0.01) on scFoundation embeddings — from sibling node4-1-1-1
7. Lighter head dropout (0.3/0.2 instead of 0.5/0.3) — sibling confirmed 0.5 was excessive
8. Reduce weight_decay to 2e-2 (from 3e-2, matching node4-2's proven setting)
9. Extend training: 300 epochs + patience=30 (sibling was externally terminated at 74/200 epochs)
10. Increase cosine floor: min_lr_ratio=0.12 (from parent's 0.05, allows late-epoch exploration)

Design philosophy: Adopt the proven node4-2 recipe (frozen GNN + weighted CE + Mixup)
while retaining the beneficial 2-layer head and embedding noise augmentation from the
parent lineage. This node is distinct from sibling node4-1-1-1 which used active GNN
+ focal loss + standard cosine (min_lr=0.05).
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
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
HEAD_HIDDEN = 256   # intermediate hidden layer in 2-layer head

SCF_MODEL_DIR = "/home/Models/scFoundation"
GNN_MODEL_DIR = "/home/Models/STRING_GNN"

# Class frequencies: down-regulated, neutral, up-regulated (remapped 0,1,2)
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Inverse-sqrt-frequency class weights for weighted CE loss."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute per-gene macro-averaged F1 score.

    preds:   [N, 3, G] (softmax probabilities)
    targets: [N, G]    (integer class labels in {0, 1, 2})
    """
    assert preds.dim() == 3 and preds.shape[1] == 3, f"Expected preds [N,3,G], got {preds.shape}"
    assert targets.dim() == 2, f"Expected targets [N,G], got {targets.shape}"
    N, C, G = preds.shape
    y_hat = preds.argmax(dim=1)           # [N, G]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)
    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0).float()
        tp = (is_pred & is_true).float().sum(0)
        fp = (is_pred & ~is_true).float().sum(0)
        fn = (~is_pred & is_true).float().sum(0)
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(prec + rec > 0, 2*prec*rec / (prec + rec + 1e-8), torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Loss: Weighted CrossEntropy + Label Smoothing (NO focal component)
# ---------------------------------------------------------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing and per-class weighting.

    Following node4-2's proven recipe: weighted CE + label smoothing (no focal).
    The focal component (used in node4-1-1 and node4-1-1-1) created conflicting
    gradient signals when combined with class weights and label smoothing.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        smoothing: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [N, C], targets: [N] long
        log_probs = F.log_softmax(logits, dim=-1)

        # Soft targets with label smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / self.num_classes)
            smooth_targets.scatter_(
                1, targets.unsqueeze(1),
                1.0 - self.smoothing + self.smoothing / self.num_classes
            )

        # Label-smoothed NLL
        smooth_ce = -(smooth_targets * log_probs).sum(dim=-1)

        # Apply per-sample class weights
        if self.weight is not None:
            sample_weight = self.weight[targets]
            smooth_ce = smooth_ce * sample_weight / self.weight.mean()

        return smooth_ce.mean()


# ---------------------------------------------------------------------------
# Gated Fusion Module
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Learnable gated fusion of scFoundation and STRING_GNN embeddings.

    gate_scf = sigmoid(W_scf * [scf_emb; gnn_emb])
    gate_gnn = sigmoid(W_gnn * [scf_emb; gnn_emb])
    output = gate_scf * proj_scf(scf_emb) + gate_gnn * proj_gnn(gnn_emb)
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

    def forward(self, scf_emb: torch.Tensor, gnn_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([scf_emb, gnn_emb], dim=-1)
        gate_s   = torch.sigmoid(self.gate_scf(combined))
        gate_g   = torch.sigmoid(self.gate_gnn(combined))
        fused    = gate_s * self.proj_scf(scf_emb) + gate_g * self.proj_gnn(gnn_emb)
        return self.dropout(self.layer_norm(fused))


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
    """Collate function that tokenizes pert_ids for scFoundation."""

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]

        # scFoundation: missing genes → 0.0 (not -1.0!)
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")

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
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=make_collate_scf(self.tokenizer),
            persistent_workers=(self.num_workers > 0),
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# LR Scheduler: Linear Warmup + Single Cosine Decay
# ---------------------------------------------------------------------------
class WarmupCosineDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup for warmup_steps epochs, then single cosine decay to min_lr_ratio.

    Using standard single cosine (no warm restarts) — CAWR was proven
    destructive in node4-1-1 (caused F1 to collapse from 0.41 to 0.30).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.12,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = float(step + 1) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay from peak to min_lr_ratio * peak
            progress = (step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            progress = min(progress, 1.0)
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
        return [base_lr * scale for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Mixup Utility
# ---------------------------------------------------------------------------
def mixup_batch(
    scf_emb: torch.Tensor,
    gnn_emb: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup augmentation applied to embedding representations.

    Mixes scF embeddings, GNN embeddings, and label tensors.
    Returns mixed embeddings, mixed labels A, mixed labels B, and lambda.
    """
    if alpha <= 0.0:
        return scf_emb, gnn_emb, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    batch_size = scf_emb.size(0)
    perm = torch.randperm(batch_size, device=scf_emb.device)

    mixed_scf = lam * scf_emb + (1 - lam) * scf_emb[perm]
    mixed_gnn = lam * gnn_emb + (1 - lam) * gnn_emb[perm]
    labels_a  = labels
    labels_b  = labels[perm]
    return mixed_scf, mixed_gnn, labels_a, labels_b, lam


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FusionDEGModel(pl.LightningModule):
    """scFoundation (top-6 layers fine-tuned) + STRING_GNN (frozen, cached) + GatedFusion + 2-layer head.

    Node 4-1-1-2: Adopts node4-2's proven recipe (frozen GNN + weighted CE + Mixup)
    while retaining the parent lineage's 2-layer head architecture and adding
    Gaussian noise augmentation from sibling node4-1-1-1.
    """

    def __init__(
        self,
        scf_finetune_layers: int = 6,
        head_dropout: float      = 0.3,
        head_dropout2: float     = 0.2,
        head_hidden: int         = HEAD_HIDDEN,
        fusion_dropout: float    = 0.2,
        lr: float                = 2e-4,
        weight_decay: float      = 2e-2,
        label_smoothing: float   = 0.1,
        noise_std: float         = 0.01,
        mixup_alpha: float       = 0.2,
        warmup_epochs: int       = 10,
        min_lr_ratio: float      = 0.12,
        max_epochs: int          = 300,
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

        # Freeze all scF parameters, then unfreeze top-k transformer layers
        for param in self.scf.parameters():
            param.requires_grad = False

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
        print(f"[Node4-1-1-2] scFoundation: {scf_train:,}/{scf_total:,} trainable (top-{hp.scf_finetune_layers} layers)")

        # ---- STRING_GNN (FROZEN: fixed PPI feature extractor) ----
        self.gnn = AutoModel.from_pretrained(str(gnn_dir), trust_remote_code=True)
        self.gnn = self.gnn.to(torch.float32)
        for param in self.gnn.parameters():
            param.requires_grad = False
        gnn_params = sum(p.numel() for p in self.gnn.parameters())
        print(f"[Node4-1-1-2] STRING_GNN: FROZEN ({gnn_params:,} params, cached embeddings)")

        # Load graph data and node name→index mapping
        graph_data = torch.load(gnn_dir / "graph_data.pt", map_location="cpu")
        node_names = json.loads((gnn_dir / "node_names.json").read_text())
        self.register_buffer("edge_index", graph_data["edge_index"].long())
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

        # ---- Gated Fusion (fusion_dropout=0.2, lighter than parent's 0.3) ----
        self.fusion = GatedFusion(
            d_scf=SCF_HIDDEN,
            d_gnn=GNN_HIDDEN,
            d_out=FUSION_DIM,
            fusion_dropout=hp.fusion_dropout,
        )

        # ---- Classification head: 2-layer MLP (512 → 256 → 19,920) ----
        # Lighter dropouts (0.3/0.2) vs parent's (0.5/0.3):
        # Combined with weighted CE (no focal), these dropouts provide better
        # gradient signal for the minority DEG classes.
        self.head = nn.Sequential(
            nn.Dropout(hp.head_dropout),
            nn.Linear(FUSION_DIM, hp.head_hidden),
            nn.BatchNorm1d(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout2),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # Cast head and fusion params to float32
        for param in self.fusion.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        for param in self.head.parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ---- Loss: Weighted CE + Label Smoothing (no focal) ----
        self.register_buffer("class_weights", get_class_weights())
        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=hp.label_smoothing,
            num_classes=N_CLASSES,
        )

        # Accumulators
        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []

        # Cache for frozen GNN node embeddings
        self._gnn_node_embs_cache: Optional[torch.Tensor] = None

    def _get_gnn_indices(self, pert_ids: List[str], device: torch.device) -> torch.Tensor:
        """Return LongTensor of node indices (0 for unknowns)."""
        indices = [self._ensembl_to_idx.get(pid, 0) for pid in pert_ids]
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _get_frozen_gnn_embs(self, device: torch.device) -> torch.Tensor:
        """Get frozen STRING_GNN node embeddings. Cached after first call."""
        if self._gnn_node_embs_cache is None or self._gnn_node_embs_cache.device != device:
            with torch.no_grad():
                ew = self.edge_weight.to(device) if self.edge_weight is not None else None
                gnn_out = self.gnn(
                    edge_index  = self.edge_index.to(device),
                    edge_weight = ew,
                )
                self._gnn_node_embs_cache = gnn_out.last_hidden_state.detach()
        return self._gnn_node_embs_cache

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids: List[str],
        scf_noise_std: float = 0.0,
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        device = input_ids.device

        # 1. scFoundation: mean pool → [B, 768] in float32
        scf_out = self.scf(input_ids=input_ids, attention_mask=attention_mask)
        scf_emb = scf_out.last_hidden_state.float().mean(dim=1)

        # 2. Optional Gaussian noise on scFoundation embedding (training only)
        if scf_noise_std > 0.0 and self.training:
            noise = torch.randn_like(scf_emb) * scf_noise_std
            scf_emb = scf_emb + noise

        # 3. STRING_GNN: frozen, cached node embeddings → [B, 256]
        node_embs    = self._get_frozen_gnn_embs(device)
        node_indices = self._get_gnn_indices(pert_ids, device)
        gnn_emb      = node_embs[node_indices]

        # 4. Gated fusion → [B, 512]
        fused = self.fusion(scf_emb, gnn_emb)

        # 5. 2-layer classification head → [B, 3, G]
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return logits

    def _forward_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (scf_emb, gnn_emb) before fusion — used for Mixup."""
        device = input_ids.device
        scf_out  = self.scf(input_ids=input_ids, attention_mask=attention_mask)
        scf_emb  = scf_out.last_hidden_state.float().mean(dim=1)

        # Gaussian noise during training
        if self.training and self.hparams.noise_std > 0.0:
            scf_emb = scf_emb + torch.randn_like(scf_emb) * self.hparams.noise_std

        node_embs    = self._get_frozen_gnn_embs(device)
        node_indices = self._get_gnn_indices(pert_ids, device)
        gnn_emb      = node_embs[node_indices]
        return scf_emb, gnn_emb

    def _fuse_and_classify(
        self, scf_emb: torch.Tensor, gnn_emb: torch.Tensor
    ) -> torch.Tensor:
        """Apply fusion + head to pre-computed embeddings."""
        B = scf_emb.size(0)
        fused  = self.fusion(scf_emb, gnn_emb)
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return logits

    def _loss(
        self, logits: torch.Tensor, targets: torch.Tensor, lam: float = 1.0,
        targets_b: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, C, G = logits.shape
        self.criterion.weight = self.class_weights
        flat_logits = logits.permute(0, 2, 1).reshape(-1, C)
        flat_tgts   = targets.reshape(-1)

        if targets_b is not None and lam < 1.0:
            flat_tgts_b = targets_b.reshape(-1)
            loss = lam * self.criterion(flat_logits, flat_tgts) + \
                   (1 - lam) * self.criterion(flat_logits, flat_tgts_b)
        else:
            loss = self.criterion(flat_logits, flat_tgts)
        return loss

    # ---- steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        hp = self.hparams

        # Extract embeddings first (for Mixup)
        scf_emb, gnn_emb = self._forward_embeddings(
            batch["input_ids"], batch["attention_mask"], batch["pert_id"]
        )

        labels = batch["labels"]

        # Mixup augmentation in embedding space
        if hp.mixup_alpha > 0.0 and self.training:
            scf_m, gnn_m, labels_a, labels_b, lam = mixup_batch(
                scf_emb, gnn_emb, labels, alpha=hp.mixup_alpha
            )
        else:
            scf_m, gnn_m, labels_a, labels_b, lam = scf_emb, gnn_emb, labels, None, 1.0

        logits = self._fuse_and_classify(scf_m, gnn_m)
        loss   = self._loss(logits, labels_a, lam=lam, targets_b=labels_b)

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

        # Sort by sample index for deduplication, then restore original order
        order     = torch.argsort(all_idx)
        s_idx     = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask      = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        uniq_idx  = s_idx[mask]
        uniq_pred = s_pred[mask]
        uniq_tgt  = s_tgt[mask]
        orig_order = torch.argsort(uniq_idx)
        f1 = compute_per_gene_f1(uniq_pred[orig_order], uniq_tgt[orig_order])
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
        all_preds   = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)

        all_pert_ids: List[List[str]] = [None] * self.trainer.world_size
        all_symbols:  List[List[str]] = [None] * self.trainer.world_size
        torch.distributed.all_gather_object(all_pert_ids, self._test_pert_ids)
        torch.distributed.all_gather_object(all_symbols,  self._test_symbols)

        if self.trainer.is_global_zero:
            flat_pids = [p for rank_pids in all_pert_ids for p in rank_pids]
            flat_syms = [s for rank_syms in all_symbols  for s in rank_syms]
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
            print(f"[Node4-1-1-2] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    # ---- checkpoint: save only trainable params + buffers ----
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
        self.print(f"Checkpoint: {trained:,}/{total:,} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        # Convert to bfloat16 so encoder params match model dtype on load
        state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=False)

    # ---- optimizer ----
    def configure_optimizers(self):
        hp = self.hparams

        # GNN is frozen — no parameter group for it
        scf_params  = [p for p in self.scf.parameters()    if p.requires_grad]
        head_params = (
            list(self.fusion.parameters()) +
            list(self.head.parameters())
        )

        param_groups = [
            {"params": scf_params,  "lr": hp.lr, "name": "scf"},
            {"params": head_params, "lr": hp.lr, "name": "head"},
        ]

        opt = torch.optim.AdamW(param_groups, weight_decay=hp.weight_decay)

        # Standard single cosine decay (no warm restarts — CAWR was destructive)
        sch = WarmupCosineDecayScheduler(
            opt,
            warmup_steps  = hp.warmup_epochs,
            total_steps   = hp.max_epochs,
            min_lr_ratio  = hp.min_lr_ratio,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval":  "epoch",
                "frequency": 1,
                "monitor":   "val/f1",
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node4-1-1-2 – scFoundation (top-6) + STRING_GNN (frozen) + Mixup + weighted CE"
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=8)
    parser.add_argument("--global-batch-size",   type=int,   default=64)
    parser.add_argument("--max-epochs",          type=int,   default=300)
    parser.add_argument("--lr",                  type=float, default=2e-4)
    parser.add_argument("--weight-decay",        type=float, default=2e-2)
    parser.add_argument("--scf-finetune-layers", type=int,   default=6,
                        dest="scf_finetune_layers")
    parser.add_argument("--head-dropout",        type=float, default=0.3,
                        dest="head_dropout")
    parser.add_argument("--head-dropout2",       type=float, default=0.2,
                        dest="head_dropout2")
    parser.add_argument("--head-hidden",         type=int,   default=HEAD_HIDDEN,
                        dest="head_hidden")
    parser.add_argument("--fusion-dropout",      type=float, default=0.2,
                        dest="fusion_dropout")
    parser.add_argument("--label-smoothing",     type=float, default=0.1,
                        dest="label_smoothing")
    parser.add_argument("--noise-std",           type=float, default=0.01,
                        dest="noise_std")
    parser.add_argument("--mixup-alpha",         type=float, default=0.2,
                        dest="mixup_alpha")
    parser.add_argument("--warmup-epochs",       type=int,   default=10,
                        dest="warmup_epochs")
    parser.add_argument("--min-lr-ratio",        type=float, default=0.12,
                        dest="min_lr_ratio")
    parser.add_argument("--patience",            type=int,   default=30)
    parser.add_argument("--num-workers",         type=int,   default=4)
    parser.add_argument("--debug-max-step",      type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",        action="store_true",
                        dest="fast_dev_run")
    parser.add_argument("--val-check-interval",  type=float, default=1.0,
                        dest="val_check_interval")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = lim_val = lim_test = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = FusionDEGModel(
        scf_finetune_layers = args.scf_finetune_layers,
        head_dropout        = args.head_dropout,
        head_dropout2       = args.head_dropout2,
        head_hidden         = args.head_hidden,
        fusion_dropout      = args.fusion_dropout,
        lr                  = args.lr,
        weight_decay        = args.weight_decay,
        label_smoothing     = args.label_smoothing,
        noise_std           = args.noise_std,
        mixup_alpha         = args.mixup_alpha,
        warmup_epochs       = args.warmup_epochs,
        min_lr_ratio        = args.min_lr_ratio,
        max_epochs          = args.max_epochs,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
        auto_insert_metric_name=False,
    )
    es_cb = EarlyStopping(
        monitor="val/f1", mode="max",
        patience=args.patience, min_delta=1e-4,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(
        save_dir=str(output_dir / "logs"), name="csv_logs"
    )
    tb_logger = TensorBoardLogger(
        save_dir=str(output_dir / "logs"), name="tensorboard_logs"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        limit_train_batches=lim_train,
        limit_val_batches=lim_val,
        limit_test_batches=lim_test,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pg_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=dm)

    if fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=dm)
    else:
        test_results = trainer.test(model, datamodule=dm, ckpt_path="best")

    print(f"[Node4-1-1-2] Test results: {test_results}")


if __name__ == "__main__":
    main()
