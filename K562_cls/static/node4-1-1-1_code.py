"""Node 1-2 – scFoundation (top-6 layers fine-tuned) + STRING_GNN (discriminative LR) + 2-Layer Head + Noise Augmentation.

Improvements over node4-1-1 (test F1=0.3895) — reverting to node4-1's well-performing base:
1. Restore scFoundation fine-tuned layers: 4 → 6 (recovers model capacity)
   Rationale: 4 layers + frozen GNN left model underfitting; node4-1 used 6 layers successfully
2. Restore STRING_GNN discriminative LR (lr×0.2 = 4e-5) — no longer frozen
   Rationale: discriminative LR (not full LR) already prevents overfitting; freezing was too aggressive
3. Retain 2-layer classification head (512→256→19,920 with BN+GELU+Dropout)
   Rationale: confirmed improvement over single 10.2M-param linear head; keeps memorization low
4. Revert to standard cosine decay (no CAWR warm restarts)
   Rationale: CAWR warm restarts caused destructive LR jumps and never recovered (F1 0.41→0.30)
5. Reduce weight_decay: 3e-2 → 1.5e-2 (lighter regularization for restored capacity)
6. Reduce focal_gamma: 2.0 → 1.5 (less aggressive focus; gamma=2.0 + class_weights was too strong)
7. Reduce head_dropout: 0.5/0.3 → 0.3/0.2 (lighter dropout to match restored model capacity)
8. Add Gaussian noise augmentation on scFoundation embeddings during training (std=0.01)
   Rationale: regularizes 1,388-sample dataset in embedding space; prevents embedding memorization
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
HEAD_HIDDEN = 256   # intermediate hidden layer in 2-layer head

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
    """Compute per-gene macro-averaged F1 score matching calc_metric.py logic.

    preds: [N, 3, G] (softmax probabilities)
    targets: [N, G] (class labels per gene, in {0,1,2})
    """
    assert preds.dim() == 3 and preds.shape[1] == 3, f"Expected preds [N,3,G], got {preds.shape}"
    assert targets.dim() == 2, f"Expected targets [N,G], got {targets.shape}"
    N, C, G = preds.shape
    y_hat = preds.argmax(dim=1)            # [N, G] predicted class per sample per gene
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)
    for c in range(3):
        is_true = (targets == c)            # [N, G]
        is_pred = (y_hat == c)              # [N, G]
        present = is_true.any(dim=0).float()  # [G]
        tp = is_pred & is_true              # [N, G]
        fp = is_pred & ~is_true             # [N, G]
        fn = ~is_pred & is_true             # [N, G]
        tp = tp.float().sum(0)              # [G]
        fp = fp.float().sum(0)              # [G]
        fn = fn.float().sum(0)              # [G]
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(prec + rec > 0, 2*prec*rec/(prec+rec+1e-8), torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


class LabelSmoothingFocalLoss(nn.Module):
    """Focal loss with label smoothing.

    Combines focal loss (to address class imbalance) with label smoothing
    (to reduce overconfidence). The label smoothing creates soft targets:
    epsilon/K for each class, with (1 - epsilon) * one_hot for the true class.

    Node1-2 change: focal_gamma reduced from 2.0 → 1.5 to reduce the
    aggressiveness of focus weighting when combined with class weights.
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
# LR Scheduler: Linear Warmup + Standard Cosine Decay (no warm restarts)
# ---------------------------------------------------------------------------
class WarmupCosineDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup for warmup_steps epochs, then standard cosine annealing decay.

    After warmup, decays from base_lr to min_lr_ratio * base_lr over
    (total_steps - warmup_steps) epochs. No warm restarts (CAWR was harmful).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.05,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps  = warmup_steps
        self.total_steps   = total_steps
        self.min_lr_ratio  = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = float(step + 1) / float(max(1, self.warmup_steps))
        else:
            # Standard cosine decay from warmup end to total steps
            progress = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            # Clamp progress to [0, 1]
            progress = min(progress, 1.0)
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
        return [base_lr * scale for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FusionDEGModel(pl.LightningModule):
    """scFoundation (top-6 layers fine-tuned) + STRING_GNN (discriminative LR) + Gated Fusion + 2-layer head.

    Key improvements over node4-1-1:
    - scFoundation fine-tuned layers: 4 → 6 (restores node4-1's capacity)
    - STRING_GNN restored to discriminative LR (lr×gnn_lr_ratio) — no longer frozen
    - 2-layer head (512→256→19,920) retained from node4-1-1 (reduces memorization)
    - Standard cosine decay replacing CAWR (eliminates destructive LR jumps)
    - weight_decay reduced from 3e-2 → 1.5e-2 (lighter regularization)
    - focal_gamma reduced from 2.0 → 1.5 (less aggressive focus with class weights)
    - head_dropout reduced from 0.5/0.3 → 0.3/0.2 (proportional to restored capacity)
    - Gaussian noise augmentation on scFoundation embeddings (std=0.01) for data regularization
    """

    def __init__(
        self,
        scf_finetune_layers: int = 6,
        head_dropout: float      = 0.3,
        head_dropout2: float     = 0.2,
        head_hidden: int         = HEAD_HIDDEN,
        fusion_dropout: float    = 0.2,
        gnn_lr_ratio: float      = 0.2,    # GNN LR = lr * gnn_lr_ratio (discriminative LR)
        lr: float                = 2e-4,
        weight_decay: float      = 1.5e-2,
        label_smoothing: float   = 0.1,
        focal_gamma: float       = 1.5,
        warmup_epochs: int       = 10,
        min_lr_ratio: float      = 0.05,
        max_epochs: int          = 200,
        noise_std: float         = 0.01,   # Gaussian noise std on scF embeddings (training only)
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
        print(f"[Node1-2] scFoundation: {scf_train:,}/{scf_total:,} trainable")

        # ---- STRING_GNN ---- (discriminative LR: gnn_lr = lr * gnn_lr_ratio)
        self.gnn = AutoModel.from_pretrained(str(gnn_dir), trust_remote_code=True)
        self.gnn = self.gnn.to(torch.float32)
        # Allow all GNN parameters to train (with discriminative lower LR via param groups)
        for param in self.gnn.parameters():
            param.requires_grad = True
        gnn_train = sum(p.numel() for p in self.gnn.parameters() if p.requires_grad)
        gnn_total = sum(p.numel() for p in self.gnn.parameters())
        gnn_lr = hp.lr * hp.gnn_lr_ratio
        print(f"[Node1-2] STRING_GNN: {gnn_train:,}/{gnn_total:,} trainable at LR={gnn_lr:.2e}")

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

        # ---- Gated Fusion (fusion_dropout=0.2, back to node4-1 level) ----
        self.fusion = GatedFusion(
            d_scf=SCF_HIDDEN,
            d_gnn=GNN_HIDDEN,
            d_out=FUSION_DIM,
            fusion_dropout=hp.fusion_dropout,
        )

        # ---- Classification head: 2-layer MLP (512 → 256 → 19,920) ----
        # Retained from node4-1-1: reduces head memorization capacity from 10.2M to 1.4M params
        # Node1-2 change: lighter dropout (0.3/0.2 vs 0.5/0.3) to match restored model capacity
        self.head = nn.Sequential(
            nn.Dropout(hp.head_dropout),
            nn.Linear(FUSION_DIM, hp.head_hidden),
            nn.BatchNorm1d(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout2),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # Cast head and fusion params to float32 for stable optimization
        for name, param in self.fusion.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()
        for name, param in self.head.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ---- Loss: Label Smoothing + Focal (gamma=1.5, reduced from 2.0) ----
        self.register_buffer("class_weights", get_class_weights())
        self.criterion = LabelSmoothingFocalLoss(
            weight=self.class_weights,
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

    def _run_gnn(self, device: torch.device) -> torch.Tensor:
        """Run GNN forward pass and return node embeddings [N_nodes, 256]."""
        ew = self.edge_weight.to(device) if self.edge_weight is not None else None
        gnn_out = self.gnn(
            edge_index  = self.edge_index.to(device),
            edge_weight = ew,
        )
        return gnn_out.last_hidden_state  # [N_nodes, 256]

    # ---- forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids: List[str],
        add_noise: bool = False,
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        device = input_ids.device

        # 1. scFoundation: [B, nnz+2, 768] → mean pool → [B, 768]
        scf_out  = self.scf(input_ids=input_ids, attention_mask=attention_mask)
        scf_emb  = scf_out.last_hidden_state.float().mean(dim=1)   # [B, 768]

        # Gaussian noise augmentation on scFoundation embeddings during training
        # Regularizes embedding-level memorization on the small 1,388-sample dataset
        if add_noise and self.hparams.noise_std > 0.0:
            noise = torch.randn_like(scf_emb) * self.hparams.noise_std
            scf_emb = scf_emb + noise

        # 2. STRING_GNN: full graph propagation (with gradients, discriminative LR)
        node_embs    = self._run_gnn(device)                        # [N_nodes, 256]
        node_indices = self._get_gnn_indices(pert_ids, device)      # [B]
        gnn_emb      = node_embs[node_indices]                      # [B, 256]

        # 3. Gated fusion (with dropout) → [B, 512]
        fused = self.fusion(scf_emb, gnn_emb)

        # 4. 2-layer classification head → [B, 3, G]
        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return logits

    # ---- loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        return self.criterion(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
        )

    # ---- steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        # add_noise=True during training for embedding regularization
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"], add_noise=True)
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        # add_noise=False during validation (deterministic evaluation)
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"], add_noise=False)
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
        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        # Deduplicate: keep first occurrence of each unique index
        mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        # Restore original sample order (important for prediction alignment)
        uniq_idx  = s_idx[mask]
        uniq_pred = s_pred[mask]
        uniq_tgt  = s_tgt[mask]
        orig_order = torch.argsort(uniq_idx)   # inverse sort to recover original order
        f1 = compute_per_gene_f1(uniq_pred[orig_order], uniq_tgt[orig_order])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"], add_noise=False)
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

        # Step 1: Deduplicate within each rank by pert_id (removes padding duplicates from DistributedSampler).
        # Real samples come first in the sampler; padding repeats the first few samples.
        seen: set[str] = set()
        local_dedup_preds: List[torch.Tensor] = []
        local_dedup_pids: List[str] = []
        local_dedup_syms: List[str] = []
        for i, pid in enumerate(self._test_pert_ids):
            if pid not in seen:
                seen.add(pid)
                local_dedup_preds.append(local_preds[i])
                local_dedup_pids.append(pid)
                local_dedup_syms.append(self._test_symbols[i])
        local_dedup_preds_t = torch.stack(local_dedup_preds) if local_dedup_preds else torch.zeros(0, N_CLASSES, N_GENES, dtype=torch.float32)

        # Step 2: All-gather deduplicated results from all ranks.
        all_preds   = self.all_gather(local_dedup_preds_t).view(-1, N_CLASSES, N_GENES)
        all_pert_ids: List[List[str]] = [None] * self.trainer.world_size
        all_symbols: List[List[str]] = [None] * self.trainer.world_size
        torch.distributed.all_gather_object(all_pert_ids, local_dedup_pids)
        torch.distributed.all_gather_object(all_symbols, local_dedup_syms)

        if self.trainer.is_global_zero:
            # Step 3: Deduplicate across ranks (handles any remaining duplicates).
            flat_pids  = [p for rank_pids in all_pert_ids  for p in rank_pids]
            flat_syms  = [s for rank_syms in all_symbols for s in rank_syms]
            n = all_preds.shape[0]
            seen2: set[str] = set()
            rows = []
            for i in range(n):
                pid = flat_pids[i]
                if pid not in seen2:
                    seen2.add(pid)
                    rows.append({
                        "idx":        pid,
                        "input":      flat_syms[i],
                        "prediction": json.dumps(all_preds[i].float().cpu().numpy().tolist()),
                    })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node1-2] Saved {len(rows)} test predictions.")
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
        # Cast checkpoint tensors to match the dtype of the corresponding model
        # parameter. scFoundation params are bfloat16; GNN/head/fusion are float32.
        sd = {}
        for k, v in state_dict.items():
            if k.startswith("scf."):
                sd[k] = v.to(torch.bfloat16)
            else:
                sd[k] = v.to(torch.float32)
        return super().load_state_dict(sd, strict=False)

    # ---- optimizer: scFoundation + GNN (discriminative LR) + head/fusion ----
    def configure_optimizers(self):
        hp = self.hparams

        # Discriminative learning rates:
        # - scFoundation fine-tuned layers: lr (full learning rate)
        # - STRING_GNN: lr * gnn_lr_ratio (discriminative, slower adaptation)
        # - Head + fusion: lr (full learning rate)
        scf_params  = [p for p in self.scf.parameters() if p.requires_grad]
        gnn_params  = [p for p in self.gnn.parameters() if p.requires_grad]
        head_params = (
            list(self.fusion.parameters()) +
            list(self.head.parameters())
        )

        gnn_lr = hp.lr * hp.gnn_lr_ratio  # e.g., 2e-4 * 0.2 = 4e-5

        param_groups = [
            {"params": scf_params,  "lr": hp.lr,    "name": "scf"},
            {"params": gnn_params,  "lr": gnn_lr,   "name": "gnn"},
            {"params": head_params, "lr": hp.lr,    "name": "head"},
        ]

        opt = torch.optim.AdamW(
            param_groups,
            weight_decay=hp.weight_decay,
        )

        # Standard cosine decay (no warm restarts — CAWR was harmful):
        # Linear warmup for warmup_epochs, then cosine to min_lr_ratio * base_lr
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

    parser = argparse.ArgumentParser(description="Node1-2 – scFoundation (6 layers) + STRING_GNN (discriminative LR) + 2-Layer Head + Noise Aug")
    parser.add_argument("--micro-batch-size",     type=int,   default=8)
    parser.add_argument("--global-batch-size",    type=int,   default=64)
    parser.add_argument("--max-epochs",           type=int,   default=200)
    parser.add_argument("--lr",                   type=float, default=2e-4)
    parser.add_argument("--weight-decay",         type=float, default=1.5e-2)
    parser.add_argument("--scf-finetune-layers",  type=int,   default=6,
                        dest="scf_finetune_layers")
    parser.add_argument("--gnn-lr-ratio",         type=float, default=0.2,
                        dest="gnn_lr_ratio")
    parser.add_argument("--head-dropout",         type=float, default=0.3,
                        dest="head_dropout")
    parser.add_argument("--head-dropout2",        type=float, default=0.2,
                        dest="head_dropout2")
    parser.add_argument("--head-hidden",          type=int,   default=HEAD_HIDDEN,
                        dest="head_hidden")
    parser.add_argument("--fusion-dropout",       type=float, default=0.2,
                        dest="fusion_dropout")
    parser.add_argument("--label-smoothing",      type=float, default=0.1,
                        dest="label_smoothing")
    parser.add_argument("--focal-gamma",          type=float, default=1.5,
                        dest="focal_gamma")
    parser.add_argument("--warmup-epochs",        type=int,   default=10,
                        dest="warmup_epochs")
    parser.add_argument("--min-lr-ratio",         type=float, default=0.05,
                        dest="min_lr_ratio")
    parser.add_argument("--noise-std",            type=float, default=0.01,
                        dest="noise_std")
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
        gnn_lr_ratio        = args.gnn_lr_ratio,
        lr                  = args.lr,
        weight_decay        = args.weight_decay,
        label_smoothing     = args.label_smoothing,
        focal_gamma         = args.focal_gamma,
        warmup_epochs       = args.warmup_epochs,
        min_lr_ratio        = args.min_lr_ratio,
        max_epochs          = args.max_epochs,
        noise_std           = args.noise_std,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
    )
    # patience=15: shorter than node4-1-1's 25 since there are no CAWR warm restart dips
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=15, min_delta=1e-4)
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
        val_check_interval      = args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
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
    print(f"[Node1-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
