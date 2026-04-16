"""Node 4-2-2 – scFoundation (6L) + STRING_GNN (frozen, cached) + GatedFusion + Gene Prior Bias

Improvements over node4-2-1 (F1=0.4801):
1. Remove NeighborhoodAttentionModule: redundant with scFoundation's perturbation signal in fusion
   context; the technique that worked in STRING_GNN-only (node1-1-1-1-1, +0.010 F1) did NOT
   transfer to the fusion architecture. Removing it eliminates the val-test gap (0.0035) it created.
2. Revert Mixup alpha=0.4 -> 0.2: alpha=0.4 created anomalous train < val loss (0.924 vs 1.079)
   indicating over-regularization. Reverting to 0.2 restores the healthy train/val relationship.
3. Raise min_lr_ratio 0.08 -> 0.10: the lower floor in node4-2-1 may have driven too-aggressive
   convergence. Middle ground between parent's 0.15 (too conservative) and 0.08 (too aggressive).
4. Add gene-specific class prior bias (GenePriorBias): a per-gene learnable bias term (shape [3, 6640])
   initialized from training label class frequencies. This encodes gene-level DEG prevalence — some
   genes are highly responsive while others rarely show DEGs — as a complementary signal orthogonal
   to both scFoundation's perturbation embedding and STRING_GNN's topology. Only 19,920 trainable
   parameters added (virtually zero overhead) but provides an information-theoretically grounded
   inductive bias for minority DEG classes.
5. Warmup 5 -> 7 epochs: slightly longer warmup (per feedback recommendation) for stability with the
   adjusted LR floor, giving gradient stability before cosine decay begins.
6. Retain extended training: max_epochs=300, patience=35 (the core improvements from node4-2-1's
   schedule design; these are sound, the other changes were the issue).

Architecture:
  pert_id -> scFoundation(top-6 layers fine-tuned)  -> mean-pool -> [B,768]
  pert_id -> STRING_GNN(frozen,cached)[18870,256]
           -> direct lookup [B,256]  (no neighborhood attention)
  -> GatedFusion(768+256->512) + LayerNorm + Dropout(0.3)
  -> [optional Mixup(alpha=0.2) during training]
  -> Head: Dropout(0.5)->Linear(512->256)->LN->GELU->Dropout(0.25)->Linear(256->19920)
  -> [B, 3, 6640] logits
  -> GenePriorBias: add per-gene class bias initialized from train label frequencies
  -> [B, 3, 6640] biased logits
"""

from __future__ import annotations

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
N_GENES     = 6640
N_CLASSES   = 3
SCF_HIDDEN  = 768    # scFoundation hidden size
GNN_HIDDEN  = 256    # STRING_GNN hidden size
FUSION_DIM  = 512    # Gated fusion output dimension
HEAD_HIDDEN = 256    # Two-layer head intermediate dimension

SCF_MODEL_DIR = "/home/Models/scFoundation"
GNN_MODEL_DIR = "/home/Models/STRING_GNN"

CLASS_FREQ = [0.0429, 0.9251, 0.0320]   # down(-1->0), neutral(0->1), up(1->2)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Inverse-sqrt-frequency class weights to handle 92.5% neutral class dominance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py logic.

    Args:
        preds:   [N, 3, G] float32 softmax probabilities
        targets: [N, G]   int64 class indices in {0, 1, 2}
    Returns:
        Scalar F1 averaged over genes.
    """
    y_hat = preds.argmax(dim=1)  # [N, G]
    G = targets.shape[1]
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
        f1_c = torch.where(
            prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-8), torch.zeros_like(prec)
        )
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


def compute_gene_prior_bias(train_tsv_path: Path) -> torch.Tensor:
    """Compute per-gene class log-frequency bias from training labels.

    For each gene g, compute log(class_count[c, g] + 1) for c in {0, 1, 2}.
    This initializes the GenePriorBias with gene-specific DEG prevalence.

    Args:
        train_tsv_path: Path to train.tsv with 'label' column

    Returns:
        Tensor of shape [3, N_GENES] with log-frequency biases, zero-mean per gene.
    """
    df = pd.read_csv(train_tsv_path, sep="\t")
    # Parse labels: each row is a JSON list of length N_GENES in {-1, 0, 1}
    # Convert to class indices {0, 1, 2}
    labels_list = [json.loads(row) for row in df["label"].tolist()]
    labels_arr = np.array(labels_list, dtype=np.int32) + 1  # shape [N, N_GENES], values {0,1,2}

    N, G = labels_arr.shape
    # Count per class per gene: [3, G]
    counts = np.zeros((3, G), dtype=np.float32)
    for c in range(3):
        counts[c] = (labels_arr == c).sum(axis=0).astype(np.float32)

    # Log-frequency bias with Laplace smoothing
    log_freq = np.log(counts + 1.0)  # [3, G]

    # Zero-mean per gene (subtract per-gene mean across 3 classes)
    # This ensures the prior bias shifts class distribution without inflating all logits
    log_freq -= log_freq.mean(axis=0, keepdims=True)  # [3, G]

    return torch.tensor(log_freq, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Gene Prior Bias Module (NEW in node4-2-2)
# ---------------------------------------------------------------------------
class GenePriorBias(nn.Module):
    """Per-gene learnable class bias for the classification head.

    This module adds a gene-specific, class-specific learnable bias term to the
    raw logits from the classification head. The bias is initialized from the
    training label class frequencies (log-frequency), encoding:
    - Which genes are frequently up/down-regulated (high DEG genes)
    - Which genes are rarely perturbed (mostly neutral)

    This provides an information-theoretically grounded inductive bias that:
    1. Complements scFoundation's perturbation-identity signal (query-side)
    2. Complements STRING_GNN's PPI topology (structural side)
    3. Encodes gene-level response tendencies (key-side) orthogonally

    The bias is fully learnable, starting from the log-frequency initialization
    and adapting through backpropagation. At ~19,920 parameters it is negligible
    overhead compared to the 42.5M+ model.

    Architecture:
        logits: [B, 3, G]
        bias:   [3, G]  (learnable, initialized from log train class frequencies)
        output: logits + bias.unsqueeze(0)  -> [B, 3, G]
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        n_genes: int = N_GENES,
        init_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if init_bias is not None:
            # Initialize directly with the provided tensor data (avoids meta-device copy issue)
            self.bias = nn.Parameter(init_bias.clone())
        else:
            self.bias = nn.Parameter(torch.zeros(n_classes, n_genes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Add per-gene class bias to logits.

        Args:
            logits: [B, 3, G] raw logits from classification head

        Returns:
            [B, 3, G] logits with per-gene class bias added
        """
        return logits + self.bias.unsqueeze(0)  # broadcast over batch


# ---------------------------------------------------------------------------
# Gated Fusion Module
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Element-wise gated fusion of two heterogeneous embeddings.

    output = LayerNorm(Dropout(gate_scf ⊙ proj_scf(scf) + gate_gnn ⊙ proj_gnn(gnn)))
    where gates are computed from the concatenation of both inputs.
    """

    def __init__(
        self,
        d_scf: int = SCF_HIDDEN,
        d_gnn: int = GNN_HIDDEN,
        d_out: int = FUSION_DIM,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        d_in = d_scf + d_gnn
        self.proj_scf   = nn.Linear(d_scf, d_out)
        self.proj_gnn   = nn.Linear(d_gnn, d_out)
        self.gate_scf   = nn.Linear(d_in,  d_out)
        self.gate_gnn   = nn.Linear(d_in,  d_out)
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, scf_emb: torch.Tensor, gnn_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([scf_emb, gnn_emb], dim=-1)     # [B, d_scf+d_gnn]
        gate_s   = torch.sigmoid(self.gate_scf(combined))     # [B, d_out]
        gate_g   = torch.sigmoid(self.gate_gnn(combined))     # [B, d_out]
        fused    = gate_s * self.proj_scf(scf_emb) + gate_g * self.proj_gnn(gnn_emb)
        return self.dropout(self.layer_norm(fused))            # [B, d_out]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List[torch.Tensor]] = (
            [
                torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
                for row in df["label"].tolist()
            ]
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
    """Collate function that tokenizes inputs for scFoundation."""

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
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

        self.train_ds = DEGDataset(pd.read_csv(TRAIN_TSV, sep="\t"))
        self.val_ds   = DEGDataset(pd.read_csv(VAL_TSV,   sep="\t"))
        self.test_ds  = DEGDataset(pd.read_csv(TEST_TSV,  sep="\t"))

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate_scf(self.tokenizer),
        )

    def train_dataloader(self) -> DataLoader: return self._loader(self.train_ds, True)
    def val_dataloader(self)   -> DataLoader: return self._loader(self.val_ds,   False)
    def test_dataloader(self)  -> DataLoader: return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FusionDEGModel(pl.LightningModule):
    """scFoundation (top-6 layers fine-tuned) + STRING_GNN (frozen, cached, DIRECT lookup)
    + GatedFusion + GenePriorBias.

    Changes vs. node4-2-1:
    1. No NeighborhoodAttentionModule — direct cached GNN lookup [B, 256].
    2. Reverted Mixup alpha: 0.4 -> 0.2 (closes train<val loss anomaly).
    3. Raised min_lr_ratio: 0.08 -> 0.10 (healthier convergence).
    4. Added GenePriorBias: per-gene class bias initialized from train log-frequencies.
    5. Warmup: 5 -> 7 epochs (slightly longer for stability with adjusted LR floor).
    6. Extended training retained: max_epochs=300, patience=35.
    """

    def __init__(
        self,
        scf_finetune_layers: int = 6,
        head_dropout: float      = 0.5,
        fusion_dropout: float    = 0.3,
        lr: float                = 2e-4,
        weight_decay: float      = 3e-2,
        warmup_epochs: int       = 7,
        max_epochs: int          = 300,
        min_lr_ratio: float      = 0.10,
        mixup_alpha: float       = 0.2,
        label_smoothing: float   = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams
        gnn_dir = Path(GNN_MODEL_DIR)

        # ----------------------------------------------------------------
        # scFoundation backbone (top-k layers fine-tuned)
        # ----------------------------------------------------------------
        self.scf = AutoModel.from_pretrained(
            SCF_MODEL_DIR,
            trust_remote_code=True,
            _use_flash_attention_2=True,
        ).to(torch.bfloat16)
        self.scf.config.use_cache = False
        self.scf.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Freeze all scF params, then unfreeze top-k transformer layers
        for param in self.scf.parameters():
            param.requires_grad = False
        n_scf_layers = len(self.scf.encoder.transformer_encoder)
        for i in range(n_scf_layers - hp.scf_finetune_layers, n_scf_layers):
            for param in self.scf.encoder.transformer_encoder[i].parameters():
                param.requires_grad = True
        # Unfreeze final LayerNorm
        for param in self.scf.encoder.norm.parameters():
            param.requires_grad = True
        # Cast trainable scF params to float32 for stable optimization
        for name, param in self.scf.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        scf_train = sum(p.numel() for p in self.scf.parameters() if p.requires_grad)
        scf_total = sum(p.numel() for p in self.scf.parameters())
        print(f"[Node4-2-2] scFoundation: {scf_train:,}/{scf_total:,} trainable params")

        # ----------------------------------------------------------------
        # STRING_GNN: fully frozen, embeddings precomputed and cached
        # ----------------------------------------------------------------
        print("[Node4-2-2] Precomputing STRING_GNN embeddings (frozen)...")
        gnn_temp    = AutoModel.from_pretrained(str(gnn_dir), trust_remote_code=True).float()
        gnn_temp.eval()
        graph_data  = torch.load(gnn_dir / "graph_data.pt", map_location="cpu")
        edge_index  = graph_data["edge_index"].long()
        edge_weight = graph_data.get("edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.float()
        with torch.no_grad():
            gnn_out  = gnn_temp(edge_index=edge_index, edge_weight=edge_weight)
            gnn_embs = gnn_out.last_hidden_state.float().detach()   # [18870, 256]
        # Register as a buffer -> auto-moved to GPU by Lightning
        self.register_buffer("gnn_embs_cached", gnn_embs)
        del gnn_temp   # free memory
        print(f"[Node4-2-2] GNN embeddings cached: {gnn_embs.shape}")

        # Build Ensembl ID -> node index lookup
        node_names = json.loads((gnn_dir / "node_names.json").read_text())
        self._ensembl_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(node_names)
        }

        # ----------------------------------------------------------------
        # Gated Fusion Module
        # ----------------------------------------------------------------
        self.fusion = GatedFusion(
            d_scf=SCF_HIDDEN, d_gnn=GNN_HIDDEN, d_out=FUSION_DIM,
            dropout=hp.fusion_dropout,
        )

        # ----------------------------------------------------------------
        # Two-layer Classification Head: 512 -> 256 -> 3*6640
        # ----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Dropout(hp.head_dropout),
            nn.Linear(FUSION_DIM, HEAD_HIDDEN),
            nn.LayerNorm(HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(hp.head_dropout * 0.5),
            nn.Linear(HEAD_HIDDEN, N_CLASSES * N_GENES),
        )

        # ----------------------------------------------------------------
        # Gene Prior Bias (NEW in node4-2-2)
        # ----------------------------------------------------------------
        # Initialize from training label log-frequencies for informed start
        print("[Node4-2-2] Computing gene-specific class prior biases from training labels...")
        prior_bias = compute_gene_prior_bias(TRAIN_TSV)  # [3, N_GENES]
        self.gene_prior = GenePriorBias(
            n_classes=N_CLASSES, n_genes=N_GENES, init_bias=prior_bias
        )
        bias_params = sum(p.numel() for p in self.gene_prior.parameters())
        print(f"[Node4-2-2] GenePriorBias: {bias_params:,} trainable params")

        # Class weights for weighted CE loss
        self.register_buffer("class_weights", get_class_weights())

        # Accumulators for validation / test
        self._val_preds:     List[torch.Tensor] = []
        self._val_tgts:      List[torch.Tensor] = []
        self._val_idx:       List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []

    # ---- GNN index lookup ----
    def _get_gnn_indices(self, pert_ids: List[str], device: torch.device) -> torch.Tensor:
        """Look up STRING_GNN node indices for a batch of Ensembl gene IDs."""
        indices = [self._ensembl_to_idx.get(pid, 0) for pid in pert_ids]
        return torch.tensor(indices, dtype=torch.long, device=device)

    # ---- Embedding computation ----
    def get_fused_emb(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids:       List[str],
    ) -> torch.Tensor:
        """Compute fused embedding: scFoundation + direct GNN lookup -> GatedFusion."""
        device = input_ids.device

        # 1. scFoundation -> mean-pool over sequence tokens
        scf_out = self.scf(input_ids=input_ids, attention_mask=attention_mask)
        scf_emb = scf_out.last_hidden_state.float().mean(dim=1)   # [B, 768]

        # 2. STRING_GNN: direct cached embedding lookup (no neighborhood attention)
        node_indices = self._get_gnn_indices(pert_ids, device)
        gnn_emb = self.gnn_embs_cached[node_indices]               # [B, 256]

        # 3. Gated fusion -> [B, 512]
        return self.fusion(scf_emb, gnn_emb)

    # ---- Forward ----
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        pert_ids:       List[str],
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        fused = self.get_fused_emb(input_ids, attention_mask, pert_ids)
        raw_logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return self.gene_prior(raw_logits)  # add gene-specific class bias

    # ---- Loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Weighted CE with label smoothing — proven better F1 alignment than focal loss."""
        B, C, G = logits.shape
        return F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, C),  # [B*G, C]
            targets.reshape(-1),                       # [B*G]
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

    # ---- Training ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pert_ids       = batch["pert_id"]
        labels         = batch["labels"]
        B = input_ids.shape[0]

        fused = self.get_fused_emb(input_ids, attention_mask, pert_ids)

        # Mixup augmentation (alpha=0.2, reverted from 0.4 for healthy train/val balance)
        if self.hparams.mixup_alpha > 0.0 and B > 1 and self.training:
            lam = float(np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha))
            perm = torch.randperm(B, device=fused.device)
            fused_mix = lam * fused + (1 - lam) * fused[perm]
            raw_logits = self.head(fused_mix).view(B, N_CLASSES, N_GENES)
            logits = self.gene_prior(raw_logits)
            loss = lam * self._loss(logits, labels) + (1 - lam) * self._loss(logits, labels[perm])
        else:
            raw_logits = self.head(fused).view(B, N_CLASSES, N_GENES)
            logits = self.gene_prior(raw_logits)
            loss = self._loss(logits, labels)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    # ---- Validation ----
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        probs = torch.softmax(logits.float(), dim=1).detach()
        self._val_preds.append(probs)
        self._val_idx.append(batch["sample_idx"].detach())
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            self._val_tgts.append(batch["labels"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds or not self._val_tgts:
            return
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        local_idx   = torch.cat(self._val_idx,   0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        # Deduplicate (DDP may duplicate samples at epoch boundaries)
        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([
            torch.ones(1, dtype=torch.bool, device=s_idx.device),
            s_idx[1:] != s_idx[:-1],
        ])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    # ---- Test ----
    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["pert_id"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "labels" in batch and batch["labels"] is not None:
            self.log("test/loss", self._loss(logits, batch["labels"]), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)

        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        if is_dist:
            all_preds    = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
            all_pert_ids = [None] * self.trainer.world_size
            all_symbols  = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(all_pert_ids, self._test_pert_ids)
            torch.distributed.all_gather_object(all_symbols,  self._test_symbols)
            flat_pids = [p for rank_pids in all_pert_ids for p in rank_pids]
            flat_syms = [s for rank_syms in all_symbols  for s in rank_syms]
        else:
            all_preds = local_preds
            flat_pids = self._test_pert_ids
            flat_syms = self._test_symbols

        if self.trainer.is_global_zero:
            n    = all_preds.shape[0]
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
            print(f"[Node4-2-2] Saved {len(rows)} test predictions -> {out_dir}/test_predictions.tsv")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    # ---- Checkpoint: save only trainable params + all buffers ----
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
        self.print(
            f"[Node4-2-2] Checkpoint: {trained:,}/{total:,} params ({100*trained/total:.2f}%)"
        )
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- Optimizer with WarmupCosine LR scheduler ----
    def configure_optimizers(self):
        hp = self.hparams
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable_params, lr=hp.lr, weight_decay=hp.weight_decay
        )

        def lr_lambda(epoch: int) -> float:
            """Linear warmup then cosine decay to min_lr_ratio floor."""
            if epoch < hp.warmup_epochs:
                return max(1e-8, epoch / max(1, hp.warmup_epochs))
            progress = (epoch - hp.warmup_epochs) / max(1, hp.max_epochs - hp.warmup_epochs)
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return hp.min_lr_ratio + (1.0 - hp.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node4-2-2 – scFoundation + STRING_GNN (frozen + direct lookup) + GatedFusion + GenePriorBias"
    )
    parser.add_argument("--micro-batch-size",    type=int,   default=8)
    parser.add_argument("--global-batch-size",   type=int,   default=64)
    parser.add_argument("--max-epochs",          type=int,   default=300)
    parser.add_argument("--lr",                  type=float, default=2e-4)
    parser.add_argument("--weight-decay",        type=float, default=3e-2)
    parser.add_argument("--scf-finetune-layers", type=int,   default=6,
                        dest="scf_finetune_layers")
    parser.add_argument("--head-dropout",        type=float, default=0.5)
    parser.add_argument("--fusion-dropout",      type=float, default=0.3)
    parser.add_argument("--warmup-epochs",       type=int,   default=7)
    parser.add_argument("--min-lr-ratio",        type=float, default=0.10)
    parser.add_argument("--mixup-alpha",         type=float, default=0.2)
    parser.add_argument("--label-smoothing",     type=float, default=0.1)
    parser.add_argument("--num-workers",         type=int,   default=4)
    parser.add_argument("--debug-max-step",      type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",        action="store_true", dest="fast_dev_run")
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

    dm = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = FusionDEGModel(
        scf_finetune_layers = args.scf_finetune_layers,
        head_dropout        = args.head_dropout,
        fusion_dropout      = args.fusion_dropout,
        lr                  = args.lr,
        weight_decay        = args.weight_decay,
        warmup_epochs       = args.warmup_epochs,
        max_epochs          = args.max_epochs,
        min_lr_ratio        = args.min_lr_ratio,
        mixup_alpha         = args.mixup_alpha,
        label_smoothing     = args.label_smoothing,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
    )
    es_cb  = EarlyStopping(monitor="val/f1", mode="max", patience=35, min_delta=1e-4)
    lr_cb  = LearningRateMonitor(logging_interval="epoch")
    pg_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

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

    score_path = Path(__file__).parent / "test_score.txt"
    with open(score_path, "w") as f:
        f.write(f"test_results: {test_results}\n")
        if test_results:
            for k, v in test_results[0].items():
                f.write(f"  {k}: {v}\n")
    print(f"[Node4-2-2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
