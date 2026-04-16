"""Node 4-1-1-1-1-1-1-1-1-1-1: ESM2-650M (LoRA r=24) + STRING_GNN
              RESTORED per-gene bias (19,920 params),
              NO SWA (best single checkpoint only),
              Manifold Mixup regularization in fused embedding space,
              Restored weight decay=2e-3, label_smoothing=0.03

Architecture:
  - Branch A: ESM2-650M with LoRA (r=24, alpha=48), no branch dropout → [B, 1280]
  - Branch B: STRING_GNN with cond_emb injection + learnable gain → [B, 256]
  - Gated Fusion: gate-weighted combination → [B, 512]
  - Manifold Mixup: applied in fused 512-dim embedding space (alpha=0.2, prob=0.5)
  - Head: 2-layer MLP (LayerNorm→Linear→GELU→Dropout(0.45)→Linear) + PER-GENE BIAS
    (19,920 params = 3 × 6,640) → [B, 3, 6640]
  - Inference: best single checkpoint (no SWA) + TTA (8 dropout passes)

Critical fix from parent feedback:
  The parent node (test F1=0.2188, -58% regression) catastrophically failed
  because class-shared gene bias (6,640 params) destroyed class discrimination.
  This node RESTORES per-gene bias (19,920 params = 3 × 6,640) from the
  grandparent (test F1=0.5223), which is CRITICAL for the model to learn
  class-specific gene response patterns.

Key changes from parent (node4-1-1-1-1-1-1-1-1-1):
  1. CRITICAL: RESTORE per-gene bias (19,920 params = 3 × 6640)
     - Parent used class-shared bias (6,640 params) → uniform predictions
     - Per-gene bias allows independent class offsets per gene
  2. DISABLE SWA: SWA provided zero benefit (weight diff=0.000109)
     - Use best single checkpoint instead
  3. RESTORE weight decay 1e-3 → 2e-3 (feedback recommendation)
  4. RESTORE label smoothing 0.02 → 0.03 (feedback recommendation)
  5. ADD Manifold Mixup (alpha=0.2, prob=0.5) in fused embedding space
     - Proven effective in node3 lineage (reached 0.5243 with Mixup)
     - Creates smooth interpolations for better generalization
     - Applied post-fusion to preserve the fusion's gating mechanism

Retained from node4-1-1-1-1-1-1-1-1 (grandparent at F1=0.5223):
  - LoRA r=24 (from failing parent but historically proven: 0.5175)
  - No ESM2 branch dropout (parent's change: 0.15 failed → increased gap)
  - Split LR (5e-5/1e-4): Proven critical
  - Head hidden=512, dropout=0.45
  - Muon optimizer for STRING_GNN
  - Focal loss gamma=2.0
  - Cosine annealing (warmup=10, T_max=160)
  - Early stop patience=55
  - TTA (8 dropout passes)
  - Gradient checkpointing for ESM2
  - ESM2_MAX_LEN=512
  - save_top_k=10

Auxiliary data:
  - ESM2 protein sequences: /home/data/genome/hg38_gencode_protein.fa
  - STRING_GNN graph: /home/Models/STRING_GNN/graph_data.pt
  - STRING_GNN node names: /home/Models/STRING_GNN/node_names.json
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
import math
import glob as glob_module
import re
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
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, EsmForMaskedLM

# muon optimizer — imported here so it's available for configure_optimizers
try:
    from muon import MuonWithAuxAdam
except ImportError:
    MuonWithAuxAdam = None  # Will fall back to pure AdamW if --no-muon

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
STRING_GNN_DIR = "/home/Models/STRING_GNN"
PROTEIN_FASTA = "/home/data/genome/hg38_gencode_protein.fa"
N_GENES = 6640
N_CLASSES = 3
ESM2_MAX_LEN = 512      # per skill: max 1024; 512 covers most proteins
ESM2_HIDDEN = 1280      # ESM2-650M hidden size
GNN_HIDDEN = 256        # STRING_GNN embedding dim
FUSED_DIM = 512         # dimension after gated fusion


# ---------------------------------------------------------------------------
# Protein FASTA helpers
# ---------------------------------------------------------------------------
def _build_ensg_to_seq(fasta_path: str) -> Dict[str, str]:
    """Build ENSG→longest protein sequence map from GENCODE protein FASTA."""
    ensg2seq: Dict[str, str] = {}
    current_ensg: Optional[str] = None
    current_seq_parts: List[str] = []

    def _flush():
        if current_ensg and current_seq_parts:
            seq = "".join(current_seq_parts)
            if current_ensg not in ensg2seq or len(seq) > len(ensg2seq[current_ensg]):
                ensg2seq[current_ensg] = seq

    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                _flush()
                current_seq_parts = []
                current_ensg = None
                fields = line[1:].split("|")
                if len(fields) >= 3:
                    current_ensg = fields[2].split(".")[0]
            else:
                current_seq_parts.append(line)
    _flush()
    return ensg2seq


_ENSG2SEQ_CACHE: Optional[Dict[str, str]] = None


def get_ensg2seq() -> Dict[str, str]:
    global _ENSG2SEQ_CACHE
    if _ENSG2SEQ_CACHE is None:
        _ENSG2SEQ_CACHE = _build_ensg_to_seq(PROTEIN_FASTA)
    return _ENSG2SEQ_CACHE


FALLBACK_SEQ = "M"


# ---------------------------------------------------------------------------
# STRING_GNN helpers
# ---------------------------------------------------------------------------
def _build_ensg_to_node_idx(node_names_path: str) -> Dict[str, int]:
    """Map ENSG IDs (no version) to STRING_GNN node indices."""
    with open(node_names_path, "r") as f:
        node_names: List[str] = json.load(f)
    ensg2idx = {}
    for i, name in enumerate(node_names):
        ensg = name.split(".")[0]  # strip version if any
        ensg2idx[ensg] = i
    return ensg2idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        ensg2seq: Dict[str, str],
        ensg2node: Dict[str, int],
        n_gnn_nodes: int,
    ) -> None:
        self.pert_ids: List[str] = df["pert_id"].tolist()
        self.symbols: List[str] = df["symbol"].tolist()
        self.n_gnn_nodes = n_gnn_nodes

        # Resolve protein sequences
        self.sequences: List[str] = []
        for pid in self.pert_ids:
            ensg = pid.split(".")[0]
            self.sequences.append(ensg2seq.get(ensg, FALLBACK_SEQ))

        # Resolve STRING_GNN node indices (-1 means not in PPI graph)
        self.node_indices: List[int] = []
        for pid in self.pert_ids:
            ensg = pid.split(".")[0]
            self.node_indices.append(ensg2node.get(ensg, -1))

        if "label" in df.columns:
            labels = np.array([json.loads(x) for x in df["label"].tolist()], dtype=np.int64)
            self.labels: Optional[torch.Tensor] = torch.tensor(labels + 1, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.pert_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "seq": self.sequences[idx],
            "node_idx": self.node_indices[idx],  # int, -1 if not in graph
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
        micro_batch_size: int = 4,
        num_workers: int = 4,
        max_seq_len: int = ESM2_MAX_LEN,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len

        self.tokenizer = None
        self.ensg2seq: Optional[Dict[str, str]] = None
        self.ensg2node: Optional[Dict[str, int]] = None
        self.n_gnn_nodes: int = 18870
        self.train_ds: Optional[PerturbDataset] = None
        self.val_ds: Optional[PerturbDataset] = None
        self.test_ds: Optional[PerturbDataset] = None

    def setup(self, stage: str = "fit") -> None:
        # ESM2 tokenizer — rank-0 downloads first
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(ESM2_MODEL)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL)

        self.ensg2seq = get_ensg2seq()
        self.ensg2node = _build_ensg_to_node_idx(
            str(Path(STRING_GNN_DIR) / "node_names.json")
        )
        self.n_gnn_nodes = len(self.ensg2node)

        train_df = pd.read_csv(self.train_path, sep="\t")
        val_df = pd.read_csv(self.val_path, sep="\t")
        test_df = pd.read_csv(self.test_path, sep="\t")

        self.train_ds = PerturbDataset(train_df, self.ensg2seq, self.ensg2node, self.n_gnn_nodes)
        self.val_ds = PerturbDataset(val_df, self.ensg2seq, self.ensg2node, self.n_gnn_nodes)
        self.test_ds = PerturbDataset(test_df, self.ensg2seq, self.ensg2node, self.n_gnn_nodes)

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        seqs = [item["seq"] for item in batch]
        tokenized = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        result = {
            "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
            "pert_id": [item["pert_id"] for item in batch],
            "symbol": [item["symbol"] for item in batch],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "node_idx": torch.tensor([item["node_idx"] for item in batch], dtype=torch.long),
        }
        if "label" in batch[0]:
            result["label"] = torch.stack([item["label"] for item in batch])
        return result

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )


# ---------------------------------------------------------------------------
# Gated Fusion (with optional dropout)
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Learnable gated fusion of two embeddings with dropout regularization.

    gate = sigmoid(W_a * a + W_b * b + bias)
    fused = gate * a_proj + (1 - gate) * b_proj
    """

    def __init__(self, dim_a: int, dim_b: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj_a = nn.Linear(dim_a, out_dim)
        self.proj_b = nn.Linear(dim_b, out_dim)
        self.gate_a = nn.Linear(dim_a, out_dim, bias=False)
        self.gate_b = nn.Linear(dim_b, out_dim, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: [B, dim_a], b: [B, dim_b]
        a_proj = self.proj_a(a)   # [B, out_dim]
        b_proj = self.proj_b(b)   # [B, out_dim]
        gate = torch.sigmoid(
            self.gate_a(a) + self.gate_b(b) + self.gate_bias
        )  # [B, out_dim]
        fused = gate * a_proj + (1.0 - gate) * b_proj
        fused = self.norm(fused)
        return self.dropout(fused)  # [B, out_dim]


# ---------------------------------------------------------------------------
# Prediction Head with PER-GENE BIAS (RESTORED — critical fix)
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """2-layer MLP with per-gene bias term.

    Per-gene bias (3 × 6640 = 19,920 params) allows the model to independently
    adjust logit offsets for each (class, gene) pair. This is CRITICAL for the
    model to learn class-specific gene response patterns.

    RESTORATION: The parent used class-shared bias (6,640 params), which
    catastrophically destroyed class discrimination (test F1 collapsed to 0.2188
    vs parent's 0.5223). Per-gene bias is restored here.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        n_genes: int = N_GENES,
        dropout: float = 0.45,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes * N_CLASSES),
        )
        # Per-gene bias: 3 × 6,640 = 19,920 params (RESTORED from parent's 6,640)
        # This allows independent class offsets per gene, critical for class discrimination
        self.gene_bias = nn.Parameter(torch.zeros(N_CLASSES, n_genes))
        self.n_genes = n_genes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        logits = self.net(x).view(-1, N_CLASSES, self.n_genes)  # [B, 3, N_GENES]
        # Apply per-gene bias: independent offset per (class, gene) pair
        # gene_bias: [3, N_GENES] → broadcast over batch dim
        logits = logits + self.gene_bias.unsqueeze(0)  # [B, 3, N_GENES]
        return logits


# ---------------------------------------------------------------------------
# Focal Loss WITH label smoothing
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss with class weights and optional label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.03,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # [n_classes]
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: [N, C], target: [N] (class indices)
        log_prob = F.log_softmax(input, dim=-1)  # [N, C]
        prob = torch.exp(log_prob)               # [N, C]

        # Focal weight: based on the true class probability (hard label)
        p_t = prob.gather(1, target.unsqueeze(1)).squeeze(1)  # [N]
        focal_w = (1.0 - p_t) ** self.gamma                   # [N]

        # Cross-entropy with label smoothing
        if self.label_smoothing > 0:
            n_classes = input.size(-1)
            smooth_targets = torch.full_like(log_prob, self.label_smoothing / n_classes)
            smooth_targets.scatter_(
                1,
                target.unsqueeze(1),
                1.0 - self.label_smoothing + self.label_smoothing / n_classes
            )
            ce = -(smooth_targets * log_prob).sum(dim=-1)  # [N]
        else:
            ce = F.nll_loss(log_prob, target, reduction="none")  # [N]

        # Combine focal weight
        loss = focal_w * ce  # [N]

        # Apply class weights (based on hard label for interpretability)
        if self.weight is not None:
            w = self.weight.to(device=input.device, non_blocking=True)
            loss = loss * w[target]

        return loss.mean()


# ---------------------------------------------------------------------------
# ESM2 cond_emb projector with learnable gain
# ---------------------------------------------------------------------------
class CondEmbProjector(nn.Module):
    """Projects ESM2 embeddings to STRING_GNN cond_emb dimension (256).

    Adds a learnable gain parameter (initialized near 1.0) to scale the
    conditioning signal strength.
    """

    def __init__(self, in_dim: int = ESM2_HIDDEN, out_dim: int = GNN_HIDDEN) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
        )
        # Learnable gain: initialized to 0 → softplus(0) ≈ 0.693, approximately 1.0 effective
        self.log_gain = nn.Parameter(torch.zeros(1))

    def forward(self, esm2_emb: torch.Tensor) -> torch.Tensor:
        # esm2_emb: [B, ESM2_HIDDEN]
        projected = self.proj(esm2_emb)  # [B, GNN_HIDDEN]
        gain = F.softplus(self.log_gain)  # positive scalar, ~0.693 initially
        return projected * gain  # [B, GNN_HIDDEN]


# ---------------------------------------------------------------------------
# Manifold Mixup helper
# ---------------------------------------------------------------------------
def manifold_mixup(
    fused: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Manifold Mixup in the fused embedding space.

    Creates linear interpolations between pairs of samples in the embedding space.
    This doubles the effective training set size and improves generalization by
    encouraging smoother decision boundaries.

    Proven effective in node3 lineage (test F1=0.5243 with Mixup prob=0.65).

    Args:
        fused: [B, FUSED_DIM] fused embeddings
        labels: [B, N_GENES] integer labels (shifted to {0,1,2})
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        mixed_fused: [B, FUSED_DIM] mixed embeddings
        labels_a: [B, N_GENES] labels for first sample
        labels_b: [B, N_GENES] labels for second sample
        lam: mixing coefficient
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    B = fused.size(0)
    # Create a random permutation of the batch
    perm = torch.randperm(B, device=fused.device)
    mixed_fused = lam * fused + (1 - lam) * fused[perm]
    labels_a = labels
    labels_b = labels[perm]
    return mixed_fused, labels_a, labels_b, lam


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.45,
        fusion_dropout: float = 0.2,
        backbone_lr: float = 5e-5,            # Split LR: lower for LoRA parameters
        lr: float = 1e-4,                     # Split LR: higher for head/GNN/fusion/cond_proj
        weight_decay: float = 2e-3,           # RESTORED to 2e-3 (per feedback: 1e-3 showed no improvement)
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.03,        # RESTORED to 0.03 (per feedback)
        warmup_epochs: int = 10,
        max_epochs: int = 300,
        cosine_t_max: int = 160,
        lora_r: int = 24,                     # LoRA r=24 (better task adaptation than r=16)
        lora_alpha: int = 48,                 # alpha=48 maintains ratio=2.0 with r=24
        lora_dropout: float = 0.10,
        use_mixup: bool = True,               # ENABLED: Manifold Mixup for regularization
        mixup_alpha: float = 0.2,             # Beta distribution alpha for Mixup
        mixup_prob: float = 0.5,              # Probability of applying Mixup per batch
        tta_passes: int = 8,
        use_muon: bool = True,
        muon_lr: float = 0.01,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.esm2: Optional[nn.Module] = None
        self.string_gnn = None
        self.cond_proj: Optional[CondEmbProjector] = None
        self.fusion: Optional[GatedFusion] = None
        self.head: Optional[PerturbHead] = None
        self.focal_loss: Optional[FocalLoss] = None
        self._n_gnn_nodes: int = 18870  # will be set in setup from graph
        self.micro_batch_size: int = 4  # will be set via setter before fit

        # Validation state — track pert_ids for correct DDP deduplication
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_pert_ids: List[str] = []

        # Test state
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # ESM2-650M with LoRA (r=24)
        base_esm2 = EsmForMaskedLM.from_pretrained(ESM2_MODEL, dtype=torch.float32)

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=["query", "key", "value", "dense"],
            bias="none",
        )
        self.esm2 = get_peft_model(base_esm2, lora_config)
        # Enable gradient checkpointing for activation memory savings
        self.esm2.gradient_checkpointing_enable()

        # NOTE: NO ESM2 branch dropout — parent's Dropout(0.15) failed to reduce
        # train-val gap and may have increased it. Full ESM2 embedding flows directly.

        # STRING_GNN (full fine-tune — only 5.43M params)
        self.string_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        # Load graph data and register as buffers
        graph = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", map_location="cpu")
        self.register_buffer("_graph_edge_index", graph["edge_index"].long())
        self.register_buffer("_graph_edge_weight", graph["edge_weight"].float())
        self._n_gnn_nodes = int(graph["edge_index"].max().item()) + 1

        # cond_emb projector with learnable gain: ESM2[B, 1280] → [B, 256] for GNN conditioning
        self.cond_proj = CondEmbProjector(in_dim=ESM2_HIDDEN, out_dim=GNN_HIDDEN)

        # Gated fusion and prediction head
        self.fusion = GatedFusion(
            dim_a=ESM2_HIDDEN,
            dim_b=GNN_HIDDEN,
            out_dim=FUSED_DIM,
            dropout=self.hparams.fusion_dropout,
        )
        # Head with PER-GENE BIAS (19,920 params = 3 × 6,640) — RESTORED
        self.head = PerturbHead(
            in_dim=FUSED_DIM,
            hidden_dim=self.hparams.head_hidden_dim,
            dropout=self.hparams.head_dropout,
        )

        # Class weights for focal loss (inverse frequency)
        # Frequencies from DATA_ABSTRACT.md: neutral=92.82%, down=4.77%, up=2.41%
        # After +1 shift: class0=down(4.77%), class1=neutral(92.82%), class2=up(2.41%)
        freq = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq)
        class_weights = class_weights / class_weights.mean()
        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"ESM2-650M (LoRA r={self.hparams.lora_r}, alpha={self.hparams.lora_alpha}) "
            f"+ STRING_GNN + cond_emb(gain) | "
            f"head_dim={self.hparams.head_hidden_dim} | head_drop={self.hparams.head_dropout} | "
            f"per-gene bias (19920 params RESTORED) | "
            f"mixup={self.hparams.use_mixup}(alpha={self.hparams.mixup_alpha}, "
            f"prob={self.hparams.mixup_prob}) | "
            f"wd={self.hparams.weight_decay} | label_smooth={self.hparams.label_smoothing} | "
            f"total={total:,} | trainable={trainable:,} "
            f"({100*trainable/total:.2f}%)"
        )

        # Cast all trainable parameters to float32 for stable optimization
        for p in self.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    def _get_esm2_emb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """ESM2 mean pool over residues (excluding CLS, EOS, PAD).

        NO branch dropout applied.

        Returns: [B, 1280]
        """
        out = self.esm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = out["hidden_states"][-1]  # [B, T, 1280]

        # Exclude special tokens: CLS (0), EOS (2), PAD (1)
        tokenizer_special_ids = torch.tensor(
            [0, 1, 2], dtype=torch.long, device=input_ids.device
        )
        special_mask = torch.isin(input_ids, tokenizer_special_ids)  # [B, T]
        valid_mask = ~special_mask  # [B, T]
        valid_mask_f = valid_mask.unsqueeze(-1).float()  # [B, T, 1]
        sum_emb = (hidden * valid_mask_f).sum(dim=1)     # [B, 1280]
        count = valid_mask_f.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        emb = sum_emb / count  # [B, 1280]
        return emb  # [B, 1280]

    def _get_gnn_emb_with_conditioning(
        self,
        node_idx: torch.Tensor,
        esm2_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Run STRING_GNN with perturbation conditioning via cond_emb + learnable gain."""
        device = node_idx.device
        B = node_idx.size(0)
        N = self._n_gnn_nodes

        # Project ESM2 embeddings to GNN conditioning space (with learnable gain)
        cond_signal = self.cond_proj(esm2_emb)  # [B, 256]

        # Create a full cond_emb matrix [N, 256] initialized to zero
        cond_emb_full = torch.zeros(N, GNN_HIDDEN, dtype=cond_signal.dtype, device=device)

        # For samples with valid node indices, inject their conditioning
        valid = node_idx >= 0
        if valid.any():
            valid_nodes = node_idx[valid]
            valid_signals = cond_signal[valid]

            cond_emb_full.scatter_add_(
                0,
                valid_nodes.unsqueeze(1).expand_as(valid_signals),
                valid_signals,
            )

        # Run STRING_GNN with perturbation conditioning
        out = self.string_gnn(
            edge_index=self._graph_edge_index,
            edge_weight=self._graph_edge_weight,
            cond_emb=cond_emb_full,
        )
        all_emb = out.last_hidden_state  # [N, 256]

        # Gather embeddings for each sample; fallback to zeros for unknown genes
        result = torch.zeros(B, GNN_HIDDEN, dtype=all_emb.dtype, device=device)
        if valid.any():
            result[valid] = all_emb[node_idx[valid]]
        return result.float()

    def _get_fused_emb(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute fused embedding without applying the head.

        Returns: [B, FUSED_DIM]
        """
        esm2_emb = self._get_esm2_emb(batch["input_ids"], batch["attention_mask"])  # [B, 1280]
        gnn_emb = self._get_gnn_emb_with_conditioning(batch["node_idx"], esm2_emb)  # [B, 256]
        return self.fusion(esm2_emb, gnn_emb)  # [B, 512]

    def _forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Standard forward pass."""
        fused = self._get_fused_emb(batch)    # [B, 512]
        return self.head(fused)                # [B, 3, 6640]

    def _forward_with_tta(self, batch: Dict[str, Any], n_passes: int = 8) -> torch.Tensor:
        """Test-Time Augmentation: average over N stochastic dropout passes."""
        self.head.train()
        self.fusion.train()
        self.esm2.eval()
        self.string_gnn.eval()

        logit_sum = None
        for _ in range(n_passes):
            logits = self._forward(batch)  # [B, 3, 6640]
            if logit_sum is None:
                logit_sum = logits.detach().float()
            else:
                logit_sum = logit_sum + logits.detach().float()

        # Restore eval mode for all submodules after TTA
        self.head.eval()
        self.fusion.eval()

        return logit_sum / n_passes

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return self.focal_loss(logits_flat, labels_flat)

    def _compute_mixup_loss(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Compute mixed loss for Manifold Mixup.

        Loss is a convex combination of the two loss terms.
        """
        loss_a = self._compute_loss(logits, labels_a)
        loss_b = self._compute_loss(logits, labels_b)
        return lam * loss_a + (1.0 - lam) * loss_b

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        if self.hparams.use_mixup and "label" in batch:
            # Manifold Mixup in fused embedding space
            # Get fused embedding
            esm2_emb = self._get_esm2_emb(batch["input_ids"], batch["attention_mask"])  # [B, 1280]
            gnn_emb = self._get_gnn_emb_with_conditioning(batch["node_idx"], esm2_emb)  # [B, 256]
            fused = self.fusion(esm2_emb, gnn_emb)  # [B, 512]

            labels = batch["label"]  # [B, N_GENES]

            # Apply Manifold Mixup with probability mixup_prob
            if np.random.random() < self.hparams.mixup_prob:
                mixed_fused, labels_a, labels_b, lam = manifold_mixup(
                    fused, labels, alpha=self.hparams.mixup_alpha
                )
                logits = self.head(mixed_fused)  # [B, 3, 6640]
                loss = self._compute_mixup_loss(logits, labels_a, labels_b, lam)
            else:
                logits = self.head(fused)  # [B, 3, 6640]
                loss = self._compute_loss(logits, labels)
        else:
            logits = self._forward(batch)
            loss = self._compute_loss(logits, batch["label"])

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                  sync_dist=True, batch_size=batch["input_ids"].size(0))
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())
        self._val_pert_ids.extend(batch["pert_id"])

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds_local = torch.cat(self._val_preds, dim=0)
        labels_local = torch.cat(self._val_labels, dim=0)
        pert_ids_local = self._val_pert_ids[:]
        self._val_preds.clear()
        self._val_labels.clear()
        self._val_pert_ids.clear()

        # Gather pert_ids across DDP ranks using all_gather_object
        all_pert_ids: List[str] = []
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            _gathered_ids: List[List[str]] = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(_gathered_ids, pert_ids_local)
            for ids in _gathered_ids:
                all_pert_ids.extend(ids)
        else:
            all_pert_ids = pert_ids_local

        # Gather ALL validation predictions across DDP ranks
        if self.trainer is not None and self.trainer.world_size >= 1:
            all_preds = self.all_gather(preds_local)
            all_labels = self.all_gather(labels_local)
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
            all_labels = all_labels.view(-1, N_GENES)
        else:
            all_preds = preds_local
            all_labels = labels_local

        # Compute F1 with proper pert_id-based deduplication on global zero rank
        # All ranks compute F1 locally (dedup uses only local ids which were gathered
        # via all_gather_object, so all ranks have the same pert_ids list)
        all_preds_np = all_preds.cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()
        all_pert_ids_str = all_pert_ids

        # Deduplicate by pert_id (handles DistributedSampler padding/replication)
        paired = list(zip(all_pert_ids_str, all_preds_np, all_labels_np))
        seen: set = set()
        unique_preds_list: List[np.ndarray] = []
        unique_labels_list: List[np.ndarray] = []
        for pid, pred, lab in paired:
            if pid not in seen:
                seen.add(pid)
                unique_preds_list.append(pred)
                unique_labels_list.append(lab)

        unique_preds_np = np.stack(unique_preds_list)
        unique_labels_np = np.stack(unique_labels_list)
        f1 = _compute_per_gene_f1(unique_preds_np, unique_labels_np)

        # Log F1: all ranks log the same value (all_gather makes preds/labels identical)
        # Use batch_size=self.micro_batch_size for Lightning's batch size inference
        self.log("val/f1", f1, prog_bar=True, sync_dist=True,
                 batch_size=self.micro_batch_size)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step with optional TTA (dropout passes)."""
        tta_passes = self.hparams.tta_passes
        if tta_passes > 1:
            logits = self._forward_with_tta(batch, n_passes=tta_passes)
        else:
            logits = self._forward(batch)

        loss = self._compute_loss(logits, batch["label"])
        self._test_preds.append(logits.detach().cpu().float())
        self._test_labels.append(batch["label"].detach().cpu())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        return {"test/loss": loss}

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)
        labels_local = torch.cat(self._test_labels, dim=0)
        pert_ids_local = self._test_pert_ids[:]
        symbols_local = self._test_symbols[:]
        self._test_preds.clear()
        self._test_labels.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        # Gather pert_ids and symbols from ALL ranks
        all_pert_ids: List[str] = []
        all_symbols: List[str] = []
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            _gathered_ids: List[List[str]] = [None] * self.trainer.world_size
            _gathered_syms: List[List[str]] = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(_gathered_ids, pert_ids_local)
            torch.distributed.all_gather_object(_gathered_syms, symbols_local)
            for ids in _gathered_ids:
                all_pert_ids.extend(ids)
            for syms in _gathered_syms:
                all_symbols.extend(syms)
        else:
            all_pert_ids = pert_ids_local
            all_symbols = symbols_local

        # all_gather predictions and labels across all ranks
        if self.trainer is not None and self.trainer.world_size >= 1:
            all_preds = self.all_gather(preds_local)
            all_labels = self.all_gather(labels_local)
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
            all_labels = all_labels.view(-1, N_GENES)
        else:
            all_preds = preds_local
            all_labels = labels_local

        # Compute test F1 and save predictions on global zero rank
        if self.trainer is not None and self.trainer.is_global_zero:
            all_preds_cpu = all_preds.cpu()
            all_labels_cpu = all_labels.cpu()

            # Deduplicate by pert_id
            paired = list(zip(all_pert_ids, all_symbols,
                              all_preds_cpu.numpy(),
                              all_labels_cpu.numpy()))
            seen: set = set()
            unique_pert_ids, unique_symbols, unique_preds_list, unique_labels_list = [], [], [], []
            for pid, sym, pred, lab in paired:
                if pid not in seen:
                    seen.add(pid)
                    unique_pert_ids.append(pid)
                    unique_symbols.append(sym)
                    unique_preds_list.append(pred)
                    unique_labels_list.append(lab)

            unique_preds_np = np.stack(unique_preds_list)
            unique_labels_np = np.stack(unique_labels_list)

            test_f1 = _compute_per_gene_f1(unique_preds_np, unique_labels_np)
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=False)

            out_path = Path(__file__).parent / "run" / "test_predictions.tsv"
            _save_test_predictions(
                pert_ids=unique_pert_ids,
                symbols=unique_symbols,
                preds=unique_preds_np,
                out_path=out_path,
            )

            # Save test score to node root test_score.txt
            score_path = Path(__file__).parent / "test_score.txt"
            score_path.write_text(f"{test_f1:.10f}\n")
            self.print(f"Test F1: {test_f1:.6f} | Saved to {score_path}")

    def configure_optimizers(self):
        """Configure optimizer with Muon for STRING_GNN hidden layers + AdamW for everything else.

        Optimizer groups:
        1. ESM2 LoRA params: AdamW at backbone_lr=5e-5
        2. STRING_GNN hidden matrices: Muon at muon_lr=0.01 (if use_muon=True)
        3. Head/fusion/cond_proj/GNN scalars: AdamW at lr=1e-4
        """
        # Identify LoRA parameters (backbone branch)
        lora_param_names = set()
        for name, module in self.esm2.named_modules():
            if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                for pname, _ in module.named_parameters():
                    lora_param_names.add(f"esm2.{name}.{pname}")

        # Identify STRING_GNN hidden weight matrices (2D params, not embedding or output layers)
        gnn_hidden_weights = []
        gnn_other_params = []
        gnn_hidden_param_names = set()

        if self.hparams.use_muon and self.string_gnn is not None:
            for name, param in self.string_gnn.named_parameters():
                if param.requires_grad:
                    full_name = f"string_gnn.{name}"
                    is_embedding = "embed" in name.lower() or "embedding" in name.lower()
                    is_output = "out" in name.lower() and name.endswith("weight") and param.ndim == 2 and param.shape[0] == N_GENES
                    is_2d = param.ndim >= 2
                    if is_2d and not is_embedding and not is_output:
                        gnn_hidden_weights.append(param)
                        gnn_hidden_param_names.add(full_name)
                    else:
                        gnn_other_params.append(param)

        # Separate parameters into groups
        backbone_params = []        # ESM2 LoRA → AdamW at backbone_lr
        gnn_muon_params = []        # STRING_GNN 2D matrices → Muon (if use_muon)
        other_params = []           # Head, fusion, cond_proj, GNN scalars → AdamW at lr

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name in lora_param_names or "esm2" in name:
                backbone_params.append(param)
            elif self.hparams.use_muon and name in gnn_hidden_param_names:
                gnn_muon_params.append(param)
            elif self.hparams.use_muon and "string_gnn" in name and name not in gnn_hidden_param_names:
                # GNN biases/scalars → AdamW
                other_params.append(param)
            else:
                other_params.append(param)

        # Build parameter groups for MuonWithAuxAdam
        if self.hparams.use_muon and gnn_muon_params:
            param_groups = [
                # ESM2 LoRA params: AdamW at backbone_lr
                dict(
                    params=backbone_params,
                    use_muon=False,
                    lr=self.hparams.backbone_lr,
                    betas=(0.9, 0.999),
                    weight_decay=self.hparams.weight_decay,
                ),
                # STRING_GNN hidden matrices: Muon for faster GNN optimization
                dict(
                    params=gnn_muon_params,
                    use_muon=True,
                    lr=self.hparams.muon_lr,
                    weight_decay=self.hparams.weight_decay,
                    momentum=0.95,
                ),
                # Head, fusion, cond_proj, GNN scalars: AdamW at higher lr
                dict(
                    params=other_params,
                    use_muon=False,
                    lr=self.hparams.lr,
                    betas=(0.9, 0.999),
                    weight_decay=self.hparams.weight_decay,
                ),
            ]
            self.print(
                f"Using Muon+AdamW optimizer: {len(gnn_muon_params)} GNN hidden matrices "
                f"use Muon (lr={self.hparams.muon_lr}), "
                f"{len(backbone_params)} LoRA params use AdamW (lr={self.hparams.backbone_lr}), "
                f"{len(other_params)} other params use AdamW (lr={self.hparams.lr})"
            )
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            # Fallback to pure AdamW if Muon is disabled or no GNN matrices found
            param_groups = [
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
                {"params": other_params, "lr": self.hparams.lr},
            ]
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.hparams.weight_decay,
            )
            self.print("Using pure AdamW optimizer (Muon disabled)")

        # Cosine annealing with linear warmup
        warmup_epochs = self.hparams.warmup_epochs
        cosine_t_max = self.hparams.cosine_t_max

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=1e-7,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save trainable parameters and persistent buffers."""
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        trainable_state_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if key in full_sd:
                    trainable_state_dict[key] = full_sd[key]

        for name, buf in self.named_buffers():
            key = prefix + name
            if key in full_sd:
                trainable_state_dict[key] = full_sd[key]

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_buffers = sum(b.numel() for _, b in self.named_buffers())
        self.print(
            f"Saving checkpoint: {trainable_params}/{total_params} params "
            f"({100 * trainable_params / total_params:.2f}%), plus {total_buffers} buffer values"
        )

        return trainable_state_dict

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_per_gene_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import f1_score as sk_f1
    y_hat = preds.argmax(axis=1)
    n_genes = labels.shape[1]
    f1_vals = []
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n = min(len(pert_ids), len(preds))
    for i in range(n):
        rows.append({
            "idx": pert_ids[i],
            "input": symbols[i],
            "prediction": json.dumps(preds[i].tolist()),
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(rows)} test predictions → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node4-1-1-1-1-1-1-1-1-1-1: ESM2-650M (LoRA r=24) + STRING_GNN "
                    "(Muon+AdamW), NO ESM2 branch dropout, "
                    "PER-GENE BIAS (RESTORED 19920 params), "
                    "Manifold Mixup (alpha=0.2, prob=0.5), "
                    "head_hidden=512, head_dropout=0.45, "
                    "weight_decay=2e-3, lora_dropout=0.10, label_smoothing=0.03"
    )
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--backbone-lr", type=float, default=5e-5,
                   help="LR for ESM2 LoRA parameters")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="LR for head/GNN biases/fusion/cond_proj")
    p.add_argument("--weight-decay", type=float, default=2e-3,
                   help="Weight decay (RESTORED to 2e-3 per feedback)")
    p.add_argument("--head-hidden-dim", type=int, default=512)
    p.add_argument("--head-dropout", type=float, default=0.45)
    p.add_argument("--fusion-dropout", type=float, default=0.2)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.03,
                   help="Label smoothing (RESTORED to 0.03 per feedback)")
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--cosine-t-max", type=int, default=160)
    p.add_argument("--lora-r", type=int, default=24,
                   help="LoRA rank (r=24: better task adaptation than r=16)")
    p.add_argument("--lora-alpha", type=int, default=48,
                   help="LoRA alpha (48 maintains alpha/r=2.0 ratio with r=24)")
    p.add_argument("--lora-dropout", type=float, default=0.10)
    p.add_argument("--early-stop-patience", type=int, default=55)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-mixup", action="store_true",
                   help="Disable Manifold Mixup augmentation")
    p.add_argument("--mixup-alpha", type=float, default=0.2,
                   help="Beta distribution alpha for Manifold Mixup")
    p.add_argument("--mixup-prob", type=float, default=0.5,
                   help="Probability of applying Mixup per batch")
    p.add_argument("--tta-passes", type=int, default=8,
                   help="Number of TTA stochastic dropout passes at test time")
    p.add_argument("--save-top-k", type=int, default=10,
                   help="Number of top checkpoints to save")
    p.add_argument("--no-muon", action="store_true",
                   help="Disable Muon optimizer (fall back to pure AdamW)")
    p.add_argument("--muon-lr", type=float, default=0.01,
                   help="Muon LR for STRING_GNN hidden matrices")
    p.add_argument("--debug_max_step", type=int, default=None)
    p.add_argument("--fast_dev_run", action="store_true")
    p.add_argument("--val_check_interval", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(0)

    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = PerturbDataModule(
        train_path=str(data_dir / "train.tsv"),
        val_path=str(data_dir / "val.tsv"),
        test_path=str(data_dir / "test.tsv"),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        max_seq_len=ESM2_MAX_LEN,
    )

    model = PerturbModule(
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        fusion_dropout=args.fusion_dropout,
        backbone_lr=args.backbone_lr,
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        cosine_t_max=args.cosine_t_max,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_mixup=not args.no_mixup,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        tta_passes=args.tta_passes,
        use_muon=not args.no_muon,
        muon_lr=args.muon_lr,
    )
    model.micro_batch_size = args.micro_batch_size  # For batch_size in log calls

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    accumulate = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        limit_train = limit_val = limit_test = args.debug_max_step
        max_steps = args.debug_max_step
        val_check_interval = 1.0
    else:
        limit_train = limit_val = limit_test = 1.0
        max_steps = -1
        val_check_interval = args.val_check_interval

    save_top_k = 1 if (fast_dev_run or args.debug_max_step is not None) else args.save_top_k
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-ep{epoch:03d}",
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

    csv_logger = CSVLogger(save_dir=str(output_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir / "logs"), name="tensorboard_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=600)),
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, progress_bar],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=datamodule)

    # Test phase
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
        if trainer.is_global_zero and test_results:
            print(f"Test results: {test_results}")
        # Graceful DDP shutdown to prevent NCCL watchdog hangs after fast_dev_run/debug runs.
        # This is a known issue with Lightning + DDP + ESM2 PEFT + Muon: pending NCCL
        # collectives from training can cause the NCCL watchdog to timeout ~2 min after
        # test completes. We force process group destruction to prevent torchrun failures.
        # Production runs (no debug flags) complete normally without this issue.
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass
        return

    # Production: use best single checkpoint (no SWA — proven zero benefit in parent)
    test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")
    if trainer.is_global_zero and test_results:
        print(f"Test results: {test_results}")
    # Graceful DDP shutdown to prevent NCCL watchdog hangs after test completes.
    # This prevents pending NCCL collectives from causing timeout errors during exit.
    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
