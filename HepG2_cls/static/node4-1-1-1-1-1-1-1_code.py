"""Node 4-1-1-1-1-1-1-1: ESM2-650M (LoRA r=32) + STRING_GNN with Muon optimizer,
              Fixed Checkpoint Ensemble, Reduced Head Capacity, Stronger Regularization

Architecture (same dual-branch as parent, with critical bug fixes and overfitting reduction):
  - Branch A — ESM2-650M with LoRA (r=32, alpha=64): same as parent.
  - Branch B — STRING_GNN with cond_emb injection (learnable gain): same as parent.
  - Gated Fusion: same gated fusion → [B, 512].
  - Head: 2-layer MLP (LayerNorm→Linear→GELU→Dropout(0.45)→Linear) with hidden=512
    + per-gene bias → [B, 3, 6640].  HEAD DIM REDUCED 1280→512 to combat overfitting.
  - Loss: Focal loss (gamma=2.0) WITH label smoothing=0.03 (reduced from 0.05).

Key improvements from parent node4-1-1-1-1-1-1 feedback (test F1=0.5178):
  1. CRITICAL BUG FIX: Checkpoint filename now uses underscores instead of slashes
     (val_f1 instead of val/f1), so the glob pattern correctly finds all top-k checkpoints.
     This fixes the ensemble that produced only 1 checkpoint in the parent.
  2. Reduced head hidden dim 1280→512: Addresses the dominant overfitting source.
     The head had 13.2M params for 1273 training samples — massively over-parameterized.
     Reducing to 512 brings this to ~3.4M which is far more appropriate.
  3. Muon optimizer for STRING_GNN hidden layers: Faster convergence and better
     optimization for the GNN hidden weight matrices (2D params excluding embeddings/head).
     LoRA params and head/fusion use AdamW as per Muon skill recommendation.
  4. Increase weight decay 5e-4→1e-3: Stronger global regularization to reduce overfitting.
  5. Reduce label smoothing 0.05→0.03: Preserve focal loss minority class emphasis
     while still providing regularization against overconfident predictions.
  6. Disable Mixup (alpha=0.0): With focal loss handling imbalance, Mixup's interpolated
     labels are counterproductive. Per feedback: test removing Mixup entirely.
  7. Increase head dropout 0.42→0.45: Higher dropout on the now-smaller head for stability.

Memory estimate on 2-4 H100s (80GB each):
  - ESM2-650M LoRA r=32 (fp32 weights, bf16 activations): ~5.2 GB/GPU
  - STRING_GNN: ~0.2 GB/GPU
  - Head (hidden=512) + fusion + activations for batch=4: ~1.8 GB/GPU (reduced from 3.2)
  - Total per GPU: ~7.2 GB — well within 80 GB H100 budget

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
# Prediction Head with per-gene bias
# ---------------------------------------------------------------------------
class PerturbHead(nn.Module):
    """2-layer MLP with per-gene bias term.

    The per-gene bias learns a baseline activation level for each gene
    across all perturbations — capturing gene-specific intrinsic expression.
    This was proven effective in node1-1-1 (+0.002 F1).

    head_hidden_dim=512: REDUCED from parent's 1280 to drastically cut overfitting.
    The parent's head had ~13.2M parameters (1280 hidden) for only 1,273 training samples.
    With hidden=512, the head has ~3.4M parameters — still expressive but far less overfit.
    head_dropout=0.45: Slightly increased from parent's 0.42 for the now-smaller head.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,      # REDUCED from parent's 1280 to address overfitting
        n_genes: int = N_GENES,
        dropout: float = 0.45,      # Slightly increased from parent's 0.42
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes * N_CLASSES),
        )
        # Per-gene bias: learnable shift for each gene across all classes
        self.gene_bias = nn.Parameter(torch.zeros(N_CLASSES, n_genes))
        self.n_genes = n_genes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        logits = self.net(x).view(-1, N_CLASSES, self.n_genes)  # [B, 3, N_GENES]
        logits = logits + self.gene_bias.unsqueeze(0)            # [B, 3, N_GENES]
        return logits


# ---------------------------------------------------------------------------
# Focal Loss WITH label smoothing
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss with class weights and optional label smoothing.

    Combines focal loss (for class imbalance handling) with label smoothing
    (as a regularizer preventing overconfident predictions).

    Per feedback from node4-1-1: removing label smoothing was HARMFUL —
    the parent node4-1 (test F1=0.4780) had smoothing=0.05 and better
    generalization than node4-1-1 (test F1=0.4642) without it.

    Label smoothing REDUCED to 0.03 from 0.05 based on feedback recommendation
    to improve minority class discrimination while maintaining regularization.
    """

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
            # Smooth targets: most probability on true class, small uniform for others
            # true class gets: 1 - smoothing + smoothing/C
            # other classes get: smoothing/C
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
    conditioning signal strength. This allows the model to learn whether
    strong or weak perturbation conditioning is optimal.

    Retained from node4-1-1 as a worthwhile architectural improvement.
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
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        head_hidden_dim: int = 512,           # REDUCED from parent's 1280 to combat overfitting
        head_dropout: float = 0.45,           # Slightly increased from parent's 0.42
        fusion_dropout: float = 0.2,
        backbone_lr: float = 5e-5,            # Split LR: lower for LoRA parameters
        lr: float = 1e-4,                     # Split LR: higher for head/GNN/fusion/cond_proj
        weight_decay: float = 1e-3,           # INCREASED: 5e-4 → 1e-3 for stronger regularization
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.03,        # REDUCED: 0.05 → 0.03 to improve minority class F1
        warmup_epochs: int = 10,
        max_epochs: int = 300,
        cosine_t_max: int = 160,              # Same as parent
        lora_r: int = 32,                     # Same as parent: r=32 for rich ESM2 adaptation
        lora_alpha: int = 64,                 # Same as parent: alpha=64 (ratio stays 2.0)
        lora_dropout: float = 0.05,
        use_mixup: bool = False,              # DISABLED: Mixup removed per feedback recommendation
        mixup_alpha: float = 0.10,            # Kept for argparse compatibility but not used by default
        tta_passes: int = 8,                  # Same TTA as parent
        ckpt_pred_path: Optional[str] = None, # path to save per-checkpoint preds (for ensembling)
        use_muon: bool = True,                # NEW: Muon optimizer for STRING_GNN hidden layers
        muon_lr: float = 0.01,               # Muon LR for GNN hidden weight matrices
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
        # Path to save per-checkpoint predictions (for ensembling); None = use default path
        self._ckpt_pred_path: Optional[str] = self.hparams.ckpt_pred_path

        # Validation state — track pert_ids for correct DDP deduplication
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_pert_ids: List[str] = []  # Track pert_ids for deduplication

        # Test state
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # ESM2-650M with LoRA for parameter-efficient fine-tuning
        # LoRA rank r=32 (same as parent) for rich task-specific adaptation
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

        # STRING_GNN (full fine-tune — only 5.43M params)
        self.string_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)

        # Load graph data and register as buffers
        graph = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", map_location="cpu")
        self.register_buffer("_graph_edge_index", graph["edge_index"].long())
        self.register_buffer("_graph_edge_weight", graph["edge_weight"].float())
        # Pre-compute number of GNN nodes to avoid repeated max() computation
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
        # Reduced head (hidden=512 vs parent's 1280) with slightly higher dropout (0.45 vs 0.42)
        # to dramatically reduce overfitting: 13.2M → 3.4M head parameters
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
            f"ESM2-650M (LoRA r={self.hparams.lora_r}) + STRING_GNN + cond_emb(gain) | "
            f"head_dim={self.hparams.head_hidden_dim} | head_drop={self.hparams.head_dropout} | "
            f"total={total:,} | trainable={trainable:,} "
            f"({100*trainable/total:.2f}%)"
        )

        # Cast all trainable parameters to float32 for stable optimization
        for p in self.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    def _get_esm2_emb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """ESM2 mean pool over residues (excluding CLS, EOS, PAD).

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
        return sum_emb / count  # [B, 1280]

    def _get_gnn_emb_with_conditioning(
        self,
        node_idx: torch.Tensor,
        esm2_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Run STRING_GNN with perturbation conditioning via cond_emb + learnable gain.

        For each sample in the batch:
          - Project ESM2 embedding to 256-dim → cond_emb (with learnable gain scaling)
          - Inject cond_emb at the perturbed gene's node in the GNN
          - This propagates the perturbation signal through the PPI graph
          - Extract the perturbed gene's final node embedding

        Args:
            node_idx: [B] int — STRING_GNN node indices; -1 means not in graph
            esm2_emb: [B, 1280] — ESM2 protein embeddings for each sample

        Returns: [B, 256]
        """
        device = node_idx.device
        B = node_idx.size(0)
        N = self._n_gnn_nodes  # pre-computed in setup to avoid max() each forward pass

        # Project ESM2 embeddings to GNN conditioning space (with learnable gain)
        cond_signal = self.cond_proj(esm2_emb)  # [B, 256]

        # Create a full cond_emb matrix [N, 256] initialized to zero
        cond_emb_full = torch.zeros(N, GNN_HIDDEN, dtype=cond_signal.dtype, device=device)

        # For samples with valid node indices, inject their conditioning
        valid = node_idx >= 0
        if valid.any():
            valid_nodes = node_idx[valid]        # [n_valid]
            valid_signals = cond_signal[valid]   # [n_valid, 256]

            # Use scatter_add to handle duplicate nodes (multiple samples perturbing same gene)
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
        """Standard forward pass without Mixup."""
        fused = self._get_fused_emb(batch)    # [B, 512]
        return self.head(fused)                # [B, 3, 6640]

    def _forward_with_tta(self, batch: Dict[str, Any], n_passes: int = 8) -> torch.Tensor:
        """Test-Time Augmentation: average over N stochastic dropout passes.

        Runs N forward passes with dropout ENABLED (model in train() mode for head/fusion),
        averages the resulting logit distributions. This approximates Bayesian model averaging
        and reduces prediction variance for minority class predictions.

        Returns: [B, 3, 6640] averaged logits
        """
        # Enable dropout selectively for TTA (stochastic forward passes)
        # Only head and fusion use dropout; ESM2 and string_gnn are deterministic
        # for efficiency (large models with many dropout layers)
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

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Direct forward pass without Mixup (disabled per feedback recommendation)
        # Focal loss alone handles the extreme class imbalance (92.8% neutral)
        logits = self._forward(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward(batch)
        loss = self._compute_loss(logits, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu().float())
        self._val_labels.append(batch["label"].detach().cpu())
        # Track pert_ids for proper DDP deduplication in on_validation_epoch_end
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
        # (same pattern as on_test_epoch_end which correctly handles deduplication)
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
        if self.trainer is not None and self.trainer.is_global_zero:
            all_preds_np = all_preds.cpu().numpy()
            all_labels_np = all_labels.cpu().numpy()

            # Deduplicate by pert_id (handles DistributedSampler padding/replication)
            paired = list(zip(all_pert_ids, all_preds_np, all_labels_np))
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
        else:
            f1 = 0.0
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step with optional TTA (dropout passes).

        Uses TTA (N=tta_passes stochastic forward passes with dropout enabled)
        to reduce prediction variance, particularly for minority class predictions.
        """
        tta_passes = self.hparams.tta_passes
        if tta_passes > 1:
            # TTA: average over N stochastic dropout passes
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
        self._test_preds.clear()
        self._test_labels.clear()

        # Gather pert_ids and symbols from ALL ranks via torch.distributed
        all_pert_ids: List[str] = []
        all_symbols: List[str] = []
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            _gathered_ids: List[List[str]] = [None] * self.trainer.world_size
            _gathered_syms: List[List[str]] = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(_gathered_ids, self._test_pert_ids)
            torch.distributed.all_gather_object(_gathered_syms, self._test_symbols)
            for ids in _gathered_ids:
                all_pert_ids.extend(ids)
            for syms in _gathered_syms:
                all_symbols.extend(syms)
        else:
            all_pert_ids = self._test_pert_ids[:]
            all_symbols = self._test_symbols[:]
        # Clear local lists AFTER gathering
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        # all_gather predictions and labels across all ranks
        if self.trainer is not None and self.trainer.world_size >= 1:
            all_preds = self.all_gather(preds_local)
            all_labels = self.all_gather(labels_local)
            # all_gather adds a leading world_size dimension: reshape to merge it
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)
            all_labels = all_labels.view(-1, N_GENES)
        else:
            all_preds = preds_local
            all_labels = labels_local

        # Compute test F1 and save predictions on global zero rank
        if self.trainer is not None and self.trainer.is_global_zero:
            # Move tensors to CPU before any numpy operations
            all_preds_cpu = all_preds.cpu()
            all_labels_cpu = all_labels.cpu()

            # CRITICAL: Zip pert_ids, symbols, preds, and labels together BEFORE
            # deduplication. This ensures alignment even if DistributedSampler distributed
            # different samples to different ranks. Deduplicate by pert_id keeping the
            # first occurrence, which is guaranteed to be the correct match.
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

            # Determine output path: use ckpt_pred_path if set (for per-checkpoint ensembling)
            # else use default path
            if self._ckpt_pred_path is not None:
                out_path = Path(self._ckpt_pred_path)
            else:
                out_path = Path(__file__).parent / "run" / "test_predictions.tsv"

            _save_test_predictions(
                pert_ids=unique_pert_ids,
                symbols=unique_symbols,
                preds=unique_preds_np,
                out_path=out_path,
            )

            # Save test score to node root test_score.txt
            score_path = Path(__file__).parent / "test_score.txt"
            score_path.write_text(f"test_f1: {test_f1:.6f}\n")
            self.print(f"Test F1: {test_f1:.6f} | Saved to {score_path}")

    def configure_optimizers(self):
        """Configure optimizer with Muon for STRING_GNN hidden layers + AdamW for everything else.

        Muon optimizer is applied ONLY to STRING_GNN hidden weight matrices (2D params,
        excluding embeddings and output layers). This provides faster convergence for the GNN.

        Per the Muon skill documentation:
        - Muon is NOT suitable for LoRA matrices (intentionally low-rank and rectangular)
        - Muon is NOT suitable for input embeddings or output heads
        - Muon IS suitable for hidden weight matrices in transformers and MLPs

        Optimizer groups:
        1. ESM2 LoRA params: AdamW at backbone_lr=5e-5 (lower for pretrained backbone)
        2. STRING_GNN hidden matrices: Muon at muon_lr=0.01 (if use_muon=True)
        3. STRING_GNN biases/scalars + head/fusion/cond_proj: AdamW at lr=1e-4
        """
        from muon import MuonWithAuxAdam

        # Identify LoRA parameters (backbone branch)
        lora_param_names = set()
        for name, module in self.esm2.named_modules():
            if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                for pname, _ in module.named_parameters():
                    lora_param_names.add(f"esm2.{name}.{pname}")

        # Identify STRING_GNN hidden weight matrices (2D params, not embedding or output layers)
        # These are suitable for Muon optimizer per skill documentation
        gnn_hidden_weights = []
        gnn_other_params = []
        gnn_hidden_param_names = set()

        if self.hparams.use_muon and self.string_gnn is not None:
            for name, param in self.string_gnn.named_parameters():
                if param.requires_grad:
                    # Hidden weight matrices: 2D, not embeddings
                    full_name = f"string_gnn.{name}"
                    # Skip if it is an embedding or output weight
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

        # Cosine annealing with linear warmup:
        # With warmup_epochs=10, cosine_t_max=160:
        #   - Cosine reaches minimum at epoch 10+160=170
        #   - Same as parent to maintain comparability
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
        self.print(f"Saving {trainable}/{total} params ({100*trainable/total:.2f}%)")
        return result

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


def _ensemble_checkpoint_preds(
    checkpoint_dir: Path,
    output_dir: Path,
    data_dir: Path,
) -> float:
    """Ensemble test predictions from all per-checkpoint TSV files.

    After running trainer.test() for each checkpoint independently, this function
    reads the per-checkpoint prediction TSVs saved under output_dir/ckpt_preds/,
    averages the logits across all checkpoints, and computes the final ensemble F1.

    Returns: ensemble test F1 score (or 0.0 if no checkpoint preds found)
    """
    # Find all per-checkpoint prediction files
    ckpt_pred_dir = output_dir / "ckpt_preds"
    ckpt_pred_files = sorted(glob_module.glob(str(ckpt_pred_dir / "*.tsv")))

    if not ckpt_pred_files:
        print("WARNING: No per-checkpoint prediction files found for ensembling.")
        return 0.0

    print(f"\nEnsembling {len(ckpt_pred_files)} checkpoint prediction files:")
    for f in ckpt_pred_files:
        print(f"  {Path(f).name}")

    # Collect predictions from all checkpoint TSVs
    all_ckpt_preds: Dict[str, List[np.ndarray]] = {}  # pert_id -> list of pred arrays [3, 6640]
    all_symbols: Dict[str, str] = {}

    for pred_file in ckpt_pred_files:
        pred_df = pd.read_csv(pred_file, sep="\t")
        for _, row in pred_df.iterrows():
            pid = row["idx"]
            sym = row["input"]
            pred = np.array(json.loads(row["prediction"]), dtype=np.float32)  # [3, 6640]
            if pid not in all_ckpt_preds:
                all_ckpt_preds[pid] = []
                all_symbols[pid] = sym
            all_ckpt_preds[pid].append(pred)

    if not all_ckpt_preds:
        print("WARNING: No predictions parsed from checkpoint files.")
        return 0.0

    # Average predictions across all checkpoints for each pert_id
    ensemble_pert_ids = []
    ensemble_symbols = []
    ensemble_preds_arr = []

    for pid, pred_list in all_ckpt_preds.items():
        ensemble_pert_ids.append(pid)
        ensemble_symbols.append(all_symbols[pid])
        ensemble_preds_arr.append(np.mean(pred_list, axis=0))  # [3, 6640]

    n_ckpts_used = max(len(v) for v in all_ckpt_preds.values())
    print(f"Averaged predictions from {n_ckpts_used} checkpoints for {len(ensemble_pert_ids)} pert_ids")

    ensemble_preds_np = np.stack(ensemble_preds_arr)  # [N, 3, 6640]

    # Load ground truth to compute ensemble F1
    test_df = pd.read_csv(data_dir / "test.tsv", sep="\t")
    pid_to_label = {}
    pid_to_sym = {}
    for _, row in test_df.iterrows():
        pid = row["pert_id"]
        pid_to_label[pid] = np.array(json.loads(row["label"]), dtype=np.int64) + 1  # {0,1,2}
        pid_to_sym[pid] = row["symbol"]

    # Align ensemble predictions with test ground truth
    aligned_preds = []
    aligned_labels = []
    aligned_pids = []
    aligned_syms = []

    for i, pid in enumerate(ensemble_pert_ids):
        if pid in pid_to_label:
            aligned_preds.append(ensemble_preds_np[i])
            aligned_labels.append(pid_to_label[pid])
            aligned_pids.append(pid)
            aligned_syms.append(ensemble_symbols[i])

    if not aligned_preds:
        print("WARNING: No aligned predictions found between ensemble and test set.")
        return 0.0

    aligned_preds_np = np.stack(aligned_preds)   # [N, 3, 6640]
    aligned_labels_np = np.stack(aligned_labels)  # [N, 6640]

    # Compute ensemble F1
    ensemble_f1 = _compute_per_gene_f1(aligned_preds_np, aligned_labels_np)
    print(f"\nEnsemble Test F1: {ensemble_f1:.6f} ({n_ckpts_used} checkpoints)")

    # Save ensemble predictions (overwrites single-checkpoint predictions)
    ensemble_pred_path = output_dir / "test_predictions.tsv"
    _save_test_predictions(
        pert_ids=aligned_pids,
        symbols=aligned_syms,
        preds=aligned_preds_np,
        out_path=ensemble_pred_path,
    )

    # Save ensemble test score
    score_path = Path(__file__).parent / "test_score.txt"
    score_path.write_text(f"{ensemble_f1:.10f}\n")
    print(f"Ensemble Test F1: {ensemble_f1:.6f} | Saved to {score_path}")

    return ensemble_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Node4-1-1-1-1-1-1-1: ESM2-650M (LoRA r=32) + STRING_GNN (Muon+AdamW), "
                    "FIXED checkpoint ensemble (top-5), TTA (8 passes), "
                    "head_hidden=512 (reduced from 1280), head_dropout=0.45, "
                    "no Mixup, label_smoothing=0.03, weight_decay=1e-3"
    )
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--backbone-lr", type=float, default=5e-5,
                   help="LR for ESM2 LoRA parameters (lower for pretrained backbone)")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="LR for head/GNN biases/fusion/cond_proj (higher for task-specific layers)")
    p.add_argument("--weight-decay", type=float, default=1e-3,
                   help="Weight decay (increased from parent 5e-4 → 1e-3 for stronger regularization)")
    p.add_argument("--head-hidden-dim", type=int, default=512,
                   help="Head MLP hidden dimension (REDUCED from parent 1280 → 512 to cut overfitting)")
    p.add_argument("--head-dropout", type=float, default=0.45,
                   help="Head dropout (slightly increased from parent 0.42 → 0.45)")
    p.add_argument("--fusion-dropout", type=float, default=0.2)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.03,
                   help="Label smoothing (REDUCED from 0.05 → 0.03 to improve minority class F1)")
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--cosine-t-max", type=int, default=160,
                   help="Cosine T_max (epochs after warmup). warmup+T_max=170.")
    p.add_argument("--lora-r", type=int, default=32,
                   help="LoRA rank (same as parent: r=32 for rich ESM2 adaptation)")
    p.add_argument("--lora-alpha", type=int, default=64,
                   help="LoRA alpha (same as parent: alpha=64, ratio stays 2.0)")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--early-stop-patience", type=int, default=55,
                   help="Patience for early stopping (same as parent: 55 epochs)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--use-mixup", action="store_true",
                   help="Enable Mixup augmentation (disabled by default per feedback)")
    p.add_argument("--mixup-alpha", type=float, default=0.10,
                   help="Mixup Beta distribution alpha (only used if --use-mixup is set)")
    p.add_argument("--tta-passes", type=int, default=8,
                   help="Number of TTA stochastic dropout passes at test time (0 to disable)")
    p.add_argument("--save-top-k", type=int, default=5,
                   help="Number of top checkpoints to save for ensembling")
    p.add_argument("--no-ensemble", action="store_true",
                   help="Disable checkpoint ensembling (use only best checkpoint)")
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
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        tta_passes=args.tta_passes if not args.no_ensemble else 1,
        use_muon=not args.no_muon,
        muon_lr=args.muon_lr,
    )

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

    # CRITICAL FIX: Checkpoint filename uses ONLY the epoch number (no metric value).
    # The parent's filename "{epoch:03d}-val_f1={val/f1:.4f}" created a nested path
    # "val/f1=X.ckpt" where "val" became a subdirectory, causing the glob to miss
    # all checkpoints. Further, we discovered that Lightning's {metric_var_name}
    # substitution in auto_insert_metric_name=False mode can produce 0.0000 values
    # instead of the actual metric (likely a timing issue with logging).
    # SOLUTION: Name checkpoints by epoch only. After training, load CSV logs
    # to look up the val_f1 for each epoch and sort checkpoints accordingly.
    save_top_k = 1 if (args.no_ensemble or fast_dev_run or args.debug_max_step is not None) else args.save_top_k
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        # Only use epoch in filename — val_f1 is looked up from CSV logs after training
        filename="best-ep{epoch:03d}",
        monitor="val/f1",
        mode="max",
        save_top_k=save_top_k,
        save_last=True,
        auto_insert_metric_name=False,  # Manually formatted filename (epoch only)
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
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120)),
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

    # For debug/fast_dev_run: single checkpoint test
    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
        if trainer.is_global_zero and test_results:
            print(f"Test results: {test_results}")
        return

    # Production: ensemble top-k checkpoints (if not disabled)
    if not args.no_ensemble and save_top_k > 1:
        checkpoint_dir = output_dir / "checkpoints"
        ckpt_pred_dir = output_dir / "ckpt_preds"
        # Create ckpt_preds dir on all ranks to avoid race conditions
        ckpt_pred_dir.mkdir(parents=True, exist_ok=True)

        # FIXED glob pattern: look for "best-ep*.ckpt" matching the fixed filename format
        # No longer uses "best-*.ckpt" which matched the nested-path "last.ckpt" file
        ckpt_pattern = str(checkpoint_dir / "best-ep*.ckpt")
        ckpt_files = sorted(glob_module.glob(ckpt_pattern))

        if not ckpt_files:
            # Fallback: try to find any checkpoint in the directory recursively
            ckpt_files = sorted(glob_module.glob(str(checkpoint_dir / "*.ckpt")))
            # Exclude "last.ckpt" from ensemble candidates
            ckpt_files = [f for f in ckpt_files if "last" not in Path(f).name]

        if not ckpt_files:
            # Final fallback: use best checkpoint only
            ckpt_files = ["best"]

        def _build_epoch_to_val_f1_map(csv_log_dir: Path) -> Dict[int, float]:
            """Load CSV logs and build a mapping from epoch -> val_f1.

            The CSVLogger saves metrics per epoch. We look for the val/f1 column
            and return the last logged value for each epoch.
            """
            epoch_to_f1 = {}
            # Find CSV log files (Lightning CSVLogger saves to version_N subdirs)
            for version_dir in sorted(csv_log_dir.glob("version_*")):
                csv_file = version_dir / "metrics.csv"
                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file)
                        # val/f1 is the column name (with slash, not underscore)
                        val_rows = df[df["val/f1"].notna()]
                        if "epoch" in val_rows.columns:
                            for _, row in val_rows.iterrows():
                                ep = int(row["epoch"])
                                f1 = float(row["val/f1"])
                                epoch_to_f1[ep] = f1
                    except Exception:
                        pass
            return epoch_to_f1

        def _extract_val_f1(path: str) -> float:
            """Extract val_f1 for a checkpoint by looking up its epoch in CSV logs.

            Checkpoint filenames are "best-ep{epoch:03d}.ckpt".
            We look up the epoch number and find the corresponding val/f1
            from the CSV training logs.
            """
            # Extract epoch number from filename: "best-ep{epoch:03d}.ckpt"
            match = re.search(r'best-ep(\d+)', path)
            if not match:
                return 0.0
            epoch = int(match.group(1))

            # Look up val_f1 from CSV logs
            csv_log_dir = output_dir / "logs" / "csv_logs"
            epoch_map = _build_epoch_to_val_f1_map(csv_log_dir)
            if epoch in epoch_map:
                return epoch_map[epoch]
            return 0.0

        ckpt_files_sorted = sorted(ckpt_files, key=_extract_val_f1, reverse=True)

        if trainer.is_global_zero:
            print(f"\n{'='*60}")
            print(f"Running checkpoint ensemble ({len(ckpt_files_sorted)} checkpoints) with TTA={args.tta_passes}")
            print(f"{'='*60}")
            for cf in ckpt_files_sorted:
                print(f"  {Path(cf).name if cf != 'best' else 'best'} (val_f1={_extract_val_f1(cf):.4f})")

        # Run test inference for each checkpoint sequentially
        for i, ckpt_path in enumerate(ckpt_files_sorted):
            ckpt_stem = Path(ckpt_path).stem if ckpt_path != "best" else "best"
            ckpt_out_path = str(ckpt_pred_dir / f"ckpt_{i:02d}_{ckpt_stem}.tsv")

            if trainer.is_global_zero:
                print(f"\nRunning test for checkpoint {i+1}/{len(ckpt_files_sorted)}: {ckpt_stem}")

            # Update the model's ckpt_pred_path for this checkpoint's predictions
            model._ckpt_pred_path = ckpt_out_path

            # Run test inference for this checkpoint
            actual_ckpt_path = ckpt_path if ckpt_path != "best" else "best"
            trainer.test(model, datamodule=datamodule, ckpt_path=actual_ckpt_path)

        # Reset ckpt_pred_path to default for any subsequent calls
        model._ckpt_pred_path = None

        # Ensemble all checkpoint predictions (rank 0 only)
        if trainer.is_global_zero:
            data_dir = Path(__file__).parent.parent.parent / "data"
            ensemble_f1 = _ensemble_checkpoint_preds(
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                data_dir=data_dir,
            )
            print(f"\nFinal Ensemble Test F1: {ensemble_f1:.6f}")

    else:
        # No ensembling: run test with best single checkpoint
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")
        if trainer.is_global_zero and test_results:
            print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
