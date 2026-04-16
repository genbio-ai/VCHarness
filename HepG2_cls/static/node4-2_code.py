"""Node 4-2: Frozen ESM2-650M + STRING_GNN (4-Layer Tiered Fine-tune) + Muon+AdamW +
CosineWarmRestarts + Manifold Mixup for HepG2 DEG prediction.

Architecture:
  - Branch A — Frozen ESM2-650M (no gradient):
      Protein sequence → ESM2 mean-pool → [B, 1280]  (fixed feature vector)
  - Branch B — STRING_GNN (4-layer tiered fine-tune, mps.4-7 + post_mp):
      Run STRING_GNN → extract node embedding for perturbed gene → [B, 256]
      Tiered LR: mps.6-7 + post_mp at gnn_lr_near=5e-5; mps.4-5 at gnn_lr_far=3e-5
  - Gated Fusion:
      Project both branches to shared dim D=512
      Gate = sigmoid(W_a * a + W_b * b + bias)
      fused = gate * a_proj + (1-gate) * b_proj → [B, 512]
      fusion_dropout = 0.15
  - Head: 3-block PreNorm Residual MLP (hidden=384) + per-gene bias → [B, 3, 6640]
  - Loss: Weighted Cross-Entropy (no focal loss — Muon incompatible with focal)
  - Optimizer: Muon+AdamW dual optimizer
  - Schedule: LambdaLR CosineWarmRestarts (T_0=100, T_mult=2) + 10-epoch linear warmup
  - Regularization: Manifold Mixup (prob=0.65, alpha=0.2)

Key improvements over node4-1 (sibling):
  1. Frozen ESM2 (not LoRA) — tree-best nodes F1=0.5175-0.5283 all use frozen ESM2
  2. STRING_GNN 4-layer tiered fine-tune (not cond_emb) — proven recipe from node2-1-2-1-1-1-1
  3. Muon+AdamW dual optimizer (not single AdamW) — accelerates MLP convergence
  4. WCE loss (not focal) — Muon + focal loss causes catastrophic collapse
  5. CosineWarmRestarts T_0=100 (not CosineAnnealing T_max=140) — warm restarts escape local optima
  6. Manifold Mixup prob=0.65 — strongest regularization for 1273-sample dataset
  7. No label smoothing — conflicts with WCE minority-class weighting
  8. Correct WCE class weights: freq=[0.0477, 0.9282, 0.0241] (bug fix from node2-1-2-1-1-1-1)

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
HEAD_HIDDEN = 384       # hidden dim for 3-block PreNorm MLP

# WCE class frequencies (correct order from DATA_ABSTRACT.md):
#   class 0 = down-regulated  = 4.77%
#   class 1 = neutral          = 92.82%
#   class 2 = up-regulated     = 2.41%
# NOTE: do NOT swap class 0 and class 1 (bug present in node2-1-2-1-1-1-1)
WCE_FREQ = torch.tensor([0.0477, 0.9282, 0.0241], dtype=torch.float32)


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
            "node_idx": self.node_indices[idx],
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
        # ESM2 tokenizer — rank 0 downloads first, then all ranks load
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
# Gated Fusion with dropout
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Learnable gated fusion of two embeddings.

    gate = sigmoid(W_a * a + W_b * b + bias)
    fused = gate * a_proj + (1 - gate) * b_proj
    """

    def __init__(self, dim_a: int, dim_b: int, out_dim: int, dropout: float = 0.15) -> None:
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
        return self.dropout(self.norm(fused))   # [B, out_dim]


# ---------------------------------------------------------------------------
# 3-Block PreNorm Residual MLP with per-gene bias
# ---------------------------------------------------------------------------
class PreNormResBlock(nn.Module):
    """Pre-LayerNorm Residual Block: LN -> Linear -> GELU -> Linear -> Dropout -> residual."""
    def __init__(self, hidden_dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class PerturbHead(nn.Module):
    """3-block PreNorm ResidualMLP with per-gene bias for perturbation DEG prediction."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = HEAD_HIDDEN,
        n_genes: int = N_GENES,
        head_dropout: float = 0.15,
        n_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        # Residual blocks
        self.blocks = nn.ModuleList([
            PreNormResBlock(hidden_dim, dropout=head_dropout) for _ in range(n_blocks)
        ])
        # Output head with dropout
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_drop = nn.Dropout(head_dropout)
        self.out_linear = nn.Linear(hidden_dim, n_genes * N_CLASSES)
        # Per-gene bias: [N_CLASSES, N_GENES] — captures gene-specific baseline tendencies
        self.gene_bias = nn.Parameter(torch.zeros(N_CLASSES, n_genes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        x = self.input_norm(self.input_proj(x))   # [B, hidden_dim]
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        x = self.out_drop(x)
        logits = self.out_linear(x).view(-1, N_CLASSES, self.n_genes)  # [B, 3, N_GENES]
        logits = logits + self.gene_bias.unsqueeze(0)                   # broadcast gene_bias
        return logits


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        head_hidden_dim: int = HEAD_HIDDEN,
        muon_lr: float = 0.01,
        adamw_lr: float = 3e-4,
        gnn_lr_near: float = 5e-5,
        gnn_lr_far: float = 3e-5,
        weight_decay: float = 8e-4,
        warmup_epochs: int = 10,
        T_0: int = 100,
        T_mult: int = 2,
        mixup_prob: float = 0.65,
        mixup_alpha: float = 0.2,
        head_dropout: float = 0.15,
        fusion_dropout: float = 0.15,
        max_epochs: int = 400,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Will be initialized in setup()
        self.esm2: Optional[EsmForMaskedLM] = None
        self.string_gnn = None
        self.fusion: Optional[GatedFusion] = None
        self.head: Optional[PerturbHead] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # ESM2-650M — frozen, float32 for stable embedding extraction
        self.esm2 = EsmForMaskedLM.from_pretrained(ESM2_MODEL, dtype=torch.float32)
        # Freeze all ESM2 parameters — no gradient flows through ESM2
        for p in self.esm2.parameters():
            p.requires_grad = False
        self.esm2.eval()

        # STRING_GNN — 4-layer tiered fine-tuning
        self.string_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        # Freeze all STRING_GNN layers first
        for p in self.string_gnn.parameters():
            p.requires_grad = False
        # Unfreeze tiered layers: near (mps.6, mps.7, post_mp) and far (mps.4, mps.5)
        for layer_name in ["mps.4", "mps.5", "mps.6", "mps.7", "post_mp"]:
            for name, p in self.string_gnn.named_parameters():
                if name.startswith(layer_name):
                    p.requires_grad = True

        # Load graph data as registered buffers
        graph = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", map_location="cpu")
        self.register_buffer("_graph_edge_index", graph["edge_index"].long())
        self.register_buffer("_graph_edge_weight", graph["edge_weight"].float())

        # Gated Fusion with dropout
        self.fusion = GatedFusion(
            dim_a=ESM2_HIDDEN,
            dim_b=GNN_HIDDEN,
            out_dim=FUSED_DIM,
            dropout=self.hparams.fusion_dropout,
        )

        # 3-block PreNorm MLP head with per-gene bias
        self.head = PerturbHead(
            in_dim=FUSED_DIM,
            hidden_dim=self.hparams.head_hidden_dim,
            n_genes=N_GENES,
            head_dropout=self.hparams.head_dropout,
        )

        # WCE loss with inverse-frequency class weights
        # IMPORTANT: correct ordering — class 0=down-reg(4.77%), class 1=neutral(92.82%), class 2=up-reg(2.41%)
        class_weights = (1.0 / WCE_FREQ)
        class_weights = class_weights / class_weights.mean()
        # Register as buffer so it automatically moves to the correct device with the module
        self.register_buffer("_wce_weights", class_weights)

        # Cast trainable parameters to float32 for stable optimization
        for name, p in self.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Node4-2: ESM2(frozen) + STRING_GNN(4-layer tiered) + Muon+AdamW | "
                   f"total={total:,} | trainable={trainable:,} ({100*trainable/total:.2f}%)")

    def _get_esm2_emb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """ESM2 mean pool over residues (excluding CLS, EOS, PAD). ESM2 is frozen.

        Returns: [B, 1280]
        """
        with torch.no_grad():
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
        sum_emb = (hidden.float() * valid_mask_f).sum(dim=1)   # [B, 1280]
        count = valid_mask_f.sum(dim=1).clamp(min=1e-9)        # [B, 1]
        return sum_emb / count                                   # [B, 1280]

    def _get_gnn_emb(self, node_idx: torch.Tensor) -> torch.Tensor:
        """Run STRING_GNN (partial fine-tune) and extract node embeddings.

        Args:
            node_idx: [B] int — STRING_GNN node indices; -1 means not in graph

        Returns: [B, 256]
        """
        device = node_idx.device
        out = self.string_gnn(
            edge_index=self._graph_edge_index,
            edge_weight=self._graph_edge_weight,
        )
        all_emb = out.last_hidden_state  # [N_nodes, 256]

        B = node_idx.size(0)
        result = torch.zeros(B, GNN_HIDDEN, dtype=all_emb.dtype, device=device)
        valid = node_idx >= 0
        if valid.any():
            result[valid] = all_emb[node_idx[valid]]
        return result.float()

    def _manifold_mixup(
        self,
        emb: torch.Tensor,
        labels: Optional[torch.Tensor],
        alpha: float = 0.2,
        prob: float = 0.65,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[float]]:
        """Apply Manifold Mixup in the fused embedding space.

        Returns: (mixed_emb, labels_a, labels_b, lam)
        """
        if labels is None or torch.rand(1).item() > prob:
            return emb, labels, None, None

        B = emb.size(0)
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(B, device=emb.device)
        mixed_emb = lam * emb + (1.0 - lam) * emb[perm]
        labels_a = labels
        labels_b = labels[perm]
        return mixed_emb, labels_a, labels_b, lam

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: Optional[torch.Tensor] = None,
        lam: Optional[float] = None,
    ) -> torch.Tensor:
        """WCE loss, optionally with mixup interpolation.
        Uses F.cross_entropy with registered buffer weights for DDP compatibility.
        """
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_a_flat = labels_a.reshape(-1)
        # _wce_weights is a registered buffer — automatically on the correct device
        w = self._wce_weights.float()

        if labels_b is not None and lam is not None:
            labels_b_flat = labels_b.reshape(-1)
            loss = lam * F.cross_entropy(logits_flat, labels_a_flat, weight=w) + \
                   (1.0 - lam) * F.cross_entropy(logits_flat, labels_b_flat, weight=w)
        else:
            loss = F.cross_entropy(logits_flat, labels_a_flat, weight=w)
        return loss

    def _forward_no_mixup(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward without mixup (for validation/test)."""
        esm2_emb = self._get_esm2_emb(batch["input_ids"], batch["attention_mask"])
        gnn_emb = self._get_gnn_emb(batch["node_idx"])
        fused = self.fusion(esm2_emb, gnn_emb)
        return self.head(fused)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        esm2_emb = self._get_esm2_emb(batch["input_ids"], batch["attention_mask"])
        gnn_emb = self._get_gnn_emb(batch["node_idx"])
        fused = self.fusion(esm2_emb, gnn_emb)  # [B, FUSED_DIM]

        # Apply Manifold Mixup in the fused embedding space
        labels = batch.get("label")
        mixed_fused, labels_a, labels_b, lam = self._manifold_mixup(
            fused, labels,
            alpha=self.hparams.mixup_alpha,
            prob=self.hparams.mixup_prob,
        )

        logits = self.head(mixed_fused)

        if labels_a is not None:
            loss = self._compute_loss(logits, labels_a, labels_b, lam)
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward_no_mixup(batch)
        if "label" in batch:
            loss = self._compute_loss(logits, batch["label"])
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(logits.detach().cpu().float())
        if "label" in batch:
            self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds, dim=0).numpy()
        self._val_preds.clear()
        if self._val_labels:
            labels = torch.cat(self._val_labels, dim=0).numpy()
            self._val_labels.clear()
            f1 = _compute_per_gene_f1(preds, labels)
            # sync_dist=True ensures cross-rank averaging for checkpoint monitoring
            self.log("val/f1", f1, prog_bar=True, sync_dist=True)
        else:
            self._val_labels.clear()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        logits = self._forward_no_mixup(batch)
        self._test_preds.append(logits.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])

    def on_test_epoch_end(self) -> None:
        preds_local = torch.cat(self._test_preds, dim=0)
        self._test_preds.clear()

        all_preds = self.all_gather(preds_local)

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
        self._test_pert_ids.clear()
        self._test_symbols.clear()

        if self.trainer is not None and self.trainer.world_size > 1:
            all_preds = all_preds.view(-1, N_CLASSES, N_GENES)

        if self.trainer is not None and self.trainer.is_global_zero:
            n = min(len(all_pert_ids), len(all_preds))
            # De-duplicate by pert_id (in case of padding in DDP)
            seen = {}
            dedup_ids, dedup_syms, dedup_preds = [], [], []
            for i in range(n):
                pid = all_pert_ids[i]
                if pid not in seen:
                    seen[pid] = True
                    dedup_ids.append(pid)
                    dedup_syms.append(all_symbols[i])
                    dedup_preds.append(all_preds[i])
            if dedup_preds:
                final_preds = torch.stack(dedup_preds).float().cpu().numpy()
                _save_test_predictions(
                    pert_ids=dedup_ids,
                    symbols=dedup_syms,
                    preds=final_preds,
                    out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
                )

    def configure_optimizers(self):
        """Muon+AdamW dual optimizer with LambdaLR CosineWarmRestarts and linear warmup.

        Parameter groups:
          - Muon group: 2D weight matrices in MLP head blocks (NOT GNN, NOT ESM2, NOT output layer)
          - AdamW group 1: GNN near layers (mps.6, mps.7, post_mp) at gnn_lr_near
          - AdamW group 2: GNN far layers (mps.4, mps.5) at gnn_lr_far
          - AdamW group 3: all other trainable params (fusion, head norms/biases/gene_bias, output linear)
        """
        try:
            from muon import MuonWithAuxAdam
            use_muon = True
        except ImportError:
            use_muon = False
            self.print("WARNING: muon not found, falling back to AdamW for all params")

        # Identify parameter groups
        # GNN near layers: mps.6, mps.7, post_mp
        gnn_near_params = []
        gnn_near_ids = set()
        for name, p in self.string_gnn.named_parameters():
            if p.requires_grad and (name.startswith("mps.6") or name.startswith("mps.7") or name.startswith("post_mp")):
                gnn_near_params.append(p)
                gnn_near_ids.add(id(p))

        # GNN far layers: mps.4, mps.5
        gnn_far_params = []
        gnn_far_ids = set()
        for name, p in self.string_gnn.named_parameters():
            if p.requires_grad and (name.startswith("mps.4") or name.startswith("mps.5")):
                gnn_far_params.append(p)
                gnn_far_ids.add(id(p))

        # MLP head 2D hidden weight matrices (for Muon): only ResBlock fc1/fc2 weights
        # Exclude: gene_bias (not a weight matrix), out_linear (output layer), input_proj (input layer)
        muon_params = []
        muon_ids = set()
        for name, p in self.head.named_parameters():
            if p.requires_grad and p.ndim >= 2 and ("blocks" in name) and ("fc1" in name or "fc2" in name):
                muon_params.append(p)
                muon_ids.add(id(p))

        # All other trainable parameters → AdamW at adamw_lr
        muon_adamw_other_params = []
        gnn_all_ids = gnn_near_ids | gnn_far_ids
        for name, p in self.named_parameters():
            if p.requires_grad and id(p) not in muon_ids and id(p) not in gnn_all_ids:
                muon_adamw_other_params.append(p)

        if use_muon and muon_params:
            param_groups = [
                # Muon group for hidden weight matrices in MLP blocks
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=self.hparams.muon_lr,
                    weight_decay=self.hparams.weight_decay,
                    momentum=0.95,
                ),
                # AdamW group for GNN near layers (higher LR)
                dict(
                    params=gnn_near_params,
                    use_muon=False,
                    lr=self.hparams.gnn_lr_near,
                    betas=(0.9, 0.95),
                    weight_decay=self.hparams.weight_decay,
                ),
                # AdamW group for GNN far layers (lower LR)
                dict(
                    params=gnn_far_params,
                    use_muon=False,
                    lr=self.hparams.gnn_lr_far,
                    betas=(0.9, 0.95),
                    weight_decay=self.hparams.weight_decay,
                ),
                # AdamW group for everything else
                dict(
                    params=muon_adamw_other_params,
                    use_muon=False,
                    lr=self.hparams.adamw_lr,
                    betas=(0.9, 0.95),
                    weight_decay=self.hparams.weight_decay,
                ),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            # Fallback: standard AdamW with different LR groups
            optimizer = torch.optim.AdamW(
                [
                    {"params": gnn_near_params, "lr": self.hparams.gnn_lr_near},
                    {"params": gnn_far_params, "lr": self.hparams.gnn_lr_far},
                    {"params": muon_params + muon_adamw_other_params, "lr": self.hparams.adamw_lr},
                ],
                weight_decay=self.hparams.weight_decay,
            )

        # LambdaLR implementing CosineWarmRestarts with linear warmup
        # Schedule: [0..warmup_epochs]: linear ramp from 0 to 1
        #           [warmup_epochs..T_0 + warmup_epochs]: cosine decay from 1 to 0
        #           then restart with T_mult expansion
        warmup_epochs = self.hparams.warmup_epochs
        T_0 = self.hparams.T_0
        T_mult = self.hparams.T_mult

        def cosine_warmrestart_with_warmup(epoch: int) -> float:
            """Lambda LR multiplier for CosineWarmRestarts with linear warmup.

            epoch 0..warmup-1: linear warmup from 0 to 1
            epoch warmup+: cosine warm restarts (T_0=100, T_mult=2)
            """
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)

            # After warmup: CosineWarmRestarts
            t = epoch - warmup_epochs
            T_cur = T_0
            t_remaining = t
            while t_remaining >= T_cur:
                t_remaining -= T_cur
                T_cur = T_cur * T_mult

            # Cosine decay within current restart cycle
            cos_val = 0.5 * (1.0 + math.cos(math.pi * t_remaining / T_cur))
            return cos_val

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=cosine_warmrestart_with_warmup
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
        """Save only trainable parameters and persistent buffers."""
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
        """Load trainable parameters from partial checkpoint."""
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
    p = argparse.ArgumentParser(description="Node4-2: Frozen ESM2 + STRING_GNN 4-layer tiered + Muon")
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=400)
    p.add_argument("--muon-lr", type=float, default=0.01,
                   help="Muon learning rate for MLP hidden weight matrices")
    p.add_argument("--adamw-lr", type=float, default=3e-4,
                   help="AdamW learning rate for non-Muon parameters")
    p.add_argument("--gnn-lr-near", type=float, default=5e-5,
                   help="LR for STRING_GNN near layers (mps.6, mps.7, post_mp)")
    p.add_argument("--gnn-lr-far", type=float, default=3e-5,
                   help="LR for STRING_GNN far layers (mps.4, mps.5)")
    p.add_argument("--weight-decay", type=float, default=8e-4)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--T0", type=int, default=100, dest="T_0",
                   help="CosineWarmRestarts period T_0")
    p.add_argument("--Tmult", type=int, default=2, dest="T_mult",
                   help="CosineWarmRestarts multiplier T_mult")
    p.add_argument("--mixup-prob", type=float, default=0.65)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--head-dropout", type=float, default=0.15)
    p.add_argument("--fusion-dropout", type=float, default=0.15)
    p.add_argument("--head-hidden-dim", type=int, default=HEAD_HIDDEN)
    p.add_argument("--early-stop-patience", type=int, default=80,
                   help="Early stopping patience — must be large enough to allow CosineWR restarts")
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
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        gnn_lr_near=args.gnn_lr_near,
        gnn_lr_far=args.gnn_lr_far,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        T_0=args.T_0,
        T_mult=args.T_mult,
        mixup_prob=args.mixup_prob,
        mixup_alpha=args.mixup_alpha,
        head_dropout=args.head_dropout,
        fusion_dropout=args.fusion_dropout,
        max_epochs=args.max_epochs,
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

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-val_f1={val/f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
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

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    if trainer.is_global_zero and test_results:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(json.dumps(test_results, indent=2))
        print(f"Test results → {score_path}")


if __name__ == "__main__":
    main()
