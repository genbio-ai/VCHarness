"""Node 4-1-1: ESM2-650M (LoRA) + STRING_GNN (cond_emb with gain) Gated Fusion
            with fixed cosine schedule, unified LR, and improved DDP metrics.

Architecture (improvements over node4-1):
  - Branch A — ESM2-650M with LoRA (r=16, alpha=32): same as node4-1.
  - Branch B — STRING_GNN with enhanced cond_emb:
      Same cond_emb injection but adds a learnable gain parameter (scalar)
      that scales the projected ESM2 conditioning signal before injection,
      allowing the model to learn the appropriate conditioning strength.
  - Gated Fusion: same as node4-1.
  - Head: same 2-layer MLP + per-gene bias.
  - Loss: Focal loss (gamma=2.0) without label smoothing (label_smoothing=0.0).

Key fixes from node4-1 feedback:
  1. Cosine T_max reduced from 140 to 60: With warmup=10, the cosine reaches its
     minimum at epoch 70, exactly where node4-1 found its best checkpoint.
     Previously T_max=140 meant LR barely decayed before early stopping at epoch 90.
  2. Unified LR=5e-5 for ALL parameters (LoRA, GNN, head, cond_proj, fusion):
     node4-1's 1e-4 head/GNN LR caused continued aggressive updates post-peak,
     driving val loss up while train loss dropped. Using 5e-5 for all parameters
     removes the LR imbalance between backbone and head.
  3. sync_dist=True for val/f1 logging: Fixes the DDP checkpoint naming bug where
     val_f1 appeared as 0.0000 in the filename due to sync_dist=False.
  4. Removed label smoothing: Focal loss already handles class imbalance; label
     smoothing conflicts by softening minority-class confidence boosted by focal.
  5. Learnable gain parameter for cond_emb conditioning: A scalar parameter
     (initialized near 1.0) scales the projected ESM2 embedding before injection
     into the GNN. This lets the model learn whether strong or weak conditioning
     is optimal, rather than fixing the conditioning strength at the projection scale.

Memory estimate on 4 H100s (80GB each):
  - ESM2-650M LoRA (fp32 weights, bf16 activations): ~4.4 GB/GPU
  - STRING_GNN: ~0.2 GB/GPU
  - Head + fusion + activations for batch=4: ~3 GB/GPU
  - Total per GPU: ~8 GB — well within 80 GB H100 budget

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
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 1024,
        n_genes: int = N_GENES,
        dropout: float = 0.3,
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
# Focal Loss (no label smoothing)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss with class weights.

    Focal loss suppresses the dominant neutral class (92.8%).
    Label smoothing is NOT used here (unlike node4-1) to avoid conflict
    with focal loss's minority-class emphasis.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # [n_classes]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: [N, C], target: [N] (class indices)
        log_prob = F.log_softmax(input, dim=-1)  # [N, C]
        prob = torch.exp(log_prob)               # [N, C]

        # Focal weight: based on the true class probability
        p_t = prob.gather(1, target.unsqueeze(1)).squeeze(1)  # [N]
        focal_w = (1.0 - p_t) ** self.gamma                   # [N]

        # Standard cross-entropy (no smoothing)
        ce = F.nll_loss(log_prob, target, reduction="none")   # [N]

        # Combine focal weight
        loss = focal_w * ce  # [N]

        # Apply class weights
        if self.weight is not None:
            w = self.weight.to(device=input.device, non_blocking=True)
            loss = loss * w[target]

        return loss.mean()


# ---------------------------------------------------------------------------
# ESM2 cond_emb projector with learnable gain
# ---------------------------------------------------------------------------
class CondEmbProjector(nn.Module):
    """Projects ESM2 embeddings to STRING_GNN cond_emb dimension (256).

    Adds a learnable gain parameter (initialized to 1.0) to scale the
    conditioning signal strength. This allows the model to learn whether
    strong or weak perturbation conditioning is optimal, rather than fixing
    it at the projection scale.

    The gain addresses the feedback concern that additive cond_emb conditioning
    may be too weak to meaningfully alter GNN node representations.
    """

    def __init__(self, in_dim: int = ESM2_HIDDEN, out_dim: int = GNN_HIDDEN) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
        )
        # Learnable gain: initialized to 1.0, allows model to scale conditioning strength
        # Using softplus(log_gain) ensures gain > 0 while allowing gradient flow
        self.log_gain = nn.Parameter(torch.zeros(1))  # exp(0) = 1.0

    def forward(self, esm2_emb: torch.Tensor) -> torch.Tensor:
        # esm2_emb: [B, ESM2_HIDDEN]
        projected = self.proj(esm2_emb)  # [B, GNN_HIDDEN]
        gain = F.softplus(self.log_gain)  # positive scalar, ~1.0 initially
        return projected * gain  # [B, GNN_HIDDEN]


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class PerturbModule(LightningModule):
    def __init__(
        self,
        head_hidden_dim: int = 1024,
        head_dropout: float = 0.3,
        fusion_dropout: float = 0.2,
        lr: float = 5e-5,          # Unified LR for all params (LoRA + GNN + head)
        weight_decay: float = 5e-4,
        focal_gamma: float = 2.0,
        warmup_epochs: int = 10,
        max_epochs: int = 150,
        cosine_t_max: int = 60,    # Reduced from 140: reaches min at epoch 70 (=warmup+T_max)
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        grad_clip_val: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.esm2: Optional[nn.Module] = None
        self.string_gnn = None
        self.cond_proj: Optional[CondEmbProjector] = None
        self.fusion: Optional[GatedFusion] = None
        self.head: Optional[PerturbHead] = None
        self.focal_loss: Optional[FocalLoss] = None

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols: List[str] = []

    def setup(self, stage: str = "fit") -> None:
        # ESM2-650M with LoRA for parameter-efficient fine-tuning
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

        # cond_emb projector with learnable gain: ESM2[B, 1280] → [B, 256] for GNN conditioning
        self.cond_proj = CondEmbProjector(in_dim=ESM2_HIDDEN, out_dim=GNN_HIDDEN)

        # Gated fusion and prediction head
        self.fusion = GatedFusion(
            dim_a=ESM2_HIDDEN,
            dim_b=GNN_HIDDEN,
            out_dim=FUSED_DIM,
            dropout=self.hparams.fusion_dropout,
        )
        self.head = PerturbHead(
            in_dim=FUSED_DIM,
            hidden_dim=self.hparams.head_hidden_dim,
            dropout=self.hparams.head_dropout,
        )

        # Class weights for focal loss (inverse frequency, no label smoothing)
        freq = torch.tensor([0.9282, 0.0477, 0.0241], dtype=torch.float32)
        class_weights = (1.0 / freq)
        class_weights = class_weights / class_weights.mean()
        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=class_weights,
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"ESM2-650M (LoRA r={self.hparams.lora_r}) + STRING_GNN + cond_emb(gain) | "
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
        """Run STRING_GNN with enhanced perturbation conditioning via cond_emb + gain.

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
        N = self._graph_edge_index.max().item() + 1

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

    def _forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        esm2_emb = self._get_esm2_emb(batch["input_ids"], batch["attention_mask"])  # [B, 1280]
        gnn_emb = self._get_gnn_emb_with_conditioning(batch["node_idx"], esm2_emb)  # [B, 256]
        fused = self.fusion(esm2_emb, gnn_emb)                                       # [B, 512]
        return self.head(fused)                                                        # [B, 3, 6640]

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.permute(0, 2, 1).reshape(-1, N_CLASSES).float()
        labels_flat = labels.reshape(-1)
        return self.focal_loss(logits_flat, labels_flat)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
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

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds, dim=0).numpy()
        labels = torch.cat(self._val_labels, dim=0).numpy()
        self._val_preds.clear()
        self._val_labels.clear()
        f1 = _compute_per_gene_f1(preds, labels)
        # FIX: sync_dist=True ensures val/f1 is properly synchronized across DDP ranks
        # node4-1 had sync_dist=False which caused checkpoint naming to show val_f1=0.0000
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
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

        # Compute test F1 for logging
        if self.trainer is not None and self.trainer.is_global_zero:
            # Deduplicate by pert_id (in case DistributedSampler caused duplicates)
            # Move tensors to CPU first before stacking
            all_preds_cpu = all_preds.cpu()
            all_labels_cpu = all_labels.cpu()
            seen = set()
            unique_preds, unique_labels = [], []
            for i, pid in enumerate(all_pert_ids):
                if pid not in seen:
                    seen.add(pid)
                    unique_preds.append(all_preds_cpu[i])
                    unique_labels.append(all_labels_cpu[i])
            unique_preds_np = torch.stack(unique_preds).numpy()
            unique_labels_np = torch.stack(unique_labels).numpy()
            test_f1 = _compute_per_gene_f1(unique_preds_np, unique_labels_np)
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=False)

            _save_test_predictions(
                pert_ids=all_pert_ids[:all_preds.shape[0]],
                symbols=all_symbols[:all_preds.shape[0]],
                preds=all_preds.float().cpu().numpy(),
                out_path=Path(__file__).parent / "run" / "test_predictions.tsv",
            )

            # Save test score to node root test_score.txt
            score_path = Path(__file__).parent / "test_score.txt"
            score_path.write_text(f"test_f1: {test_f1:.6f}\n")
            self.print(f"Test F1: {test_f1:.6f} | Saved to {score_path}")

    def configure_optimizers(self):
        # Unified LR=5e-5 for ALL trainable parameters.
        # node4-1 had LR=1e-4 for head/GNN but only 5e-5 for LoRA.
        # Feedback showed the 1e-4 components continued making aggressive updates
        # post-peak (epoch 70), driving val loss up while train loss dropped.
        # Unifying to 5e-5 reduces destructive high-LR updates in later epochs.
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing with linear warmup:
        # - warmup: epochs 0→warmup_epochs, LR linearly increases from ~0 to lr
        # - cosine: epochs warmup→(warmup+T_max), LR decays to eta_min
        # With warmup_epochs=10, cosine_t_max=60:
        #   - cosine reaches minimum at epoch 10+60=70
        # This matches node4-1's best checkpoint at epoch 70.
        # Previously T_max=140 only reached 57% of its decay at early stop (epoch 90).
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Node4-1-1: ESM2-650M (LoRA) + STRING_GNN (cond_emb+gain), fixed cosine schedule")
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Unified LR for all trainable params (LoRA, GNN, head, fusion, cond_proj)")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--head-hidden-dim", type=int, default=1024)
    p.add_argument("--head-dropout", type=float, default=0.3)
    p.add_argument("--fusion-dropout", type=float, default=0.2)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--cosine-t-max", type=int, default=60,
                   help="Cosine T_max (epochs after warmup). warmup+T_max=70 matches node4-1 best epoch.")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--grad-clip-val", type=float, default=1.0,
                   help="Gradient clipping value for training stability")
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
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        cosine_t_max=args.cosine_t_max,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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
        filename="best-{epoch:03d}-{val_f1:.4f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
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
        gradient_clip_val=args.grad_clip_val,
    )

    trainer.fit(model, datamodule=datamodule)

    if args.fast_dev_run or args.debug_max_step is not None:
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    if trainer.is_global_zero and test_results:
        print(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
