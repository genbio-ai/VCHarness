"""
Node 3-2-1 – AIDO.Cell-100M + LoRA (r=32) + Realistic Multi-Gene Input + Deep Residual Head

Architecture:
  - AIDO.Cell-100M backbone (hidden_size=640, 18 transformer layers)
    loaded from /home/Models/AIDO.Cell-100M
  - REALISTIC perturbation input: all 19,264 genes at expression=1.0, perturbed gene at 0.0
    (simulates knockdown: target gene absent, all others at baseline expression)
    KEY CHANGE from node3-2: moves input from out-of-distribution (single gene=1.0, all others=-1.0)
    to IN-DISTRIBUTION (full transcriptome context), allowing AIDO.Cell-100M to leverage
    its learned gene-gene co-expression patterns from pretraining.
  - LoRA r=32 (increased from r=16): more expressive backbone fine-tuning
  - Gene-position extraction: hidden state at the perturbed gene's vocabulary index
    (now enriched with full transcriptome context via attention)
  - Deep residual prediction head:
      input_proj(640→640) →
      4 × ResidualBlock(640→2560→640, LN+GELU+Dropout) →
      output_norm(640) →
      output_proj(640→6640×3)
  - Focal cross-entropy loss (gamma=2.0) with inverse-frequency class weights [10.91, 1.0, 29.62]
  - No label smoothing (proven to hurt with class weights in this tree)
  - Pure AdamW (betas=(0.9, 0.95)): backbone LoRA at lr=1e-4, head at lr=3e-4
  - CosineAnnealingLR (T_max=100) with 10-epoch linear warmup

Key improvements vs parent Node 3-2 (AIDO.Cell-100M + LoRA r=16 + single-gene input + 2-layer head):
  1. REALISTIC INPUT: all genes=1.0, perturbed gene=0.0 (primary bottleneck fix)
     - Parent: {perturbed_gene: 1.0, all others: -1.0} → highly OOD, 1 of 19264 genes expressed
     - This node: {all genes: 1.0, perturbed_gene: 0.0} → in-distribution, full transcriptome
     - AIDO.Cell can now leverage co-expression patterns to predict downstream effects
  2. DEEP RESIDUAL HEAD: 4 ResidualBlocks(640→2560→640) instead of 2-layer Linear head
     - Parent: Linear(640→2048)→GELU→LN→Dropout→Linear(2048→19920) [42.1M params]
     - This node: 4-block ResidualMLP with expand=4 [~26M params, deeper and fewer params]
     - Residual connections enable better gradient flow through deep supervision
  3. LORA RANK=32: doubles LoRA expressiveness (1.1M → 2.2M LoRA params)
     - LoRA r=16 tried in node3-2 (F1=0.4096); r=8 in node3-1-1 (F1=0.4096)
     - r=32 provides richer low-rank updates at negligible memory cost (~3.41 GiB for 100M LoRA)

Root cause from Node 3-2 feedback:
  "The single most impactful change is providing a realistic genetic context to AIDO.Cell-100M
  rather than a synthetic single-gene activation. This change would fundamentally alter the
  information content available to the model and potentially unlock significantly higher F1 scores."
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─── Constants ────────────────────────────────────────────────────────────────

AIDO_CELL_DIR = "/home/Models/AIDO.Cell-100M"
N_GENES_OUT   = 6640
N_CLASSES     = 3
HIDDEN_SIZE   = 640   # AIDO.Cell-100M hidden size
N_GENE_VOCAB  = 19264 # AIDO.Cell gene space

# Class weights: inverse-frequency based on train split label distribution
# down-regulated (-1): 8.14%, neutral (0): 88.86%, up-regulated (+1): 3.00%
# Shifted to {0,1,2}: class 0 = down (8.14%), class 1 = neutral (88.86%), class 2 = up (3.00%)
# Weights: neutral_freq / class_freq, normalized so neutral weight = 1.0
# neutral/down = 88.86/8.14 ≈ 10.91;  neutral/up = 88.86/3.00 ≈ 29.62
CLASS_WEIGHTS = torch.tensor([10.91, 1.0, 29.62], dtype=torch.float32)


# ─── Focal Loss ───────────────────────────────────────────────────────────────

def focal_cross_entropy(
    logits: torch.Tensor,         # [B, C, G]
    labels: torch.Tensor,         # [B, G] class indices in {0, 1, 2}
    class_weights: torch.Tensor,  # [C]
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal cross-entropy loss for 3D logits [B, C, G].

    Focal loss modulation: (1 - p_t)^gamma downweights well-classified examples.
    Combined with inverse-frequency class weights to address 88.9% neutral imbalance.
    Matches calc_metric.py's evaluation contract.
    """
    # Standard cross-entropy per position: [B, G]
    ce = F.cross_entropy(
        logits, labels,
        weight=class_weights.to(logits.device),
        reduction="none",
    )  # [B, G]

    # Compute p_t (probability of the true class) for focal modulation
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)       # [B, C, G]
        pt    = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B, G]

    # Focal modulation: (1 - p_t)^gamma
    focal_weight = (1.0 - pt) ** gamma  # [B, G]

    return (focal_weight * ce).mean()


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """
    Compute macro-averaged per-gene F1 score matching calc_metric.py logic.

    Args:
        pred_np:   [N, 3, G] softmax probabilities (float)
        labels_np: [N, G] class indices in {0, 1, 2} (already shifted from {-1, 0, 1})
    Returns:
        float: mean per-gene macro-F1 over all G genes
    """
    pred_cls = pred_np.argmax(axis=1)  # [N, G]
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1   = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AIDOCellPerturbDataset(Dataset):
    def __init__(self, pert_ids, symbols, input_ids, gene_positions, labels=None):
        self.pert_ids       = pert_ids
        self.symbols        = symbols
        self.input_ids      = input_ids        # [N, 19264] float32
        self.gene_positions = gene_positions   # [N] long
        self.labels         = labels           # [N, 6640] long or None

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":       self.pert_ids[idx],
            "symbol":        self.symbols[idx],
            "input_ids":     self.input_ids[idx],
            "gene_position": self.gene_positions[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    out = {
        "pert_id":       [b["pert_id"]   for b in batch],
        "symbol":        [b["symbol"]    for b in batch],
        "input_ids":     torch.stack([b["input_ids"]     for b in batch]),
        "gene_position": torch.stack([b["gene_position"] for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class AIDOCell100MDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data", micro_batch_size=8, num_workers=2):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        # Rank 0 downloads first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        def tokenize_symbols_realistic(symbols):
            """
            Realistic knockdown perturbation input representation.

            KEY CHANGE from parent node3-2:
            Instead of {perturbed_gene: 1.0, all others: -1.0} (highly OOD),
            we provide {all_genes: 1.0, perturbed_gene: 0.0}.

            This simulates a knockdown experiment where:
            - The target gene has zero expression (knocked down / absent)
            - All other 19,263 genes maintain a uniform baseline expression of 1.0

            The model's internal _prepare_inputs() then applies CP10K normalization:
            - total counts = 19263 (all non-perturbed genes at 1.0)
            - expressed genes: log1p(1.0 / 19263 * 10000) ≈ log1p(0.519) ≈ 0.416
            - perturbed gene: log1p(0.0) = 0.0

            This creates an in-distribution input (AIDO.Cell-100M was pretrained on full
            transcriptomes), allowing the model to leverage learned gene-gene
            co-expression patterns to predict downstream perturbation effects.
            """
            # Step 1: Tokenize with only the perturbed gene to discover its vocabulary position.
            # The tokenizer maps gene symbols to fixed positions in the 19264-dim vector.
            # The perturbed gene's slot gets value=1.0; all others get -1.0 (missing).
            batch_input = [{"gene_names": [s], "expression": [1.0]} for s in symbols]
            tok_out     = tokenizer(batch_input, return_tensors="pt")
            ids         = tok_out["input_ids"]   # [N, 19264] float32; -1.0 = missing

            # Find the perturbed gene's position (the only slot with value > 0.5)
            # If gene symbol is not in AIDO.Cell vocabulary, all values remain -1.0
            # and argmax returns 0 (graceful fallback).
            gene_found = (ids > 0.5).any(dim=1)                      # [N] bool
            gpos       = (ids > 0.5).float().argmax(dim=1).long()    # [N] position index

            # Step 2: Build realistic input tensor.
            # Start from the original ids (perturbed=1.0, others=-1.0)
            # then flip: replace all -1.0 (missing) with baseline=1.0,
            # and set the perturbed gene to 0.0 (knockdown).
            realistic_ids = ids.clone()

            # Replace all missing genes (-1.0) with baseline expression (1.0)
            realistic_ids[realistic_ids < 0] = 1.0

            # Set the perturbed gene to 0.0 (knocked down / absent)
            # Use vectorized scatter for efficiency
            batch_size = realistic_ids.shape[0]
            for i in range(batch_size):
                if gene_found[i]:
                    realistic_ids[i, gpos[i]] = 0.0
                # If gene not in vocabulary: all positions stay 1.0 (baseline only, no knockdown)
                # The gene_position (gpos=0) used downstream still indexes consistently.

            return realistic_ids, gpos

        def load_split(fname, has_lbl):
            df        = pd.read_csv(self.data_dir / fname, sep="\t")
            ids, gpos = tokenize_symbols_realistic(df["symbol"].tolist())
            labels    = None
            if has_lbl and "label" in df.columns:
                # Shift labels from {-1, 0, 1} to class indices {0, 1, 2}
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return AIDOCellPerturbDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(), ids, gpos, labels
            )

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=self.num_workers,
            pin_memory=True, drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Deep Residual Head ───────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Single pre-norm residual MLP block.

    Architecture:
        LN(hidden_dim)
        → Linear(hidden_dim → hidden_dim * expand_factor) → GELU → Dropout
        → Linear(hidden_dim * expand_factor → hidden_dim) → Dropout
        + residual connection

    Using pre-norm (LayerNorm before the linear layers) for stable gradient flow,
    consistent with transformer-style residual blocks.
    """
    def __init__(self, hidden_dim: int, expand_factor: int = 4, dropout: float = 0.2):
        super().__init__()
        expanded = hidden_dim * expand_factor
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, expanded),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DeepResidualHead(nn.Module):
    """
    Deep residual MLP prediction head following node1-2's proven architecture.

    Replaces the shallow 2-layer head from node3-2 with a deeper design
    using residual connections for better gradient flow through the deep supervision
    signal (6640 output positions per sample).

    Architecture:
        input_proj: Linear(input_dim → hidden_dim)          [optional if input_dim == hidden_dim]
        n_blocks × ResidualBlock(hidden_dim, expand_factor, dropout)
        output_norm: LayerNorm(hidden_dim)
        output_proj: Linear(hidden_dim → n_genes_out × n_classes)

    Parameter count (hidden_dim=640, expand=4, n_blocks=4, n_genes_out=6640, n_classes=3):
        input_proj:  640×640 + 640 = 410,240
        4 blocks:    4 × (LN(640) + 640×2560 + 2560×640) ≈ 13.1M
        output_norm: 640×2 = 1,280
        output_proj: 640×19920 + 19920 ≈ 12.75M
        Total: ~26.3M  (vs parent's 42.1M for the 2-layer head)

    The output_proj has 12.75M params (vs 40.8M in parent's Linear(2048→19920))
    — significantly fewer, with the deep residual blocks compensating expressiveness.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_genes_out: int,
        n_classes: int,
        n_blocks: int = 4,
        expand_factor: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.blocks      = nn.ModuleList([
            ResidualBlock(hidden_dim, expand_factor, dropout)
            for _ in range(n_blocks)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, n_genes_out * n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        return self.output_proj(x)


# ─── Model ────────────────────────────────────────────────────────────────────

class AIDOCell100MLoRAModel(nn.Module):
    """
    AIDO.Cell-100M with LoRA (r=32) + realistic multi-gene input + deep residual head.

    Three key changes from parent (node3-2):
    1. Input representation: realistic knockdown (all=1.0, perturbed=0.0) vs single-gene
    2. Prediction head: 4-block DeepResidualHead vs 2-layer Linear head
    3. LoRA rank: 32 vs 16 (more expressive backbone fine-tuning)
    """

    def __init__(
        self,
        lora_rank: int       = 32,
        lora_alpha: int      = 64,
        lora_dropout: float  = 0.05,
        n_genes_out: int     = N_GENES_OUT,
        n_classes: int       = N_CLASSES,
        head_hidden_dim: int = 640,
        head_n_blocks: int   = 4,
        head_expand: int     = 4,
        head_dropout: float  = 0.2,
    ):
        super().__init__()

        # ── Backbone: AIDO.Cell-100M in bfloat16 ──────────────────────────────
        backbone = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False
        # NOTE: Do NOT call gradient_checkpointing_enable() before get_peft_model().
        # PEFT's __init__ calls _prepare_model_for_gradient_checkpointing() when GC is
        # already enabled, which invokes get_input_embeddings() — not implemented in
        # AidoCellModel (raises NotImplementedError). Enable GC on the underlying
        # model AFTER PEFT wrapping.

        # ── LoRA configuration (rank=32, doubled from node3-2's rank=16) ───────
        # flash_self shares Q/K/V weight tensors with self, so LoRA on
        # attention.self.{query,key,value} automatically applies to flash_self too.
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
            layers_to_transform=None,   # all 18 transformer layers
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Enable gradient checkpointing on the underlying AidoCellModel
        # (bypasses PEFT's _prepare_model_for_gradient_checkpointing which fails due
        # to unimplemented get_input_embeddings() in AidoCellModel)
        self.backbone.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA parameters to float32 for stable optimization
        # (frozen backbone weights remain in bfloat16)
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ── Deep residual prediction head ─────────────────────────────────────
        # 4-block ResidualMLP: 640 → (640→2560→640) × 4 → 640 → 6640×3
        self.head = DeepResidualHead(
            input_dim    = HIDDEN_SIZE,
            hidden_dim   = head_hidden_dim,
            n_genes_out  = n_genes_out,
            n_classes    = n_classes,
            n_blocks     = head_n_blocks,
            expand_factor= head_expand,
            dropout      = head_dropout,
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(
            f"[Node3-2-1] Trainable params: {n_trainable:,} / {n_total:,} "
            f"({100 * n_trainable / n_total:.2f}%)"
        )

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, 19264] float32 — realistic input (0.0 or 1.0)
        gene_positions: torch.Tensor,  # [B] long — perturbed gene's vocab position
    ) -> torch.Tensor:
        # AIDO.Cell overrides attention_mask to all-ones internally, but we pass it
        # explicitly for compatibility (the model ignores it)
        attn_mask = torch.ones(
            input_ids.shape[0], input_ids.shape[1],
            dtype=torch.long, device=input_ids.device,
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(input_ids=input_ids, attention_mask=attn_mask)

        # last_hidden_state: [B, 19266, 640]
        # Slice to gene space [B, 19264, 640] (exclude the 2 appended summary tokens)
        gene_states = out.last_hidden_state[:, :N_GENE_VOCAB, :]  # [B, 19264, 640]

        # Gene-position extraction: pick the hidden state at the perturbed gene's position.
        # With the realistic input, this token now has rich contextual embeddings from
        # attending to all 19,263 other expressed genes — much more informative than
        # the parent's single-gene input where context was nearly empty.
        # gene_positions: [B] long, each in [0, 19263]
        pos_idx   = gene_positions.unsqueeze(1).unsqueeze(2).expand(-1, 1, HIDDEN_SIZE)
        gene_repr = gene_states.gather(1, pos_idx).squeeze(1)  # [B, 640]

        # Cast to float32 for the head (mixed precision boundary)
        gene_repr = gene_repr.float()

        # Deep residual head → [B, 6640×3]
        logits = self.head(gene_repr)
        return logits.view(-1, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    """All-gather tensors across DDP ranks, handling variable sizes per rank."""
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))
    pad    = max_sz - local_p.shape[0]
    p = local_p.to(device)
    l = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)
    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p)
    dist.all_gather(gl, l)
    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class AIDOCell100MLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_backbone: float   = 1e-4,
        lr_head: float       = 3e-4,
        weight_decay: float  = 0.01,
        focal_gamma: float   = 2.0,
        lora_rank: int       = 32,
        lora_alpha: int      = 64,
        lora_dropout: float  = 0.05,
        head_hidden_dim: int = 640,
        head_n_blocks: int   = 4,
        head_expand: int     = 4,
        head_dropout: float  = 0.2,
        t_max: int           = 100,
        warmup_steps: int    = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str]          = []
        self._test_symbols:  List[str]          = []
        self._test_labels:   List[torch.Tensor] = []

    def setup(self, stage=None):
        self.model = AIDOCell100MLoRAModel(
            lora_rank      = self.hparams.lora_rank,
            lora_alpha     = self.hparams.lora_alpha,
            lora_dropout   = self.hparams.lora_dropout,
            head_hidden_dim= self.hparams.head_hidden_dim,
            head_n_blocks  = self.hparams.head_n_blocks,
            head_expand    = self.hparams.head_expand,
            head_dropout   = self.hparams.head_dropout,
        )
        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(self, input_ids, gene_positions):
        return self.model(input_ids, gene_positions)

    def _loss(self, logits, labels):
        return focal_cross_entropy(
            logits, labels,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"])
        if "label" in batch:
            loss = self._loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True,
                     prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        lp = torch.cat(self._val_preds,  0)
        ll = torch.cat(self._val_labels, 0)
        if self.trainer.world_size > 1:
            lp, ll = _gather_tensors(lp, ll, self.device, self.trainer.world_size)
        f1 = compute_per_gene_f1(lp.numpy(), ll.numpy())
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"])
        probs  = torch.softmax(logits, dim=1)
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (
            torch.cat(self._test_labels, 0) if self._test_labels
            else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        )
        if self.trainer.world_size > 1:
            all_probs, all_labels = _gather_tensors(
                local_probs, dummy_labels, self.device, self.trainer.world_size
            )
            all_pert = [None] * self.trainer.world_size
            all_syms = [None] * self.trainer.world_size
            dist.all_gather_object(all_pert, self._test_pert_ids)
            dist.all_gather_object(all_syms, self._test_symbols)
            all_pert = [p for sub in all_pert for p in sub]
            all_syms = [s for sub in all_syms for s in sub]
        else:
            all_probs, all_labels = local_probs, dummy_labels
            all_pert, all_syms   = self._test_pert_ids, self._test_symbols

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id: DDP DistributedSampler may pad the dataset
            # to equalize counts across ranks, introducing duplicate pert_ids.
            seen_pids: set = set()
            dedup_perts, dedup_syms, dedup_probs_list, dedup_label_rows = [], [], [], []
            for pid, sym, prob_row, lbl_row in zip(
                all_pert, all_syms, all_probs.numpy(), all_labels.numpy()
            ):
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    dedup_perts.append(pid)
                    dedup_syms.append(sym)
                    dedup_probs_list.append(prob_row)
                    dedup_label_rows.append(lbl_row)

            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for pid, sym, probs in zip(dedup_perts, dedup_syms, dedup_probs_list):
                    fh.write(f"{pid}\t{sym}\t{json.dumps(probs.tolist())}\n")

            self.print(
                f"[Node3-2-1] Saved {len(dedup_perts)} test predictions → {pred_path}"
            )
            if all_labels.any():
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[Node3-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    # ── Two-group AdamW + Sequential LR (warmup + cosine annealing) ───────────

    def configure_optimizers(self):
        hp = self.hparams

        # Separate backbone (LoRA) parameters from prediction head parameters
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        head_params     = list(self.model.head.parameters())

        param_groups = [
            dict(params=backbone_params, lr=hp.lr_backbone, weight_decay=hp.weight_decay),
            dict(params=head_params,     lr=hp.lr_head,     weight_decay=hp.weight_decay),
        ]
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

        # SequentialLR: 10-epoch linear warmup → CosineAnnealingLR(T_max=90)
        # Total effective schedule: 100 epochs, calibrated to expected early-stopping point.
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=hp.warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, hp.t_max - hp.warmup_steps),
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[hp.warmup_steps],
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable params (LoRA + head) + buffers ────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save only trainable parameters (LoRA + head) and persistent buffers."""
        full_sd        = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items() if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained:,}/{total:,} params ({100 * trained / total:.2f}%)")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        """Load trainable parameters and buffers from partial checkpoint."""
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 3-2-1 – AIDO.Cell-100M + LoRA r=32 + Realistic Input + Deep Residual Head"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--lr-backbone",        type=float, default=1e-4)
    p.add_argument("--lr-head",            type=float, default=3e-4)
    p.add_argument("--weight-decay",       type=float, default=0.01)
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--lora-rank",          type=int,   default=32)
    p.add_argument("--lora-alpha",         type=int,   default=64)
    p.add_argument("--lora-dropout",       type=float, default=0.05)
    p.add_argument("--head-hidden-dim",    type=int,   default=640)
    p.add_argument("--head-n-blocks",      type=int,   default=4)
    p.add_argument("--head-expand",        type=int,   default=4)
    p.add_argument("--head-dropout",       type=float, default=0.2)
    p.add_argument("--t-max",              type=int,   default=100)
    p.add_argument("--warmup-steps",       type=int,   default=10)
    p.add_argument("--micro-batch-size",   type=int,   default=8)
    p.add_argument("--global-batch-size",  type=int,   default=32)
    p.add_argument("--max-epochs",         type=int,   default=200)
    p.add_argument("--patience",           type=int,   default=40)
    p.add_argument("--num-workers",        type=int,   default=2)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",     type=int,   default=None)
    p.add_argument("--fast-dev-run",       action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm  = AIDOCell100MDataModule(
        data_dir       = args.data_dir,
        micro_batch_size = args.micro_batch_size,
        num_workers    = args.num_workers,
    )
    lit = AIDOCell100MLitModule(
        lr_backbone    = args.lr_backbone,
        lr_head        = args.lr_head,
        weight_decay   = args.weight_decay,
        focal_gamma    = args.focal_gamma,
        lora_rank      = args.lora_rank,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        head_hidden_dim= args.head_hidden_dim,
        head_n_blocks  = args.head_n_blocks,
        head_expand    = args.head_expand,
        head_dropout   = args.head_dropout,
        t_max          = args.t_max,
        warmup_steps   = args.warmup_steps,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath   = str(out_dir / "checkpoints"),
        filename  = "best-{epoch:04d}-{val_f1:.4f}",
        monitor   = "val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb   = EarlyStopping(monitor="val_f1", mode="max",
                            patience=args.patience, min_delta=1e-5)
    lr_cb   = LearningRateMonitor(logging_interval="epoch")
    pb_cb   = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps:   int        = -1
    limit_train: float|int  = 1.0
    limit_val:   float|int  = 1.0
    limit_test:  float|int  = 1.0
    fast_dev_run            = False

    if args.debug_max_step is not None:
        max_steps   = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2
        limit_test  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    accum    = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator          = "gpu",
        devices              = n_gpus,
        num_nodes            = 1,
        strategy             = strategy,
        precision            = "bf16-mixed",
        max_epochs           = args.max_epochs,
        max_steps            = max_steps,
        accumulate_grad_batches = accum,
        limit_train_batches  = limit_train,
        limit_val_batches    = limit_val,
        limit_test_batches   = limit_test,
        val_check_interval   = (
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps = 2,
        callbacks            = [ckpt_cb, es_cb, lr_cb, pb_cb],
        logger               = [csv_logger, tb_logger],
        log_every_n_steps    = 10,
        deterministic        = "warn_only",   # nll_loss may lack deterministic CUDA impl
        default_root_dir     = str(out_dir),
        fast_dev_run         = fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 3-2-1 – AIDO.Cell-100M + LoRA r=32 + Realistic Multi-Gene Input + "
            "Deep Residual Head\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
