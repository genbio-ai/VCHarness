"""
Node 3-1-3 – AIDO.Cell-100M + LoRA(r=16) + Full Multi-Gene Baseline + Dual-Stream + STRING_GNN Fusion

Architecture:
  - AIDO.Cell-100M backbone (hidden_size=640, 18 transformer layers)
    loaded from /home/Models/AIDO.Cell-100M
  - Full multi-gene baseline input: ALL 19,264 genes at expression=1.0,
    perturbed gene set to 0.0 (true knockdown representation)
    This is in-distribution for AIDO.Cell's pre-training corpus and provides
    rich co-expression context for the attention mechanism.
  - LoRA fine-tuning (rank=16, alpha=32) on QKV matrices across all 18 layers
    (~5.4M LoRA trainable parameters — appropriate for 1,416 training samples)
  - Dual-stream AIDO.Cell representation:
      Stream A: gene-position extraction at perturbed gene's vocab index [B, 640]
      Stream B: mean-pool over all 19,264 gene hidden states [B, 640]
  - STRING_GNN PPI embedding stream (frozen, precomputed):
      Stream C: STRING_GNN PPI node embedding for perturbed gene [B, 256]
      OOV genes (~6.4% of training) get a learned OOV embedding [256]
  - Fusion: Concatenate [A; B; C] → [B, 1536]
      LayerNorm(1536) → Linear(1536, 1024) → GELU → Dropout(0.2) → Linear(1024 → 6640×3)
  - Focal cross-entropy loss (gamma=2.0, class_weights=[10.91, 1.0, 29.62], label_smoothing=0.05)
  - AdamW optimizer: lr_backbone=1e-4 (LoRA), lr_head=3e-4 (head + STRING_GNN)
  - Cosine annealing with 5-epoch warmup (T_max=100)

Key design rationale:
  - Full multi-gene baseline: node3-1-2 found +0.031 F1 gain from knockdown input
    (node3-2-1 evidence), but only partially implemented it. This node provides
    the complete baseline with all 19,264 genes expressed.
  - Dual-stream (position + mean-pool): Captures both gene-specific context
    (Stream A: the perturbed gene's position embedding) and global perturbation
    state (Stream B: how the transcriptome-wide representation shifts). node3-1-2
    feedback identified mean-pool as "primary high-priority unexplored direction."
  - STRING_GNN PPI stream: node1-1-3-1-1 demonstrated +0.0479 F1 gain from adding
    STRING_GNN to AIDO.Cell (from 0.4379 to 0.4858). PPI topology encodes
    protein-protein interaction structure that directly predicts DEG cascades.
  - This combination addresses the core AIDO.Cell lineage bottleneck: single-gene
    extraction on sparse OOD input limits F1 to ~0.41. Full baseline + mean-pool +
    STRING_GNN fusion should break through to 0.46-0.50.

Differentiation from siblings:
  - node3-1-1: rank=8, no STRING_GNN, sparse OOD input (1.0 probe), no mean-pool
  - node3-1-2: rank=16, partial knockdown (0.0 probe, still sparse), no STRING_GNN, no mean-pool
  - node3-1-3 (this): rank=16, FULL baseline (all 19264 genes at 1.0), STRING_GNN fusion,
    dual-stream (position + mean-pool)
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
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─── Constants ────────────────────────────────────────────────────────────────

AIDO_CELL_DIR  = "/home/Models/AIDO.Cell-100M"
STRING_GNN_DIR = "/home/Models/STRING_GNN"

N_GENES_OUT   = 6640
N_CLASSES     = 3
HIDDEN_SIZE   = 640    # AIDO.Cell-100M hidden size
N_GENE_VOCAB  = 19264  # AIDO.Cell gene space
STRING_DIM    = 256    # STRING_GNN node embedding dimension

# Class weights: inverse-frequency based on train split label distribution
# down (-1): 8.14%, neutral (0): 88.86%, up (+1): 3.00%
# Shifted to {0,1,2}: class 0=down(8.14%), class 1=neutral(88.86%), class 2=up(3.00%)
CLASS_WEIGHTS = torch.tensor([10.91, 1.0, 29.62], dtype=torch.float32)

# LoRA configuration
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """
    Compute macro-averaged per-gene F1 score matching calc_metric.py logic.

    Args:
        pred_np: [N, 3, G] softmax probabilities
        labels_np: [N, G] class indices in {0, 1, 2} (already shifted)
    Returns:
        float: mean per-gene macro-F1 over all G genes
    """
    pred_cls = pred_np.argmax(axis=1)  # [N, G]
    f1_vals = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal cross-entropy loss with optional class weights and label smoothing.
    Focuses training on hard examples, especially important for the 88.9% neutral class.
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, G] — raw class scores for each gene position
            targets: [B, G]    — class indices in {0, 1, 2}
        Returns:
            scalar loss
        """
        B, C, G = logits.shape
        # Reshape to [B*G, C] and [B*G]
        logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)   # [B*G, C]
        targets_flat = targets.reshape(-1)                        # [B*G]

        # Standard cross-entropy (with label smoothing) as base
        ce = F.cross_entropy(
            logits_flat, targets_flat,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )   # [B*G]

        # Compute p_t (probability of the true class)
        with torch.no_grad():
            probs   = torch.softmax(logits_flat, dim=1)  # [B*G, C]
            # Clip targets to valid range to avoid out-of-bounds indexing
            clipped = targets_flat.clamp(0, C - 1)
            p_t = probs.gather(1, clipped.unsqueeze(1)).squeeze(1)  # [B*G]
            focal_weight = (1.0 - p_t) ** self.gamma

        loss = (focal_weight * ce).mean()

        # Normalize by class weight sum for scale invariance
        if self.weight is not None:
            loss = loss / self.weight.sum()

        return loss


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    """
    Dataset using full multi-gene baseline input with true knockdown encoding.
    All 19,264 genes are provided at expression=1.0; the perturbed gene is set to 0.0.
    """

    def __init__(self, pert_ids, symbols, input_ids, gene_positions,
                 string_node_indices, labels=None):
        """
        Args:
            pert_ids:           [N] list of Ensembl gene IDs (perturbed gene)
            symbols:            [N] list of gene symbols (perturbed gene)
            input_ids:          [N, 19264] float32 — full baseline expression input
            gene_positions:     [N] long — AIDO.Cell vocab index of perturbed gene
            string_node_indices:[N] long — STRING_GNN node index (-1 for OOV)
            labels:             [N, 6640] long or None
        """
        self.pert_ids            = pert_ids
        self.symbols             = symbols
        self.input_ids           = input_ids
        self.gene_positions      = gene_positions
        self.string_node_indices = string_node_indices
        self.labels              = labels

    def __len__(self): return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":          self.pert_ids[idx],
            "symbol":           self.symbols[idx],
            "input_ids":        self.input_ids[idx],
            "gene_position":    self.gene_positions[idx],
            "string_node_idx":  self.string_node_indices[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def perturb_collate_fn(batch):
    out = {
        "pert_id":         [b["pert_id"]        for b in batch],
        "symbol":          [b["symbol"]         for b in batch],
        "input_ids":       torch.stack([b["input_ids"]       for b in batch]),
        "gene_position":   torch.stack([b["gene_position"]   for b in batch]),
        "string_node_idx": torch.stack([b["string_node_idx"] for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class TripleStreamDataModule(pl.LightningDataModule):
    """
    DataModule that prepares:
    1. AIDO.Cell full multi-gene baseline inputs (all 19264 genes at 1.0, perturbed at 0.0)
    2. STRING_GNN node indices for each perturbed gene
    3. Labels (shifted from {-1,0,1} to {0,1,2})
    """

    def __init__(self, data_dir="data", micro_batch_size=4, num_workers=2):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        # ── Tokenizer: rank-0 downloads first ─────────────────────────────────
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        # ── Build full baseline expression array ───────────────────────────────
        all_gene_ids = tokenizer.gene_ids   # list of 19264 Ensembl IDs (vocab order)
        gene_id_to_vocab_idx = tokenizer.gene_id_to_index  # ENSEMBL_ID → vocab position

        # Store as contiguous arrays for fast sample-level construction
        all_gene_ids_list = list(all_gene_ids)   # [19264] for tokenizer input
        full_baseline_expr = [1.0] * N_GENE_VOCAB  # base: all genes at 1.0

        # ── STRING_GNN node index map ──────────────────────────────────────────
        node_names = json.loads((Path(STRING_GNN_DIR) / "node_names.json").read_text())
        node_name_to_idx = {n: i for i, n in enumerate(node_names)}
        OOV_IDX = -1  # sentinel for genes absent from STRING_GNN

        def tokenize_split(symbols, pert_ids):
            """
            Build full multi-gene baseline inputs for a list of (symbol, pert_id) pairs.
            The perturbed gene is identified by its pert_id (Ensembl ID) in the tokenizer.
            """
            input_ids_list   = []
            gene_pos_list    = []
            string_idx_list  = []

            for sym, pid in zip(symbols, pert_ids):
                # Get vocab index of this perturbed gene
                vocab_idx = gene_id_to_vocab_idx.get(pid)

                if vocab_idx is not None:
                    # Build full baseline expression: all 1.0, perturbed at 0.0
                    expr = full_baseline_expr.copy()
                    expr[vocab_idx] = 0.0  # knockdown
                    sample_input = {"gene_ids": all_gene_ids_list, "expression": expr}
                else:
                    # OOV for AIDO.Cell: fall back to sparse single-gene probe (1.0)
                    # This affects ~98 samples (~6.9% of 1416 training samples)
                    sample_input = {"gene_ids": [sym], "expression": [1.0]}

                tok_out   = tokenizer(sample_input, return_tensors="pt")
                ids_1d    = tok_out["input_ids"]  # [19264] float32 (single sample)
                if ids_1d.dim() == 2:
                    ids_1d = ids_1d.squeeze(0)

                input_ids_list.append(ids_1d)

                # Gene position: use vocab_idx directly if available; else detect via threshold
                if vocab_idx is not None:
                    gpos = torch.tensor(vocab_idx, dtype=torch.long)
                else:
                    # Detect position: the single gene at 1.0 in sparse OOD input is the max
                    gpos = ids_1d.argmax().long()

                gene_pos_list.append(gpos)

                # STRING_GNN node index
                str_idx = node_name_to_idx.get(pid, OOV_IDX)
                string_idx_list.append(torch.tensor(str_idx, dtype=torch.long))

            return (
                torch.stack(input_ids_list),              # [N, 19264] float32
                torch.stack(gene_pos_list),               # [N] long
                torch.stack(string_idx_list),             # [N] long
            )

        def load_split(fname, has_lbl):
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            symbols  = df["symbol"].tolist()
            pert_ids = df["pert_id"].tolist()

            input_ids, gene_pos, str_idx = tokenize_split(symbols, pert_ids)

            labels = None
            if has_lbl and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)

            return PerturbDataset(pert_ids, symbols, input_ids, gene_pos, str_idx, labels)

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  False)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size, shuffle=shuffle,
            collate_fn=perturb_collate_fn, num_workers=self.num_workers,
            pin_memory=True, drop_last=shuffle,
            persistent_workers=self.num_workers > 0
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Model ────────────────────────────────────────────────────────────────────

class TripleStreamModel(nn.Module):
    """
    Triple-stream perturbation DEG predictor:

    Stream A: AIDO.Cell-100M gene-specific position embedding
        → hidden_state[:, perturbed_gene_vocab_idx, :] → [B, 640]
    Stream B: AIDO.Cell-100M mean-pool over all 19264 gene positions
        → hidden_state[:, :19264, :].mean(dim=1) → [B, 640]
    Stream C: STRING_GNN PPI frozen embedding for perturbed gene
        → precomputed_embeddings[string_node_idx, :] → [B, 256]

    Fusion:
        concat([A; B; C]) → [B, 1536]
        LayerNorm(1536) → Linear(1536, 1024) → GELU → Dropout(0.2)
        → Linear(1024, 6640×3) → reshape [B, 3, 6640]
    """

    def __init__(self, n_genes_out=N_GENES_OUT, n_classes=N_CLASSES):
        super().__init__()

        # ── AIDO.Cell-100M with LoRA ───────────────────────────────────────────
        backbone = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["query", "key", "value"],
        )
        # Apply LoRA BEFORE enabling gradient checkpointing (critical ordering)
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA parameters to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        n_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Node3-1-3] AIDO Trainable: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.2f}%)")

        # ── STRING_GNN (frozen, precomputed embeddings) ────────────────────────
        # Load STRING_GNN and precompute all node embeddings
        string_gnn = AutoModel.from_pretrained(STRING_GNN_DIR, trust_remote_code=True)
        for param in string_gnn.parameters():
            param.requires_grad = False

        graph_data   = torch.load(Path(STRING_GNN_DIR) / "graph_data.pt", weights_only=False)
        edge_index   = graph_data["edge_index"]
        edge_weight  = graph_data.get("edge_weight")

        with torch.no_grad():
            outputs = string_gnn(
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            str_embs = outputs.last_hidden_state.float()  # [18870, 256]

        # Store as a frozen parameter buffer for DDP compatibility
        self.register_buffer("str_embs", str_embs)  # [18870, 256]

        # Learnable OOV embedding for genes absent from STRING_GNN (~6.4%)
        self.oov_emb = nn.Parameter(torch.zeros(STRING_DIM))

        # ── Fusion + Prediction Head ───────────────────────────────────────────
        # Input: [A; B; C] = [640 + 640 + 256] = 1536-dim
        fusion_dim = HIDDEN_SIZE + HIDDEN_SIZE + STRING_DIM  # 1536

        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, n_genes_out * n_classes),
        )

        # Initialize head output layer near zero for training stability
        nn.init.zeros_(self.head[-1].bias if hasattr(self.head[-1], 'bias') and
                        self.head[-1].bias is not None else
                        nn.Parameter(torch.zeros(1)))
        nn.init.normal_(self.head[-1].weight, std=0.01)

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, 19264] float32
        gene_positions: torch.Tensor,   # [B] long — AIDO vocab idx of perturbed gene
        string_node_idx: torch.Tensor,  # [B] long — STRING_GNN node idx (-1 for OOV)
    ) -> torch.Tensor:
        # Attention mask (overridden inside AIDO.Cell, but required for API)
        attn_mask = torch.ones(
            input_ids.shape[0], input_ids.shape[1],
            dtype=torch.long, device=input_ids.device
        )

        # AIDO.Cell-100M forward pass in bfloat16
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(input_ids=input_ids, attention_mask=attn_mask)

        # last_hidden_state: [B, 19266, 640] — includes 2 summary tokens at end
        # Slice to gene space only: [B, 19264, 640]
        gene_states = out.last_hidden_state[:, :N_GENE_VOCAB, :].float()  # [B, 19264, 640]

        # ── Stream A: Gene-position extraction ────────────────────────────────
        # Extract hidden state at the specific perturbed gene's position
        batch_size = gene_states.shape[0]
        pos_idx    = gene_positions.view(batch_size, 1, 1).expand(batch_size, 1, HIDDEN_SIZE)
        stream_a   = gene_states.gather(1, pos_idx).squeeze(1)   # [B, 640]

        # ── Stream B: Mean-pool over all 19,264 gene positions ─────────────────
        stream_b = gene_states.mean(dim=1)  # [B, 640]

        # ── Stream C: STRING_GNN PPI embedding ────────────────────────────────
        # For OOV genes (string_node_idx == -1), use learned OOV embedding
        valid_mask = string_node_idx >= 0  # [B] bool
        str_lookup = string_node_idx.clamp(min=0)  # [B] — clamp OOV to 0 temporarily
        str_emb_b  = self.str_embs[str_lookup]     # [B, 256]

        if not valid_mask.all():
            oov_expanded = self.oov_emb.unsqueeze(0).expand(batch_size, -1)  # [B, 256]
            str_emb_b = torch.where(
                valid_mask.unsqueeze(1), str_emb_b, oov_expanded
            )
        stream_c = str_emb_b  # [B, 256]

        # ── Fusion ────────────────────────────────────────────────────────────
        fused  = torch.cat([stream_a, stream_b, stream_c], dim=1)  # [B, 1536]
        logits = self.head(fused)                                    # [B, 6640*3]
        return logits.view(-1, N_CLASSES, N_GENES_OUT)               # [B, 3, 6640]


# ─── DDP Helpers ──────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    """Gather tensors from all DDP ranks with variable-size padding."""
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))
    pad    = max_sz - local_p.shape[0]
    p = local_p.to(device); l = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)
    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p); dist.all_gather(gl, l)
    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class TripleStreamLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_backbone:      float = 1e-4,   # LoRA params LR (AdamW)
        lr_head:          float = 3e-4,   # Head + OOV embedding LR (AdamW)
        weight_decay:     float = 0.01,
        label_smoothing:  float = 0.05,
        focal_gamma:      float = 2.0,
        max_epochs:       int   = 200,
        warmup_epochs:    int   = 5,
        cosine_t_max:     int   = 100,    # Aligned with expected early stopping window
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []
        self._test_labels:   List[torch.Tensor] = []

    def setup(self, stage=None):
        self.model = TripleStreamModel()
        self.register_buffer("class_weights", CLASS_WEIGHTS)
        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=CLASS_WEIGHTS,
            label_smoothing=self.hparams.label_smoothing,
        )

    def forward(self, input_ids, gene_positions, string_node_idx):
        return self.model(input_ids, gene_positions, string_node_idx)

    def _loss(self, logits, labels):
        return self.focal_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"], batch["string_node_idx"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"], batch["string_node_idx"])
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
        probs_np  = torch.softmax(lp, dim=1).numpy()  # [N, 3, G]
        labels_np = ll.numpy()                         # [N, G]
        f1 = compute_per_gene_f1(probs_np, labels_np)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"], batch["string_node_idx"])
        probs  = torch.softmax(logits, dim=1)  # [B, 3, G]
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs  = torch.cat(self._test_preds, 0)
        dummy_labels = (torch.cat(self._test_labels, 0) if self._test_labels
                        else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long))

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
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate: DDP DistributedSampler may pad the dataset
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

            self.print(f"[Node3-1-3] Saved {len(dedup_perts)} predictions → {pred_path}")

            # Self-evaluate if labels are available
            if any(any(row) for row in dedup_label_rows):
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"[Node3-1-3] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    # ── Optimizer: AdamW with two parameter groups ────────────────────────────

    def configure_optimizers(self):
        hp = self.hparams

        # Group 1: LoRA parameters (backbone) at lower LR
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters() if p.requires_grad
        ]
        backbone_ids = {id(p) for p in backbone_params}

        # Group 2: Head + OOV embedding (fresh parameters) at higher LR
        other_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in backbone_ids
        ]

        param_groups = [
            dict(params=backbone_params, lr=hp.lr_backbone, weight_decay=hp.weight_decay),
            dict(params=other_params,    lr=hp.lr_head,     weight_decay=hp.weight_decay),
        ]
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

        # Cosine annealing with linear warmup
        # T_max=100 aligned with expected early stopping around epoch 70-100
        # (node3-1-1 best at epoch 12, but with full baseline + STRING_GNN it should converge slower)
        warmup_epochs = hp.warmup_epochs
        cosine_t_max  = hp.cosine_t_max

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(warmup_epochs)
            progress = float(current_epoch - warmup_epochs) / float(
                max(1, cosine_t_max - warmup_epochs)
            )
            progress = min(progress, 1.0)  # clamp to prevent second cycle
            return max(1e-6, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ── Checkpoint: save only trainable parameters + buffers ─────────────────

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items()
              if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(
            f"Saving ckpt: {trained:,}/{total:,} params ({100*trained/total:.2f}%)"
        )
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Node 3-1-3 – AIDO.Cell-100M LoRA + Full Baseline + Dual-Stream + STRING_GNN"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--lr-backbone",        type=float, default=1e-4)
    p.add_argument("--lr-head",            type=float, default=3e-4)
    p.add_argument("--weight-decay",       type=float, default=0.01)
    p.add_argument("--label-smoothing",    type=float, default=0.05)
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--micro-batch-size",   type=int,   default=4)
    p.add_argument("--global-batch-size",  type=int,   default=32)
    p.add_argument("--max-epochs",         type=int,   default=200)
    p.add_argument("--warmup-epochs",      type=int,   default=5)
    p.add_argument("--cosine-t-max",       type=int,   default=100)
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

    dm = TripleStreamDataModule(
        args.data_dir, args.micro_batch_size, args.num_workers
    )
    lit = TripleStreamLitModule(
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        cosine_t_max=args.cosine_t_max,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(
        monitor="val_f1", mode="max",
        patience=args.patience, min_delta=1e-5
    )
    lr_cb  = LearningRateMonitor(logging_interval="epoch")
    pb_cb  = TQDMProgressBar(refresh_rate=10)
    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps    = -1
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False

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
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        limit_test_batches=limit_test,
        val_check_interval=(
            args.val_check_interval
            if (args.debug_max_step is None and not args.fast_dev_run)
            else 1.0
        ),
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 3-1-3 – AIDO.Cell-100M LoRA r=16 + Full Baseline + Dual-Stream + STRING_GNN\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
