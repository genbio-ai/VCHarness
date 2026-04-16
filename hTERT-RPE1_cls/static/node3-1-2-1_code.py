"""
Node 3-1-2-1 – AIDO.Cell-100M + LoRA rank=32 (QKV+FFN) + Full Baseline Input + Mean-Pool

Architecture:
  - AIDO.Cell-100M backbone (hidden_size=640, 18 transformer layers)
    loaded from /home/Models/AIDO.Cell-100M
  - FULL MULTI-GENE BASELINE INPUT: all 19,264 genes at expression=1.0, perturbed gene at 0.0
    (knockdown representation: all other genes at baseline, perturbed gene silenced)
    Verified: node3-2-1 with this approach achieved F1=0.4405 (+0.031 vs OOD sparse input)
  - LoRA rank=32 (alpha=64) on QKV matrices AND FFN layers (gate_proj, up_proj, down_proj)
    across all 18 transformer layers (~6.2M LoRA trainable params)
  - MEAN-POOL AGGREGATION: mean over last_hidden_state[:, :19264, :] → [B, 640]
    captures the entire "perturbed cell state" vs single-gene extraction in parent
    This is the primary unexplored direction identified in node3-1-2 and node3-2-1 feedbacks
  - Prediction head: Linear(640→1536) + GELU + LayerNorm(1536) + Dropout(0.3)
                   + Linear(1536 → 6640×3)
  - Focal cross-entropy loss (gamma=2.0) with capped class weights [10.91, 1.0, 15.0]
  - Label smoothing: 0.05
  - AdamW: backbone LoRA lr=1e-4, head lr=2e-4
  - 2-epoch linear warmup + cosine annealing (T_max=100)

Key improvements vs Parent Node 3-1-2 (test F1=0.41475):
  1. FULL BASELINE INPUT: all 19,264 genes at 1.0, perturbed at 0.0
     Parent implemented only {perturbed=0.0, others=-1.0} (not the full multi-gene baseline).
     Full baseline proven: node3-2-1 F1=0.4405 vs OOD sparse F1=0.4096 → +0.031 gain.
  2. MEAN-POOL AGGREGATION: mean over all 19,264 gene hidden states (vs single-gene extraction)
     The parent's single-gene extraction (one 640-dim vector from 19,266) is an extreme
     information bottleneck. Mean-pool captures the whole perturbed cell state.
     Identified as "primary high-priority unexplored direction" in both node3-1-2 and node3-2-1
     feedback reports.
  3. LoRA RANK=32 (from 16): more expressive backbone adaptation.
     Evidence: node3-2-1 with r=32 + full baseline achieved F1=0.4405.
  4. LoRA ON FFN LAYERS (gate_proj, up_proj, down_proj): ~4M additional LoRA params.
     Allows backbone to adapt feature transformation (not just attention patterns).
     Identified as MODERATE PRIORITY in node3-1-2 feedback.
  5. SHORTER WARMUP (2 epochs vs 5): parent's 5-epoch warmup delayed best epoch to 85.
  6. CAPPED CLASS WEIGHTS at [10.91, 1.0, 15.0]: reduces training instability from 29.62 up-class.
  7. HEAD DROPOUT=0.3 (from 0.2): stronger regularization given val loss divergence in parent.
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
N_GENE_VOCAB  = 19264 # AIDO.Cell gene vocabulary size

# Class weights: inverse-frequency based on train split label distribution
# down-regulated (-1): 8.14%, neutral (0): 88.86%, up-regulated (+1): 3.00%
# Shifted to {0,1,2}: class 0=down, class 1=neutral, class 2=up
# Capped: up-class weight at 15.0 (was 29.62) to reduce training instability
# Evidence from node3-1-2 feedback: "29.62 is very high and may cause training instability"
CLASS_WEIGHTS = torch.tensor([10.91, 1.0, 15.0], dtype=torch.float32)


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal cross-entropy loss with class weights and label smoothing.
    focal_weight = (1 - p_t)^gamma
    Downweights easy examples (neutral class) and focuses on hard/minority ones.
    """

    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, G] raw logits
            targets: [B, G] class indices in {0, 1, 2}
        Returns:
            scalar focal loss
        """
        B, C, G = logits.shape
        logits_2d  = logits.permute(0, 2, 1).reshape(-1, C)   # [B*G, C]
        targets_1d = targets.reshape(-1)                        # [B*G]

        with torch.no_grad():
            probs   = F.softmax(logits_2d.float(), dim=1)       # [B*G, C]
            p_t     = probs.gather(1, targets_1d.unsqueeze(1)).squeeze(1)  # [B*G]
            focal_w = (1.0 - p_t.clamp(0.0, 1.0)) ** self.gamma  # [B*G]

        w = self.weight.to(logits_2d.device) if self.weight is not None else None
        per_sample_ce = F.cross_entropy(
            logits_2d, targets_1d,
            weight=w,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )   # [B*G]

        focal_loss = focal_w * per_sample_ce  # [B*G]

        if w is not None:
            class_w_sum = w.sum()
            return focal_loss.sum() / class_w_sum.clamp(min=1.0)
        else:
            return focal_loss.mean()


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
    def __init__(self, pert_ids, symbols, input_ids, labels=None):
        self.pert_ids   = pert_ids
        self.symbols    = symbols
        self.input_ids  = input_ids   # [N, 19264] float32 – full multi-gene baseline
        self.labels     = labels      # [N, 6640] long or None

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id":   self.pert_ids[idx],
            "symbol":    self.symbols[idx],
            "input_ids": self.input_ids[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch):
    out = {
        "pert_id":   [b["pert_id"]   for b in batch],
        "symbol":    [b["symbol"]    for b in batch],
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
    }
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


# ─── DataModule ───────────────────────────────────────────────────────────────

class AIDOCell100MDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data", micro_batch_size=4, num_workers=2):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        # Rank-0 downloads/verifies tokenizer first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        def tokenize_symbols_realistic(symbols):
            """
            Build FULL MULTI-GENE BASELINE knockdown input.

            Key insight (from node3-2-1 feedback, +0.031 F1 gain):
            Instead of {perturbed_gene: 0.0, others: -1.0} (sparse, OOD for AIDO.Cell),
            we provide {ALL 19,264 genes: 1.0, perturbed_gene: 0.0}.

            This simulates a knockdown in a baseline expressing cell:
            - All 19,263 non-perturbed genes: expression=1.0 (baseline active)
            - Perturbed gene: expression=0.0 (knocked down / silenced)

            After AIDO.Cell's internal _prepare_inputs() CP10K normalization:
            - total counts = 19263 (baseline genes at 1.0)
            - normalized value: log1p(1.0 / 19263 * 10000) ≈ 0.416 per expressed gene
            - perturbed gene: log1p(0.0) = 0.0

            This keeps AIDO.Cell in-distribution and enables it to leverage
            its learned gene-gene co-expression patterns from pretraining.

            Implementation (same as node3-2-1, proven approach):
            Step 1: Tokenize each sample with only the perturbed gene at 1.0
                    to discover its vocabulary position.
            Step 2: Replace all -1.0 (missing) positions with 1.0 (baseline).
            Step 3: Set the perturbed gene's position back to 0.0 (knockdown).
            """
            # Step 1: Get the perturbed gene's vocab position
            batch_input = [{"gene_names": [s], "expression": [1.0]} for s in symbols]
            tok_out     = tokenizer(batch_input, return_tensors="pt")
            ids         = tok_out["input_ids"]   # [N, 19264] float32; -1.0 = missing

            # Find perturbed gene position (only slot with value > 0.5)
            gene_found = (ids > 0.5).any(dim=1)                   # [N] bool
            gpos       = (ids > 0.5).float().argmax(dim=1).long() # [N]

            # Step 2: Build full baseline input
            realistic_ids = ids.clone()
            # Replace all missing genes (-1.0) with baseline expression (1.0)
            realistic_ids[realistic_ids < 0] = 1.0
            # Step 3: Set the perturbed gene to 0.0 (knockdown)
            batch_size = realistic_ids.shape[0]
            for i in range(batch_size):
                if gene_found[i]:
                    realistic_ids[i, gpos[i]] = 0.0
                # If gene not in vocabulary: all stay at 1.0 (baseline only, no knockdown)

            return realistic_ids

        def load_split(fname, has_lbl):
            df  = pd.read_csv(self.data_dir / fname, sep="\t")
            ids = tokenize_symbols_realistic(df["symbol"].tolist())
            labels = None
            if has_lbl and "label" in df.columns:
                # Shift labels from {-1, 0, 1} to class indices {0, 1, 2}
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return AIDOCellPerturbDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(), ids, labels
            )

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=self.num_workers,
            pin_memory=True, drop_last=shuffle,
            persistent_workers=self.num_workers > 0
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Model ────────────────────────────────────────────────────────────────────

class AIDOCell100MModel(nn.Module):
    """
    AIDO.Cell-100M with LoRA (rank=32) fine-tuning on QKV + FFN + mean-pool head.

    Key design decisions:
    1. FULL MULTI-GENE INPUT: all 19,264 genes at 1.0, perturbed at 0.0.
       Provides in-distribution context for AIDO.Cell-100M's co-expression attention.
       Proven: node3-2-1 achieved F1=0.4405 (+0.031) with this representation.

    2. MEAN-POOL AGGREGATION: mean over last_hidden_state[:, :19264, :] → [B, 640].
       Primary unexplored direction from node3-1-2 and node3-2-1 feedbacks.
       Captures the entire "perturbed cell state" rather than a single gene's context.
       More information-rich than single-gene extraction (used in all AIDO.Cell nodes so far).

    3. LoRA rank=32 on QKV + FFN (gate_proj, up_proj, down_proj):
       - QKV: ~2.2M LoRA params (18 layers × 3 matrices × r=32)
       - FFN: ~4.0M LoRA params (18 layers × 3 proj × r=32)
       - Total LoRA: ~6.2M trainable parameters
       Allows backbone to adapt both attention patterns AND feature transformations.
    """

    def __init__(self, lora_rank: int = 32, lora_alpha: int = 64,
                 lora_dropout: float = 0.05,
                 n_genes_out: int = N_GENES_OUT, n_classes: int = N_CLASSES):
        super().__init__()
        # Load backbone in bfloat16 for memory efficiency
        self.backbone = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        self.backbone = self.backbone.to(torch.bfloat16)
        self.backbone.config.use_cache = False

        # Apply LoRA BEFORE enabling gradient checkpointing (required ordering per PEFT)
        # Target QKV attention matrices + FFN projection layers
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value", "gate_proj", "up_proj", "down_proj"],
            # QKV: bert.encoder.layer.*.attention.self.{query,key,value}
            # FFN: bert.encoder.layer.*.mlp.{gate_proj,up_proj,down_proj}
            # flash_self shares same weight tensors as self, so LoRA applies automatically
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

        # Enable gradient checkpointing AFTER LoRA (required to avoid NotImplementedError)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA parameters to float32 for stable optimization
        # (backbone main weights remain bfloat16)
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Prediction head: Linear(640→1536) + GELU + LayerNorm(1536) + Dropout(0.3) + Linear(1536→6640×3)
        # Head dropout=0.3 (vs 0.2 in parent): stronger regularization given val loss divergence in parent
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 1536),
            nn.GELU(),
            nn.LayerNorm(1536),
            nn.Dropout(0.3),
            nn.Linear(1536, n_genes_out * n_classes),
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Node3-1-2-1] Trainable params: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.2f}%)")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, 19264] float32 – full multi-gene baseline expression values
                       (all genes=1.0, perturbed gene=0.0)
        Returns:
            logits: [B, 3, 6640] class logits
        """
        # Attention mask: all ones (model overrides this anyway, but API requires it)
        attn_mask = torch.ones(
            input_ids.shape[0], input_ids.shape[1],
            dtype=torch.long, device=input_ids.device
        )

        # Forward through AIDO.Cell-100M backbone with bfloat16 autocast
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(input_ids=input_ids, attention_mask=attn_mask)

        # last_hidden_state: [B, 19266, 640]
        # Slice to gene positions only (exclude the 2 appended summary tokens)
        gene_states = out.last_hidden_state[:, :N_GENE_VOCAB, :].float()  # [B, 19264, 640]

        # MEAN-POOL AGGREGATION: mean over all 19,264 gene positions → [B, 640]
        # This captures the entire "perturbed cell state" representation.
        # With full multi-gene input, each gene attends to all 19,263 other genes via
        # self-attention, so the mean-pool integrates the global transcriptomic context.
        # This is fundamentally richer than extracting a single gene's hidden state.
        cell_repr = gene_states.mean(dim=1)  # [B, 640]

        logits = self.head(cell_repr)                              # [B, 6640*3]
        return logits.view(-1, N_CLASSES, N_GENES_OUT)            # [B, 3, 6640]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    """Gather tensors from all DDP ranks, handling variable-size padding."""
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

class AIDOCell100MLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_backbone:    float = 1e-4,
        lr_head:        float = 2e-4,
        weight_decay:   float = 0.01,
        label_smoothing: float = 0.05,
        focal_gamma:    float = 2.0,
        max_epochs:     int   = 200,
        warmup_epochs:  int   = 2,    # Shorter warmup (vs 5 in parent) to avoid epoch-85 delay
        t_max:          int   = 100,
        lora_rank:      int   = 32,
        lora_alpha:     int   = 64,
        lora_dropout:   float = 0.05,
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
        self.model = AIDOCell100MModel(
            lora_rank=self.hparams.lora_rank,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
        )
        self.register_buffer("class_weights", CLASS_WEIGHTS)
        self.focal_loss_fn = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=CLASS_WEIGHTS,
            label_smoothing=self.hparams.label_smoothing,
        )

    def forward(self, input_ids):
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"])
        loss   = self.focal_loss_fn(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"])
        if "label" in batch:
            loss = self.focal_loss_fn(logits, batch["label"])
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
        labels_np = ll.numpy()                          # [N, G]
        f1 = compute_per_gene_f1(probs_np, labels_np)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"])
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
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id: DDP DistributedSampler may pad the dataset
            # to make sample counts equal across ranks → duplicate pert_ids.
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

            self.print(f"[Node3-1-2-1] Saved {len(dedup_perts)} test predictions → {pred_path}")

            # Self-evaluate if labels are available
            if any(r.any() for r in dedup_label_rows):
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[Node3-1-2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    # ── Optimizer: AdamW with two LR groups + warmup + cosine annealing ──

    def configure_optimizers(self):
        hp = self.hparams

        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        head_params     = list(self.model.head.parameters())

        param_groups = [
            {"params": backbone_params, "lr": hp.lr_backbone, "weight_decay": hp.weight_decay},
            {"params": head_params,     "lr": hp.lr_head,     "weight_decay": hp.weight_decay},
        ]
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

        # Shorter warmup (2 epochs) + cosine annealing (T_max epochs)
        # Shorter warmup vs parent's 5 epochs: parent's warmup delayed best epoch to 85;
        # 2 epochs provides stability without excessive delay
        import math
        warmup_epochs = hp.warmup_epochs
        t_max         = hp.t_max
        eta_min_ratio = 1e-6 / hp.lr_backbone  # ratio of eta_min to initial LR

        def lr_lambda_warmup_cosine(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            else:
                progress = (epoch - warmup_epochs) / float(max(1, t_max - warmup_epochs))
                progress = min(progress, 1.0)
                return max(eta_min_ratio,
                           0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda_warmup_cosine
        )
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
        description="Node 3-1-2-1 – AIDO.Cell-100M + LoRA rank=32 (QKV+FFN) + Full Baseline + Mean-Pool"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--lr-backbone",        type=float, default=1e-4)
    p.add_argument("--lr-head",            type=float, default=2e-4)
    p.add_argument("--weight-decay",       type=float, default=0.01)
    p.add_argument("--label-smoothing",    type=float, default=0.05)
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--lora-rank",          type=int,   default=32)
    p.add_argument("--lora-alpha",         type=int,   default=64)
    p.add_argument("--lora-dropout",       type=float, default=0.05)
    p.add_argument("--micro-batch-size",   type=int,   default=4)
    p.add_argument("--global-batch-size",  type=int,   default=32)
    p.add_argument("--max-epochs",         type=int,   default=200)
    p.add_argument("--warmup-epochs",      type=int,   default=2)
    p.add_argument("--t-max",              type=int,   default=100)
    p.add_argument("--patience",           type=int,   default=35)
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
        args.data_dir, args.micro_batch_size, args.num_workers
    )
    lit = AIDOCell100MLitModule(
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        t_max=args.t_max,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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

    max_steps:  int          = -1
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
        deterministic="warn",  # cross_entropy on [B, C, 6640] has no deterministic CUDA impl
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 3-1-2-1 – AIDO.Cell-100M + LoRA rank=32 (QKV+FFN) + Full Baseline + Mean-Pool\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
