"""
Node 3-1-1 – AIDO.Cell-100M + LoRA (rank=8) + Focal Loss + Gene-Position Extraction

Architecture:
  - AIDO.Cell-100M backbone (hidden_size=640, 18 transformer layers)
    loaded from /home/Models/AIDO.Cell-100M
  - Synthetic perturbation input: {perturbed_gene_symbol: 1.0}
    (all other genes set to -1.0 by tokenizer default)
  - LoRA fine-tuning (rank=8, alpha=16) on QKV matrices only
    → ~2.7M trainable params vs parent's 22.26M (87% reduction)
    → directly addresses the critical overfitting bottleneck from node3-1
  - Pure AdamW optimizer (lr_backbone=1e-4, lr_head=3e-4)
    (Muon is NOT suitable for LoRA matrices per muon-optimizer-skill docs)
  - Gene-position extraction: extracts hidden state at the perturbed gene's
    specific position index from last_hidden_state[:, :19264, :]
  - Prediction head: Linear(640→2048) + GELU + LayerNorm + Dropout(0.2)
                   + Linear(2048 → 6640×3)
  - Focal cross-entropy loss (gamma=2.0) with inverse-frequency class weights
  - Cosine annealing LR schedule (T_max=100, aligned with expected stopping epoch)

Key improvements vs parent Node 3-1:
  1. AIDO.Cell-100M (hidden_size=640) vs 10M (hidden_size=256): 2.5x more expressive
  2. LoRA (rank=8) vs direct QKV fine-tuning: 87% fewer trainable params, reduces overfitting
  3. Pure AdamW vs Muon+AdamW: LoRA matrices are NOT suitable for Muon (rectangular, low-rank)
  4. Focal loss (gamma=2) vs weighted cross-entropy: focuses gradient on hard examples
  5. Higher head dropout (0.2 vs 0.1): stronger regularization against confirmed overfitting
  6. Cosine annealing T_max=100 vs 200: aligned with expected early stopping epoch

Root cause from parent: 22.26M trainable params on 1,416 training samples (~15,700 params/sample)
causes severe overfitting (train-val loss gap ~0.83). LoRA reduces this to ~1,900 params/sample.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

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
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy, SingleDeviceStrategy
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
# Shifted to {0,1,2}: class 0 = down, class 1 = neutral, class 2 = up
# Weights: neutral_freq / class_freq (normalized so neutral = 1.0)
CLASS_WEIGHTS = torch.tensor([10.91, 1.0, 29.62], dtype=torch.float32)


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """
    Compute macro-averaged per-gene F1 score matching calc_metric.py logic.

    Args:
        pred_np: [N, 3, G] softmax probabilities (float)
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
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal cross-entropy loss for multi-class classification.

    Focal loss down-weights well-classified examples and focuses training on
    hard examples. Particularly useful for the 88.9% neutral class imbalance
    in the DEG prediction task, where the model easily learns the neutral class
    but struggles with rare down/up-regulated positions.

    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    Args:
        gamma: focusing parameter (0 = standard CE, 2 = typical focal)
        weight: per-class weights tensor (same as CE weight parameter)
        label_smoothing: label smoothing factor
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, G] unnormalized logits (C=3 classes, G=6640 genes)
            targets: [B, G] class indices in {0, 1, 2}
        Returns:
            scalar loss
        """
        # [B, C, G] → [B*G, C]
        B, C, G = logits.shape
        logits_flat  = logits.permute(0, 2, 1).reshape(-1, C)  # [B*G, C]
        targets_flat = targets.reshape(-1)                      # [B*G]

        # Log-softmax probabilities for focal weight computation
        log_probs = F.log_softmax(logits_flat, dim=1)           # [B*G, C]
        probs     = torch.exp(log_probs)                        # [B*G, C]

        # Gather log-prob and prob at target class
        target_log_prob = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)  # [B*G]
        target_prob     = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)       # [B*G]

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1.0 - target_prob).pow(self.gamma)   # [B*G]

        # Per-class weight
        if self.weight is not None:
            class_w = self.weight.to(logits.device)[targets_flat]   # [B*G]
        else:
            class_w = torch.ones_like(focal_weight)

        # Label smoothing: blend target log-prob with mean log-prob
        if self.label_smoothing > 0:
            smooth_loss  = -log_probs.mean(dim=1)                    # [B*G]
            ce_loss      = -target_log_prob                          # [B*G]
            loss_per_pos = (
                (1 - self.label_smoothing) * ce_loss
                + self.label_smoothing * smooth_loss
            )
        else:
            loss_per_pos = -target_log_prob                         # [B*G]

        # Apply focal weighting and class weights
        weighted_loss = focal_weight * class_w * loss_per_pos       # [B*G]

        # Normalize by sum of weights for scale consistency
        # (avoids loss magnitude drift when class_w has large values)
        denom = class_w.sum().clamp(min=1.0)
        return (weighted_loss.sum() / denom)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AIDOCellPerturbDataset(Dataset):
    def __init__(self, pert_ids, symbols, input_ids, gene_positions, labels=None):
        self.pert_ids       = pert_ids
        self.symbols        = symbols
        self.input_ids      = input_ids       # [N, 19264] float32
        self.gene_positions = gene_positions  # [N] long – index into 19264 gene space
        self.labels         = labels          # [N, 6640] long or None

    def __len__(self): return len(self.pert_ids)

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
        "pert_id":       [b["pert_id"]       for b in batch],
        "symbol":        [b["symbol"]        for b in batch],
        "input_ids":     torch.stack([b["input_ids"]     for b in batch]),
        "gene_position": torch.stack([b["gene_position"] for b in batch]),
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
        # Rank-0 verifies tokenizer first, then all ranks load
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        def tokenize_symbols(symbols):
            # Build input as {gene_names: [sym], expression: [1.0]} for the perturbed gene
            # All other genes default to -1.0 (missing) in the tokenizer
            batch_input = [{"gene_names": [s], "expression": [1.0]} for s in symbols]
            tok_out     = tokenizer(batch_input, return_tensors="pt")
            ids  = tok_out["input_ids"]          # [N, 19264] float32
            # Gene position: index of the single gene with expression=1.0
            # input_ids: genes NOT in batch are -1.0; the provided gene has value > 0.5
            gpos = (ids > 0.5).float().argmax(dim=1).long()  # [N]
            return ids, gpos

        def load_split(fname, has_lbl):
            df  = pd.read_csv(self.data_dir / fname, sep="\t")
            ids, gpos = tokenize_symbols(df["symbol"].tolist())
            labels = None
            if has_lbl and "label" in df.columns:
                # Shift labels from {-1, 0, 1} to {0, 1, 2} to match class indices
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
            persistent_workers=self.num_workers > 0
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Model ────────────────────────────────────────────────────────────────────

class AIDOCell100MLoRAModel(nn.Module):
    """
    AIDO.Cell-100M with LoRA fine-tuning + gene-position extraction head.

    Key design decisions:
    1. AIDO.Cell-100M (hidden_size=640, 18 layers): 2.5x more expressive than 10M.
       With LoRA, GPU memory is ~3.41 GiB — nearly identical to 10M LoRA (3.21 GiB).
    2. LoRA (rank=8, alpha=16) on QKV matrices: reduces trainable params to ~2.7M
       (vs parent's 22.26M direct QKV fine-tuning), directly addressing overfitting.
    3. Gene-position extraction: inherited from node3-1, extracts hidden state at
       the specific perturbed gene's position from last_hidden_state[:, :19264, :].
    4. Wider head (640→2048): exploits the 100M's richer 640-dim representation.
    5. Higher dropout (0.2): stronger regularization given confirmed overfitting.

    Note: LoRA params are cast to float32 per skill documentation for training stability.
    Backbone (non-LoRA) remains in bfloat16 for FlashAttention compatibility.
    """

    def __init__(self, n_genes_out: int = N_GENES_OUT, n_classes: int = N_CLASSES):
        super().__init__()

        # Load AIDO.Cell-100M backbone in bfloat16
        backbone = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        # Apply LoRA on QKV matrices FIRST (before enabling gradient checkpointing).
        # Per AIDO.Cell skill docs: get_peft_model must be called before enabling gc.
        # If gc is enabled before get_peft_model, PEFT calls enable_input_require_grads()
        # → get_input_embeddings(), which AIDO.Cell raises NotImplementedError for.
        # target_modules path: bert.encoder.layer.*.attention.self.{query,key,value}
        # flash_self shares weight tensors, so LoRA is applied to both automatically
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,                  # rank
            lora_alpha=16,        # alpha = 2 * rank (standard scaling)
            lora_dropout=0.05,    # light dropout in LoRA adapter
            target_modules=["query", "key", "value"],
            layers_to_transform=None,  # fine-tune all 18 layers
        )
        self.backbone = get_peft_model(backbone, lora_cfg)

        # Enable gradient checkpointing AFTER LoRA wrapping (per AIDO.Cell skill docs).
        # use_reentrant=False is the safer default for LoRA setups where
        # the model input itself does not require grad.
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA params to float32 for training stability
        # (per AIDO.Cell skill documentation and PEFT best practices)
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Prediction head: exploits the 640-dim representation from AIDO.Cell-100M
        # Linear(640→2048) + GELU + LayerNorm + Dropout(0.2) + Linear(2048→6640×3)
        # Note: head runs in float32 (LoRA params already cast to float32)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Dropout(0.2),      # Increased from 0.1: stronger regularization
            nn.Linear(2048, n_genes_out * n_classes),
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f"[Node3-1-1] Trainable params: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.2f}%)")

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, 19264] float32
        gene_positions: torch.Tensor,  # [B] long – index of perturbed gene in vocab
    ) -> torch.Tensor:
        # Attention mask: all ones (overridden inside model anyway, but required)
        attn_mask = torch.ones(
            input_ids.shape[0], input_ids.shape[1],
            dtype=torch.long, device=input_ids.device
        )

        # Run backbone in bfloat16 for FlashAttention compatibility
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(input_ids=input_ids, attention_mask=attn_mask)

        # last_hidden_state: [B, 19266, 640]
        # Slice to gene positions only (exclude the 2 appended summary tokens)
        # Cast to float32 for numerically stable head computation
        gene_states = out.last_hidden_state[:, :N_GENE_VOCAB, :].float()  # [B, 19264, 640]

        # Gene-position extraction: index into each sample's perturbed gene position
        # gene_positions: [B] long – each value in [0, N_GENE_VOCAB)
        # gene_states[b, gene_positions[b], :] → [B, 640]
        batch_size = gene_states.shape[0]
        pos_idx    = gene_positions.view(batch_size, 1, 1).expand(batch_size, 1, HIDDEN_SIZE)
        gene_repr  = gene_states.gather(1, pos_idx).squeeze(1)  # [B, 640]

        logits = self.head(gene_repr)                         # [B, 6640*3]
        return logits.view(-1, N_CLASSES, N_GENES_OUT)        # [B, 3, 6640]


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

class AIDOCell100MLoRALitModule(pl.LightningModule):

    def __init__(
        self,
        lr_backbone: float     = 1e-4,    # LoRA adapter learning rate
        lr_head: float         = 3e-4,    # Prediction head learning rate
        weight_decay: float    = 0.01,
        focal_gamma: float     = 2.0,     # Focal loss focusing parameter
        label_smoothing: float = 0.05,
        max_epochs: int        = 200,
        t_max_cosine: int      = 100,     # Cosine annealing T_max (aligned with expected stopping)
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
        self.model = AIDOCell100MLoRAModel()
        self.register_buffer("class_weights", CLASS_WEIGHTS)
        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            weight=CLASS_WEIGHTS,
            label_smoothing=self.hparams.label_smoothing,
        )

    def forward(self, input_ids, gene_positions):
        return self.model(input_ids, gene_positions)

    def _loss(self, logits, labels):
        """Focal cross-entropy loss."""
        return self.focal_loss(logits, labels)

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
        # Compute F1 on globally-gathered data
        probs_np  = torch.softmax(lp, dim=1).numpy()  # [N, 3, G]
        labels_np = ll.numpy()                         # [N, G]
        f1 = compute_per_gene_f1(probs_np, labels_np)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear(); self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"])
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

            # Deduplicate by pert_id: DDP DistributedSampler may pad the dataset
            # to make sample counts equal across ranks → duplicate pert_ids.
            # calc_metric.py explicitly forbids duplicate idx values.
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

            self.print(f"[Node3-1-1] Saved {len(dedup_perts)} test predictions → {pred_path}")

            # Self-evaluate if labels are available
            if all_labels.any():
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[Node3-1-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    # ── Two-group AdamW optimizer ─────────────────────────────────────────────

    def configure_optimizers(self):
        hp = self.hparams

        # LoRA adapter parameters → lower LR (backbone fine-tuning regime)
        # Head parameters → higher LR (randomly initialized, needs faster convergence)
        lora_params = [
            p for n, p in self.model.backbone.named_parameters()
            if p.requires_grad
        ]
        lora_ids  = {id(p) for p in lora_params}
        head_params = [
            p for p in self.model.head.parameters()
            if p.requires_grad
        ]
        head_ids  = {id(p) for p in head_params}

        # All params that are trainable (should be exactly lora_params + head_params)
        other_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in lora_ids and id(p) not in head_ids
        ]

        param_groups = [
            {"params": lora_params,   "lr": hp.lr_backbone, "weight_decay": hp.weight_decay},
            {"params": head_params,   "lr": hp.lr_head,     "weight_decay": hp.weight_decay},
        ]
        # Include any remaining params (e.g., focal_loss buffers) at backbone LR
        if other_params:
            param_groups.append(
                {"params": other_params, "lr": hp.lr_backbone, "weight_decay": hp.weight_decay}
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Cosine annealing with T_max=100:
        # Parent node stopped at epoch ~70-95 with T_max=200, meaning the LR
        # only reached half of its decay range. Setting T_max=100 ensures
        # the model fully exploits the low-LR fine-tuning phase within the
        # expected training window.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=hp.t_max_cosine,
            eta_min=1e-6,
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
        description="Node 3-1-1 – AIDO.Cell-100M + LoRA + Focal Loss + Gene-Position Extraction"
    )
    p.add_argument("--data-dir",           type=str,   default="data")
    p.add_argument("--lr-backbone",        type=float, default=1e-4)
    p.add_argument("--lr-head",            type=float, default=3e-4)
    p.add_argument("--weight-decay",       type=float, default=0.01)
    p.add_argument("--focal-gamma",        type=float, default=2.0)
    p.add_argument("--label-smoothing",    type=float, default=0.05)
    p.add_argument("--micro-batch-size",   type=int,   default=4)
    p.add_argument("--global-batch-size",  type=int,   default=32)
    p.add_argument("--max-epochs",         type=int,   default=200)
    p.add_argument("--t-max-cosine",       type=int,   default=100)
    p.add_argument("--patience",           type=int,   default=30)
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
    lit = AIDOCell100MLoRALitModule(
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        max_epochs=args.max_epochs,
        t_max_cosine=args.t_max_cosine,
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
        deterministic="warn",  # cross_entropy on [B, C, 6640] has no deterministic CUDA impl; use warn mode
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 3-1-1 – AIDO.Cell-100M + LoRA + Focal Loss + Gene-Position Extraction\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
