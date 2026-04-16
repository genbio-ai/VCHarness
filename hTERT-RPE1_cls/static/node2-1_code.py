"""
Node 2-1 — AIDO.Cell-100M + LoRA r=32 + Gene-Position Extraction
             with Realistic Multi-Gene Input Profile

Architecture:
  - AIDO.Cell-100M backbone loaded from /home/Models/AIDO.Cell-100M
  - Realistic multi-gene input: all 19,264 genes at expression=1.0 (baseline),
    perturbed gene overridden to expression=10.0.
    Fixes the critical OOD issue in prior nodes where only 1 gene was expressed.
  - LoRA fine-tuning (r=32, alpha=64) on Q/K/V attention layers (all 18 layers)
  - Gene representation extracted at the perturbed gene's vocabulary position
    (gene-position extraction, as in node3-1-1 which achieved F1=0.4258)
  - Multi-layer feature fusion: weighted average of the last 6 transformer layers
  - Deep prediction head: LayerNorm → Linear(640→2048) → GELU → Dropout(0.1) → Linear(2048→6640×3)
  - Focal loss only (no class weights, to avoid double-penalization)
  - Cosine Annealing LR with warmup
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

AIDO_CELL_DIR = "/home/Models/AIDO.Cell-100M"
N_GENES_OUT   = 6640
N_CLASSES     = 3
HIDDEN_SIZE   = 640   # AIDO.Cell-100M
N_LAYERS      = 18   # total transformer layers
FUSION_LAYERS = 6    # number of trailing layers to fuse
LORA_R        = 32   # LoRA rank (increased from r=16 in parent for richer adaptation)
LORA_ALPHA    = 64   # LoRA alpha = 2 × rank


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    No class weights — focal mechanism handles imbalance via (1-pt)^gamma.
    Avoids double-penalization that occurs when combining class weights AND focal loss.
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, C] (2D, already reshaped)
        targets: [N] long
        """
        n_classes = logits.shape[-1]
        # Compute standard cross-entropy with optional label smoothing
        ce_loss = F.cross_entropy(
            logits, targets,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        # Get probability of the true class for focal weighting
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction='none'))
        # Focal weight: down-weight easy examples
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Per-gene macro F1 matching calc_metric.py.  pred_np: [N,3,G], labels_np: [N,G]."""
    pred_cls = pred_np.argmax(axis=1)
    f1_vals = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g]
        yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PerturbDataset(Dataset):
    """Stores pert_ids, symbols, and labels (gene_positions computed on-the-fly in collate)."""

    def __init__(
        self,
        pert_ids: List[str],
        symbols: List[str],
        labels: Optional[torch.Tensor] = None,  # [N, 6640] long or None
    ):
        self.pert_ids = pert_ids
        self.symbols  = symbols
        self.labels   = labels

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id": self.pert_ids[idx],
            "symbol":  self.symbols[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def build_collate_fn(tokenizer):
    """
    Returns a collate function that builds realistic multi-gene input profiles.

    For each sample: all 19,264 genes at expression=1.0 (baseline),
    perturbed gene overridden to expression=10.0.

    This resolves the critical OOD issue in parent nodes where a single-gene
    profile was provided to AIDO.Cell (which was trained on rich multi-gene profiles).
    """
    # Pre-build the list of all gene names in vocabulary order
    all_gene_names = list(tokenizer.gene_to_index.keys())  # 19264 gene names
    gene_name_set  = set(all_gene_names)

    def collate_fn(batch):
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        B = len(batch)

        # Build per-sample expression dicts with realistic multi-gene context
        expr_dicts = []
        gene_vocab_positions = []  # index in 19264-gene vocab for each sample

        for sym in symbols:
            # Baseline: all genes at expression=1.0
            expr = {g: 1.0 for g in all_gene_names}
            # Perturbed gene: override to 10.0 (10x elevated signal)
            if sym in gene_name_set:
                expr[sym] = 10.0
                gene_vocab_positions.append(tokenizer.gene_to_index[sym])
            else:
                # Out-of-vocabulary: keep baseline for all genes, use position 0 as placeholder
                # Mean-pool at inference will average over all genes anyway, but we track
                # a valid index for the extraction path
                gene_vocab_positions.append(0)
            expr_dicts.append(expr)

        # Tokenize batch
        tok_out = tokenizer(expr_dicts, return_tensors="pt")
        input_ids = tok_out["input_ids"]  # [B, 19264] float32

        gene_positions = torch.tensor(gene_vocab_positions, dtype=torch.long)  # [B]

        out = {
            "pert_id":       pert_ids,
            "symbol":        symbols,
            "input_ids":     input_ids,      # [B, 19264] float32
            "gene_position": gene_positions, # [B] long — index in 19264-gene vocab
        }
        if "label" in batch[0]:
            out["label"] = torch.stack([b["label"] for b in batch])
        return out

    return collate_fn


# ─── DataModule ───────────────────────────────────────────────────────────────

class AIDOCellDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data",
        micro_batch_size: int = 4,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage: Optional[str] = None):
        # ── Load tokenizer (DDP-safe barrier) ─────────────────────────────
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        # ── Helper: load a split ───────────────────────────────────────────
        def load_split(fname: str, has_label: bool):
            df = pd.read_csv(self.data_dir / fname, sep="\t")
            pert_ids = df["pert_id"].tolist()
            symbols  = df["symbol"].tolist()
            labels   = None
            if has_label and "label" in df.columns:
                # Shift labels from {-1,0,1} to class indices {0,1,2}
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return PerturbDataset(pert_ids, symbols, labels)

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  False)

        # Build collate function (uses tokenizer internals)
        self.collate = build_collate_fn(self.tokenizer)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds, batch_size=self.micro_batch_size,
            shuffle=shuffle, collate_fn=self.collate,
            num_workers=self.num_workers, pin_memory=True,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Model ────────────────────────────────────────────────────────────────────

class AIDOCellPerturbModel(nn.Module):
    """
    AIDO.Cell-100M + LoRA r=32 backbone.

    Key improvements over parent (node2):
    1. Multi-gene realistic input profile (resolved via collate, not model)
    2. Higher LoRA rank r=32 (vs r=16) for richer backbone adaptation
    3. Gene-position extraction at the perturbed gene's vocabulary index
       (proven effective in node3-1-1 which achieved F1=0.4258)
    4. Multi-layer feature fusion (last 6 layers) — same as parent
    5. Deeper MLP head: 640→2048→19920 (adds non-linear capacity)
    """

    def __init__(self, n_genes_out: int = N_GENES_OUT, n_classes: int = N_CLASSES,
                 head_dropout: float = 0.1):
        super().__init__()
        # Load backbone
        backbone = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False

        # Patch enable_input_require_grads for AIDO.Cell
        # (AIDO.Cell does not implement get_input_embeddings(), which PEFT calls)
        def _safe_enable_input_require_grads():
            def _make_inputs_require_grad(module, input, output):
                if isinstance(output, torch.Tensor):
                    output.requires_grad_(True)
            backbone.bert.gene_embedding.register_forward_hook(_make_inputs_require_grad)
        backbone.enable_input_require_grads = _safe_enable_input_require_grads

        # Apply LoRA r=32 (increased from r=16) to Q/K/V in ALL 18 layers
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.05,
            target_modules=["query", "key", "value"],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Enable gradient checkpointing after PEFT wrapping
        self.backbone.base_model.model.config.use_cache = False
        self.backbone.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Cast LoRA params to float32 for training stability
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # Learnable layer-fusion weights for last FUSION_LAYERS layers
        self.layer_weights = nn.Parameter(torch.zeros(FUSION_LAYERS))

        # Deeper prediction head: LayerNorm → Linear(640→2048) → GELU → Dropout → Linear(2048→6640×3)
        # Adds non-linear capacity to map gene representation to 6,640 ternary DEG predictions
        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Linear(HIDDEN_SIZE, 2048),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(2048, n_genes_out * n_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, 19264] float32
        gene_positions: torch.Tensor,  # [B] long — index in 19264-gene vocab
    ) -> torch.Tensor:
        """Returns logits [B, 3, 6640]."""
        attn_mask = torch.ones(
            input_ids.shape[0], input_ids.shape[1],
            dtype=torch.long, device=input_ids.device
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

        # hidden_states: tuple of 19 tensors [B, 19266, 640]
        # indices 0..18; use last FUSION_LAYERS layers: [13..18]
        hidden_states = torch.stack(
            [out.hidden_states[i].float()
             for i in range(N_LAYERS - FUSION_LAYERS + 1, N_LAYERS + 1)],
            dim=0,
        )  # [FUSION_LAYERS, B, 19266, 640]

        # Weighted combination across fusion layers
        weights = torch.softmax(self.layer_weights, dim=0)  # [FUSION_LAYERS]
        fused = (hidden_states * weights[:, None, None, None]).sum(0)  # [B, 19266, 640]

        # Extract the representation at each sample's perturbed gene position in vocab
        # Gene-position extraction: most discriminative because it directly encodes
        # the perturbed gene's contextually-informed representation from the full
        # multi-gene baseline profile (now that the input is not OOD).
        B = fused.shape[0]
        gene_repr = fused[torch.arange(B, device=fused.device), gene_positions, :]  # [B, 640]

        # Decode to DEG signature via deep head
        logits = self.head(gene_repr)  # [B, 6640*3]
        return logits.view(-1, N_CLASSES, N_GENES_OUT)  # [B, 3, 6640]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))

    pad = max_sz - local_p.shape[0]
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

class AIDOCellLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_backbone: float = 5e-5,   # conservative LR for pretrained LoRA backbone
        lr_head: float = 3e-4,       # standard LR for fresh prediction head
        weight_decay: float = 1e-3,  # stronger regularization than parent (1e-4)
        focal_gamma: float = 2.0,
        head_dropout: float = 0.1,
        warmup_steps: int = 100,
        max_steps_total: int = 2000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds:     List[torch.Tensor] = []
        self._val_labels:    List[torch.Tensor] = []
        self._test_preds:    List[torch.Tensor] = []
        self._test_pert_ids: List[str] = []
        self._test_symbols:  List[str] = []
        self._test_labels:   List[torch.Tensor] = []

    def setup(self, stage: Optional[str] = None):
        self.model = AIDOCellPerturbModel(
            head_dropout=self.hparams.head_dropout
        )
        self.focal_loss = FocalLoss(
            gamma=self.hparams.focal_gamma,
            label_smoothing=0.0,  # No label smoothing to let focal loss handle hard examples
        )

    def forward(self, input_ids, gene_positions):
        return self.model(input_ids, gene_positions)

    def _loss(self, logits, labels):
        # Reshape to 2D before cross_entropy (deterministic nll_loss kernel)
        # logits: [B, 3, 6640] -> [B*6640, 3];  labels: [B, 6640] -> [B*6640]
        logits_2d = logits.float().permute(0, 2, 1).reshape(-1, N_CLASSES)
        labels_1d = labels.reshape(-1)
        return self.focal_loss(logits_2d, labels_1d)

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
            torch.cat(self._test_labels, 0)
            if self._test_labels
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
            all_pert, all_syms    = self._test_pert_ids, self._test_symbols

        if self.trainer.is_global_zero:
            out_dir   = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"

            # Deduplicate by pert_id (DDP may pad with duplicates)
            seen_pids: set = set()
            dedup_indices: List[int] = []
            for i, pid in enumerate(all_pert):
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    dedup_indices.append(i)

            all_probs_np = all_probs.numpy()
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for i in dedup_indices:
                    fh.write(
                        f"{all_pert[i]}\t{all_syms[i]}\t"
                        f"{json.dumps(all_probs_np[i].tolist())}\n"
                    )
            self.print(f"[Node2-1] Saved {len(dedup_indices)} test predictions → {pred_path}")

            if self._test_labels:
                dedup_probs  = all_probs_np[dedup_indices]
                dedup_labels = all_labels[dedup_indices].numpy()
                f1 = compute_per_gene_f1(dedup_probs, dedup_labels)
                self.print(f"[Node2-1] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        hp = self.hparams

        # Separate parameter groups for backbone (LoRA) and head
        backbone_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and ("backbone" in n or "layer_weights" in n)
        ]
        head_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and "head" in n
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": hp.lr_backbone,
                 "weight_decay": hp.weight_decay},
                {"params": head_params, "lr": hp.lr_head,
                 "weight_decay": hp.weight_decay},
            ]
        )

        # Cosine annealing with linear warmup
        # - Linear warmup: steps 0..warmup_steps
        # - Cosine decay: steps warmup_steps..max_steps_total
        warmup = hp.warmup_steps
        total  = hp.max_steps_total

        def lr_lambda(current_step: int):
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            progress = float(current_step - warmup) / float(max(1, total - warmup))
            return max(1e-6, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

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
        self.print(f"Saving ckpt: {trained}/{total} params ({100*trained/total:.2f}%)")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Node 2-1 — AIDO.Cell-100M LoRA r=32 + Gene-Position Extraction")
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--lr-backbone",       type=float, default=5e-5,
                   help="Learning rate for LoRA backbone parameters")
    p.add_argument("--lr-head",           type=float, default=3e-4,
                   help="Learning rate for prediction head parameters")
    p.add_argument("--weight-decay",      type=float, default=1e-3)
    p.add_argument("--focal-gamma",       type=float, default=2.0)
    p.add_argument("--head-dropout",      type=float, default=0.1)
    p.add_argument("--warmup-steps",      type=int,   default=100)
    p.add_argument("--micro-batch-size",  type=int,   default=4)
    p.add_argument("--global-batch-size", type=int,   default=32)
    p.add_argument("--max-epochs",        type=int,   default=150)
    p.add_argument("--patience",          type=int,   default=20)
    p.add_argument("--num-workers",       type=int,   default=2)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step",    type=int,   default=None,
                   help="Limit train/val/test steps (debug mode)")
    p.add_argument("--fast-dev-run",      action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus  = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Estimate total training steps for LR schedule
    train_size  = 1416
    steps_per_epoch = max(1, train_size // (args.micro_batch_size * n_gpus))
    accum       = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    # Effective steps per epoch after gradient accumulation
    eff_steps_per_epoch = max(1, steps_per_epoch // accum)
    max_steps_total = eff_steps_per_epoch * args.max_epochs

    dm  = AIDOCellDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = AIDOCellLitModule(
        lr_backbone       = args.lr_backbone,
        lr_head           = args.lr_head,
        weight_decay      = args.weight_decay,
        focal_gamma       = args.focal_gamma,
        head_dropout      = args.head_dropout,
        warmup_steps      = args.warmup_steps,
        max_steps_total   = max_steps_total,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="step")
    pb_cb  = TQDMProgressBar(refresh_rate=10)

    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps_trainer: int = -1
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps_trainer = args.debug_max_step
        limit_train = args.debug_max_step
        limit_val   = 2
        limit_test  = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps_trainer,
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
        gradient_clip_val=1.0,  # Gradient clipping for stability
    )

    trainer.fit(lit, datamodule=dm)

    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 2-1 — AIDO.Cell-100M + LoRA r=32 + Gene-Position Extraction "
            "+ Multi-Gene Input Profile\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
