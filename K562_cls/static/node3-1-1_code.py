"""Node 1-2 – AIDO.Cell-10M + QKV+Output Fine-tuning + Muon + Concat Fusion + Label-Smooth CE.

This node is an improved version of the parent (node1-2 / node3-1), which catastrophically
failed due to focal loss + class weights causing prediction collapse (test F1=0.188).

Key design decisions informed by memory analysis:
1. REVERT focal loss → label-smoothed CE (ε=0.1): Proven safe in node3 (F1=0.426). Focal
   loss with class weights created 48× gradient amplification causing near-random 33% per
   class predictions. Label-smoothed CE with sqrt-inverse-frequency class weights is safe.
2. Restore concat fusion (4 layers → 1024-dim): The 256-dim learnable weighted sum from
   node1-2 bottlenecked information. Concatenation of 4 layers (1024-dim) is proven.
3. Restore head_hidden=512: Combined with the reduced 256-dim bottleneck and 0.4 dropout,
   the parent collapsed. Restore proven 512-dim head from node3.
4. head_dropout=0.3 (mild reduction from 0.4, slight increase from 0.2): Balance between
   node3 (0.2 = too little) and node1-2 (0.4 = too much when combined with reduced capacity).
5. Extend QKV to QKV+Output: Additionally unfreeze the attention output projection
   (output.dense), which is also a square 256×256 matrix. This adds 4× more matrix-Muon
   parameters per layer while staying within memory budget (~+0.5M params).
6. Cosine Annealing LR schedule (T_max=50): Replaces ReduceLROnPlateau. Cosine annealing
   is more predictable and doesn't require tuning patience parameter. Warm restarts every
   50 epochs allow the model to escape local minima.
7. Tighter early stopping (patience=10, min_delta=0.002): Preventing the 16 post-peak
   epochs of node3, while being slightly more permissive than the node1-2 parent's 8/0.003
   which may have been too aggressive.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES    = 6640
N_CLASSES  = 3
AIDO_GENES = 19264
MODEL_DIR  = "/home/Models/AIDO.Cell-10M"
HIDDEN_DIM = 256      # AIDO.Cell-10M hidden size
N_LAYERS   = 8        # AIDO.Cell-10M transformer layers

# Class frequencies: down-regulated, neutral, up-regulated (remapped to 0,1,2)
CLASS_FREQ = [0.0429, 0.9251, 0.0320]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights for class imbalance."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute per-gene macro F1, matching the calc_metric.py evaluation logic.

    Args:
        preds: [N, 3, G] softmax probabilities
        targets: [N, G] integer class labels (0=down, 1=neutral, 2=up)

    Returns:
        Scalar: mean per-gene macro F1 over all G genes.
    """
    y_hat       = preds.argmax(dim=1)  # [N, G]
    G           = targets.shape[1]
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
        f1_c = torch.where(prec + rec > 0, 2*prec*rec/(prec+rec+1e-8), torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present
    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


# ---------------------------------------------------------------------------
# Dataset / DataModule
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        has_label = "label" in df.columns and df["label"].notna().all()
        self.labels: Optional[List] = (
            [torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
             for row in df["label"].tolist()]
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


def make_collate(tokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]
        expr_dicts = [{"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids]
        tokenized  = tokenizer(expr_dicts, return_tensors="pt")
        input_ids  = tokenized["input_ids"]  # [B, 19264] float32
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),
            torch.zeros(len(batch), dtype=torch.long),
        )
        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      input_ids,
            "attention_mask": tokenized["attention_mask"],
            "gene_positions": gene_positions,
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out
    return collate_fn


class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.tokenizer   = None

    def setup(self, stage: Optional[str] = None) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

        train_df = pd.read_csv(TRAIN_TSV, sep="\t")
        val_df   = pd.read_csv(VAL_TSV,   sep="\t")
        test_df  = pd.read_csv(TEST_TSV,  sep="\t")
        self.train_ds = DEGDataset(train_df)
        self.val_ds   = DEGDataset(val_df)
        self.test_ds  = DEGDataset(test_df)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True,
                          collate_fn=make_collate(self.tokenizer))

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCellQKVOutputModel(pl.LightningModule):
    """AIDO.Cell-10M + QKV+Output fine-tuning + Muon + concat fusion + label-smooth CE.

    Key differences from parent node (node1-2 / node3-1) which catastrophically failed:
    - Label-smoothed CE (ε=0.1) + mild class weights: Safe, proven in node3 (F1=0.426)
    - NO focal loss: The focal loss caused prediction collapse to 33% per class
    - Concatenation fusion (1024-dim): Restores proven information-rich representation
    - head_hidden=512: Restore proven node3 capacity
    - head_dropout=0.3: Slightly higher than node3 (0.2) for mild additional regularization
    - QKV + output.dense fine-tuning: Extends trainable backbone params for richer adaptation
    - Cosine annealing LR: More predictable schedule than ReduceLROnPlateau
    """

    def __init__(
        self,
        fusion_layers: int   = 4,        # last N transformer layers to concatenate
        head_hidden: int     = 512,       # restored from 512 (node3 proven value)
        head_dropout: float  = 0.3,       # between node3 (0.2) and failed node1-2 (0.4)
        lr_muon: float       = 0.02,      # Muon lr for QKV+output weight matrices
        lr_adamw: float      = 2e-4,      # AdamW lr for head
        weight_decay: float  = 1e-2,
        label_smoothing: float = 0.1,     # label smoothing for CE loss
        cosine_t_max: int    = 50,        # T_max for CosineAnnealingWarmRestarts
        cosine_eta_min: float = 1e-7,     # minimum LR floor for cosine annealing
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load backbone ----
        self.backbone = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
        self.backbone = self.backbone.to(torch.bfloat16)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # ---- Enable FlashAttention to avoid OOM ----
        self.backbone.config._use_flash_attention_2 = True

        # ---- Share QKV weight tensors between flash_self and self.self ----
        # After enabling flash_attn, each attention layer has a separate
        # flash_self (BertSelfFlashAttention) with its own QKV weight tensors.
        # For QKV-only fine-tuning we only train self.self's QKV weights,
        # so we make flash_self's QKV tensors alias self.self's QKV tensors.
        for layer in self.backbone.bert.encoder.layer:
            ss = layer.attention.flash_self  # BertSelfFlashAttention
            mm = layer.attention.self       # CellFoundationSelfAttention (regular)
            # Share Q/K/V weights so both paths see the same trainable parameters
            ss.query.weight = mm.query.weight
            ss.key.weight   = mm.key.weight
            ss.value.weight = mm.value.weight
            ss.query.bias   = mm.query.bias
            ss.key.bias     = mm.key.bias
            ss.value.bias   = mm.value.bias

        # ---- Freeze all, then unfreeze QKV weights + attention output projection ----
        # Innovation: Additionally unfreeze attention.output.dense (the output projection)
        # which is also a square 256×256 matrix ideal for Muon.
        # This extends the trainable attention components from 3 (QKV) to 4 (QKV+Output)
        # per layer, giving ~2.1M trainable backbone params vs ~1.57M (QKV-only).
        for param in self.backbone.parameters():
            param.requires_grad = False

        qkv_out_patterns = (
            "attention.self.query.weight",
            "attention.self.key.weight",
            "attention.self.value.weight",
            "attention.output.dense.weight",  # NEW: output projection matrix
        )
        for name, param in self.backbone.named_parameters():
            if any(name.endswith(p) for p in qkv_out_patterns):
                param.requires_grad = True

        qkv_count = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total      = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Node] Trainable backbone params: {qkv_count:,} / {total:,}")

        # ---- Concatenation fusion head ----
        # Proven approach from node3: concatenate last 4 layers at perturbed gene position
        # Input: [B, 4 * 256] = [B, 1024]
        in_dim = hp.fusion_layers * HIDDEN_DIM
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )
        # Cast head to float32 for stable optimization
        self.head = self.head.float()

        # ---- Loss: label-smoothed CE + class weights ----
        # Restoring node3's proven approach. Focal loss caused prediction collapse in parent.
        class_weights = get_class_weights()
        self.register_buffer("class_weights", class_weights)
        self.label_smoothing = hp.label_smoothing

        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
        self._test_tgts: List[torch.Tensor]  = []
        self._test_meta:  List[Tuple]        = []

    # ---- forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_positions: torch.Tensor,
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,   # needed for multi-layer fusion
        )
        # hidden_states: tuple of (N_LAYERS+1) tensors, each [B, AIDO_GENES+2, 256]
        # Use last `fusion_layers` transformer layers (exclude layer 0 = embedding output)
        hidden_states = out.hidden_states   # len = N_LAYERS + 1 = 9

        n = self.hparams.fusion_layers
        # Concatenation fusion: last n layers at the perturbed gene position
        # Indices: -1 (last), -2, -3, -4 → transformer layers 8,7,6,5
        layer_embs = []
        for i in range(n):
            hs = hidden_states[-(i + 1)]             # [B, AIDO_GENES+2, 256]
            # Extract at perturbed gene position and cast to float32 for head
            ge = hs[torch.arange(B, device=hs.device), gene_positions, :].float()  # [B, 256]
            layer_embs.append(ge)

        # Concatenate layers: [B, n * 256]
        fused = torch.cat(layer_embs, dim=1)  # [B, 1024]

        logits = self.head(fused).view(B, N_CLASSES, N_GENES)
        return logits

    # ---- loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        # Reshape for CE loss: [B*G, C] and [B*G]
        flat_logits  = logits.permute(0, 2, 1).reshape(-1, C)
        flat_targets = targets.reshape(-1)
        # Label-smoothed CE with class weights — proven safe in node3
        return F.cross_entropy(
            flat_logits,
            flat_targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    # ---- steps ----
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_positions"])
        loss   = self._loss(logits, batch["labels"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_positions"])
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("val/loss", loss, sync_dist=True)
            probs = torch.softmax(logits.float(), dim=1).detach()
            self._val_preds.append(probs)
            self._val_tgts.append(batch["labels"].detach())
            self._val_idx.append(batch["sample_idx"].detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        local_preds = torch.cat(self._val_preds, 0)
        local_tgts  = torch.cat(self._val_tgts,  0)
        local_idx   = torch.cat(self._val_idx,   0)
        self._val_preds.clear(); self._val_tgts.clear(); self._val_idx.clear()

        all_preds = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        all_tgts  = self.all_gather(local_tgts).view(-1, N_GENES)
        all_idx   = self.all_gather(local_idx).view(-1)

        order  = torch.argsort(all_idx)
        s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=False)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_positions"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        for i, (pid, sym) in enumerate(zip(batch["pert_id"], batch["symbol"])):
            self._test_meta.append((pid, sym, batch["sample_idx"][i].item()))
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)
            self._test_tgts.append(batch["labels"].detach())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        all_preds   = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)

        # Gather sample indices for proper alignment
        local_idx_list = torch.tensor(
            [m[2] for m in self._test_meta], dtype=torch.long, device=all_preds.device
        )
        all_idx = self.all_gather(local_idx_list).view(-1)

        # Compute test F1 if targets are available
        if self._test_tgts:
            local_tgts = torch.cat(self._test_tgts, 0)
            all_tgts   = self.all_gather(local_tgts).view(-1, N_GENES)
            order  = torch.argsort(all_idx)
            s_idx  = all_idx[order]; s_pred = all_preds[order]; s_tgt = all_tgts[order]
            mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
            test_f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
            self.log("test/f1", test_f1, prog_bar=True, sync_dist=True)

        # Gather metadata from ALL ranks using torch.distributed.all_gather_object
        world_size = self.trainer.world_size if hasattr(self.trainer, "world_size") else 1
        all_meta_flat: List[Tuple] = []
        if world_size > 1:
            # Manually gather metadata from all ranks
            gathered_meta: List[List] = [None] * world_size
            torch.distributed.all_gather_object(gathered_meta, list(self._test_meta))
            for meta_list in gathered_meta:
                all_meta_flat.extend(meta_list)
        else:
            all_meta_flat = list(self._test_meta)

        if self.trainer.is_global_zero:
            # Align metadata by sample index using a dict for O(1) lookup
            meta_dict: Dict[int, Tuple] = {m[2]: m for m in all_meta_flat}
            n_samples = all_preds.shape[0]
            rows = []
            for i in range(n_samples):
                idx_val = all_idx[i].item()
                if idx_val in meta_dict:
                    pid, sym, _ = meta_dict[idx_val]
                else:
                    pid, sym = f"unknown_{idx_val}", f"unknown_{idx_val}"
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(all_preds[i].float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_tgts.clear()
        self._test_meta.clear()

    # ---- checkpoint: save only trainable parameters ----
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
        self.print(f"Checkpoint: {trained}/{total} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- optimizer: Muon for QKV+Output weight matrices, AdamW for head ----
    def configure_optimizers(self):
        hp = self.hparams

        # All trainable backbone matrices (QKV + output.dense) → Muon
        # These are all square 256×256 matrices, ideal for Muon's orthogonalization
        backbone_matrices = [
            p for name, p in self.backbone.named_parameters()
            if p.requires_grad and p.ndim >= 2
        ]
        # Backbone bias params (if any) + head parameters → AdamW
        backbone_biases = [
            p for name, p in self.backbone.named_parameters()
            if p.requires_grad and p.ndim < 2
        ]
        head_params = list(self.head.parameters())

        param_groups = [
            # Muon group: QKV + output.dense weight matrices (4 matrices per layer × 8 layers = 32)
            dict(
                params      = backbone_matrices,
                use_muon    = True,
                lr          = hp.lr_muon,
                weight_decay = hp.weight_decay,
                momentum    = 0.95,
            ),
            # AdamW group: head + any backbone biases
            dict(
                params      = head_params + backbone_biases,
                use_muon    = False,
                lr          = hp.lr_adamw,
                betas       = (0.9, 0.95),
                weight_decay = hp.weight_decay,
            ),
        ]
        use_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        opt_cls = MuonWithAuxAdam if use_distributed else SingleDeviceMuonWithAuxAdam
        optimizer = opt_cls(param_groups)

        # Cosine Annealing with Warm Restarts:
        # - More predictable than ReduceLROnPlateau (no plateau detection required)
        # - Warm restarts every T_max epochs allow escaping local minima
        # - η_min prevents LR from collapsing to near zero
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hp.cosine_t_max,
            T_mult=1,
            eta_min=hp.cosine_eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(
        description="Node – AIDO.Cell-10M + QKV+Output + Muon + Concat Fusion + Label-Smooth CE"
    )
    parser.add_argument("--micro-batch-size",   type=int,   default=16)
    parser.add_argument("--global-batch-size",  type=int,   default=128)
    parser.add_argument("--max-epochs",         type=int,   default=150)
    parser.add_argument("--lr-muon",            type=float, default=0.02)
    parser.add_argument("--lr-adamw",           type=float, default=2e-4)
    parser.add_argument("--weight-decay",       type=float, default=1e-2)
    parser.add_argument("--fusion-layers",      type=int,   default=4)
    parser.add_argument("--head-hidden",        type=int,   default=512)
    parser.add_argument("--head-dropout",       type=float, default=0.3)
    parser.add_argument("--label-smoothing",    type=float, default=0.1)
    parser.add_argument("--cosine-t-max",       type=int,   default=50)
    parser.add_argument("--cosine-eta-min",     type=float, default=1e-7)
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--debug_max_step",   type=int,   default=None)
    parser.add_argument("--fast_dev_run",     action="store_true")
    parser.add_argument("--val-check-interval", type=float, default=1.0)
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

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOCellQKVOutputModel(
        fusion_layers    = args.fusion_layers,
        head_hidden      = args.head_hidden,
        head_dropout     = args.head_dropout,
        lr_muon          = args.lr_muon,
        lr_adamw         = args.lr_adamw,
        weight_decay     = args.weight_decay,
        label_smoothing  = args.label_smoothing,
        cosine_t_max     = args.cosine_t_max,
        cosine_eta_min   = args.cosine_eta_min,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val/f1:.4f}",
        monitor="val/f1", mode="max", save_top_k=1,
    )
    # Tighter early stopping: patience=10 (vs node3's 15) prevents post-peak overfitting
    # min_delta=0.002: meaningful improvement threshold (between node3's 1e-4 and node1-2's 0.003)
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=10, min_delta=0.002)
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    pg_cb = TQDMProgressBar(refresh_rate=10)

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
        val_check_interval      = args.val_check_interval if (args.debug_max_step is None and not fast_dev_run) else 1.0,
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
    print(f"[Node] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
