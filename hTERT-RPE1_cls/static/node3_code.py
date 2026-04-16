"""
Node 3 – AIDO.Cell-3M + QKV Direct Fine-tuning + Muon Optimizer

Architecture:
  - AIDO.Cell-3M backbone loaded from /home/Models/AIDO.Cell-3M
  - Synthetic perturbation input: {perturbed_gene_symbol: 1.0}
  - QKV-only direct fine-tuning (no LoRA) – freeze all non-QKV parameters
  - Muon optimizer for QKV weight matrices; AdamW for the prediction head
  - Learned attention pooling over all 19264 gene positions
    (the pooled representation captures which gene is expressed)
  - Prediction head: Linear(128→512) + GELU + LayerNorm + Linear(512 → 6640×3)
  - Label smoothing cross-entropy loss (0.1)

Key diversity vs Node 2:
  - Different model size (3M vs 100M)
  - Direct QKV tuning instead of LoRA
  - Muon optimizer with momentum orthogonalisation
  - Attention pooling instead of extracting specific gene position
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
from muon import MuonWithAuxAdam
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─── Constants ────────────────────────────────────────────────────────────────

AIDO_CELL_DIR = "/home/Models/AIDO.Cell-3M"
N_GENES_OUT   = 6640
N_CLASSES     = 3
HIDDEN_SIZE   = 128   # AIDO.Cell-3M
N_GENE_VOCAB  = 19264 # AIDO.Cell gene space

CLASS_WEIGHTS = torch.tensor([12.28, 1.12, 33.33], dtype=torch.float32)


# ─── Metric ───────────────────────────────────────────────────────────────────

def compute_per_gene_f1(pred_np: np.ndarray, labels_np: np.ndarray) -> float:
    pred_cls = pred_np.argmax(axis=1)
    f1_vals  = []
    for g in range(labels_np.shape[1]):
        yt = labels_np[:, g];  yh = pred_cls[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AIDOCellPerturbDataset(Dataset):
    def __init__(self, pert_ids, symbols, input_ids, gene_positions, labels=None):
        self.pert_ids      = pert_ids
        self.symbols       = symbols
        self.input_ids     = input_ids      # [N, 19264] float32
        self.gene_positions = gene_positions  # [N] long
        self.labels        = labels          # [N, 6640] long or None

    def __len__(self):  return len(self.pert_ids)

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

class AIDOCell3MDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data", micro_batch_size=8, num_workers=2):
        super().__init__()
        self.data_dir         = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)

        def tokenize_symbols(symbols):
            batch_input = [{"gene_names": [s], "expression": [1.0]} for s in symbols]
            tok_out     = tokenizer(batch_input, return_tensors="pt")
            ids  = tok_out["input_ids"]          # [N, 19264] float32
            gpos = (ids > 0.5).float().argmax(dim=1).long()  # [N]
            return ids, gpos

        def load_split(fname, has_lbl):
            df  = pd.read_csv(self.data_dir / fname, sep="\t")
            ids, gpos = tokenize_symbols(df["symbol"].tolist())
            labels = None
            if has_lbl and "label" in df.columns:
                rows   = [[x + 1 for x in json.loads(s)] for s in df["label"]]
                labels = torch.tensor(rows, dtype=torch.long)
            return AIDOCellPerturbDataset(
                df["pert_id"].tolist(), df["symbol"].tolist(), ids, gpos, labels
            )

        self.train_ds = load_split("train.tsv", True)
        self.val_ds   = load_split("val.tsv",   True)
        self.test_ds  = load_split("test.tsv",  True)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.micro_batch_size, shuffle=shuffle,
                          collate_fn=collate_fn, num_workers=self.num_workers,
                          pin_memory=True, drop_last=shuffle,
                          persistent_workers=self.num_workers > 0)

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds,   False)
    def test_dataloader(self):  return self._loader(self.test_ds,  False)


# ─── Model ────────────────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """Learnable single-query attention pooling over sequence dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.query_weight = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale        = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]  →  [B, D]"""
        # attn_score: [B, T, 1]
        attn = torch.matmul(x, self.query_weight.expand(x.shape[0], -1, -1).transpose(-1, -2))
        attn = torch.softmax(attn * self.scale, dim=1)          # [B, T, 1]
        return (x * attn).sum(dim=1)                            # [B, D]


class AIDOCell3MModel(nn.Module):
    """AIDO.Cell-3M with QKV-only direct fine-tuning + attention-pooled head."""

    def __init__(self, n_genes_out: int = N_GENES_OUT, n_classes: int = N_CLASSES):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(AIDO_CELL_DIR, trust_remote_code=True)
        self.backbone = self.backbone.to(torch.bfloat16)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Freeze everything; then unfreeze QKV projection weights
        for param in self.backbone.parameters():
            param.requires_grad = False
        qkv_names = ("attention.self.query.weight",
                      "attention.self.key.weight",
                      "attention.self.value.weight")
        for name, param in self.backbone.named_parameters():
            if name.endswith(qkv_names):
                param.requires_grad = True

        # Keep QKV weights in bfloat16 (same as the rest of the backbone).
        # DO NOT cast to float32: modeling_cellfoundation.py dispatches to
        # FlashAttention only when ln_outputs.dtype is fp16/bf16.  It converts
        # ln_outputs to query.weight.dtype before the check, so float32 QKV
        # weights would keep ln_outputs in float32 → FlashAttention disabled →
        # quadratic O(seq_len²) standard attention → guaranteed OOM.
        # Muon's Newton-Schulz orthogonalisation runs stably in bfloat16.

        # Attention pooling over gene positions
        self.attn_pool = AttentionPool(HIDDEN_SIZE)

        # Prediction head: narrow → wider → output
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, n_genes_out * n_classes),
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Node3] Trainable params: {n_trainable:,}")

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, 19264] float32
        gene_positions: torch.Tensor,  # [B] long  (unused here; kept for API consistency)
    ) -> torch.Tensor:
        attn_mask = torch.ones(input_ids.shape[0], input_ids.shape[1],
                               dtype=torch.long, device=input_ids.device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.backbone(input_ids=input_ids, attention_mask=attn_mask)

        # last_hidden_state: [B, 19266, 128]
        # Use gene positions 0..19263 for attention pooling
        gene_states = out.last_hidden_state[:, :N_GENE_VOCAB, :].float()  # [B, 19264, 128]

        # Attention pooling → [B, 128]
        pooled = self.attn_pool(gene_states)

        logits = self.head(pooled)         # [B, 6640*3]
        return logits.view(-1, N_CLASSES, N_GENES_OUT)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gather_tensors(local_p, local_l, device, world_size):
    local_size = torch.tensor([local_p.shape[0]], dtype=torch.long, device=device)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_sz = int(max(s.item() for s in all_sizes))
    pad    = max_sz - local_p.shape[0]
    p = local_p.to(device);  l = local_l.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], 0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], 0)
    gp = [torch.zeros_like(p) for _ in range(world_size)]
    gl = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(gp, p);  dist.all_gather(gl, l)
    rp = torch.cat([gp[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    rl = torch.cat([gl[i][:int(all_sizes[i].item())].cpu() for i in range(world_size)], 0)
    return rp, rl


# ─── LightningModule ──────────────────────────────────────────────────────────

class AIDOCell3MLitModule(pl.LightningModule):

    def __init__(
        self,
        lr_muon: float  = 0.02,
        lr_adam: float  = 3e-4,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.1,
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
        self.model = AIDOCell3MModel()
        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(self, input_ids, gene_positions):
        return self.model(input_ids, gene_positions)

    def _loss(self, logits, labels):
        return F.cross_entropy(
            logits, labels,
            weight=self.class_weights.to(logits.device),
            label_smoothing=self.hparams.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"])
        loss   = self._loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["gene_position"])
        if "label" in batch:
            loss = self._loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
        # f1 is already computed on globally-gathered data (all ranks hold the same value);
        # sync_dist=True averages identical values across ranks → no change in result.
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear();  self._val_labels.clear()

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
        dummy_labels = (torch.cat(self._test_labels, 0) if self._test_labels
                        else torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long))
        if self.trainer.world_size > 1:
            all_probs, all_labels = _gather_tensors(local_probs, dummy_labels, self.device, self.trainer.world_size)
            all_pert = [None]*self.trainer.world_size;  all_syms = [None]*self.trainer.world_size
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
            # Deduplicate by pert_id: DDP DistributedSampler may pad the dataset to
            # make sample counts equal across ranks, introducing duplicate pert_ids.
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
            self.print(f"[Node3] Saved {len(dedup_perts)} test predictions → {pred_path}")
            if all_labels.any():
                dedup_probs_np  = np.array(dedup_probs_list)
                dedup_labels_np = np.array(dedup_label_rows)
                f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                self.print(f"[Node3] Self-computed test F1 = {f1:.4f}")

        self._test_preds.clear();   self._test_pert_ids.clear()
        self._test_symbols.clear(); self._test_labels.clear()

    # ── Muon + AdamW dual optimiser ───────────────────────────────────────

    def configure_optimizers(self):
        hp = self.hparams

        # QKV weight matrices (ndim >= 2, requires_grad) → Muon
        qkv_weights = [
            p for n, p in self.model.backbone.named_parameters()
            if p.requires_grad and p.ndim >= 2
        ]
        # Everything else (head, attn_pool, 1-D params) → AdamW
        qkv_ids   = {id(p) for p in qkv_weights}
        other_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in qkv_ids
        ]

        param_groups = [
            dict(params=qkv_weights,   use_muon=True,  lr=hp.lr_muon,
                 weight_decay=hp.weight_decay, momentum=0.95),
            dict(params=other_params,  use_muon=False, lr=hp.lr_adam,
                 betas=(0.9, 0.95), weight_decay=hp.weight_decay),
        ]
        optimizer = MuonWithAuxAdam(param_groups)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys    = {prefix + n for n, _ in self.named_buffers()}
        sd = {k: v for k, v in full_sd.items() if k in trainable_keys or k in buffer_keys}
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} params ({100*trained/total:.2f}%)")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Node 3 – AIDO.Cell-3M QKV + Muon")
    p.add_argument("--data-dir",          type=str,   default="data")
    p.add_argument("--lr-muon",           type=float, default=0.02)
    p.add_argument("--lr-adam",           type=float, default=3e-4)
    p.add_argument("--weight-decay",      type=float, default=0.01)
    p.add_argument("--label-smoothing",   type=float, default=0.1)
    p.add_argument("--micro-batch-size",  type=int,   default=8)
    p.add_argument("--global-batch-size", type=int,   default=32)
    p.add_argument("--max-epochs",        type=int,   default=200)
    p.add_argument("--patience",           type=int,   default=20)
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

    dm  = AIDOCell3MDataModule(args.data_dir, args.micro_batch_size, args.num_workers)
    lit = AIDOCell3MLitModule(args.lr_muon, args.lr_adam, args.weight_decay, args.label_smoothing)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
    )
    es_cb  = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-5)
    lr_cb  = LearningRateMonitor(logging_interval="epoch")
    pb_cb  = TQDMProgressBar(refresh_rate=10)
    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger  = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps = -1
    limit_train: float | int = 1.0
    limit_val:   float | int = 1.0
    limit_test:  float | int = 1.0
    fast_dev_run = False
    if args.debug_max_step is not None:
        max_steps = args.debug_max_step;  limit_train = args.debug_max_step
        limit_val = 2;  limit_test = 2
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
        val_check_interval=args.val_check_interval if (args.debug_max_step is None and not args.fast_dev_run) else 1.0,
        num_sanity_val_steps=2,
        callbacks=[ckpt_cb, es_cb, lr_cb, pb_cb],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        deterministic="warn_only",  # nll_loss2d (cross-entropy over [B, 6640]) has no deterministic CUDA impl
        default_root_dir=str(out_dir),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(lit, datamodule=dm)
    ckpt = "best" if (args.debug_max_step is None and not args.fast_dev_run) else None
    trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        (Path(__file__).parent / "test_score.txt").write_text(
            "Node 3 – AIDO.Cell-3M QKV + Muon\n"
            "(Final score computed by EvaluateAgent via calc_metric.py)\n"
        )


if __name__ == "__main__":
    main()
