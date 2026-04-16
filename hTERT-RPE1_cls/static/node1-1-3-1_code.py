"""
Node 1-1-3-1 – AIDO.Cell-100M + Fixed Multi-Gene Input + Hybrid Gene-Position/Summary Representation
             + Bilinear Head + LR Schedule Fix + Larger LoRA + Label Smoothing + Adjusted Class Weights

Key improvements over node1-1-3 (F1=0.4411):
1. Fixed LR schedule mismatch: max_epochs=60, patience=25 ensures cosine schedule covers actual training window
2. Larger LoRA rank: r=32, alpha=64 for more backbone expressivity
3. Improved LR differential: lr_backbone=3e-5 (more conservative), lr_head=5e-4 (faster head adaptation)
4. Label smoothing (eps=0.05) in focal loss to reduce overconfidence and curb overfitting
5. Larger global batch size (64) for more stable gradient estimates
6. Adjusted class weights [2.5, 1.0, 5.5] for stronger minority class push
7. Increased warmup steps (150) to accommodate larger LoRA rank
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

N_GENES_OUT = 6640
N_CLASSES = 3
MODEL_DIR = "/home/Models/AIDO.Cell-100M"
AIDO_HIDDEN_DIM = 640
AIDO_N_GENES = 19264
# Adjusted class weights: stronger push toward minority classes (down=2.5x, up=5.5x)
CLASS_WEIGHTS = torch.tensor([2.5, 1.0, 5.5], dtype=torch.float32)
PERTURB_EXPRESSION = 10.0
BASELINE_EXPRESSION = 1.0


def compute_per_gene_f1(pred_logits_np, labels_np):
    pred_classes = pred_logits_np.argmax(axis=1)
    n_genes = labels_np.shape[1]
    f1_vals = []
    for g in range(n_genes):
        yt = labels_np[:, g]
        yh = pred_classes[:, g]
        present = np.array([(yt == c).any() for c in [0, 1, 2]])
        pc_f1 = f1_score(yt, yh, labels=[0, 1, 2], average=None, zero_division=0)
        f1_vals.append(float(pc_f1[present].mean()))
    return float(np.mean(f1_vals))


def focal_cross_entropy(logits, targets, class_weights, gamma=2.0, label_smoothing=0.0):
    """Focal cross-entropy with optional label smoothing to reduce overconfidence."""
    ce = F.cross_entropy(
        logits, targets,
        weight=class_weights.to(logits.device),
        reduction="none",
        label_smoothing=label_smoothing
    )
    pt = torch.exp(-ce)
    focal = (1.0 - pt) ** gamma * ce
    return focal.mean()


class PerturbationDataset(Dataset):
    def __init__(self, df, gene_to_pos, all_gene_names, has_labels=True):
        self.pert_ids = df["pert_id"].tolist()
        self.symbols = df["symbol"].tolist()
        self.gene_positions = [gene_to_pos.get(sym, -1) for sym in self.symbols]
        self.all_gene_names = all_gene_names
        self.has_labels = has_labels
        if has_labels and "label" in df.columns:
            rows = []
            for lbl_str in df["label"]:
                rows.append([x + 1 for x in json.loads(lbl_str)])
            self.labels = torch.tensor(rows, dtype=torch.long)
        else:
            self.has_labels = False

    def __len__(self):
        return len(self.pert_ids)

    def __getitem__(self, idx):
        item = {
            "pert_id": self.pert_ids[idx],
            "symbol": self.symbols[idx],
            "gene_pos": self.gene_positions[idx]
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item


def build_collate_fn(tokenizer, all_gene_names):
    base_expression = [BASELINE_EXPRESSION] * len(all_gene_names)

    def collate_fn(batch):
        expr_inputs = []
        for item in batch:
            expr = list(base_expression)
            gene_pos_in_vocab = item["gene_pos"]
            if gene_pos_in_vocab >= 0:
                expr[gene_pos_in_vocab] = PERTURB_EXPRESSION
            expr_inputs.append({"gene_names": all_gene_names, "expression": expr})
        tokenized = tokenizer(expr_inputs, return_tensors="pt")
        gene_positions = torch.tensor([item["gene_pos"] for item in batch], dtype=torch.long)
        pert_ids = [item["pert_id"] for item in batch]
        symbols = [item["symbol"] for item in batch]
        result = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "gene_pos": gene_positions,
            "pert_id": pert_ids,
            "symbol": symbols,
        }
        if "label" in batch[0]:
            result["label"] = torch.stack([item["label"] for item in batch], dim=0)
        return result

    return collate_fn


class PerturbationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", micro_batch_size=4, num_workers=4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.gene_to_pos = {}
        self.all_gene_names = []
        self.tokenizer = None

    def setup(self, stage=None):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        self.gene_to_pos = {sym: pos for sym, pos in self.tokenizer.gene_to_index.items()}
        self.all_gene_names = [""] * len(self.gene_to_pos)
        for sym, pos in self.gene_to_pos.items():
            self.all_gene_names[pos] = sym
        dfs = {}
        for split in ("train", "val", "test"):
            dfs[split] = pd.read_csv(self.data_dir / f"{split}.tsv", sep="\t")
        self.train_ds = PerturbationDataset(dfs["train"], self.gene_to_pos, self.all_gene_names, True)
        self.val_ds = PerturbationDataset(dfs["val"], self.gene_to_pos, self.all_gene_names, True)
        self.test_ds = PerturbationDataset(dfs["test"], self.gene_to_pos, self.all_gene_names, True)
        self._collate = build_collate_fn(self.tokenizer, self.all_gene_names)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.micro_batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            collate_fn=self._collate, persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
            collate_fn=self._collate, persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.micro_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
            collate_fn=self._collate, persistent_workers=self.num_workers > 0
        )


class HybridFusionHead(nn.Module):
    def __init__(self, hidden_dim=640):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.act = nn.GELU()
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, gene_emb, summary_emb):
        combined = torch.cat([gene_emb, summary_emb], dim=-1)
        combined = self.norm(combined)
        fused = self.act(self.proj(combined))
        return fused


class AIDOCellHybridModel(nn.Module):
    def __init__(self, lora_r=32, lora_alpha=64, lora_dropout=0.05, bilinear_rank=256,
                 head_dropout=0.15, n_genes_out=N_GENES_OUT, n_classes=N_CLASSES,
                 hidden_dim=AIDO_HIDDEN_DIM):
        super().__init__()
        backbone = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"]
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.fusion = HybridFusionHead(hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(head_dropout)
        self.proj_bilinear = nn.Linear(hidden_dim, n_classes * bilinear_rank, bias=True)
        self.out_gene_emb = nn.Embedding(n_genes_out, bilinear_rank)
        nn.init.normal_(self.out_gene_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.proj_bilinear.weight)
        nn.init.zeros_(self.proj_bilinear.bias)
        self.n_classes = n_classes
        self.bilinear_rank = bilinear_rank
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask, gene_pos):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, 19266, 640]
        B = input_ids.shape[0]
        valid_mask = gene_pos >= 0
        safe_pos = gene_pos.clone()
        safe_pos[~valid_mask] = 0
        gene_emb = hidden[torch.arange(B, device=hidden.device), safe_pos]
        if (~valid_mask).any():
            mean_emb = hidden[:, :AIDO_N_GENES, :].mean(dim=1)
            gene_emb = gene_emb.clone()
            gene_emb[~valid_mask] = mean_emb[~valid_mask]
        summary_emb = hidden[:, AIDO_N_GENES:, :].mean(dim=1)
        gene_emb = gene_emb.float()
        summary_emb = summary_emb.float()
        fused = self.fusion(gene_emb, summary_emb)
        fused = self.dropout(fused)
        proj = self.proj_bilinear(fused)
        proj = proj.view(B, self.n_classes, self.bilinear_rank)
        out_emb = self.out_gene_emb.weight
        logits = torch.einsum("bcr,gr->bcg", proj, out_emb)
        return logits


def _gather_tensors(local_preds, local_labels, device, world_size):
    local_size = torch.tensor([local_preds.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = int(max(s.item() for s in all_sizes))
    pad = max_size - local_preds.shape[0]
    p = local_preds.to(device)
    l = local_labels.to(device)
    if pad > 0:
        p = torch.cat([p, p.new_zeros(pad, *p.shape[1:])], dim=0)
        l = torch.cat([l, l.new_zeros(pad, *l.shape[1:])], dim=0)
    g_preds = [torch.zeros_like(p) for _ in range(world_size)]
    g_labels = [torch.zeros_like(l) for _ in range(world_size)]
    dist.all_gather(g_preds, p)
    dist.all_gather(g_labels, l)
    real_preds = torch.cat([g_preds[i][:all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    real_labels = torch.cat([g_labels[i][:all_sizes[i].item()].cpu() for i in range(world_size)], 0)
    return real_preds, real_labels


class PerturbationLitModule(pl.LightningModule):
    def __init__(self, lora_r=32, lora_alpha=64, lora_dropout=0.05, bilinear_rank=256,
                 head_dropout=0.15, lr_backbone=3e-5, lr_head=5e-4, weight_decay=1e-3,
                 warmup_steps=150, focal_gamma=2.0, max_steps_total=10000,
                 label_smoothing=0.05):
        super().__init__()
        self.save_hyperparameters()
        self._val_preds = []
        self._val_labels = []
        self._test_preds = []
        self._test_pert_ids = []
        self._test_symbols = []

    def setup(self, stage=None):
        hp = self.hparams
        self.model = AIDOCellHybridModel(
            lora_r=hp.lora_r, lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout, bilinear_rank=hp.bilinear_rank,
            head_dropout=hp.head_dropout
        )
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()
        self.register_buffer("class_weights", CLASS_WEIGHTS)

    def forward(self, input_ids, attention_mask, gene_pos):
        return self.model(input_ids, attention_mask, gene_pos)

    def _compute_loss(self, logits, labels):
        return focal_cross_entropy(
            logits, labels,
            class_weights=self.class_weights,
            gamma=self.hparams.focal_gamma,
            label_smoothing=self.hparams.label_smoothing
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_pos"])
        loss = self._compute_loss(logits, batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_pos"])
        if "label" in batch:
            loss = self._compute_loss(logits, batch["label"])
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self._val_preds.append(logits.detach().cpu().float())
            self._val_labels.append(batch["label"].cpu())
        return logits

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        local_p = torch.cat(self._val_preds, dim=0)
        local_l = torch.cat(self._val_labels, dim=0)
        if self.trainer.world_size > 1:
            all_p, all_l = _gather_tensors(local_p, local_l, self.device, self.trainer.world_size)
        else:
            all_p, all_l = local_p, local_l
        f1 = compute_per_gene_f1(all_p.numpy(), all_l.numpy())
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_pos"])
        probs = torch.softmax(logits, dim=1)
        self._test_preds.append(probs.detach().cpu().float())
        self._test_pert_ids.extend(batch["pert_id"])
        self._test_symbols.extend(batch["symbol"])
        if "label" in batch:
            if not hasattr(self, "_test_labels"):
                self._test_labels = []
            self._test_labels.append(batch["label"].cpu())

    def on_test_epoch_end(self):
        local_probs = torch.cat(self._test_preds, dim=0)
        dummy_labels = torch.zeros(local_probs.shape[0], N_GENES_OUT, dtype=torch.long)
        if hasattr(self, "_test_labels") and self._test_labels:
            dummy_labels = torch.cat(self._test_labels, dim=0)
            del self._test_labels
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
            all_probs = local_probs
            all_labels = dummy_labels
            all_pert = self._test_pert_ids
            all_syms = self._test_symbols
        if self.trainer.is_global_zero:
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / "test_predictions.tsv"
            seen_ids = set()
            dedup_probs = []
            dedup_labels = []
            with open(pred_path, "w") as fh:
                fh.write("idx\tinput\tprediction\n")
                for i, (pert_id, symbol, probs) in enumerate(zip(all_pert, all_syms, all_probs.numpy())):
                    if pert_id not in seen_ids:
                        seen_ids.add(pert_id)
                        fh.write(f"{pert_id}\t{symbol}\t{json.dumps(probs.tolist())}\n")
                        dedup_probs.append(probs)
                        dedup_labels.append(all_labels[i].numpy())
            self.print(f"Saved test predictions -> {pred_path} ({len(seen_ids)} unique samples)")
            if dedup_probs and dedup_labels:
                dedup_probs_np = np.stack(dedup_probs, axis=0)
                dedup_labels_np = np.stack(dedup_labels, axis=0)
                if dedup_labels_np.any():
                    f1 = compute_per_gene_f1(dedup_probs_np, dedup_labels_np)
                    self.print(f"Self-computed test F1 = {f1:.4f}")
        self._test_preds.clear()
        self._test_pert_ids.clear()
        self._test_symbols.clear()

    def configure_optimizers(self):
        hp = self.hparams
        backbone_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and "backbone" in n
        ]
        head_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and "backbone" not in n
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": hp.lr_backbone},
                {"params": head_params, "lr": hp.lr_head},
            ],
            weight_decay=hp.weight_decay
        )

        def lr_lambda(current_step):
            if current_step < hp.warmup_steps:
                return float(current_step) / max(1, hp.warmup_steps)
            progress = float(current_step - hp.warmup_steps) / max(
                1, hp.max_steps_total - hp.warmup_steps
            )
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        trainable_keys = {prefix + n for n, p in self.named_parameters() if p.requires_grad}
        buffer_keys = {prefix + n for n, _ in self.named_buffers()}
        trainable_sd = {k: v for k, v in full_sd.items() if k in trainable_keys or k in buffer_keys}
        total = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Saving ckpt: {trained}/{total} trainable params ({100*trained/total:.1f}%)")
        return trainable_sd

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)


def parse_args():
    p = argparse.ArgumentParser(description="node1-1-3-1: AIDO.Cell-100M with improved LR schedule and regularization")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--bilinear-rank", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.15)
    p.add_argument("--lr-backbone", type=float, default=3e-5)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=150)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-check-interval", type=float, default=1.0)
    p.add_argument("--debug-max-step", type=int, default=None)
    p.add_argument("--fast-dev-run", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(0)

    n_gpus = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
    out_dir = Path(__file__).parent / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = PerturbationDataModule(
        data_dir=args.data_dir,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers
    )

    _train_df_size = pd.read_csv(Path(args.data_dir) / "train.tsv", sep="\t", usecols=["pert_id"]).shape[0]
    steps_per_epoch = _train_df_size // (args.micro_batch_size * n_gpus)
    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))
    effective_steps_per_epoch = max(1, steps_per_epoch // accum)
    max_steps_total = effective_steps_per_epoch * args.max_epochs

    lit = PerturbationLitModule(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bilinear_rank=args.bilinear_rank,
        head_dropout=args.head_dropout,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        focal_gamma=args.focal_gamma,
        max_steps_total=max(max_steps_total, 1),
        label_smoothing=args.label_smoothing,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-{epoch:04d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    es_cb = EarlyStopping(monitor="val_f1", mode="max", patience=args.patience, min_delta=1e-4)
    lr_cb = LearningRateMonitor(logging_interval="step")
    pb_cb = TQDMProgressBar(refresh_rate=10)
    csv_logger = CSVLogger(save_dir=str(out_dir / "logs"), name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=str(out_dir / "logs"), name="tensorboard_logs")

    max_steps = -1
    limit_train_batches = 1.0
    limit_val_batches = 1.0
    limit_test_batches = 1.0
    fast_dev_run = False

    if args.debug_max_step is not None:
        max_steps = args.debug_max_step
        limit_train_batches = args.debug_max_step
        limit_val_batches = 2
        limit_test_batches = 2
    if args.fast_dev_run:
        fast_dev_run = True

    strategy = (
        DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=120))
        if n_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        num_nodes=1,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        gradient_clip_val=1.0,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
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
    test_results = trainer.test(lit, datamodule=dm, ckpt_path=ckpt)

    if trainer.is_global_zero:
        score_path = Path(__file__).parent / "test_score.txt"
        score_path.write_text(f"Test results from trainer: {test_results}\n")


if __name__ == "__main__":
    main()
