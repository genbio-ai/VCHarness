"""Node 2 – AIDO.Cell-100M + LoRA + Focal Loss.

Strategy:
- Encode the perturbed gene using AIDO.Cell-100M: set only the perturbed gene's
  expression to 1.0 and all others to -1.0 (missing). The model's position+gene
  embeddings provide a rich, pretrained biological representation.
- Extract: concat(perturbed-gene hidden state, mean-pool of all gene states) → [B, 1280]
- LoRA (r=16) on QKV attention projections for parameter-efficient fine-tuning.
- Focal Loss (γ=2) to address the severe 92.5% neutral class imbalance.
- AdamW + ReduceLROnPlateau.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_GENES   = 6640
N_CLASSES = 3
AIDO_GENES = 19264    # AIDO.Cell vocabulary size
MODEL_DIR  = "/home/Models/AIDO.Cell-100M"
HIDDEN_DIM = 640      # AIDO.Cell-100M hidden size

CLASS_FREQ = [0.0429, 0.9251, 0.0320]  # down, neutral, up (remapped 0,1,2)

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TRAIN_TSV = DATA_ROOT / "train.tsv"
VAL_TSV   = DATA_ROOT / "val.tsv"
TEST_TSV  = DATA_ROOT / "test.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_class_weights() -> torch.Tensor:
    """Sqrt-inverse-frequency class weights."""
    w = [1.0 / (f + 1e-9) ** 0.5 for f in CLASS_FREQ]
    m = sum(w) / len(w)
    return torch.tensor([x / m for x in w], dtype=torch.float32)


def compute_per_gene_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-gene macro-averaged F1 matching calc_metric.py."""
    y_hat = preds.argmax(dim=1)       # [N, G]
    G = targets.shape[1]
    f1_per_gene = torch.zeros(G, device=preds.device)
    n_present   = torch.zeros(G, device=preds.device)

    for c in range(3):
        is_true = (targets == c)
        is_pred = (y_hat == c)
        present = is_true.any(dim=0).float()
        tp = (is_pred & is_true).float().sum(0)
        fp = (is_pred & ~is_true).float().sum(0)
        fn = (~is_pred & is_true).float().sum(0)
        prec = torch.where(tp + fp > 0, tp / (tp + fp + 1e-8), torch.zeros_like(tp))
        rec  = torch.where(tp + fn > 0, tp / (tp + fn + 1e-8), torch.zeros_like(tp))
        f1_c = torch.where(prec + rec > 0,
                           2 * prec * rec / (prec + rec + 1e-8),
                           torch.zeros_like(prec))
        f1_per_gene += f1_c * present
        n_present   += present

    return (f1_per_gene / n_present.clamp(min=1)).mean().item()


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced multi-class classification."""

    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.pert_ids = df["pert_id"].tolist()
        self.symbols  = df["symbol"].tolist()
        has_label = "label" in df.columns and df["label"].notna().all()
        if has_label:
            self.labels: Optional[List] = [
                torch.tensor([x + 1 for x in json.loads(row)], dtype=torch.long)
                for row in df["label"].tolist()
            ]
        else:
            self.labels = None

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
    """Factory for collate_fn with AIDO.Cell tokenizer."""

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        pert_ids = [b["pert_id"] for b in batch]
        symbols  = [b["symbol"]  for b in batch]

        # Tokenize: each sample gets only its perturbed gene with expression=1.0
        expr_dicts = [
            {"gene_ids": [pid], "expression": [1.0]} for pid in pert_ids
        ]
        tokenized = tokenizer(expr_dicts, return_tensors="pt")  # input_ids: [B, 19264] float32

        # Find perturbed gene position for each sample (position where value != -1.0)
        input_ids = tokenized["input_ids"]   # [B, 19264]
        # Gene not in vocab → all -1.0 → argmax returns 0 (fallback)
        gene_in_vocab  = (input_ids > -1.0).any(dim=1)                       # [B]
        gene_positions = torch.where(
            gene_in_vocab,
            (input_ids > -1.0).float().argmax(dim=1),
            torch.zeros(len(batch), dtype=torch.long),
        )   # [B]

        out: Dict[str, Any] = {
            "sample_idx":     torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long),
            "pert_id":        pert_ids,
            "symbol":         symbols,
            "input_ids":      input_ids,
            "attention_mask": tokenized["attention_mask"],
            "gene_positions": gene_positions,
            "gene_in_vocab":  gene_in_vocab,
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch])
        return out

    return collate_fn


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class DEGDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_workers: int = 4) -> None:
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

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=make_collate(self.tokenizer),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AIDOCellLoRAModel(pl.LightningModule):
    """AIDO.Cell-100M + LoRA + compact linear head + Focal Loss."""

    def __init__(
        self,
        lora_r: int         = 16,
        lora_alpha: int     = 32,
        lora_dropout: float = 0.05,
        head_hidden: int    = 512,
        head_dropout: float = 0.3,
        lr: float           = 2e-4,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        # ---- Load backbone ----
        backbone = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
        backbone = backbone.to(torch.bfloat16)
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # PEFT's get_peft_model calls model.enable_input_require_grads() which
        # internally calls get_input_embeddings() — this raises NotImplementedError
        # for AIDO.Cell (custom GeneEmbedding, no standard word_embeddings).
        # Monkey-patch it to a no-op before calling get_peft_model; we'll
        # register a proper forward hook on the gene_embedding module below.
        backbone.enable_input_require_grads = lambda: None

        # ---- Wrap with LoRA ----
        lora_cfg = LoraConfig(
            task_type       = TaskType.FEATURE_EXTRACTION,
            r               = hp.lora_r,
            lora_alpha      = hp.lora_alpha,
            lora_dropout    = hp.lora_dropout,
            target_modules  = ["query", "key", "value"],
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        # Register forward hook on gene_embedding to enable input require_grads.
        # This ensures gradients flow back through the embedding to LoRA adapters.
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        gene_emb = self.backbone.model.bert.gene_embedding
        gene_emb.register_forward_hook(_make_inputs_require_grad)

        # Cast LoRA params to float32 for stable optimization
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

        # ---- Head ----
        # concat(pert_gene_emb, mean_pool) → [B, 2*HIDDEN_DIM=1280]
        in_dim = 2 * HIDDEN_DIM
        self.head = nn.Sequential(
            nn.Linear(in_dim, hp.head_hidden),
            nn.LayerNorm(hp.head_hidden),
            nn.GELU(),
            nn.Dropout(hp.head_dropout),
            nn.Linear(hp.head_hidden, N_CLASSES * N_GENES),
        )

        # ---- Loss ----
        self.register_buffer("class_weights", get_class_weights())
        self.focal_loss = FocalLoss(gamma=2.0)

        # Accumulators
        self._val_preds: List[torch.Tensor]  = []
        self._val_tgts:  List[torch.Tensor]  = []
        self._val_idx:   List[torch.Tensor]  = []
        self._test_preds: List[torch.Tensor] = []
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
            output_hidden_states=False,
        )
        # last_hidden_state: [B, 19266, HIDDEN_DIM]
        lhs = out.last_hidden_state  # [B, 19266, 640]

        # 1. Perturbed gene embedding
        pert_emb = lhs[torch.arange(B, device=lhs.device), gene_positions, :]  # [B, 640]

        # 2. Mean pool over gene positions (exclude 2 summary tokens)
        mean_emb = lhs[:, :AIDO_GENES, :].mean(dim=1)  # [B, 640]

        # 3. Concatenate
        emb = torch.cat([pert_emb, mean_emb], dim=-1).float()  # [B, 1280], cast to fp32 for head

        logits = self.head(emb).view(B, N_CLASSES, N_GENES)   # [B, 3, G]
        return logits

    # ---- loss ----
    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, G = logits.shape
        self.focal_loss.weight = self.class_weights
        return self.focal_loss(
            logits.permute(0, 2, 1).reshape(-1, C),
            targets.reshape(-1),
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
        s_idx  = all_idx[order];  s_pred = all_preds[order];  s_tgt = all_tgts[order]
        mask   = torch.cat([torch.tensor([True], device=s_idx.device), s_idx[1:] != s_idx[:-1]])
        f1 = compute_per_gene_f1(s_pred[mask], s_tgt[mask])
        self.log("val/f1", f1, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"], batch["gene_positions"])
        probs  = torch.softmax(logits.float(), dim=1).detach()
        self._test_preds.append(probs)
        for i, (pid, sym) in enumerate(zip(batch["pert_id"], batch["symbol"])):
            self._test_meta.append((pid, sym, batch["sample_idx"][i].item()))
        if "labels" in batch:
            loss = self._loss(logits, batch["labels"])
            self.log("test/loss", loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        local_preds = torch.cat(self._test_preds, 0)
        all_preds   = self.all_gather(local_preds).view(-1, N_CLASSES, N_GENES)
        if self.trainer.is_global_zero:
            n = min(len(self._test_meta), all_preds.shape[0])
            rows = []
            for i in range(n):
                pid, sym, _ = self._test_meta[i]
                rows.append({
                    "idx":        pid,
                    "input":      sym,
                    "prediction": json.dumps(all_preds[i].float().cpu().numpy().tolist()),
                })
            out_dir = Path(__file__).parent / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "test_predictions.tsv", sep="\t", index=False)
            print(f"[Node2] Saved {len(rows)} test predictions.")
        self._test_preds.clear()
        self._test_meta.clear()

    # ---- checkpoint ----
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
        total  = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.print(f"Checkpoint: {trained}/{total} params ({100*trained/total:.2f}%)")
        return trainable

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    # ---- optimizer ----
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=8, min_lr=1e-7
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val/f1"}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(description="Node2 – AIDO.Cell-100M + LoRA + Focal Loss")
    parser.add_argument("--micro-batch-size",  type=int,   default=4)
    parser.add_argument("--global-batch-size", type=int,   default=32)
    parser.add_argument("--max-epochs",        type=int,   default=150)
    parser.add_argument("--lr",                type=float, default=2e-4)
    parser.add_argument("--weight-decay",      type=float, default=1e-2)
    parser.add_argument("--lora-r",            type=int,   default=16)
    parser.add_argument("--lora-alpha",        type=int,   default=32)
    parser.add_argument("--lora-dropout",      type=float, default=0.05)
    parser.add_argument("--head-hidden",       type=int,   default=512)
    parser.add_argument("--head-dropout",      type=float, default=0.3)
    parser.add_argument("--num-workers",       type=int,   default=4)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--debug-max-step",    type=int,   default=None,
                        dest="debug_max_step")
    parser.add_argument("--fast-dev-run",      action="store_true", dest="fast_dev_run")
    args = parser.parse_args()

    n_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    n_gpus = max(n_gpus, 1)

    output_dir = Path(__file__).parent / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    fast_dev_run = args.fast_dev_run
    if args.debug_max_step is not None:
        lim_train = args.debug_max_step
        lim_val   = args.debug_max_step
        lim_test  = args.debug_max_step
        max_steps = args.debug_max_step
    else:
        lim_train = lim_val = lim_test = 1.0
        max_steps = -1

    accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpus))

    dm    = DEGDataModule(batch_size=args.micro_batch_size, num_workers=args.num_workers)
    model = AIDOCellLoRAModel(
        lora_r       = args.lora_r,
        lora_alpha   = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        head_hidden  = args.head_hidden,
        head_dropout = args.head_dropout,
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath    = str(output_dir / "checkpoints"),
        filename   = "best-{epoch:03d}-{val/f1:.4f}",
        monitor    = "val/f1",
        mode       = "max",
        save_top_k = 1,
    )
    es_cb = EarlyStopping(monitor="val/f1", mode="max", patience=15, min_delta=1e-4)
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
        val_check_interval      = 1.0 if (args.debug_max_step is not None or fast_dev_run) else args.val_check_interval,
        num_sanity_val_steps    = 2,
        callbacks               = [ckpt_cb, es_cb, lr_cb, pg_cb],
        logger                  = [csv_logger, tb_logger],
        log_every_n_steps       = 5,
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
    print(f"[Node2] Done. Results saved to {score_path}")


if __name__ == "__main__":
    main()
